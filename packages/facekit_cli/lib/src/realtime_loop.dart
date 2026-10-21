// S6 (doc/KR/linux_port_scope.md) — wires a continuous FrameSource into
// detect→align→embed→match, running until stopped. Deliberately takes
// FrameSource (the S5b abstraction) rather than V4l2FrameSource directly,
// so this loop — and its tests — don't care whether frames come from a
// real camera or a fake in-memory source.
//
// Design decisions fixed before writing this (see linux_port_scope.md "S6
// DoD"):
//   - A corrupt/short frame (wrong byte count for its format) is counted
//     and skipped — never thrown past this loop.
//   - A V4L2 driver-side drop (gap in the per-frame sequence counter) is
//     counted and logged — no retry/recovery, same conclusion as S5b.
//   - No face detected, or the pose gate rejecting alignment, counts as
//     "no identification this frame" and the loop continues — never a
//     crash.
// None of that is a recovery *strategy* to design; it's "skip and keep
// going", already decided. What this file adds is just the plumbing.

import 'dart:async';
import 'dart:io';

import 'package:facekit_core/facekit_core.dart';

import 'frame_source.dart';

/// Outcome of processing one non-corrupt frame through the pipeline.
class RealtimeFrameResult {
  final int
  frameIndex; // 1-based count of frames handed to the pipeline (corrupt frames excluded)
  final int?
  sequence; // the source's driver-assigned frame counter, if it has one
  final bool faceDetected;
  final bool aligned; // only meaningful when faceDetected is true
  final MatchResult? match; // null unless faceDetected && aligned
  final Map<String, int>
  ms; // stage timings actually run this frame: subset of detect/align/embed/total

  RealtimeFrameResult({
    required this.frameIndex,
    required this.sequence,
    required this.faceDetected,
    required this.aligned,
    required this.match,
    required this.ms,
  });

  Map<String, dynamic> toJson() => {
    'frame': frameIndex,
    'seq': sequence,
    'face_detected': faceDetected,
    'aligned': aligned,
    'matched_id': match?.matchedId,
    'similarity': match?.similarity,
    'accepted': match?.accepted,
    'ms': ms,
  };
}

/// Running counters for a [runRealtimeLoop] call — the basis for the S6 DoD
/// checks (frame-drop/corruption rate staying flat over a long run, no
/// crash, memory not growing unbounded — same shape as S5b's checks, now
/// with the recognition stages included).
class RealtimeStats {
  int framesSeen = 0; // non-corrupt frames handed to the pipeline
  int corruptFrames =
      0; // wrong byte count for the frame's declared format — skipped
  int droppedFrames =
      0; // gaps in the source's sequence counter — logged only, see file header
  int noFace = 0;
  int alignRejected = 0; // face found but the pose gate rejected it
  int accepted = 0; // matched a gallery entry above threshold
  int rejected = 0; // embedded but didn't match any gallery entry
  int rssStartBytes = 0;
  int rssMaxBytes = 0;

  Map<String, dynamic> toJson() => {
    'frames_seen': framesSeen,
    'corrupt_frames': corruptFrames,
    'dropped_frames': droppedFrames,
    'no_face': noFace,
    'align_rejected': alignRejected,
    'accepted': accepted,
    'rejected': rejected,
    'rss_start_mb': rssStartBytes / 1024 / 1024,
    'rss_max_mb': rssMaxBytes / 1024 / 1024,
    'rss_growth_mb': (rssMaxBytes - rssStartBytes) / 1024 / 1024,
  };
}

/// Returns the exact byte length a non-corrupt [frame] must have, or null
/// if [format] isn't one this loop knows how to size-check (treated as
/// "can't tell, don't flag as corrupt").
int? _expectedFrameBytes(RawFrame frame) {
  switch (frame.format) {
    case PixelFormat.yuyv422:
      return frame.width * frame.height * 2;
    case PixelFormat.bgra8888:
      return frame.width * frame.height * 4;
    case PixelFormat.yuv420_888:
    case PixelFormat.mjpeg:
      return null;
  }
}

/// Converts [frame] to the RGB888 [FaceImage] the pipeline expects.
/// Throws [StateError] for formats this loop doesn't decode — currently
/// only YUYV422 (V4l2FrameSource) and BGRA8888 are wired up; yuv420_888
/// needs plane strides this flat-buffer [RawFrame] doesn't carry, and
/// mjpeg needs a JPEG decode neither producer in this repo emits yet.
FaceImage _toFaceImage(RawFrame frame) {
  switch (frame.format) {
    case PixelFormat.yuyv422:
      return yuyv422ToFaceImage(
        yuyv: frame.bytes,
        width: frame.width,
        height: frame.height,
      );
    case PixelFormat.bgra8888:
      return bgraToFaceImage(
        bgra: frame.bytes,
        width: frame.width,
        height: frame.height,
      );
    case PixelFormat.yuv420_888:
    case PixelFormat.mjpeg:
      throw StateError(
        'realtime loop: ${frame.format}는 아직 지원하지 않음 (V4l2FrameSource는 yuyv422만 냄)',
      );
  }
}

/// Consumes [source].frames() one at a time — never starting a frame while
/// the previous one is still being processed, so a slow detect/embed never
/// builds a backlog against the camera's own frame rate (S4 benchmark:
/// ~350ms/frame on 4 threads vs. a ~33-40ms camera frame interval, so this
/// loop is expected to process well under the camera's native FPS; that's
/// an accepted trade-off, not a bug — see S6 DoD "성능" in
/// linux_port_scope.md).
///
/// Stops when [duration] elapses (wall-clock, measured from loop start),
/// [stopSignal] completes, or [stopRequested] returns true after a frame.
/// A duration or stop signal also ends an idle stream; if a frame is being
/// processed, the loop waits for it to finish before returning. Errors from
/// the source or pipeline are rethrown. Neither closes [source]; the caller
/// owns that.
Future<RealtimeStats> runRealtimeLoop({
  required FrameSource source,
  required FaceDetector detector,
  required FaceAligner aligner,
  required FaceEmbedder embedder,
  required FaceMatcher matcher,
  required List<Enrollment> gallery,
  Duration? duration,
  Future<void>? stopSignal,
  bool Function()? stopRequested,
  void Function(RealtimeFrameResult result)? onResult,
  void Function(String message)? onProgress,
}) async {
  final stats = RealtimeStats()..rssStartBytes = ProcessInfo.currentRss;
  stats.rssMaxBytes = stats.rssStartBytes;

  int? lastSequence;
  final completer = Completer<void>();
  late StreamSubscription<RawFrame> sub;
  final loopStopwatch = Stopwatch()..start();
  var frameInProgress = false;
  var stopRequestedInternally = false;

  void complete() {
    if (!completer.isCompleted) completer.complete();
  }

  void fail(Object error, StackTrace stackTrace) {
    if (!completer.isCompleted) completer.completeError(error, stackTrace);
  }

  void requestStop() {
    stopRequestedInternally = true;
    if (!frameInProgress) complete();
  }

  void finishFrame() {
    frameInProgress = false;
    if (duration != null && loopStopwatch.elapsed >= duration) {
      stopRequestedInternally = true;
    }
    final stopped = stopRequestedInternally || (stopRequested?.call() ?? false);
    if (stopped) {
      complete();
    } else {
      sub.resume();
    }
  }

  sub = source.frames().listen(null);
  sub.onData((frame) async {
    sub.pause();
    frameInProgress = true;
    try {
      final rss = ProcessInfo.currentRss;
      if (rss > stats.rssMaxBytes) stats.rssMaxBytes = rss;

      if (frame.sequence != null) {
        if (lastSequence != null && frame.sequence! != lastSequence! + 1) {
          stats.droppedFrames += frame.sequence! - lastSequence! - 1;
        }
        lastSequence = frame.sequence;
      }

      final expected = _expectedFrameBytes(frame);
      if (expected != null && frame.bytes.length != expected) {
        stats.corruptFrames++;
        onProgress?.call(
          '프레임(seq=${frame.sequence}): 손상된 프레임, 건너뜀 '
          '(기대 $expected bytes, 실제 ${frame.bytes.length} bytes)',
        );
        finishFrame();
        return;
      }

      stats.framesSeen++;
      final frameIndex = stats.framesSeen;

      final totalSw = Stopwatch()..start();
      final image = _toFaceImage(frame);

      final detectSw = Stopwatch()..start();
      final detected = await detector.detect(image);
      final detectMs = detectSw.elapsedMilliseconds;

      if (detected.isEmpty) {
        stats.noFace++;
        onResult?.call(
          RealtimeFrameResult(
            frameIndex: frameIndex,
            sequence: frame.sequence,
            faceDetected: false,
            aligned: false,
            match: null,
            ms: {'detect': detectMs, 'total': totalSw.elapsedMilliseconds},
          ),
        );
        finishFrame();
        return;
      }

      final best = detected.reduce((a, b) => a.score >= b.score ? a : b);

      final alignSw = Stopwatch()..start();
      final aligned = aligner.align(image, best);
      final alignMs = alignSw.elapsedMilliseconds;

      if (aligned == null) {
        stats.alignRejected++;
        onResult?.call(
          RealtimeFrameResult(
            frameIndex: frameIndex,
            sequence: frame.sequence,
            faceDetected: true,
            aligned: false,
            match: null,
            ms: {
              'detect': detectMs,
              'align': alignMs,
              'total': totalSw.elapsedMilliseconds,
            },
          ),
        );
        finishFrame();
        return;
      }

      final embedSw = Stopwatch()..start();
      final embedding = await embedder.embed(aligned);
      final embedMs = embedSw.elapsedMilliseconds;

      final match = matcher.match(embedding, gallery);
      if (match.accepted) {
        stats.accepted++;
      } else {
        stats.rejected++;
      }

      onResult?.call(
        RealtimeFrameResult(
          frameIndex: frameIndex,
          sequence: frame.sequence,
          faceDetected: true,
          aligned: true,
          match: match,
          ms: {
            'detect': detectMs,
            'align': alignMs,
            'embed': embedMs,
            'total': totalSw.elapsedMilliseconds,
          },
        ),
      );
      finishFrame();
    } catch (e, st) {
      frameInProgress = false;
      fail(e, st);
    }
  });
  sub.onError((Object e, StackTrace st) {
    fail(e, st);
  });
  sub.onDone(complete);

  final durationTimer = duration == null ? null : Timer(duration, requestStop);
  stopSignal?.then((_) => requestStop(), onError: fail);

  try {
    await completer.future;
    return stats;
  } finally {
    durationTimer?.cancel();
    await sub.cancel();
  }
}

/// Enrolls the most prominent face in [image] (detect→align→embed, same as
/// a single-shot facekit_cli run) as [id]. Throws [StateError] if no face
/// is found or the pose gate rejects it — enrollment needs one clean shot,
/// unlike the main loop which is built to shrug off a bad frame.
Future<Enrollment> enrollFromImage({
  required FaceImage image,
  required FaceDetector detector,
  required FaceAligner aligner,
  required FaceEmbedder embedder,
  required String id,
}) async {
  final detected = await detector.detect(image);
  if (detected.isEmpty) {
    throw StateError('등록용 이미지에서 얼굴을 찾지 못함');
  }
  final best = detected.reduce((a, b) => a.score >= b.score ? a : b);
  final aligned = aligner.align(image, best);
  if (aligned == null) {
    throw StateError('등록용 이미지 정렬 실패(포즈 게이트) — 더 정면에 가까운 사진을 쓸 것');
  }
  final embedding = await embedder.embed(aligned);
  return Enrollment(id: id, embedding: embedding);
}
