// Orchestrates the full face recognition pipeline.
// Heavy inference runs in a Dart isolate to avoid dropping UI frames.
// Source: design spec §7 Pipeline + §5 R5.

import 'dart:convert' show base64Encode;
import 'dart:isolate';
import 'package:flutter/foundation.dart' show debugPrint, kDebugMode;
import '../core/contracts.dart';
import '../core/debug_flags.dart';
import '../core/models.dart';

// DEBUG (see core/debug_flags.dart) — prints the aligned-face RGB crop as
// base64, chunked across many short debugPrint lines, so it can be
// reconstructed from a plain `flutter run` console copy-paste. Three other
// approaches were tried and failed on a real device: a clipboard round-trip
// silently truncated a 37KB string, writing to app-external storage was
// blocked (scoped storage requires the Android Context API, not plain
// dart:io), and a loopback HttpServer.bind failed with EPERM (device policy
// blocked server sockets). Only dumps once per label per app run
// (identify() fires every frame once liveness passes, and re-dumping every
// frame would flood the log).
const _debugChunkSize = 500;
bool _debugEnrollDumped = false;
bool _debugIdentifyDumped = false;

void _debugDumpAlignedFace(String label, AlignedFace face) {
  if (!kFacekitVerboseDebug) return;
  final alreadyDumped = label == 'enroll' ? _debugEnrollDumped : _debugIdentifyDumped;
  if (alreadyDumped) return;
  if (label == 'enroll') {
    _debugEnrollDumped = true;
  } else {
    _debugIdentifyDumped = true;
  }

  final b64 = base64Encode(face.rgbBytes);
  final chunks = <String>[];
  for (var i = 0; i < b64.length; i += _debugChunkSize) {
    final end = (i + _debugChunkSize < b64.length) ? i + _debugChunkSize : b64.length;
    chunks.add(b64.substring(i, end));
  }
  debugPrint(
    '[FacePipeline] TEMP DEBUG BEGIN $label '
    '(${face.size}x${face.size}, ${face.rgbBytes.length} bytes, ${chunks.length} chunks)',
  );
  for (var i = 0; i < chunks.length; i++) {
    debugPrint('[FacePipeline] TEMP DEBUG CHUNK $label ${i + 1}/${chunks.length} ${chunks[i]}');
  }
  debugPrint('[FacePipeline] TEMP DEBUG END $label');
}

// DEBUG (see core/debug_flags.dart) — logs the raw detector landmarks
// (pre-alignment) every time, to check whether their order/positions (e.g.
// left/right eye) stay consistent between calls — useful for diagnosing
// camera-orientation/alignment issues.
void _debugLogLandmarks(String label, FaceImage image, DetectedFace face) {
  if (!kFacekitVerboseDebug) return;
  final box = face.boundingBox;
  final pts = face.landmarks
      .asMap()
      .entries
      .map((e) => '${e.key}:(${e.value.x.toStringAsFixed(1)},${e.value.y.toStringAsFixed(1)})')
      .join(' ');
  debugPrint(
    '[FacePipeline] TEMP DEBUG LANDMARKS $label '
    'imageSize=${image.width}x${image.height} '
    'bbox=(${box.left.toStringAsFixed(1)},${box.top.toStringAsFixed(1)})'
    '-(${box.right.toStringAsFixed(1)},${box.bottom.toStringAsFixed(1)}) '
    'score=${face.score.toStringAsFixed(3)} pts=$pts',
  );
}

/// Full pipeline: detect → align → embed → match.
///
/// Construct once and reuse. Call [dispose] when done.
class FacePipeline {
  final FaceDetector detector;
  final FaceAligner aligner;
  final FaceEmbedder embedder;
  final FaceMatcher matcher;

  const FacePipeline({
    required this.detector,
    required this.aligner,
    required this.embedder,
    required this.matcher,
  });

  /// Identifies the most prominent face in [image] against [gallery].
  ///
  /// Returns null if no face is detected.
  /// Heavy work (embedding inference) runs in an isolate.
  Future<MatchResult?> identify(
    FaceImage image,
    List<Enrollment> gallery,
  ) async {
    final faces = await detector.detect(image);
    _log('detect: ${faces.length} face(s)');
    if (faces.isEmpty) return null;

    // Use the highest-confidence face only (first after descending sort by score).
    final best = faces.reduce((a, b) => a.score >= b.score ? a : b);
    _log('detect: best score=${best.score.toStringAsFixed(3)}');
    _debugLogLandmarks('identify', image, best);
    final aligned = aligner.align(image, best);
    _log('align: ${aligned.size}x${aligned.size} patch ready');
    _debugDumpAlignedFace('identify', aligned);
    final embedding = await _embedInIsolate(aligned);
    _log('embed: ${embedding.dim}-dim vector');
    final result = matcher.match(embedding, gallery);
    _log('match: id=${result.matchedId} similarity=${result.similarity.toStringAsFixed(3)} accepted=${result.accepted}');
    return result;
  }

  /// Enrolls the most prominent face in [image] and returns its embedding.
  ///
  /// Returns null if no face is detected.
  Future<Embedding?> enroll(FaceImage image) async {
    final faces = await detector.detect(image);
    _log('detect: ${faces.length} face(s)');
    if (faces.isEmpty) return null;

    final best = faces.reduce((a, b) => a.score >= b.score ? a : b);
    _log('detect: best score=${best.score.toStringAsFixed(3)}');
    _debugLogLandmarks('enroll', image, best);
    final aligned = aligner.align(image, best);
    _log('align: ${aligned.size}x${aligned.size} patch ready');
    _debugDumpAlignedFace('enroll', aligned);
    final embedding = await _embedInIsolate(aligned);
    _log('embed: ${embedding.dim}-dim vector');
    return embedding;
  }

  /// Diagnostic-only logging for following the pipeline step by step via
  /// `flutter logs`/`adb logcat -s flutter`. Debug-build only — never prints
  /// in release builds.
  void _log(String message) {
    if (kDebugMode) debugPrint('[FacePipeline] $message');
  }

  /// Runs [embedder.embed] in a separate isolate.
  ///
  /// Isolate communication uses [_IsolateEmbedRequest] to pass the aligned
  /// face bytes and a [SendPort] for the result. The embedder interface is
  /// not directly serialisable, so we run inference inside a closure that
  /// captures [embedder] (valid because the isolate shares heap in Dart 2+
  /// when using [Isolate.run]).
  Future<Embedding> _embedInIsolate(AlignedFace face) async {
    // Isolate.run is the idiomatic way since Dart 2.19 — spawns, runs, returns.
    return Isolate.run(() => embedder.embed(face));
  }
}

/// Lightweight pipeline that runs everything on the calling isolate.
/// Use in tests or when the caller already manages isolate scheduling.
class SyncFacePipeline {
  final FaceDetector detector;
  final FaceAligner aligner;
  final FaceEmbedder embedder;
  final FaceMatcher matcher;

  const SyncFacePipeline({
    required this.detector,
    required this.aligner,
    required this.embedder,
    required this.matcher,
  });

  Future<MatchResult?> identify(FaceImage image, List<Enrollment> gallery) async {
    final faces = await detector.detect(image);
    if (faces.isEmpty) return null;

    final best = faces.reduce((a, b) => a.score >= b.score ? a : b);
    final aligned = aligner.align(image, best);
    final embedding = await embedder.embed(aligned);
    return matcher.match(embedding, gallery);
  }

  Future<Embedding?> enroll(FaceImage image) async {
    final faces = await detector.detect(image);
    if (faces.isEmpty) return null;

    final best = faces.reduce((a, b) => a.score >= b.score ? a : b);
    final aligned = aligner.align(image, best);
    return embedder.embed(aligned);
  }
}
