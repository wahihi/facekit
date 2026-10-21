// S6 (doc/KR/linux_port_scope.md) — exercises runRealtimeLoop end to end
// against the real YuNet+AuraFace models (same fixtures as
// golden_vector_test.dart), through a FakeFrameSource instead of a real
// camera, so the wiring (frame → convert → detect → align → embed →
// match → stats) is verified without needing V4L2 hardware.
//
// Needs libtensorflowlite_c for this platform findable by TfliteFfiRunner
// — same pre-existing CI gap as golden_vector_test.dart and facekit_core's
// own model-loading smoke tests.
import 'dart:async';
import 'dart:typed_data';

import 'package:test/test.dart';

import 'package:facekit_cli/facekit_cli.dart';
import 'package:facekit_core/facekit_core.dart';

/// Replays a fixed list of frames, then ends the stream — lets the test
/// drive runRealtimeLoop without a real V4L2 device.
class FakeFrameSource implements FrameSource {
  final List<RawFrame> _frames;
  FakeFrameSource(this._frames);

  @override
  Stream<RawFrame> frames() async* {
    for (final frame in _frames) {
      yield frame;
    }
  }

  @override
  Future<void> close() async {}
}

class _IdleFrameSource implements FrameSource {
  final StreamController<RawFrame> _controller = StreamController<RawFrame>();

  @override
  Stream<RawFrame> frames() => _controller.stream;

  @override
  Future<void> close() => _controller.close();
}

class _UnusedDetector implements FaceDetector {
  @override
  Future<List<DetectedFace>> detect(FaceImage image) async => const [];
}

class _ThrowingDetector implements FaceDetector {
  @override
  Future<List<DetectedFace>> detect(FaceImage image) async {
    throw StateError('detector failed');
  }
}

class _BlockingDetector implements FaceDetector {
  final started = Completer<void>();
  final release = Completer<void>();

  @override
  Future<List<DetectedFace>> detect(FaceImage image) async {
    started.complete();
    await release.future;
    return const [];
  }
}

class _UnusedAligner implements FaceAligner {
  @override
  AlignedFace? align(FaceImage image, DetectedFace face) => null;
}

class _UnusedEmbedder implements FaceEmbedder {
  @override
  Future<Embedding> embed(AlignedFace face) async =>
      throw StateError('embedder should not be called');
}

class _UnusedMatcher implements FaceMatcher {
  @override
  MatchResult match(Embedding probe, List<Enrollment> gallery) =>
      const MatchResult(matchedId: null, similarity: 0, accepted: false);
}

/// Expands an RGB888 FaceImage to a flat BGRA8888 buffer — the inverse of
/// bgraToFaceImage, used here only to manufacture a RawFrame from a real
/// photo for the fake source (alpha is unused by bgraToFaceImage, 0xFF).
Uint8List _rgbToBgra(FaceImage image) {
  final out = Uint8List(image.width * image.height * 4);
  for (var i = 0; i < image.width * image.height; i++) {
    final src = i * 3, dst = i * 4;
    out[dst] = image.rgbBytes[src + 2]; // B
    out[dst + 1] = image.rgbBytes[src + 1]; // G
    out[dst + 2] = image.rgbBytes[src]; // R
    out[dst + 3] = 0xFF;
  }
  return out;
}

RawFrame _bgraFrameFrom(FaceImage image, int? sequence) => RawFrame(
  bytes: _rgbToBgra(image),
  width: image.width,
  height: image.height,
  format: PixelFormat.bgra8888,
  sequence: sequence,
);

void main() {
  test(
    'detect→align→embed→match loop: corrupt-skip, drop-count, accept/reject over real models',
    () async {
      const detectorDir = '../../assets/models/yunet_160';
      const embedderDir = '../../example/assets/models/auraface';

      final detectorManifest = loadManifest('$detectorDir/manifest.json');
      final embedderManifest = loadManifest('$embedderDir/manifest.json');

      final detector = YuNetDetector.fromFile(
        tflitePath: '$detectorDir/${detectorManifest.file}',
        manifest: detectorManifest,
      );
      final embedder = TfliteFaceEmbedder.fromFile(
        tflitePath: '$embedderDir/${embedderManifest.file}',
        manifest: embedderManifest,
      );
      final aligner = AffineAligner.arcface112(nativeFivePoint: true);
      final matcher = CosineMatcher.fromManifest(embedderManifest);

      addTearDown(() {
        detector.dispose();
        embedder.dispose();
      });

      final me1 = decodeImageFile('../../tool/model_verification/me_1.jpg');
      final me2 = decodeImageFile('../../tool/model_verification/me_2.jpg');
      final other1 = decodeImageFile(
        '../../tool/model_verification/other_1.jpg',
      );

      final gallery = [
        await enrollFromImage(
          image: me1,
          detector: detector,
          aligner: aligner,
          embedder: embedder,
          id: 'me',
        ),
      ];

      // seq 1: corrupt (declares bgra8888 but is 4 bytes short of w*h*4).
      final corrupt = RawFrame(
        bytes: Uint8List(me2.width * me2.height * 4 - 4),
        width: me2.width,
        height: me2.height,
        format: PixelFormat.bgra8888,
        sequence: 1,
      );
      // seq 3 (seq 2 never arrives → one dropped frame): me_2, same person as the gallery.
      final samePerson = _bgraFrameFrom(me2, 3);
      // seq 4: other_1, a different person.
      final differentPerson = _bgraFrameFrom(other1, 4);

      final results = <RealtimeFrameResult>[];
      final progress = <String>[];

      final stats = await runRealtimeLoop(
        source: FakeFrameSource([corrupt, samePerson, differentPerson]),
        detector: detector,
        aligner: aligner,
        embedder: embedder,
        matcher: matcher,
        gallery: gallery,
        onResult: results.add,
        onProgress: progress.add,
      );

      expect(stats.corruptFrames, 1, reason: '$progress');
      expect(stats.droppedFrames, 1); // seq 1 → 3, seq 2 missing
      expect(stats.framesSeen, 2); // corrupt frame excluded
      expect(stats.noFace, 0);
      expect(stats.alignRejected, 0);
      expect(stats.accepted, 1); // me_2 vs me_1 gallery
      expect(stats.rejected, 1); // other_1 vs me_1 gallery

      expect(results, hasLength(2));
      expect(results[0].match!.accepted, isTrue);
      expect(results[0].match!.matchedId, 'me');
      expect(results[1].match!.accepted, isFalse);
    },
  );

  test(
    'a stream with no frames completes cleanly with all-zero stats',
    () async {
      const detectorDir = '../../assets/models/yunet_160';
      const embedderDir = '../../example/assets/models/auraface';
      final detectorManifest = loadManifest('$detectorDir/manifest.json');
      final embedderManifest = loadManifest('$embedderDir/manifest.json');

      final detector = YuNetDetector.fromFile(
        tflitePath: '$detectorDir/${detectorManifest.file}',
        manifest: detectorManifest,
      );
      final embedder = TfliteFaceEmbedder.fromFile(
        tflitePath: '$embedderDir/${embedderManifest.file}',
        manifest: embedderManifest,
      );
      addTearDown(() {
        detector.dispose();
        embedder.dispose();
      });

      final stats = await runRealtimeLoop(
        source: FakeFrameSource(const []),
        detector: detector,
        aligner: AffineAligner.arcface112(nativeFivePoint: true),
        embedder: embedder,
        matcher: CosineMatcher.fromManifest(embedderManifest),
        gallery: const [],
      );

      expect(stats.framesSeen, 0);
      expect(stats.corruptFrames, 0);
      expect(stats.droppedFrames, 0);
    },
  );

  test('stopRequested stops the loop even with frames still queued', () async {
    const detectorDir = '../../assets/models/yunet_160';
    const embedderDir = '../../example/assets/models/auraface';
    final detectorManifest = loadManifest('$detectorDir/manifest.json');
    final embedderManifest = loadManifest('$embedderDir/manifest.json');

    final detector = YuNetDetector.fromFile(
      tflitePath: '$detectorDir/${detectorManifest.file}',
      manifest: detectorManifest,
    );
    final embedder = TfliteFaceEmbedder.fromFile(
      tflitePath: '$embedderDir/${embedderManifest.file}',
      manifest: embedderManifest,
    );
    final aligner = AffineAligner.arcface112(nativeFivePoint: true);
    addTearDown(() {
      detector.dispose();
      embedder.dispose();
    });

    final me1 = decodeImageFile('../../tool/model_verification/me_1.jpg');
    final me2 = decodeImageFile('../../tool/model_verification/me_2.jpg');

    var framesHandled = 0;
    final stats = await runRealtimeLoop(
      source: FakeFrameSource([
        _bgraFrameFrom(me1, 1),
        _bgraFrameFrom(me2, 2),
        _bgraFrameFrom(me2, 3),
      ]),
      detector: detector,
      aligner: aligner,
      embedder: embedder,
      matcher: CosineMatcher.fromManifest(embedderManifest),
      gallery: const [],
      onResult: (_) => framesHandled++,
      stopRequested: () => framesHandled >= 1,
    );

    expect(
      stats.framesSeen,
      1,
      reason: 'stopRequested should cut the loop short after the first frame',
    );
  });

  test('duration stops an idle source without waiting for a frame', () async {
    final source = _IdleFrameSource();
    final stats = await runRealtimeLoop(
      source: source,
      detector: _UnusedDetector(),
      aligner: _UnusedAligner(),
      embedder: _UnusedEmbedder(),
      matcher: _UnusedMatcher(),
      gallery: const [],
      duration: const Duration(milliseconds: 20),
    );

    expect(stats.framesSeen, 0);
    await source.close();
  });

  test('duration expires during a frame and stops after that frame', () async {
    final detector = _BlockingDetector();
    final loop = runRealtimeLoop(
      source: FakeFrameSource([
        for (var sequence = 1; sequence <= 3; sequence++)
          RawFrame(
            bytes: Uint8List(16),
            width: 2,
            height: 2,
            format: PixelFormat.bgra8888,
            sequence: sequence,
          ),
      ]),
      detector: detector,
      aligner: _UnusedAligner(),
      embedder: _UnusedEmbedder(),
      matcher: _UnusedMatcher(),
      gallery: const [],
      duration: const Duration(milliseconds: 10),
    );

    await detector.started.future;
    await Future<void>.delayed(const Duration(milliseconds: 30));
    detector.release.complete();

    final stats = await loop;
    expect(stats.framesSeen, 1);
  });

  test('stopSignal stops an idle source cleanly', () async {
    final source = _IdleFrameSource();
    final stopSignal = Completer<void>();
    final loop = runRealtimeLoop(
      source: source,
      detector: _UnusedDetector(),
      aligner: _UnusedAligner(),
      embedder: _UnusedEmbedder(),
      matcher: _UnusedMatcher(),
      gallery: const [],
      stopSignal: stopSignal.future,
    );
    stopSignal.complete();

    final stats = await loop;
    expect(stats.framesSeen, 0);
    await source.close();
  });

  test('stopSignal waits for the in-progress frame to finish', () async {
    final detector = _BlockingDetector();
    final stopSignal = Completer<void>();
    final loop = runRealtimeLoop(
      source: FakeFrameSource([
        RawFrame(
          bytes: Uint8List(16),
          width: 2,
          height: 2,
          format: PixelFormat.bgra8888,
        ),
      ]),
      detector: detector,
      aligner: _UnusedAligner(),
      embedder: _UnusedEmbedder(),
      matcher: _UnusedMatcher(),
      gallery: const [],
      stopSignal: stopSignal.future,
    );
    await detector.started.future;
    stopSignal.complete();

    var loopCompleted = false;
    loop.then((_) => loopCompleted = true);
    await Future<void>.delayed(Duration.zero);
    expect(loopCompleted, isFalse);

    detector.release.complete();
    final stats = await loop;
    expect(stats.framesSeen, 1);
  });

  test(
    'source errors are propagated instead of returning success stats',
    () async {
      final source = _FailingFrameSource();

      await expectLater(
        runRealtimeLoop(
          source: source,
          detector: _UnusedDetector(),
          aligner: _UnusedAligner(),
          embedder: _UnusedEmbedder(),
          matcher: _UnusedMatcher(),
          gallery: const [],
        ),
        throwsA(isA<StateError>()),
      );
    },
  );

  test(
    'pipeline errors are propagated instead of returning success stats',
    () async {
      final source = FakeFrameSource([
        RawFrame(
          bytes: Uint8List(16),
          width: 2,
          height: 2,
          format: PixelFormat.bgra8888,
        ),
      ]);

      await expectLater(
        runRealtimeLoop(
          source: source,
          detector: _ThrowingDetector(),
          aligner: _UnusedAligner(),
          embedder: _UnusedEmbedder(),
          matcher: _UnusedMatcher(),
          gallery: const [],
        ),
        throwsA(isA<StateError>()),
      );
    },
  );
}

class _FailingFrameSource implements FrameSource {
  @override
  Stream<RawFrame> frames() =>
      Stream<RawFrame>.error(StateError('source failed'));

  @override
  Future<void> close() async {}
}
