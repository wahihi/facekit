// On-device timing harness for doc/KR/benchmark.md's "실기기 수치 별도 측정" gap.
//
// Re-runs detect→align→embed on a single already-captured FaceImage (the
// caller must supply one with a face in it) so every iteration does the same
// work — only wall-clock varies. Embedding runs via Isolate.run, matching
// FacePipeline's real per-frame cost (isolate spawn included), not a bare
// inference microbenchmark.
import 'dart:isolate';

import 'package:facekit/facekit.dart';

class StageStats {
  final double avgMs;
  final double p50Ms;
  final double p95Ms;

  const StageStats({
    required this.avgMs,
    required this.p50Ms,
    required this.p95Ms,
  });

  factory StageStats.fromSamplesMs(List<double> samples) {
    final sorted = [...samples]..sort();
    final avg = sorted.reduce((a, b) => a + b) / sorted.length;
    double percentile(double p) =>
        sorted[((sorted.length - 1) * p).round()];
    return StageStats(
      avgMs: avg,
      p50Ms: percentile(0.50),
      p95Ms: percentile(0.95),
    );
  }

  String get summary =>
      '평균 ${avgMs.toStringAsFixed(1)}ms / p50 ${p50Ms.toStringAsFixed(1)}ms / p95 ${p95Ms.toStringAsFixed(1)}ms';
}

class BenchmarkResult {
  final StageStats detect;
  final StageStats embed;
  final StageStats total;
  final int sampleCount;

  const BenchmarkResult({
    required this.detect,
    required this.embed,
    required this.total,
    required this.sampleCount,
  });

  String toReportText() => '''
n=$sampleCount (warmup 제외)
검출: ${detect.summary}
임베딩: ${embed.summary}
전체 1프레임(검출+정렬+임베딩): ${total.summary}
''';
}

/// Throws [StateError] if [image] stops yielding a detected face mid-run —
/// shouldn't happen since [image] is a frozen single frame, but the model
/// call itself could still fail (e.g. interpreter error), so callers should
/// still wrap this in a try/catch.
Future<BenchmarkResult> runBenchmark({
  required FaceDetector detector,
  required FaceAligner aligner,
  required FaceEmbedder embedder,
  required FaceImage image,
  int iterations = 30,
  int warmup = 5,
}) async {
  final detectMs = <double>[];
  final embedMs = <double>[];
  final totalMs = <double>[];

  for (var i = 0; i < warmup + iterations; i++) {
    final frameSw = Stopwatch()..start();

    final detectSw = Stopwatch()..start();
    final faces = await detector.detect(image);
    final detectElapsed = detectSw.elapsedMicroseconds / 1000.0;
    if (faces.isEmpty) {
      throw StateError('벤치마크용 캡처 프레임에서 얼굴을 잃었습니다. 다시 시도해주세요.');
    }

    final face = faces.reduce((a, b) => a.score >= b.score ? a : b);
    final aligned = aligner.align(image, face);

    final embedSw = Stopwatch()..start();
    await Isolate.run(() => embedder.embed(aligned));
    final embedElapsed = embedSw.elapsedMicroseconds / 1000.0;

    final totalElapsed = frameSw.elapsedMicroseconds / 1000.0;

    if (i >= warmup) {
      detectMs.add(detectElapsed);
      embedMs.add(embedElapsed);
      totalMs.add(totalElapsed);
    }
  }

  return BenchmarkResult(
    detect: StageStats.fromSamplesMs(detectMs),
    embed: StageStats.fromSamplesMs(embedMs),
    total: StageStats.fromSamplesMs(totalMs),
    sampleCount: iterations,
  );
}
