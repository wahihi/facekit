// Headless detect→align→embed over a single still image — the reusable
// core behind bin/facekit_cli.dart, factored out so test/golden_vector_test.dart
// can call it directly instead of shelling out to the compiled binary.

import 'dart:io';

import 'package:image/image.dart' as img;

import 'package:facekit_core/facekit_core.dart';

export 'src/frame_source.dart';
export 'src/v4l2_frame_source.dart';

class FaceResult {
  final List<double> bbox; // [left, top, right, bottom]
  final double score;
  final List<double>? embedding; // null if alignment's pose gate rejected it
  final Map<String, int?> ms; // {'detect':.., 'align':.., 'embed':..}

  FaceResult({
    required this.bbox,
    required this.score,
    required this.embedding,
    required this.ms,
  });

  Map<String, dynamic> toJson() => {
        'bbox': bbox,
        'score': score,
        'embedding': embedding,
        'ms': ms,
      };
}

/// Runs YuNet detect → arcface112 align → embed on [imagePath], returning
/// the highest-confidence face's result (empty list if none detected).
///
/// [detectorDir]/[embedderDir] are directories containing a manifest.json
/// and the .tflite file it names — the same layout as assets/models/*.
/// [libraryPath] overrides libtensorflowlite_c discovery (see
/// TfliteFfiRunner.fromFile / bindings/bindings.dart).
Future<List<FaceResult>> runDetectAlignEmbed({
  required String imagePath,
  required String detectorDir,
  required String embedderDir,
  String? libraryPath,
  int numThreads = 1,
}) async {
  final image = decodeImageFile(imagePath);

  final detectorManifest = loadManifest('$detectorDir/manifest.json');
  final embedderManifest = loadManifest('$embedderDir/manifest.json');

  final detector = YuNetDetector.fromFile(
    tflitePath: '$detectorDir/${detectorManifest.file}',
    manifest: detectorManifest,
    tfliteLibraryPath: libraryPath,
    numThreads: numThreads,
  );
  final embedder = TfliteFaceEmbedder.fromFile(
    tflitePath: '$embedderDir/${embedderManifest.file}',
    manifest: embedderManifest,
    tfliteLibraryPath: libraryPath,
    numThreads: numThreads,
  );
  final aligner = AffineAligner.arcface112(nativeFivePoint: true);

  final sw = Stopwatch()..start();
  final detected = await detector.detect(image);
  final detectMs = sw.elapsedMilliseconds;

  if (detected.isEmpty) return [];

  final best = detected.reduce((a, b) => a.score >= b.score ? a : b);
  final bbox = [
    best.boundingBox.left,
    best.boundingBox.top,
    best.boundingBox.right,
    best.boundingBox.bottom,
  ];

  sw.reset();
  final aligned = aligner.align(image, best);
  final alignMs = sw.elapsedMilliseconds;

  if (aligned == null) {
    return [
      FaceResult(
        bbox: bbox,
        score: best.score,
        embedding: null,
        ms: {'detect': detectMs, 'align': alignMs, 'embed': null},
      ),
    ];
  }

  sw.reset();
  final embedding = await embedder.embed(aligned);
  final embedMs = sw.elapsedMilliseconds;
  embedder.dispose();

  return [
    FaceResult(
      bbox: bbox,
      score: best.score,
      embedding: embedding.vector,
      ms: {'detect': detectMs, 'align': alignMs, 'embed': embedMs},
    ),
  ];
}

class StageStats {
  final double medianMs, meanMs, minMs, maxMs;
  StageStats({required this.medianMs, required this.meanMs, required this.minMs, required this.maxMs});

  factory StageStats.fromMicros(List<int> micros) {
    final ms = micros.map((u) => u / 1000.0).toList()..sort();
    final mid = ms.length ~/ 2;
    final median = ms.length.isOdd ? ms[mid] : (ms[mid - 1] + ms[mid]) / 2;
    return StageStats(
      medianMs: median,
      meanMs: ms.reduce((a, b) => a + b) / ms.length,
      minMs: ms.first,
      maxMs: ms.last,
    );
  }

  Map<String, dynamic> toJson() => {
        'median_ms': medianMs,
        'mean_ms': meanMs,
        'min_ms': minMs,
        'max_ms': maxMs,
      };
}

class BenchResult {
  final int n;
  final StageStats detect, align, embed, total;
  BenchResult({required this.n, required this.detect, required this.align, required this.embed, required this.total});

  Map<String, dynamic> toJson() => {
        'n': n,
        'detect': detect.toJson(),
        'align': align.toJson(),
        'embed': embed.toJson(),
        'total': total.toJson(),
      };
}

/// Repeatedly runs detect→align→embed on the same (already-decoded) image
/// with the same (already-loaded) models, warming up [warmup] iterations
/// before recording [iterations] samples per stage. Loads everything once,
/// unlike calling [runDetectAlignEmbed] in a loop — S4 DoD (doc/KR/
/// linux_port_scope.md): real per-stage timing on the target board, n>=30.
Future<BenchResult> runBenchmark({
  required String imagePath,
  required String detectorDir,
  required String embedderDir,
  String? libraryPath,
  required int iterations,
  int warmup = 3,
  int numThreads = 1,
}) async {
  final image = decodeImageFile(imagePath);

  final detectorManifest = loadManifest('$detectorDir/manifest.json');
  final embedderManifest = loadManifest('$embedderDir/manifest.json');

  final detector = YuNetDetector.fromFile(
    tflitePath: '$detectorDir/${detectorManifest.file}',
    manifest: detectorManifest,
    tfliteLibraryPath: libraryPath,
    numThreads: numThreads,
  );
  final embedder = TfliteFaceEmbedder.fromFile(
    tflitePath: '$embedderDir/${embedderManifest.file}',
    manifest: embedderManifest,
    tfliteLibraryPath: libraryPath,
    numThreads: numThreads,
  );
  final aligner = AffineAligner.arcface112(nativeFivePoint: true);

  final detectUs = <int>[], alignUs = <int>[], embedUs = <int>[], totalUs = <int>[];
  final sw = Stopwatch();
  final swTotal = Stopwatch();

  for (var i = 0; i < warmup + iterations; i++) {
    swTotal
      ..reset()
      ..start();

    sw
      ..reset()
      ..start();
    final detected = await detector.detect(image);
    final dUs = sw.elapsedMicroseconds;
    if (detected.isEmpty) {
      throw StateError('벤치마크용 이미지에서 얼굴을 못 찾음: $imagePath');
    }
    final best = detected.reduce((a, b) => a.score >= b.score ? a : b);

    sw
      ..reset()
      ..start();
    final aligned = aligner.align(image, best);
    final aUs = sw.elapsedMicroseconds;
    if (aligned == null) {
      throw StateError('벤치마크용 이미지 정렬 실패(포즈 게이트): $imagePath');
    }

    sw
      ..reset()
      ..start();
    await embedder.embed(aligned);
    final eUs = sw.elapsedMicroseconds;

    final totalUsThisRun = swTotal.elapsedMicroseconds;

    if (i >= warmup) {
      detectUs.add(dUs);
      alignUs.add(aUs);
      embedUs.add(eUs);
      totalUs.add(totalUsThisRun);
    }
  }
  embedder.dispose();

  return BenchResult(
    n: iterations,
    detect: StageStats.fromMicros(detectUs),
    align: StageStats.fromMicros(alignUs),
    embed: StageStats.fromMicros(embedUs),
    total: StageStats.fromMicros(totalUs),
  );
}

ModelManifest loadManifest(String path) {
  final file = File(path);
  if (!file.existsSync()) {
    throw StateError('모델 매니페스트를 찾을 수 없음: $path');
  }
  final manifest = ModelManifest.fromJsonString(file.readAsStringSync());
  manifest.validate();
  return manifest;
}

/// Decodes [path] (any format package:image supports) into an RGB888
/// FaceImage — facekit_core's own image type, decoder-agnostic.
FaceImage decodeImageFile(String path) {
  final file = File(path);
  if (!file.existsSync()) {
    throw StateError('이미지 파일을 찾을 수 없음: $path');
  }
  final decoded = img.decodeImage(file.readAsBytesSync());
  if (decoded == null) {
    throw StateError('이미지 디코딩 실패(지원하지 않는 포맷?): $path');
  }
  final rgb = decoded.convert(numChannels: 3);
  return FaceImage(
    rgbBytes: rgb.getBytes(order: img.ChannelOrder.rgb),
    width: rgb.width,
    height: rgb.height,
  );
}
