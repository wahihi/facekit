// Headless detect→align→embed over a single still image — the reusable
// core behind bin/facekit_cli.dart, factored out so test/golden_vector_test.dart
// can call it directly instead of shelling out to the compiled binary.

import 'dart:io';

import 'package:image/image.dart' as img;

import 'package:facekit_core/facekit_core.dart';

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
}) async {
  final image = decodeImageFile(imagePath);

  final detectorManifest = loadManifest('$detectorDir/manifest.json');
  final embedderManifest = loadManifest('$embedderDir/manifest.json');

  final detector = YuNetDetector.fromFile(
    tflitePath: '$detectorDir/${detectorManifest.file}',
    manifest: detectorManifest,
    tfliteLibraryPath: libraryPath,
  );
  final embedder = TfliteFaceEmbedder.fromFile(
    tflitePath: '$embedderDir/${embedderManifest.file}',
    manifest: embedderManifest,
    tfliteLibraryPath: libraryPath,
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
