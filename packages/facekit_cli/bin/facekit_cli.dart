// facekit_cli — headless detect→align→embed over a single still image.
// No Flutter; runs on any platform with a Dart SDK + libtensorflowlite_c.
//
// Usage:
//   dart run bin/facekit_cli.dart \
//     --image path/to/photo.jpg \
//     --detector ../../assets/models/yunet_160 \
//     --embedder ../../example/assets/models/auraface \
//     [--tflite-library path/to/libtensorflowlite_c.so] \
//     [--json]
//
// --detector/--embedder are directories containing a manifest.json and the
// .tflite file it names (manifest.file) — the same layout as assets/models/*.
//
// Output (stdout): one JSON object per run —
//   {"faces":[{"bbox":[l,t,r,b],"score":..,"embedding":[512 floats],
//              "ms":{"detect":..,"align":..,"embed":..}}]}
// or {"faces":[]} if no face was detected. Everything else (progress,
// errors) goes to stderr so stdout stays parseable.

import 'dart:convert';
import 'dart:io';

import 'package:args/args.dart';
import 'package:image/image.dart' as img;

import 'package:facekit_core/facekit_core.dart';

Future<void> main(List<String> arguments) async {
  final parser = ArgParser()
    ..addOption('image', help: '입력 이미지 경로 (JPEG/PNG 등, package:image가 지원하는 포맷)')
    ..addOption('detector', help: 'manifest.json + .tflite가 있는 검출기 모델 디렉토리')
    ..addOption('embedder', help: 'manifest.json + .tflite가 있는 임베더 모델 디렉토리')
    ..addOption('tflite-library', help: 'libtensorflowlite_c 경로 (미지정 시 FACEKIT_TFLITE_C_LIBRARY 환경변수 또는 실행파일 옆 기본 경로)')
    ..addFlag('json', help: '결과를 JSON 한 줄로 stdout에 출력', defaultsTo: true)
    ..addFlag('help', abbr: 'h', negatable: false);

  final ArgResults args;
  try {
    args = parser.parse(arguments);
  } on FormatException catch (e) {
    stderr.writeln('인자 오류: ${e.message}\n');
    stderr.writeln(parser.usage);
    exitCode = 64;
    return;
  }

  if (args['help'] as bool ||
      args['image'] == null ||
      args['detector'] == null ||
      args['embedder'] == null) {
    stderr.writeln('facekit_cli — headless detect→align→embed\n');
    stderr.writeln(parser.usage);
    exitCode = args['help'] as bool ? 0 : 64;
    return;
  }

  final imagePath = args['image'] as String;
  final detectorDir = args['detector'] as String;
  final embedderDir = args['embedder'] as String;
  final libraryPath = args['tflite-library'] as String?;

  final image = _decodeImage(imagePath);

  final detectorManifest = _loadManifest('$detectorDir/manifest.json');
  final embedderManifest = _loadManifest('$embedderDir/manifest.json');

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

  if (detected.isEmpty) {
    stdout.writeln(jsonEncode({'faces': []}));
    return;
  }

  final best = detected.reduce((a, b) => a.score >= b.score ? a : b);

  sw.reset();
  final aligned = aligner.align(image, best);
  final alignMs = sw.elapsedMilliseconds;

  if (aligned == null) {
    stderr.writeln('경고: 정렬 실패(포즈 게이트 거부) — bbox/score만 출력');
    stdout.writeln(jsonEncode({
      'faces': [
        {
          'bbox': [best.boundingBox.left, best.boundingBox.top, best.boundingBox.right, best.boundingBox.bottom],
          'score': best.score,
          'embedding': null,
          'ms': {'detect': detectMs, 'align': alignMs, 'embed': null},
        },
      ],
    }));
    return;
  }

  sw.reset();
  final embedding = await embedder.embed(aligned);
  final embedMs = sw.elapsedMilliseconds;

  stdout.writeln(jsonEncode({
    'faces': [
      {
        'bbox': [best.boundingBox.left, best.boundingBox.top, best.boundingBox.right, best.boundingBox.bottom],
        'score': best.score,
        'embedding': embedding.vector,
        'ms': {'detect': detectMs, 'align': alignMs, 'embed': embedMs},
      },
    ],
  }));

  embedder.dispose();
}

ModelManifest _loadManifest(String path) {
  final file = File(path);
  if (!file.existsSync()) {
    stderr.writeln('모델 매니페스트를 찾을 수 없음: $path');
    exit(66);
  }
  final manifest = ModelManifest.fromJsonString(file.readAsStringSync());
  manifest.validate();
  return manifest;
}

/// Decodes [path] (any format package:image supports) into an RGB888
/// FaceImage — facekit_core's own image type, decoder-agnostic.
FaceImage _decodeImage(String path) {
  final file = File(path);
  if (!file.existsSync()) {
    stderr.writeln('이미지 파일을 찾을 수 없음: $path');
    exit(66);
  }
  final decoded = img.decodeImage(file.readAsBytesSync());
  if (decoded == null) {
    stderr.writeln('이미지 디코딩 실패(지원하지 않는 포맷?): $path');
    exit(65);
  }
  final rgb = decoded.convert(numChannels: 3);
  return FaceImage(
    rgbBytes: rgb.getBytes(order: img.ChannelOrder.rgb),
    width: rgb.width,
    height: rgb.height,
  );
}
