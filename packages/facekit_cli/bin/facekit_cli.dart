// facekit_cli — headless detect→align→embed over a single still image.
// No Flutter; runs on any platform with a Dart SDK + libtensorflowlite_c.
//
// Usage:
//   dart run bin/facekit_cli.dart \
//     --image path/to/photo.jpg \
//     --detector ../../assets/models/yunet_160 \
//     --embedder ../../example/assets/models/auraface \
//     [--tflite-library path/to/libtensorflowlite_c.so]
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

import 'package:facekit_cli/facekit_cli.dart';

Future<void> main(List<String> arguments) async {
  final parser = ArgParser()
    ..addOption('image', help: '입력 이미지 경로 (JPEG/PNG 등, package:image가 지원하는 포맷)')
    ..addOption('detector', help: 'manifest.json + .tflite가 있는 검출기 모델 디렉토리')
    ..addOption('embedder', help: 'manifest.json + .tflite가 있는 임베더 모델 디렉토리')
    ..addOption('tflite-library', help: 'libtensorflowlite_c 경로 (미지정 시 FACEKIT_TFLITE_C_LIBRARY 환경변수 또는 실행파일 옆 기본 경로)')
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

  try {
    final results = await runDetectAlignEmbed(
      imagePath: args['image'] as String,
      detectorDir: args['detector'] as String,
      embedderDir: args['embedder'] as String,
      libraryPath: args['tflite-library'] as String?,
    );
    stdout.writeln(jsonEncode({'faces': results.map((r) => r.toJson()).toList()}));
  } on StateError catch (e) {
    stderr.writeln(e.message);
    exit(66);
  }
}
