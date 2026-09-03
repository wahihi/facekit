// facekit_cli — headless detect→align→embed over a single still image.
// No Flutter; runs on any platform with a Dart SDK + libtensorflowlite_c.
//
// Usage:
//   dart run bin/facekit_cli.dart \
//     --image path/to/photo.jpg \
//     --detector ../../assets/models/yunet_160 \
//     --embedder ../../example/assets/models/auraface \
//     [--tflite-library path/to/libtensorflowlite_c.so] \
//     [--bench 30]
//
// --detector/--embedder are directories containing a manifest.json and the
// .tflite file it names (manifest.file) — the same layout as assets/models/*.
//
// Output (stdout): one JSON object per run —
//   {"faces":[{"bbox":[l,t,r,b],"score":..,"embedding":[512 floats],
//              "ms":{"detect":..,"align":..,"embed":..}}]}
// or {"faces":[]} if no face was detected. Everything else (progress,
// errors) goes to stderr so stdout stays parseable.
//
// With --bench N: loads the image/models once, runs detect->align->embed
// N times (+3 warmup iterations, discarded) on the same input, and prints
// per-stage median/mean/min/max ms instead —
//   {"bench":{"n":30,"detect":{"median_ms":..,...},"align":{...},
//             "embed":{...},"total":{...}}}

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
    ..addOption('bench', help: '단발 실행 대신, 같은 이미지/모델로 N회 반복해 단계별 ms 통계 출력 (워밍업 3회 별도)')
    ..addOption('num-threads', defaultsTo: '1', help: 'TFLite 인터프리터 CPU 스레드 수')
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

  final benchArg = args['bench'] as String?;
  final numThreads = int.tryParse(args['num-threads'] as String) ?? 1;
  if (numThreads < 1) {
    stderr.writeln('--num-threads는 1 이상의 정수여야 함: ${args['num-threads']}');
    exit(64);
  }

  try {
    if (benchArg != null) {
      final iterations = int.tryParse(benchArg);
      if (iterations == null || iterations < 1) {
        stderr.writeln('--bench는 1 이상의 정수여야 함: $benchArg');
        exit(64);
      }
      final result = await runBenchmark(
        imagePath: args['image'] as String,
        detectorDir: args['detector'] as String,
        embedderDir: args['embedder'] as String,
        libraryPath: args['tflite-library'] as String?,
        iterations: iterations,
        numThreads: numThreads,
      );
      stdout.writeln(jsonEncode({'bench': result.toJson()}));
    } else {
      final results = await runDetectAlignEmbed(
        imagePath: args['image'] as String,
        detectorDir: args['detector'] as String,
        embedderDir: args['embedder'] as String,
        libraryPath: args['tflite-library'] as String?,
        numThreads: numThreads,
      );
      stdout.writeln(jsonEncode({'faces': results.map((r) => r.toJson()).toList()}));
    }
  } on StateError catch (e) {
    stderr.writeln(e.message);
    exit(66);
  }
}
