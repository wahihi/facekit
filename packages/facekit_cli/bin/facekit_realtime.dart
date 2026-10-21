// facekit_realtime — S6 (doc/KR/linux_port_scope.md): the real-time
// recognition loop. Opens a V4L2 camera, enrolls one identity from a still
// image, then continuously detect→align→embed→match's each camera frame
// against that single-person gallery until stopped.
//
// Usage:
//   dart run bin/facekit_realtime.dart \
//     --device /dev/video2 --width 640 --height 480 \
//     --detector ../../assets/models/yunet_160 \
//     --embedder ../../example/assets/models/auraface \
//     --enroll-image tool/model_verification/me_1.jpg \
//     [--seconds 30] [--num-threads 4] \
//     [--tflite-library path] [--v4l2-library path]
//
// Output:
//   stdout — one JSON object per processed frame (see
//     RealtimeFrameResult.toJson), so a consumer can read results live,
//     line by line, while the loop runs.
//   stderr — human-readable progress (every 10th frame) and, at the end,
//     one JSON summary line (RealtimeStats.toJson) with the S6 DoD numbers
//     (frames/corrupt/dropped/accepted/rss).
//
// Stops on --seconds elapsing (omit to run until Ctrl+C) or on SIGINT —
// either way shuts down cleanly: closes the V4L2 device and disposes the
// TFLite interpreters before exiting.

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:args/args.dart';

import 'package:facekit_cli/facekit_cli.dart';
import 'package:facekit_core/facekit_core.dart';

Future<void> main(List<String> arguments) async {
  final parser = ArgParser()
    ..addOption('device', help: 'V4L2 장치 경로, 예: /dev/video2')
    ..addOption('width', defaultsTo: '640')
    ..addOption('height', defaultsTo: '480')
    ..addOption('detector', help: 'manifest.json + .tflite가 있는 검출기 모델 디렉토리')
    ..addOption('embedder', help: 'manifest.json + .tflite가 있는 임베더 모델 디렉토리')
    ..addOption(
      'enroll-image',
      help: '시작 시 이 사진에서 얼굴 1개를 등록 (이후 전체 세션의 유일한 갤러리 엔트리)',
    )
    ..addOption(
      'enroll-id',
      defaultsTo: 'enrolled',
      help: '등록할 식별자 (매칭 결과의 matched_id로 출력됨)',
    )
    ..addOption('tflite-library', help: 'libtensorflowlite_c 경로')
    ..addOption(
      'v4l2-library',
      help: 'libv4l2_capture.so 경로 (미지정 시 native/libv4l2_capture.so)',
    )
    ..addOption('num-threads', defaultsTo: '1', help: 'TFLite 인터프리터 CPU 스레드 수')
    ..addOption('seconds', help: '이 시간(초) 후 정상 종료 (미지정 시 Ctrl+C까지 계속)')
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
      args['device'] == null ||
      args['detector'] == null ||
      args['embedder'] == null ||
      args['enroll-image'] == null) {
    stderr.writeln('facekit_realtime — S6 실시간 인식 루프\n');
    stderr.writeln(parser.usage);
    exitCode = args['help'] as bool ? 0 : 64;
    return;
  }

  final numThreads = int.tryParse(args['num-threads'] as String) ?? 1;
  if (numThreads < 1) {
    stderr.writeln('--num-threads는 1 이상의 정수여야 함: ${args['num-threads']}');
    exit(64);
  }
  final secondsArg = args['seconds'] as String?;
  Duration? duration;
  if (secondsArg != null) {
    final seconds = int.tryParse(secondsArg);
    if (seconds == null || seconds < 1) {
      stderr.writeln('--seconds는 1 이상의 정수여야 함: $secondsArg');
      exit(64);
    }
    duration = Duration(seconds: seconds);
  }

  final libraryPath = args['tflite-library'] as String?;
  final detectorDir = args['detector'] as String;
  final embedderDir = args['embedder'] as String;

  YuNetDetector? detector;
  TfliteFaceEmbedder? embedder;
  V4l2FrameSource? source;
  StreamSubscription<ProcessSignal>? sigintSub;
  final sigintCompleter = Completer<void>();
  try {
    sigintSub = ProcessSignal.sigint.watch().listen((_) {
      if (sigintCompleter.isCompleted) {
        stderr.writeln('\nCtrl+C가 다시 들어왔습니다. 정리 완료까지 기다립니다.');
        return;
      }
      stderr.writeln('\nCtrl+C 받음 — 현재 작업을 마치고 정리 후 종료');
      exitCode = 130;
      sigintCompleter.complete();
    });

    final detectorManifest = loadManifest('$detectorDir/manifest.json');
    final embedderManifest = loadManifest('$embedderDir/manifest.json');

    final detectorInstance = YuNetDetector.fromFile(
      tflitePath: '$detectorDir/${detectorManifest.file}',
      manifest: detectorManifest,
      tfliteLibraryPath: libraryPath,
      numThreads: numThreads,
    );
    detector = detectorInstance;
    await Future<void>.delayed(Duration.zero);
    if (sigintCompleter.isCompleted) return;

    final embedderInstance = TfliteFaceEmbedder.fromFile(
      tflitePath: '$embedderDir/${embedderManifest.file}',
      manifest: embedderManifest,
      tfliteLibraryPath: libraryPath,
      numThreads: numThreads,
    );
    embedder = embedderInstance;
    await Future<void>.delayed(Duration.zero);
    if (sigintCompleter.isCompleted) return;

    final aligner = AffineAligner.arcface112(nativeFivePoint: true);
    final matcher = CosineMatcher.fromManifest(embedderManifest);

    final Enrollment enrollment;
    try {
      enrollment = await enrollFromImage(
        image: decodeImageFile(args['enroll-image'] as String),
        detector: detectorInstance,
        aligner: aligner,
        embedder: embedderInstance,
        id: args['enroll-id'] as String,
      );
    } on StateError catch (e) {
      stderr.writeln('등록 실패: ${e.message}');
      exitCode = 66;
      return;
    }
    if (sigintCompleter.isCompleted) return;
    stderr.writeln('등록 완료: id="${enrollment.id}"');

    try {
      source = V4l2FrameSource.open(
        device: args['device'] as String,
        width: int.parse(args['width'] as String),
        height: int.parse(args['height'] as String),
        libraryPath: args['v4l2-library'] as String?,
      );
    } on StateError catch (e) {
      stderr.writeln('카메라 열기 실패: ${e.message}');
      exitCode = 69;
      return;
    }
    final sourceInstance = source;
    stderr.writeln(
      '카메라 열림: ${sourceInstance.width}x${sourceInstance.height} YUYV422'
      '${duration != null ? ' — ${duration.inSeconds}초 후 종료' : ' — Ctrl+C로 종료'}',
    );

    final stats = await runRealtimeLoop(
      source: sourceInstance,
      detector: detectorInstance,
      aligner: aligner,
      embedder: embedderInstance,
      matcher: matcher,
      gallery: [enrollment],
      duration: duration,
      stopSignal: sigintCompleter.future,
      onResult: (result) => stdout.writeln(jsonEncode(result.toJson())),
      onProgress: stderr.writeln,
    );

    stderr.writeln(jsonEncode(stats.toJson()));
  } catch (e, st) {
    stderr.writeln('실시간 인식 실패: $e\n$st');
    if (exitCode == 0) exitCode = 1;
  } finally {
    try {
      await sigintSub?.cancel();
    } finally {
      try {
        await source?.close();
      } finally {
        try {
          detector?.dispose();
        } finally {
          embedder?.dispose();
        }
      }
    }
  }
}
