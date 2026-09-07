// S5a — ffmpeg PoC (doc/KR/linux_port_scope.md). Captures raw RGB24 frames
// via ffmpeg (V4L2 camera, or a synthetic --testsrc pattern for testing
// this file's plumbing without a camera) over dart:io Process, and runs
// each through YuNetDetector — sanity-checks frame format/rotation/
// mirroring before writing any V4L2 C/dart:ffi code (S5b). Zero C code;
// ffmpeg does the capture, this file just parses its raw stdout stream.
//
// ffmpeg itself is GPL — fine for this throwaway dev-time probe, but per
// CLAUDE.md/S5b notes it must not ship in the final distribution; S5b
// replaces this with a native V4L2 dart:ffi capture path.
//
// Usage (real camera, run on the target board):
//   dart run bin/camera_probe.dart \
//     --device /dev/video2 --width 640 --height 480 \
//     --detector ../../assets/models/yunet_160 \
//     --frames 30 --save-frame /tmp/probe_frame.png \
//     [--tflite-library path/to/libtensorflowlite_c.so]
//
// Usage (no camera — validates only the Dart-side frame parsing):
//   dart run bin/camera_probe.dart --testsrc --width 640 --height 480 \
//     --detector ../../assets/models/yunet_160 --frames 10 \
//     --save-frame /tmp/probe_frame.png

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:args/args.dart';
import 'package:image/image.dart' as img;

import 'package:facekit_core/facekit_core.dart';

Future<void> main(List<String> arguments) async {
  final parser = ArgParser()
    ..addOption('device', help: 'V4L2 장치 경로, 예: /dev/video2')
    ..addFlag('testsrc', help: '실카메라 대신 ffmpeg 합성 테스트 패턴 사용(플러밍 검증용)', defaultsTo: false)
    ..addOption('width', defaultsTo: '640')
    ..addOption('height', defaultsTo: '480')
    ..addOption('detector', help: 'manifest.json + .tflite가 있는 검출기 모델 디렉토리')
    ..addOption('tflite-library', help: 'libtensorflowlite_c 경로 (미지정 시 FACEKIT_TFLITE_C_LIBRARY 환경변수 또는 기본 경로)')
    ..addOption('frames', defaultsTo: '30', help: '처리할 프레임 수')
    ..addOption('save-frame', help: '첫 프레임을 PNG로 저장할 경로(회전/미러링 육안 확인용)')
    ..addFlag('help', abbr: 'h', negatable: false);

  final args = parser.parse(arguments);
  if (args['help'] as bool || (args['device'] == null && !(args['testsrc'] as bool)) || args['detector'] == null) {
    stderr.writeln('camera_probe — S5a ffmpeg PoC\n');
    stderr.writeln(parser.usage);
    exit(args['help'] as bool ? 0 : 64);
  }

  final width = int.parse(args['width'] as String);
  final height = int.parse(args['height'] as String);
  final frameCount = int.parse(args['frames'] as String);
  final frameBytes = width * height * 3; // rgb24

  final detectorDir = args['detector'] as String;
  final detectorManifest = ModelManifest.fromJsonString(
    File('$detectorDir/manifest.json').readAsStringSync(),
  )..validate();
  final detector = YuNetDetector.fromFile(
    tflitePath: '$detectorDir/${detectorManifest.file}',
    manifest: detectorManifest,
    tfliteLibraryPath: args['tflite-library'] as String?,
  );

  final useTestsrc = args['testsrc'] as bool;
  final ffmpegArgs = useTestsrc
      ? [
          '-f', 'lavfi',
          '-i', 'testsrc=size=${width}x$height:rate=5',
          '-pix_fmt', 'rgb24',
          '-f', 'rawvideo',
          '-',
        ]
      : [
          '-f', 'v4l2',
          '-input_format', 'yuyv422',
          '-video_size', '${width}x$height',
          '-i', args['device'] as String,
          '-pix_fmt', 'rgb24',
          '-f', 'rawvideo',
          '-',
        ];

  stderr.writeln('ffmpeg ${ffmpegArgs.join(' ')}');
  final process = await Process.start('ffmpeg', ffmpegArgs);
  process.stderr.transform(utf8.decoder).listen((_) {}); // discard ffmpeg's own logs

  var framesSeen = 0;
  var facesFound = 0;
  var savedFirstFrame = false;
  final buffer = BytesBuilder(copy: false);
  final sw = Stopwatch()..start();

  await for (final chunk in process.stdout) {
    buffer.add(chunk);
    var bytes = buffer.toBytes();

    while (bytes.length >= frameBytes && framesSeen < frameCount) {
      final frame = Uint8List.sublistView(bytes, 0, frameBytes);
      bytes = Uint8List.sublistView(bytes, frameBytes);
      framesSeen++;

      final image = FaceImage(rgbBytes: Uint8List.fromList(frame), width: width, height: height);

      if (!savedFirstFrame && args['save-frame'] != null) {
        savedFirstFrame = true;
        final png = img.Image.fromBytes(width: width, height: height, bytes: frame.buffer, numChannels: 3);
        File(args['save-frame'] as String).writeAsBytesSync(img.encodePng(png));
        stderr.writeln('첫 프레임을 ${args['save-frame']}에 저장함 — 회전/미러링 육안 확인할 것');
      }

      final detected = await detector.detect(image);
      if (detected.isNotEmpty) {
        facesFound++;
        final best = detected.reduce((a, b) => a.score >= b.score ? a : b);
        stderr.writeln(
          '프레임 $framesSeen: 얼굴 검출됨 score=${best.score.toStringAsFixed(3)} '
          'bbox=(${best.boundingBox.left.toStringAsFixed(0)},${best.boundingBox.top.toStringAsFixed(0)})'
          '-(${best.boundingBox.right.toStringAsFixed(0)},${best.boundingBox.bottom.toStringAsFixed(0)})',
        );
      } else {
        stderr.writeln('프레임 $framesSeen: 얼굴 없음');
      }
    }

    buffer.clear();
    if (bytes.isNotEmpty) buffer.add(bytes);
    if (framesSeen >= frameCount) break;
  }

  process.kill();
  final elapsedMs = sw.elapsedMilliseconds;

  stdout.writeln(jsonEncode({
    'frames': framesSeen,
    'faces_found': facesFound,
    'elapsed_ms': elapsedMs,
    'fps': framesSeen / (elapsedMs / 1000.0),
  }));
}
