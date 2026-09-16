// S5b DoD check (doc/KR/linux_port_scope.md): captures from the native
// V4L2 path for a fixed duration and reports frame drops (via the
// driver's own per-frame sequence counter) and RSS memory growth, so the
// "30초 연속 캡처 중 프레임 드롭·메모리 증가 없음" requirement is a real
// measurement, not a eyeball guess. Also runs YuNet on each frame (same
// as S5a's camera_probe.dart) and saves the first frame as a PNG for a
// visual rotation/mirroring check.
//
// Usage:
//   dart run bin/v4l2_probe.dart \
//     --device /dev/video2 --width 640 --height 480 \
//     --detector ../../assets/models/yunet_160 \
//     --seconds 30 --save-frame /tmp/v4l2_probe_frame.png \
//     [--tflite-library path] [--v4l2-library path]

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:args/args.dart';
import 'package:image/image.dart' as img;

import 'package:facekit_cli/facekit_cli.dart';
import 'package:facekit_core/facekit_core.dart';

Future<void> main(List<String> arguments) async {
  final parser = ArgParser()
    ..addOption('device', help: 'V4L2 장치 경로, 예: /dev/video2')
    ..addOption('width', defaultsTo: '640')
    ..addOption('height', defaultsTo: '480')
    ..addOption('detector', help: 'manifest.json + .tflite가 있는 검출기 모델 디렉토리')
    ..addOption('tflite-library', help: 'libtensorflowlite_c 경로')
    ..addOption('v4l2-library', help: 'libv4l2_capture.so 경로 (미지정 시 native/libv4l2_capture.so)')
    ..addOption('seconds', defaultsTo: '30', help: '캡처 지속 시간(초)')
    ..addOption('save-frame', help: '첫 프레임을 PNG로 저장할 경로(회전/미러링 육안 확인용)')
    ..addFlag('help', abbr: 'h', negatable: false);

  final args = parser.parse(arguments);
  if (args['help'] as bool || args['device'] == null || args['detector'] == null) {
    stderr.writeln('v4l2_probe — S5b DoD 체크 (프레임 드롭 / 메모리 증가)\n');
    stderr.writeln(parser.usage);
    exit(args['help'] as bool ? 0 : 64);
  }

  final width = int.parse(args['width'] as String);
  final height = int.parse(args['height'] as String);
  final seconds = int.parse(args['seconds'] as String);

  final detectorDir = args['detector'] as String;
  final detectorManifest = ModelManifest.fromJsonString(
    File('$detectorDir/manifest.json').readAsStringSync(),
  )..validate();
  final detector = YuNetDetector.fromFile(
    tflitePath: '$detectorDir/${detectorManifest.file}',
    manifest: detectorManifest,
    tfliteLibraryPath: args['tflite-library'] as String?,
  );

  final source = V4l2FrameSource.open(
    device: args['device'] as String,
    width: width,
    height: height,
    libraryPath: args['v4l2-library'] as String?,
  );
  stderr.writeln('열림: ${source.width}x${source.height} YUYV422');

  var framesSeen = 0;
  var facesFound = 0;
  var drops = 0;
  var corruptFrames = 0;
  int? lastSequence;
  var savedFirstFrame = false;
  final rssStart = ProcessInfo.currentRss;
  var rssMax = rssStart;

  final sw = Stopwatch()..start();
  final sub = source.frames().listen(null);
  final done = Completer<void>();

  sub.onData((frame) async {
    sub.pause();
    try {
      framesSeen++;

      if (frame.sequence != null) {
        if (lastSequence != null && frame.sequence! != lastSequence! + 1) {
          drops += frame.sequence! - lastSequence! - 1;
        }
        lastSequence = frame.sequence;
      }

      final rss = ProcessInfo.currentRss;
      if (rss > rssMax) rssMax = rss;

      final expectedBytes = frame.width * frame.height * 2;
      if (frame.bytes.length != expectedBytes) {
        // Occasionally seen on the C270 over V4L2: a frame a few bytes
        // short of a full YUYV422 frame (a dropped USB packet) — a real
        // capture-time event, not a bug to crash on. Count it separately
        // from V4L2 sequence-counter drops and move on to the next frame.
        corruptFrames++;
        stderr.writeln(
          '프레임 $framesSeen (seq=${frame.sequence}): 손상된 프레임, 건너뜀 '
          '(기대 $expectedBytes bytes, 실제 ${frame.bytes.length} bytes)',
        );
        if (sw.elapsed.inSeconds >= seconds) {
          if (!done.isCompleted) done.complete();
        } else {
          sub.resume();
        }
        return;
      }

      final image = yuyv422ToFaceImage(yuyv: frame.bytes, width: frame.width, height: frame.height);

      if (!savedFirstFrame && args['save-frame'] != null) {
        savedFirstFrame = true;
        final png = img.Image.fromBytes(width: image.width, height: image.height, bytes: image.rgbBytes.buffer, numChannels: 3);
        File(args['save-frame'] as String).writeAsBytesSync(img.encodePng(png));
        stderr.writeln('첫 프레임을 ${args['save-frame']}에 저장함');
      }

      final detected = await detector.detect(image);
      if (detected.isNotEmpty) facesFound++;

      if (framesSeen % 10 == 0) {
        stderr.writeln(
          '프레임 $framesSeen (seq=${frame.sequence}) rss=${(rss / 1024 / 1024).toStringAsFixed(1)}MB '
          '누적 얼굴검출=$facesFound 누적드롭=$drops 누적손상=$corruptFrames',
        );
      }

      if (sw.elapsed.inSeconds >= seconds) {
        if (!done.isCompleted) done.complete();
      } else {
        sub.resume();
      }
    } catch (e, st) {
      stderr.writeln('처리 중 오류: $e\n$st');
      if (!done.isCompleted) done.complete();
    }
  });

  await done.future;
  await sub.cancel();
  await source.close();
  detector.dispose();

  final elapsedS = sw.elapsedMilliseconds / 1000.0;
  stdout.writeln(jsonEncode({
    'seconds': elapsedS,
    'frames': framesSeen,
    'fps': framesSeen / elapsedS,
    'faces_found': facesFound,
    'dropped_frames': drops,
    'corrupt_frames': corruptFrames,
    'rss_start_mb': rssStart / 1024 / 1024,
    'rss_max_mb': rssMax / 1024 / 1024,
    'rss_growth_mb': (rssMax - rssStart) / 1024 / 1024,
  }));
}
