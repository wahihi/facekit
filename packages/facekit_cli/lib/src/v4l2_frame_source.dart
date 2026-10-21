// Native V4L2 mmap capture (S5b), implementing FrameSource. Replaces the
// S5a ffmpeg PoC for the actual shipped path — no GPL dependency, and no
// subprocess/pipe overhead.
//
// Concurrency note: v4l2_read_frame() blocks the calling isolate (via
// poll() in C) for up to its timeout — fine for this CLI's sequential
// capture->detect->align->embed loop (S6), but if a future caller wants
// capture running concurrently with other work on the same isolate,
// this needs to move onto its own Isolate first.

import 'dart:async';
import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'frame_source.dart';
import 'v4l2_bindings.dart';

class V4l2FrameSource implements FrameSource {
  final V4l2Bindings _bindings;
  final Pointer<Void> _capture;
  final int _frameSize;
  final int width;
  final int height;

  Pointer<Uint8>? _scratch;
  Pointer<Uint32>? _sequenceOut;
  bool _closed = false;

  V4l2FrameSource._(
    this._bindings,
    this._capture,
    this.width,
    this.height,
    this._frameSize,
  );

  /// Opens [device] (e.g. `/dev/video2`) and negotiates [width]x[height] in
  /// YUYV422 — the only format this camera path supports (see
  /// v4l2PixFmtYuyv's doc comment). The driver may adjust the negotiated
  /// size; read it back from [width]/[height] on the returned instance
  /// rather than assuming the requested values stuck.
  static V4l2FrameSource open({
    required String device,
    required int width,
    required int height,
    String? libraryPath,
  }) {
    final bindings = V4l2Bindings(openV4l2Library(explicitPath: libraryPath));
    final devicePtr = device.toNativeUtf8();
    final Pointer<Void> capture;
    try {
      capture = bindings.open(devicePtr.cast(), width, height, v4l2PixFmtYuyv);
    } finally {
      calloc.free(devicePtr);
    }
    if (capture == nullptr) {
      throw StateError(
        'V4L2 장치를 열 수 없음: $device ($width x $height, YUYV) — 장치 경로/포맷/'
        '해상도를 v4l2-ctl --list-formats-ext로 확인할 것',
      );
    }
    return V4l2FrameSource._(
      bindings,
      capture,
      bindings.getWidth(capture),
      bindings.getHeight(capture),
      bindings.getFrameSize(capture),
    );
  }

  @override
  Stream<RawFrame> frames() async* {
    final scratch = _scratch ??= calloc<Uint8>(_frameSize);
    final sequenceOut = _sequenceOut ??= calloc<Uint32>();
    while (!_closed) {
      final n = _bindings.readFrame(_capture, scratch, 100, sequenceOut);
      if (_closed) break;
      if (n < 0) {
        throw StateError('V4L2 프레임 읽기 실패 (code $n)');
      }
      if (n == 0) {
        // Yield so duration timers and stop signals run even when no camera
        // frames arrive.
        await Future<void>.delayed(Duration.zero);
        continue;
      }

      yield RawFrame(
        // Copy out of the scratch buffer — it's reused (and may be
        // overwritten) on the next readFrame call.
        bytes: Uint8List.fromList(scratch.asTypedList(n)),
        width: width,
        height: height,
        format: PixelFormat.yuyv422,
        sequence: sequenceOut.value,
      );
    }
  }

  @override
  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    _bindings.close(_capture);
    if (_scratch != null) {
      calloc.free(_scratch!);
      _scratch = null;
    }
    if (_sequenceOut != null) {
      calloc.free(_sequenceOut!);
      _sequenceOut = null;
    }
  }
}
