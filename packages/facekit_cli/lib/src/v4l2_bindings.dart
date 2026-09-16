// dart:ffi bindings for native/libv4l2_capture.so (S5b, doc/KR/
// linux_port_scope.md). The C side does the actual V4L2 ioctl/mmap work;
// this file just declares the small function surface it exports.

import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart' show Utf8;

/// V4L2_PIX_FMT_YUYV fourcc ('Y','U','Y','V') — see `linux/videodev2.h`.
/// The only format this binding requests; the Logitech C270 used for Q6A
/// development only offers YUYV/MJPG and MJPG tops out at the same
/// resolution YUYV does on this camera (see doc/KR/linux_port_scope.md),
/// so there's no reason to add MJPG/libjpeg-turbo decoding here.
const int v4l2PixFmtYuyv = 0x56595559;

typedef V4l2OpenNative = Pointer<Void> Function(
    Pointer<Utf8> device, Uint32 width, Uint32 height, Uint32 pixelformat);
typedef V4l2OpenDart = Pointer<Void> Function(
    Pointer<Utf8> device, int width, int height, int pixelformat);

typedef V4l2GetU32Native = Uint32 Function(Pointer<Void> capture);
typedef V4l2GetU32Dart = int Function(Pointer<Void> capture);

typedef V4l2ReadFrameNative = Int32 Function(Pointer<Void> capture,
    Pointer<Uint8> outBuffer, Int32 timeoutMs, Pointer<Uint32> outSequence);
typedef V4l2ReadFrameDart = int Function(Pointer<Void> capture,
    Pointer<Uint8> outBuffer, int timeoutMs, Pointer<Uint32> outSequence);

typedef V4l2CloseNative = Void Function(Pointer<Void> capture);
typedef V4l2CloseDart = void Function(Pointer<Void> capture);

class V4l2Bindings {
  final DynamicLibrary _lib;

  late final V4l2OpenDart open =
      _lib.lookupFunction<V4l2OpenNative, V4l2OpenDart>('v4l2_open');
  late final V4l2GetU32Dart getWidth =
      _lib.lookupFunction<V4l2GetU32Native, V4l2GetU32Dart>('v4l2_get_width');
  late final V4l2GetU32Dart getHeight = _lib
      .lookupFunction<V4l2GetU32Native, V4l2GetU32Dart>('v4l2_get_height');
  late final V4l2GetU32Dart getFrameSize = _lib.lookupFunction<
      V4l2GetU32Native, V4l2GetU32Dart>('v4l2_get_frame_size');
  late final V4l2ReadFrameDart readFrame = _lib.lookupFunction<
      V4l2ReadFrameNative, V4l2ReadFrameDart>('v4l2_read_frame');
  late final V4l2CloseDart close =
      _lib.lookupFunction<V4l2CloseNative, V4l2CloseDart>('v4l2_close');

  V4l2Bindings(this._lib);
}

/// Locates and opens libv4l2_capture.so, in priority order: [explicitPath],
/// the FACEKIT_V4L2_LIBRARY environment variable, or `native/
/// libv4l2_capture.so` relative to the current working directory (built
/// via native/build.sh — matches how this CLI is invoked throughout
/// doc/KR/linux_port_scope.md, always from the facekit_cli package root).
DynamicLibrary openV4l2Library({String? explicitPath}) {
  final path = explicitPath ??
      Platform.environment['FACEKIT_V4L2_LIBRARY'] ??
      'native/libv4l2_capture.so';
  try {
    return DynamicLibrary.open(path);
  } on ArgumentError catch (e) {
    throw StateError(
      'Could not load libv4l2_capture from "$path" ($e). Build it with '
      'native/build.sh, or pass an explicit path / set '
      'FACEKIT_V4L2_LIBRARY.',
    );
  }
}
