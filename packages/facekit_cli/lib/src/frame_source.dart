// Frame-source abstraction (S5b, doc/KR/linux_port_scope.md) — a single
// interface any camera/capture backend can implement, so the recognition
// loop (S6) doesn't care whether frames come from V4L2, ffmpeg (the S5a
// PoC), or a file replay for tests.

import 'dart:typed_data';

enum PixelFormat { yuv420_888, yuyv422, mjpeg, bgra8888 }

class RawFrame {
  final Uint8List bytes;
  final int width;
  final int height;
  final PixelFormat format;

  /// Driver-assigned per-frame counter, when the source can provide one
  /// (V4l2FrameSource can; a gap between consecutive values means a
  /// dropped frame). Null for sources that don't track this (e.g. the
  /// ffmpeg-based S5a source).
  final int? sequence;

  RawFrame({
    required this.bytes,
    required this.width,
    required this.height,
    required this.format,
    this.sequence,
  });
}

abstract class FrameSource {
  /// Frame stream. Implementations should keep producing frames until the
  /// stream is cancelled or [close] is called — callers that only need a
  /// few frames should cancel their subscription rather than expecting
  /// this to complete on its own.
  Stream<RawFrame> frames();

  /// Releases the underlying capture resource (device handle, mmap'd
  /// buffers, subprocess, ...). Safe to call more than once.
  Future<void> close();
}
