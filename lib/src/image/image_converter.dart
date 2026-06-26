// Converts raw camera byte planes to the RGB888 FaceImage the pipeline expects.
//
// Supported formats:
//   YUV420 (yuv420_888) — standard Android camera2 output
//   BGRA8888             — standard iOS / macOS AVFoundation output
//
// Source: camera plugin documentation + ITU-R BT.601 YCbCr→RGB coefficients.
// Pure Dart — no Flutter / dart:ui import.

import 'dart:typed_data';
import '../core/models.dart';

/// Converts a YUV420 (yuv420_888) camera frame to [FaceImage].
///
/// [yPlane], [uPlane], [vPlane] are the raw byte buffers from the camera plugin.
/// [yRowStride] / [uvRowStride] / [uvPixelStride] match the camera plugin plane
/// metadata fields of the same name.
FaceImage yuv420ToFaceImage({
  required Uint8List yPlane,
  required Uint8List uPlane,
  required Uint8List vPlane,
  required int width,
  required int height,
  required int yRowStride,
  required int uvRowStride,
  required int uvPixelStride,
}) {
  final rgb = Uint8List(width * height * 3);
  int outIdx = 0;

  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      final yIdx = row * yRowStride + col;
      final uvRow = row >> 1;
      final uvCol = col >> 1;
      final uvIdx = uvRow * uvRowStride + uvCol * uvPixelStride;

      final y = yPlane[yIdx] & 0xFF;
      final u = uPlane[uvIdx] & 0xFF;
      final v = vPlane[uvIdx] & 0xFF;

      // ITU-R BT.601 full-range YCbCr → RGB
      final yShifted = y - 16;
      final uShifted = u - 128;
      final vShifted = v - 128;

      final r = (1.164 * yShifted + 1.596 * vShifted).round().clamp(0, 255);
      final g = (1.164 * yShifted - 0.392 * uShifted - 0.813 * vShifted).round().clamp(0, 255);
      final b = (1.164 * yShifted + 2.017 * uShifted).round().clamp(0, 255);

      rgb[outIdx++] = r;
      rgb[outIdx++] = g;
      rgb[outIdx++] = b;
    }
  }

  return FaceImage(rgbBytes: rgb, width: width, height: height);
}

/// Converts a BGRA8888 camera frame to [FaceImage].
///
/// [bgra] is the flat byte buffer (4 bytes per pixel: B G R A).
FaceImage bgraToFaceImage({
  required Uint8List bgra,
  required int width,
  required int height,
}) {
  final rgb = Uint8List(width * height * 3);
  final pixelCount = width * height;

  for (int i = 0; i < pixelCount; i++) {
    final src = i * 4;
    final dst = i * 3;
    rgb[dst]     = bgra[src + 2]; // R
    rgb[dst + 1] = bgra[src + 1]; // G
    rgb[dst + 2] = bgra[src];     // B
    // alpha (bgra[src + 3]) discarded
  }

  return FaceImage(rgbBytes: rgb, width: width, height: height);
}

/// Resizes an RGB888 [FaceImage] to [targetWidth] × [targetHeight]
/// using nearest-neighbour sampling.
///
/// Used to scale a full-frame image down before detection (speed vs accuracy).
FaceImage resizeNearest(FaceImage src, int targetWidth, int targetHeight) {
  final dst = Uint8List(targetWidth * targetHeight * 3);
  final xScale = src.width / targetWidth;
  final yScale = src.height / targetHeight;

  for (int row = 0; row < targetHeight; row++) {
    final srcRow = (row * yScale).floor().clamp(0, src.height - 1);
    for (int col = 0; col < targetWidth; col++) {
      final srcCol = (col * xScale).floor().clamp(0, src.width - 1);
      final srcIdx = (srcRow * src.width + srcCol) * 3;
      final dstIdx = (row * targetWidth + col) * 3;
      dst[dstIdx]     = src.rgbBytes[srcIdx];
      dst[dstIdx + 1] = src.rgbBytes[srcIdx + 1];
      dst[dstIdx + 2] = src.rgbBytes[srcIdx + 2];
    }
  }

  return FaceImage(rgbBytes: dst, width: targetWidth, height: targetHeight);
}

/// Crops [region] out of [src], clamped to the image bounds.
///
/// Used to extract a square face ROI (with margin) ahead of a landmark model
/// that — unlike the detector — expects an already-cropped, roughly-centred
/// face rather than a full frame.
FaceImage cropFaceImage(FaceImage src, Rect region) {
  final left = region.left.floor().clamp(0, src.width - 1);
  final top = region.top.floor().clamp(0, src.height - 1);
  final right = region.right.ceil().clamp(left + 1, src.width);
  final bottom = region.bottom.ceil().clamp(top + 1, src.height);
  final width = right - left;
  final height = bottom - top;

  final dst = Uint8List(width * height * 3);
  for (int row = 0; row < height; row++) {
    final srcRowStart = ((top + row) * src.width + left) * 3;
    final dstRowStart = row * width * 3;
    dst.setRange(dstRowStart, dstRowStart + width * 3, src.rgbBytes, srcRowStart);
  }

  return FaceImage(rgbBytes: dst, width: width, height: height);
}
