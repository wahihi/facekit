// Pure-Dart tensor prep/allocation helpers shared by both InferenceRunner
// implementations' callers (detectors, embedders, landmarker) — no engine
// dependency, just plain nested Lists in the shape TfliteRunner/
// TfliteFfiRunner both expect.

import 'dart:typed_data';

/// Builds a nested zero-filled `List` matching [shape], for use as a
/// pre-allocated buffer with [InferenceRunner.runForMultipleOutputs] when a
/// tensor's contents aren't needed (e.g. AdaFace's secondary "norm" output).
Object zeroTensor(List<int> shape) {
  if (shape.length == 1) return List.filled(shape[0], 0.0);
  return List.generate(shape[0], (_) => zeroTensor(shape.sublist(1)));
}

// ── input preparation utilities ───────────────────────────────────────────────

/// Converts an RGB888 byte array into a float32 NHWC tensor normalised by
/// (pixel - mean) / std, matching the manifest InputSpec.
///
/// [swapToBgr] reorders channels to B,G,R before normalising — set this when
/// `manifest.input.color == 'BGR'` (e.g. AdaFace, which assumes cv2-style BGR
/// input, unlike InsightFace/ArcFace's RGB convention). [mean]/[std] are
/// still indexed 0/1/2 in the *output* channel order, matching how the
/// manifest's normalize arrays are written.
///
/// Returns a [List] shaped [1, height, width, 3] for use as tflite input.
List<List<List<List<double>>>> prepareInputTensor({
  required Uint8List rgbBytes,
  required int width,
  required int height,
  required List<double> mean,
  required List<double> std,
  bool swapToBgr = false,
}) {
  return List.generate(
    1,
    (_) => List.generate(
      height,
      (row) => List.generate(
        width,
        (col) {
          final idx = (row * width + col) * 3;
          final r = rgbBytes[idx], g = rgbBytes[idx + 1], b = rgbBytes[idx + 2];
          final c0 = swapToBgr ? b : r;
          final c2 = swapToBgr ? r : b;
          return [
            (c0 - mean[0]) / std[0],
            (g  - mean[1]) / std[1],
            (c2 - mean[2]) / std[2],
          ];
        },
      ),
    ),
  );
}
