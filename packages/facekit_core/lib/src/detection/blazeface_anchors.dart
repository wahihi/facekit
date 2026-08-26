// BlazeFace SSD anchor generation — short-range model (128×128 input).
//
// Source: MediaPipe SsdAnchorsCalculator (Apache 2.0)
// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tflite/ssd_anchors_calculator.cc
//
// Config used by face_detection_short_range.tflite:
//   num_layers: 4,  strides: [8, 16, 16, 16]
//   min_scale: 0.1484375, max_scale: 0.75
//   input_size: 128×128,  anchor_offset: 0.5
//   aspect_ratios: [1.0],  fixed_anchor_size: true
//   interpolated_scale_aspect_ratio: 1.0  (adds one extra anchor per cell)
//
// Result: 896 anchors — (16×16 + 8×8 + 8×8 + 8×8) × 2 = 512 + 384 = 896.

import 'dart:typed_data';

/// A single SSD anchor represented as (x_center, y_center, width, height)
/// in normalised coordinates [0, 1].
class Anchor {
  final double xCenter;
  final double yCenter;
  final double width;
  final double height;

  const Anchor({
    required this.xCenter,
    required this.yCenter,
    required this.width,
    required this.height,
  });
}

/// Generates the 896 anchors for the BlazeFace short-range model.
/// Called once at detector initialisation; result is cached.
List<Anchor> generateBlazeFaceShortAnchors() {
  const int inputH = 128, inputW = 128;
  const List<int> strides = [8, 16, 16, 16];
  const double minScale = 0.1484375, maxScale = 0.75;
  const double anchorOffsetX = 0.5, anchorOffsetY = 0.5;
  // Each cell gets 2 anchors: one for aspect_ratio=1.0, one for the
  // interpolated scale (also aspect_ratio=1.0 at the interpolated scale).
  const int anchorsPerCell = 2;

  final anchors = <Anchor>[];
  final numLayers = strides.length;

  for (int layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    final stride = strides[layerIdx];
    final featureMapH = (inputH / stride).ceil();
    final featureMapW = (inputW / stride).ceil();

    // Interpolated scale between min and max for this layer.
    // MediaPipe uses: scale = min + (max - min) * layer / (num_layers - 1)
    final scale = _interpolateScale(minScale, maxScale, layerIdx, numLayers);

    for (int row = 0; row < featureMapH; row++) {
      for (int col = 0; col < featureMapW; col++) {
        final xCenter = (col + anchorOffsetX) / featureMapW;
        final yCenter = (row + anchorOffsetY) / featureMapH;

        // With fixed_anchor_size = true, width and height are always 1.0.
        // The two anchors per cell are identical in this config.
        for (int ai = 0; ai < anchorsPerCell; ai++) {
          anchors.add(Anchor(
            xCenter: xCenter,
            yCenter: yCenter,
            width:   1.0,
            height:  1.0,
          ));
        }
      }
    }
  }

  assert(anchors.length == 896,
      'Expected 896 anchors, got ${anchors.length}');
  return anchors;
}

double _interpolateScale(
    double minScale, double maxScale, int layerIdx, int numLayers) {
  if (numLayers == 1) return (minScale + maxScale) / 2;
  return minScale + (maxScale - minScale) * layerIdx / (numLayers - 1);
}

/// Lazily computed and cached anchors for the short-range model.
final List<Anchor> blazeFaceShortAnchors = generateBlazeFaceShortAnchors();
