// Decodes raw BlazeFace TFLite output tensors into DetectedFace objects.
//
// Source:
//   MediaPipe TensorToDetectionsCalculator (Apache 2.0)
//   https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
//
// Tensor layout (short-range model):
//   regressors    [1, 896, 16]  — 4 box coords + 12 landmark coords (6 × 2)
//   classificators[1, 896,  1]  — raw logit scores (apply sigmoid for probability)
//
// Box encoding (YXHW, scale = 128):
//   decoded_cx = anchor_cx + raw[1] / scale * anchor_w
//   decoded_cy = anchor_cy + raw[0] / scale * anchor_h
//   decoded_w  = exp(raw[3] / scale) * anchor_w
//   decoded_h  = exp(raw[2] / scale) * anchor_h
//
// Landmark encoding (6 keypoints, pairs of [y, x]):
//   lm_x = anchor_cx + raw[4 + 2*i + 1] / scale * anchor_w
//   lm_y = anchor_cy + raw[4 + 2*i + 0] / scale * anchor_h
//
// Keypoint order: rightEye, leftEye, nose, mouth, rightEar, leftEar
// (from the model's perspective; right/left are mirrored to viewer's left/right)

import 'dart:math' as math;
import 'dart:typed_data';

import '../core/models.dart';
import 'blazeface_anchors.dart';

const double _kScale = 128.0; // matches input_size_width / height

/// Decodes raw tensors into a list of [DetectedFace].
///
/// [regressors]     — flat list shaped [896, 16].
/// [classificators] — flat list shaped [896, 1].
/// [anchors]        — the 896 pre-generated anchors.
/// [scoreThreshold] — sigmoid score threshold (typically 0.5).
/// [iouThreshold]   — IoU threshold for NMS (typically 0.3).
/// [maxFaces]       — maximum number of returned detections.
List<DetectedFace> decodeBlazeFace({
  required List<List<double>> regressors,
  required List<List<double>> classificators,
  required List<Anchor> anchors,
  required double scoreThreshold,
  required double iouThreshold,
  required int maxFaces,
}) {
  assert(regressors.length == 896 && classificators.length == 896);

  final candidates = <DetectedFace>[];

  for (int i = 0; i < 896; i++) {
    final score = _sigmoid(classificators[i][0]);
    if (score < scoreThreshold) continue;

    final anchor = anchors[i];
    final raw = regressors[i];

    // Decode bounding box (YXHW encoding)
    final cx = anchor.xCenter + raw[1] / _kScale * anchor.width;
    final cy = anchor.yCenter + raw[0] / _kScale * anchor.height;
    final w  = math.exp(raw[3] / _kScale) * anchor.width;
    final h  = math.exp(raw[2] / _kScale) * anchor.height;

    final box = Rect(
      left:   (cx - w / 2).clamp(0.0, 1.0),
      top:    (cy - h / 2).clamp(0.0, 1.0),
      right:  (cx + w / 2).clamp(0.0, 1.0),
      bottom: (cy + h / 2).clamp(0.0, 1.0),
    );

    // Decode 6 landmarks
    final landmarks = <Point>[];
    for (int k = 0; k < 6; k++) {
      final lmX = anchor.xCenter + raw[4 + 2 * k + 1] / _kScale * anchor.width;
      final lmY = anchor.yCenter + raw[4 + 2 * k + 0] / _kScale * anchor.height;
      landmarks.add(Point(lmX.clamp(0.0, 1.0), lmY.clamp(0.0, 1.0)));
    }

    candidates.add(DetectedFace(
      boundingBox: box,
      landmarks:   landmarks,
      score:       score,
    ));
  }

  return _nms(candidates, iouThreshold: iouThreshold, maxFaces: maxFaces);
}

// ── Non-Maximum Suppression ───────────────────────────────────────────────────

List<DetectedFace> _nms(
  List<DetectedFace> detections, {
  required double iouThreshold,
  required int maxFaces,
}) {
  // Sort descending by score
  detections.sort((a, b) => b.score.compareTo(a.score));

  final kept = <DetectedFace>[];

  for (final det in detections) {
    if (kept.length >= maxFaces) break;

    bool suppressed = false;
    for (final k in kept) {
      if (_iou(det.boundingBox, k.boundingBox) >= iouThreshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) kept.add(det);
  }

  return kept;
}

double _iou(Rect a, Rect b) {
  final interLeft   = math.max(a.left,   b.left);
  final interTop    = math.max(a.top,    b.top);
  final interRight  = math.min(a.right,  b.right);
  final interBottom = math.min(a.bottom, b.bottom);

  final interW = (interRight  - interLeft).clamp(0.0, 1.0);
  final interH = (interBottom - interTop ).clamp(0.0, 1.0);
  final interArea = interW * interH;

  if (interArea == 0.0) return 0.0;

  final aArea = a.width * a.height;
  final bArea = b.width * b.height;
  return interArea / (aArea + bArea - interArea);
}

/// Converts detections with normalised [0,1] coordinates (as produced by
/// [decodeBlazeFace]) into pixel coordinates for an image of size
/// [width] × [height].
///
/// Normalised coordinates are scale-invariant (the model always runs on a
/// resized 128×128 copy), so this maps them back onto whatever image size
/// the caller's original [FaceImage] actually has — which is what every
/// downstream consumer (e.g. [AffineAligner]) expects per the pixel-space
/// contract documented on [Rect].
List<DetectedFace> denormalizeDetections(
  List<DetectedFace> faces,
  int width,
  int height,
) {
  return [
    for (final f in faces)
      DetectedFace(
        boundingBox: Rect(
          left:   f.boundingBox.left   * width,
          top:    f.boundingBox.top    * height,
          right:  f.boundingBox.right  * width,
          bottom: f.boundingBox.bottom * height,
        ),
        landmarks: [
          for (final p in f.landmarks) Point(p.x * width, p.y * height),
        ],
        score: f.score,
      ),
  ];
}

double _sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));

// ── Tensor reshaping helpers ──────────────────────────────────────────────────

/// Reshapes the flat TFLite output List into [numAnchors][valuesPerAnchor].
/// tflite_flutter returns outputs as nested lists; this normalises them.
List<List<double>> reshapeTensor2D(dynamic raw, int dim0, int dim1) {
  // tflite_flutter may return List<List<List<double>>> shaped [1, dim0, dim1]
  // or List<List<double>> shaped [dim0, dim1]. Handle both.
  final inner = (raw is List && raw.isNotEmpty && raw[0] is List && raw[0][0] is List)
      ? raw[0] as List  // unwrap batch dimension
      : raw as List;

  return List<List<double>>.generate(
    dim0,
    (i) => List<double>.from((inner[i] as List).map((v) => (v as num).toDouble())),
  );
}
