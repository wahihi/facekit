// Decodes raw YuNet TFLite output tensors into DetectedFace objects.
//
// Source:
//   YuNet (libfacedetection / OpenCV Zoo, MIT licence)
//   https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
//   Decode formula (prior centre, bbox, keypoints, score) ported from
//   OpenCV's own postprocessing:
//   https://github.com/opencv/opencv/blob/4.x/modules/objdetect/src/face_detect.cpp
//
// Verification: this exact formula was cross-checked in Python against
// cv2.FaceDetectorYN's own reference output on a real photo (same 160×160
// resized input fed to both) and matched byte-for-byte on score, bbox, and
// all 5 keypoints — see tool/model_verification/compare_with_yunet.py and
// doc/KR/postmortem/2026-08-12-yunet-landmark-order.md.
//
// Tensor layout (per stride s in {8, 16, 32}; grid = inputSize / s):
//   cls_s  [grid*grid, 1]  — sigmoid'd class score, already in [0,1]
//   obj_s  [grid*grid, 1]  — sigmoid'd objectness score, already in [0,1]
//   bbox_s [grid*grid, 4]  — raw regression [dx, dy, dw, dh]
//   kps_s  [grid*grid, 10] — 5 keypoints, raw offsets, pairs of [x, y]
// Row-major grid: flat index = row * grid + col (row 0..grid-1 outer,
// col 0..grid-1 inner) — matches the network's own feature-map layout.
//
// Decode (all coordinates come out in "model input" pixel space
// [0, inputSize], not [0,1] — unlike BlazeFace's decoder):
//   score = sqrt(clamp(cls,0,1) * clamp(obj,0,1))
//   cx = (col + bbox[0]) * stride;  cy = (row + bbox[1]) * stride
//   w  = exp(bbox[2]) * stride;     h  = exp(bbox[3]) * stride
//   kp_x[k] = (col + kps[2k])   * stride
//   kp_y[k] = (row + kps[2k+1]) * stride
//
// Keypoint order is YuNet's own raw output order, unmodified: index 0/1 are
// the two eyes, index 2 is the nose tip, index 3/4 are the two mouth
// corners. IMPORTANT — do not relabel or reorder these to try to match
// ArcFace's named left/right convention: YuNet's own author-given
// right_eye/left_eye labels turned out to be the *opposite* handedness
// from facekit's arcface112Ref point order. Feeding this raw positional
// order directly into AffineAligner's native 5-point mode (no reordering)
// is the verified-correct behaviour; "correcting" it collapses the fitted
// scale to ~1/7 of the expected value. See
// doc/KR/postmortem/2026-08-12-yunet-landmark-order.md for the full
// investigation (same class of bug as the BlazeFace eye-order fix in
// affine_aligner.dart).

import 'dart:math' as math;

import '../core/models.dart';

/// Decodes one stride's worth of raw tensors into candidate detections, in
/// "model input" pixel space [0, inputSize].
///
/// [cls]/[obj]/[bbox]/[kps] are shaped [grid*grid, 1]/[grid*grid, 1]/
/// [grid*grid, 4]/[grid*grid, 10] where grid = inputSize ~/ stride.
List<DetectedFace> _decodeStride({
  required List<List<double>> cls,
  required List<List<double>> obj,
  required List<List<double>> bbox,
  required List<List<double>> kps,
  required int stride,
  required int inputSize,
  required double scoreThreshold,
}) {
  final grid = inputSize ~/ stride;
  assert(cls.length == grid * grid, 'expected ${grid * grid} cells for stride $stride, got ${cls.length}');

  final maxCoord = inputSize.toDouble();
  final out = <DetectedFace>[];
  int idx = 0;
  for (int row = 0; row < grid; row++) {
    for (int col = 0; col < grid; col++) {
      final clsScore = cls[idx][0].clamp(0.0, 1.0);
      final objScore = obj[idx][0].clamp(0.0, 1.0);
      final score = math.sqrt(clsScore * objScore);

      if (score >= scoreThreshold) {
        final b = bbox[idx];
        final cx = (col + b[0]) * stride;
        final cy = (row + b[1]) * stride;
        final w = math.exp(b[2]) * stride;
        final h = math.exp(b[3]) * stride;

        final box = Rect(
          left:   (cx - w / 2).clamp(0.0, maxCoord),
          top:    (cy - h / 2).clamp(0.0, maxCoord),
          right:  (cx + w / 2).clamp(0.0, maxCoord),
          bottom: (cy + h / 2).clamp(0.0, maxCoord),
        );

        final k = kps[idx];
        final landmarks = <Point>[
          for (int p = 0; p < 5; p++)
            Point(
              ((col + k[2 * p])     * stride).clamp(0.0, maxCoord),
              ((row + k[2 * p + 1]) * stride).clamp(0.0, maxCoord),
            ),
        ];

        out.add(DetectedFace(boundingBox: box, landmarks: landmarks, score: score));
      }
      idx++;
    }
  }
  return out;
}

/// Decodes all 3 strides (8, 16, 32) of raw YuNet output tensors into a
/// list of [DetectedFace], in "model input" pixel space [0, inputSize].
/// Use [scaleYunetDetections] to map the result onto an actual image size.
List<DetectedFace> decodeYunet({
  required List<List<double>> cls8,
  required List<List<double>> obj8,
  required List<List<double>> bbox8,
  required List<List<double>> kps8,
  required List<List<double>> cls16,
  required List<List<double>> obj16,
  required List<List<double>> bbox16,
  required List<List<double>> kps16,
  required List<List<double>> cls32,
  required List<List<double>> obj32,
  required List<List<double>> bbox32,
  required List<List<double>> kps32,
  required int inputSize,
  required double scoreThreshold,
  required double iouThreshold,
  required int maxFaces,
}) {
  final candidates = <DetectedFace>[
    ..._decodeStride(cls: cls8,  obj: obj8,  bbox: bbox8,  kps: kps8,
        stride: 8,  inputSize: inputSize, scoreThreshold: scoreThreshold),
    ..._decodeStride(cls: cls16, obj: obj16, bbox: bbox16, kps: kps16,
        stride: 16, inputSize: inputSize, scoreThreshold: scoreThreshold),
    ..._decodeStride(cls: cls32, obj: obj32, bbox: bbox32, kps: kps32,
        stride: 32, inputSize: inputSize, scoreThreshold: scoreThreshold),
  ];

  return _nms(candidates, iouThreshold: iouThreshold, maxFaces: maxFaces);
}

/// Converts detections in "model input" pixel space [0, inputSize] (as
/// produced by [decodeYunet]) into pixel coordinates for an image of size
/// [width] × [height] — the pixel-space contract every downstream consumer
/// (e.g. [AffineAligner]) expects, matching how BlazeFace's
/// `denormalizeDetections` bridges its own [0,1]-normalised output.
List<DetectedFace> scaleYunetDetections(
  List<DetectedFace> faces,
  int inputSize,
  int width,
  int height,
) {
  final sx = width / inputSize;
  final sy = height / inputSize;
  return [
    for (final f in faces)
      DetectedFace(
        boundingBox: Rect(
          left:   f.boundingBox.left   * sx,
          top:    f.boundingBox.top    * sy,
          right:  f.boundingBox.right  * sx,
          bottom: f.boundingBox.bottom * sy,
        ),
        landmarks: [
          for (final p in f.landmarks) Point(p.x * sx, p.y * sy),
        ],
        score: f.score,
      ),
  ];
}

// ── Centred first-pass crop geometry ────────────────────────────────────────
// A second, independent mitigation for the same whole-frame-squeeze problem
// the refinement pass below addresses — see real-device evidence in
// doc/KR/postmortem/2026-08-14-camera-orientation-and-auto-zoom.md for why
// this exists *in addition to* refinement rather than instead of it:
// refinement crops around the first pass's own bbox, so when that bbox is
// itself imprecise (exactly the small/far-face case it's meant to help),
// the crop can miss the face and refinement falls back with nothing to show
// for it — 100% fallback rate across every real-device trigger logged so
// far. A centred crop needs no prior detection to know where to look, so it
// can't inherit that particular failure mode.

/// The largest centred square crop of a [imageWidth] × [imageHeight] frame
/// — side length = min(imageWidth, imageHeight), so it needs no padding and
/// (for a landscape-shaped frame) discards only the left/right margins.
/// Squeezing a 480×480 crop into YuNet's 160×160 input is a uniform 3x
/// squeeze on both axes, versus a whole 720×480 frame's uneven 4.5x/3x
/// squeeze — more native pixels per face *and* no aspect-ratio distortion,
/// unconditionally, with no detection-confidence-dependent decision logic
/// needed at all (unlike auto-zoom or refinement, both of which need a
/// stable signal from detection to know when/how to act, which is exactly
/// what's unstable for a small/far face in the first place).
Rect centerSquareCropRegion(int imageWidth, int imageHeight) {
  final side = (imageWidth < imageHeight ? imageWidth : imageHeight).toDouble();
  final left = (imageWidth - side) / 2;
  final top = (imageHeight - side) / 2;
  return Rect(left: left, top: top, right: left + side, bottom: top + side);
}

// ── Second-pass refinement geometry ─────────────────────────────────────────
// Pure helpers for YuNetDetector's two-pass refinement: resizeNearest
// squeezes the *whole* camera frame down to inputSize×inputSize, so a face
// that's only a small fraction of the frame gets very few input pixels —
// ordinary per-frame camera sensor noise then moves the decoded landmarks by
// a large fraction of the face's own size. Measured 160-288x higher jitter
// below ~20% face-width than above it, consistently across three test
// photos — see doc/KR/postmortem/2026-08-13-landmark-jitter-frame-fill.md.
// The fix: when the first pass's best face is below that threshold, crop
// tightly around it and re-detect at native resolution.

/// Face-width fraction of the full frame below which [YuNetDetector.detect]
/// re-runs detection against a tighter, native-resolution crop around the
/// first pass's best face.
const double kRefineFaceFraction = 0.20;

/// Crop margin for the refinement pass: fraction of the bbox's larger side
/// added on *each* side of the square crop — same convention as
/// MediaPipeFaceLandmarker's `_cropMarginFraction`
/// (lib/src/landmark/face_landmarker.dart). 0.5 -> crop side = 2x the
/// bbox's own larger side, matching the postmortem sweep's margin=2 point:
/// well inside the low-jitter range measured at margin<=5, with slack for
/// pass-1's own bbox being imprecise for a face this small.
const double kRefineCropMarginFraction = 0.5;

/// True if [boundingBox] (in an image [frameWidth] px wide) is small enough
/// relative to the frame that [YuNetDetector.detect] should refine it with
/// a second, cropped pass.
bool needsYunetRefinement(
  Rect boundingBox,
  int frameWidth, {
  double threshold = kRefineFaceFraction,
}) {
  if (frameWidth <= 0) return false;
  return boundingBox.width / frameWidth < threshold;
}

/// The square, image-bounds-clamped crop region to re-detect within for the
/// refinement pass — centred on [boundingBox], sized via
/// [kRefineCropMarginFraction]. Clamping matches `cropFaceImage`'s
/// (lib/src/image/image_converter.dart) own floor/clamp behaviour exactly,
/// so callers can crop with `cropFaceImage(image, region)` and trust
/// `region.left`/`region.top` as the crop's *actual* top-left for mapping
/// results back afterwards — same pattern as
/// MediaPipeFaceLandmarker.detectLandmarks.
Rect yunetRefinementCropRegion(
  Rect boundingBox,
  int imageWidth,
  int imageHeight, {
  double marginFraction = kRefineCropMarginFraction,
}) {
  final side =
      (boundingBox.width > boundingBox.height ? boundingBox.width : boundingBox.height) *
          (1 + 2 * marginFraction);
  final cx = boundingBox.centerX;
  final cy = boundingBox.centerY;

  final left = (cx - side / 2).floor().clamp(0, imageWidth - 1);
  final top = (cy - side / 2).floor().clamp(0, imageHeight - 1);
  final right = (cx + side / 2).ceil().clamp(left + 1, imageWidth);
  final bottom = (cy + side / 2).ceil().clamp(top + 1, imageHeight);
  return Rect(
    left: left.toDouble(),
    top: top.toDouble(),
    right: right.toDouble(),
    bottom: bottom.toDouble(),
  );
}

/// Maps a [DetectedFace] decoded in a refinement crop's own local pixel
/// space back into the original image's coordinates, given the crop's
/// [region] (as returned by [yunetRefinementCropRegion]).
DetectedFace mapYunetRefinedFace(DetectedFace localFace, Rect region) {
  final b = localFace.boundingBox;
  return DetectedFace(
    boundingBox: Rect(
      left: region.left + b.left,
      top: region.top + b.top,
      right: region.left + b.right,
      bottom: region.top + b.bottom,
    ),
    landmarks: [
      for (final p in localFace.landmarks) Point(region.left + p.x, region.top + p.y),
    ],
    score: localFace.score,
  );
}

// ── Non-Maximum Suppression ───────────────────────────────────────────────────
// Not shared with blazeface_decoder.dart's _nms/_iou: that one clamps
// intersection width/height to [0,1] (valid for its normalised coordinate
// space), which would silently break here where coordinates range over
// [0, inputSize].

List<DetectedFace> _nms(
  List<DetectedFace> detections, {
  required double iouThreshold,
  required int maxFaces,
}) {
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

  final interW = math.max(0.0, interRight  - interLeft);
  final interH = math.max(0.0, interBottom - interTop);
  final interArea = interW * interH;

  if (interArea == 0.0) return 0.0;

  final aArea = a.width * a.height;
  final bArea = b.width * b.height;
  return interArea / (aArea + bArea - interArea);
}
