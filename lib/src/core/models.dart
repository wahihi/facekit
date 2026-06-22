// Source: design spec §6 — clean-room data models, no proprietary reference.
// dart:typed_data is the only external dependency (no Flutter, no dart:ui).

import 'dart:typed_data';

/// Raw image fed into the pipeline. RGB888, row-major, top-left origin.
class FaceImage {
  final Uint8List rgbBytes;
  final int width;
  final int height;

  const FaceImage({
    required this.rgbBytes,
    required this.width,
    required this.height,
  });

  int get byteLength => width * height * 3;
}

/// Axis-aligned bounding box in pixel coordinates (top-left origin).
class Rect {
  final double left;
  final double top;
  final double right;
  final double bottom;

  const Rect({
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
  });

  double get width => right - left;
  double get height => bottom - top;
  double get centerX => (left + right) / 2;
  double get centerY => (top + bottom) / 2;

  @override
  String toString() =>
      'Rect(l:$left, t:$top, r:$right, b:$bottom)';
}

/// 2-D point in pixel coordinates.
class Point {
  final double x;
  final double y;

  const Point(this.x, this.y);

  @override
  String toString() => 'Point($x, $y)';
}

/// Single detected face from the detector.
/// [landmarks]: BlazeFace 6 keypoints — [leftEye, rightEye, nose, mouth, leftEar, rightEar]
class DetectedFace {
  final Rect boundingBox;
  final List<Point> landmarks;
  final double score;

  const DetectedFace({
    required this.boundingBox,
    required this.landmarks,
    required this.score,
  });
}

/// Cropped & aligned face patch ready for the embedder.
/// Always square: [size] × [size] pixels, RGB888, row-major.
class AlignedFace {
  final Uint8List rgbBytes;
  final int size; // 112 for ArcFace/AdaFace/MobileFaceNet, 160 for FaceNet

  const AlignedFace({required this.rgbBytes, required this.size});
}

/// L2-normalised embedding vector produced by the embedder.
class Embedding {
  final Float32List vector;

  const Embedding(this.vector);

  int get dim => vector.length;
}

/// One enrolled identity in the gallery.
class Enrollment {
  final String id;
  final Embedding embedding;
  final Map<String, String> meta;

  const Enrollment({
    required this.id,
    required this.embedding,
    this.meta = const {},
  });
}

/// Result of a 1:N gallery match.
class MatchResult {
  /// null when no gallery entry passes the threshold.
  final String? matchedId;
  final double similarity; // cosine similarity, 0–1
  final bool accepted;     // true when similarity >= threshold

  const MatchResult({
    this.matchedId,
    required this.similarity,
    required this.accepted,
  });
}
