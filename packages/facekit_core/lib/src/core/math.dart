// Source: standard linear-algebra definitions (L2 norm, cosine similarity)
// and the Eye Aspect Ratio formula from Soukupová & Čech, "Real-Time Eye
// Blink Detection using Facial Landmarks" (2016) — a public, widely-cited
// formula, not proprietary code.
// Pure functions — no side effects, no Flutter dependency.

import 'dart:math' as math;
import 'dart:typed_data';

import 'models.dart';

/// Returns a new vector that is [v] scaled to unit length (L2 norm = 1).
/// Throws [ArgumentError] if [v] is empty or all-zero.
Float32List l2Normalize(Float32List v) {
  if (v.isEmpty) throw ArgumentError('vector must not be empty');

  double sumSq = 0.0;
  for (final x in v) {
    sumSq += x * x;
  }

  final norm = math.sqrt(sumSq);
  if (norm == 0.0) throw ArgumentError('zero vector cannot be normalised');

  final out = Float32List(v.length);
  for (int i = 0; i < v.length; i++) {
    out[i] = v[i] / norm;
  }
  return out;
}

/// Cosine similarity between two L2-normalised vectors.
/// Returns a value in [−1, 1]; higher = more similar.
/// Both vectors must have the same [length].
double cosineSimilarity(Float32List a, Float32List b) {
  if (a.length != b.length) {
    throw ArgumentError('vectors must have equal length: ${a.length} vs ${b.length}');
  }
  if (a.isEmpty) throw ArgumentError('vectors must not be empty');

  double dot = 0.0;
  for (int i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
  }
  // Clamp to [-1, 1] to guard against floating-point drift beyond unit sphere.
  return dot.clamp(-1.0, 1.0);
}

/// L2 norm (Euclidean length) of [v].
double l2Norm(Float32List v) {
  double sumSq = 0.0;
  for (final x in v) {
    sumSq += x * x;
  }
  return math.sqrt(sumSq);
}

double _distance(Point a, Point b) {
  final dx = a.x - b.x;
  final dy = a.y - b.y;
  return math.sqrt(dx * dx + dy * dy);
}

/// Eye Aspect Ratio (EAR) — Soukupová & Čech (2016).
///
/// [eyePoints] must be exactly 6 points around one eye, in this order:
/// [outerCorner, upperOuter, upperInner, innerCorner, lowerInner, lowerOuter].
/// Open eyes give a higher ratio; a closed/blinking eye's ratio drops sharply
/// because the vertical (eyelid) distances shrink while the horizontal
/// (corner-to-corner) distance stays roughly constant.
///
/// EAR = (‖upperOuter−lowerOuter‖ + ‖upperInner−lowerInner‖) / (2·‖outerCorner−innerCorner‖)
double eyeAspectRatio(List<Point> eyePoints) {
  if (eyePoints.length != 6) {
    throw ArgumentError('eyePoints must have exactly 6 points, got ${eyePoints.length}');
  }
  final horizontal = _distance(eyePoints[0], eyePoints[3]);
  if (horizontal == 0.0) throw ArgumentError('eye corner points coincide — degenerate input');

  final vertical1 = _distance(eyePoints[1], eyePoints[5]);
  final vertical2 = _distance(eyePoints[2], eyePoints[4]);
  return (vertical1 + vertical2) / (2.0 * horizontal);
}
