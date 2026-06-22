// Source: standard linear-algebra definitions (L2 norm, cosine similarity).
// Pure functions — no side effects, no Flutter dependency.

import 'dart:math' as math;
import 'dart:typed_data';

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
