// 5-point similarity transform (scale + rotation + translation, no shear).
// Maps detected landmarks onto the model-specific canonical reference positions,
// then bilinear-samples the aligned patch.
//
// Source:
//   Umeyama (1991) "Least-squares estimation of transformation parameters
//   between two point patterns." IEEE TPAMI 13(4):376–380.
//   Reference coordinates from ArcFace/InsightFace open-source code (MIT):
//   https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py

import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show debugPrint, visibleForTesting;
import '../core/debug_flags.dart';
import '../core/models.dart';
import '../core/contracts.dart';

/// Canonical 5-point reference coordinates for a 112×112 aligned patch.
/// Order: leftEye, rightEye, nose, leftMouthCorner, rightMouthCorner.
/// Source: ArcFace paper + InsightFace open implementation (MIT licence).
const arcface112Ref = [
  Point(38.2946, 51.6963),
  Point(73.5318, 51.5014),
  Point(56.0252, 71.7366),
  Point(41.5493, 92.3655),
  Point(70.7299, 92.2041),
];

/// Canonical 5-point reference coordinates for a 160×160 aligned patch (FaceNet).
/// Source: FaceNet paper alignment implementation (Apache 2.0).
const facenet160Ref = [
  Point(55.0, 67.0),
  Point(105.0, 67.0),
  Point(80.0, 100.0),
  Point(60.0, 133.0),
  Point(100.0, 133.0),
];

// DEBUG (see core/debug_flags.dart) — counts align() calls so the printed
// matrix below can be matched, in log order, against face_pipeline.dart's
// LANDMARKS log (printed just before align() is called) and its crop dump
// (printed just after).
int _debugAlignCallCount = 0;

/// Implements [FaceAligner] using a 5-point similarity transform.
class AffineAligner implements FaceAligner {
  final List<Point> referencePoints;
  final int outputSize;

  const AffineAligner({
    required this.referencePoints,
    required this.outputSize,
  });

  factory AffineAligner.arcface112() =>
      const AffineAligner(referencePoints: arcface112Ref, outputSize: 112);

  factory AffineAligner.facenet160() =>
      const AffineAligner(referencePoints: facenet160Ref, outputSize: 160);

  @override
  AlignedFace align(FaceImage image, DetectedFace face) {
    // BlazeFace gives 6 keypoints, order rightEye[0], leftEye[1], nose[2],
    // mouth[3], rightEar[4], leftEar[5] (see blazeface_decoder.dart) — note
    // this is opposite the eye order this file used to assume. Confirmed by
    // reconstructing real landmark geometry from a device debug dump: with
    // the old (wrong) eye order the fitted rotation was 42-90° and unstable
    // across frames; swapping the two eye indices below drops it to a
    // consistent 13-24° (ordinary head tilt). Ears are dropped entirely —
    // BlazeFace only gives a single mouth-centre point, not two corners, so
    // there's no good BlazeFace point to pair with the 5th ArcFace reference
    // point (right mouth corner); an ear midpoint is not a mouth corner and
    // pulled the fit even further off. 4-point similarity (eyes+nose+mouth,
    // matched against the ArcFace reference with its two mouth-corner points
    // collapsed to their midpoint) avoids that bad correspondence.
    final src = _pickFourPoints(face.landmarks);
    final dst = _fourPointReference(referencePoints);

    final m = _umeyamaSimilarity(src, dst);
    if (kFacekitVerboseDebug) {
      _debugAlignCallCount++;
      debugPrint(
        '[AffineAligner] DEBUG MATRIX #$_debugAlignCallCount '
        'a=${m[0].toStringAsFixed(4)} b=${m[1].toStringAsFixed(4)} tx=${m[2].toStringAsFixed(1)} '
        'c=${m[3].toStringAsFixed(4)} d=${m[4].toStringAsFixed(4)} ty=${m[5].toStringAsFixed(1)} '
        'src=$src',
      );
    }
    final rgb = _warpBilinear(image, m, outputSize);

    return AlignedFace(rgbBytes: rgb, size: outputSize);
  }

  /// Pick 4 usable landmark points from the detector output.
  /// BlazeFace order: rightEye[0], leftEye[1], noseTip[2], mouth[3],
  /// rightEar[4], leftEar[5] (see blazeface_decoder.dart). Reordered here to
  /// [leftEye, rightEye, nose, mouth] to match the ArcFace reference point
  /// order; ears are dropped (see comment in [align]).
  static List<Point> _pickFourPoints(List<Point> lm) {
    if (lm.length < 5) throw ArgumentError('Need ≥5 landmarks, got ${lm.length}');
    return [lm[1], lm[0], lm[2], lm[3]];
  }

  /// Collapses a 5-point ArcFace-style reference (two mouth corners) into 4
  /// points by averaging the mouth corners into a single mouth-centre point,
  /// matching what BlazeFace's single mouth landmark can actually provide.
  static List<Point> _fourPointReference(List<Point> ref5) {
    assert(ref5.length == 5);
    final mouthMid = Point(
      (ref5[3].x + ref5[4].x) / 2,
      (ref5[3].y + ref5[4].y) / 2,
    );
    return [ref5[0], ref5[1], ref5[2], mouthMid];
  }
}

/// Testing hook — lets tests exercise the private similarity solver in
/// isolation, independent of BlazeFace landmark picking/reordering.
@visibleForTesting
List<double> umeyamaSimilarityForTest(List<Point> s, List<Point> d) =>
    _umeyamaSimilarity(s, d);

/// Testing hook — computes the same alignment matrix [AffineAligner.align]
/// does (4-point pick + reference collapse + Umeyama fit), without needing a
/// [FaceImage]/[DetectedFace] or running the pixel warp. [landmarks] is raw
/// BlazeFace order (rightEye, leftEye, nose, mouth, [rightEar, leftEar]);
/// [referencePoints5] is a 5-point ArcFace-style reference (e.g.
/// [arcface112Ref]).
@visibleForTesting
List<double> alignmentMatrixForTest(
  List<Point> landmarks,
  List<Point> referencePoints5,
) =>
    _umeyamaSimilarity(
      AffineAligner._pickFourPoints(landmarks),
      AffineAligner._fourPointReference(referencePoints5),
    );

/// Estimates the optimal similarity matrix M (2×3) that maps [src] → [dst]
/// using Umeyama's closed-form least-squares method.
///
/// Returns [a, b, tx, c, d, ty] where the 2×2 rotation-scale part is [[a,b],[c,d]].
List<double> _umeyamaSimilarity(List<Point> src, List<Point> dst) {
  assert(src.length == dst.length && src.isNotEmpty);
  final n = src.length;

  // Centroids
  double srcMx = 0, srcMy = 0, dstMx = 0, dstMy = 0;
  for (int i = 0; i < n; i++) {
    srcMx += src[i].x; srcMy += src[i].y;
    dstMx += dst[i].x; dstMy += dst[i].y;
  }
  srcMx /= n; srcMy /= n;
  dstMx /= n; dstMy /= n;

  // Variance and cross-covariance of the centred point sets
  double srcVar = 0;
  double cov00 = 0, cov01 = 0, cov10 = 0, cov11 = 0;

  for (int i = 0; i < n; i++) {
    final sx = src[i].x - srcMx, sy = src[i].y - srcMy;
    final dx = dst[i].x - dstMx, dy = dst[i].y - dstMy;
    srcVar += sx * sx + sy * sy;
    cov00 += dx * sx; cov01 += dx * sy;
    cov10 += dy * sx; cov11 += dy * sy;
  }
  srcVar /= n;
  cov00 /= n; cov01 /= n; cov10 /= n; cov11 /= n;

  // 2×2 SVD of covariance matrix via Jacobi (analytic for 2×2)
  final svd = _svd2x2(cov00, cov01, cov10, cov11);
  final u = svd.$1; // 2×2 unitary
  final s = svd.$2; // singular values [s0, s1]
  final vt = svd.$3; // 2×2 unitary transposed

  // det(U) * det(V) sign correction (reflection guard)
  final detU = u[0] * u[3] - u[1] * u[2];
  final detVt = vt[0] * vt[3] - vt[1] * vt[2];
  final sign = (detU * detVt < 0) ? -1.0 : 1.0;

  final scaledS = [s[0], sign * s[1]];

  // Scale
  final sigma = (scaledS[0] + scaledS[1]) / srcVar;

  // Rotation R = U * diag(1, sign) * Vt
  final r00 = u[0] * vt[0] + u[1] * sign * vt[2];
  final r01 = u[0] * vt[1] + u[1] * sign * vt[3];
  final r10 = u[2] * vt[0] + u[3] * sign * vt[2];
  final r11 = u[2] * vt[1] + u[3] * sign * vt[3];

  final a = sigma * r00;
  final b = sigma * r01;
  final c = sigma * r10;
  final d = sigma * r11;
  final tx = dstMx - a * srcMx - b * srcMy;
  final ty = dstMy - c * srcMx - d * srcMy;

  return [a, b, tx, c, d, ty];
}

/// Analytically computes the SVD of a 2×2 matrix [[m00,m01],[m10,m11]].
/// Returns (U, [s0,s1], Vt) where U and Vt are flat row-major 2×2 matrices.
(List<double>, List<double>, List<double>) _svd2x2(
    double m00, double m01, double m10, double m11) {
  // Use the standard 2×2 SVD formula via the cross product method.
  final e = (m00 + m11) / 2;
  final f = (m00 - m11) / 2;
  final g = (m10 + m01) / 2;
  final h = (m10 - m01) / 2;

  final q = math.sqrt(e * e + h * h);
  final r = math.sqrt(f * f + g * g);

  final s0 = q + r;
  final s1 = q - r;

  final a1 = math.atan2(g, f);
  final a2 = math.atan2(h, e);
  final theta = (a2 - a1) / 2;
  final phi   = (a2 + a1) / 2;

  // U  = rot(phi)
  final uCos = math.cos(phi), uSin = math.sin(phi);
  final u = [uCos, -uSin, uSin, uCos];

  // Vt = rot(-theta) transposed = rot(theta)
  final vCos = math.cos(theta), vSin = math.sin(theta);
  final vt = [vCos, -vSin, vSin, vCos];

  return (u, [s0, s1], vt);
}

/// Applies the 2×3 affine matrix [m] = [a,b,tx, c,d,ty] to [image],
/// sampling with bilinear interpolation, and returns the output RGB patch.
Uint8List _warpBilinear(FaceImage image, List<double> m, int size) {
  final a = m[0], b = m[1], tx = m[2];
  final c = m[3], d = m[4], ty = m[5];

  // Invert 2×2 to map output pixel → input pixel
  final det = a * d - b * c;
  if (det.abs() < 1e-10) throw StateError('singular transform');
  final ia = d / det, ib = -b / det;
  final ic = -c / det, id = a / det;
  final itx = (b * ty - d * tx) / det;
  final ity = (c * tx - a * ty) / det;

  final out = Uint8List(size * size * 3);
  final src = image.rgbBytes;
  final W = image.width, H = image.height;

  for (int row = 0; row < size; row++) {
    for (int col = 0; col < size; col++) {
      final srcX = ia * col + ib * row + itx;
      final srcY = ic * col + id * row + ity;

      final x0 = srcX.floor().clamp(0, W - 1);
      final y0 = srcY.floor().clamp(0, H - 1);
      final x1 = (x0 + 1).clamp(0, W - 1);
      final y1 = (y0 + 1).clamp(0, H - 1);

      final wx = srcX - x0;
      final wy = srcY - y0;
      final wx1 = 1.0 - wx, wy1 = 1.0 - wy;

      final dstIdx = (row * size + col) * 3;
      for (int ch = 0; ch < 3; ch++) {
        final p00 = src[(y0 * W + x0) * 3 + ch];
        final p10 = src[(y0 * W + x1) * 3 + ch];
        final p01 = src[(y1 * W + x0) * 3 + ch];
        final p11 = src[(y1 * W + x1) * 3 + ch];
        out[dstIdx + ch] =
            (wy1 * (wx1 * p00 + wx * p10) + wy * (wx1 * p01 + wx * p11))
                .round()
                .clamp(0, 255);
      }
    }
  }

  return out;
}
