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
    // BlazeFace gives 6 keypoints: leftEye[0], rightEye[1], nose[2],
    // mouth[3], leftEar[4], rightEar[5].
    // We map the first 5 (skip ears) to the 5-point reference.
    final src = _pickFivePoints(face.landmarks);
    final dst = referencePoints;

    final m = _umeyamaSimilarity(src, dst);
    final rgb = _warpBilinear(image, m, outputSize);

    return AlignedFace(rgbBytes: rgb, size: outputSize);
  }

  /// Pick 5 usable landmark points from the detector output.
  /// BlazeFace order: leftEye, rightEye, noseTip, mouth, leftEar, rightEar.
  static List<Point> _pickFivePoints(List<Point> lm) {
    if (lm.length < 5) throw ArgumentError('Need ≥5 landmarks, got ${lm.length}');
    // indices 0,1,2,3 and derive 5th as midpoint of ears if available,
    // otherwise reuse mouth.
    final fifth = lm.length >= 6
        ? Point((lm[4].x + lm[5].x) / 2, (lm[4].y + lm[5].y) / 2)
        : lm[3];
    return [lm[0], lm[1], lm[2], lm[3], fifth];
  }
}

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
  final vt = [vCos, vSin, -vSin, vCos];

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
