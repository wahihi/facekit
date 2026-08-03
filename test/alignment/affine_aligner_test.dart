import 'dart:math' as math;
import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:facekit/src/core/models.dart';
import 'package:facekit/src/alignment/affine_aligner.dart';

/// Builds a solid-colour FaceImage of given size.
FaceImage _solid(int w, int h, int r, int g, int b) {
  final bytes = Uint8List(w * h * 3);
  for (int i = 0; i < w * h; i++) {
    bytes[i * 3]     = r;
    bytes[i * 3 + 1] = g;
    bytes[i * 3 + 2] = b;
  }
  return FaceImage(rgbBytes: bytes, width: w, height: h);
}

/// Builds a [DetectedFace] with landmarks that match the ArcFace reference
/// (i.e. no transform needed — output should equal the input crop).
DetectedFace _identityFace() {
  // Use arcface112Ref as both source and reference → identity transform.
  return DetectedFace(
    boundingBox: Rect(left: 0, top: 0, right: 112, bottom: 112),
    landmarks: arcface112Ref,
    score: 0.99,
  );
}

void main() {
  group('AffineAligner', () {
    test('output size is 112×112 for arcface', () {
      final aligner = AffineAligner.arcface112();
      final image = _solid(200, 200, 128, 64, 32);
      final face = _identityFace();
      final aligned = aligner.align(image, face);
      expect(aligned.size, 112);
      expect(aligned.rgbBytes.length, 112 * 112 * 3);
    });

    test('output size is 160×160 for facenet', () {
      final aligner = AffineAligner.facenet160();
      final image = _solid(300, 300, 100, 100, 100);
      // Build a minimal face with 5 landmarks positioned near facenet160Ref
      final face = DetectedFace(
        boundingBox: Rect(left: 0, top: 0, right: 160, bottom: 160),
        landmarks: facenet160Ref,
        score: 0.9,
      );
      final aligned = aligner.align(image, face);
      expect(aligned.size, 160);
      expect(aligned.rgbBytes.length, 160 * 160 * 3);
    });

    test('solid colour image stays same colour after identity transform', () {
      final aligner = AffineAligner.arcface112();
      final image = _solid(200, 200, 200, 100, 50);
      final face = _identityFace();
      final aligned = aligner.align(image, face);

      // After identity-like transform on a solid image every pixel should
      // stay the same colour (within bilinear rounding tolerance ±2).
      // NOTE: this cannot detect a wrong rotation/scale/reflection — any
      // affine transform of a solid-colour image is still that same solid
      // colour. It only exercises output plumbing (size, byte layout). The
      // tests below pin down actual transform *correctness* with known
      // numeric answers instead.
      final bytes = aligned.rgbBytes;
      for (int i = 0; i < bytes.length; i += 3) {
        expect(bytes[i],     closeTo(200, 2));
        expect(bytes[i + 1], closeTo(100, 2));
        expect(bytes[i + 2], closeTo(50,  2));
      }
    });

    test('throws when fewer than 5 landmarks provided', () {
      final aligner = AffineAligner.arcface112();
      final image = _solid(100, 100, 0, 0, 0);
      final face = DetectedFace(
        boundingBox: Rect(left: 0, top: 0, right: 100, bottom: 100),
        landmarks: [const Point(10, 10), const Point(20, 10)], // only 2
        score: 0.8,
      );
      expect(() => aligner.align(image, face), throwsArgumentError);
    });
  });

  group('_umeyamaSimilarity (via umeyamaSimilarityForTest)', () {
    test('identical point sets produce the identity matrix', () {
      final m = umeyamaSimilarityForTest(arcface112Ref, arcface112Ref);
      expect(m[0], closeTo(1, 1e-6)); // a
      expect(m[1], closeTo(0, 1e-6)); // b
      expect(m[2], closeTo(0, 1e-4)); // tx
      expect(m[3], closeTo(0, 1e-6)); // c
      expect(m[4], closeTo(1, 1e-6)); // d
      expect(m[5], closeTo(0, 1e-4)); // ty
    });

    test('Umeyama (1991) §III worked example matches the published values', () {
      // src/dst and expected c, R, t are the paper's own numeric example
      // (a reflection case): src=[(0,0),(1,0),(0,2)], dst=[(0,0),(-1,0),(0,2)].
      const src = [Point(0, 0), Point(1, 0), Point(0, 2)];
      const dst = [Point(0, 0), Point(-1, 0), Point(0, 2)];
      final m = umeyamaSimilarityForTest(src, dst);
      final a = m[0], b = m[1], tx = m[2];
      final c = m[3], d = m[4], ty = m[5];

      final det = a * d - b * c; // sigma^2 since R is a pure rotation (det=1)
      expect(det, closeTo(0.52, 1e-2)); // sigma^2 ≈ 0.7211^2 ≈ 0.52

      expect(a, closeTo(0.6, 1e-3));
      expect(b, closeTo(0.4, 1e-3));
      expect(c, closeTo(-0.4, 1e-3));
      expect(d, closeTo(0.6, 1e-3));
      expect(tx, closeTo(-0.8, 1e-3));
      expect(ty, closeTo(0.4, 1e-3));
    });

    test('pure scale+rotation recovers the exact known scale factor', () {
      // Scale arcface112Ref by 2.2x and rotate 12° about its own centroid;
      // fitting back onto arcface112Ref should recover scale = 1/2.2 exactly.
      final theta = 12 * math.pi / 180;
      final cosT = math.cos(theta), sinT = math.sin(theta);
      double cx = 0, cy = 0;
      for (final p in arcface112Ref) { cx += p.x; cy += p.y; }
      cx /= arcface112Ref.length; cy /= arcface112Ref.length;

      final scaled = arcface112Ref.map((p) {
        final dx = p.x - cx, dy = p.y - cy;
        final rx = dx * cosT - dy * sinT, ry = dx * sinT + dy * cosT;
        return Point(cx + 2.2 * rx + 15, cy + 2.2 * ry - 8);
      }).toList();

      final m = umeyamaSimilarityForTest(scaled, arcface112Ref);
      final det = m[0] * m[4] - m[1] * m[3];
      expect(det, closeTo(1 / (2.2 * 2.2), 1e-6)); // sigma^2 = (1/2.2)^2
    });

    test('mirrored correspondence is fit as a rotation, not a reflection', () {
      // Horizontally mirror arcface112Ref about its own centroid — this is a
      // genuine reflection case. The fitted R must still have det=+1 (no
      // reflection artefact), matching Umeyama's reflection-guarded least
      // squares (verified independently against a NumPy reference solver).
      double cx = 0;
      for (final p in arcface112Ref) { cx += p.x; }
      cx /= arcface112Ref.length;
      final mirrored = arcface112Ref.map((p) => Point(2 * cx - p.x, p.y)).toList();

      final m = umeyamaSimilarityForTest(mirrored, arcface112Ref);
      final sigma2 = m[0] * m[0] + m[3] * m[3]; // a^2 + c^2 = sigma^2 (R orthonormal)
      final det = (m[0] * m[4] - m[1] * m[3]) / sigma2;
      expect(det, closeTo(1, 1e-6));
    });
  });

  group('AffineAligner 4-point landmark correspondence', () {
    test(
      'raw BlazeFace eye order (rightEye, leftEye, ...) round-trips to '
      'identity once reordered to match the ArcFace reference',
      () {
        // BlazeFace's real landmark order is rightEye, leftEye, nose, mouth,
        // [rightEar, leftEar] (see blazeface_decoder.dart) — the opposite of
        // the ArcFace reference's [leftEye, rightEye, ...] order. Feeding the
        // reference's own points in *raw BlazeFace order* must still align
        // back to identity; this pins down the eye-index swap in
        // AffineAligner._pickFourPoints (previously missing, which produced
        // a spurious 40-90°+ rotation on real device landmarks — see
        // doc/KR/postmortem for the investigation that found this).
        final mouthMid = Point(
          (arcface112Ref[3].x + arcface112Ref[4].x) / 2,
          (arcface112Ref[3].y + arcface112Ref[4].y) / 2,
        );
        final rawBlazeFaceOrder = [
          arcface112Ref[1], // rightEye (BlazeFace idx 0) == ArcFace ref[1]
          arcface112Ref[0], // leftEye  (BlazeFace idx 1) == ArcFace ref[0]
          arcface112Ref[2], // nose
          mouthMid,         // mouth (single point, no separate corners)
          mouthMid,         // 5th point: unused by _pickFourPoints, only
                             // present to satisfy the ≥5-landmarks check.
        ];

        final m = alignmentMatrixForTest(rawBlazeFaceOrder, arcface112Ref);
        expect(m[0], closeTo(1, 1e-6)); // a
        expect(m[1], closeTo(0, 1e-6)); // b
        expect(m[3], closeTo(0, 1e-6)); // c
        expect(m[4], closeTo(1, 1e-6)); // d
      },
    );

    test(
      'NOT reordering (old buggy eye order) does not round-trip to identity',
      () {
        // Regression guard in the other direction: confirms the test above
        // is actually discriminating, by checking the pre-fix behaviour
        // (feeding points already in ArcFace order, which after
        // _pickFourPoints's swap become mismatched) is measurably NOT
        // identity.
        final mouthMid = Point(
          (arcface112Ref[3].x + arcface112Ref[4].x) / 2,
          (arcface112Ref[3].y + arcface112Ref[4].y) / 2,
        );
        final alreadyArcfaceOrder = [
          arcface112Ref[0],
          arcface112Ref[1],
          arcface112Ref[2],
          mouthMid,
          mouthMid, // 5th point: unused, only satisfies the ≥5-landmarks check.
        ];
        final m = alignmentMatrixForTest(alreadyArcfaceOrder, arcface112Ref);
        expect((m[0] - 1).abs() > 1e-3 || m[1].abs() > 1e-3, isTrue);
      },
    );
  });
}
