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
}
