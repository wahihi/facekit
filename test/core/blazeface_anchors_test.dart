import 'package:test/test.dart';
// Direct imports — avoids pulling in tflite_flutter → flutter → dart:ui
import 'package:facekit/src/core/models.dart';
import 'package:facekit/src/detection/blazeface_anchors.dart';
import 'package:facekit/src/detection/blazeface_decoder.dart';

void main() {
  group('BlazeFace anchor generation', () {
    late List<Anchor> anchors;

    setUpAll(() => anchors = generateBlazeFaceShortAnchors());

    test('generates exactly 896 anchors', () {
      expect(anchors.length, 896);
    });

    test('all anchor centers are in (0, 1)', () {
      for (final a in anchors) {
        expect(a.xCenter, greaterThan(0.0));
        expect(a.xCenter, lessThan(1.0));
        expect(a.yCenter, greaterThan(0.0));
        expect(a.yCenter, lessThan(1.0));
      }
    });

    test('fixed anchor size is 1.0', () {
      for (final a in anchors) {
        expect(a.width,  closeTo(1.0, 1e-9));
        expect(a.height, closeTo(1.0, 1e-9));
      }
    });

    test('first anchor is at top-left of first feature map (16×16)', () {
      // stride=8, featureMap=16×16; first cell center = (0+0.5)/16
      expect(anchors[0].xCenter, closeTo(0.5 / 16, 1e-6));
      expect(anchors[0].yCenter, closeTo(0.5 / 16, 1e-6));
    });

    test('anchors 0 and 1 share same center (2 per cell)', () {
      expect(anchors[0].xCenter, anchors[1].xCenter);
      expect(anchors[0].yCenter, anchors[1].yCenter);
    });
  });

  group('BlazeFace decoder', () {
    test('empty output → no detections', () {
      final regressors    = List.generate(896, (_) => List.filled(16, 0.0));
      final classificators = List.generate(896, (_) => [-100.0]); // sigmoid≈0

      final faces = decodeBlazeFace(
        regressors:     regressors,
        classificators: classificators,
        anchors:        blazeFaceShortAnchors,
        scoreThreshold: 0.5,
        iouThreshold:   0.3,
        maxFaces:       10,
      );

      expect(faces, isEmpty);
    });

    test('high-score detection is returned', () {
      final regressors    = List.generate(896, (_) => List.filled(16, 0.0));
      final classificators = List.generate(896, (_) => [-100.0]);

      // Anchor 0 at xCenter≈0.031, yCenter≈0.031
      // raw box all zeros → cx=anchor_cx, cy=anchor_cy, w=exp(0)=1.0, h=1.0
      classificators[0] = [100.0]; // sigmoid≈1.0

      final faces = decodeBlazeFace(
        regressors:     regressors,
        classificators: classificators,
        anchors:        blazeFaceShortAnchors,
        scoreThreshold: 0.5,
        iouThreshold:   0.3,
        maxFaces:       10,
      );

      expect(faces.length, 1);
      expect(faces[0].score, greaterThan(0.99));
      expect(faces[0].landmarks.length, 6);
    });

    test('NMS suppresses overlapping detections', () {
      final regressors    = List.generate(896, (_) => List.filled(16, 0.0));
      final classificators = List.generate(896, (_) => [-100.0]);

      // Anchors 0 and 1 share the same center → IoU = 1.0 → should be merged to 1
      classificators[0] = [100.0];
      classificators[1] = [100.0];

      final faces = decodeBlazeFace(
        regressors:     regressors,
        classificators: classificators,
        anchors:        blazeFaceShortAnchors,
        scoreThreshold: 0.5,
        iouThreshold:   0.3,
        maxFaces:       10,
      );

      expect(faces.length, 1);
    });

    test('maxFaces cap respected', () {
      final regressors    = List.generate(896, (_) => List.filled(16, 0.0));
      // Make every anchor score 1.0
      final classificators = List.generate(896, (_) => [100.0]);

      final faces = decodeBlazeFace(
        regressors:     regressors,
        classificators: classificators,
        anchors:        blazeFaceShortAnchors,
        scoreThreshold: 0.5,
        iouThreshold:   0.0, // disable NMS suppression
        maxFaces:       3,
      );

      expect(faces.length, lessThanOrEqualTo(3));
    });
  });

  group('denormalizeDetections', () {
    test('scales bounding box and landmarks to the target image size', () {
      const face = DetectedFace(
        boundingBox: Rect(left: 0.25, top: 0.5, right: 0.75, bottom: 1.0),
        landmarks: [Point(0.5, 0.5)],
        score: 0.9,
      );

      final out = denormalizeDetections([face], 200, 400);

      expect(out.length, 1);
      expect(out[0].boundingBox.left,   closeTo(50,  1e-9));
      expect(out[0].boundingBox.top,    closeTo(200, 1e-9));
      expect(out[0].boundingBox.right,  closeTo(150, 1e-9));
      expect(out[0].boundingBox.bottom, closeTo(400, 1e-9));
      expect(out[0].landmarks[0].x, closeTo(100, 1e-9));
      expect(out[0].landmarks[0].y, closeTo(200, 1e-9));
      expect(out[0].score, 0.9);
    });

    test('is a no-op on a unit (1×1) image', () {
      const face = DetectedFace(
        boundingBox: Rect(left: 0.1, top: 0.2, right: 0.3, bottom: 0.4),
        landmarks: [Point(0.5, 0.6)],
        score: 0.5,
      );

      final out = denormalizeDetections([face], 1, 1);

      expect(out[0].boundingBox.left, closeTo(0.1, 1e-9));
      expect(out[0].landmarks[0].y,   closeTo(0.6, 1e-9));
    });
  });
}
