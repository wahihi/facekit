// Smoke test: loads the real bundled face_landmark_478 .tflite and runs
// inference on a synthetic image to verify it produces 478 landmarks.
//
// Bundled (Apache 2.0), not BYOM — unlike the embedder smoke tests, this
// always runs, no graceful skip.
//
// Run with: flutter test test/landmark/face_landmarker_smoke_test.dart
// (requires flutter test runner — tflite_flutter depends on dart:ui)

import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:facekit/src/core/models.dart';
import 'package:facekit/src/inference/model_manifest.dart';
import 'package:facekit/src/landmark/face_landmarker.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  const modelPath = 'assets/models/face_landmark_478/face_landmark_478.tflite';
  const manifestPath = 'assets/models/face_landmark_478/manifest.json';

  test('face_landmark_478 model loads and returns 478 landmarks', () async {
    final manifestJson = File(manifestPath).readAsStringSync();
    final manifest = ModelManifest.fromJsonString(manifestJson);

    final landmarker = await MediaPipeFaceLandmarker.fromFile(
      tflitePath: modelPath,
      manifest: manifest,
    );

    // Solid grey 320×320 image with a fake centred "face" box — no real
    // face, but the model must not crash and must return 478 points.
    const w = 320, h = 320;
    final grey = Uint8List(w * h * 3)..fillRange(0, w * h * 3, 128);
    final image = FaceImage(rgbBytes: grey, width: w, height: h);
    const face = DetectedFace(
      boundingBox: Rect(left: 60, top: 60, right: 260, bottom: 260),
      landmarks: [],
      score: 0.9,
    );

    final landmarks = await landmarker.detectLandmarks(image, face);

    expect(landmarks, isNotNull);
    expect(landmarks!.points.length, 478);
    // Landmarks should land within (or very near) the original image bounds
    // given the crop maps back to image space.
    for (final p in landmarks.points) {
      expect(p.x, inInclusiveRange(-50.0, w + 50.0));
      expect(p.y, inInclusiveRange(-50.0, h + 50.0));
    }

    landmarker.dispose();
  });
}
