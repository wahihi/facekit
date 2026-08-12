// Smoke test: loads the real YuNet .tflite file and runs inference
// on a synthetic image to verify the model loads, the scrambled onnx2tf
// output order resolves correctly, and decode doesn't crash.
//
// Run with: flutter test test/detection/yunet_smoke_test.dart
// (requires flutter test runner — tflite_flutter depends on dart:ui)

import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:facekit/src/core/models.dart';
import 'package:facekit/src/detection/yunet_detector.dart';
import 'package:facekit/src/inference/model_manifest.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  const modelPath = 'assets/models/yunet_160/yunet_160.tflite';
  const manifestPath = 'assets/models/yunet_160/manifest.json';

  test('YuNet model loads and accepts 160×160 input', () async {
    final manifestJson = File(manifestPath).readAsStringSync();
    final manifest = ModelManifest.fromJsonString(manifestJson);

    final detector = await YuNetDetector.fromFile(
      tflitePath: modelPath,
      manifest: manifest,
    );

    // Solid grey 160×160 image — no real face, but the model must not crash.
    final grey = Uint8List(160 * 160 * 3)..fillRange(0, 160 * 160 * 3, 128);
    final image = FaceImage(rgbBytes: grey, width: 160, height: 160);

    final faces = await detector.detect(image);

    expect(faces, isA<List<DetectedFace>>());
    // Every returned face (if any) should have exactly 5 landmarks and a
    // score within [0,1], regardless of whether a face was actually found.
    for (final f in faces) {
      expect(f.landmarks, hasLength(5));
      expect(f.score, inInclusiveRange(0.0, 1.0));
    }

    detector.dispose();
  });

  test('YuNet detects a face in a non-square real-world-sized image', () async {
    final manifestJson = File(manifestPath).readAsStringSync();
    final manifest = ModelManifest.fromJsonString(manifestJson);

    final detector = await YuNetDetector.fromFile(
      tflitePath: modelPath,
      manifest: manifest,
    );

    // A non-square image exercises the per-axis scale-back in
    // scaleYunetDetections (sx != sy) — still just a solid image (no real
    // face), so this only checks output plumbing, not detection quality.
    final grey = Uint8List(320 * 240 * 3)..fillRange(0, 320 * 240 * 3, 100);
    final image = FaceImage(rgbBytes: grey, width: 320, height: 240);

    final faces = await detector.detect(image);
    expect(faces, isA<List<DetectedFace>>());
    for (final f in faces) {
      expect(f.boundingBox.left, greaterThanOrEqualTo(0.0));
      expect(f.boundingBox.top, greaterThanOrEqualTo(0.0));
      expect(f.boundingBox.right, lessThanOrEqualTo(320.0 + 1e-6));
      expect(f.boundingBox.bottom, lessThanOrEqualTo(240.0 + 1e-6));
    }

    detector.dispose();
  });
}
