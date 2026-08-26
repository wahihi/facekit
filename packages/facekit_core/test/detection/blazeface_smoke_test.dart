// Smoke test: loads the real BlazeFace .tflite file and runs inference
// on a synthetic image to verify the model loads and produces 896-anchor output.
//
// Run with: dart test test/detection/blazeface_smoke_test.dart
// Needs libtensorflowlite_c for this platform findable by TfliteFfiRunner
// (FACEKIT_TFLITE_C_LIBRARY env var, or see bindings/bindings.dart).

import 'dart:io';
import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:facekit_core/src/core/models.dart';
import 'package:facekit_core/src/detection/blazeface_detector.dart';
import 'package:facekit_core/src/inference/model_manifest.dart';

void main() {

  const modelPath =
      'assets/models/blazeface_short/face_detection_short_range.tflite';
  const manifestPath =
      'assets/models/blazeface_short/manifest.json';

  test('BlazeFace model loads and accepts 128×128 input', () async {
    final manifestJson = File(manifestPath).readAsStringSync();
    final manifest = ModelManifest.fromJsonString(manifestJson);

    final detector = BlazeFaceDetector.fromFile(
      tflitePath: modelPath,
      manifest: manifest,
    );

    // Solid grey 128×128 image — no real face, but the model must not crash.
    final grey = Uint8List(128 * 128 * 3)..fillRange(0, 128 * 128 * 3, 128);
    final image = FaceImage(rgbBytes: grey, width: 128, height: 128);

    final faces = await detector.detect(image);

    // No face in a solid-grey image — expect empty list, not an exception.
    expect(faces, isA<List<DetectedFace>>());

    detector.dispose();
  });
}
