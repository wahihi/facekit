// Smoke test: loads the real ArcFace (buffalo_l) .tflite model and runs
// inference on a synthetic aligned face to verify the embedder produces a
// valid 512-dim L2-normalised embedding.
//
// The model is Demo/research-tier (non-redistributable) and therefore not
// bundled with the repo — see assets/models/arcface_buffalo_l/manifest.json.
// This test looks for it at a local, developer-provided path and skips
// gracefully if not found, so the suite stays green on machines that haven't
// fetched the model.
//
// Run with: flutter test test/embedding/face_embedder_smoke_test.dart
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:facekit/src/core/models.dart';
import 'package:facekit/src/embedding/face_embedder.dart';
import 'package:facekit/src/inference/model_manifest.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  const manifestPath = 'assets/models/arcface_buffalo_l/manifest.json';
  final modelPath =
      '${Platform.environment['HOME']}/development/models/w600k_r50_tf/w600k_r50_float32.tflite';

  final modelExists = File(modelPath).existsSync();

  test(
    'ArcFace model loads and produces a 512-dim L2-normalised embedding',
    () async {
      final manifestJson = File(manifestPath).readAsStringSync();
      final manifest = ModelManifest.fromJsonString(manifestJson);

      final embedder = await TfliteFaceEmbedder.fromFile(
        tflitePath: modelPath,
        manifest: manifest,
      );

      // Solid grey 112×112 aligned face — no real face, but the model must
      // not crash and must return a well-formed embedding.
      final grey = Uint8List(112 * 112 * 3)..fillRange(0, 112 * 112 * 3, 128);
      final face = AlignedFace(rgbBytes: grey, size: 112);

      final embedding = await embedder.embed(face);

      expect(embedding.dim, 512);
      double normSq = 0.0;
      for (final v in embedding.vector) {
        normSq += v * v;
      }
      expect(normSq, closeTo(1.0, 1e-3)); // L2-normalised → unit length

      embedder.dispose();
    },
    skip: modelExists ? false : 'w600k_r50_float32.tflite not present locally',
  );

  // AdaFace's exported graph has two output tensors — "feature" [1,512] and
  // a pre-normalisation "norm" [1,1] scalar (confirmed via the ONNX export,
  // see export_onnx.py output_names=['feature','norm']) — unlike ArcFace's
  // single-output graph above. embed() must consume only the first output
  // and not crash on the second; see TfliteFaceEmbedder.embed and
  // TfliteRunner.zeroTensor.
  const adafaceManifestPath = 'assets/models/adaface_ir101_webface12m/manifest.json';
  final adafaceModelPath =
      '${Platform.environment['HOME']}/development/models/adaface/adaface_ir101_webface12m.tflite';

  final adafaceModelExists = File(adafaceModelPath).existsSync();

  test(
    'AdaFace model (2 output tensors: feature+norm) loads and produces a 512-dim embedding',
    () async {
      final manifestJson = File(adafaceManifestPath).readAsStringSync();
      final manifest = ModelManifest.fromJsonString(manifestJson);

      final embedder = await TfliteFaceEmbedder.fromFile(
        tflitePath: adafaceModelPath,
        manifest: manifest,
      );

      final grey = Uint8List(112 * 112 * 3)..fillRange(0, 112 * 112 * 3, 128);
      final face = AlignedFace(rgbBytes: grey, size: 112);

      final embedding = await embedder.embed(face);

      expect(embedding.dim, 512);
      double normSq = 0.0;
      for (final v in embedding.vector) {
        normSq += v * v;
      }
      expect(normSq, closeTo(1.0, 1e-3));

      embedder.dispose();
    },
    skip: adafaceModelExists
        ? false
        : 'adaface_ir101_webface12m.tflite not present locally '
            '(local TFLite conversion currently OOMs on this machine — see doc/KR/adaface_verification.md)',
  );
}
