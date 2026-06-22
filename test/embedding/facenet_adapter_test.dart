// facenet_adapter.dart has no Flutter dependency (unlike arcface_adapter.dart,
// which pulls in tflite_runner.dart → tflite_flutter), so this runs under
// plain `dart test`.
import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:facekit/src/core/models.dart';
import 'package:facekit/src/inference/model_manifest.dart';
import 'package:facekit/src/embedding/adapters/facenet_adapter.dart';

const _facenetManifestJson = '''
{
  "name": "facenet512", "family": "facenet", "file": "facenet512.tflite",
  "input": {"width":160,"height":160,"color":"RGB","layout":"NHWC",
            "normalize":{"mean":[0.0,0.0,0.0],"std":[1.0,1.0,1.0]}},
  "output": {"dim":512,"l2_normalize":true},
  "license": {"tier":"research","redistributable":false,"source":"x","note":"y"}
}
''';

void main() {
  group('FacenetAdapter', () {
    const adapter = FacenetAdapter();
    final manifest = ModelManifest.fromJsonString(_facenetManifestJson);

    test('preprocess produces a [1, 160, 160, 3] nested tensor', () {
      final face = AlignedFace(rgbBytes: Uint8List(160 * 160 * 3), size: 160);
      final tensor = adapter.preprocess(face, manifest) as List;

      expect(tensor.length, 1);
      expect((tensor[0] as List).length, 160);
      expect((tensor[0][0] as List).length, 160);
      expect((tensor[0][0][0] as List).length, 3);
    });

    test('preprocess prewhitens using the image\'s own mean/std', () {
      // Two-value image: half the pixels at 0, half at 100.
      // mean = 50, population std = 50 → every pixel maps to ±1.
      final bytes = Uint8List(160 * 160 * 3);
      for (int i = 0; i < bytes.length; i++) {
        bytes[i] = (i % 2 == 0) ? 0 : 100;
      }
      final face = AlignedFace(rgbBytes: bytes, size: 160);

      final tensor = adapter.preprocess(face, manifest) as List;
      final firstPixel = (tensor[0][0][0] as List).cast<double>();
      // bytes[0]=0, bytes[1]=100, bytes[2]=0 → channel values map to -1,+1,-1
      expect(firstPixel[0], closeTo(-1.0, 1e-6));
      expect(firstPixel[1], closeTo(1.0, 1e-6));
      expect(firstPixel[2], closeTo(-1.0, 1e-6));
    });

    test('preprocess does not divide by zero on a solid-colour image', () {
      final bytes = Uint8List(160 * 160 * 3)..fillRange(0, 160 * 160 * 3, 128);
      final face = AlignedFace(rgbBytes: bytes, size: 160);

      final tensor = adapter.preprocess(face, manifest) as List;
      final firstPixel = (tensor[0][0][0] as List).cast<double>();
      // std=0 → stdAdj falls back to 1/sqrt(N), so output is finite, not NaN/Inf.
      expect(firstPixel[0].isFinite, isTrue);
    });

    test('postprocess L2-normalises when manifest requests it', () {
      final raw = List<double>.filled(512, 0.0);
      raw[0] = 3.0;
      raw[1] = 4.0;
      final embedding = adapter.postprocess(raw, manifest);

      expect(embedding.dim, 512);
      expect(embedding.vector[0], closeTo(0.6, 1e-6));
      expect(embedding.vector[1], closeTo(0.8, 1e-6));
    });
  });
}
