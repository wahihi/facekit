// Uses flutter_test (not plain `dart test`) because arcface_adapter.dart
// imports tflite_runner.dart, which depends on tflite_flutter → dart:ui.
// Run with: flutter test test/embedding/arcface_adapter_test.dart
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:facekit/src/core/models.dart';
import 'package:facekit/src/inference/model_manifest.dart';
import 'package:facekit/src/embedding/adapters/arcface_adapter.dart';
import 'package:facekit/src/embedding/adapters/facenet_adapter.dart';
import 'package:facekit/src/embedding/face_embedder.dart';

const _arcfaceManifestJson = '''
{
  "name": "arcface_buffalo_l", "family": "arcface", "file": "w600k_r50.tflite",
  "input": {"width":112,"height":112,"color":"RGB","layout":"NHWC",
            "normalize":{"mean":[127.5,127.5,127.5],"std":[127.5,127.5,127.5]}},
  "output": {"dim":512,"l2_normalize":true},
  "license": {"tier":"research","redistributable":false,"source":"x","note":"y"}
}
''';

const _bgrManifestJson = '''
{
  "name": "adaface_ir101_webface12m", "family": "adaface", "file": "x.tflite",
  "input": {"width":112,"height":112,"color":"BGR","layout":"NHWC",
            "normalize":{"mean":[127.5,127.5,127.5],"std":[127.5,127.5,127.5]}},
  "output": {"dim":512,"l2_normalize":true},
  "license": {"tier":"research","redistributable":false,"source":"x","note":"y"}
}
''';

const _noNormalizeManifestJson = '''
{
  "name": "raw_512", "family": "mobilefacenet", "file": "x.tflite",
  "input": {"width":112,"height":112,"color":"RGB","layout":"NHWC",
            "normalize":{"mean":[127.5,127.5,127.5],"std":[127.5,127.5,127.5]}},
  "output": {"dim":512,"l2_normalize":false},
  "license": {"tier":"byom","redistributable":false,"source":"x","note":"y"}
}
''';

void main() {
  group('adapterForFamily', () {
    test('arcface/adaface/mobilefacenet all resolve to ArcfaceAdapter', () {
      expect(adapterForFamily('arcface'), isA<ArcfaceAdapter>());
      expect(adapterForFamily('adaface'), isA<ArcfaceAdapter>());
      expect(adapterForFamily('mobilefacenet'), isA<ArcfaceAdapter>());
    });

    test('facenet resolves to FacenetAdapter', () {
      expect(adapterForFamily('facenet'), isA<FacenetAdapter>());
    });

    test('unknown family throws', () {
      expect(() => adapterForFamily('unknown_family'), throwsArgumentError);
    });
  });

  group('ArcfaceAdapter', () {
    const adapter = ArcfaceAdapter();

    test('preprocess produces a [1, height, width, 3] nested tensor', () {
      final manifest = ModelManifest.fromJsonString(_arcfaceManifestJson);
      final face = AlignedFace(rgbBytes: Uint8List(112 * 112 * 3), size: 112);

      final tensor = adapter.preprocess(face, manifest) as List;
      expect(tensor.length, 1);
      expect((tensor[0] as List).length, 112);
      expect((tensor[0][0] as List).length, 112);
      expect((tensor[0][0][0] as List).length, 3);
    });

    test('postprocess L2-normalises when manifest requests it', () {
      final manifest = ModelManifest.fromJsonString(_arcfaceManifestJson);
      final raw = List<double>.generate(512, (i) => i == 0 ? 3.0 : 0.0)
        ..[1] = 4.0; // vector (3,4,0,0,...) → norm 5
      final embedding = adapter.postprocess(raw, manifest);

      expect(embedding.dim, 512);
      expect(embedding.vector[0], closeTo(0.6, 1e-6));
      expect(embedding.vector[1], closeTo(0.8, 1e-6));
    });

    test('preprocess swaps R/B channels when manifest.input.color is BGR', () {
      final manifest = ModelManifest.fromJsonString(_bgrManifestJson);
      // Single pixel, repeated: R=10, G=20, B=30.
      final bytes = Uint8List(112 * 112 * 3);
      for (int i = 0; i < bytes.length; i += 3) {
        bytes[i] = 10;
        bytes[i + 1] = 20;
        bytes[i + 2] = 30;
      }
      final face = AlignedFace(rgbBytes: bytes, size: 112);

      final tensor = adapter.preprocess(face, manifest) as List;
      final pixel = (tensor[0][0][0] as List).cast<double>();
      // BGR output order: channel0=B(30), channel1=G(20), channel2=R(10).
      expect(pixel[0], closeTo((30 - 127.5) / 127.5, 1e-6));
      expect(pixel[1], closeTo((20 - 127.5) / 127.5, 1e-6));
      expect(pixel[2], closeTo((10 - 127.5) / 127.5, 1e-6));
    });

    test('postprocess leaves vector raw when l2_normalize is false', () {
      final manifest = ModelManifest.fromJsonString(_noNormalizeManifestJson);
      final raw = List<double>.generate(512, (i) => i == 0 ? 3.0 : 0.0)
        ..[1] = 4.0;
      final embedding = adapter.postprocess(raw, manifest);

      expect(embedding.vector[0], closeTo(3.0, 1e-6));
      expect(embedding.vector[1], closeTo(4.0, 1e-6));
    });
  });
}
