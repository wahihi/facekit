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
