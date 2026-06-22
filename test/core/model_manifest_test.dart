import 'package:test/test.dart';
// Direct import — avoids pulling in tflite_flutter → flutter → dart:ui
import 'package:facekit/src/inference/model_manifest.dart';

const _blazefaceJson = '''
{
  "name": "blazeface_short",
  "family": "blazeface",
  "file": "face_detection_short_range.tflite",
  "input": {
    "width": 128,
    "height": 128,
    "color": "RGB",
    "layout": "NHWC",
    "normalize": { "mean": [127.5, 127.5, 127.5], "std": [127.5, 127.5, 127.5] }
  },
  "output": {
    "regressors_index": 0,
    "classificators_index": 1
  },
  "detection": {
    "score_threshold": 0.5,
    "iou_threshold": 0.3,
    "max_faces": 10
  },
  "license": {
    "tier": "bundled",
    "redistributable": true,
    "source": "MediaPipe BlazeFace",
    "note": "Apache 2.0."
  }
}
''';

void main() {
  group('ModelManifest', () {
    late ModelManifest m;

    setUp(() => m = ModelManifest.fromJsonString(_blazefaceJson));

    test('parses name and family', () {
      expect(m.name,   'blazeface_short');
      expect(m.family, 'blazeface');
    });

    test('parses input spec', () {
      expect(m.input.width,  128);
      expect(m.input.height, 128);
      expect(m.input.color,  'RGB');
      expect(m.input.normalize.mean, [127.5, 127.5, 127.5]);
      expect(m.input.normalize.std,  [127.5, 127.5, 127.5]);
    });

    test('parses detection spec', () {
      expect(m.detection, isNotNull);
      expect(m.detection!.scoreThreshold, 0.5);
      expect(m.detection!.iouThreshold,   0.3);
      expect(m.detection!.maxFaces,       10);
    });

    test('parses license tier', () {
      expect(m.license.tier,            LicenseTier.bundled);
      expect(m.license.redistributable, isTrue);
    });

    test('validate passes for bundled redistributable model', () {
      expect(() => m.validate(), returnsNormally);
    });

    test('validate throws for research model marked redistributable', () {
      const badJson = '''
      {
        "name": "bad", "family": "arcface", "file": "x.tflite",
        "input": {"width":112,"height":112,"color":"RGB","layout":"NHWC",
                  "normalize":{"mean":[127.5,127.5,127.5],"std":[127.5,127.5,127.5]}},
        "output": {"dim":512,"l2_normalize":true},
        "license": {"tier":"research","redistributable":true,"source":"x","note":"y"}
      }''';
      final bad = ModelManifest.fromJsonString(badJson);
      expect(() => bad.validate(), throwsStateError);
    });

    test('assertLoadable allows a redistributable model in release builds', () {
      expect(() => m.assertLoadable(isReleaseBuild: true), returnsNormally);
    });

    test('assertLoadable allows a non-redistributable model in debug builds', () {
      const researchJson = '''
      {
        "name": "demo_arcface", "family": "arcface", "file": "x.tflite",
        "input": {"width":112,"height":112,"color":"RGB","layout":"NHWC",
                  "normalize":{"mean":[127.5,127.5,127.5],"std":[127.5,127.5,127.5]}},
        "output": {"dim":512,"l2_normalize":true},
        "license": {"tier":"research","redistributable":false,"source":"x","note":"y"}
      }''';
      final demo = ModelManifest.fromJsonString(researchJson);
      expect(() => demo.assertLoadable(isReleaseBuild: false), returnsNormally);
    });

    test('assertLoadable throws for a non-redistributable model in release builds', () {
      const researchJson = '''
      {
        "name": "demo_arcface", "family": "arcface", "file": "x.tflite",
        "input": {"width":112,"height":112,"color":"RGB","layout":"NHWC",
                  "normalize":{"mean":[127.5,127.5,127.5],"std":[127.5,127.5,127.5]}},
        "output": {"dim":512,"l2_normalize":true},
        "license": {"tier":"research","redistributable":false,"source":"x","note":"y"}
      }''';
      final demo = ModelManifest.fromJsonString(researchJson);
      expect(() => demo.assertLoadable(isReleaseBuild: true), throwsStateError);
    });
  });
}
