import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:facekit/src/core/math.dart';
import 'package:facekit/src/core/models.dart';
import 'package:facekit/src/inference/model_manifest.dart';
import 'package:facekit/src/matching/cosine_matcher.dart';

const _arcfaceManifestJson = '''
{
  "name": "arcface_buffalo_l", "family": "arcface", "file": "w600k_r50.tflite",
  "input": {"width":112,"height":112,"color":"RGB","layout":"NHWC",
            "normalize":{"mean":[127.5,127.5,127.5],"std":[127.5,127.5,127.5]}},
  "output": {"dim":512,"l2_normalize":true},
  "matching": {"metric":"cosine","threshold":0.40,"threshold_note":"PoC"},
  "license": {"tier":"research","redistributable":false,"source":"x","note":"y"}
}
''';

Enrollment _enroll(String id, List<double> v) =>
    Enrollment(id: id, embedding: Embedding(l2Normalize(Float32List.fromList(v))));

void main() {
  final matcher = CosineMatcher(threshold: 0.40);

  group('CosineMatcher', () {
    test('exact match → accepted', () {
      final probe = Embedding(l2Normalize(Float32List.fromList([1.0, 0.0, 0.0])));
      final gallery = [_enroll('alice', [1.0, 0.0, 0.0])];
      final result = matcher.match(probe, gallery);
      expect(result.accepted, isTrue);
      expect(result.matchedId, 'alice');
      expect(result.similarity, closeTo(1.0, 1e-5));
    });

    test('orthogonal → rejected', () {
      final probe = Embedding(l2Normalize(Float32List.fromList([1.0, 0.0])));
      final gallery = [_enroll('bob', [0.0, 1.0])];
      final result = matcher.match(probe, gallery);
      expect(result.accepted, isFalse);
      expect(result.matchedId, isNull);
    });

    test('picks best match from multiple enrollments', () {
      final probe = Embedding(l2Normalize(Float32List.fromList([1.0, 0.1, 0.0])));
      final gallery = [
        _enroll('alice', [1.0, 0.0, 0.0]),
        _enroll('bob',   [0.0, 1.0, 0.0]),
      ];
      final result = matcher.match(probe, gallery);
      expect(result.matchedId, 'alice');
    });

    test('empty gallery → not accepted', () {
      final probe = Embedding(l2Normalize(Float32List.fromList([1.0, 0.0])));
      final result = matcher.match(probe, []);
      expect(result.accepted, isFalse);
      expect(result.similarity, 0.0);
    });

    test('custom threshold respected', () {
      final high = CosineMatcher(threshold: 0.99);
      // [1,1] vs [1,0] → cosine ≈ 0.707, well below 0.99
      final probe = Embedding(l2Normalize(Float32List.fromList([1.0, 1.0])));
      final gallery = [_enroll('alice', [1.0, 0.0])];
      final result = high.match(probe, gallery);
      expect(result.accepted, isFalse);
    });

    test('fromManifest reads threshold from manifest.matching', () {
      final manifest = ModelManifest.fromJsonString(_arcfaceManifestJson);
      final fromManifest = CosineMatcher.fromManifest(manifest);
      expect(fromManifest.threshold, 0.40);
    });

    test('fromManifest throws when manifest has no matching spec', () {
      const noMatchingJson = '''
      {
        "name": "x", "family": "arcface", "file": "x.tflite",
        "input": {"width":112,"height":112,"color":"RGB","layout":"NHWC",
                  "normalize":{"mean":[127.5,127.5,127.5],"std":[127.5,127.5,127.5]}},
        "output": {"dim":512,"l2_normalize":true},
        "license": {"tier":"research","redistributable":false,"source":"x","note":"y"}
      }''';
      final manifest = ModelManifest.fromJsonString(noMatchingJson);
      expect(() => CosineMatcher.fromManifest(manifest), throwsArgumentError);
    });
  });
}
