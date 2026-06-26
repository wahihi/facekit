// Run with: flutter test test/inference/tflite_runner_test.dart
// (flutter_test, not plain `dart test` — tflite_runner.dart depends on
// tflite_flutter → dart:ui, same constraint as the embedding adapter tests.)
import 'package:flutter_test/flutter_test.dart';
import 'package:facekit/src/inference/tflite_runner.dart';

void main() {
  group('zeroTensor', () {
    test('rank-1 shape produces a flat zero-filled list', () {
      final t = zeroTensor([4]) as List;
      expect(t, [0.0, 0.0, 0.0, 0.0]);
    });

    test('rank-2 shape produces a nested zero-filled list, e.g. AdaFace\'s [1,1] norm output', () {
      final t = zeroTensor([1, 1]) as List;
      expect(t.length, 1);
      expect((t[0] as List), [0.0]);
    });

    test('rank-3 shape nests correctly, e.g. a [1, 896, 16] detector regressors tensor', () {
      final t = zeroTensor([1, 896, 16]) as List;
      expect(t.length, 1);
      expect((t[0] as List).length, 896);
      expect(((t[0] as List)[0] as List).length, 16);
      expect(((t[0] as List)[0] as List)[0], 0.0);
    });
  });
}
