import 'dart:math' as math;
import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:facekit/src/core/math.dart';
import 'package:facekit/src/core/models.dart';

void main() {
  group('l2Normalize', () {
    test('unit vector stays unit', () {
      final v = Float32List.fromList([1.0, 0.0, 0.0]);
      final n = l2Normalize(v);
      expect(n[0], closeTo(1.0, 1e-6));
      expect(n[1], closeTo(0.0, 1e-6));
    });

    test('normalises arbitrary vector', () {
      final v = Float32List.fromList([3.0, 4.0]);
      final n = l2Normalize(v);
      final norm = math.sqrt(n[0] * n[0] + n[1] * n[1]);
      expect(norm, closeTo(1.0, 1e-6));
    });

    test('throws on empty vector', () {
      expect(() => l2Normalize(Float32List(0)), throwsArgumentError);
    });

    test('throws on zero vector', () {
      expect(() => l2Normalize(Float32List.fromList([0.0, 0.0])),
          throwsArgumentError);
    });
  });

  group('cosineSimilarity', () {
    test('identical vectors → 1.0', () {
      final v = l2Normalize(Float32List.fromList([1.0, 2.0, 3.0]));
      expect(cosineSimilarity(v, v), closeTo(1.0, 1e-6));
    });

    test('orthogonal vectors → 0.0', () {
      final a = l2Normalize(Float32List.fromList([1.0, 0.0]));
      final b = l2Normalize(Float32List.fromList([0.0, 1.0]));
      expect(cosineSimilarity(a, b), closeTo(0.0, 1e-6));
    });

    test('opposite vectors → -1.0', () {
      final a = l2Normalize(Float32List.fromList([1.0, 0.0]));
      final b = l2Normalize(Float32List.fromList([-1.0, 0.0]));
      expect(cosineSimilarity(a, b), closeTo(-1.0, 1e-6));
    });

    test('throws on length mismatch', () {
      final a = Float32List.fromList([1.0, 0.0]);
      final b = Float32List.fromList([1.0, 0.0, 0.0]);
      expect(() => cosineSimilarity(a, b), throwsArgumentError);
    });

    test('result clamped to [-1, 1]', () {
      // Force a near-overflow scenario with pre-normalised vectors
      final a = l2Normalize(Float32List.fromList([1.0, 1e-7]));
      final b = l2Normalize(Float32List.fromList([1.0, 1e-7]));
      final sim = cosineSimilarity(a, b);
      expect(sim, lessThanOrEqualTo(1.0));
      expect(sim, greaterThanOrEqualTo(-1.0));
    });
  });

  group('l2Norm', () {
    test('3-4-5 right triangle', () {
      final v = Float32List.fromList([3.0, 4.0]);
      expect(l2Norm(v), closeTo(5.0, 1e-6));
    });
  });

  group('eyeAspectRatio', () {
    // Point order: [outerCorner, upperOuter, upperInner, innerCorner, lowerInner, lowerOuter].
    test('open eye → higher ratio (~0.4)', () {
      final openEye = [
        const Point(0, 0),   // outerCorner
        const Point(2, -2),  // upperOuter
        const Point(8, -2),  // upperInner
        const Point(10, 0),  // innerCorner
        const Point(8, 2),   // lowerInner
        const Point(2, 2),   // lowerOuter
      ];
      expect(eyeAspectRatio(openEye), closeTo(0.4, 1e-6));
    });

    test('closed eye → much lower ratio (~0.04)', () {
      final closedEye = [
        const Point(0, 0),
        const Point(2, -0.2),
        const Point(8, -0.2),
        const Point(10, 0),
        const Point(8, 0.2),
        const Point(2, 0.2),
      ];
      expect(eyeAspectRatio(closedEye), closeTo(0.04, 1e-6));
    });

    test('closed eye ratio is well below open eye ratio', () {
      final openEye = [
        const Point(0, 0), const Point(2, -2), const Point(8, -2),
        const Point(10, 0), const Point(8, 2), const Point(2, 2),
      ];
      final closedEye = [
        const Point(0, 0), const Point(2, -0.2), const Point(8, -0.2),
        const Point(10, 0), const Point(8, 0.2), const Point(2, 0.2),
      ];
      expect(eyeAspectRatio(closedEye), lessThan(eyeAspectRatio(openEye)));
    });

    test('throws when given other than 6 points', () {
      expect(() => eyeAspectRatio([const Point(0, 0), const Point(1, 1)]),
          throwsArgumentError);
    });

    test('throws on degenerate input (corner points coincide)', () {
      final degenerate = List.generate(6, (_) => const Point(5, 5));
      expect(() => eyeAspectRatio(degenerate), throwsArgumentError);
    });
  });
}
