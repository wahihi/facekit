import 'package:test/test.dart';
import 'package:facekit/src/core/contracts.dart';
import 'package:facekit/src/core/models.dart';
import 'package:facekit/src/liveness/blink_liveness_detector.dart';

const _rightEyeIndices = [33, 159, 158, 133, 153, 145];
const _leftEyeIndices = [362, 380, 374, 263, 386, 385];

/// Builds a 478-point FaceLandmarks with both eyes shaped to the given
/// open/closed state (everything else is a placeholder zero point — unused
/// by BlinkLivenessDetector, which only reads the eye indices).
FaceLandmarks _landmarks({required bool eyesOpen}) {
  final points = List.generate(478, (_) => const Point3D(0, 0, 0));
  // Point order: [outerCorner, upperOuter, upperInner, innerCorner, lowerInner, lowerOuter].
  final shape = eyesOpen
      ? const [
          Point3D(0, 0, 0), Point3D(2, -2, 0), Point3D(8, -2, 0),
          Point3D(10, 0, 0), Point3D(8, 2, 0), Point3D(2, 2, 0),
        ] // EAR ≈ 0.4
      : const [
          Point3D(0, 0, 0), Point3D(2, -0.2, 0), Point3D(8, -0.2, 0),
          Point3D(10, 0, 0), Point3D(8, 0.2, 0), Point3D(2, 0.2, 0),
        ]; // EAR ≈ 0.04
  for (var i = 0; i < 6; i++) {
    points[_rightEyeIndices[i]] = shape[i];
    points[_leftEyeIndices[i]] = shape[i];
  }
  return FaceLandmarks(points: points);
}

void main() {
  group('BlinkLivenessDetector', () {
    test('eyes always open within the window → stays pending', () {
      final detector = BlinkLivenessDetector(observationWindowMs: 4000);
      LivenessResult result = detector.update(_landmarks(eyesOpen: true), 0);
      expect(result.state, LivenessState.pending);

      result = detector.update(_landmarks(eyesOpen: true), 2000);
      expect(result.state, LivenessState.pending);
    });

    test('eyes always open past the window → pending with a reason', () {
      final detector = BlinkLivenessDetector(observationWindowMs: 4000);
      detector.update(_landmarks(eyesOpen: true), 0);
      final result = detector.update(_landmarks(eyesOpen: true), 5000);
      expect(result.state, LivenessState.pending);
      expect(result.failReason, isNotNull);
    });

    test('a brief closure-then-reopen within the window counts as a blink → passed', () {
      final detector = BlinkLivenessDetector(observationWindowMs: 4000);
      detector.update(_landmarks(eyesOpen: true), 0);
      detector.update(_landmarks(eyesOpen: false), 200); // eyes close
      final result = detector.update(_landmarks(eyesOpen: true), 350); // reopen ~150ms later
      expect(result.state, LivenessState.passed);
    });

    test('a too-slow closure (> max blink duration) is not counted as a blink', () {
      final detector = BlinkLivenessDetector(observationWindowMs: 4000);
      detector.update(_landmarks(eyesOpen: true), 0);
      detector.update(_landmarks(eyesOpen: false), 200); // eyes close
      // Reopen 1500ms later — too slow to be a genuine blink (default max 1000ms).
      final result = detector.update(_landmarks(eyesOpen: true), 1700);
      expect(result.state, LivenessState.pending);
    });

    test('reset() clears blink count and window state', () {
      final detector = BlinkLivenessDetector(observationWindowMs: 4000);
      detector.update(_landmarks(eyesOpen: true), 0);
      detector.update(_landmarks(eyesOpen: false), 200);
      var result = detector.update(_landmarks(eyesOpen: true), 350);
      expect(result.state, LivenessState.passed);

      detector.reset();
      result = detector.update(_landmarks(eyesOpen: true), 10000);
      expect(result.state, LivenessState.pending);
    });
  });
}
