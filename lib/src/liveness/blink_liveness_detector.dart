// Free-tier basic liveness: blink detection via Eye Aspect Ratio (EAR).
//
// Honest scope: this defeats a static printed/displayed photo (a flat photo
// never produces an EAR dip), but it is NOT a substitute for Pro-tier
// anti-spoofing — it does not defend against a video replay of a blinking
// face, nor a photo with cut-out eye holes held in front of a real mouth/
// eyes. That stronger, multi-signal liveness stays a Pro-tier concern behind
// the same [LivenessDetector] contract.
//
// The EAR threshold (~0.2) is the published Soukupová & Čech (2016)
// rule-of-thumb starting point, not yet re-derived against our own captured
// data — unlike the ArcFace/AdaFace cosine thresholds (doc/KR/adaface_verification.md),
// there's no real blink-video sample available to measure an EER-style value
// in this environment. Revisit once on-device testing is possible.
//
// Eye landmark indices below are positions into the 478-point MediaPipe face
// mesh topology (public, widely-cited — e.g.
// https://github.com/Pushtogithub23/Eye-Blink-Detection-using-MediaPipe-and-OpenCV),
// ordered [outerCorner, upperOuter, upperInner, innerCorner, lowerInner,
// lowerOuter] to match core/math.dart's eyeAspectRatio() point order.

import '../core/contracts.dart';
import '../core/math.dart';
import '../core/models.dart';

const List<int> _rightEyeIndices = [33, 159, 158, 133, 153, 145];
const List<int> _leftEyeIndices = [362, 380, 374, 263, 386, 385];

/// A closed→open transition longer than this is treated as "eyes were
/// closed" (e.g. squinting, looking down) rather than a genuine blink, and
/// is not counted. Real blinks are typically 100–400ms.
const int _maxBlinkDurationMs = 1000;

class BlinkLivenessDetector implements LivenessDetector {
  final double earThreshold;
  final int observationWindowMs;

  bool _eyeClosed = false;
  int? _closedSinceMs;
  int _blinkCount = 0;
  int? _windowStartMs;

  /// [earThreshold]: average two-eye EAR below this counts as "closed".
  /// [observationWindowMs]: how long to wait for at least one blink before
  /// reporting [LivenessState.pending] with a reason, instead of staying
  /// silently pending forever.
  BlinkLivenessDetector({
    this.earThreshold = 0.2,
    this.observationWindowMs = 4000,
  });

  @override
  LivenessResult update(FaceLandmarks landmarks, int timestampMs) {
    _windowStartMs ??= timestampMs;

    final ear = _averageEar(landmarks);
    final closedNow = ear < earThreshold;

    if (closedNow && !_eyeClosed) {
      _eyeClosed = true;
      _closedSinceMs = timestampMs;
    } else if (!closedNow && _eyeClosed) {
      _eyeClosed = false;
      final closedSince = _closedSinceMs;
      _closedSinceMs = null;
      if (closedSince != null && (timestampMs - closedSince) <= _maxBlinkDurationMs) {
        _blinkCount++;
      }
    }

    if (_blinkCount >= 1) {
      return const LivenessResult(state: LivenessState.passed);
    }

    if (timestampMs - _windowStartMs! > observationWindowMs) {
      // No blink observed in the window — stay `pending` rather than jump to
      // `failed`, so a legitimate user who simply hasn't blinked yet isn't
      // rejected outright. Callers wanting a hard cutoff should apply their
      // own grace period on top of a sustained `pending`.
      return const LivenessResult(
        state: LivenessState.pending,
        failReason: 'no blink observed within the observation window',
      );
    }

    return const LivenessResult(state: LivenessState.pending);
  }

  double _averageEar(FaceLandmarks landmarks) {
    final right = eyeAspectRatio(_pointsAt(landmarks, _rightEyeIndices));
    final left = eyeAspectRatio(_pointsAt(landmarks, _leftEyeIndices));
    return (right + left) / 2.0;
  }

  List<Point> _pointsAt(FaceLandmarks landmarks, List<int> indices) {
    return [
      for (final i in indices) Point(landmarks.points[i].x, landmarks.points[i].y),
    ];
  }

  @override
  void reset() {
    _eyeClosed = false;
    _closedSinceMs = null;
    _blinkCount = 0;
    _windowStartMs = null;
  }
}
