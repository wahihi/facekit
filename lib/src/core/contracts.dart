// Abstract interfaces for each pipeline stage.
// Concrete implementations live outside core/ and depend on these types only.
// Source: design spec §7.

import 'models.dart';

abstract class FaceDetector {
  /// Returns all detected faces in [image], ordered by score descending.
  Future<List<DetectedFace>> detect(FaceImage image);
}

abstract class FaceAligner {
  /// Crops and aligns [face] from [image] into a square patch.
  AlignedFace align(FaceImage image, DetectedFace face);
}

abstract class FaceEmbedder {
  /// Runs inference and returns an L2-normalised embedding.
  Future<Embedding> embed(AlignedFace face);
}

abstract class FaceMatcher {
  /// Scores [probe] against every [Enrollment] in [gallery].
  /// Pure function — no I/O, no state mutation.
  MatchResult match(Embedding probe, List<Enrollment> gallery);
}

abstract class FaceLandmarker {
  /// Returns dense landmarks for [face] (detected within [image]), or null
  /// if the model can't find a face in the cropped region.
  Future<FaceLandmarks?> detectLandmarks(FaceImage image, DetectedFace face);
}

// ── Liveness ───────────────────────────────────────────────────────────────
// The interface is shared across tiers; Free ships a basic blink-only
// implementation (BlinkLivenessDetector), Pro adds stronger multi-signal
// checks (e.g. replay-attack resistance) behind the same contract.

abstract class LivenessDetector {
  /// Accumulates [landmarks] observations (captured at [timestampMs]) and
  /// returns the current liveness verdict. [timestampMs] is supplied by the
  /// caller (e.g. `DateTime.now().millisecondsSinceEpoch`) rather than read
  /// internally, so implementations stay deterministic/testable.
  LivenessResult update(FaceLandmarks landmarks, int timestampMs);
  void reset();
}

class LivenessResult {
  final LivenessState state;
  final String? failReason;

  const LivenessResult({required this.state, this.failReason});
}

enum LivenessState { pending, passed, failed }

abstract class TemplateStore {
  Future<void> save(Enrollment enrollment);
  Future<List<Enrollment>> loadAll();
  Future<void> delete(String id);
}
