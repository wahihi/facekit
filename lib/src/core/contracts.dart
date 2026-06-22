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

// ── Pro tier (defined here so Free code can reference the types) ─────────────

abstract class LivenessDetector {
  /// Accumulates [face] observations and returns the current liveness verdict.
  LivenessResult update(DetectedFace face);
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
