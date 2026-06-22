// 1:N cosine similarity matcher.
// Pure function — stateless, no I/O.
// Source: standard cosine similarity + nearest-neighbour search.

import '../core/contracts.dart';
import '../core/models.dart';
import '../core/math.dart';
import '../inference/model_manifest.dart';

class CosineMatcher implements FaceMatcher {
  /// Minimum cosine similarity to accept a match.
  /// Default 0.40 is a PoC placeholder — tune per model using ROC on your dataset.
  final double threshold;

  const CosineMatcher({this.threshold = 0.40});

  /// Builds a matcher using the threshold declared in [manifest].matching,
  /// so each embedding model's own (yet-to-be-ROC-tuned) threshold travels
  /// with its manifest instead of being re-specified at call sites.
  factory CosineMatcher.fromManifest(ModelManifest manifest) {
    final matching = manifest.matching;
    if (matching == null) {
      throw ArgumentError('manifest "${manifest.name}" has no matching spec');
    }
    return CosineMatcher(threshold: matching.threshold);
  }

  @override
  MatchResult match(Embedding probe, List<Enrollment> gallery) {
    if (gallery.isEmpty) {
      return const MatchResult(matchedId: null, similarity: 0.0, accepted: false);
    }

    double bestSim = -1.0;
    String? bestId;

    for (final enrollment in gallery) {
      final sim = cosineSimilarity(probe.vector, enrollment.embedding.vector);
      if (sim > bestSim) {
        bestSim = sim;
        bestId = enrollment.id;
      }
    }

    final accepted = bestSim >= threshold;
    return MatchResult(
      matchedId: accepted ? bestId : null,
      similarity: bestSim,
      accepted: accepted,
    );
  }
}
