// Per-model-family pre/post-processing for embedding models.
// Lets a new embedding family be added by registering one adapter, without
// touching face_embedder.dart's loading/inference logic.
// Source: design spec §7 — clean-room adapter interface.

import '../../core/models.dart';
import '../../inference/model_manifest.dart';

abstract class EmbedderAdapter {
  /// Builds the TFLite input tensor for [face] per [manifest].input rules.
  /// Shape/type is whatever the underlying TfliteRunner expects (typically a
  /// nested NHWC List), so this returns Object rather than a fixed type.
  Object preprocess(AlignedFace face, ModelManifest manifest);

  /// Converts the raw flat model output into an [Embedding], applying
  /// L2-normalisation when [ModelManifest.output] requests it.
  Embedding postprocess(List<double> raw, ModelManifest manifest);
}
