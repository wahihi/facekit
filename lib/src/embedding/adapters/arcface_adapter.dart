// ArcFace-family embedder adapter.
//
// Also covers AdaFace and MobileFaceNet: all three share the same 112x112
// RGB, (pixel-mean)/std input convention and a flat (non-spatial) embedding
// output, so one adapter implementation serves all three `family` values.
//
// Source: InsightFace model_zoo recognition preprocessing convention (MIT) —
// only the pixel-normalisation constants/shape convention are used here, no
// model weights or proprietary code.

import 'dart:typed_data';

import '../../core/math.dart';
import '../../core/models.dart';
import '../../inference/model_manifest.dart';
import '../../inference/tflite_runner.dart';
import 'embedder_adapter.dart';

class ArcfaceAdapter implements EmbedderAdapter {
  const ArcfaceAdapter();

  @override
  Object preprocess(AlignedFace face, ModelManifest manifest) {
    final input = manifest.input;
    return prepareInputTensor(
      rgbBytes: face.rgbBytes,
      width: input.width,
      height: input.height,
      mean: input.normalize.mean,
      std: input.normalize.std,
    );
  }

  @override
  Embedding postprocess(List<double> raw, ModelManifest manifest) {
    final vector = Float32List.fromList(raw);
    return Embedding(manifest.output.l2Normalize ? l2Normalize(vector) : vector);
  }
}
