// FaceNet-family embedder adapter.
//
// FaceNet's input convention differs from ArcFace/AdaFace/MobileFaceNet:
// instead of a fixed (pixel - mean) / std using manifest-supplied constants,
// each image is "prewhitened" using its OWN per-image mean/std. This is also
// why FaceNet needs a different input size (160x160, not 112x112) — see
// design spec §8 "(입력크기 예외)".
//
// Source: prewhiten formula per the davidsandberg/facenet reference
// preprocessing convention (MIT) — formula/shape only, no model weights or
// proprietary code.

import 'dart:math' as math;
import 'dart:typed_data';

import '../../core/math.dart';
import '../../core/models.dart';
import '../../inference/model_manifest.dart';
import 'embedder_adapter.dart';

class FacenetAdapter implements EmbedderAdapter {
  const FacenetAdapter();

  @override
  Object preprocess(AlignedFace face, ModelManifest manifest) {
    final width = manifest.input.width;
    final height = manifest.input.height;
    final rgb = face.rgbBytes;
    final pixelCount = width * height * 3;

    double sum = 0.0;
    for (int i = 0; i < pixelCount; i++) {
      sum += rgb[i];
    }
    final mean = sum / pixelCount;

    double sumSq = 0.0;
    for (int i = 0; i < pixelCount; i++) {
      final d = rgb[i] - mean;
      sumSq += d * d;
    }
    final std = math.sqrt(sumSq / pixelCount);
    final stdAdj = math.max(std, 1.0 / math.sqrt(pixelCount));

    return List.generate(
      1,
      (_) => List.generate(
        height,
        (row) => List.generate(
          width,
          (col) {
            final idx = (row * width + col) * 3;
            return [
              (rgb[idx] - mean) / stdAdj,
              (rgb[idx + 1] - mean) / stdAdj,
              (rgb[idx + 2] - mean) / stdAdj,
            ];
          },
        ),
      ),
    );
  }

  @override
  Embedding postprocess(List<double> raw, ModelManifest manifest) {
    final vector = Float32List.fromList(raw);
    return Embedding(manifest.output.l2Normalize ? l2Normalize(vector) : vector);
  }
}
