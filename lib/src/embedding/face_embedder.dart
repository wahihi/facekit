// Loads a TFLite embedding model per its manifest, selects the adapter
// matching `manifest.family`, and implements the FaceEmbedder contract.
// Source: design spec §4, §7, §8.

import 'package:flutter/foundation.dart' show kReleaseMode;

import '../core/contracts.dart';
import '../core/models.dart';
import '../inference/model_manifest.dart';
import '../inference/tflite_runner.dart';
import 'adapters/arcface_adapter.dart';
import 'adapters/embedder_adapter.dart';
import 'adapters/facenet_adapter.dart';

/// Maps a manifest's `family` field to the adapter that knows how to
/// pre/post-process tensors for that model family.
///
/// arcface/adaface/mobilefacenet share the same 112x112 input convention and
/// are served by [ArcfaceAdapter]. facenet uses a different 160x160 +
/// per-image "prewhiten" convention and is served by [FacenetAdapter].
EmbedderAdapter adapterForFamily(String family) {
  switch (family) {
    case 'arcface':
    case 'adaface':
    case 'mobilefacenet':
      return const ArcfaceAdapter();
    case 'facenet':
      return const FacenetAdapter();
    default:
      throw ArgumentError('No embedder adapter registered for family "$family"');
  }
}

class TfliteFaceEmbedder implements FaceEmbedder {
  final TfliteRunner _runner;
  final ModelManifest _manifest;
  final EmbedderAdapter _adapter;

  TfliteFaceEmbedder._({
    required TfliteRunner runner,
    required ModelManifest manifest,
    required EmbedderAdapter adapter,
  })  : _runner = runner,
        _manifest = manifest,
        _adapter = adapter;

  /// Loads from a Flutter asset path.
  static Future<TfliteFaceEmbedder> fromAsset({
    required String tfliteAssetPath,
    required ModelManifest manifest,
  }) async {
    manifest.validate();
    manifest.assertLoadable(isReleaseBuild: kReleaseMode);
    final runner = await TfliteRunner.fromAsset(tfliteAssetPath);
    return TfliteFaceEmbedder._(
      runner: runner,
      manifest: manifest,
      adapter: adapterForFamily(manifest.family),
    );
  }

  /// Loads from an absolute file-system path (useful in tests and for
  /// BYOM/Demo models that are deliberately not bundled as Flutter assets).
  static Future<TfliteFaceEmbedder> fromFile({
    required String tflitePath,
    required ModelManifest manifest,
  }) async {
    manifest.validate();
    manifest.assertLoadable(isReleaseBuild: kReleaseMode);
    final runner = await TfliteRunner.fromFile(tflitePath);
    return TfliteFaceEmbedder._(
      runner: runner,
      manifest: manifest,
      adapter: adapterForFamily(manifest.family),
    );
  }

  @override
  Future<Embedding> embed(AlignedFace face) async {
    final input = _adapter.preprocess(face, _manifest);
    final dim = _manifest.output.dim!;
    // Output tensor 0 is always the embedding. Some families (e.g. AdaFace,
    // whose graph also exposes a pre-normalisation "norm" scalar) have
    // additional output tensors that this SDK doesn't use — those still need
    // a correctly-shaped buffer or tflite_flutter's runForMultipleInputs
    // throws on the unfilled map entry.
    final outputs = <int, Object>{0: List.generate(1, (_) => List.filled(dim, 0.0))};
    for (var i = 1; i < _runner.outputCount; i++) {
      outputs[i] = zeroTensor(_runner.outputShape(i));
    }
    _runner.runForMultipleOutputs(input, outputs);
    final embedding = (outputs[0] as List)[0] as List<double>;
    return _adapter.postprocess(embedding, _manifest);
  }

  void dispose() => _runner.close();
}
