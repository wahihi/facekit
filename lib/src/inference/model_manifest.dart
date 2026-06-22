// Parses and validates manifest.json files that describe TFLite models.
// The manifest is the single source of truth for a model's I/O contract.
// Source: design spec §8 — clean-room implementation.

import 'dart:convert';

class NormalizeSpec {
  final List<double> mean;
  final List<double> std;

  const NormalizeSpec({required this.mean, required this.std});

  factory NormalizeSpec.fromJson(Map<String, dynamic> j) => NormalizeSpec(
        mean: List<double>.from((j['mean'] as List).map((e) => (e as num).toDouble())),
        std:  List<double>.from((j['std']  as List).map((e) => (e as num).toDouble())),
      );
}

class InputSpec {
  final int width;
  final int height;
  final String color;   // 'RGB' | 'BGR'
  final String layout;  // 'NHWC' | 'NCHW'
  final NormalizeSpec normalize;

  const InputSpec({
    required this.width,
    required this.height,
    required this.color,
    required this.layout,
    required this.normalize,
  });

  factory InputSpec.fromJson(Map<String, dynamic> j) => InputSpec(
        width:     j['width'] as int,
        height:    j['height'] as int,
        color:     j['color'] as String,
        layout:    j['layout'] as String,
        normalize: NormalizeSpec.fromJson(j['normalize'] as Map<String, dynamic>),
      );
}

class OutputSpec {
  /// For embedder models: embedding dimension.
  final int? dim;
  final bool l2Normalize;

  /// For detector models: index of the regressors (box+landmarks) tensor.
  final int? regressorsIndex;
  /// For detector models: index of the classificators (scores) tensor.
  final int? classificatorsIndex;

  const OutputSpec({
    this.dim,
    this.l2Normalize = false,
    this.regressorsIndex,
    this.classificatorsIndex,
  });

  factory OutputSpec.fromJson(Map<String, dynamic> j) => OutputSpec(
        dim:                   j['dim'] as int?,
        l2Normalize:           (j['l2_normalize'] as bool?) ?? false,
        regressorsIndex:       j['regressors_index'] as int?,
        classificatorsIndex:   j['classificators_index'] as int?,
      );
}

class DetectionSpec {
  final double scoreThreshold;
  final double iouThreshold;
  final int maxFaces;

  const DetectionSpec({
    required this.scoreThreshold,
    required this.iouThreshold,
    required this.maxFaces,
  });

  factory DetectionSpec.fromJson(Map<String, dynamic> j) => DetectionSpec(
        scoreThreshold: (j['score_threshold'] as num).toDouble(),
        iouThreshold:   (j['iou_threshold']   as num).toDouble(),
        maxFaces:        j['max_faces'] as int,
      );
}

class AlignmentSpec {
  final String type;       // e.g. 'five_point_affine'
  final String reference;  // e.g. 'arcface_112'

  const AlignmentSpec({required this.type, required this.reference});

  factory AlignmentSpec.fromJson(Map<String, dynamic> j) => AlignmentSpec(
        type:      j['type'] as String,
        reference: j['reference'] as String,
      );
}

class MatchingSpec {
  final String metric;
  final double threshold;
  final String? thresholdNote;

  const MatchingSpec({
    required this.metric,
    required this.threshold,
    this.thresholdNote,
  });

  factory MatchingSpec.fromJson(Map<String, dynamic> j) => MatchingSpec(
        metric:        j['metric'] as String,
        threshold:     (j['threshold'] as num).toDouble(),
        thresholdNote: j['threshold_note'] as String?,
      );
}

enum LicenseTier { bundled, research, byom, licensed }

class LicenseSpec {
  final LicenseTier tier;
  final bool redistributable;
  final String source;
  final String note;

  const LicenseSpec({
    required this.tier,
    required this.redistributable,
    required this.source,
    required this.note,
  });

  factory LicenseSpec.fromJson(Map<String, dynamic> j) {
    final tierStr = j['tier'] as String;
    final tier = switch (tierStr) {
      'bundled'  => LicenseTier.bundled,
      'research' => LicenseTier.research,
      'byom'     => LicenseTier.byom,
      'licensed' => LicenseTier.licensed,
      _          => throw ArgumentError('Unknown license tier: $tierStr'),
    };
    return LicenseSpec(
      tier:            tier,
      redistributable: j['redistributable'] as bool,
      source:          j['source'] as String,
      note:            j['note'] as String,
    );
  }
}

class ModelManifest {
  final String name;
  final String family; // 'blazeface' | 'arcface' | 'adaface' | 'facenet' | 'mobilefacenet'
  final String file;   // .tflite filename relative to the manifest directory
  final InputSpec input;
  final OutputSpec output;
  final DetectionSpec? detection;
  final AlignmentSpec? alignment;
  final MatchingSpec? matching;
  final LicenseSpec license;

  const ModelManifest({
    required this.name,
    required this.family,
    required this.file,
    required this.input,
    required this.output,
    required this.license,
    this.detection,
    this.alignment,
    this.matching,
  });

  factory ModelManifest.fromJson(Map<String, dynamic> j) => ModelManifest(
        name:      j['name'] as String,
        family:    j['family'] as String,
        file:      j['file'] as String,
        input:     InputSpec.fromJson(j['input'] as Map<String, dynamic>),
        output:    OutputSpec.fromJson(j['output'] as Map<String, dynamic>),
        detection: j['detection'] != null
            ? DetectionSpec.fromJson(j['detection'] as Map<String, dynamic>)
            : null,
        alignment: j['alignment'] != null
            ? AlignmentSpec.fromJson(j['alignment'] as Map<String, dynamic>)
            : null,
        matching: j['matching'] != null
            ? MatchingSpec.fromJson(j['matching'] as Map<String, dynamic>)
            : null,
        license: LicenseSpec.fromJson(j['license'] as Map<String, dynamic>),
      );

  /// Parses a manifest from a JSON string.
  factory ModelManifest.fromJsonString(String jsonStr) =>
      ModelManifest.fromJson(jsonDecode(jsonStr) as Map<String, dynamic>);

  /// Throws if the manifest is in an inconsistent state.
  void validate() {
    if (file.isEmpty) throw StateError('manifest "$name": file must not be empty');
    if (license.tier == LicenseTier.research && license.redistributable) {
      throw StateError('manifest "$name": research-tier models must not be redistributable');
    }
  }

  /// Throws if this model must not be loaded under the current build mode.
  ///
  /// Per CLAUDE.md license policy: non-redistributable models (Demo/research
  /// tier) are for evaluation only and must be blocked in release builds.
  /// [isReleaseBuild] is passed in by the caller (e.g. `kReleaseMode`) so this
  /// file stays pure Dart with no Flutter dependency.
  void assertLoadable({required bool isReleaseBuild}) {
    if (isReleaseBuild && !license.redistributable) {
      throw StateError(
        'manifest "$name": license.redistributable=false (tier=${license.tier.name}, '
        'source=${license.source}) — this model must not be loaded in a release build',
      );
    }
  }
}
