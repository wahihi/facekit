// BlazeFace face detector — implements FaceDetector.
//
// Pipeline:
//   FaceImage (any size, RGB888)
//     → resize to 128×128
//     → normalise to [-1, 1]
//     → TFLite inference
//     → decode anchors + NMS (normalised coords [0,1])
//     → denormalise back to the input image's own width/height
//     → List<DetectedFace> (pixel coords, matching the input FaceImage)
//
// Source:
//   BlazeFace paper: https://arxiv.org/abs/1907.05047
//   MediaPipe face detection module (Apache 2.0):
//   https://github.com/google/mediapipe/tree/master/mediapipe/modules/face_detection

import 'package:flutter/foundation.dart' show kReleaseMode;

import '../core/contracts.dart';
import '../core/models.dart';
import '../image/image_converter.dart';
import '../inference/model_manifest.dart';
import '../inference/tflite_runner.dart';
import 'blazeface_anchors.dart';
import 'blazeface_decoder.dart';

class BlazeFaceDetector implements FaceDetector {
  final TfliteRunner _runner;
  final DetectionSpec _spec;
  final List<Anchor> _anchors;

  BlazeFaceDetector._({
    required TfliteRunner runner,
    required DetectionSpec spec,
    required List<Anchor> anchors,
  })  : _runner = runner,
        _spec = spec,
        _anchors = anchors;

  /// Loads from a Flutter asset path.
  /// [manifestAssetPath] — e.g. 'assets/models/blazeface_short/manifest.json'
  /// The .tflite file is resolved relative to the manifest directory.
  static Future<BlazeFaceDetector> fromAsset({
    required String tfliteAssetPath,
    required ModelManifest manifest,
    bool useNnApi = false,
  }) async {
    manifest.validate();
    manifest.assertLoadable(isReleaseBuild: kReleaseMode);
    final runner = await TfliteRunner.fromAsset(tfliteAssetPath, useNnApi: useNnApi);
    return BlazeFaceDetector._(
      runner:  runner,
      spec:    manifest.detection!,
      anchors: blazeFaceShortAnchors,
    );
  }

  /// Loads from absolute file-system paths (useful in tests).
  static Future<BlazeFaceDetector> fromFile({
    required String tflitePath,
    required ModelManifest manifest,
    bool useNnApi = false,
  }) async {
    manifest.validate();
    manifest.assertLoadable(isReleaseBuild: kReleaseMode);
    final runner = await TfliteRunner.fromFile(tflitePath, useNnApi: useNnApi);
    return BlazeFaceDetector._(
      runner:  runner,
      spec:    manifest.detection!,
      anchors: blazeFaceShortAnchors,
    );
  }

  @override
  Future<List<DetectedFace>> detect(FaceImage image) async {
    // 1. Resize to 128×128
    final resized = resizeNearest(image, 128, 128);

    // 2. Build input tensor [1, 128, 128, 3] normalised to [-1, 1]
    final input = prepareInputTensor(
      rgbBytes: resized.rgbBytes,
      width:    128,
      height:   128,
      mean:     [127.5, 127.5, 127.5],
      std:      [127.5, 127.5, 127.5],
    );

    // 3. Allocate outputs
    //   regressors    [1, 896, 16]
    //   classificators[1, 896,  1]
    final rawRegressors    = List.generate(1, (_) => List.generate(896, (_) => List.filled(16, 0.0)));
    final rawClassificators = List.generate(1, (_) => List.generate(896, (_) => List.filled(1,  0.0)));

    _runner.runForMultipleOutputs(input, {
      0: rawRegressors,
      1: rawClassificators,
    });

    // 4. Reshape and decode
    final regressors    = reshapeTensor2D(rawRegressors,    896, 16);
    final classificators = reshapeTensor2D(rawClassificators, 896, 1);

    final normalized = decodeBlazeFace(
      regressors:     regressors,
      classificators: classificators,
      anchors:        _anchors,
      scoreThreshold: _spec.scoreThreshold,
      iouThreshold:   _spec.iouThreshold,
      maxFaces:       _spec.maxFaces,
    );

    // decodeBlazeFace returns [0,1]-normalised coordinates relative to the
    // 128×128 model input; convert to pixel coordinates of the *original*
    // image so AffineAligner (which expects pixel-space landmarks) works.
    return denormalizeDetections(normalized, image.width, image.height);
  }

  void dispose() => _runner.close();
}
