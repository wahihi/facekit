// MediaPipe Face Landmarker (478-point face mesh) — implements FaceLandmarker.
//
// Unlike BlazeFaceDetector (which scans the whole frame), this model expects
// an already-cropped, roughly-centred face patch — so detectLandmarks() first
// crops a square region (with margin) around the upstream DetectedFace's box,
// resizes it to the model's 256×256 input, runs inference, then maps the
// resulting points back from crop-local model space into the original
// FaceImage's pixel coordinates.
//
// Source:
//   MediaPipe Face Landmarker (Apache 2.0):
//   https://github.com/google/mediapipe
//   https://www.kaggle.com/models/mediapipe/face-landmarks-detection
//   .tflite extracted from the official face_landmarker.task bundle — see
//   assets/models/face_landmark_478/manifest.json for provenance.

import 'package:flutter/foundation.dart' show kReleaseMode;

import '../core/contracts.dart';
import '../core/models.dart';
import '../image/image_converter.dart';
import '../inference/model_manifest.dart';
import '../inference/tflite_runner.dart';

/// Fraction of the detected face box's larger side added as margin on each
/// side of the square crop fed to the landmark model — empirically, face
/// mesh models need slack around a tight detector box so the eyes/jaw
/// aren't clipped under expression or slight pose change. Not derived from
/// any measured dataset; a reasonable starting point to revisit once this
/// can be tested against real camera frames.
const double _cropMarginFraction = 0.25;

class MediaPipeFaceLandmarker implements FaceLandmarker {
  final TfliteRunner _runner;
  final int _inputWidth;
  final int _inputHeight;

  MediaPipeFaceLandmarker._({
    required TfliteRunner runner,
    required int inputWidth,
    required int inputHeight,
  })  : _runner = runner,
        _inputWidth = inputWidth,
        _inputHeight = inputHeight;

  /// Loads from a Flutter asset path.
  static Future<MediaPipeFaceLandmarker> fromAsset({
    required String tfliteAssetPath,
    required ModelManifest manifest,
  }) async {
    manifest.validate();
    manifest.assertLoadable(isReleaseBuild: kReleaseMode);
    final runner = await TfliteRunner.fromAsset(tfliteAssetPath);
    return MediaPipeFaceLandmarker._(
      runner: runner,
      inputWidth: manifest.input.width,
      inputHeight: manifest.input.height,
    );
  }

  /// Loads from an absolute file-system path (useful in tests).
  static Future<MediaPipeFaceLandmarker> fromFile({
    required String tflitePath,
    required ModelManifest manifest,
  }) async {
    manifest.validate();
    manifest.assertLoadable(isReleaseBuild: kReleaseMode);
    final runner = await TfliteRunner.fromFile(tflitePath);
    return MediaPipeFaceLandmarker._(
      runner: runner,
      inputWidth: manifest.input.width,
      inputHeight: manifest.input.height,
    );
  }

  @override
  Future<FaceLandmarks?> detectLandmarks(FaceImage image, DetectedFace face) async {
    final box = face.boundingBox;
    final side = (box.width > box.height ? box.width : box.height) * (1 + 2 * _cropMarginFraction);
    final cx = box.centerX;
    final cy = box.centerY;

    // Pre-clamp to the image bounds ourselves (mirroring cropFaceImage's own
    // clamping) so `region.left`/`region.top` below are guaranteed to match
    // the crop's *actual* top-left in image space — required for mapping
    // model-space landmarks back correctly when the requested crop runs off
    // the edge of the frame.
    final left = (cx - side / 2).floor().clamp(0, image.width - 1);
    final top = (cy - side / 2).floor().clamp(0, image.height - 1);
    final right = (cx + side / 2).ceil().clamp(left + 1, image.width);
    final bottom = (cy + side / 2).ceil().clamp(top + 1, image.height);
    final region = Rect(
      left: left.toDouble(),
      top: top.toDouble(),
      right: right.toDouble(),
      bottom: bottom.toDouble(),
    );

    final cropped = cropFaceImage(image, region);
    final resized = resizeNearest(cropped, _inputWidth, _inputHeight);

    final input = prepareInputTensor(
      rgbBytes: resized.rgbBytes,
      width: _inputWidth,
      height: _inputHeight,
      mean: [0.0, 0.0, 0.0],
      std: [255.0, 255.0, 255.0],
    );

    // Output 0 is the 478×3 landmark tensor; the model also exposes one or
    // more face-presence/score scalars (outputs 1+) that this SDK doesn't
    // use yet — filled with zero buffers so runForMultipleOutputs doesn't
    // throw on the unfilled slots (same pattern as TfliteFaceEmbedder.embed).
    final landmarkCount = 478;
    final outputs = <int, Object>{
      0: List.generate(1, (_) => List.generate(1, (_) => List.generate(1, (_) => List.filled(landmarkCount * 3, 0.0)))),
    };
    for (var i = 1; i < _runner.outputCount; i++) {
      outputs[i] = zeroTensor(_runner.outputShape(i));
    }
    _runner.runForMultipleOutputs(input, outputs);

    final raw = (((outputs[0] as List)[0] as List)[0] as List)[0] as List;

    // Model emits landmark x/y in the *crop's* 0..inputWidth/inputHeight
    // pixel space; map back into the original image's pixel coordinates.
    final scaleX = cropped.width / _inputWidth;
    final scaleY = cropped.height / _inputHeight;
    final points = <Point3D>[];
    for (var i = 0; i < landmarkCount; i++) {
      final mx = (raw[i * 3] as num).toDouble();
      final my = (raw[i * 3 + 1] as num).toDouble();
      final mz = (raw[i * 3 + 2] as num).toDouble();
      points.add(Point3D(
        region.left + mx * scaleX,
        region.top + my * scaleY,
        mz,
      ));
    }

    return FaceLandmarks(points: points);
  }

  void dispose() => _runner.close();
}
