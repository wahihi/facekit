// YuNet face detector — implements FaceDetector.
//
// Pipeline:
//   FaceImage (any size, RGB888)
//     → resize to 160×160
//     → raw 0-255 pixels, BGR channel order, no normalisation
//       (verified empirically: the exported ONNX graph's first op consumes
//       'input' directly with no Sub/Div/Mul preprocessing node, and
//       matching cv2.FaceDetectorYN's own reference output required no
//       normalisation either — see tool/model_verification/)
//     → TFLite inference (12 output tensors: cls/obj/bbox/kps × 3 strides)
//     → decodeYunet (pixel-space [0, 160]) → NMS
//     → scaleYunetDetections back onto the input image's own width/height
//     → List<DetectedFace> (pixel coords, matching the input FaceImage)
//
// Source:
//   YuNet (libfacedetection / OpenCV Zoo, MIT licence):
//   https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

import 'package:flutter/foundation.dart' show kReleaseMode;

import '../core/contracts.dart';
import '../core/models.dart';
import '../image/image_converter.dart';
import '../inference/model_manifest.dart';
import '../inference/tflite_runner.dart';
import 'blazeface_decoder.dart' show reshapeTensor2D;
import 'yunet_decoder.dart';

const List<int> _kStrides = [8, 16, 32];

/// Which physical TFLite output tensor index holds each role, per stride.
/// Resolved once at load time from tensor shapes rather than hardcoded,
/// because the onnx→tflite conversion (onnx2tf) does not preserve the
/// exported graph's output order — verified empirically: the 12 outputs
/// come back from the interpreter in a scrambled order that doesn't match
/// their alphabetical/semantic names. Shape alone disambiguates them:
/// grid size (400/100/25 for a 160 input) picks the stride, channel count
/// (4=bbox, 10=kps, 1=score) picks the role. A stride has *two* channel-1
/// score tensors (cls and obj); which one is "cls" vs "obj" doesn't matter
/// since decodeYunet only ever needs their symmetric product
/// sqrt(cls*obj).
class _StrideIo {
  final int bboxIndex;
  final int kpsIndex;
  final int scoreIndexA;
  final int scoreIndexB;
  const _StrideIo({
    required this.bboxIndex,
    required this.kpsIndex,
    required this.scoreIndexA,
    required this.scoreIndexB,
  });
}

Map<int, _StrideIo> _resolveStrideIo(TfliteRunner runner, int inputSize) {
  final bboxByStride = <int, int>{};
  final kpsByStride = <int, int>{};
  final scoresByStride = <int, List<int>>{};

  for (int i = 0; i < runner.outputCount; i++) {
    final shape = runner.outputShape(i); // [1, N, C]
    if (shape.length != 3) {
      throw StateError('YuNetDetector: output $i has unexpected shape $shape');
    }
    final n = shape[1];
    final c = shape[2];

    int? stride;
    for (final s in _kStrides) {
      final grid = inputSize ~/ s;
      if (n == grid * grid) {
        stride = s;
        break;
      }
    }
    if (stride == null) {
      throw StateError(
        'YuNetDetector: output $i has $n cells, which matches no stride '
        'grid for a ${inputSize}x$inputSize input (expected one of '
        '${_kStrides.map((s) => (inputSize ~/ s) * (inputSize ~/ s)).toList()})',
      );
    }

    switch (c) {
      case 4:
        bboxByStride[stride] = i;
      case 10:
        kpsByStride[stride] = i;
      case 1:
        (scoresByStride[stride] ??= []).add(i);
      default:
        throw StateError('YuNetDetector: output $i has unexpected channel count $c');
    }
  }

  final result = <int, _StrideIo>{};
  for (final s in _kStrides) {
    final bbox = bboxByStride[s];
    final kps = kpsByStride[s];
    final scores = scoresByStride[s];
    if (bbox == null || kps == null || scores == null || scores.length != 2) {
      throw StateError(
        'YuNetDetector: could not resolve outputs for stride $s '
        '(bbox=$bbox kps=$kps scores=$scores) — model does not match the '
        'expected YuNet 12-output layout',
      );
    }
    result[s] = _StrideIo(
      bboxIndex: bbox,
      kpsIndex: kps,
      scoreIndexA: scores[0],
      scoreIndexB: scores[1],
    );
  }
  return result;
}

class YuNetDetector implements FaceDetector {
  final TfliteRunner _runner;
  final DetectionSpec _spec;
  final int _inputSize;
  final Map<int, _StrideIo> _strideIo;

  YuNetDetector._({
    required TfliteRunner runner,
    required DetectionSpec spec,
    required int inputSize,
    required Map<int, _StrideIo> strideIo,
  })  : _runner = runner,
        _spec = spec,
        _inputSize = inputSize,
        _strideIo = strideIo;

  /// Loads from a Flutter asset path.
  /// [manifestAssetPath] — e.g. 'assets/models/yunet_160/manifest.json'
  /// The .tflite file is resolved relative to the manifest directory.
  static Future<YuNetDetector> fromAsset({
    required String tfliteAssetPath,
    required ModelManifest manifest,
    bool useNnApi = false,
  }) async {
    manifest.validate();
    manifest.assertLoadable(isReleaseBuild: kReleaseMode);
    final runner = await TfliteRunner.fromAsset(tfliteAssetPath, useNnApi: useNnApi);
    return _build(runner: runner, manifest: manifest);
  }

  /// Loads from absolute file-system paths (useful in tests).
  static Future<YuNetDetector> fromFile({
    required String tflitePath,
    required ModelManifest manifest,
    bool useNnApi = false,
  }) async {
    manifest.validate();
    manifest.assertLoadable(isReleaseBuild: kReleaseMode);
    final runner = await TfliteRunner.fromFile(tflitePath, useNnApi: useNnApi);
    return _build(runner: runner, manifest: manifest);
  }

  static YuNetDetector _build({
    required TfliteRunner runner,
    required ModelManifest manifest,
  }) {
    final inputSize = manifest.input.width;
    if (manifest.input.height != inputSize) {
      throw StateError('YuNetDetector requires a square input, got ${manifest.input.width}x${manifest.input.height}');
    }
    return YuNetDetector._(
      runner: runner,
      spec: manifest.detection!,
      inputSize: inputSize,
      strideIo: _resolveStrideIo(runner, inputSize),
    );
  }

  @override
  Future<List<DetectedFace>> detect(FaceImage image) async {
    // 1. Resize to inputSize×inputSize
    final resized = resizeNearest(image, _inputSize, _inputSize);

    // 2. Build input tensor [1, size, size, 3], raw 0-255 BGR (no normalisation)
    final input = prepareInputTensor(
      rgbBytes: resized.rgbBytes,
      width: _inputSize,
      height: _inputSize,
      mean: const [0.0, 0.0, 0.0],
      std: const [1.0, 1.0, 1.0],
      swapToBgr: true,
    );

    // 3. Allocate all 12 outputs and run
    final outputs = <int, Object>{
      for (int i = 0; i < _runner.outputCount; i++) i: zeroTensor(_runner.outputShape(i)),
    };
    _runner.runForMultipleOutputs(input, outputs);

    // 4. Reshape each stride's tensors and decode
    List<List<double>> reshape(int index, int channels) {
      final grid = _inputSize ~/ _strideForIndex(index);
      return reshapeTensor2D(outputs[index], grid * grid, channels);
    }

    final io8 = _strideIo[8]!;
    final io16 = _strideIo[16]!;
    final io32 = _strideIo[32]!;

    final pixelSpace = decodeYunet(
      cls8:  reshape(io8.scoreIndexA, 1),
      obj8:  reshape(io8.scoreIndexB, 1),
      bbox8: reshape(io8.bboxIndex, 4),
      kps8:  reshape(io8.kpsIndex, 10),
      cls16:  reshape(io16.scoreIndexA, 1),
      obj16:  reshape(io16.scoreIndexB, 1),
      bbox16: reshape(io16.bboxIndex, 4),
      kps16:  reshape(io16.kpsIndex, 10),
      cls32:  reshape(io32.scoreIndexA, 1),
      obj32:  reshape(io32.scoreIndexB, 1),
      bbox32: reshape(io32.bboxIndex, 4),
      kps32:  reshape(io32.kpsIndex, 10),
      inputSize: _inputSize,
      scoreThreshold: _spec.scoreThreshold,
      iouThreshold: _spec.iouThreshold,
      maxFaces: _spec.maxFaces,
    );

    return scaleYunetDetections(pixelSpace, _inputSize, image.width, image.height);
  }

  int _strideForIndex(int index) {
    for (final entry in _strideIo.entries) {
      final io = entry.value;
      if (io.bboxIndex == index || io.kpsIndex == index ||
          io.scoreIndexA == index || io.scoreIndexB == index) {
        return entry.key;
      }
    }
    throw StateError('YuNetDetector: output index $index is not part of any resolved stride');
  }

  void dispose() => _runner.close();
}
