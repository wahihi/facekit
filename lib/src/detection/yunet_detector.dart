// YuNet face detector — implements FaceDetector.
//
// Per-pass pipeline (see _detectOnFrame), run against whatever FaceImage
// it's given:
//   FaceImage (any size, RGB888)
//     → resize to 160×160
//     → raw 0-255 pixels, BGR channel order, no normalisation
//       (verified empirically: the exported ONNX graph's first op consumes
//       'input' directly with no Sub/Div/Mul preprocessing node, and
//       matching cv2.FaceDetectorYN's own reference output required no
//       normalisation either — see tool/model_verification/)
//     → TFLite inference (12 output tensors: cls/obj/bbox/kps × 3 strides)
//     → decodeYunet (pixel-space [0, 160]) → NMS
//     → scaleYunetDetections back onto the input frame's own width/height
//     → List<DetectedFace> (pixel coords, matching the input frame)
//
// detect() runs the above up to three times, all in service of the same
// goal: never squeeze more of the frame down to 160×160 than necessary,
// because that starves a small-in-frame face of input pixels, letting
// ordinary frame-to-frame sensor noise swing its decoded landmarks (and
// detection confidence) by a large fraction of the face's own size — see
// doc/KR/postmortem/2026-08-13-landmark-jitter-frame-fill.md.
//   1. First pass: a centred square crop of the frame (_detectFirstPass),
//      not the whole frame — needs no prior detection to know where to
//      look, so it carries no risk of a bad crop from imprecise data.
//      Falls back to the whole frame if nothing's found there (e.g. an
//      off-centre face).
//   2. If the best face found is still a small fraction of the frame (see
//      needsYunetRefinement), a second pass re-detects in a tight,
//      native-resolution crop centred on *that* face's own bbox (see
//      yunetRefinementCropRegion) and uses the refined result instead.
//      Unlike step 1, this one does depend on a prior (pass-1) detection to
//      know where to crop, which historically has made it fall back more
//      often than not on-device when that bbox was itself imprecise — see
//      doc/KR/postmortem/2026-08-14-camera-orientation-and-auto-zoom.md.
//
// Source:
//   YuNet (libfacedetection / OpenCV Zoo, MIT licence):
//   https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

import 'package:flutter/foundation.dart' show debugPrint, kReleaseMode;

import '../core/contracts.dart';
import '../core/debug_flags.dart';
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
    final passOne = _detectFirstPass(image);
    if (passOne.isEmpty) return passOne;

    var bestIndex = 0;
    for (var i = 1; i < passOne.length; i++) {
      if (passOne[i].score > passOne[bestIndex].score) bestIndex = i;
    }
    final best = passOne[bestIndex];
    final faceFraction = image.width > 0 ? best.boundingBox.width / image.width : 0.0;
    if (!needsYunetRefinement(best.boundingBox, image.width)) {
      if (kFacekitVerboseDebug) {
        debugPrint('[YuNetDetector] DEBUG REFINE skip: face is '
            '${(faceFraction * 100).toStringAsFixed(1)}% of frame width '
            '(threshold ${(kRefineFaceFraction * 100).toStringAsFixed(0)}%)');
      }
      return passOne;
    }

    if (kFacekitVerboseDebug) {
      debugPrint('[YuNetDetector] DEBUG REFINE triggered: face is '
          '${(faceFraction * 100).toStringAsFixed(1)}% of frame width '
          '(threshold ${(kRefineFaceFraction * 100).toStringAsFixed(0)}%), '
          're-detecting in a native-res crop');
    }
    final refined = _refine(image, best);
    if (refined == null) {
      if (kFacekitVerboseDebug) {
        // Region logged so a fallback can be checked against where the
        // face actually was (e.g. a follow-up manual crop of the raw frame)
        // — every real-device trigger logged so far has fallen back, and
        // it's not yet confirmed whether that's because the crop is
        // missing the face (imprecise pass-1 bbox) or something else.
        final region = yunetRefinementCropRegion(best.boundingBox, image.width, image.height);
        debugPrint('[YuNetDetector] DEBUG REFINE fallback: no face found in '
            'the refinement crop, keeping the whole-frame result '
            '(pass-1 bbox=${_fmtRect(best.boundingBox)}, '
            'crop region=${_fmtRect(region)})');
      }
      return passOne;
    }
    if (kFacekitVerboseDebug) {
      final refinedFraction = image.width > 0 ? refined.boundingBox.width / image.width : 0.0;
      debugPrint('[YuNetDetector] DEBUG REFINE applied: score '
          '${best.score.toStringAsFixed(3)} -> ${refined.score.toStringAsFixed(3)}, '
          'face fraction ${(faceFraction * 100).toStringAsFixed(1)}% -> '
          '${(refinedFraction * 100).toStringAsFixed(1)}%');
    }

    return [
      for (var i = 0; i < passOne.length; i++) i == bestIndex ? refined : passOne[i],
    ];
  }

  /// First-pass detection: tries a centred square crop of [image] first
  /// (see [centerSquareCropRegion]) rather than squeezing the whole frame —
  /// cheaper on pixels-per-face, and (unlike [_refine]'s bbox-centred crop)
  /// needs no prior detection to know where to look, so it can't inherit
  /// refinement's "imprecise bbox → crop misses the face" failure mode.
  /// Falls back to the whole, uncropped frame if the centred crop finds
  /// nothing — e.g. a face off-centre enough to fall outside it.
  List<DetectedFace> _detectFirstPass(FaceImage image) {
    final region = centerSquareCropRegion(image.width, image.height);
    final cropped = cropFaceImage(image, region);
    final inCrop = _detectOnFrame(cropped);
    if (inCrop.isNotEmpty) {
      if (kFacekitVerboseDebug) {
        debugPrint('[YuNetDetector] DEBUG CENTERCROP hit: found in centred '
            '${_fmtRect(region)} crop');
      }
      return [for (final f in inCrop) mapYunetRefinedFace(f, region)];
    }
    if (kFacekitVerboseDebug) {
      debugPrint('[YuNetDetector] DEBUG CENTERCROP fallback: nothing found '
          'in the centred ${_fmtRect(region)} crop, trying the whole frame');
    }
    return _detectOnFrame(image);
  }

  /// Re-detects within a tight, native-resolution crop around [face]'s box
  /// (see [yunetRefinementCropRegion]) — the second pass of [detect]'s
  /// whole-frame-then-refine strategy. Returns null (caller falls back to
  /// the whole-frame result) if nothing is found in the crop.
  DetectedFace? _refine(FaceImage image, DetectedFace face) {
    final region = yunetRefinementCropRegion(face.boundingBox, image.width, image.height);
    final cropped = cropFaceImage(image, region);
    final passTwo = _detectOnFrame(cropped);
    if (passTwo.isEmpty) return null;

    var bestIndex = 0;
    for (var i = 1; i < passTwo.length; i++) {
      if (passTwo[i].score > passTwo[bestIndex].score) bestIndex = i;
    }
    return mapYunetRefinedFace(passTwo[bestIndex], region);
  }

  /// Runs one whole-model pass (resize → tensor → inference → decode) on
  /// [frame], returning detections scaled into [frame]'s own pixel space.
  /// Used for both the initial whole-frame pass and the refinement pass —
  /// the only difference between them is what [frame] contains.
  List<DetectedFace> _detectOnFrame(FaceImage frame) {
    // 1. Resize to inputSize×inputSize
    final resized = resizeNearest(frame, _inputSize, _inputSize);

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

    return scaleYunetDetections(pixelSpace, _inputSize, frame.width, frame.height);
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

String _fmtRect(Rect r) =>
    '(${r.left.toStringAsFixed(0)},${r.top.toStringAsFixed(0)})-'
    '(${r.right.toStringAsFixed(0)},${r.bottom.toStringAsFixed(0)})';
