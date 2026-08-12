// Unit tests for the pure decodeYunet() function.
//
// The "real detection regression" group below embeds raw pre-decode tensor
// values captured directly from the actual yunet_2026may.onnx model run on
// a real photo (tool/model_verification/, cross-checked byte-for-byte
// against cv2.FaceDetectorYN's own reference output — see
// doc/KR/postmortem/2026-08-12-yunet-landmark-order.md) and asserts
// decodeYunet() reproduces the same score/bbox/keypoints. This pins the
// exact decode formula (grid offset, exp-based bbox, sqrt(cls*obj) score)
// against known-good numbers, not just internal self-consistency.

import 'package:test/test.dart';
import 'package:facekit/src/core/models.dart';
import 'package:facekit/src/detection/yunet_decoder.dart';

const _inputSize = 160;

/// Builds a [grid*grid]-length zero-filled tensor with [channels] columns
/// per row, used as a stand-in for one stride's cls/obj/bbox/kps output.
List<List<double>> _zeros(int grid, int channels) =>
    List.generate(grid * grid, (_) => List.filled(channels, 0.0));

List<DetectedFace> _decodeWithStride8({
  required List<List<double>> cls8,
  required List<List<double>> obj8,
  required List<List<double>> bbox8,
  required List<List<double>> kps8,
  double scoreThreshold = 0.5,
}) =>
    decodeYunet(
      cls8: cls8, obj8: obj8, bbox8: bbox8, kps8: kps8,
      cls16: _zeros(10, 1), obj16: _zeros(10, 1), bbox16: _zeros(10, 4), kps16: _zeros(10, 10),
      cls32: _zeros(5, 1), obj32: _zeros(5, 1), bbox32: _zeros(5, 4), kps32: _zeros(5, 10),
      inputSize: _inputSize,
      scoreThreshold: scoreThreshold,
      iouThreshold: 0.3,
      maxFaces: 5,
    );

void main() {
  group('decodeYunet — real detection regression', () {
    // Captured from yunet_2026may.onnx on tool/model_verification/me_1.jpg,
    // resized to 160×160: stride 8, grid row 12, col 11 (flat index 251 of
    // the 20×20=400-cell stride-8 grid) — the single highest-scoring cell.
    final cls8 = _zeros(20, 1);
    final obj8 = _zeros(20, 1);
    final bbox8 = _zeros(20, 4);
    final kps8 = _zeros(20, 10);
    const flatIdx = 251;
    cls8[flatIdx][0] = 0.87964743;
    obj8[flatIdx][0] = 0.9974295;
    bbox8[flatIdx] = [-0.6264444589614868, 0.06728613376617432, 2.134089231491089, 2.263803482055664];
    kps8[flatIdx] = [
      -2.788578987121582, -0.9985904097557068,
      0.7519518136978149, -1.053585410118103,
      -1.209341049194336, 0.880354106426239,
      -2.413376569747925, 2.4346084594726562,
      0.6207417249679565, 2.151059627532959,
    ];

    List<DetectedFace> decode({double scoreThreshold = 0.5}) => _decodeWithStride8(
          cls8: cls8, obj8: obj8, bbox8: bbox8, kps8: kps8,
          scoreThreshold: scoreThreshold,
        );

    test('score matches sqrt(cls*obj) from the real model output', () {
      final faces = decode();
      expect(faces, hasLength(1));
      expect(faces[0].score, closeTo(0.936689, 1e-4));
    });

    test('bbox matches the manually-verified decode (cross-checked against cv2.FaceDetectorYN)', () {
      final faces = decode();
      final box = faces[0].boundingBox;
      expect(box.left,   closeTo(49.191, 1e-2));
      expect(box.top,    closeTo(58.060, 1e-2));
      expect(box.right,  closeTo(49.191 + 67.595, 1e-2));
      expect(box.bottom, closeTo(58.060 + 76.957, 1e-2));
    });

    test('keypoints match the manually-verified decode (cross-checked against cv2.FaceDetectorYN)', () {
      final faces = decode();
      final lm = faces[0].landmarks;
      expect(lm, hasLength(5));
      const expected = [
        [65.69137,  88.01128],
        [94.01561,  87.57132],
        [78.32527, 103.04283],
        [68.69299, 115.47687],
        [92.96593, 113.20848],
      ];
      for (int i = 0; i < 5; i++) {
        expect(lm[i].x, closeTo(expected[i][0], 1e-2));
        expect(lm[i].y, closeTo(expected[i][1], 1e-2));
      }
    });

    test('a stricter score threshold filters the detection out entirely', () {
      expect(decode(scoreThreshold: 0.99), isEmpty);
    });
  });

  group('decodeYunet — synthetic sanity checks', () {
    List<List<double>> onesAtOrigin(int grid, int channels) {
      final t = _zeros(grid, channels);
      if (channels == 1) t[0][0] = 1.0;
      return t;
    }

    test('an all-zero raw cell at grid (0,0) decodes to a box centred at the origin', () {
      // bbox raw = [0,0,0,0] -> cx=(0+0)*stride=0, cy=0, w=exp(0)*stride=stride, h=stride
      // kps raw = 0 for all 5 points -> (0+0)*stride = 0 for every point
      // left/top would be cx-w/2 = 0-8/2 = -4, but decodeYunet clamps every
      // coordinate to [0, inputSize] (a real detection can't extend past
      // the model's own input frame), so the negative half is clamped to 0.
      final faces = _decodeWithStride8(
        cls8: onesAtOrigin(20, 1),
        obj8: onesAtOrigin(20, 1),
        bbox8: _zeros(20, 4),
        kps8: _zeros(20, 10),
      );

      expect(faces, hasLength(1));
      expect(faces[0].score, closeTo(1.0, 1e-9));
      expect(faces[0].boundingBox.left,   closeTo(0.0, 1e-9)); // clamp(cx - w/2, 0, inputSize)
      expect(faces[0].boundingBox.top,    closeTo(0.0, 1e-9));
      expect(faces[0].boundingBox.right,  closeTo(4.0, 1e-9));
      expect(faces[0].boundingBox.bottom, closeTo(4.0, 1e-9));
      for (final p in faces[0].landmarks) {
        expect(p.x, closeTo(0.0, 1e-9));
        expect(p.y, closeTo(0.0, 1e-9));
      }
    });

    test('scaleYunetDetections rescales pixel-space output onto an image size', () {
      final faces = _decodeWithStride8(
        cls8: onesAtOrigin(20, 1),
        obj8: onesAtOrigin(20, 1),
        bbox8: _zeros(20, 4),
        kps8: _zeros(20, 10),
      );
      // inputSize=160 -> scale to a 320x480 image is 2x horizontally, 3x vertically.
      final scaled = scaleYunetDetections(faces, _inputSize, 320, 480);
      expect(scaled[0].boundingBox.left, closeTo(faces[0].boundingBox.left * 2, 1e-9));
      expect(scaled[0].boundingBox.top,  closeTo(faces[0].boundingBox.top * 3, 1e-9));
    });

    test('score below threshold produces no detections', () {
      final faces = _decodeWithStride8(
        cls8: _zeros(20, 1),
        obj8: _zeros(20, 1),
        bbox8: _zeros(20, 4),
        kps8: _zeros(20, 10),
      );
      expect(faces, isEmpty);
    });
  });
}
