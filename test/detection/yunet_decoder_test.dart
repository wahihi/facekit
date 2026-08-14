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

  group('needsYunetRefinement', () {
    test('bbox well below the frame-fraction threshold needs refinement', () {
      // 100px face in a 1000px-wide frame = 10% -- below the 20% default.
      final box = Rect(left: 450, top: 450, right: 550, bottom: 550);
      expect(needsYunetRefinement(box, 1000), isTrue);
    });

    test('bbox well above the frame-fraction threshold does not need refinement', () {
      // 900px face in a 1000px-wide frame = 90% -- well above 20%.
      final box = Rect(left: 50, top: 50, right: 950, bottom: 950);
      expect(needsYunetRefinement(box, 1000), isFalse);
    });

    test('bbox exactly at the threshold does not need refinement (strict <)', () {
      final box = Rect(left: 0, top: 0, right: 200, bottom: 200); // 200/1000 = 0.20
      expect(needsYunetRefinement(box, 1000, threshold: 0.20), isFalse);
    });

    test('non-positive frame width is treated as never needing refinement', () {
      final box = Rect(left: 0, top: 0, right: 10, bottom: 10);
      expect(needsYunetRefinement(box, 0), isFalse);
    });

    test('a custom threshold is honoured', () {
      final box = Rect(left: 0, top: 0, right: 300, bottom: 300); // 30% of frame
      expect(needsYunetRefinement(box, 1000, threshold: 0.20), isFalse);
      expect(needsYunetRefinement(box, 1000, threshold: 0.50), isTrue);
    });
  });

  group('yunetRefinementCropRegion', () {
    test('centres a square crop at 2x the bbox\'s larger side away from image edges', () {
      // 100x60 bbox centred at (500, 500), well inside a 1000x1000 image ->
      // no clamping. side = 100 * (1 + 2*0.5) = 200 (larger side wins).
      final box = Rect(left: 450, top: 470, right: 550, bottom: 530);
      final region = yunetRefinementCropRegion(box, 1000, 1000);

      expect(region.width, closeTo(200, 1e-9));
      expect(region.height, closeTo(200, 1e-9));
      expect(region.centerX, closeTo(box.centerX, 1e-9));
      expect(region.centerY, closeTo(box.centerY, 1e-9));
    });

    test('a taller-than-wide bbox sizes the crop off its height', () {
      final box = Rect(left: 480, top: 400, right: 520, bottom: 600); // 40w x 200h
      final region = yunetRefinementCropRegion(box, 1000, 1000);
      expect(region.width, closeTo(400, 1e-9)); // 200 * (1 + 2*0.5)
      expect(region.height, closeTo(400, 1e-9));
    });

    test('clamps to the image bounds instead of extending past them', () {
      // bbox right at the top-left corner of a small image -- an unclamped
      // crop would go negative on both axes.
      final box = Rect(left: 0, top: 0, right: 40, bottom: 40);
      final region = yunetRefinementCropRegion(box, 200, 200);

      expect(region.left, greaterThanOrEqualTo(0.0));
      expect(region.top, greaterThanOrEqualTo(0.0));
      expect(region.right, lessThanOrEqualTo(200.0));
      expect(region.bottom, lessThanOrEqualTo(200.0));
      // Still a valid non-empty region.
      expect(region.width, greaterThan(0.0));
      expect(region.height, greaterThan(0.0));
    });

    test('a custom margin fraction changes the crop size', () {
      final box = Rect(left: 400, top: 400, right: 600, bottom: 600); // 200x200
      final tight = yunetRefinementCropRegion(box, 2000, 2000, marginFraction: 0.0);
      final wide = yunetRefinementCropRegion(box, 2000, 2000, marginFraction: 1.0);
      expect(tight.width, closeTo(200, 1e-9)); // 200 * (1 + 0)
      expect(wide.width, closeTo(600, 1e-9)); // 200 * (1 + 2)
    });
  });

  group('mapYunetRefinedFace', () {
    test('offsets bbox and landmarks by the crop region\'s top-left, keeps score', () {
      final region = Rect(left: 300, top: 400, right: 700, bottom: 800);
      final local = DetectedFace(
        boundingBox: Rect(left: 10, top: 20, right: 110, bottom: 220),
        landmarks: const [Point(50, 60), Point(70, 80)],
        score: 0.87,
      );

      final mapped = mapYunetRefinedFace(local, region);

      expect(mapped.boundingBox.left, closeTo(310, 1e-9));
      expect(mapped.boundingBox.top, closeTo(420, 1e-9));
      expect(mapped.boundingBox.right, closeTo(410, 1e-9));
      expect(mapped.boundingBox.bottom, closeTo(620, 1e-9));
      expect(mapped.landmarks[0].x, closeTo(350, 1e-9));
      expect(mapped.landmarks[0].y, closeTo(460, 1e-9));
      expect(mapped.landmarks[1].x, closeTo(370, 1e-9));
      expect(mapped.landmarks[1].y, closeTo(480, 1e-9));
      expect(mapped.score, closeTo(0.87, 1e-9));
    });
  });

  group('centerSquareCropRegion', () {
    test('a landscape-shaped frame crops to a centred square sized off the shorter side', () {
      final region = centerSquareCropRegion(720, 480);
      expect(region.width, closeTo(480, 1e-9));
      expect(region.height, closeTo(480, 1e-9));
      expect(region.left, closeTo(120, 1e-9)); // (720-480)/2
      expect(region.top, closeTo(0, 1e-9));
      expect(region.right, closeTo(600, 1e-9));
      expect(region.bottom, closeTo(480, 1e-9));
    });

    test('a portrait-shaped frame crops off the top/bottom instead', () {
      final region = centerSquareCropRegion(480, 720);
      expect(region.width, closeTo(480, 1e-9));
      expect(region.height, closeTo(480, 1e-9));
      expect(region.left, closeTo(0, 1e-9));
      expect(region.top, closeTo(120, 1e-9)); // (720-480)/2
    });

    test('an already-square frame crops to itself', () {
      final region = centerSquareCropRegion(300, 300);
      expect(region.left, closeTo(0, 1e-9));
      expect(region.top, closeTo(0, 1e-9));
      expect(region.right, closeTo(300, 1e-9));
      expect(region.bottom, closeTo(300, 1e-9));
    });
  });
}
