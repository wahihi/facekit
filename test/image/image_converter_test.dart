import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:facekit/src/core/models.dart';
import 'package:facekit/src/image/image_converter.dart';

void main() {
  group('bgraToFaceImage', () {
    test('single red pixel', () {
      // BGRA: B=0, G=0, R=255, A=255
      final bgra = Uint8List.fromList([0, 0, 255, 255]);
      final img = bgraToFaceImage(bgra: bgra, width: 1, height: 1);
      expect(img.width, 1);
      expect(img.height, 1);
      expect(img.rgbBytes[0], 255); // R
      expect(img.rgbBytes[1], 0);   // G
      expect(img.rgbBytes[2], 0);   // B
    });

    test('2×1 image channel order', () {
      // pixel0: B=10,G=20,R=30,A=255  pixel1: B=40,G=50,R=60,A=255
      final bgra = Uint8List.fromList([10, 20, 30, 255, 40, 50, 60, 255]);
      final img = bgraToFaceImage(bgra: bgra, width: 2, height: 1);
      expect(img.rgbBytes, [30, 20, 10, 60, 50, 40]);
    });

    test('output byte count is width*height*3', () {
      final bgra = Uint8List(4 * 4 * 4); // 4×4 BGRA
      final img = bgraToFaceImage(bgra: bgra, width: 4, height: 4);
      expect(img.rgbBytes.length, 4 * 4 * 3);
    });
  });

  group('yuv420ToFaceImage', () {
    test('pure Y plane (grey) → grey RGB', () {
      const w = 4, h = 4;
      // Y=128 → near-grey output (slight offset from BT.601)
      final y = Uint8List(w * h)..fillRange(0, w * h, 128);
      final u = Uint8List((w ~/ 2) * (h ~/ 2))..fillRange(0, (w ~/ 2) * (h ~/ 2), 128);
      final v = Uint8List((w ~/ 2) * (h ~/ 2))..fillRange(0, (w ~/ 2) * (h ~/ 2), 128);

      final img = yuv420ToFaceImage(
        yPlane: y, uPlane: u, vPlane: v,
        width: w, height: h,
        yRowStride: w, uvRowStride: w ~/ 2, uvPixelStride: 1,
      );

      expect(img.width, w);
      expect(img.height, h);
      expect(img.rgbBytes.length, w * h * 3);

      // With Y=128, U=128, V=128 the BT.601 formula gives approximately equal R=G=B
      final r = img.rgbBytes[0], g = img.rgbBytes[1], b = img.rgbBytes[2];
      expect((r - g).abs(), lessThan(5));
      expect((g - b).abs(), lessThan(5));
    });

    test('output dimensions match', () {
      const w = 8, h = 6;
      final y = Uint8List(w * h);
      final u = Uint8List((w ~/ 2) * (h ~/ 2));
      final v = Uint8List((w ~/ 2) * (h ~/ 2));
      final img = yuv420ToFaceImage(
        yPlane: y, uPlane: u, vPlane: v,
        width: w, height: h,
        yRowStride: w, uvRowStride: w ~/ 2, uvPixelStride: 1,
      );
      expect(img.width, w);
      expect(img.height, h);
      expect(img.rgbBytes.length, w * h * 3);
    });
  });

  group('resizeNearest', () {
    test('2×2 → 1×1 picks top-left pixel', () {
      final rgb = Uint8List.fromList([
        255, 0, 0,  // (0,0) red
        0, 255, 0,  // (1,0) green
        0, 0, 255,  // (0,1) blue
        255, 255, 0, // (1,1) yellow
      ]);
      final src = FaceImage(rgbBytes: rgb, width: 2, height: 2);
      final dst = resizeNearest(src, 1, 1);
      expect(dst.rgbBytes[0], 255); // R of top-left
      expect(dst.rgbBytes[1], 0);
      expect(dst.rgbBytes[2], 0);
    });

    test('output size correct', () {
      final src = FaceImage(rgbBytes: Uint8List(10 * 10 * 3), width: 10, height: 10);
      final dst = resizeNearest(src, 5, 5);
      expect(dst.width, 5);
      expect(dst.height, 5);
      expect(dst.rgbBytes.length, 5 * 5 * 3);
    });
  });
}
