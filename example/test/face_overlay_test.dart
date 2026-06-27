// Pure-math tests for the camera-image → preview coordinate mapping used by
// FaceOverlayPainter. This is the one part of the box-overlay feature that
// can be verified without a real device/camera (see doc/KR/liveness.md and
// the face_overlay.dart header comment for what couldn't be).
import 'package:flutter/services.dart' show DeviceOrientation;
import 'package:flutter/foundation.dart' show TargetPlatform;
import 'package:flutter/material.dart' show Size, Offset;
import 'package:flutter_test/flutter_test.dart';

import 'package:facekit_example/face_overlay.dart';

void main() {
  group('quarterTurnsForOrientation', () {
    test('portraitUp is 0 turns on Android', () {
      expect(
        quarterTurnsForOrientation(DeviceOrientation.portraitUp, platform: TargetPlatform.android),
        0,
      );
    });

    test('landscapeRight is 1 turn on Android', () {
      expect(
        quarterTurnsForOrientation(DeviceOrientation.landscapeRight, platform: TargetPlatform.android),
        1,
      );
    });

    test('portraitDown is 2 turns, landscapeLeft is 3 turns on Android', () {
      expect(
        quarterTurnsForOrientation(DeviceOrientation.portraitDown, platform: TargetPlatform.android),
        2,
      );
      expect(
        quarterTurnsForOrientation(DeviceOrientation.landscapeLeft, platform: TargetPlatform.android),
        3,
      );
    });

    test('always 0 on iOS regardless of orientation — native preview is already upright', () {
      for (final o in DeviceOrientation.values) {
        expect(quarterTurnsForOrientation(o, platform: TargetPlatform.iOS), 0);
      }
    });
  });

  group('mapImagePointToPreview', () {
    // A 640x480 (landscape sensor) raw image, displayed in a 480x640
    // (portrait) preview box — the common phone-held-upright case.
    const imageSize = Size(640, 480);
    const previewBoxSize = Size(480, 640);

    test('0 turns: identity scale, no rotation', () {
      final p = mapImagePointToPreview(
        x: 100,
        y: 50,
        imageSize: const Size(480, 640),
        previewBoxSize: const Size(480, 640),
        quarterTurns: 0,
      );
      expect(p, const Offset(100, 50));
    });

    test('1 turn: raw image top-left corner maps to preview top-right corner', () {
      final p = mapImagePointToPreview(
        x: 0,
        y: 0,
        imageSize: imageSize,
        previewBoxSize: previewBoxSize,
        quarterTurns: 1,
      );
      expect(p, const Offset(480, 0));
    });

    test('1 turn: raw image bottom-right corner maps to preview bottom-left corner', () {
      final p = mapImagePointToPreview(
        x: 640,
        y: 480,
        imageSize: imageSize,
        previewBoxSize: previewBoxSize,
        quarterTurns: 1,
      );
      expect(p, const Offset(0, 640));
    });

    test('1 turn: raw image centre maps to preview centre', () {
      final p = mapImagePointToPreview(
        x: 320,
        y: 240,
        imageSize: imageSize,
        previewBoxSize: previewBoxSize,
        quarterTurns: 1,
      );
      expect(p.dx, closeTo(240, 1e-9));
      expect(p.dy, closeTo(320, 1e-9));
    });

    test('mirror flips horizontally after rotation/scale', () {
      final unmirrored = mapImagePointToPreview(
        x: 0,
        y: 0,
        imageSize: imageSize,
        previewBoxSize: previewBoxSize,
        quarterTurns: 1,
      );
      final mirrored = mapImagePointToPreview(
        x: 0,
        y: 0,
        imageSize: imageSize,
        previewBoxSize: previewBoxSize,
        quarterTurns: 1,
        mirror: true,
      );
      expect(mirrored.dx, previewBoxSize.width - unmirrored.dx);
      expect(mirrored.dy, unmirrored.dy);
    });

    test('2 turns: image scaled directly onto a same-orientation preview box, point inverted', () {
      final p = mapImagePointToPreview(
        x: 0,
        y: 0,
        imageSize: imageSize,
        previewBoxSize: imageSize,
        quarterTurns: 2,
      );
      expect(p, const Offset(640, 480));
    });

    test('3 turns: raw image top-right corner maps to preview top-left corner', () {
      final p = mapImagePointToPreview(
        x: 640,
        y: 0,
        imageSize: imageSize,
        previewBoxSize: previewBoxSize,
        quarterTurns: 3,
      );
      expect(p, const Offset(0, 0));
    });
  });
}
