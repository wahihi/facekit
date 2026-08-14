// Maps facekit's raw-sensor-space points (DetectedFace.boundingBox,
// FaceLandmarks) onto the CameraPreview widget's displayed box, and paints
// a box + eye-landmark overlay for the live recognition/liveness demo.
//
// Why this is needed: `CameraController.startImageStream` delivers frames in
// the *raw sensor orientation* (e.g. 1920x1080 even when the phone is held
// portrait) — facekit's detector/landmarker operate on that raw space. The
// `CameraPreview` widget, however, displays an already-rotated texture: on
// Android it wraps the native preview in a `RotatedBox` keyed off
// `CameraValue.deviceOrientation`, using the same quarter-turn table as
// camera's own `CameraPreview._getQuarterTurns()` (see
// camera-0.11.4/lib/src/camera_preview.dart). To draw a box that visually
// lines up with the live preview, every point must go through the same
// rotation before being scaled into the preview box's pixel size.
//
// Source: rotation/quarter-turn table replicated from the `camera` package's
// own (BSD-licensed, public) CameraPreview implementation — not a guess.
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart' show TargetPlatform, defaultTargetPlatform;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show DeviceOrientation;

import 'package:facekit/facekit.dart' show DetectedFace, FaceLandmarks, Point;

const Map<DeviceOrientation, int> _quarterTurnsByOrientation = {
  DeviceOrientation.portraitUp: 0,
  DeviceOrientation.landscapeRight: 1,
  DeviceOrientation.portraitDown: 2,
  DeviceOrientation.landscapeLeft: 3,
};

/// Quarter-turns (0..3, clockwise) the live preview texture is rotated by,
/// mirroring camera's own `CameraPreview._getQuarterTurns()`. RotatedBox
/// correction is Android-only — iOS's native preview is already upright.
/// [platform] defaults to [defaultTargetPlatform] (not a constant, hence the
/// nullable-parameter indirection rather than a default-value expression).
int quarterTurnsForOrientation(
  DeviceOrientation orientation, {
  TargetPlatform? platform,
}) {
  if ((platform ?? defaultTargetPlatform) != TargetPlatform.android) return 0;
  return _quarterTurnsByOrientation[orientation] ?? 0;
}

/// Rotates a point clockwise by [quarterTurns] * 90° within an
/// [imageWidth] x [imageHeight] box, returning the point in the
/// rotated box's own coordinate space (width/height swapped for odd turns).
Offset _rotatePoint(
  double x,
  double y,
  double imageWidth,
  double imageHeight,
  int quarterTurns,
) {
  switch (quarterTurns % 4) {
    case 1:
      return Offset(imageHeight - y, x);
    case 2:
      return Offset(imageWidth - x, imageHeight - y);
    case 3:
      return Offset(y, imageWidth - x);
    default:
      return Offset(x, y);
  }
}

/// Maps a point in raw camera-image pixel space into the displayed preview
/// box's local pixel space (origin top-left, matching [previewBoxSize]).
///
/// [mirror] flips horizontally — set for the front camera, whose preview is
/// conventionally shown mirrored (selfie convention).
Offset mapImagePointToPreview({
  required double x,
  required double y,
  required Size imageSize,
  required Size previewBoxSize,
  required int quarterTurns,
  bool mirror = false,
}) {
  final rotated = _rotatePoint(x, y, imageSize.width, imageSize.height, quarterTurns);
  final rotatedSize = quarterTurns % 2 == 1
      ? Size(imageSize.height, imageSize.width)
      : Size(imageSize.width, imageSize.height);

  final scaleX = rotatedSize.width == 0 ? 1.0 : previewBoxSize.width / rotatedSize.width;
  final scaleY = rotatedSize.height == 0 ? 1.0 : previewBoxSize.height / rotatedSize.height;

  var px = rotated.dx * scaleX;
  final py = rotated.dy * scaleY;
  if (mirror) px = previewBoxSize.width - px;
  return Offset(px, py);
}

/// Eye landmark indices used by [BlinkLivenessDetector] — drawn as small
/// dots so the demo visibly shows what's being tracked for the blink check.
const List<int> rightEyeIndices = [33, 159, 158, 133, 153, 145];
const List<int> leftEyeIndices = [362, 380, 374, 263, 386, 385];

class FaceOverlayPainter extends CustomPainter {
  final DetectedFace? face;
  final FaceLandmarks? landmarks;
  final Size imageSize;
  final int quarterTurns;
  final bool mirror;
  final Color boxColor;
  final String? label;

  /// Static framing guide, in image space — drawn regardless of whether a
  /// face is currently detected, so the user has something to aim for even
  /// before the first detection. Null hides it (e.g. before the frame size
  /// is known yet).
  final Rect? guideRegion;

  FaceOverlayPainter({
    required this.face,
    required this.landmarks,
    required this.imageSize,
    required this.quarterTurns,
    required this.mirror,
    required this.boxColor,
    this.label,
    this.guideRegion,
  });

  Offset _map(double x, double y, Size previewBoxSize) => mapImagePointToPreview(
        x: x,
        y: y,
        imageSize: imageSize,
        previewBoxSize: previewBoxSize,
        quarterTurns: quarterTurns,
        mirror: mirror,
      );

  @override
  void paint(Canvas canvas, Size size) {
    final guide = guideRegion;
    if (guide != null && !imageSize.isEmpty) {
      final guideRect = Rect.fromPoints(
        _map(guide.left, guide.top, size),
        _map(guide.right, guide.bottom, size),
      );
      final guidePaint = Paint()
        ..color = Colors.white.withValues(alpha: 0.7)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2;
      _drawDashedRect(canvas, guideRect, guidePaint);

      final hint = TextPainter(
        text: const TextSpan(
          text: '이 안에 얼굴을 맞춰주세요',
          style: TextStyle(
            color: Colors.white,
            fontSize: 14,
            fontWeight: FontWeight.w600,
            backgroundColor: Color(0xAA000000),
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout(maxWidth: guideRect.width);
      hint.paint(
        canvas,
        Offset(guideRect.left + (guideRect.width - hint.width) / 2, guideRect.top - hint.height - 4),
      );
    }

    final detected = face;
    if (detected == null || imageSize.isEmpty) return;

    final box = detected.boundingBox;
    final topLeft = _map(box.left, box.top, size);
    final bottomRight = _map(box.right, box.bottom, size);
    final rect = Rect.fromPoints(topLeft, bottomRight);

    final boxPaint = Paint()
      ..color = boxColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;
    canvas.drawRect(rect, boxPaint);

    final eyeLandmarks = landmarks;
    if (eyeLandmarks != null) {
      final dotPaint = Paint()..color = boxColor;
      for (final i in [...rightEyeIndices, ...leftEyeIndices]) {
        final p = eyeLandmarks.points[i];
        canvas.drawCircle(_map(p.x, p.y, size), 2.5, dotPaint);
      }
    }

    final text = label;
    if (text != null && text.isNotEmpty) {
      final painter = TextPainter(
        text: TextSpan(
          text: text,
          style: TextStyle(
            color: boxColor,
            fontSize: 16,
            fontWeight: FontWeight.bold,
            backgroundColor: const Color(0xAA000000),
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout(maxWidth: size.width);
      final labelTop = rect.top - painter.height - 4;
      painter.paint(canvas, Offset(rect.left, labelTop < 0 ? rect.top + 4 : labelTop));
    }
  }

  @override
  bool shouldRepaint(FaceOverlayPainter oldDelegate) =>
      oldDelegate.face != face ||
      oldDelegate.landmarks != landmarks ||
      oldDelegate.label != label ||
      oldDelegate.boxColor != boxColor ||
      oldDelegate.quarterTurns != quarterTurns ||
      oldDelegate.mirror != mirror ||
      oldDelegate.guideRegion != guideRegion;
}

/// Dashed rectangle outline — distinguishes the static framing guide from
/// the live face-detection box (solid line), so the two aren't confused
/// even when they briefly overlap.
void _drawDashedRect(Canvas canvas, Rect rect, Paint paint, {double dashLength = 8, double gapLength = 5}) {
  void dashedLine(Offset start, Offset end) {
    final total = (end - start).distance;
    if (total == 0) return;
    final direction = (end - start) / total;
    var covered = 0.0;
    while (covered < total) {
      final segmentEnd = (covered + dashLength).clamp(0.0, total);
      canvas.drawLine(start + direction * covered, start + direction * segmentEnd, paint);
      covered += dashLength + gapLength;
    }
  }

  dashedLine(rect.topLeft, rect.topRight);
  dashedLine(rect.topRight, rect.bottomRight);
  dashedLine(rect.bottomRight, rect.bottomLeft);
  dashedLine(rect.bottomLeft, rect.topLeft);
}

/// Convenience for reading the quarter-turns to use for a given controller's
/// current device orientation — mirrors `CameraPreview`'s own (simplified:
/// this app never pauses/locks capture orientation, so deviceOrientation is
/// the only input that matters here).
int quarterTurnsForController(CameraController controller) =>
    quarterTurnsForOrientation(controller.value.deviceOrientation);

Point boxTopLeft(DetectedFace face) => Point(face.boundingBox.left, face.boundingBox.top);
