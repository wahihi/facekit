// facekit example — register a face, then recognize it live from the camera.
//
// Loads YuNet (bundled with the facekit package, MIT licence) for detection,
// MediaPipe Face Landmarker (also bundled, Apache 2.0) for blink-based
// liveness, and AuraFace (bundled/redistributable, Apache 2.0 — see
// assets/models/auraface/) for embedding, then drives FacePipeline end-to-end
// against the device camera feed. The AuraFace .tflite weight itself is not
// committed to git (see tool/fetch_models.sh); arcface_buffalo_l is kept
// alongside it as a BYOM example — swap `_embedderDir`/`_embedderFile` below
// to switch.
//
// YuNet replaced BlazeFace as the default detector (doc/KR/postmortem/
// 2026-08-12-yunet-landmark-order.md) — it natively outputs 5 landmarks
// (2 eyes, nose, 2 mouth corners), letting AffineAligner use the true
// ArcFace 5-point reference instead of BlazeFace's 4-point compromise
// (BlazeFace only ever gave a single mouth-centre point). Real-device
// validation of this switch is still pending; BlazeFaceDetector remains
// available (lib/src/detection/blazeface_detector.dart) as a fallback if
// YuNet underperforms on-device.
//
// Every frame draws a box overlay (see face_overlay.dart) over the detected
// face, and gates enroll/identify on `BlinkLivenessDetector` passing first —
// holding up a static photo never blinks, so it never reaches the matcher.
import 'dart:async' show unawaited;
import 'dart:math' as math;
import 'dart:ui' as ui show Rect;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart'
    show Clipboard, ClipboardData, DeviceOrientation, rootBundle;

import 'package:facekit/facekit.dart';

import 'benchmark.dart';
import 'face_overlay.dart';

// Demo-mode delays for video recording — flip to false to remove all delays.
// Dart const bool is tree-shaken: when false the if-blocks below are
// eliminated from the compiled binary, identical to C's #ifdef _DEMO_MODE.
const _kDemoMode = true;

// Temporarily gates out the blink-liveness check entirely (enroll/identify
// treat every frame as already "passed") — a debugging toggle, not a
// permanent policy decision. Real-device sessions (log3.txt, 2026-08-14)
// showed liveness contributing its own friction on top of whatever was
// making recognition itself unstable (frequent face-lost resets discarding
// in-progress blink progress, genuine blinks being rare during a framing/
// zoom test), which made it hard to tell how much of the instability was
// recognition-quality vs liveness-gating. Flip back to true once
// recognition stability is confirmed independently. `MediaPipeFaceLandmarker`
// .detectLandmarks is skipped too while this is off, not just the gate —
// no point paying for landmark inference nothing consumes.
const _kLivenessEnabled = false;

// Embedding model in use — swap these two constants to switch models (e.g.
// back to 'assets/models/arcface_buffalo_l' / 'w600k_r50.tflite', kept in the
// repo as a BYOM example). See the manifest.json alongside each .tflite for
// that model's family/license.
const _embedderDir = 'assets/models/auraface';
const _embedderFile = 'auraface_r100_fp16.tflite';

late List<CameraDescription> _cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  _cameras = await availableCameras();
  runApp(const FacekitExampleApp());
}

class FacekitExampleApp extends StatelessWidget {
  const FacekitExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'facekit example',
      theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
      home: const RecognitionPage(),
    );
  }
}

class RecognitionPage extends StatefulWidget {
  const RecognitionPage({super.key});

  @override
  State<RecognitionPage> createState() => _RecognitionPageState();
}

class _RecognitionPageState extends State<RecognitionPage> {
  CameraController? _controller;
  FacePipeline? _pipeline;
  MediaPipeFaceLandmarker? _landmarker;
  final List<Enrollment> _gallery = [];
  final _nameController = TextEditingController(text: '나');

  bool _busy = false; // guards against overlapping inference calls per frame
  String? _pendingEnrollName; // set by the 등록 button, consumed by next frame
  bool _identifying = false;
  String _status = '모델을 불러오는 중...';
  CameraLensDirection _lensDirection =
      CameraLensDirection.front; //CameraLensDirection.back; misowish: front로 변경
  bool _switchingCamera = false;

  // Liveness — Free-tier blink check. Re-created on face loss via .reset()
  // rather than allocated fresh, since it carries no per-face identity state.
  final _liveness = BlinkLivenessDetector();
  LivenessState _livenessState = LivenessState.pending;

  // Drives the overlay painter — updated every frame regardless of
  // enroll/identify state so the box is always visible while a face is in
  // frame (see FaceOverlayPainter in face_overlay.dart).
  DetectedFace? _overlayFace;
  FaceLandmarks? _overlayLandmarks;
  Size _overlayImageSize = Size.zero;
  String? _matchLabel; // "이름 (유사도 0.xx)" while a live identify match holds

  // Persistent framing tip, shown separately from `_status` (which gets
  // overwritten by every enroll/identify/liveness message, so a one-time
  // status string is invisible again the moment anything else happens) —
  // only non-null while auto-zoom has already maxed out and the face is
  // still below the target fraction, i.e. the one case auto-zoom genuinely
  // can't fix on its own and needs the user to physically move.
  String? _framingHint;

  // Latest frame that had a detected face, frozen for the benchmark button —
  // yuv420ToFaceImage always allocates a fresh buffer, so holding this
  // reference across frames is safe (the camera plugin can't mutate it).
  FaceImage? _lastFaceImage;
  bool _benchmarking = false;

  // Logged-value cache purely for debugPrint de-duplication below — keeps
  // `flutter logs` readable by printing only on transitions, not every frame.
  bool? _loggedHasFace;
  LivenessState? _loggedLivenessState;
  bool _useNnApiForBenchmark = false;

  // How long a face can be briefly undetected before liveness progress is
  // discarded. Originally measured on real Pixel 7 hardware with BlazeFace:
  // its detection score could hover right around its 0.5 threshold (logged
  // score 0.5-0.7), flickering found/lost every ~100-200ms. Without this
  // grace period, _liveness.reset() fired on every single missed frame and
  // the 4s blink window restarted from zero each time, making `passed` take
  // far longer than intended. 500ms was a round starting value sized to
  // cover a few missed-detection frames, not derived from measured footage.
  // Carried over as-is for YuNet (score_threshold 0.6, see
  // assets/models/yunet_160/manifest.json) — not yet re-measured on-device
  // for the new detector's own flicker behaviour; revisit if it proves too
  // forgiving or still resets too often.
  static const _faceLossGraceMs = 500;
  int? _faceLostSinceMs;

  // Auto-zoom: resizeNearest squeezes the *whole* camera frame down to
  // YuNet's fixed 160x160 input, so a face that's only a small fraction of
  // the frame gets very few input pixels, which lets ordinary frame-to-frame
  // camera noise swing decoded landmarks by a large fraction of the face's
  // own size — measured 160-288x higher jitter below ~20% face-width than
  // above it (doc/KR/postmortem/2026-08-13-landmark-jitter-frame-fill.md).
  // Camera-level zoom (CameraController.setZoomLevel) fixes this more
  // fundamentally than any post-capture software crop can: ResolutionPreset
  // .medium already caps capture at a fraction of the sensor's native
  // resolution, so a crop *after* that capture can't recover detail the
  // camera driver already discarded — zooming the *capture* itself asks the
  // driver for a tighter, still-native-resolution-backed region instead.
  //
  // Target range is deliberately a band, not a single point: the postmortem
  // sweep's jitter was already low and roughly flat across ~20-90% face
  // width, so there's no benefit to hunting for an exact fraction, only a
  // need to stay out of the unstable <20% tail. The band's lower edge sits
  // just above that tail with some margin; the upper edge avoids the very-
  // close-range false rejections seen in real device logs (log4.txt), which
  // looked more like focus/framing artefacts than a benefit of zooming
  // further in.
  //
  // First real-device run (log2.txt, 2026-08-14) oscillated hard instead of
  // settling: zoom bounced 0.90<->1.20<->1.50 repeatedly and faceFraction
  // swung between ~25% and ~70-78% — overshooting the target band in *both*
  // directions every single time, never landing inside it — and the
  // rejected-match frames clustered almost exactly inside that oscillating
  // stretch (a calmer stretch elsewhere in the same log had 8 consecutive
  // accepted matches). Two compounding causes: (1) 400ms was shorter than
  // this device's actual zoom-settle latency, so the next frame's
  // faceFraction reading still reflected the *previous* zoom level, making
  // each correction react to stale data and overshoot; (2) getMinZoomLevel()
  // returned <1.0 (0.8958...) on this device, meaning the low end of the
  // range crosses into an ultra-wide *physical lens switch*, which is a
  // discontinuous FOV jump, not smooth zoom — a 33% zoom-level increase
  // (0.90->1.20) produced a >2x faceFraction jump because it crossed that
  // seam. Fixed by clamping the zoom floor to 1.0 (never cross the lens
  // seam), lengthening the cooldown well past the observed settle lag,
  // shrinking the step so any residual overshoot is smaller, and requiring
  // a few *consecutive* out-of-band readings before reacting at all (a
  // single noisy frame shouldn't retrigger a hardware zoom call).
  static const _targetFaceFractionLow = 0.30;
  static const _targetFaceFractionHigh = 0.60;
  static const _zoomStep = 0.15; // in the controller's own zoom-level units
  static const _zoomCooldownMs = 900; // longer than the observed settle lag
  static const _zoomOutOfRangeStreakThreshold = 3;
  double _minZoom = 1.0;
  double _maxZoom = 1.0;
  double _currentZoom = 1.0;
  int? _lastZoomAdjustMs;
  int _zoomOutOfRangeStreak = 0;
  int _zoomOutOfRangeDirection = 0; // -1 too small, +1 too big, 0 none yet

  // Lazily built only when the NNAPI toggle is first used — a second
  // detector/embedder pair pointed at the same models but with the NNAPI
  // delegate on, so the benchmark can compare CPU-only vs. NNAPI on the same
  // device without touching the main pipeline used for real enroll/identify.
  ModelManifest? _detectorManifest;
  ModelManifest? _embedderManifest;
  YuNetDetector? _nnapiDetector;
  TfliteFaceEmbedder? _nnapiEmbedder;

  @override
  void initState() {
    super.initState();
    _setup();
  }

  Future<void> _setup() async {
    try {
      final detectorManifest = ModelManifest.fromJsonString(
        await rootBundle.loadString(
          'packages/facekit/assets/models/yunet_160/manifest.json',
        ),
      );
      final detector = await YuNetDetector.fromAsset(
        tfliteAssetPath:
            'packages/facekit/assets/models/yunet_160/yunet_160.tflite',
        manifest: detectorManifest,
      );
      _detectorManifest = detectorManifest;

      final embedderManifest = ModelManifest.fromJsonString(
        await rootBundle.loadString('$_embedderDir/manifest.json'),
      );
      final embedder = await TfliteFaceEmbedder.fromAsset(
        tfliteAssetPath: '$_embedderDir/$_embedderFile',
        manifest: embedderManifest,
      );
      _embedderManifest = embedderManifest;

      _pipeline = FacePipeline(
        detector: detector,
        aligner: AffineAligner.arcface112(nativeFivePoint: true),
        embedder: embedder,
        matcher: CosineMatcher.fromManifest(embedderManifest),
      );

      final landmarkerManifest = ModelManifest.fromJsonString(
        await rootBundle.loadString(
          'packages/facekit/assets/models/face_landmark_478/manifest.json',
        ),
      );
      _landmarker = await MediaPipeFaceLandmarker.fromAsset(
        tfliteAssetPath:
            'packages/facekit/assets/models/face_landmark_478/face_landmark_478.tflite',
        manifest: landmarkerManifest,
      );

      await _initCamera(_lensDirection);

      setState(() {
        // Framing is now handled automatically (auto-zoom nudges toward a
        // safe face-to-frame ratio; the ML rotation path adapts to whatever
        // orientation the phone is actually held in) — this no longer needs
        // to tell the user to hold it any particular way. If auto-zoom ever
        // maxes out and still can't get the face big enough, `_framingHint`
        // (a persistent hint that survives status-text overwrites, unlike
        // this one-shot message) picks up the slack.
        _status = '준비 완료 — "등록"을 눌러보세요.';
      });
      if (_kDemoMode) await Future.delayed(const Duration(seconds: 2));
    } catch (e) {
      final message = e.toString();
      // rootBundle throws this when a declared asset isn't actually on disk —
      // the case for model .tflite files, which are fetched separately via
      // tool/fetch_models.sh rather than committed to git.
      final hint = message.contains('Unable to load asset')
          ? '\n모델 파일이 없습니다. tool/fetch_models.sh 를 실행해 받아주세요.\n'
              'Model file is missing — run tool/fetch_models.sh to download it.'
          : '';
      setState(() => _status = '초기화 실패: $message$hint');
    }
  }

  /// (Re)initialises the camera controller for the given lens direction and
  /// starts streaming frames into [_onFrame]. Used both at startup and when
  /// the user taps the front/back switch button.
  Future<void> _initCamera(CameraLensDirection direction) async {
    final previous = _controller;
    if (previous != null) {
      if (previous.value.isStreamingImages) {
        await previous.stopImageStream();
      }
      await previous.dispose();
    }

    final camera = _cameras.firstWhere(
      (c) => c.lensDirection == direction,
      orElse: () => _cameras.first,
    );
    final controller = CameraController(
      camera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    await controller.initialize();
    // Clamped to 1.0: some devices report getMinZoomLevel() below 1.0,
    // which crosses into a physical ultra-wide lens switch (a discontinuous
    // FOV jump, not smooth zoom) — see _targetFaceFractionLow's doc comment.
    final rawMinZoom = await controller.getMinZoomLevel();
    _minZoom = math.max(1.0, rawMinZoom);
    _maxZoom = await controller.getMaxZoomLevel();
    _currentZoom = _minZoom;
    debugPrint(
      '[Zoom] init: lensDirection=$direction rawMinZoom=$rawMinZoom '
      'clampedMinZoom=$_minZoom maxZoom=$_maxZoom '
      'capable=${_maxZoom > _minZoom}',
    );
    await controller.setZoomLevel(_currentZoom);
    _lastZoomAdjustMs = null;
    _zoomOutOfRangeStreak = 0;
    await controller.startImageStream(_onFrame);

    if (!mounted) return;
    setState(() {
      _controller = controller;
      _lensDirection = camera.lensDirection;
    });
  }

  Future<void> _switchCamera() async {
    if (_cameras.length < 2 || _switchingCamera) return;
    final next = _lensDirection == CameraLensDirection.back
        ? CameraLensDirection.front
        : CameraLensDirection.back;

    setState(() {
      _switchingCamera = true;
      _status = '카메라 전환 중...';
    });
    try {
      await _initCamera(next);
      _setStatus(
        next == CameraLensDirection.front ? '전면 카메라로 전환됨' : '후면 카메라로 전환됨',
      );
    } catch (e) {
      _setStatus('카메라 전환 실패: $e');
    } finally {
      if (mounted) setState(() => _switchingCamera = false);
    }
  }

  FaceImage? _toFaceImage(CameraImage image) {
    if (image.format.group != ImageFormatGroup.yuv420) return null;
    final raw = yuv420ToFaceImage(
      yPlane: image.planes[0].bytes,
      uPlane: image.planes[1].bytes,
      vPlane: image.planes[2].bytes,
      width: image.width,
      height: image.height,
      yRowStride: image.planes[0].bytesPerRow,
      uvRowStride: image.planes[1].bytesPerRow,
      uvPixelStride: image.planes[1].bytesPerPixel ?? 2,
    );
    return rotateFaceImage(raw, _cameraQuarterTurns());
  }

  /// Camera sensors are mounted independently of how the phone is held, so
  /// the raw YUV buffer above is in the sensor's native (usually landscape)
  /// orientation regardless of device orientation.
  ///
  /// [_baseQuarterTurns] below is a *fixed* rotation, empirically calibrated
  /// on a real Pixel 7 (front camera, sensorOrientation=270) to produce an
  /// upright frame when the device is held in [DeviceOrientation.portraitUp]
  /// — the textbook ML Kit-style front-camera formula `(360 -
  /// sensorOrientation) % 360` does NOT hold here (it produced a
  /// 90°-rotated crop, verified by reconstructing raw landmark geometry
  /// from a device debug dump: eyes came out separated vertically instead
  /// of horizontally). The +90° adjustment is empirically calibrated to
  /// that one measurement, not re-derived from camera theory.
  ///
  /// That fixed value alone is only correct for *that one* physical
  /// orientation, though — this app used to assume the device was always
  /// held portraitUp and never re-checked, which is a real bug: the ML
  /// path (this function) had no way to know if that assumption held,
  /// while `CameraPreview`/`face_overlay.dart`'s `quarterTurnsForOrientation`
  /// already reads the *live* `CameraValue.deviceOrientation` to keep the
  /// on-screen preview correctly upright regardless of how the phone is
  /// actually held right now. That means a user could hold the device in
  /// any orientation, see a perfectly normal-looking preview (since that
  /// path already compensates), while the ML pipeline silently received a
  /// rotated frame underneath — degrading detection/landmark/liveness
  /// quality with no visible symptom pointing at "orientation". See
  /// doc/KR/postmortem/2026-08-14-camera-orientation-and-auto-zoom.md.
  ///
  /// Fix: combine the fixed base calibration (still exactly correct for
  /// portraitUp, so that case is unchanged/non-regressing) with the same
  /// live-orientation delta table `quarterTurnsForOrientation` already
  /// uses for the preview, so the ML path stays correctly upright in any
  /// held orientation, not just the one it happened to be calibrated in.
  /// The back-camera branch is unchanged and untested — this app is only
  /// ever run with the front camera in practice.
  int get _baseQuarterTurns {
    final sensorOrientation = _controller?.description.sensorOrientation ?? 0;
    final degrees = _lensDirection == CameraLensDirection.front
        ? (450 - sensorOrientation) % 360
        : sensorOrientation % 360;
    return degrees ~/ 90;
  }

  int? _loggedQuarterTurns;

  int _cameraQuarterTurns() {
    final liveOrientation =
        _controller?.value.deviceOrientation ?? DeviceOrientation.portraitUp;
    final orientationDelta = quarterTurnsForOrientation(liveOrientation);
    final turns = (_baseQuarterTurns + orientationDelta) % 4;
    if (_loggedQuarterTurns != turns) {
      _loggedQuarterTurns = turns;
      debugPrint(
        '[_cameraQuarterTurns] lensDirection=$_lensDirection '
        'baseQuarterTurns=$_baseQuarterTurns liveOrientation=$liveOrientation '
        'orientationDelta=$orientationDelta turns=$turns',
      );
    }
    return turns;
  }

  /// Runs every camera frame, independent of enroll/identify state, so the
  /// box overlay + liveness check are always live. The detect→landmark→
  /// liveness chain is cheap (YuNet + a small 256×256 landmark model;
  /// see doc/KR/benchmark.md) and runs on the calling isolate, matching how
  /// `FacePipeline.identify/enroll` already run detection synchronously —
  /// only embedding inference (the expensive step) goes to an isolate.
  Future<void> _onFrame(CameraImage frame) async {
    if (_busy) return; // drop frames while the previous one is still running
    final pipeline = _pipeline;
    final landmarker = _landmarker;
    if (pipeline == null || landmarker == null) return;

    _busy = true;
    final frameStopwatch = Stopwatch()..start();
    try {
      final image = _toFaceImage(frame);
      if (image == null) return;

      // Tracked on every frame (not just ones with a face) so the framing
      // guide has a size to draw against before the very first detection.
      final newImageSize = Size(image.width.toDouble(), image.height.toDouble());
      if (_overlayImageSize != newImageSize) {
        setState(() => _overlayImageSize = newImageSize);
      }

      final faces = await pipeline.detector.detect(image);
      debugPrint('[Timing] detect() (onFrame pass): ${frameStopwatch.elapsedMilliseconds}ms');
      final face = faces.isEmpty
          ? null
          : faces.reduce((a, b) => a.score >= b.score ? a : b);

      if (face == null) {
        final now = DateTime.now().millisecondsSinceEpoch;
        _faceLostSinceMs ??= now;
        if (now - _faceLostSinceMs! >= _faceLossGraceMs) {
          // Genuinely gone (not just a 1-2 frame detection flicker) — safe
          // to discard blink progress now.
          if (_loggedHasFace != false) {
            debugPrint(
              '[onFrame] face lost (>${_faceLossGraceMs}ms, resetting liveness)',
            );
            _loggedHasFace = false;
            _loggedLivenessState = null;
          }
          _liveness.reset();
        }
        setState(() {
          _overlayFace = null;
          _overlayLandmarks = null;
          _livenessState = LivenessState.pending;
          _matchLabel = null;
          _framingHint = null;
        });
        if (_identifying) _setStatus('얼굴 없음');
        return;
      }
      _faceLostSinceMs = null;
      if (_loggedHasFace != true) {
        debugPrint(
          '[onFrame] face found, score=${face.score.toStringAsFixed(3)}',
        );
        _loggedHasFace = true;
      }
      _maybeAdjustZoom(face.boundingBox.width / image.width);

      _lastFaceImage = image;

      final landmarks = _kLivenessEnabled
          ? await landmarker.detectLandmarks(image, face)
          : null;
      final liveness = !_kLivenessEnabled
          ? const LivenessResult(state: LivenessState.passed)
          : landmarks == null
          ? const LivenessResult(state: LivenessState.pending)
          : _liveness.update(landmarks, DateTime.now().millisecondsSinceEpoch);
      if (_loggedLivenessState != liveness.state) {
        debugPrint(
          '[onFrame] liveness: ${liveness.state}'
          '${liveness.failReason != null ? " (${liveness.failReason})" : ""}',
        );
        _loggedLivenessState = liveness.state;
      }

      setState(() {
        _overlayFace = face;
        _overlayLandmarks = landmarks;
        _livenessState = liveness.state;
      });

      final enrollName = _pendingEnrollName;
      if (liveness.state != LivenessState.passed) {
        if (_matchLabel != null) setState(() => _matchLabel = null);
        if (enrollName != null || _identifying) {
          _setStatus('라이브니스 확인 중 — 카메라를 보고 눈을 깜빡여주세요.');
        }
        return;
      }

      if (enrollName != null) {
        _pendingEnrollName = null;
        _liveness.reset();
        final embedding = await pipeline.enroll(image);
        debugPrint('[Timing] pipeline.enroll() done: ${frameStopwatch.elapsedMilliseconds}ms total');
        if (embedding == null) {
          _setStatus('얼굴을 찾지 못했어요. 카메라에 얼굴이 잘 보이게 해주세요.');
        } else {
          _gallery.add(Enrollment(id: enrollName, embedding: embedding));
          _setStatus('"$enrollName" 등록 완료 (총 ${_gallery.length}명)');
          if (_kDemoMode) await Future.delayed(const Duration(seconds: 3));
        }
      } else if (_identifying) {
        final result = await pipeline.identify(image, _gallery);
        debugPrint('[Timing] pipeline.identify() done: ${frameStopwatch.elapsedMilliseconds}ms total');
        if (result == null) {
          setState(() => _matchLabel = null);
          _setStatus('얼굴 없음');
        } else if (result.accepted) {
          final label =
              '${result.matchedId} (유사도 ${result.similarity.toStringAsFixed(2)})';
          setState(() => _matchLabel = label);
          _setStatus('인식됨: $label');
        } else {
          setState(() => _matchLabel = null);
          _setStatus('모르는 얼굴 (유사도 ${result.similarity.toStringAsFixed(2)})');
        }
      }
    } catch (e) {
      _setStatus('오류: $e');
    } finally {
      debugPrint('[Timing] onFrame total: ${frameStopwatch.elapsedMilliseconds}ms');
      _busy = false;
    }
  }

  void _setStatus(String text) {
    if (!mounted) return;
    setState(() => _status = text);
  }

  /// Nudges the camera's optical/hybrid zoom toward keeping the detected
  /// face's width within [_targetFaceFractionLow, _targetFaceFractionHigh]
  /// of the frame — see the field doc comment above for why this matters
  /// more than any post-capture software crop, and for what went wrong the
  /// first time (log2.txt's oscillation) that the streak/cooldown/step
  /// tuning below is responding to. A single out-of-band frame doesn't
  /// trigger anything — [_zoomOutOfRangeStreakThreshold] consecutive frames
  /// in the *same* direction are required first, so one noisy reading can't
  /// retrigger a hardware zoom call while a previous one is still settling.
  ///
  /// Also drives [_framingHint]: zoom alone can't help once it's already
  /// maxed out, so that's the one case left for the user to fix by
  /// physically moving closer — surfaced as a persistent hint rather than
  /// a one-shot status message (see [_framingHint]'s doc comment).
  void _maybeAdjustZoom(double faceFraction) {
    final controller = _controller;
    final capable = controller != null && _maxZoom > _minZoom;
    debugPrint(
      '[Zoom] check: faceFraction=${faceFraction.toStringAsFixed(3)} '
      'capable=$capable minZoom=$_minZoom maxZoom=$_maxZoom '
      'currentZoom=$_currentZoom streak=$_zoomOutOfRangeStreak '
      'streakDirection=$_zoomOutOfRangeDirection',
    );
    if (!capable) return;

    final tooFarEvenAtMaxZoom =
        faceFraction < _targetFaceFractionLow && _currentZoom >= _maxZoom;
    _setFramingHint(
      tooFarEvenAtMaxZoom ? '카메라에 조금 더 가까이 와주세요 (줌 최대)' : null,
    );

    // +1 = zoom in (face too small), -1 = zoom out (face too big). This sign
    // was inverted in an earlier revision (log3.txt, 2026-08-14) — zoom kept
    // climbing past 95% face-width instead of backing off, because "too big"
    // mapped to +1 (zoom in further) instead of -1 (zoom out).
    final direction = faceFraction < _targetFaceFractionLow
        ? 1
        : faceFraction > _targetFaceFractionHigh
        ? -1
        : 0;
    if (direction == 0 || direction != _zoomOutOfRangeDirection) {
      _zoomOutOfRangeStreak = direction == 0 ? 0 : 1;
      _zoomOutOfRangeDirection = direction;
    } else {
      _zoomOutOfRangeStreak++;
    }
    if (direction == 0 || _zoomOutOfRangeStreak < _zoomOutOfRangeStreakThreshold) {
      return;
    }

    final now = DateTime.now().millisecondsSinceEpoch;
    if (_lastZoomAdjustMs != null && now - _lastZoomAdjustMs! < _zoomCooldownMs) {
      return;
    }

    final next = (_currentZoom + direction * _zoomStep).clamp(_minZoom, _maxZoom);
    if (next == _currentZoom) return;

    _currentZoom = next;
    _lastZoomAdjustMs = now;
    _zoomOutOfRangeStreak = 0;
    debugPrint(
      '[AutoZoom] faceFraction=${faceFraction.toStringAsFixed(3)} -> '
      'zoom=${next.toStringAsFixed(2)} (range $_minZoom-$_maxZoom)',
    );
    // Fire-and-forget: awaiting here would hold _busy for the platform
    // channel round-trip, dropping camera frames for no benefit — the next
    // frame just reads whatever zoom level is in effect by then.
    unawaited(controller.setZoomLevel(next));
  }

  void _setFramingHint(String? hint) {
    if (_framingHint == hint || !mounted) return;
    setState(() => _framingHint = hint);
  }

  /// Lazily builds a second detector+embedder pair pointed at the same model
  /// files but with the NNAPI delegate on, for the benchmark toggle. Built
  /// once and cached — switching the toggle on and off doesn't reload.
  Future<void> _ensureNnApiModels() async {
    if (_nnapiDetector != null && _nnapiEmbedder != null) return;
    final detectorManifest = _detectorManifest;
    final embedderManifest = _embedderManifest;
    if (detectorManifest == null || embedderManifest == null) return;

    _nnapiDetector = await YuNetDetector.fromAsset(
      tfliteAssetPath:
          'packages/facekit/assets/models/yunet_160/yunet_160.tflite',
      manifest: detectorManifest,
      useNnApi: true,
    );
    _nnapiEmbedder = await TfliteFaceEmbedder.fromAsset(
      tfliteAssetPath: '$_embedderDir/$_embedderFile',
      manifest: embedderManifest,
      useNnApi: true,
    );
  }

  /// Runs the benchmark on the last frame that had a face in it, then shows
  /// the result in a dialog (with a copy button) so the numbers can be read
  /// off-device without adb — see doc/KR/benchmark.md. Toggling
  /// [_useNnApiForBenchmark] swaps in NNAPI-delegated detector/embedder
  /// instances instead of the CPU-only ones the main pipeline uses, so the
  /// two runs are comparable on the same device.
  Future<void> _runBenchmark() async {
    final pipeline = _pipeline;
    final image = _lastFaceImage;
    if (pipeline == null || image == null) {
      _setStatus('먼저 카메라에 얼굴을 비춰주세요.');
      return;
    }

    setState(() => _benchmarking = true);
    _setStatus('벤치마크 실행 중... (35회 추론, 잠시 기다려주세요)');
    try {
      FaceDetector detector = pipeline.detector;
      FaceEmbedder embedder = pipeline.embedder;
      if (_useNnApiForBenchmark) {
        await _ensureNnApiModels();
        detector = _nnapiDetector ?? detector;
        embedder = _nnapiEmbedder ?? embedder;
      }

      final result = await runBenchmark(
        detector: detector,
        aligner: pipeline.aligner,
        embedder: embedder,
        image: image,
      );
      if (!mounted) return;
      _setStatus('벤치마크 완료');
      final modeLabel = _useNnApiForBenchmark ? '[NNAPI]' : '[CPU]';
      final reportText = '$modeLabel\n${result.toReportText()}';
      await showDialog<void>(
        context: context,
        builder: (context) => AlertDialog(
          title: Text('벤치마크 결과 $modeLabel'),
          content: SelectableText(reportText),
          actions: [
            TextButton(
              onPressed: () {
                Clipboard.setData(ClipboardData(text: reportText));
                _setStatus('결과를 클립보드에 복사했어요.');
              },
              child: const Text('복사'),
            ),
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('닫기'),
            ),
          ],
        ),
      );
    } catch (e) {
      _setStatus('벤치마크 실패: $e');
    } finally {
      if (mounted) setState(() => _benchmarking = false);
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    _landmarker?.dispose();
    _nnapiDetector?.dispose();
    _nnapiEmbedder?.dispose();
    _nameController.dispose();
    super.dispose();
  }

  /// Static framing guide, in image space — the centred square
  /// [YuNetDetector]'s first pass now crops to (see
  /// `centerSquareCropRegion` in facekit), shrunk by [_guideRegionMargin]
  /// so a user who's slightly off still lands inside the actual crop
  /// rather than right on its edge. Drawn regardless of whether a face is
  /// currently detected — see [FaceOverlayPainter.guideRegion].
  static const _guideRegionMargin = 0.85;

  ui.Rect? get _guideRegion {
    if (_overlayImageSize.isEmpty) return null;
    // centerSquareCropRegion returns facekit's own Rect (core/models.dart) —
    // converted to dart:ui's Rect below since that's what FaceOverlayPainter
    // (a Canvas/CustomPainter type) expects.
    final full = centerSquareCropRegion(
      _overlayImageSize.width.round(),
      _overlayImageSize.height.round(),
    );
    final shrink = full.width * (1 - _guideRegionMargin) / 2;
    return ui.Rect.fromLTRB(
      full.left + shrink,
      full.top + shrink,
      full.right - shrink,
      full.bottom - shrink,
    );
  }

  @override
  Widget build(BuildContext context) {
    final controller = _controller;
    return Scaffold(
      appBar: AppBar(
        title: const Text('facekit example'),
        actions: [
          IconButton(
            icon: const Icon(Icons.cameraswitch),
            tooltip: '전면/후면 카메라 전환',
            onPressed: (_pipeline == null || _switchingCamera)
                ? null
                : _switchCamera,
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: controller == null || !controller.value.isInitialized
                ? const Center(child: CircularProgressIndicator())
                : CameraPreview(
                    controller,
                    child: CustomPaint(
                      painter: FaceOverlayPainter(
                        face: _overlayFace,
                        landmarks: _overlayLandmarks,
                        imageSize: _overlayImageSize,
                        // `_overlayImageSize` is already the *orientation-
                        // corrected* FaceImage's size (rotated by
                        // _cameraQuarterTurns(), which now folds in live
                        // deviceOrientation) — it's already upright, same as
                        // what CameraPreview itself already displays, so no
                        // further rotation is needed here. Using
                        // quarterTurnsForController(controller) (raw-sensor-
                        // space delta) on top of an already-corrected image
                        // would double-rotate the overlay away from the face
                        // for any non-portraitUp hold.
                        quarterTurns: 0,
                        mirror: _lensDirection == CameraLensDirection.front,
                        guideRegion: _guideRegion,
                        boxColor: _livenessState == LivenessState.passed
                            ? Colors.greenAccent
                            : Colors.amber,
                        label: _overlayFace == null
                            ? null
                            : _matchLabel ??
                                  (_livenessState == LivenessState.passed
                                      ? '라이브 확인됨'
                                      : '눈을 깜빡여주세요'),
                      ),
                    ),
                  ),
          ),
          Padding(
            padding: const EdgeInsets.all(12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Text(
                  _status,
                  textAlign: TextAlign.center,
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                if (_framingHint != null) ...[
                  const SizedBox(height: 4),
                  Text(
                    _framingHint!,
                    textAlign: TextAlign.center,
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: Colors.orange.shade800,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
                const SizedBox(height: 12),
                TextField(
                  controller: _nameController,
                  decoration: const InputDecoration(
                    labelText: '등록할 이름',
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    Expanded(
                      child: ElevatedButton(
                        onPressed: _pipeline == null
                            ? null
                            : () => setState(() {
                                _pendingEnrollName =
                                    _nameController.text.trim().isEmpty
                                    ? '나'
                                    : _nameController.text.trim();
                                _status = '등록 중... 카메라를 바라봐주세요.';
                              }),
                        child: const Text('현재 얼굴 등록'),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: ElevatedButton(
                        onPressed: _pipeline == null
                            ? null
                            : () => setState(() {
                                _identifying = !_identifying;
                                _status = _identifying
                                    ? '실시간 인식 중...'
                                    : '인식 중지됨';
                              }),
                        child: Text(_identifying ? '인식 중지' : '실시간 인식 시작'),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 4),
                SwitchListTile(
                  contentPadding: EdgeInsets.zero,
                  dense: true,
                  title: const Text('NNAPI 가속 사용 (벤치마크용)'),
                  value: _useNnApiForBenchmark,
                  onChanged: _benchmarking
                      ? null
                      : (v) => setState(() => _useNnApiForBenchmark = v),
                ),
                OutlinedButton(
                  onPressed: (_pipeline == null || _benchmarking)
                      ? null
                      : _runBenchmark,
                  child: Text(_benchmarking ? '벤치마크 실행 중...' : '벤치마크 실행'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
