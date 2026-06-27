// facekit example — register a face, then recognize it live from the camera.
//
// Loads BlazeFace (bundled with the facekit package) for detection,
// MediaPipe Face Landmarker (also bundled, Apache 2.0) for blink-based
// liveness, and ArcFace buffalo_l (BYOM/Demo, bundled only in this example
// app's own assets — see assets/models/arcface_buffalo_l/) for embedding,
// then drives FacePipeline end-to-end against the device camera feed.
//
// Every frame draws a box overlay (see face_overlay.dart) over the detected
// face, and gates enroll/identify on `BlinkLivenessDetector` passing first —
// holding up a static photo never blinks, so it never reaches the matcher.
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;

import 'package:facekit/facekit.dart';

import 'face_overlay.dart';

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
  CameraLensDirection _lensDirection = CameraLensDirection.back;
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

  @override
  void initState() {
    super.initState();
    _setup();
  }

  Future<void> _setup() async {
    try {
      final detectorManifest = ModelManifest.fromJsonString(
        await rootBundle.loadString(
          'packages/facekit/assets/models/blazeface_short/manifest.json',
        ),
      );
      final detector = await BlazeFaceDetector.fromAsset(
        tfliteAssetPath:
            'packages/facekit/assets/models/blazeface_short/face_detection_short_range.tflite',
        manifest: detectorManifest,
      );

      final embedderManifest = ModelManifest.fromJsonString(
        await rootBundle.loadString(
          'assets/models/arcface_buffalo_l/manifest.json',
        ),
      );
      final embedder = await TfliteFaceEmbedder.fromAsset(
        tfliteAssetPath: 'assets/models/arcface_buffalo_l/w600k_r50.tflite',
        manifest: embedderManifest,
      );

      _pipeline = FacePipeline(
        detector: detector,
        aligner: AffineAligner.arcface112(),
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
        _status = '준비 완료 — 카메라를 가로로 들고 "등록"을 눌러보세요.';
      });
    } catch (e) {
      setState(() => _status = '초기화 실패: $e');
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
    return yuv420ToFaceImage(
      yPlane: image.planes[0].bytes,
      uPlane: image.planes[1].bytes,
      vPlane: image.planes[2].bytes,
      width: image.width,
      height: image.height,
      yRowStride: image.planes[0].bytesPerRow,
      uvRowStride: image.planes[1].bytesPerRow,
      uvPixelStride: image.planes[1].bytesPerPixel ?? 2,
    );
  }

  /// Runs every camera frame, independent of enroll/identify state, so the
  /// box overlay + liveness check are always live. The detect→landmark→
  /// liveness chain is cheap (BlazeFace + a small 256×256 landmark model;
  /// see doc/KR/benchmark.md) and runs on the calling isolate, matching how
  /// `FacePipeline.identify/enroll` already run detection synchronously —
  /// only embedding inference (the expensive step) goes to an isolate.
  Future<void> _onFrame(CameraImage frame) async {
    if (_busy) return; // drop frames while the previous one is still running
    final pipeline = _pipeline;
    final landmarker = _landmarker;
    if (pipeline == null || landmarker == null) return;

    _busy = true;
    try {
      final image = _toFaceImage(frame);
      if (image == null) return;

      final faces = await pipeline.detector.detect(image);
      final face = faces.isEmpty
          ? null
          : faces.reduce((a, b) => a.score >= b.score ? a : b);

      if (face == null) {
        _liveness.reset();
        setState(() {
          _overlayFace = null;
          _overlayLandmarks = null;
          _livenessState = LivenessState.pending;
          _matchLabel = null;
        });
        if (_identifying) _setStatus('얼굴 없음');
        return;
      }

      final landmarks = await landmarker.detectLandmarks(image, face);
      final liveness = landmarks == null
          ? const LivenessResult(state: LivenessState.pending)
          : _liveness.update(landmarks, DateTime.now().millisecondsSinceEpoch);

      setState(() {
        _overlayFace = face;
        _overlayLandmarks = landmarks;
        _overlayImageSize = Size(image.width.toDouble(), image.height.toDouble());
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
        if (embedding == null) {
          _setStatus('얼굴을 찾지 못했어요. 카메라에 얼굴이 잘 보이게 해주세요.');
        } else {
          _gallery.add(Enrollment(id: enrollName, embedding: embedding));
          _setStatus('"$enrollName" 등록 완료 (총 ${_gallery.length}명)');
        }
      } else if (_identifying) {
        final result = await pipeline.identify(image, _gallery);
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
      _busy = false;
    }
  }

  void _setStatus(String text) {
    if (!mounted) return;
    setState(() => _status = text);
  }

  @override
  void dispose() {
    _controller?.dispose();
    _landmarker?.dispose();
    _nameController.dispose();
    super.dispose();
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
                        quarterTurns: quarterTurnsForController(controller),
                        mirror: _lensDirection == CameraLensDirection.front,
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
              ],
            ),
          ),
        ],
      ),
    );
  }
}
