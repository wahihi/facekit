// Pipeline integration smoke test.
//
// Validates that FacePipeline correctly wires detect→align→embed→match
// together, and specifically that the Isolate.run()-based embedding step
// works when the embedder wraps a native TFLite interpreter. This was an
// open architectural risk from code review: Dart isolates do not share
// heap, so sending a closure that captures an FFI-backed Interpreter across
// isolates needed to be verified empirically rather than assumed correct.
//
// Uses a fixed/fake FaceDetector (not real BlazeFace) so the test doesn't
// depend on a real face photo — alignment, embedding (isolate + real ArcFace
// model) and matching are all real.
//
// Run with: flutter test test/pipeline/face_pipeline_smoke_test.dart
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:facekit/src/core/contracts.dart';
import 'package:facekit/src/core/models.dart';
import 'package:facekit/src/alignment/affine_aligner.dart';
import 'package:facekit/src/embedding/face_embedder.dart';
import 'package:facekit/src/inference/model_manifest.dart';
import 'package:facekit/src/matching/cosine_matcher.dart';
import 'package:facekit/src/pipeline/face_pipeline.dart';

class _FixedFaceDetector implements FaceDetector {
  final DetectedFace face;
  const _FixedFaceDetector(this.face);

  @override
  Future<List<DetectedFace>> detect(FaceImage image) async => [face];
}

FaceImage _solidImage(int w, int h, int r, int g, int b) {
  final bytes = Uint8List(w * h * 3);
  for (int i = 0; i < w * h; i++) {
    bytes[i * 3] = r;
    bytes[i * 3 + 1] = g;
    bytes[i * 3 + 2] = b;
  }
  return FaceImage(rgbBytes: bytes, width: w, height: h);
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  const manifestPath = 'assets/models/arcface_buffalo_l/manifest.json';
  const modelPath =
      '/home/wahihi/development/models/w600k_r50_tf/w600k_r50_float32.tflite';
  final modelExists = File(modelPath).existsSync();

  test(
    'FacePipeline.enroll/identify run detect→align→embed(isolate)→match end-to-end',
    () async {
      final manifest = ModelManifest.fromJsonString(
        File(manifestPath).readAsStringSync(),
      );

      final embedder = await TfliteFaceEmbedder.fromFile(
        tflitePath: modelPath,
        manifest: manifest,
      );

      // Landmarks identical to the ArcFace reference points → identity
      // transform, so AffineAligner crops a clean, undistorted 112×112 patch.
      final face = DetectedFace(
        boundingBox: const Rect(left: 0, top: 0, right: 112, bottom: 112),
        landmarks: arcface112Ref,
        score: 0.99,
      );

      final pipeline = FacePipeline(
        detector: _FixedFaceDetector(face),
        aligner: AffineAligner.arcface112(),
        embedder: embedder,
        matcher: CosineMatcher.fromManifest(manifest),
      );

      final image = _solidImage(200, 200, 180, 140, 120);

      final enrolled = await pipeline.enroll(image);
      expect(enrolled, isNotNull);
      expect(enrolled!.dim, 512);

      final gallery = [Enrollment(id: 'p1', embedding: enrolled)];
      final result = await pipeline.identify(image, gallery);

      expect(result, isNotNull);
      expect(result!.matchedId, 'p1');
      expect(result.accepted, isTrue);
      expect(result.similarity, closeTo(1.0, 1e-3));

      embedder.dispose();
    },
    skip: modelExists ? false : 'w600k_r50_float32.tflite not present locally',
  );
}
