// Flutter-asset-loading convenience wrappers around facekit_core's
// detector/embedder/landmarker classes. These exist only because loading
// a model from a Flutter asset bundle (rootBundle) is inherently
// Flutter-specific — facekit_core only knows how to load from a plain
// file-system path (TfliteFfiRunner.fromFile) or an already-built
// InferenceRunner. Each function here builds a TfliteRunner (this
// package's tflite_flutter-based InferenceRunner impl) from the asset,
// then hands it to the matching facekit_core class's `.withRunner(...)`.
//
// example/lib/main.dart uses these; facekit_cli (Linux, no Flutter asset
// bundle) uses facekit_core's own `.fromFile(...)` instead.

import 'package:flutter/foundation.dart' show kReleaseMode;

import 'package:facekit_core/facekit_core.dart';

import 'inference/tflite_runner.dart';

Future<YuNetDetector> loadYuNetDetectorFromAsset({
  required String tfliteAssetPath,
  required ModelManifest manifest,
  bool useNnApi = false,
}) async {
  manifest.validate();
  manifest.assertLoadable(isReleaseBuild: kReleaseMode);
  final runner = await TfliteRunner.fromAsset(tfliteAssetPath, useNnApi: useNnApi);
  return YuNetDetector.withRunner(runner: runner, manifest: manifest);
}

Future<BlazeFaceDetector> loadBlazeFaceDetectorFromAsset({
  required String tfliteAssetPath,
  required ModelManifest manifest,
  bool useNnApi = false,
}) async {
  manifest.validate();
  manifest.assertLoadable(isReleaseBuild: kReleaseMode);
  final runner = await TfliteRunner.fromAsset(tfliteAssetPath, useNnApi: useNnApi);
  return BlazeFaceDetector.withRunner(runner: runner, manifest: manifest);
}

Future<TfliteFaceEmbedder> loadFaceEmbedderFromAsset({
  required String tfliteAssetPath,
  required ModelManifest manifest,
  bool useNnApi = false,
}) async {
  manifest.validate();
  manifest.assertLoadable(isReleaseBuild: kReleaseMode);
  final runner = await TfliteRunner.fromAsset(tfliteAssetPath, useNnApi: useNnApi);
  return TfliteFaceEmbedder.withRunner(runner: runner, manifest: manifest);
}

Future<MediaPipeFaceLandmarker> loadFaceLandmarkerFromAsset({
  required String tfliteAssetPath,
  required ModelManifest manifest,
}) async {
  manifest.validate();
  manifest.assertLoadable(isReleaseBuild: kReleaseMode);
  final runner = await TfliteRunner.fromAsset(tfliteAssetPath);
  return MediaPipeFaceLandmarker.withRunner(runner: runner, manifest: manifest);
}
