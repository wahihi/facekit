// Compile-time switch for verbose pipeline-internal debug logging (aligned-
// face crop dumps, landmark logging, alignment matrix logging — see
// face_pipeline.dart, face_embedder.dart, affine_aligner.dart). Off by
// default: a `const bool` is tree-shaken by the Dart compiler, so none of
// the gated code or its string formatting ships in a normal build, exactly
// like `_kDemoMode` in the example app.
//
// Enable for a debugging session without touching source:
//   flutter run --dart-define=FACEKIT_VERBOSE_DEBUG=true
const kFacekitVerboseDebug = bool.fromEnvironment(
  'FACEKIT_VERBOSE_DEBUG',
  defaultValue: false,
);
