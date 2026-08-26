// Adapted from tflite_flutter 0.12.1's lib/src/bindings/bindings.dart
// (Apache 2.0, Copyright 2023 The TensorFlow Authors — see
// ../../../../third_party/tflite_flutter/). Modified from the original:
// the original resolves libtensorflowlite_c.so relative to the Flutter
// engine's own cache layout (only meaningful inside a Flutter app); this
// version has no Flutter SDK to anchor to, so the caller supplies an
// explicit path (or FACEKIT_TFLITE_C_LIBRARY env var, or a same-directory
// default) instead. See tflite_ffi_runner.dart for how this is used.

import 'dart:ffi';
import 'dart:io';

import 'tensorflow_lite_bindings_generated.dart';

/// Locates and opens `libtensorflowlite_c`, in priority order:
/// 1. [explicitPath], if given.
/// 2. the `FACEKIT_TFLITE_C_LIBRARY` environment variable.
/// 3. a platform-named copy next to the running executable
///    (e.g. `libtensorflowlite_c-linux.so` on Linux).
DynamicLibrary openTfliteLibrary({String? explicitPath}) {
  final path = explicitPath ??
      Platform.environment['FACEKIT_TFLITE_C_LIBRARY'] ??
      _defaultLibraryPath();
  try {
    return DynamicLibrary.open(path);
  } on ArgumentError catch (e) {
    throw StateError(
      'Could not load libtensorflowlite_c from "$path" ($e). Pass an '
      'explicit path to TfliteFfiRunner.fromFile(libraryPath: ...) or set '
      'the FACEKIT_TFLITE_C_LIBRARY environment variable.',
    );
  }
}

String _defaultLibraryPath() {
  final dir = File(Platform.resolvedExecutable).parent.path;
  if (Platform.isLinux) return '$dir/libtensorflowlite_c-linux.so';
  if (Platform.isMacOS) return '$dir/libtensorflowlite_c-mac.dylib';
  if (Platform.isWindows) return '$dir/libtensorflowlite_c-win.dll';
  throw UnsupportedError(
    'No default libtensorflowlite_c path for ${Platform.operatingSystem}; '
    'pass an explicit path.',
  );
}

/// Bindings for the loaded `libtensorflowlite_c`, built lazily on first
/// use so opening the library only happens once a runner is actually
/// constructed (not at import time).
TensorFlowLiteBindings? _cached;

TensorFlowLiteBindings tfliteBinding({String? explicitPath}) {
  return _cached ??= TensorFlowLiteBindings(
    openTfliteLibrary(explicitPath: explicitPath),
  );
}
