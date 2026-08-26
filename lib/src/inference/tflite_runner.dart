// Thin wrapper around tflite_flutter's Interpreter.
// Keeps tflite_flutter confined to this file — all other layers depend on
// TfliteRunner, not on tflite_flutter directly.
// Source: tflite_flutter 0.12.x public API (Apache 2.0).

import 'dart:io' show File;
import 'package:tflite_flutter/tflite_flutter.dart';

import 'package:facekit_core/facekit_core.dart';

export 'package:facekit_core/facekit_core.dart' show zeroTensor, prepareInputTensor;

class TfliteRunner implements InferenceRunner {
  final Interpreter _interpreter;

  TfliteRunner._(this._interpreter);

  /// Loads a model from a Flutter asset path (e.g. 'assets/models/foo/bar.tflite').
  ///
  /// [useNnApi] delegates inference to Android's NNAPI (NPU/DSP/GPU, whatever
  /// the device exposes) instead of the default CPU-only reference kernels.
  /// No-op on non-Android platforms. Off by default — callers should turn it
  /// on deliberately, since not every device has a working NNAPI driver.
  static Future<TfliteRunner> fromAsset(
    String assetPath, {
    bool useNnApi = false,
  }) async {
    // Interpreter.fromAsset is async in 0.12.x (uses rootBundle).
    final options = useNnApi
        ? (InterpreterOptions()..useNnApiForAndroid = true)
        : null;
    final interpreter = await Interpreter.fromAsset(assetPath, options: options);
    return TfliteRunner._(interpreter);
  }

  /// Loads a model from an absolute file-system path.
  /// Useful in tests and non-Flutter Dart contexts.
  /// Note: Interpreter.fromFile is synchronous in 0.12.x.
  static Future<TfliteRunner> fromFile(
    String filePath, {
    bool useNnApi = false,
  }) async {
    final options = useNnApi
        ? (InterpreterOptions()..useNnApiForAndroid = true)
        : null;
    final interpreter = Interpreter.fromFile(File(filePath), options: options);
    return TfliteRunner._(interpreter);
  }

  /// Shape of the first input tensor, e.g. [1, 128, 128, 3].
  @override
  List<int> get inputShape => _interpreter.getInputTensors().first.shape;

  /// Shape of the i-th output tensor.
  @override
  List<int> outputShape(int index) => _interpreter.getOutputTensors()[index].shape;

  /// Number of output tensors.
  @override
  int get outputCount => _interpreter.getOutputTensors().length;

  /// Runs the model with a single input and single output.
  @override
  void run(Object input, Object output) {
    _interpreter.run(input, output);
  }

  /// Runs the model with a single input and multiple outputs.
  /// [outputs] maps output-tensor-index → pre-allocated buffer.
  @override
  void runForMultipleOutputs(Object input, Map<int, Object> outputs) {
    _interpreter.runForMultipleInputs([input], outputs);
  }

  @override
  void close() => _interpreter.close();
}
