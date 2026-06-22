// Thin wrapper around tflite_flutter's Interpreter.
// Keeps tflite_flutter confined to this file — all other layers depend on
// TfliteRunner, not on tflite_flutter directly.
// Source: tflite_flutter 0.12.x public API (Apache 2.0).

import 'dart:io' show File;
import 'dart:typed_data';
import 'package:tflite_flutter/tflite_flutter.dart';

class TfliteRunner {
  final Interpreter _interpreter;

  TfliteRunner._(this._interpreter);

  /// Loads a model from a Flutter asset path (e.g. 'assets/models/foo/bar.tflite').
  static Future<TfliteRunner> fromAsset(String assetPath) async {
    // Interpreter.fromAsset is async in 0.12.x (uses rootBundle).
    final interpreter = await Interpreter.fromAsset(assetPath);
    return TfliteRunner._(interpreter);
  }

  /// Loads a model from an absolute file-system path.
  /// Useful in tests and non-Flutter Dart contexts.
  /// Note: Interpreter.fromFile is synchronous in 0.12.x.
  static Future<TfliteRunner> fromFile(String filePath) async {
    final interpreter = Interpreter.fromFile(File(filePath));
    return TfliteRunner._(interpreter);
  }

  /// Shape of the first input tensor, e.g. [1, 128, 128, 3].
  List<int> get inputShape => _interpreter.getInputTensors().first.shape;

  /// Shape of the i-th output tensor.
  List<int> outputShape(int index) => _interpreter.getOutputTensors()[index].shape;

  /// Number of output tensors.
  int get outputCount => _interpreter.getOutputTensors().length;

  /// Runs the model with a single input and single output.
  void run(Object input, Object output) {
    _interpreter.run(input, output);
  }

  /// Runs the model with a single input and multiple outputs.
  /// [outputs] maps output-tensor-index → pre-allocated buffer.
  void runForMultipleOutputs(Object input, Map<int, Object> outputs) {
    _interpreter.runForMultipleInputs([input], outputs);
  }

  void close() => _interpreter.close();
}

// ── input preparation utilities ───────────────────────────────────────────────

/// Converts an RGB888 byte array into a float32 NHWC tensor normalised by
/// (pixel - mean) / std, matching the manifest InputSpec.
///
/// Returns a [List] shaped [1, height, width, 3] for use as tflite input.
List<List<List<List<double>>>> prepareInputTensor({
  required Uint8List rgbBytes,
  required int width,
  required int height,
  required List<double> mean,
  required List<double> std,
}) {
  return List.generate(
    1,
    (_) => List.generate(
      height,
      (row) => List.generate(
        width,
        (col) {
          final idx = (row * width + col) * 3;
          return [
            (rgbBytes[idx]     - mean[0]) / std[0],
            (rgbBytes[idx + 1] - mean[1]) / std[1],
            (rgbBytes[idx + 2] - mean[2]) / std[2],
          ];
        },
      ),
    ),
  );
}
