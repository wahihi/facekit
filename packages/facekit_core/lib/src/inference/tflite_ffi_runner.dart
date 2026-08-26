// Pure-Dart dart:ffi implementation of InferenceRunner, calling
// libtensorflowlite_c directly — no Flutter dependency. Counterpart to
// the outer facekit package's lib/src/inference/tflite_runner.dart
// (Flutter/tflite_flutter-based, used on Android).
//
// Float32 tensors only — every model this SDK ships is float32 in/out
// (see prepareInputTensor/zeroTensor call sites). Extend if a future
// model needs quantized (int8/uint8) I/O.
//
// The load/allocate/invoke/copy sequence below mirrors the public shape
// of tflite_flutter's Model/Interpreter/Tensor classes (Apache 2.0) as a
// design reference, but is independent code written for this file — the
// only vendored-verbatim pieces are bindings/tensorflow_lite_bindings_
// generated.dart and the dlopen logic in bindings/bindings.dart. See
// ../../../third_party/tflite_flutter/NOTICE.md.

import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../core/contracts.dart';
import 'bindings/bindings.dart';
import 'bindings/tensorflow_lite_bindings_generated.dart';

class TfliteFfiRunner implements InferenceRunner {
  final TensorFlowLiteBindings _bindings;
  final Pointer<TfLiteModel> _model;
  final Pointer<TfLiteInterpreter> _interpreter;

  TfliteFfiRunner._(this._bindings, this._model, this._interpreter);

  /// Loads a model from an absolute file-system path.
  ///
  /// [libraryPath] overrides libtensorflowlite_c discovery (see
  /// bindings/bindings.dart's openTfliteLibrary); [numThreads] sets the
  /// CPU thread count used for inference.
  static TfliteFfiRunner fromFile(
    String modelPath, {
    String? libraryPath,
    int numThreads = 1,
  }) {
    final bindings = tfliteBinding(explicitPath: libraryPath);

    final cPath = modelPath.toNativeUtf8();
    final Pointer<TfLiteModel> model;
    try {
      model = bindings.TfLiteModelCreateFromFile(cPath.cast());
    } finally {
      calloc.free(cPath);
    }
    if (model == nullptr) {
      throw StateError('Unable to load TFLite model from $modelPath');
    }

    final options = bindings.TfLiteInterpreterOptionsCreate();
    bindings.TfLiteInterpreterOptionsSetNumThreads(options, numThreads);
    final interpreter = bindings.TfLiteInterpreterCreate(model, options);
    bindings.TfLiteInterpreterOptionsDelete(options);
    if (interpreter == nullptr) {
      bindings.TfLiteModelDelete(model);
      throw StateError('Unable to create TFLite interpreter for $modelPath');
    }

    if (bindings.TfLiteInterpreterAllocateTensors(interpreter) !=
        TfLiteStatus.kTfLiteOk) {
      bindings.TfLiteInterpreterDelete(interpreter);
      bindings.TfLiteModelDelete(model);
      throw StateError('Failed to allocate tensors for $modelPath');
    }

    return TfliteFfiRunner._(bindings, model, interpreter);
  }

  Pointer<TfLiteTensor> _inputTensor(int index) =>
      _bindings.TfLiteInterpreterGetInputTensor(_interpreter, index);

  Pointer<TfLiteTensor> _outputTensor(int index) =>
      _bindings.TfLiteInterpreterGetOutputTensor(_interpreter, index);

  List<int> _tensorShape(Pointer<TfLiteTensor> tensor) => List.generate(
        _bindings.TfLiteTensorNumDims(tensor),
        (i) => _bindings.TfLiteTensorDim(tensor, i),
      );

  @override
  List<int> get inputShape => _tensorShape(_inputTensor(0));

  @override
  List<int> outputShape(int index) => _tensorShape(_outputTensor(index));

  @override
  int get outputCount =>
      _bindings.TfLiteInterpreterGetOutputTensorCount(_interpreter);

  @override
  void run(Object input, Object output) {
    runForMultipleOutputs(input, {0: output});
  }

  @override
  void runForMultipleOutputs(Object input, Map<int, Object> outputs) {
    _writeInput(0, input);
    if (_bindings.TfLiteInterpreterInvoke(_interpreter) !=
        TfLiteStatus.kTfLiteOk) {
      throw StateError('TFLite invoke failed');
    }
    outputs.forEach(_readOutput);
  }

  void _writeInput(int index, Object nested) {
    final tensor = _inputTensor(index);
    final n = _numElements(_tensorShape(tensor));
    final buf = calloc<Float>(n);
    try {
      final view = buf.asTypedList(n);
      _flattenInto(nested, view, 0);
      final status = _bindings.TfLiteTensorCopyFromBuffer(
        tensor,
        buf.cast<Void>(),
        n * sizeOf<Float>(),
      );
      if (status != TfLiteStatus.kTfLiteOk) {
        throw StateError('TfLiteTensorCopyFromBuffer failed for input $index');
      }
    } finally {
      calloc.free(buf);
    }
  }

  void _readOutput(int index, Object dst) {
    final tensor = _outputTensor(index);
    final n = _numElements(_tensorShape(tensor));
    final buf = calloc<Float>(n);
    try {
      final status = _bindings.TfLiteTensorCopyToBuffer(
        tensor,
        buf.cast<Void>(),
        n * sizeOf<Float>(),
      );
      if (status != TfLiteStatus.kTfLiteOk) {
        throw StateError('TfLiteTensorCopyToBuffer failed for output $index');
      }
      _fillNested(dst as List, buf.asTypedList(n), 0);
    } finally {
      calloc.free(buf);
    }
  }

  @override
  void close() {
    _bindings.TfLiteInterpreterDelete(_interpreter);
    _bindings.TfLiteModelDelete(_model);
  }
}

int _numElements(List<int> shape) => shape.fold(1, (a, b) => a * b);

/// Flattens an arbitrarily-nested num List (e.g. [1,H,W,3]) into [out],
/// starting at [cursor]. Returns the cursor position after the last
/// element written.
int _flattenInto(Object node, Float32List out, int cursor) {
  if (node is List) {
    for (final child in node) {
      cursor = _flattenInto(child, out, cursor);
    }
    return cursor;
  }
  out[cursor] = (node as num).toDouble();
  return cursor + 1;
}

/// Inverse of [_flattenInto]: fills the leaf (1-D) lists of an existing
/// nested List structure [node] with values from [flat], starting at
/// [cursor]. [node] must already be shaped correctly (e.g. via
/// `zeroTensor`) — only leaf slots are assigned, matching the in-place-
/// mutation contract InferenceRunner.runForMultipleOutputs callers rely on.
int _fillNested(List node, Float32List flat, int cursor) {
  if (node.isEmpty) return cursor;
  if (node.first is List) {
    for (var i = 0; i < node.length; i++) {
      cursor = _fillNested(node[i] as List, flat, cursor);
    }
  } else {
    for (var i = 0; i < node.length; i++) {
      node[i] = flat[cursor++];
    }
  }
  return cursor;
}
