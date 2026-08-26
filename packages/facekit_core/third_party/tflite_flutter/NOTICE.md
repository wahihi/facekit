# Vendored code notice

`lib/src/inference/bindings/tensorflow_lite_bindings_generated.dart` and
`lib/src/inference/bindings/bindings.dart` are adapted from
[tflite_flutter](https://github.com/tensorflow/flutter-tflite) 0.12.1
(Copyright 2023 The TensorFlow Authors), licensed under the Apache
License, Version 2.0 — see `LICENSE` in this directory. The first file is
vendored verbatim (ffigen-generated raw C API bindings, no Flutter
dependency); the second is adapted to drop Flutter-plugin-specific
library resolution. Both are marked as modified/adapted at their own
file headers per License §4(b).

`lib/src/inference/tflite_ffi_runner.dart` is original code written for
facekit_core, not vendored — it re-implements (independently, referencing
tflite_flutter's public Model/Interpreter/Tensor API shape as a guide,
not by copying their source) the load/allocate/invoke/copy sequence
those Flutter-coupled classes perform, using the bindings above.
