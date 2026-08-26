/// facekit_core — Flutter-free face-recognition pipeline.
/// Shared by the facekit Flutter SDK and facekit_cli (Linux headless).
library facekit_core;

export 'src/core/models.dart';
export 'src/core/contracts.dart';
export 'src/core/math.dart';

export 'src/image/image_converter.dart';

export 'src/alignment/affine_aligner.dart';

export 'src/matching/cosine_matcher.dart';

export 'src/pipeline/face_pipeline.dart';

export 'src/inference/model_manifest.dart';
export 'src/inference/tensor_utils.dart';
export 'src/inference/tflite_ffi_runner.dart';

export 'src/detection/blazeface_anchors.dart';
export 'src/detection/blazeface_decoder.dart';
export 'src/detection/blazeface_detector.dart';
export 'src/detection/yunet_decoder.dart';
export 'src/detection/yunet_detector.dart';

export 'src/embedding/adapters/embedder_adapter.dart';
export 'src/embedding/adapters/arcface_adapter.dart';
export 'src/embedding/adapters/facenet_adapter.dart';
export 'src/embedding/face_embedder.dart';

export 'src/landmark/face_landmarker.dart';

export 'src/liveness/blink_liveness_detector.dart';
