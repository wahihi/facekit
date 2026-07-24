🇰🇷 [한국어 원문](../KR/adaface_verification.md)

---

# Verifying AdaFace against real weights (2026-06-25)

Embeddings were pulled from real ArcFace (`arcface_buffalo_l`/w600k_r50) and
AdaFace (`adaface_ir101_webface12m`) weights, genuine/impostor pair cosine
similarities were compared, and both manifests' `matching.threshold` values
were updated from the result. The script is
[tool/model_verification/compare_arcface_adaface.py](../../tool/model_verification/compare_arcface_adaface.py).

## Method

- **Dataset**: [huggingface.co/datasets/logasja/lfw](https://huggingface.co/datasets/logasja/lfw),
  `pairs/test` split (LFW deepfunneled, 250×250). A random sample of 100
  genuine pairs + 100 impostor pairs (seed=42).
- **Preprocessing**: only a 250×250 → 112×112 resize was applied (no 5-point
  landmark alignment — the LFW funneled crop is already roughly centered,
  but this still differs from the actual SDK's `AffineAligner` output).
- **Degraded-quality condition**: 112×112 → 24×24 downsample → 112×112
  upsample, as a proxy for low resolution / long-distance capture (a single
  proxy — it does not cover blur, compression, or pose variation).
- **Model execution**: ArcFace was run via the TFLite Interpreter on
  `example/assets/models/arcface_buffalo_l/w600k_r50.tflite`; AdaFace was run
  directly via onnxruntime on a checkpoint→ONNX conversion (the TFLite
  conversion OOM'd on this machine — see "TFLite conversion completed"
  below). Both are genuine, publicly released weights; the weight files
  themselves are not committed to the repo per license (BYOM).
- **AdaFace output**: the ONNX graph outputs two tensors — `feature`
  (L2-normalized, 512-d) and `norm` (the pre-normalization L2 norm scalar,
  a quality signal used for AdaFace's adaptive margin during training).
  Only `feature` is used for matching. **This two-output structure is what
  actually broke the Dart SDK's `TfliteFaceEmbedder.embed()`, which assumed
  a single output** — fixed separately (see below).
- **Threshold**: the cosine-similarity ROC point where FPR=FNR (EER) was
  adopted as the threshold.

## Results

| Model | Condition | Genuine mean | Impostor mean | EER | EER threshold | Accuracy at that threshold |
|---|---|---|---|---|---|---|
| ArcFace | clean | 0.4428 | 0.1134 | 8.5% | 0.263 | 91.5% |
| ArcFace | degraded | 0.4392 | 0.2542 | 25.0% | 0.333 | (70.0% if the clean threshold is reused) |
| AdaFace | clean | 0.4617 | 0.0527 | 2.0% | 0.211 | 98.0% |
| AdaFace | degraded | 0.3938 | 0.1145 | 14.0% | 0.223 | (83.5% if the clean threshold is reused) |

**Key finding**: in this small-scale measurement, AdaFace beat ArcFace on
(1) a lower EER even under clean conditions (2.0% vs 8.5%), (2) a smaller
EER degradation under low-quality conditions (8.5%→14.0%, +5.5pt, vs
ArcFace's 8.5%→25.0%, +16.5pt), and (3) a smaller accuracy drop when the
clean-condition threshold is reused as-is under degraded quality
(91.5%→83.5%, -8pt, vs ArcFace's 91.5%→70.0%, -21.5pt). In particular, the
fact that **the optimal threshold itself barely moves with image quality**
(AdaFace 0.211→0.223, +0.012, vs ArcFace 0.263→0.333, +0.070) matters in
practice for a mobile-app scenario that has to serve a range of capture
qualities with a single fixed threshold.

## Limitations (not a rigorous, paper-grade measurement)

- The sample (100+100 pairs) is small — confidence intervals are wide.
- Measured without 5-point landmark alignment (resize only) — absolute
  numbers may run lower than the real pipeline's. Since the same condition
  was applied to both models, the relative comparison is still considered
  valid.
- "Low quality" is a single proxy (24px down/upsample) only; real-world
  low light, blur, compression, and off-axis pose aren't covered.
- AdaFace was run as `.onnx`; full end-to-end verification in the `.tflite`
  format the real SDK uses hadn't been completed yet at this point (see
  below).

## TFLite conversion completed and verified end-to-end (added 2026-06-26)

The `adaface_ir101_webface12m.onnx` → `.tflite` conversion (onnx2tf) was
retried on a machine with more RAM (27GB) and succeeded (confirming the
earlier machine's OOM was simply a RAM shortage; the conversion itself was
light work once a few onnx2tf/numpy/tf_keras version-compatibility issues
were patched). ArcFace's `w600k_r50.onnx` (a public InsightFace buffalo_l
mirror) was also freshly converted to `.tflite` on the same machine.

Opening the converted `.tflite` directly in an interpreter confirmed the
AdaFace graph has, as expected, **two outputs** (`feature` [1,512], `norm`
[1,1]) — reconfirming that the multi-output fix to
`TfliteFaceEmbedder.embed()` really was necessary.

**Dart SDK end-to-end verification**: the ArcFace/AdaFace tests in
`test/embedding/face_embedder_smoke_test.dart` and
`test/pipeline/face_pipeline_smoke_test.dart` were run against the real
`.tflite` files and all passed (the full path: `TfliteFaceEmbedder.fromFile`
→ adapter selection → inference → 512-d L2-normalized embedding). Until now,
the adapter logic had only been indirectly verified by reimplementing it in
Python (onnxruntime); this time it was directly proven that **the Dart
adapter code itself runs against real weights without crashing.** (The test
file's model path, which had been hardcoded to one developer's account, was
generalized to use `Platform.environment['HOME']`.)

**Confirming ONNX ↔ TFLite result parity**: a `--adaface-tflite` option was
added to `compare_arcface_adaface.py`, and re-measuring the same 200 LFW
pairs against `.tflite`
(`results_tflite_2026-06-26.json`) showed AdaFace's EER/threshold
essentially matching the `.onnx`-based measurement
(`results_2026-06-25.json`), differing only in the 5th–6th decimal place
(noise from a different float32 execution path) — clean EER 2.0%/threshold
0.2106 vs 0.2106, degraded EER 14.0%/threshold 0.2234 vs 0.2234. In other
words, the conclusions in the "Results" section above and the manifest
threshold values remain valid in the actual deployed format (`.tflite`)
too.

## Dart SDK fix: multi-output embedder bug

`TfliteFaceEmbedder.embed()` assumed a single output tensor
(`_runner.run(input, output)`); loading a real model with two outputs
(`feature`, `norm`), as AdaFace has, made tflite_flutter's
`runForMultipleInputs` crash unconditionally on a null-assertion failure for
the unfilled second output slot. A `zeroTensor()` helper was added to
`lib/src/inference/tflite_runner.dart`, and `embed()` in
`lib/src/embedding/face_embedder.dart` was changed to use only output 0
(the embedding) and fill the remaining outputs with a discarded buffer.
Unit test: `test/inference/tflite_runner_test.dart`.
