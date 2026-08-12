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

## AuraFace EER measurement (added 2026-08-09)

AuraFace (glintr100), the default bundled embedding model, had until now
only ever had genuine-pair similarity checked (see the 2026-07-24 and
2026-08-03 postmortems) — impostor pairs had never been measured, and
`matching.threshold` was an unvalidated placeholder (0.40). A real-device
session reproduced a concrete failure: showing the camera an unrelated
person's photo scored almost the same similarity (both ~0.9) as the
enrolled user. Tracing the cause led to a wrong value in `manifest.json`'s
`input.normalize` — pixels were being fed in with no normalization at all,
instead of being normalized. Loading the same `glintr100.onnx` through the
*official* `insightface` Python package
(`tool/model_verification/verify_auraface_official.py`) showed that
`ArcFaceONNX`'s own graph-inspection logic auto-selects
`input_mean=127.5, input_std=127.5` (the standard ArcFace convention) for
this model — not raw pixels. The wrong raw-pixel value was likely first
adopted because, at the time it was chosen (2026-07-24), the alignment
pipeline still had unfixed bugs (reversed eye-left/right indices, an SVD
sign bug, etc. — all fixed only on 2026-08-03), so the crops being compared
were themselves mis-rotated; feeding those bad crops through *correct*
normalization produced low similarity, which got misdiagnosed as a
normalization problem. Nobody re-validated the normalization choice after
the alignment bugs were actually fixed a month later, so the wrong raw-pixel
value silently persisted.

After correcting `input.normalize` to the standard values (mean/std 127.5),
the model was re-measured with the same methodology already used for
ArcFace/AdaFace (`compare_arcface_adaface.py --auraface-tflite`, the same
200 LFW pairs, seed=42, resize-only preprocessing with no 5-point
alignment):

| Model | Condition | Genuine mean | Impostor mean | EER | EER threshold | Accuracy at that threshold |
|---|---|---|---|---|---|---|
| AuraFace | clean | 0.4803 | 0.1952 | 10.0% | 0.300 | 90.0% |
| AuraFace | degraded | 0.3560 | 0.2402 | 28.0% | 0.274 | (68.5% if the clean threshold is reused) |

This is roughly on par with ArcFace (8.5% EER) and clearly behind AdaFace
(2.0% EER) — AuraFace turned out to have the weakest discriminative power of
the three. An earlier quick, ad-hoc test (2 photos of one person + 1 of
someone else, via `verify_auraface_official.py`) had shown a dramatic split
(genuine 0.77 vs impostor ≈0.03), but that was a 3-sample fluke; the 200-pair
statistic here is the number worth trusting. `matching.threshold` was
updated from 0.40 (an unvalidated placeholder) to 0.30 (the measured EER
point).

**Lesson**: not just thresholds — any setting that looks like it "only has
one correct value" (like `input.normalize`) can get stuck wrong if a
different bug (here, alignment) was tangled up with it at the moment the
value was first chosen. Re-validating a value later means also
re-questioning whatever assumption was true (or wasn't) back when it was
set — here, "the crop is actually upright."

## Re-measuring with alignment, and a real-device-based threshold (added 2026-08-11)

The AuraFace EER (10.0%) above used the same methodology as ArcFace/AdaFace
— **only resizing** each 250x250 LFW funneled crop to 112x112, with no
5-point alignment applied (every `threshold_note` says so explicitly).
facekit's real pipeline does run detection + 5-point alignment, so this
checked whether applying alignment changes the numbers.

Rather than porting facekit's own BlazeFace decoder + Umeyama solver to
Python (deferred as a separate, larger task), this reused the official
`insightface` package's SCRFD detector + `face_align.norm_crop()`
(`tool/model_verification/compare_with_alignment.py`). `norm_crop`'s
reference points (`arcface_dst`) are identical to facekit's own
`arcface112Ref` down to the decimal, so this measures "the same target
geometry, via a different but standard detector+solver."

| Model | Condition | Genuine mean | Impostor mean | EER | EER threshold | Accuracy at that threshold |
|---|---|---|---|---|---|---|
| ArcFace | clean (aligned) | 0.6550 | 0.0108 | 3.0% | 0.156 | 97.0% |
| ArcFace | degraded (aligned) | 0.5234 | 0.0186 | 5.0% | 0.127 | (96.0% if the clean threshold is reused) |
| AuraFace | clean (aligned) | 0.5926 | 0.0577 | 6.5% | 0.137 | 93.5% |
| AuraFace | degraded (aligned) | 0.4205 | 0.1009 | 8.0% | 0.215 | (82.0% if the clean threshold is reused) |

Both models improved substantially over the unaligned numbers (ArcFace
clean EER 8.5%→3.0%, degraded 25%→5%; AuraFace clean 10.0%→6.5%, degraded
28%→8%) — confirming the "Limitations" section's prediction above that
unaligned numbers are pessimistic relative to the real pipeline. But
**the relative ranking didn't change** — AuraFace is still weaker than
ArcFace even with alignment applied (roughly 2.2x worse at clean EER).
Skipping alignment wasn't uniquely unfair to AuraFace; it affected both
models roughly proportionally.

**`matching.threshold` was not switched straight to this EER point
(0.137).** SCRFD appears to produce more stable landmarks than the
BlazeFace-short detector facekit actually uses on-device — BlazeFace is
already known to be unstable enough that 28-42% of real-device frames get
flagged for anomalous rotation
(`doc/EN/postmortem/2026-08-03-affine-aligner-alignment.md`). So 0.137 may
be an optimistic lower bound relative to facekit's real operating
conditions. Instead, the update was grounded in **values actually observed
on a real device**: after the `input.normalize` fix, 3 of 11 real-device
genuine-similarity readings (0.285/0.288/0.291) were rejected just under
the old threshold (0.30), while 4 impostor (photo) readings in the same
session ranged 0.060-0.094. `matching.threshold` was lowered **0.30 →
0.25** to pass all three of those genuine readings while keeping a 0.156
margin above the highest impostor value actually observed on-device.

## Giving up on threshold tuning in favor of a pose gate (2026-08-11, continued)

After lowering to 0.25, two more real-device rounds followed (impostor
subjects: Elon Musk, 73 readings; Jensen Huang, 60 readings). As the
sample grew, the observed genuine/impostor gap kept shrinking:

| Sample size | Genuine/impostor gap |
|---|---|
| 4 | 0.156 |
| 77 | 0.036 |
| 137 (cumulative) | 0.022 |

Thresholds of 0.25/0.27 "passed" at each point in time (e.g. 0.27 sat
0.021 above the highest impostor reading, 0.249, but only 0.001 below the
lowest genuine reading, 0.271) — but the gap shrinking every time the
sample grew wasn't a sign of "this value is safe," it was a sign of "the
worst case hasn't shown up yet." Chasing the threshold with more
real-device data was judged to be a losing game.

**Focus shifted back to the root cause.** Genuine similarity swinging
widely frame to frame (0.27-0.59) looked like a direct consequence of the
BlazeFace landmark instability already confirmed above (24-42% of frames
flagged for anomalous rotation). So instead of continuing to lower the
threshold, **a pose gate was added to `AffineAligner`**
(`lib/src/alignment/affine_aligner.dart`) — if the fitted transform's
rotation exceeds ±45° or its scale falls outside `[0.05, 3.0]`, `align()`
returns `null` instead of an `AlignedFace`, and the pipeline treats that
exactly like "no face detected" (bounds match what
`tool/analyze_alignment_log.py` already used — the same repo, independently
derived from the same real-device investigation). The decision logic was
split into a pure function, `_poseWithinBounds`, unit-tested via
`poseWithinBoundsForTest` (`test/alignment/affine_aligner_test.dart`);
`FaceAligner.align()`'s return type changed to `AlignedFace?`, and
`FacePipeline`/`SyncFacePipeline`/the benchmark harness were all updated to
handle the null case.

Since the pose gate addresses the root cause (alignment instability)
directly, `matching.threshold` was **fixed back at 0.30 (the LFW-200-pair
EER value)** — real-device-observation-driven threshold tuning stops here.
Re-validating the pose gate itself on a real device is left as a follow-up.

## Real-device pose-gate validation, and a backlighting discovery (2026-08-11, continued)

With the pose gate on, a real-device round (self vs. Mike Tyson,
`FACEKIT_VERBOSE_DEBUG=true`) gave:

```
107 align attempts:  48 passed (45%) / 59 rejected (55%)
Rejection cause: all 59 were rotation violations (0 scale violations), rotation 45.5-153.4 deg
Of the passed frames — self: 6/6 accepted (0.318-0.430); Mike Tyson: 40/40 rejected (0.078-0.270)
Gap = 0.048 (up from 0.022 measured without the gate)
```

The gate worked exactly as designed (only rotation violations, zero false
scale triggers) — but **55% of alignment attempts being thrown away**
means, in practice, "no face" showing on screen while a face is clearly
visible more than half the time, which was a real concern.

**Chasing the cause turned up an environmental variable: backlighting.**
During this test the light source was behind the subject's head; the
camera's auto-exposure balanced for the bright background, leaving the
face relatively underexposed (near-silhouette), which likely degraded
landmark precision and widened the rotation error. Physically blocking the
light source with the head (normalizing face exposure) produced a clearly
better felt recognition rate on the same device (not yet logged
quantitatively). In other words, a meaningful part of what's been treated
as "BlazeFace's landmarks are just inherently unstable" may actually be
explained by capture conditions (backlighting) rather than a pure
algorithmic limit — the 55% rejection rate above may be a worst-case number
specific to backlit conditions.

**Conclusion**: the pose gate is adopted (it correctly filters rotation
false-positives, and the genuine/impostor gap on the frames that pass did
improve). Whether to further tune the rotation bound (45°) should wait for
a re-measurement under non-backlit conditions; a quantitative re-test plus
adding "avoid backlighting" capture guidance to the installation docs are
left as follow-ups.

## What about swapping the detector itself? — investigating YuNet (added 2026-08-12)

BlazeFace only gives one mouth point out of its 6 landmarks, forcing
`AffineAligner` to compromise ArcFace's true 5-point reference (which wants
2 mouth corners) down to 4 points (2 eyes + nose + mouth center). This
started from asking: if a detector natively gave ArcFace's real 5 points
(2 eyes + nose + 2 mouth corners), would that compromise become
unnecessary?

**Candidate research**: both SCRFD (InsightFace) and YuNet (OpenCV
Zoo/libfacedetection) natively support 5-point output. Their licenses were
checked directly:

- **SCRFD**: `detection/scrfd/LICENSE` is Apache 2.0, but InsightFace's
  top-level README explicitly separates code license from model license:
  "The training data containing the annotation (and the models trained
  with these data) are available for non-commercial research purposes
  only" — naming `buffalo_l` (already treated as research-only in this
  project) as a concrete example. Independently, SCRFD's training data,
  WIDER FACE, is itself confirmed CC BY-NC-ND licensed. Two independent
  lines of evidence both converge on "non-commercial" — **ruled out.**
- **YuNet**: its distributor (the OpenCV Zoo README) explicitly labels the
  model itself MIT. `CLAUDE.md:15` already said "Detection (BlazeFace
  Apache2.0 / YuNet MIT): may be bundled" — this project's own license
  policy had apparently already anticipated YuNet as an alternative.

**First Python validation
(`tool/model_verification/compare_with_yunet.py`) — EER 48-49%, essentially
random.** 5 points from `cv2.FaceDetectorYN` were fed into
`insightface.utils.face_align.norm_crop()` (the same alignment function
used for the SCRFD validation, reusing the same `arcface_dst` reference)
over the same 200 LFW pairs. Both ArcFace and AuraFace came back with
genuine and impostor means nearly identical (e.g. ArcFace genuine 0.78 vs.
impostor 0.75) — even the worst BlazeFace conditions measured so far never
collapsed this completely.

**Debugging**: raw detections (bbox, 5 points, score) looked completely
normal across several sample images (score 0.92-0.95, sane eye-nose-mouth
geometry). Reconstructing an actual aligned crop as a PNG showed a tiny
face surrounded by solid black — visualizing revealed what the numbers
alone hadn't. Computing the fitted transform directly gave a scale of
0.125 (about 1/7 of the ~0.83-0.91 expected from eye spacing alone) —
reproduced identically by both insightface's `norm_crop` and facekit's own
Umeyama solver (independently ported to Python and cross-checked), which
ruled out a library bug and pointed at the input itself.

**Root cause**: YuNet's own `right_eye`/`left_eye` (and
`right_mouth`/`left_mouth`) labels turned out to be the exact opposite
handedness from what facekit assumed — the same class of mistake as the
**BlazeFace eye-order bug** (2026-08-03 postmortem), made again. Feeding
YuNet's raw order straight into `arcface_dst` with no relabeling
immediately normalized to scale 0.906, rotation 1.46°.

**Re-measured results** (same 200 LFW pairs, identical methodology):

| Method | ArcFace EER (clean/degraded) | AuraFace EER (clean/degraded) |
|---|---|---|
| Resize only (no alignment) | 8.5% / 25.0% | 10.0% / 28.0% |
| SCRFD-aligned (ruled out on license) | 3.0% / 5.0% | 6.5% / 8.0% |
| **YuNet-aligned (after the fix)** | **2.5% / 4.0%** | **2.5% / 6.0%** |

Better than SCRFD, and **ArcFace and AuraFace's clean EER landed at exactly
the same 2.5%** — the "AuraFace is the weakest of the three" pattern that
ran through this entire session may not have been a limit of the model
itself after all, but AuraFace being unusually sensitive to alignment
quality (BlazeFace's 4-point compromise specifically).

**Lesson**: "don't trust a model author's own landmark names without
verifying them" was confirmed for the second time. Porting this to a real
Dart implementation needs the corrected left/right mapping pinned down
explicitly in code comments and a regression test, or the same mistake is
easy to repeat. Full debugging write-up:
[doc/EN/postmortem/2026-08-12-yunet-landmark-order.md](postmortem/2026-08-12-yunet-landmark-order.md).
