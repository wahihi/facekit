🇰🇷 [한국어 문서](doc/KR/README.md)

---

# facekit

[![test](https://github.com/wahihi/facekit/actions/workflows/test.yml/badge.svg)](https://github.com/wahihi/facekit/actions/workflows/test.yml)

**An on-device face recognition SDK for Flutter.** A clean-room
implementation written from public models, papers, and official docs only —
no proprietary or third-party commercial code was referenced or copied.

- Code: **Apache License 2.0** ([LICENSE](LICENSE))
- Embedding: ships with **AuraFace (Apache 2.0, commercially usable)** as
  the default, bundled model; other models can still be swapped in via the
  manifest (BYOM structure preserved) — see
  [License / Model Policy](#license--model-policy-byom) below

---

## What this SDK does

Camera (or gallery image) → face detection → alignment → embedding →
matching, the full face recognition pipeline runs entirely on-device (no
network calls).

- **Detection**: BlazeFace (MediaPipe, Apache 2.0) — bundled with the SDK,
  no separate download needed
- **Embedding**: a swappable-model architecture — ships with **AuraFace**
  (Apache 2.0) as the default, plus adapters built in for ArcFace / AdaFace
  / MobileFaceNet / FaceNet, whose weights remain BYOM (bring your own)
- **Matching**: cosine similarity, accept/reject decided by the manifest's
  threshold
- **Liveness (Free)**: blink detection (EAR) — holding up a static photo
  never passes
- **On-device only**: heavy inference (embedding) runs in a separate Dart
  isolate so it never blocks the UI

See [doc/EN/architecture.md](doc/EN/architecture.md) /
[doc/KR/architecture.md](doc/KR/architecture.md) for the full design.

## Quick start

New to Flutter/Android dev entirely? See the step-by-step
[installation guide](doc/EN/installation.md) (also
[in Korean](doc/KR/installation.md)) covering OS/hardware requirements, SDK
setup, the default AuraFace path, building, and installing on a device.

```dart
import 'package:facekit/facekit.dart';

// 1) Detector — bundled with the SDK, loads directly
final detectorManifest = ModelManifest.fromJsonString(
  await rootBundle.loadString('packages/facekit/assets/models/blazeface_short/manifest.json'),
);
final detector = await BlazeFaceDetector.fromAsset(
  tfliteAssetPath: 'packages/facekit/assets/models/blazeface_short/face_detection_short_range.tflite',
  manifest: detectorManifest,
);

// 2) Embedder — AuraFace ships by default (Apache 2.0, fetched via
//    tool/fetch_models.sh); swap the path to use a BYOM model instead
//    (see the license section below)
final embedderManifest = ModelManifest.fromJsonString(
  await rootBundle.loadString('assets/models/auraface/manifest.json'),
);
final embedder = await TfliteFaceEmbedder.fromAsset(
  tfliteAssetPath: 'assets/models/auraface/auraface_r100_fp16.tflite',
  manifest: embedderManifest,
);

// 3) Build the pipeline, then enroll/identify
final pipeline = FacePipeline(
  detector: detector,
  aligner: AffineAligner.arcface112(),
  embedder: embedder,
  matcher: CosineMatcher.fromManifest(embedderManifest),
);

final embedding = await pipeline.enroll(faceImage);            // enroll
final result = await pipeline.identify(faceImage, gallery);     // identify → MatchResult?
```

A full example app — camera integration, box overlay, liveness, and a
benchmark button — lives in [example/](example/).

## Benchmark

Measured on real devices using the example app's built-in benchmark button
(n=30, profile build). Full methodology, VM comparison, and accuracy (EER)
tables are in [doc/KR/benchmark.md](doc/KR/benchmark.md).

| Device | Mode | Detection (BlazeFace) | Embedding (ArcFace buffalo_l) | Full frame |
|---|---|---|---|---|
| Pixel 7 | CPU (default) | avg 65.1ms | avg 729.4ms | avg 795.6ms |
| Pixel 7 | NNAPI | avg 76.6ms | avg 876.2ms | avg 954.4ms |
| Galaxy S25 (SM-S931N) | CPU (default) | avg 48.0ms | avg 256.6ms | avg 305.0ms |
| Galaxy S25 (SM-S931N) | NNAPI | avg 44.7ms | avg 244.4ms | avg 289.5ms |

The default is CPU-only. On Pixel 7 (Google Tensor G2), NNAPI is slower due
to float32 fallback overhead. On Galaxy S25 (Snapdragon 8 Elite), NNAPI
averages ~15ms faster but with much higher p95 variance (386ms vs 314ms) —
CPU remains the safer default for real-time use.

Accuracy (EER on 200 LFW pairs) is 8.5% for ArcFace and 2.0% for AdaFace,
with AdaFace staying more robust under low-resolution conditions (full
numbers in [doc/KR/adaface_verification.md](doc/KR/adaface_verification.md)).

## Liveness / Free vs. Pro boundary

This repository (Free) ships **blink-detection liveness only** — holding up
a static photo or a screen capture never passes, since EAR (eye-aspect
ratio) never changes. It does **not** defend against a mask with eye holes
cut out or a video-replay attack — that kind of multi-signal defense is a
separate implementation (Pro) behind the same `LivenessDetector` interface,
and **is not included in this repository.** See
[doc/KR/liveness.md](doc/KR/liveness.md) for the full limitations.

## License / Model policy (BYOM)

All code written for this project is **Apache License 2.0**
([LICENSE](LICENSE)).

This SDK ships **AuraFace (Apache 2.0), a commercially usable embedding
model, by default** — its weight is fetched at build/dev time via
`tool/fetch_models.sh` rather than committed directly (a size concern, not
a license one). The manifest-driven adapter architecture underneath is
still BYOM-capable: any other embedding model can be swapped in by
dropping in its `manifest.json` + `.tflite`, and `.gitignore` excludes any
non-redistributable weight file from the repository:

| Model | Role | License | Bundled |
|---|---|---|---|
| BlazeFace short-range | Detection | Apache 2.0 (MediaPipe) | ✅ Yes |
| MediaPipe Face Landmarker (478-pt) | Liveness landmarks | Apache 2.0 (MediaPipe) | ✅ Yes |
| AuraFace (glintr100 / ResNet100) | Embedding (default) | Apache 2.0 (fal.ai) | ✅ Yes (fetched via `tool/fetch_models.sh`) |
| ArcFace (buffalo_l / w600k_r50) | Embedding (BYOM example) | Non-commercial research (InsightFace) | ❌ BYOM |
| AdaFace (IR-101 / WebFace12M) | Embedding | Non-commercial research (mk-minchul/AdaFace) | ❌ BYOM |
| FaceNet512 | Embedding | Non-commercial research | ❌ BYOM |
| MobileFaceNet | Embedding | Varies by distribution (verify before use) | ❌ BYOM |

BYOM models must be sourced directly from the repository listed in each
manifest's `license.source` and placed in that model's folder — see
[Appendix A](doc/EN/installation.md#appendix-a--using-other-embedding-models-byom)
of the installation guide for the full walkthrough.
`ModelManifest.assertLoadable()` actively **blocks loading any
`redistributable:false` model in a release build**, enforcing the license
boundary in code, not just in docs
([lib/src/inference/model_manifest.dart](lib/src/inference/model_manifest.dart)).

## Open source used

| Component | License | Source |
|---|---|---|
| BlazeFace short-range | Apache 2.0 | https://github.com/google/mediapipe |
| MediaPipe Face Landmarker | Apache 2.0 | https://github.com/google/mediapipe |
| tflite_flutter | Apache 2.0 | https://pub.dev/packages/tflite_flutter |
| camera (Flutter plugin) | BSD-3-Clause | https://pub.dev/packages/camera |

For the BYOM embedding models (ArcFace/AdaFace/FaceNet/MobileFaceNet), see
the [License / Model policy](#license--model-policy-byom) table above and
each model's `manifest.json`.

## AI-assisted development

This project was developed in collaboration with **Claude Code** (Anthropic)
— from architecture decisions through implementation, debugging, and
documentation — under the author's direction and review throughout. For a
concrete, unfiltered look at what that collaboration actually looks like
(including the wrong turns), see the debugging write-up in
[doc/EN/postmortem/](doc/EN/postmortem/) (also available
[in Korean](doc/KR/postmortem/)).

Commits from 2026-07-24 onward carry a `Co-Authored-By: Claude` trailer.
Earlier commits predate this disclosure practice but were developed the same
way.

## Directory layout

```
lib/src/
  core/        pure Dart data models, math, interfaces (no Flutter dependency)
  inference/   TFLite plumbing, manifest parsing/license guard
  image/       camera frame (YUV420) → RGB conversion, resize/crop
  detection/   BlazeFace
  alignment/   5-point affine alignment
  embedding/   embedding adapters (ArcFace/AdaFace/FaceNet) + manifest-driven loader
  matching/    cosine matcher
  landmark/    MediaPipe Face Landmarker (478-pt)
  liveness/    blink-based liveness
  pipeline/    detect→align→embed→match orchestration (isolate dispatch)
example/       demo app: camera integration, box overlay, liveness, benchmark button
doc/EN, doc/KR  design docs, verification records, benchmarks
```

Dependencies always flow one way: `UI/example → pipeline →
detection/alignment/embedding/matching → inference → core`. `core/` depends
on nothing above it, not even Flutter.
