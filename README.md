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

- **Detection**: YuNet (OpenCV Zoo / libfacedetection, MIT) — bundled with
  the SDK, no separate download needed. It emits the **full 5-point landmark
  set (both eyes, nose, both mouth corners) natively**, so the standard
  ArcFace alignment can be used without compromise. BlazeFace (MediaPipe,
  Apache 2.0) is still bundled but only as a fallback — see
  [Why YuNet is the default detector](#why-yunet-is-the-default-detector)
- **Embedding**: a swappable-model architecture — ships with **AuraFace**
  (Apache 2.0) as the default, plus adapters built in for ArcFace / AdaFace
  / MobileFaceNet / FaceNet, whose weights remain BYOM (bring your own)
- **Matching**: cosine similarity, accept/reject decided by the manifest's
  threshold
- **Liveness (Free)**: blink detection (EAR) — **currently disabled by
  default in the example app**
  ([#4](https://github.com/wahihi/facekit/issues/4); see the liveness
  section below)
- **On-device only**: heavy inference (embedding) runs in a separate Dart
  isolate so it never blocks the UI

See [doc/EN/architecture.md](doc/EN/architecture.md) /
[doc/KR/architecture.md](doc/KR/architecture.md) for the full design.

## Why YuNet is the default detector

BlazeFace short-range was the original default. The problem was landmarks:
of the six keypoints BlazeFace emits, the mouth is a **single centre
point**, so the standard 5-point alignment that ArcFace-family embedders
assume (both eyes, nose, **both mouth corners**) could not be filled in.
`AffineAligner` compromised down to four points, and that distortion ate
directly into recognition accuracy.

The measurements bear this out. On the same 200 LFW pairs:

| Alignment | AuraFace clean EER | Threshold |
|---|---|---|
| None (resize only) | 10.0% | 0.30 |
| YuNet native 5-point | **2.5%** | **0.211** |

Every attempt to fix low recognition rates by lowering the threshold failed
(the 0.30 → 0.25 → 0.27 history is in
[doc/EN/adaface_verification.md](doc/EN/adaface_verification.md#giving-up-on-threshold-tuning-in-favor-of-a-pose-gate-2026-08-11-continued));
the real cause was that alignment had bad inputs to begin with. **Alignment
quality comes before threshold tuning** is the most expensive lesson this
project has produced.

SCRFD (InsightFace) also emits 5 points natively, but was ruled out on
licensing: separately from its Apache 2.0 detection code, the upstream
repository marks the training data — and models trained on it — as
non-commercial research only, and its training set (WIDER FACE) carries a
non-commercial licence of its own. YuNet's distributor (OpenCV Zoo) states
MIT for the weights themselves. Full write-up in the
[2026-08-12 postmortem](doc/EN/postmortem/2026-08-12-yunet-landmark-order.md).

BlazeFace was kept rather than removed (`BlazeFaceDetector`, weights still
bundled), so it can be swapped back in if YuNet misbehaves on a given
device.

## Quick start

New to Flutter/Android dev entirely? See the step-by-step
[installation guide](doc/EN/installation.md) (also
[in Korean](doc/KR/installation.md)) covering OS/hardware requirements, SDK
setup, the default AuraFace path, building, and installing on a device.

```dart
import 'package:facekit/facekit.dart';

// 1) Detector — bundled with the SDK, loads directly (YuNet, MIT)
final detectorManifest = ModelManifest.fromJsonString(
  await rootBundle.loadString('packages/facekit/assets/models/yunet_160/manifest.json'),
);
final detector = await YuNetDetector.fromAsset(
  tfliteAssetPath: 'packages/facekit/assets/models/yunet_160/yunet_160.tflite',
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
(n=30). Full methodology, VM comparison, and accuracy (EER) tables are in
[doc/EN/benchmark.md](doc/EN/benchmark.md) (also
[in Korean](doc/KR/benchmark.md)).

**AuraFace (the default embedding model)** — Pixel 7, `--release` build
(the same APK published to
[GitHub Releases](https://github.com/wahihi/facekit/releases)), two
independent runs after a reinstall:

| Device | Mode | Detection (BlazeFace) | Embedding (AuraFace) | Full frame |
|---|---|---|---|---|
| Pixel 7 | CPU (default) | avg 51–52ms | avg 1083–1186ms | avg 1135–1238ms |
| Pixel 7 | NNAPI | avg 50–61ms | avg 1106–1205ms | avg 1156–1267ms |
| Galaxy S25 (SM-S931N) | CPU (default) | avg 56.3ms | avg 598.8ms | avg 655.7ms |
| Galaxy S25 (SM-S931N) | NNAPI | avg 46.3ms | avg 480.2ms | avg 527.2ms |

The Galaxy S25 row is a **single measurement** on a borrowed device (not
the tester's own), with no re-test possible — treat the NNAPI-faster
result as a reference point, not a confirmed conclusion (see the full doc
for the caveats). The ~9–10% spread between the two Pixel 7 runs
(detection stayed stable) looks like ordinary device variance (thermal,
background load, cold state after reinstall), not a bug. AuraFace
(ResNet100) is slower than ArcFace buffalo_l (ResNet50) below, but by a
device-dependent margin (full-frame basis: ~1.5x on Pixel 7, ~2.2x on
Galaxy S25) — likely reflecting how efficiently each chip handles the
deeper backbone.

**ArcFace buffalo_l (a BYOM example, non-default)** — `--profile` build
(ArcFace's research-tier license blocks it from loading in `--release` at
all; see below):

| Device | Mode | Detection (BlazeFace) | Embedding (ArcFace buffalo_l) | Full frame |
|---|---|---|---|---|
| Pixel 7 | CPU (default) | avg 65.1ms | avg 729.4ms | avg 795.6ms |
| Pixel 7 | NNAPI | avg 76.6ms | avg 876.2ms | avg 954.4ms |
| Galaxy S25 (SM-S931N) | CPU (default) | avg 48.0ms | avg 256.6ms | avg 305.0ms |
| Galaxy S25 (SM-S931N) | NNAPI | avg 44.7ms | avg 244.4ms | avg 289.5ms |

The default is CPU-only in both cases. On Pixel 7, NNAPI's full-frame mean
is slightly slower than CPU's for both models (more so for ArcFace), and
p95 is consistently worse under NNAPI. On Galaxy S25 (Snapdragon 8 Elite),
NNAPI averages ~15ms faster than CPU for ArcFace but with much higher p95
variance (386ms vs 314ms) — CPU remains the safer default for real-time use
across the board. Build modes differ between the two tables above (see
each doc section for why), so treat the ratio between them as a rough
reference rather than a controlled comparison.

Accuracy (EER on 200 LFW pairs) is **2.5% for AuraFace** with YuNet's
native 5-point alignment applied (threshold 0.211). The older figures —
8.5% ArcFace / 2.0% AdaFace / 10.0% AuraFace — were measured **without
alignment (resize only)**, so don't read the two AuraFace numbers as a
like-for-like pair. ArcFace *was* re-measured under YuNet's native 5-point
alignment in the same Python-side validation and tied AuraFace exactly at
2.5% (see the
[2026-08-12 postmortem](doc/EN/postmortem/2026-08-12-yunet-landmark-order.md))
— but `arcface_buffalo_l`'s shipped `manifest.json` threshold hasn't been
updated to adopt that measurement yet (still `0.26`, from the unaligned
figure). AdaFace has not been re-measured under YuNet alignment at all.
AdaFace remains the most robust under low-resolution
conditions (full numbers and the AuraFace `input.normalize` bug this
uncovered in [doc/EN/adaface_verification.md](doc/EN/adaface_verification.md)).

**Note**: the `Detection (BlazeFace)` column in the tables above predates
the detector switch (2026-08-12). On-device detection timings for YuNet
have not been re-measured.

## Liveness / Free vs. Pro boundary

> ⚠️ **Liveness is currently disabled by default in the example app**
> (`_kLivenessEnabled = false` in `example/lib/main.dart`). Real-device
> re-verification on 2026-08-16 surfaced two problems: (1) running the
> 478-point landmark model on every frame pushed frame processing from a
> 55ms median to 186ms, which made genuine blinks hard to catch, and (2)
> `LivenessState.passed` is not re-verified per `identify()` attempt and is
> not tied to *which* face is in frame — so right after passing, a monitor
> photo of a different person sailed straight through the liveness gate
> (it was rejected downstream, by embedding similarity, only). The second
> is a spoofing defect rather than a UX one, so it is tracked openly as
> [#4](https://github.com/wahihi/facekit/issues/4) instead of being
> quick-patched, and **until it is fixed this SDK does not claim to block
> photo spoofing.** The published demo video was recorded with liveness
> enabled, so cloning and building today will not reproduce its
> photo-rejection scene.

What follows describes the behaviour and design limits when it is enabled.

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
| YuNet (160×160) | Detection (default) | MIT (OpenCV Zoo) | ✅ Yes |
| BlazeFace short-range | Detection (fallback) | Apache 2.0 (MediaPipe) | ✅ Yes |
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
| YuNet (face_detection_yunet) | MIT | https://github.com/opencv/opencv_zoo |
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
  detection/   YuNet (default) + BlazeFace (fallback)
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
