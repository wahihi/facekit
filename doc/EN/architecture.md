# facekit Architecture & Recognition Pipeline

facekit is a clean-room, on-device face recognition SDK for Flutter. This
document describes how the codebase is structured and exactly what happens,
step by step, from a raw camera frame to a "this is person X" answer.

## 1. Design principles

- **Clean-room implementation.** Every module's source comment cites the
  public model/paper/spec it's based on (BlazeFace paper, MediaPipe, ArcFace
  reference alignment, etc.). No proprietary or third-party commercial code
  is referenced.
- **Pure core.** `lib/src/core/` never imports `package:flutter/*` or
  `dart:ui` — it's plain Dart, runnable and testable without Flutter at all.
- **Interfaces over implementations.** Every pipeline stage is an abstract
  contract (`FaceDetector`, `FaceAligner`, `FaceEmbedder`, `FaceMatcher`).
  Concrete implementations (BlazeFace, ArcFace, etc.) are swappable behind
  these contracts, and tests inject fakes/stubs through them.
- **BYOM licensing model.** Detection (BlazeFace, Apache 2.0) is bundled with
  the package. Embedding models are *not* bundled — they're research-licensed
  weights the developer brings themselves (Bring Your Own Model). This keeps
  the SDK itself free of non-commercial licensing risk.

## 2. Layered architecture

Dependencies flow in one direction only — nothing in a lower layer ever
imports from a higher one:

```
UI / example app
        │
        ▼
   pipeline/            (orchestration, isolate dispatch)
        │
        ▼
 detection · alignment · embedding · matching   (one Dart file per concern)
        │
        ▼
   inference/           (TFLite plumbing, manifest parsing)
        │
        ▼
     core/              (pure data models + math + contracts — no Flutter)
```

`core/` has zero dependents below it — it depends on nothing but
`dart:typed_data` and `dart:math`.

## 3. Core data model (`lib/src/core/models.dart`)

| Type | Role |
|---|---|
| `FaceImage` | Decoded RGB888 image (any source: camera frame, gallery photo) |
| `Rect`, `Point` | Pixel-space geometry primitives |
| `DetectedFace` | One detection result: bounding box + 6 landmarks + score, in **pixel coordinates of the input `FaceImage`** |
| `AlignedFace` | A square (112×112 or 160×160), aligned, cropped RGB patch — ready for the embedder |
| `Embedding` | An L2-normalised `Float32List` vector |
| `Enrollment` | One gallery entry: `id` + `Embedding` + metadata |
| `MatchResult` | `matchedId` (nullable) + `similarity` + `accepted` |

All math (`lib/src/core/math.dart`: `l2Normalize`, `cosineSimilarity`,
`l2Norm`) is pure — no I/O, no mutation, no Flutter.

## 4. The recognition pipeline, step by step

`FacePipeline` (`lib/src/pipeline/face_pipeline.dart`) is the orchestrator.
Its `enroll()` and `identify()` methods both run the same four-stage chain;
`identify()` adds a final matching step against a gallery.

```
FaceImage
   │  (1) DETECT — BlazeFaceDetector
   ▼
DetectedFace  (bbox + 6 landmarks, pixel-space)
   │  (2) ALIGN — AffineAligner
   ▼
AlignedFace  (112×112 or 160×160 RGB, face-frontalised)
   │  (3) EMBED — TfliteFaceEmbedder (runs inside an Isolate)
   ▼
Embedding  (512-dim, L2-normalised)
   │  (4) MATCH — CosineMatcher          [identify() only]
   ▼
MatchResult  (matchedId, similarity, accepted)
```

### Stage 1 — Detect (`lib/src/detection/`)

`BlazeFaceDetector` wraps Google's **BlazeFace short-range** model
(Apache 2.0, bundled at `assets/models/blazeface_short/`):

1. The input `FaceImage` is resized to 128×128 and normalised to `[-1, 1]`.
2. TFLite inference produces two raw tensors: `[1, 896, 16]` regressors
   (box + 6 keypoints, 2 values each) and `[1, 896, 1]` classificator scores.
3. `blazeface_anchors.dart` generates the 896 SSD anchor boxes once (cached);
   `blazeface_decoder.dart` decodes the raw tensors against those anchors,
   applies sigmoid to scores, filters by `scoreThreshold`, and runs
   greedy NMS (`iouThreshold`) — all in **normalised `[0,1]` coordinates**,
   since that's anchor space.
4. `denormalizeDetections()` then scales every box/landmark by the
   *original* image's width/height, so the `DetectedFace` list returned to
   the caller is in pixel space matching the input `FaceImage` — not the
   128×128 model input. This conversion step exists specifically so the
   aligner (next stage) doesn't have to know anything about the detector's
   internal resolution.

### Stage 2 — Align (`lib/src/alignment/affine_aligner.dart`)

Faces come in at arbitrary scale, rotation, and position. `AffineAligner`
fixes that with a **5-point similarity transform** (Umeyama's 1991
closed-form least-squares method — rotation + uniform scale + translation,
no shear):

1. Five of BlazeFace's six landmarks (left eye, right eye, nose, mouth, and
   a midpoint of the two ear points standing in for a fifth point) are
   matched against a canonical reference layout — e.g.
   `arcface112Ref`, the standard ArcFace 112×112 reference coordinates.
2. The closed-form 2×2 similarity matrix is computed via an analytic 2×2 SVD
   (no external linear-algebra dependency needed for a 2×2 case).
3. The image is bilinearly resampled through the inverse transform into a
   square `AlignedFace` (112×112 for ArcFace/AdaFace/MobileFaceNet, 160×160
   for FaceNet).

Because different embedding model families expect different canonical
layouts, `AffineAligner` is parameterised by reference points + output size
(`AffineAligner.arcface112()`, `AffineAligner.facenet160()`).

### Stage 3 — Embed (`lib/src/embedding/`)

This is the one stage that's intentionally *not* a single fixed model —
it's manifest + adapter driven so new embedding families can be added
without touching the pipeline:

- **`ModelManifest`** (`lib/src/inference/model_manifest.dart`) is the single
  source of truth for a model's I/O contract: input size/colour/normalisation,
  output dimension, alignment reference, matching threshold, and license
  metadata. It's pure Dart, parsed from a `manifest.json` next to the
  `.tflite` file.
- **`EmbedderAdapter`** is the per-family pre/post-processing strategy:
  - `ArcfaceAdapter` — serves the `arcface`, `adaface`, and `mobilefacenet`
    families, all of which share a 112×112 input and a fixed
    `(pixel - mean) / std` normalisation taken straight from the manifest.
  - `FacenetAdapter` — FaceNet is the documented "exception": 160×160 input,
    and instead of fixed constants it **prewhitens** each image using *that
    image's own* mean/std (`(x - mean(x)) / max(std(x), 1/sqrt(N))`), the
    classic davidsandberg/facenet preprocessing convention. No Flutter
    dependency at all — pure Dart, unit-testable with `dart test`.
- **`TfliteFaceEmbedder`** ties it together: loads the `.tflite` + manifest,
  picks the adapter via `adapterForFamily(manifest.family)`, and implements
  the `FaceEmbedder` contract (`embed(AlignedFace) → Future<Embedding>`).
- Output post-processing applies `l2Normalize` whenever
  `manifest.output.l2Normalize` says to — this is what lets `CosineMatcher`
  treat every embedding as a unit vector regardless of model family.

**Why this stage runs inside an `Isolate`:** `FacePipeline._embedInIsolate`
wraps the embed call in `Isolate.run(() => embedder.embed(face))`. TFLite
inference is the heaviest step in the pipeline; running it off the UI
isolate keeps the app's frame rate from dropping during inference. This was
specifically verified to work correctly with a closure capturing an
FFI-backed `Interpreter` object across the isolate boundary (see §6).

### Stage 4 — Match (`lib/src/matching/cosine_matcher.dart`)

`CosineMatcher` is a pure, stateless function: cosine similarity between the
probe embedding and every gallery `Enrollment`, picking the best score and
accepting it only if it clears `threshold`. The threshold travels with the
model via `CosineMatcher.fromManifest(manifest)`, reading
`manifest.matching.threshold` instead of being hardcoded per call site.

## 5. License tiers and the BYOM model

`ModelManifest.license` carries one of four tiers:

| Tier | Meaning | Redistributable |
|---|---|---|
| `bundled` | Ships inside the facekit package (e.g. BlazeFace, Apache 2.0) | yes |
| `research` | Evaluation/PoC only (e.g. ArcFace buffalo_l, trained on non-commercial data) | no |
| `byom` | Developer supplies their own commercially-licensed model | n/a |
| `licensed` | A commercial license has been purchased for a specific model | yes (under that license) |

`ModelManifest.assertLoadable({required bool isReleaseBuild})` throws if a
non-redistributable model is loaded in a release build (`kReleaseMode`).
Every loader (`BlazeFaceDetector.fromAsset/fromFile`,
`TfliteFaceEmbedder.fromAsset/fromFile`) calls this right after
`validate()`, so a Demo/research model can be used freely for local
development and testing (debug/profile builds) but is hard-blocked the
moment someone tries to ship it in a release APK.

In practice this means: the ArcFace `buffalo_l` model used during
development (`assets/models/arcface_buffalo_l/manifest.json`) is **not**
committed to the repository — only its manifest is. The actual `.tflite`
weights are downloaded and converted locally by each developer (see §7) and
excluded via `.gitignore` (`**/assets/models/*/*.tflite`, with an explicit
exception carved out for the bundled BlazeFace model).

## 6. Concurrency model

- `BlazeFaceDetector.detect()` currently runs synchronously on the calling
  isolate (detection is comparatively cheap relative to embedding for a
  single face).
- `TfliteFaceEmbedder.embed()` is dispatched through `Isolate.run()` by
  `FacePipeline`, so the (heavier) embedding inference never blocks the UI
  isolate's frame rendering.
- This was validated end-to-end with an integration test
  (`test/pipeline/face_pipeline_smoke_test.dart`) that runs the *real* ArcFace
  TFLite interpreter from inside a spawned isolate and confirms the returned
  embedding is numerically correct (self-similarity ≈ 1.0 across two
  independent isolate spawns for the same input).

## 7. Getting an embedding model running locally

Because embedding weights aren't bundled, getting a working demo requires:

1. Obtain a model export (e.g. InsightFace `buffalo_l` → `w600k_r50.onnx`,
   the ArcFace ResNet50 embedding model trained on WebFace600K).
2. Convert ONNX → TensorFlow → TFLite (`onnx2tf`), producing a float32
   `.tflite` with input `[1,112,112,3]` and output `[1,512]`.
3. Numerically verify the conversion (compare ONNX vs TFLite output on the
   same input — cosine similarity should be ≈ 1.0).
4. Place the `.tflite` next to its `manifest.json` under
   `assets/models/arcface_buffalo_l/` (library tests) or
   `example/assets/models/arcface_buffalo_l/` (the example app).

## 8. Example app

`example/` is a minimal Flutter app that exercises the full pipeline against
a live camera feed: it loads BlazeFace (via the facekit package's bundled
asset, `packages/facekit/assets/...`) and ArcFace (via the example app's own
local asset), wires them into a `FacePipeline`, and offers two actions —
**register** the current camera frame under a name, and **identify**
continuously against the in-memory gallery — plus a front/back camera
switch. Because the embedding model is research-licensed, the app is only
ever built in debug/profile mode; a release build would hit the
`assertLoadable` guard described in §5.
