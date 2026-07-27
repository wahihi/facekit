🇰🇷 [한국어 원문](../KR/benchmark.md)

---

# Benchmark (2026-06-26)

A record of inference speed, model size, and accuracy measurements for the
facekit pipeline. For accuracy (EER) methodology and raw data, see
[adaface_verification.md](adaface_verification.md) and
[tool/model_verification/](../../tool/model_verification/).

## Measurement environment

Measured on this machine (Linux VM, 2 CPU cores, 27GB RAM) using the TFLite
Python interpreter (XNNPACK delegate, single-threaded). **These are not real
mobile-device numbers (Pixel 7, etc.)** — they differ from an NNAPI/GPU
delegate or a real mobile SoC's actual throughput. This table is for
relative comparison between models and confirming "it actually runs" —
real-device numbers need to be measured separately and kept up to date.

## Model size (.tflite, float32)

| Model | Role | Input | Output | Size |
|---|---|---|---|---|
| BlazeFace short-range | Detection | 128×128 RGB | 896 anchors × (bbox+16, score) | 0.2 MB |
| ArcFace (buffalo_l/w600k_r50) | Embedding | 112×112 RGB | 512-d | 166 MB |
| AdaFace (IR-101/WebFace12M) | Embedding | 112×112 BGR | 512-d (+ 1 norm output) | 249 MB |

## Inference time (single inference, n=50, excluding 5 warmup runs)

| Model | Mean | p50 | p95 |
|---|---|---|---|
| BlazeFace short-range (detection) | 0.87 ms | 0.81 ms | 1.23 ms |
| ArcFace w600k_r50 (embedding) | 109.3 ms | 108.5 ms | 115.1 ms |
| AdaFace IR-101/WebFace12M (embedding) | 203.6 ms | 202.1 ms | 215.5 ms |

Detection is a lightweight model and nearly free; embedding (especially
AdaFace's IR-101 backbone) accounts for most of the pipeline's latency.
AdaFace is slower than ArcFace (R100) simply because its backbone is
deeper — a trade-off against the accuracy gain documented in
[adaface_verification.md](adaface_verification.md).

## Accuracy (LFW 200 pairs, EER — see adaface_verification.md for detail)

| Model | Clean EER | Degraded EER | Note |
|---|---|---|---|
| ArcFace w600k_r50 | 8.5% | 25.0% | Threshold shifts substantially with quality, 0.26→0.33 |
| AdaFace IR-101/WebFace12M | 2.0% | 14.0% | Threshold barely moves, 0.21→0.22 (favorable for a fixed threshold) |

## Limitations (this section, VM measurements)

- This machine is not a mobile device — do not read the absolute ms figures
  as mobile-SoC performance.
- n=50, repeated on a single input (random tensor) — cold-start (model load
  time) is excluded, and only batch=1 was measured.
- BlazeFace is measured against a single dummy tensor and does not include
  the decode/resize/NMS cost of a real camera frame (the Dart-side
  preprocessing cost of `yuv420ToFaceImage`/`resizeNearest` etc. is separate).

## Real device (Pixel 7, 2026-06-27)

Measured via the benchmark button built into the example app
(`example/lib/benchmark.dart`). Measured with a **`flutter build apk
--profile`** build — `--debug` uses unoptimized JIT, which inflates the
Dart-side pre/post-processing cost (anchor decode/NMS/image resize) and
isn't trustworthy (in practice it measured almost the same as debug, which
if anything shows "JIT overhead isn't the cause"); `--release` can't even
run at all, since `arcface_buffalo_l` is `redistributable:false` and
`assertLoadable()` blocks loading it. Profile mode has `kReleaseMode==false`
so it isn't blocked by that guard, while its AOT optimization matches
release — making it the right build for this measurement.

The target measured is a live camera frame (detection → `AffineAligner`
alignment → embedding); embedding was run exactly as `FacePipeline` actually
runs it, via `Isolate.run()` spawning a fresh isolate each time (isolate
spawn cost is included, on the reasoning that this is closer to the real
per-frame cost). n=30, excluding 5 warmup runs, repeated on one fixed frame
captured by the camera.

| Mode | Detection (BlazeFace) | Embedding (ArcFace buffalo_l) | Full frame |
|---|---|---|---|
| CPU (default, no acceleration) | mean 65.1ms / p50 66.4ms / p95 79.0ms | mean 729.4ms / p50 736.2ms / p95 846.7ms | mean 795.6ms / p50 798.5ms / p95 920.3ms |
| NNAPI (`useNnApiForAndroid=true`) | mean 76.6ms / p50 76.3ms / p95 101.3ms | mean 876.2ms / p50 890.5ms / p95 959.5ms | mean 954.4ms / p50 965.4ms / p95 1065.1ms |

**NNAPI was actually slower than CPU.** Both models are float32 graphs, and
NNAPI is usually optimized to accelerate int8-quantized models — it's common
for float32 to fail to get accelerated by the Pixel 7's NNAPI vendor driver
and fall back to CPU, **adding only NNAPI dispatch/IPC overhead** on top.
The smaller a model already is (BlazeFace at 0.2MB), the more that overhead
shows up proportionally (detection itself slowed from 65→77ms). This result
also empirically confirms that `TfliteRunner`'s current default of
`useNnApi = false` was the right call — not "acceleration off by default"
as an assumption, but now backed by evidence that "acceleration is a net
loss for this particular model combination."

Compared to the VM numbers above, real-device CPU embedding (729ms) being
~6x slower than the VM (109ms) has a consistent explanation too — the VM
measurement used Python TFLite with the XNNPACK delegate enabled, but this
SDK's `TfliteRunner` never configures `InterpreterOptions` at all
(`tflite_runner.dart`), so it runs on the plain CPU reference kernels with
neither XNNPACK nor multithreading turned on. In other words, the current
real-device numbers are an honest lower bound with "no acceleration
whatsoever," and turning on the XNNPACK delegate (separately from NNAPI) is
a worthwhile next optimization candidate.

### Limitations (real-device section)

- Only one Pixel 7 was measured. See the Galaxy S25 section below for
  another device's numbers.
- n=30, repeated on one arbitrary frame captured by the camera as a fixed
  input — variance across frames with different lighting/angle wasn't
  measured.
- Detection time includes Dart-side preprocessing (resize) and
  post-processing (anchor decode + NMS), so this may account for more of
  the cost than the native inference itself (code review found that
  `prepareInputTensor`/`zeroTensor` allocate a fresh boxed
  `List<List<...>>>` every frame — switching to a typed buffer like
  `Float32List` could reduce this, but that's left as separate future work).

## Real device (Pixel 7, AuraFace, release build, 2026-07-25)

Measured by downloading and installing the **`--release` APK** published to
the GitHub Release (`v0.1.0`), using the same in-app benchmark button.
Unlike the ArcFace numbers above, this is a release build — AuraFace is
`redistributable: true`, so it isn't blocked from loading in a release
build the way ArcFace was (ArcFace needed `--profile` because of the
`assertLoadable()` guard, see above). `--profile` and `--release` are both
AOT-compiled, so inference speed should in principle be similar, but since
this wasn't measured under the exact same build mode, treat any direct
ratio against the ArcFace numbers below as a rough reference only.

The same APK was uninstalled and reinstalled, and measured twice (n=30
each, excluding warmup, repeated on one fixed frame captured by the
camera):

| Mode | Run | Detection (BlazeFace) | Embedding (AuraFace) | Full frame |
|---|---|---|---|---|
| CPU | 1st | mean 51.5ms / p50 49.8ms / p95 63.6ms | mean 1082.7ms / p50 1119.9ms / p95 1215.7ms | mean 1135.3ms / p50 1171.8ms / p95 1275.7ms |
| CPU | 2nd | mean 50.9ms / p50 49.0ms / p95 62.1ms | mean 1186.2ms / p50 1179.9ms / p95 1332.6ms | mean 1238.1ms / p50 1230.6ms / p95 1391.5ms |
| NNAPI | 1st | mean 49.5ms / p50 49.0ms / p95 59.7ms | mean 1106.0ms / p50 1086.4ms / p95 1367.0ms | mean 1156.2ms / p50 1136.9ms / p95 1418.7ms |
| NNAPI | 2nd | mean 61.1ms / p50 60.4ms / p95 90.4ms | mean 1204.5ms / p50 1221.1ms / p95 1377.8ms | mean 1267.3ms / p50 1276.2ms / p95 1432.6ms |

**Embedding alone shifted by about 9–10% between the two independent
runs** (detection stayed stable at 50–61ms). This looks like ordinary
real-device benchmark variance (a cold state right after reinstalling,
thermal effects, background processes) rather than anything indicating a
bug — but that much swing is itself a reason to report a range instead of
treating a single run as a fixed value.

**About 1.2–1.5x slower than ArcFace (R50)** — ArcFace buffalo_l uses a
ResNet50 backbone, while AuraFace (glintr100) uses ResNet100, which is
simply more compute. The fp16 conversion only shrinks the file size
(130MB); as long as `TfliteRunner` runs on the plain CPU reference kernels
with neither XNNPACK nor multithreading enabled (the same cause noted in
the ArcFace section above), it doesn't speed up inference itself.

**The NNAPI conclusion is the same as ArcFace's — keeping CPU as the
default is still right.** In both runs, NNAPI's full-frame mean was
slightly slower than CPU's (+1.8% on the 1st run, +2.4% on the 2nd), and
p95 was always worse under NNAPI (1418.7 vs 1275.7ms on the 1st run,
1432.6 vs 1391.5ms on the 2nd). The gap is smaller than ArcFace's (where
NNAPI was 20% slower), but points the same direction.

**A note on how live recognition feels**: while "실시간 인식 시작" (start
live recognition) is active, detection+alignment+embedding all run on
every single frame. A full frame taking 1.1–1.4 seconds means recognition
mode runs at under 1 fps, which reads as noticeably more sluggish than
ArcFace's ~0.8s.

### Limitations (this section)

- Only one Pixel 7, measured twice after a reinstall — still a small
  sample.
- This is a `--release` build, a different build mode from the ArcFace
  numbers above (`--profile`) — since the methodology isn't identical,
  treat any ratio between them as a rough reference only.
- No real-device AuraFace measurement on the Galaxy S25 yet.

## Real device (Galaxy S25 SM-S931N, 2026-06-30)

Same methodology as the Pixel 7: the example app's benchmark button, a
`flutter build apk --profile` build, n=30 excluding warmup, repeated on one
fixed frame captured by the camera. Embedding runs in a fresh isolate each
time via `Isolate.run()` (isolate spawn cost included).

| Mode | Detection (BlazeFace) | Embedding (ArcFace buffalo_l) | Full frame |
|---|---|---|---|
| CPU (default, no acceleration) | mean 48.0ms / p50 47.8ms / p95 55.6ms | mean 256.6ms / p50 250.7ms / p95 268.1ms | mean 305.0ms / p50 301.9ms / p95 314.4ms |
| NNAPI (`useNnApiForAndroid=true`) | mean 44.7ms / p50 37.0ms / p95 66.6ms | mean 244.4ms / p50 205.0ms / p95 325.6ms | mean 289.5ms / p50 241.4ms / p95 385.8ms |

**Compared to the Pixel 7:** embedding is about 2.8x faster, 729ms → 257ms —
a direct reflection of the raw CPU throughput difference of the Snapdragon
8 Elite (in the Korean/North American SM-S931N model).

**The NNAPI result is the opposite of the Pixel 7's.** NNAPI was slower
than CPU on the Pixel 7 (Google Tensor G2), but on the S25 (Qualcomm
Snapdragon 8 Elite), NNAPI's mean is 15ms faster than CPU (305.0ms →
289.5ms), suggesting the Qualcomm AI Engine driver accepts at least some
delegation even for a float32 graph. That said, **variance widened
significantly** — full-frame p95 is actually higher under NNAPI (386ms) than
CPU (314ms). The gap between p50 (241ms) and p95 (386ms) reaches 145ms,
suggesting driver scheduling / thermal throttling makes latency irregular.
Since real-time camera applications care more about p95 stability than the
mean, the conclusion that **keeping CPU as the default is still the right
call** holds equally on the S25.

### Limitations (Galaxy S25 section)

- Only one SM-S931N (Snapdragon 8 Elite) unit was measured — the Exynos
  variant (released in some regions) wasn't measured.
- n=30, repeated on one arbitrary frame captured by the camera as a fixed
  input — variance across lighting/angle wasn't measured.
- Same preprocessing-cost structure as the Pixel 7 section applies (a
  lower-bound figure with no typed-buffer optimization).

## Real device (Galaxy S25 SM-S931N, AuraFace, release build, 2026-07-27)

Measured using the `--release` APK downloaded from the GitHub Release
(`v0.1.0`) — the same conditions as the Pixel 7 AuraFace measurement above.
The face in frame belonged to someone other than the person running the
benchmark, but since this benchmark button **only times
detection→alignment→embedding and never computes a match/similarity**
(there's no `similarity`/matching-related code in
`example/lib/benchmark.dart`), whose face it is has no bearing on this
result.

**This wasn't the tester's own device and couldn't be re-measured** — so
unlike the Pixel 7 section, this one has **only a single run**. Read the
numbers below, especially the NNAPI conclusion, with that in mind.

n=30, excluding warmup, repeated on one fixed frame captured by the camera:

| Mode | Detection (BlazeFace) | Embedding (AuraFace) | Full frame |
|---|---|---|---|
| CPU (default) | mean 56.3ms / p50 54.3ms / p95 78.2ms | mean 598.8ms / p50 606.3ms / p95 643.5ms | mean 655.7ms / p50 662.7ms / p95 722.5ms |
| NNAPI | mean 46.3ms / p50 44.0ms / p95 59.3ms | mean 480.2ms / p50 441.5ms / p95 621.3ms | mean 527.2ms / p50 481.9ms / p95 720.1ms |

**The cross-device ratio is consistent**: AuraFace is also ~1.8–2x faster
on the S25 (598.8ms) than the Pixel 7 (1082.7–1186.2ms) — the same
direction as the ArcFace cross-device ratio above.

**The cross-model ratio differs by device.** On the Pixel 7, AuraFace was
about 1.2–1.5x slower than ArcFace; on the S25, it's about 2.3x slower
(598.8ms ÷ 256.6ms). Same pair of models, different relative gap per
device — this looks like different chips handling a deeper compute
structure (ResNet50 → ResNet100) with different relative efficiency,
rather than anything indicating a bug.

**NNAPI is clearly better here, this time.** Full-frame mean dropped from
655.7ms to 527.2ms (19.6% faster), and p95 was slightly better too
(722.5ms → 720.1ms) — unlike the pattern in the ArcFace section above
("mean is a bit faster, but p95 variance makes it risky"). **But this is a
single run (no repeat possible), and the Pixel 7 section already showed
~9–10% run-to-run swings on embedding alone — one result isn't enough to
treat as settled.** Whether NNAPI is genuinely better for S25 + AuraFace
is left as an open question needing more measurements — not yet enough to
justify switching the default from CPU to NNAPI.

### Limitations (this section)

- **Single run, no repeat possible** (not the tester's own device) — the
  most important limitation here.
- Only one SM-S931N unit measured.
- `--release` build, a different build mode from the ArcFace (`--profile`)
  numbers above.
