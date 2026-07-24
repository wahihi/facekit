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
