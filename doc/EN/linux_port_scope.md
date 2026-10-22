🇰🇷 [한국어 원문](../KR/linux_port_scope.md)

---

# Linux (Q6A) headless port — S1 dependency boundary survey (2026-08-26)

A one-day investigation, done before introducing `facekit_cli` (a headless recognition process
that runs with `dart run`, no Flutter), to measure how many days "Option A — keep Dart" would
actually take. Only the method and conclusions are recorded here — the actual refactor happens
in S2.

## Method

1. `grep -rn "package:flutter\|dart:ui" lib/src/` — scan Flutter dependencies layer by layer
2. `tool/probe/` — create a pure-Dart console package that depends on `facekit` via `path:` and
   try `dart pub get`

**Caveat (a trap found during this very investigation)**: on this dev VM, `dart` is
`~/flutter/bin/dart`, i.e. the Dart SDK bundled with Flutter. Running `pub get` there silently
resolves the `flutter: sdk: flutter` dependency and yields a false positive ("it works"). The Q6A
board has only a standalone Dart SDK without Flutter, so to verify, a separate standalone Dart SDK
(`~/tools/dart-standalone/`) had to be downloaded on this VM as well and used to reproduce.

## Finding 1 — the real gate is the `pubspec.yaml`, not individual files

```
$ ~/tools/dart-standalone/dart-sdk/bin/dart pub get   # standalone Dart SDK, no Flutter
Resolving dependencies...
Because probe depends on facekit from path which requires the Flutter SDK,
version solving failed.
Flutter users should use `flutter pub` instead of `dart pub`.
```

`facekit/pubspec.yaml` declares `dependencies: flutter: sdk: flutter` at the root, so **no matter
which file you try to import**, it fails immediately at pub dependency resolution. `tflite_flutter`
is the same: its own `pubspec.yaml` has `dependencies: flutter: sdk: flutter`
(`environment.flutter: '>=1.10.0'`), so even if facekit dropped that declaration, it would still
fail as long as `tflite_flutter` stays in the dependency graph.

**In other words, this isn't a "swap out a few `debugPrint`s" job — the package has to be split.**

## Finding 2 — `core/` is already completely clean

`lib/src/core/models.dart`: zero dependencies beyond `dart:typed_data`; `Rect`/`Point`/`Point3D`
are all self-defined (not dart:ui). Zero `dart:ui` matches in `grep`. No R1 rule violations —
core can be reused as is.

## Finding 3 — camera coupling is not inside the library

Zero references to `package:camera` / `CameraImage` anywhere in `lib/src/`. All camera coupling
lives in `example/` (the Flutter demo app). So S5 (V4L2 capture) won't touch library code at all;
it's purely a new layer on top.

## Finding 4 — 7 files use `package:flutter/foundation.dart`, all replaceable

| File | Symbols used |
|---|---|
| `alignment/affine_aligner.dart` | `debugPrint`, `visibleForTesting` |
| `detection/blazeface_detector.dart` | `kReleaseMode` |
| `detection/yunet_detector.dart` | `debugPrint`, `kReleaseMode` |
| `embedding/face_embedder.dart` | `kReleaseMode`, `debugPrint` |
| `landmark/face_landmarker.dart` | `debugPrint`, `kReleaseMode` |
| `liveness/blink_liveness_detector.dart` | `debugPrint` |
| `pipeline/face_pipeline.dart` | `debugPrint`, `kDebugMode` |

All four are pure utilities unrelated to the Flutter engine:
- `kReleaseMode`/`kDebugMode` are implemented inside `foundation.dart` itself as three lines of `const bool.fromEnvironment('dart.vm.product')` — copy them and you're done.
- `visibleForTesting` originates in `package:meta/meta.dart` (a pure Dart package); `foundation.dart` merely re-exports it — just change the import to `package:meta/meta.dart`.
- `debugPrint` can be replaced with plain `print`, or with an unbuffered in-house implementation.

`lib/src/core/debug_flags.dart` already exists, but it's unrelated (an on/off switch for verbose
logging) — the new shim can simply sit next to it.

## Finding 5 — the `tflite_flutter` coupling is already isolated in one file, and reusable pure-Dart FFI code sits inside it

`inference/tflite_runner.dart` is already written on the principle "Keeps tflite_flutter confined
to this file", and `TfliteRunner.fromFile()` (loading from a filesystem path) already exists — it
was designed with non-Flutter contexts somewhat in mind from the start.

Opening the source of `tflite_flutter` 0.12.1 itself (Apache 2.0):

```
lib/src/bindings/tensorflow_lite_bindings_generated.dart  ← ffigen output, no Flutter dependency
lib/src/bindings/bindings.dart                            ← same as above
lib/src/ffi/helper.dart                                   ← same as above
lib/src/interpreter.dart, tensor.dart                      ← flutter imports start here
```

So pure-Dart FFI bindings that call `libtensorflowlite_c.so` **already exist and are Apache 2.0**.
Instead of generating 300 new lines with ffigen in S2, these binding files can be brought over as
is (with attribution) and only a thin `TfliteFfiRunner` wrapper needs to be written. This reduces
the S2 workload below the estimate.

## Conclusion — Option A is valid, but its scope is larger than the plan's "add one interface"

The original S2 description ("cost effectively zero, one more implementation behind an
interface") rested on **the false premise that no package split was needed**. Per the
measurements, what's actually needed is:

1. **A package split.** Move `core/`, `contracts.dart`, `alignment/`, `matching/`,
   `detection/` (YuNet), `embedding/` (adapters), `landmark/`, `liveness/`, `pipeline/` into a
   separate pubspec (e.g. `packages/facekit_core/` in the same repo) that depends on neither
   `flutter: sdk: flutter` nor `tflite_flutter`. The existing `facekit` package stays as a thin
   layer on top, providing only `TfliteRunner` (the Flutter/tflite_flutter implementation) and
   Flutter-only exports — **`example/` (the Flutter demo app) stays untouched.**
2. Replace the `package:flutter/foundation.dart` imports in the 7 files (`debugPrint`,
   `kReleaseMode`, etc.) with a pure-Dart shim (Finding 4).
3. Port `tflite_flutter`'s existing FFI binding files and write `TfliteFfiRunner` (Finding 5 —
   easier than originally planned).
4. Add an `InferenceRunner` abstract interface to `contracts.dart`, implemented by both
   `TfliteRunner` and `TfliteFfiRunner` — opening the slot for a future QNN runner now (zero cost,
   keeps the existing decision).

**Updated estimate**: the package split makes S2 somewhat larger than the original estimate
(2–3 days), partly offset by reusing the bindings (Finding 5) — still seen as 2–4 days. The
overall 2–3 week estimate stands.

## Commit

Just this document. No `dart:ui` removal refactor was needed (Finding 2). `tool/probe/` is kept as
the minimal reproduction fixture used for the investigation (`pubspec.yaml` and `main.dart` only —
generated artifacts are in `.gitignore`).

## S2 complete (2026-08-26)

Following Finding 1 (package split needed), `packages/facekit_core/` (pure Dart, depending on
neither `flutter:` nor `tflite_flutter`) was actually split out.
`core/alignment/detection/embedding/landmark/liveness/matching/pipeline/image/inference`
(excluding `tflite_runner.dart`) all moved, and the existing `facekit` package remains as a thin
layer on top, providing only `TfliteRunner` (the Flutter/tflite_flutter implementation) and the
asset loaders (`lib/src/asset_loaders.dart`). In `example/`, only 5 call sites changed
mechanically (`X.fromAsset(...)` → `loadXFromAsset(...)`); behavior is unchanged.

Following Finding 5 (reusable FFI bindings), `TfliteFfiRunner`
(`packages/facekit_core/lib/src/inference/tflite_ffi_runner.dart`) was written — `tflite_flutter`'s
ffigen output (`bindings/tensorflow_lite_bindings_generated.dart`, Apache 2.0, attribution in
`third_party/tflite_flutter/NOTICE.md`) was ported as is, with a new float32-only tensor
marshalling layer on top. An `InferenceRunner` abstract interface was added to
`core/contracts.dart`, implemented by both `TfliteRunner` and `TfliteFfiRunner` — the slot for a
future QNN runner.

**Verification**:
- `dart analyze` (standalone Dart SDK, `~/tools/dart-standalone/`) reports no errors for
  `facekit_core` — zero Flutter dependency reconfirmed by measurement.
- `dart test` (same standalone SDK): 106 of 110 pass. The 4 failures (`blazeface_smoke_test`,
  `yunet_smoke_test`×2, `face_landmarker_smoke_test`) are **the same 4** as before the refactor,
  with the same cause (no `libtensorflowlite_c.so` in this environment) — not a regression, the
  same kind of gap as CI's existing exclusion list (`yunet_smoke_test.dart` had been missing from
  the CI exclusion list; that was fixed too — `.github/workflows/test.yml`).
- `flutter analyze`/`flutter test` (outer) report no errors, and `flutter build apk --debug`
  (example) succeeds — no regression on the Android path.

## libtensorflowlite_c.so for x86_64 built successfully (2026-08-26, continued)

Obtained it by building directly with CMake. Following the official guide
(developers.google.com/edge/litert/build/cmake) verbatim failed once; the cause and fix are
recorded here, since the same problem would hit arm64 (Q6A) too.

**Why the first attempt failed**: after cloning with `git clone` (no flags, i.e. master HEAD) and
running cmake configure, `tensorflow/lite/CMakeLists.txt` **internally fetches tensorflow a second
time** via `FetchContent` (pinned to `GIT_TAG v2.21.0-rc0`). My clone (master HEAD) and the
internal re-clone (v2.21.0-rc0) ended up at different commits, the two trees got mixed, and the
build failed with a declaration mismatch between `mmap_handle.cc` and `.h`
(`no declaration matches ...`). Not a TF bug — **a version mismatch**.

**Fix**: clone exactly the tag CMakeLists.txt expects from the start
(`git clone --depth 1 --branch v2.21.0-rc0`), and skip the internal re-clone entirely with
`cmake -DTENSORFLOW_SOURCE_DIR=<path to my clone> ...`. It then built normally — since this VM has
no sudo, cmake itself also had to be installed separately from Kitware's official binaries into
`~/tools/cmake` (apt not available).

**Verification**: reran all of `facekit_core` with `FACEKIT_TFLITE_C_LIBRARY=<so path> dart test`
— **114 of 114 pass** (3 BYOM models skipped as expected). The 4 previously failing tests
(`blazeface_smoke_test`, `yunet_smoke_test`×2, `face_landmarker_smoke_test`) all pass through
actual inference, and `face_pipeline_smoke_test.dart` (which checks that `Isolate.run()` correctly
hands FFI pointers to an isolated isolate) passes too — meaning `TfliteFfiRunner` really works.

**Next (for Q6A/arm64)**: use the same recipe (pinned tag + `TENSORFLOW_SOURCE_DIR` override) as
is. The build output (`libtensorflowlite_c.so`, 8.8MB) is not committed to the repo — as a
platform-specific binary it's treated like a BYOM/native artifact and kept local only
(`~/tools/tflite_c_build/tflite_build/`).

## S3 — facekit_cli started (2026-08-26, continued)

Wrote `packages/facekit_cli/` (path dependency on `facekit_core`, JPEG decoding via
`package:image`, CLI argument parsing via `package:args`):

```
dart run bin/facekit_cli.dart --image a.jpg \
  --detector assets/models/yunet_160 --embedder example/assets/models/auraface
→ {"faces":[{"bbox":[...],"score":..,"embedding":[512 values],"ms":{"detect":..,"align":..,"embed":..}}]}
```

Ran all three real photos already in `tool/model_verification/` (`me_1.jpg`, `me_2.jpg`,
`other_1.jpg` — photos used for model verification since the start of the project, committed to
the repo) to confirm self-consistency:

| Comparison | Cosine similarity | Verdict (threshold 0.211) |
|---|---|---|
| me_1 vs me_2 (same person) | **0.745** | Pass (correctly recognized as the same person) |
| me_1 vs other_1 (different person) | 0.026 | Correctly rejected |
| me_2 vs other_1 (different person) | 0.0006 | Correctly rejected |

Strong circumstantial evidence, even without an Android comparison, that YuNet detection (bbox
score 0.92–0.95) plus AuraFace embedding actually work correctly.

**However, this is not yet the original S3 DoD (cosine ≥ 0.999 vs. Android).** What was confirmed
is "does it correctly tell the same person from different people"; what the plan was meant to
catch is "do Linux and Android produce the same numbers for the same input with no platform
differences" (to catch platform-specific bugs such as missing rotation correction or channel-order
bugs). That comparison needs real embedding values from running these three photos (or any photo)
through the same pipeline on a real Android device, so it can't be finished within this session —
it needs one run on the maintainer's Pixel 7.

## S3 DoD passed (2026-09-03)

Added a temporary "golden vector dump" button to `example/` (`_dumpGoldenVector`,
`assets/golden/me_1.jpg` — byte-identical to `tool/model_verification/me_1.jpg`). Pressing it on a
real Pixel 7 copied the JSON straight to the clipboard (no adb/logcat needed). The same photo was
run through the Linux `facekit_cli` and compared:

| | Value |
|---|---|
| Cosine similarity | **0.99999991** |
| bbox difference | around the 4th decimal place (floating-point rounding error) |
| score | exactly identical (0.9324665440) |

Far above the bar (≥ 0.999) — **S3 DoD passed.** Android and Linux agree on rotation, channel
order, and normalization order.

This Android embedding was committed as `packages/facekit_cli/test/fixtures/me_1_android_golden.json`
and pinned as the `packages/facekit_cli/test/golden_vector_test.dart` regression test — every later
change to the pipeline is checked against this real-device value on the Linux side. As with the
other model-loading tests (no `libtensorflowlite_c.so` in that environment), the test step isn't in
CI yet; only `pub get`/`dart analyze` are.

The logic in `bin/facekit_cli.dart` was extracted into `lib/facekit_cli.dart`
(`runDetectAlignEmbed`) so the binary and the tests share the same code — the CLI's own behavior is
unchanged.

**S3 complete.** Next is S4 (moving to Q6A/arm64).

## S4 complete (2026-09-03)

Moved the code to the board via tar.gz + USB (no direct scp/ping from this environment to the
board — the VirtualBox NAT adapter reaches the internet but not other devices on the same physical
LAN). Built `libtensorflowlite_c.so` for arm64 on the board with the same recipe as x86_64
(pinned tag `v2.21.0-rc0` + `TENSORFLOW_SOURCE_DIR` override) — blocked once by a missing `g++`
(fixed with `sudo apt install build-essential`), then succeeded.

- **DoD #1 (golden vector re-verified on arm64)**: all of `dart test` passes — arm64 also matches
  the real Android device value.
- **DoD #2 (real-device benchmark)**: [benchmark_q6a.md](benchmark_q6a.md) — single-threaded
  721ms → 350ms multi-threaded on four A78 cores. **Conclusion: CPU alone, without the NPU, meets
  the commercial bar (perceived latency under 0.5s); no case for investing in QNN/INT8.**

In line with this finding, a `numThreads` option was added to
`YuNetDetector.fromFile`/`TfliteFaceEmbedder.fromFile`/`TfliteFfiRunner`, plus
`facekit_cli --num-threads` (default 1, existing behavior unchanged).

**S4 complete.** Next is S5 (V4L2 camera capture).

## S5b complete (2026-09-16) — in-house V4L2 capture, verified on real hardware

Removed the ffmpeg dependency (S5a) with `native/v4l2_capture.c` (V4L2 mmap streaming capture,
implemented directly against the public kernel UAPI `<linux/videodev2.h>`) + `dart:ffi` bindings
(`lib/src/v4l2_bindings.dart`) + an implementation of the `FrameSource` abstraction
(`V4l2FrameSource`). In `facekit_core`, a YUYV422→RGB888 conversion (`yuyv422ToFaceImage`) was
added alongside YUV420/BGRA.

**Real-hardware verification (Q6A + Logitech C270) — DoD checked 3 times**:

| Test | Duration | Frames | FPS | Face detection rate | V4L2 drops | Corrupt frames | Memory growth |
|---|---|---|---|---|---|---|---|
| 1st | 30s | 722 | 24.0 | 98.8% | 3 | 9 (1.2%) | +32.5MB |
| 2nd (USB3) | 30s | 841 | 28.0 | 99.2% | 4 | 7 (0.8%) | +31.9MB |
| 3rd | **300s** | 7274 | 24.2 | 98.3% | 4 | 71 (0.98%) | +50.3MB |

**Verdict — not the literal DoD wording (zero drops, zero growth), but treated as a pass in
substance:**

- **Memory**: running 300 seconds (10× the time, 10× the frames) grew memory only 1.5×
  (+50.3MB) — a real leak would grow ~10× in proportion to time. The full log shows memory rising
  into the 320MB range within the first 30 seconds, then flattening out, oscillating between
  319–331MB — the normal pattern of the Dart GC/native allocator reaching its working-set size and
  holding there. **Judged not to be a leak.**
- **V4L2 drops**: 4 in the whole 5 minutes, all early, none in the following 4 min 30 s — not
  worsening over time.
- **Corrupt frames** (YUYV frames exactly 4 bytes short — crashed when first found, now skipped
  cleanly after length validation was added in commit `ecc41a5`): a steady ~1% rate for the whole
  5 minutes — the same rate on a USB3 port (not bus congestion, but a trait of this camera). No
  worsening trend.

**Conclusion**: all three metrics (memory, drops, corruption) were confirmed not to degrade over
time. The DoD's real intent (is long continuous capture stable?) is considered met, and S5b is
closed. Corrupt frames (~1%) are already handled, and for the low-frequency V4L2 drops (0.05%),
"skip that frame and continue with the next" in S6 is enough — no separate retry/recovery logic is
deemed necessary.

**S5b complete.** Next is S6 (real-time recognition loop) — that's the completion point of
Option A.

## S6 started — DoD definition (2026-10-20)

Resumed after roughly a month's gap. The DoD is defined before starting (as requested).

**Gallery design decision**: enroll once at startup from a single `--enroll-image`, then match
against a single-entry gallery for the whole session. Loading a JSON gallery file or persistent
multi-person enrollment is out of scope (S6's purpose is "wiring the camera→recognition loop
itself", not gallery management) — a separate step if it becomes necessary.

**DoD**:

1. **Function**: `facekit_realtime` opens a V4L2 camera, enrolls one face from `--enroll-image`,
   then continuously prints each frame's detect→align→embed→match result to stdout as NDJSON.
2. **Corrupt/dropped frames**: exactly the design already settled in S5b — frames with the wrong
   length are counted and skipped; drops by V4L2 sequence number are only logged and processing
   continues. No separate retry/recovery logic.
3. **No face detected / alignment failure**: handled as "no identification for this frame"
   without crashing, and processing continues.
4. **Shutdown**: besides `--seconds N` for testing, add a clean SIGINT (Ctrl+C) shutdown for real
   use — closing the V4L2 handle and disposing the TFLite interpreters before exiting.
5. **Performance**: processing every frame at the camera's native FPS (24–30) is not the goal —
   per the S4 bench (350ms/frame at 4 threads) that's impossible anyway. The goal is that slow
   frame processing doesn't build a backlog on the capture side (= always process sequentially
   from the most recent frame). Passes if the end-to-end (detect+align+embed+match) latency of
   processed frames doesn't stray far from the S4 bench level (hundreds of ms).
6. **Long-run stability**: same bar as S5b — in a 5-minute (300-second) continuous run, memory
   growth flattens instead of growing with time, and no crash. (S5b ran capture only; this time
   the same bar is rechecked with the full recognition pipeline on top.)
7. **Identification accuracy**: no quantitative (%) bar — visually confirm on real hardware that
   `accepted=true` appears in most frames in front of the enrolled person, and
   `accepted=false`/no identification for a different person or no face. (Fine-tuning the
   threshold itself is out of scope — issue #2.)

**What can and can't be done in this session (dev VM, no camera)**: 2–6 can be implemented and
verified with automated tests (for #5, only rechecking the bench numbers, not live-camera timing).
1 and 7, plus the live-camera part of 5 and the real-hardware long run of 6, **can only be
verified on the Q6A board** — this VM has no V4L2 device.

## S6 implementation (2026-10-20)

Added to `packages/facekit_cli/`:

- `lib/src/realtime_loop.dart` — the core loop, which takes a `FrameSource` (the S5b abstraction)
  and consumes it via `runRealtimeLoop()`. It depends only on the `FrameSource` interface, not
  directly on `V4l2FrameSource`, so tests can run with a fake source and no real camera. While
  processing, it consumes the stream sequentially as pause→(detect→align→embed→match)→resume so
  slow frames don't build a backlog (DoD 5). Corrupt-frame skipping, drop counting, and
  no-face/alignment-failure handling are implemented exactly per DoD 2 and 3. `enrollFromImage()`
  is a dedicated enrollment path (errors immediately if there's no face or alignment fails —
  unlike the loop, "just skip it" makes no sense for enrollment).
- `bin/facekit_realtime.dart` — the actual CLI. `--device/--width/--height/--detector/--embedder/
  --enroll-image/--enroll-id/--seconds/--num-threads/--tflite-library/--v4l2-library`.
  One SIGINT means "finish the current frame, clean up, exit"; a second means exit immediately
  (exit 130). Per-frame results go to stdout as NDJSON, progress/summary to stderr (the summary is
  one line of `RealtimeStats.toJson()`). Uses each model's matching threshold as is via
  `CosineMatcher.fromManifest(embedderManifest)` (issue #2's tuning stays open — outside the scope
  of this step).
- `test/realtime_loop_test.dart` — runs the real YuNet+AuraFace models through a
  `FakeFrameSource` (with the same real photos as golden_vector_test.dart,
  `me_1/me_2/other_1.jpg`) and verifies corrupt-frame skipping, drop counting (sequence 1→3),
  same-person accept / different-person reject, an empty stream, and stopping midway via
  `stopRequested`. Pointing `FACEKIT_TFLITE_C_LIBRARY` at the locally built x86_64 `.so`, all 3
  pass (the whole `dart test` — all 4 including golden_vector_test pass, no regressions). Not yet
  in CI for the same reason as before (no `.so` in that environment) — joins the existing gap.

**Status at the time (2026-10-20)**: implementation and fake-source tests were complete, but the
Q6A real-hardware DoD was not yet verified. See the 2026-10-21 entry below for the later
real-hardware results.

## S6 shutdown and error-path hardening (2026-10-21)

- The `--seconds` limit is now measured from loop start, and the loop exits cleanly even if not a
  single frame arrives. Elapsed time is also checked when a frame finishes processing, so even if
  the timer callback is delayed during back-to-back inference, no new frame starts after the
  limit.
- SIGINT also wakes the loop while waiting between frames; if a frame is in progress, it exits
  after that frame finishes.
- The V4L2 poll timeout is capped at 100ms, yielding to the Dart event loop on each timeout so
  timer/SIGINT handling isn't starved.
- Frame source and pipeline exceptions are propagated to the caller instead of being turned into a
  normal result with stats. The CLI prints a failure message and a non-zero exit code, and cleans
  up the subscription, camera, and both interpreters in `finally`.
- Added automated tests for duration-based shutdown with no frames / mid-processing, SIGINT
  shutdown, source errors, and pipeline errors.

## Q6A port complete, and remaining real-hardware checks (2026-10-21)

Ran `feature/linux-port`'s real-time recognition CLI on the Q6A (Radxa Dragon Q6A, Linux arm64)
with a real V4L2 camera and the arm64 `libtensorflowlite_c.so`. **The original goal — the Q6A
headless port and running real-time camera recognition — is therefore judged complete.**

- Pinned to cores 4–7, 30-second run: 77 frames processed, face detection/alignment 77 times,
  enrolled entry accept 69 / reject 8, 2 corrupt frames, RSS growth about 19.2MB. Frame
  processing time mostly 340–390ms.
- Same core setup, 300-second continuous run: 665 frames processed, face detection/alignment and
  enrolled entry accept 665 times, 4 corrupt frames, RSS start 1067.2MB / max 1071.1MB / growth
  about 3.9MB. Printed final stats and exited cleanly without a crash. Median frame processing
  time about 453ms.
- Because inference is slower than camera capture, frames were skipped by V4L2 sequence (710 in
  30 seconds, 8148 in 300 seconds). Processing every frame at native FPS isn't a goal of the
  sequential design, and the measured inference time is in the same hundreds-of-ms range as the
  existing Q6A benchmark.

**Verification remaining apart from the port completion**:

1. **Clean shutdown on SIGINT (Ctrl+C)**: the automated-path tests pass, but it hasn't been
   confirmed on the Q6A hardware yet. Earlier runs were stopped with Ctrl+Z and then force-killed.
2. **Rejecting other people and no-face frames**: the device log did contain reject results, but
   it wasn't confirmed whether a different person was present at those moments. Verifying on real
   hardware that a different person yields `accepted=false`, and that no identification is
   produced when no face is present, is left as follow-up work.

So **the Q6A port is complete**, and the two items above are recorded as separately remaining
real-hardware behavior checks.

## Raspberry Pi 4 branch (`feature/rpi4`, 2026-10-22)

Measured the same headless CLI, without code changes, on a Raspberry Pi 4 (4× Cortex-A72
@1.8GHz, 64-bit Raspberry Pi OS). Environment setup and measurement are pinned down by
`tool/rpi4/setup.sh` and `tool/rpi4/bench.sh`. Details in [rpi4_port.md](rpi4_port.md).

- Golden vector PASS — the Pi 4 also produces the same embedding as the real Android device.
- 4-thread total **1254.3ms** (about 3.6× the Q6A's 349.8ms). Live-camera (C270) recognition
  works, no memory leak.
- **Conclusion: functionality passes; AuraFace R100 is not suitable for real-time use on the
  Pi 4. The Pi 4 port is closed.** Lightweight embedders (SFace, etc.) won't be added just for the
  Pi 4; they'll be evaluated separately in low-end device work.
- The two remaining Q6A checks above (rejecting others, SIGINT) are also unverified on the Pi 4 —
  follow-up items shared by all Linux targets.
