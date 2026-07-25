🇰🇷 [한국어 원문](../KR/installation.md)

---

# Installation guide (from scratch)

Every step needed to go from a fresh clone of facekit to running the
example app on a real device, in order. Written against a brand-new PC with
no Flutter/Android dev environment at all, and only lists commands actually
verified in this environment (Ubuntu 22.04, Flutter 3.44.2).

> facekit itself is a **library (package)** — it has no runnable screen of
> its own. The runnable app lives separately in [example/](../../example/).
> Running `flutter run` from the repo root will correctly produce a
> `Target file "lib/main.dart" not found` error — follow this doc and run
> from inside `example/` instead.

## 0. System requirements

| Item | Requirement |
|---|---|
| OS | Linux / macOS / Windows (anything Flutter supports) — this doc is written against **Ubuntu 22.04 (Linux)** |
| CPU | x86_64 or ARM64, 64-bit required |
| RAM | 8GB+ recommended (16GB+ recommended if also using an Android emulator) |
| Free disk space | **10GB+** — Flutter SDK (~2.3GB) + Android SDK (~3GB) + build cache + model files |
| Build tools | JDK **17** (required by the Android Gradle Plugin), Android SDK (compileSdk **36**), Flutter **3.44.2** stable, Dart **3.12.2** |
| Test device | A real Android device (minSdk **24** = Android 7.0+) — one with USB debugging available is recommended |

> iOS hasn't been verified in this environment (neither build nor run has
> been confirmed). This doc covers Android only.

## 1. Prerequisite — install the Flutter SDK

If Flutter is already installed, skip to step 2.

```bash
# Unpack anywhere you like (e.g. ~/development)
cd ~
git clone https://github.com/flutter/flutter.git -b stable development/flutter
# or download the archive from https://docs.flutter.dev/get-started/install
```

Add it to PATH in `~/.bashrc` (or `~/.zshrc` for zsh):

```bash
export PATH="$HOME/development/flutter/bin:$PATH"
```

Open a new terminal or run `source ~/.bashrc`, then confirm:

```bash
flutter --version
# OK if it prints "Flutter 3.44.2 • channel stable ..."
```

## 2. Install the Android SDK / JDK

Installing Android Studio brings the SDK along with it, but the SDK alone
is also fine.

1. Install **JDK 17** (Temurin recommended): get it from
   https://adoptium.net, or via a package manager:
   ```bash
   # Ubuntu example
   sudo apt install openjdk-17-jdk
   ```
2. Install the **Android SDK cmdline-tools** — already present at
   `~/Android/Sdk` if Android Studio is installed. Otherwise, download just
   the cmdline-tools from
   https://developer.android.com/studio#command-tools and unpack them into
   `~/Android/Sdk/cmdline-tools/latest/`.
3. Add to `~/.bashrc`:
   ```bash
   export JAVA_HOME="$HOME/development/jdk17"          # your actual JDK 17 install path
   export ANDROID_HOME="$HOME/Android/Sdk"
   export ANDROID_SDK_ROOT="$HOME/Android/Sdk"
   export PATH="$JAVA_HOME/bin:$ANDROID_HOME/platform-tools:$ANDROID_HOME/cmdline-tools/latest/bin:$PATH"
   ```
   After applying it:
   ```bash
   "$JAVA_HOME/bin/java" -version   # should print openjdk 17.x
   ```
4. Install the needed SDK packages and accept the licenses:
   ```bash
   sdkmanager "platform-tools" "platforms;android-36" "build-tools;36.0.0"
   flutter doctor --android-licenses   # answer y to all
   ```
5. Check:
   ```bash
   flutter doctor -v
   ```
   You should see `[✓] Android toolchain`, `[✓] Connected device`, etc. The
   `Linux toolchain` entry (clang/ninja/gtk3 for desktop builds) **can be
   ignored if you're only targeting Android.**

## 3. Prepare a device — USB debugging

You need a device connected ahead of time to install straight to it with
`flutter run`.

1. On the phone: **Settings → About phone (device info)** → tap **Build
   number** 7 times in a row
2. **Settings → System → Developer options** → turn on **USB debugging**
3. Connect to the PC via USB → when **"Allow USB debugging?"** pops up on
   the phone → check **Always allow from this computer** and allow it
4. Confirm it's recognized from the PC:
   ```bash
   adb devices -l
   ```
   It's fine if the device serial shows up with state `device`.

   - **If it shows `no permissions (missing udev rules?)` (Linux-only
     issue)** — there's no udev permission rule for Google devices (vendor
     ID `18d1`):
     ```bash
     echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="18d1", MODE="0666", GROUP="plugdev"' \
       | sudo tee /etc/udev/rules.d/51-android.rules
     sudo udevadm control --reload-rules
     sudo udevadm trigger
     ```
     Then unplug and replug the USB cable and confirm the allow-prompt on
     the phone again. (Your account needs to be in the `plugdev` group —
     check with the `groups` command.)
   - **If it shows `unauthorized`** — you haven't tapped the USB-debugging
     allow prompt on the phone screen yet. Reconnect the cable and check
     for the prompt.

## 4. Quick start (the default path)

Once the environment (steps 0–3) is set up, **the four steps below are all
it takes to get enroll/recognize working** — no need to source or convert
an embedding model yourself. The default embedding model, **AuraFace**
(Apache 2.0, commercially redistributable), is fetched automatically from a
GitHub Release.

```bash
# 1. Clone
git clone https://github.com/wahihi/facekit.git
cd facekit

# 2. Install dependencies (for the example app)
cd example
flutter pub get

# 3. Download the embedding model (AuraFace, ~130MB, one-time)
bash ../tool/fetch_models.sh

# 4. Run
flutter run
```

`tool/fetch_models.sh` fetches the `.tflite` weight from a GitHub Release,
verifies its integrity via SHA256, and places it under
`example/assets/models/auraface/`. It's a no-op and exits immediately if
the file is already present, so it's safe to run again any number of
times.

Once `flutter run` launches, you'll see the camera screen. **Tap "등록"
(Register) to enroll a name, then tap "실시간 인식 시작" (Start live
recognition) to start re-recognizing right away.** If you want to use a
different embedding model (like ArcFace) or want to understand the
manifest structure in more depth, see
[Appendix A](#appendix-a--using-other-embedding-models-byom); for the
license of the bundled models, see
[Appendix B](#appendix-b--licensing).

## 5. Viewing logs (watching what's happening)

```bash
flutter logs
# or
adb logcat -s flutter:*
```

The pipeline's core steps (detection/alignment/embedding/matching/liveness)
log via `debugPrint` (only active in `kDebugMode`, not printed in a release
build). For example:
```
[onFrame] face found, score=0.656
[onFrame] liveness: LivenessState.passed
[FacePipeline] detect: 1 face(s)
[FacePipeline] embed: 512-dim vector
[FacePipeline] match: id=나 similarity=0.873 accepted=true
```

For more detailed internal pipeline logging (the alignment matrix, raw
embedding stats, etc.), turn it on with
`flutter run --dart-define=FACEKIT_VERBOSE_DEBUG=true` — it's off by
default and doesn't ship in a normal build at all.

## 6. Running the tests

```bash
flutter test               # unit tests for the root (facekit package) — needs its own flutter pub get at the root
cd example && flutter test # widget tests for the example app
```

## 7. Build modes reference (if you want release/profile)

`flutter run` builds in debug mode by default. If you want to look at
performance or need a distributable APK:

```bash
flutter build apk --debug      # for debugging — fastest build, no optimization
flutter build apk --profile    # for performance measurement — AOT-optimized, some debug tooling limited
flutter build apk --release    # for distribution — maximum optimization
```

The output location is the same pattern across all modes:
```
example/build/app/outputs/flutter-apk/app-{debug,profile,release}.apk
```

**The default embedding model (AuraFace) is `bundled`/`redistributable:
true`, so it loads fine even in a `--release` build.** However, if you
follow [Appendix A](#appendix-a--using-other-embedding-models-byom) and
switch to a research-tier model like `arcface_buffalo_l`,
`ModelManifest.assertLoadable()` **blocks loading the model at the code
level in release builds** (to protect the license — see
[Appendix B](#appendix-b--licensing) for exactly how). In that case, use
`--profile` to look at performance instead — its AOT optimization matches
release without tripping this guard. See [benchmark.md](benchmark.md) for
the full reasoning.

To install the APK directly:
```bash
adb install -r build/app/outputs/flutter-apk/app-profile.apk
```
If you'd rather install over the same Wi-Fi network without USB, you can
also spin up a temporary download server on the PC and fetch it from the
phone's browser:
```bash
cd build/app/outputs/flutter-apk
python3 -m http.server 8765 --bind 0.0.0.0
# On the phone's browser, visit http://<PC's LAN IP>:8765/app-profile.apk and download
```

## 8. Common issues

| Symptom | Cause / fix |
|---|---|
| `Target file "lib/main.dart" not found` | You ran `flutter run` from the repo root — `cd example` first |
| `dart:ffi` error on web (`flutter run -d chrome`) | `tflite_flutter` uses `dart:ffi`, which the web doesn't support — this SDK doesn't support web; run on Android (or iOS/desktop) |
| "Initialization failed: ... Model file is missing" (the app's first screen) | You likely haven't run `tool/fetch_models.sh` yet — see step 4 (Quick start) |
| Model fails to load in a `--release` build after switching to a BYOM model | See step 7 above — the research-tier model's license guard. Use `--profile` or `--debug` instead |
| `adb devices` shows `no permissions` | Add the udev rule from step 3 |
| `adb devices` shows `unauthorized` | You haven't tapped the USB-debugging allow prompt on the phone screen — reconnect the cable and check for the prompt |
| A single `CameraException(Disposed CameraController...)` log line on first launch | Appears to be one-off noise from installing an update over a running app with `adb install -r` (previous process cleanup timing) — safe to ignore if everything works afterward; if it recurs after a full restart, investigate separately |
| "Kotlin Gradle Plugin (KGP)" warning during the Gradle build | Related to the `camera_android_camerax` plugin, doesn't currently block the build — safe to ignore |
| `flutter`/`adb` commands not found | The PATH change wasn't loaded into the current shell — `source ~/.bashrc` or open a new terminal |

---

## Appendix A — Using other embedding models (BYOM)

For when you want to source and use a different embedding model (ArcFace,
AdaFace, etc.) instead of the default (AuraFace). facekit swaps embedding
models via a manifest-driven structure, so **the model itself is never
committed to the repo** — you just place a manifest.json + `.tflite` at the
expected path.

### A.1 Example — preparing ArcFace (buffalo_l)

`example/assets/models/arcface_buffalo_l/manifest.json` already exists in
the repo and is kept as a BYOM example. You just need to supply the
`.tflite` weight yourself.

```bash
ls example/assets/models/arcface_buffalo_l/
# It's normal to see only manifest.json, with no .tflite (excluded via .gitignore)
```

1. Get `w600k_r50.onnx` (or an equivalent ArcFace R100 weight) from the
   `buffalo_l` model pack at
   https://github.com/deepinsight/insightface.
   - This is a non-commercial research license — it must not be used
     commercially (see the `license` field in
     `example/assets/models/arcface_buffalo_l/manifest.json` for details).
2. Convert the ONNX to TFLite:
   ```bash
   # TODO: fill in the conversion command here (onnx2tf / onnx-tf / etc. — varies by model/tool version)

   ```
3. Place the converted file at exactly this path/name:
   ```
   example/assets/models/arcface_buffalo_l/w600k_r50.tflite
   ```
4. Switch the active model by changing the constants at the top of
   `example/lib/main.dart`:
   ```dart
   const _embedderDir = 'assets/models/arcface_buffalo_l';
   const _embedderFile = 'w600k_r50.tflite';
   ```
5. Running the app without this file present will fail model loading in
   `_setup()` and show an "Initialization failed: ..." message on screen —
   that's expected behavior, don't be alarmed.

### A.2 manifest.json schema

The fields parsed by `ModelManifest.fromJson()`
([lib/src/inference/model_manifest.dart](../../lib/src/inference/model_manifest.dart)):

```jsonc
{
  "name": "a name to identify the model",
  "family": "arcface",              // the adapter-selection key, see A.3 below
  "file": "weight_file.tflite",     // relative to the same folder as manifest.json
  "input": {
    "width": 112, "height": 112,     // the square input size the model expects
    "color": "RGB",                  // "RGB" | "BGR"
    "layout": "NHWC",
    "normalize": {
      "mean": [127.5, 127.5, 127.5], // per-channel (pixel - mean) / std
      "std": [127.5, 127.5, 127.5]
    }
  },
  "output": {
    "dim": 512,                      // embedding dimension
    "l2_normalize": true             // whether postprocessing L2-normalizes it
  },
  "alignment": {
    "type": "five_point_affine",
    "reference": "arcface_112"       // one of the reference point sets in affine_aligner.dart
  },
  "matching": {
    "metric": "cosine",
    "threshold": 0.40,               // read by CosineMatcher.fromManifest()
    "threshold_note": "record the measurement this threshold is based on here"
  },
  "license": {
    "tier": "bundled",               // "bundled" | "research" | "byom" | "licensed"
    "redistributable": false,        // whether it's allowed to load in a release build (see Appendix B)
    "source": "origin / source repository",
    "note": "explanation of the license terms"
  }
}
```

`input`/`output`/`alignment` values differ per model, so always check the
original model's training/preprocessing convention before filling them
in — for example, AdaFace uses `color: "BGR"`, and FaceNet takes 160
(not 112) input with a different normalization scheme (prewhiten)
entirely.

### A.3 Registering a new family (wiring up an adapter)

The `family` value is the key `adapterForFamily()` in
`lib/src/embedding/face_embedder.dart` uses to pick a pre/post-processing
adapter:

```dart
EmbedderAdapter adapterForFamily(String family) {
  switch (family) {
    case 'arcface':
    case 'adaface':
    case 'mobilefacenet':
    case 'auraface':
      return const ArcfaceAdapter();   // shared 112x112, (pixel-mean)/std convention
    case 'facenet':
      return const FacenetAdapter();   // 160x160, per-image prewhiten
    default:
      throw ArgumentError('No embedder adapter registered for family "$family"');
  }
}
```

- If your model shares the same input convention as an existing adapter
  (`ArcfaceAdapter`/`FacenetAdapter`) — as AuraFace did — **just add a new
  `case` and reuse the existing adapter.**
- If it needs completely different preprocessing (a different
  normalization formula, extra output tensors to handle, etc.), implement
  the `EmbedderAdapter` interface
  (`preprocess`/`postprocess`) in
  `lib/src/embedding/adapters/embedder_adapter.dart` with a new adapter,
  and register it in this `switch`.

## Appendix B — Licensing

### Bundled models

| Model | License | Source |
|---|---|---|
| BlazeFace short-range (detection) | Apache 2.0 | https://github.com/google/mediapipe |
| MediaPipe Face Landmarker (478-point, used for liveness) | Apache 2.0 | https://github.com/google/mediapipe |
| AuraFace (glintr100/ResNet100, default embedding) | Apache 2.0 | [fal/AuraFace-v1](https://huggingface.co/fal/AuraFace-v1) — the `.tflite` weight itself isn't committed to the repo (fetched via `tool/fetch_models.sh` from a GitHub Release) purely because of its size, not a license restriction |

All three have commercially usable licenses, so they work out of the box
with no BYOM steps needed. The additional models covered in Appendix A
(`arcface_buffalo_l` etc.) are mostly non-commercial research licenses that
you need to source yourself.

### What the `assertLoadable()` guard blocks

Each model's `manifest.json` has `license.tier`
(`bundled`/`research`/`byom`/`licensed`) and `license.redistributable`
(a boolean).
[`ModelManifest.assertLoadable()`](../../lib/src/inference/model_manifest.dart)
checks this every time a model is loaded (`TfliteFaceEmbedder.fromAsset`/`fromFile`):

```dart
void assertLoadable({required bool isReleaseBuild}) {
  if (isReleaseBuild && !license.redistributable) {
    throw StateError(
      'manifest "$name": license.redistributable=false (tier=${license.tier.name}, '
      'source=${license.source}) — this model must not be loaded in a release build',
    );
  }
}
```

In other words, **a model with `redistributable: false` is blocked from
loading at the code level in a release build (`kReleaseMode == true`)** —
this isn't just a "no commercial distribution" note left in the docs; the
app is forced to throw at that point even if such a model accidentally
ends up in a distribution build. `--debug`/`--profile` builds don't trip
this guard, so development and performance measurement with a
research-tier model can still continue.

`ModelManifest.validate()` checks this too — a contradictory combination
of `tier: "research"` with `redistributable: true` is rejected right at
manifest-parsing time.

---

For deeper material (architecture, benchmark methodology), see
[architecture.md](architecture.md), [benchmark.md](benchmark.md), and
[liveness.md](liveness.md).
