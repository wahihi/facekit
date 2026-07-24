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

## 3. Clone the repo

```bash
git clone https://github.com/wahihi/facekit.git
cd facekit
```

## 4. Install dependencies

facekit contains two independent Flutter projects in one repo — the
package itself (root) and the example app — **both** need their
dependencies fetched.

```bash
flutter pub get            # root (the facekit package)
cd example
flutter pub get            # the example app
cd ..
```

## 5. Prepare the embedding model (BYOM) — the app won't launch without this

facekit only bundles the detection model (BlazeFace) and the landmark
model used for liveness — **the embedding model is not in the repo** (see
the license policy in
[README.md](../../README.md#license--model-policy-byom)). The example app
defaults to using ArcFace (`buffalo_l`), whose weight file you need to
obtain and place yourself.

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
2. Convert the ONNX to TFLite (`onnx2tf`, `onnx-tf`, etc. — the exact
   conversion steps vary by model/tool version, so this doc doesn't cover
   them).
3. Place the converted file at exactly this path/name:
   ```
   example/assets/models/arcface_buffalo_l/w600k_r50.tflite
   ```
4. Running the app without this file present will fail model loading in
   `_setup()` and show an "Initialization failed: ..." message on screen —
   that's expected behavior, don't be alarmed.

> To swap in a different embedding model such as AdaFace, see
> `lib/src/embedding/adapters/` and
> [architecture.md](architecture.md). If you just want to look at the code
> structure without using any actual model, you can skip this step and go
> straight to step 6 — but in that case the example app will get as far as
> the camera preview and then not function at the enroll/identify step.

## 6. Build

Run from the `example/` directory.

```bash
cd example
```

### 6-1. During development — run straight on a device (the command you'll use most)

```bash
flutter run
```

If multiple devices are connected, check with `flutter devices` and target
one with `-d <device-id>`. This builds in debug mode and runs immediately,
with hot reload (the `r` key) available.

### 6-2. Build an APK only (when you just need the install file)

```bash
flutter build apk --debug      # for debugging — fastest build, no optimization
flutter build apk --profile    # for performance measurement — AOT-optimized, some debug tooling limited
flutter build apk --release    # for distribution — maximum optimization
```

The output location is the same pattern across all modes:
```
example/build/app/outputs/flutter-apk/app-{debug,profile,release}.apk
```

> ⚠️ **`--release` fails at run time in this example.** Since the ArcFace
> demo model's manifest has `redistributable: false`,
> `ModelManifest.assertLoadable()` **blocks loading the model at the code
> level in release builds** (to protect the license, see
> [lib/src/inference/model_manifest.dart](../../lib/src/inference/model_manifest.dart)).
> Use `--profile` to look at performance instead — its AOT optimization
> matches release without tripping this guard. See
> [benchmark.md](benchmark.md) for the full reasoning.

## 7. Install onto a device

### 7-1. Turn on developer options / USB debugging on the phone

1. **Settings → About phone (device info)** → tap **Build number** 7 times
   in a row
2. **Settings → System → Developer options** → turn on **USB debugging**
3. Connect to the PC via USB → when **"Allow USB debugging?"** pops up on
   the phone → check **Always allow from this computer** and allow it

### 7-2. Confirm recognition from the PC

```bash
adb devices -l
```

It's fine if the device serial shows up with state `device`.

- **If it shows `no permissions (missing udev rules?)` (Linux-only issue)**
  — there's no udev permission rule for Google devices (vendor ID `18d1`):
  ```bash
  echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="18d1", MODE="0666", GROUP="plugdev"' \
    | sudo tee /etc/udev/rules.d/51-android.rules
  sudo udevadm control --reload-rules
  sudo udevadm trigger
  ```
  Then unplug and replug the USB cable and confirm the allow-prompt on the
  phone again. (Your account needs to be in the `plugdev` group — check
  with the `groups` command.)

### 7-3. Install the APK

If connected via USB:
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

## 8. Viewing logs (watching what's happening)

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

## 9. Running the tests

```bash
flutter test               # unit tests for the root (facekit package)
cd example && flutter test # widget tests for the example app
```

## 10. Common issues

| Symptom | Cause / fix |
|---|---|
| `Target file "lib/main.dart" not found` | You ran `flutter run` from the repo root — `cd example` first |
| `dart:ffi` error on web (`flutter run -d chrome`) | `tflite_flutter` uses `dart:ffi`, which the web doesn't support — this SDK doesn't support web; run on Android (or iOS/desktop) |
| Model fails to load in a `--release` build | See 6-2 above — the BYOM demo model's license guard. Use `--profile` or `--debug` instead |
| "Initialization failed: ..." (the app's first screen) | Most likely step 5 (preparing the BYOM model) wasn't done — check the `.tflite` file's location |
| `adb devices` shows `no permissions` | Add the udev rule from 7-2 |
| `adb devices` shows `unauthorized` | You haven't tapped the USB-debugging allow prompt on the phone screen — reconnect the cable and check for the prompt |
| A single `CameraException(Disposed CameraController...)` log line on first launch | Appears to be one-off noise from installing an update over a running app with `adb install -r` (previous process cleanup timing) — safe to ignore if everything works afterward; if it recurs after a full restart, investigate separately |
| "Kotlin Gradle Plugin (KGP)" warning during the Gradle build | Related to the `camera_android_camerax` plugin, doesn't currently block the build — safe to ignore |
| `flutter`/`adb` commands not found | The PATH change wasn't loaded into the current shell — `source ~/.bashrc` or open a new terminal |

---

For deeper material (architecture, how to swap models, benchmark
methodology), see [architecture.md](architecture.md),
[benchmark.md](benchmark.md), and [liveness.md](liveness.md).
