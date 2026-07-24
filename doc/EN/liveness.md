🇰🇷 [한국어 원문](../KR/liveness.md)

---

# Basic liveness: blink detection (2026-06-26)

A minimal liveness check added to facekit's Free tier — the most basic line
of defense of "holding up a static photo doesn't pass." The
`LivenessDetector` interface
([lib/src/core/contracts.dart](../../lib/src/core/contracts.dart)) is shared
between Free and Pro; this adds the Free-tier implementation,
`BlinkLivenessDetector`.

## Why a new model was needed

The existing detector (BlazeFace) only gives each eye as a **single point**
(one of the 6 keypoints in `DetectedFace.landmarks`). EAR (Eye Aspect Ratio)
-based blink detection needs the upper/lower eyelid contour coordinates,
which simply cannot be computed from a single point. So a new stage was
added after detection: **MediaPipe Face Landmarker (a 478-point face
mesh)**.

- Source: `unzip`ped from `face_landmarker.task` (the official MediaPipe
  release, Apache 2.0), extracting only `face_landmarks_detector.tflite`
  from inside it — the face detector and blendshape model bundled in the
  same archive were excluded, since BlazeFace already handles detection and
  blendshapes aren't used.
- Same project, same license as BlazeFace (Apache 2.0, commercially usable),
  so it's **bundled without BYOM** — registered under
  `assets/models/face_landmark_478/` and in `pubspec.yaml`'s
  `flutter.assets`.
- Input is 256×256 RGB, `(pixel-0)/255` normalization. Output is one
  478×3 (x,y,z) landmark tensor plus a face-confidence scalar (unused,
  discarded with the `zeroTensor()` helper built for AdaFace).

## Pipeline

```
DetectedFace (BlazeFace box)
  → MediaPipeFaceLandmarker.detectLandmarks()   [lib/src/landmark/face_landmarker.dart]
      crop the box as a square with a 25% margin → resize to 256×256 → infer
      → map the 478 points from crop-space back to the original image's coordinates
  → FaceLandmarks (478 Point3D values)
  → BlinkLivenessDetector.update(landmarks, timestampMs)  [lib/src/liveness/blink_liveness_detector.dart]
      pick 6 points each for the left/right eye, compute eyeAspectRatio() (core/math.dart) → average EAR
      → if it drops below a threshold (0.2 by default) and rises back within 1 second, that's "one blink"
      → LivenessState.passed once at least one blink happens within the observation window (4 seconds by default)
```

`eyeAspectRatio()` is Soukupová & Čech's (2016) published formula, and the
6-point eye indices (out of the 478-point topology: right eye
`[33,159,158,133,153,145]`, left eye `[362,380,374,263,386,385]`) are a
publicly documented standard indexing (used identically by several public
implementations, e.g.
[Pushtogithub23/Eye-Blink-Detection-using-MediaPipe-and-OpenCV](https://github.com/Pushtogithub23/Eye-Blink-Detection-using-MediaPipe-and-OpenCV)).

## Threshold — this time a published recommendation, not a measurement

For ArcFace/AdaFace, the threshold was newly derived by directly measuring
EER against 200 LFW pairs
([adaface_verification.md](adaface_verification.md)). This EAR threshold
(0.2) **did not get that level of real measurement** — there's no sample of
actual blinking-face footage available in this environment. It's simply the
published recommended starting value from the Soukupová & Čech paper.
CLAUDE.md's rule 6 ("tune thresholds from scratch") is meant to forbid
carrying over a value from a non-public source such as a previous
employer's code; since this is a constant from a public paper, it isn't a
violation of that rule — but **the limitation of it being an unverified
value clearly remains.** Recapturing real blink footage on a real device and
re-measuring is the next step.

## Example app integration (added 2026-06-27)

`MediaPipeFaceLandmarker` + `BlinkLivenessDetector` are now wired into
`example/lib/main.dart`. detect→landmark→liveness now runs on every camera
frame (regardless of the enroll/identify button state), and the result is
drawn via `FaceOverlayPainter`
([example/lib/face_overlay.dart](../../example/lib/face_overlay.dart)) on
top of the `CameraPreview`:

- A face box + 6 dots each for the left/right eye (the points used for EAR)
  are overlaid.
- A yellow box + "please blink" while liveness is `pending`; a green box
  once it's `passed`.
- **Both `enroll`/`identify` defer matching until liveness is `passed`** —
  holding up a static photo never triggers a blink, so it never reaches the
  matcher. This is the actual code path behind the "hold up a photo → gets
  blocked" demo scenario.
- On a successful match, the name + similarity is also shown in the box
  label (separately from the status text).

`MediaPipeFaceLandmarker`/`BlinkLivenessDetector` were added to the
library's public API (`lib/facekit.dart`) exports — previously neither
could be imported from outside the package (including from the example
app).

### Coordinate transforms — the trickiest part

The `CameraImage` that `startImageStream` delivers is in the **raw sensor
orientation** (typically a landscape shape like 1920×1080) as-is. The
`CameraPreview` widget applies an additional correction on top of the
native preview via a `RotatedBox`, keyed off
`controller.value.deviceOrientation` (the same logic as camera 0.11.4's
`CameraPreview._getQuarterTurns()`, Android-only — iOS's native preview is
already in the correct orientation). So drawing box/landmark coordinates
as-is would be misaligned whenever the device isn't held in its default
orientation (`portraitUp`).

`face_overlay.dart`'s `mapImagePointToPreview()` transforms coordinates by
**reimplementing the exact same rotation table** as the camera package
(ported directly from the camera package's source, not guessed). The front
camera also gets a horizontal mirror applied (the selfie convention).

## Limitations (honestly)

- **This defends against a static photo, and nothing more.** A flat photo's
  EAR never changes at all, so it can't pass. However:
  - A photo/mask with holes cut out for the eyes isn't defended against (if
    real eyes show through, EAR genuinely changes).
  - A replay attack using recorded video of a blinking face isn't defended
    against either (EAR genuinely changes in the video too).
  - Defending against these attacks is left to the Pro tier's more
    sophisticated multi-signal liveness (a stronger implementation plugs in
    behind the same `LivenessDetector` interface).
- The crop margin (25%), max blink duration (1000ms), and observation
  window (4s) are all arbitrary starting values — no real measurement
  behind them.
- `face_landmarker_smoke_test.dart` only confirms the model spits out 478
  points without crashing, on a synthetic gray image. Whether landmark
  coordinates actually land on the real eye/mouth positions for a real
  face photo — this environment has no camera/real device, so **it could
  not be visually verified.**
- **The coordinate-transform logic couldn't be visually verified on a real
  device for the same reason.** The rotation/scale formulas in
  `mapImagePointToPreview()` themselves are covered by unit tests
  (`example/test/face_overlay_test.dart`, 11 corner-point mapping cases)
  and follow the camera package's rotation table exactly, but whether the
  front-camera mirroring direction or the `deviceOrientation` callback
  actually arrive as expected on a real device couldn't be confirmed
  without a camera/real device. The next step needs a real device to
  visually confirm the mirroring direction and rotation compensation.
- Since detect+landmark now always runs every frame (previously detection
  only ran during enroll/identify), CPU usage increased — whether this
  causes frame drops on a real mobile device is unmeasured (`doc/KR/benchmark.md`
  is also not a real-device measurement at this point — same limitation).

## Tests

- `test/core/math_test.dart` — the pure `eyeAspectRatio` function (synthetic
  open-eye/closed-eye coordinates)
- `test/image/image_converter_test.dart` — `cropFaceImage` boundary clamping
- `test/landmark/face_landmarker_smoke_test.dart` — loads the actual bundled
  model and confirms it outputs 478 points (not BYOM, so this always runs,
  no graceful skip)
- `test/liveness/blink_liveness_detector_test.dart` — verifies the state
  machine against a synthetic EAR sequence
- `example/test/face_overlay_test.dart` — tests the pure coordinate
  rotation/scale/mirroring functions (the only part that can be verified
  without a real device)
