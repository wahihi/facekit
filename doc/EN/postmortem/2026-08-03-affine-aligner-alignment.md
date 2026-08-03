🇰🇷 [한국어 원문](../../KR/postmortem/2026-08-03-affine-aligner-alignment.md)

---

# The day the alignment matrix was wrong — four stacked bugs

**Date:** 2026-08-03
**Trigger:** This didn't start from a reproduced symptom — it started from a
**code review**. Re-reading the public repo and cross-checking
`AffineAligner`'s Umeyama/SVD implementation against the Umeyama (1991) paper
turned up two suspicious spots before the code was even run. Following that
thread all the way to a real device and back turned up **four completely
different layers of bugs**.
**Conclusion (spoiler):** (1) an SVD sign bug, (2) a missing covariance
normalization, (3) a wrong front-camera rotation formula, (4) a BlazeFace
landmark-correspondence bug (swapped eye order plus a forced 5th point).
All four had different root causes, and fixing one at a time kept producing
"better, but still wrong" three times in a row. The aligned crop was finally
confirmed by eye to be an upright, recognizable face, and a device
re-measurement showed genuine-match similarity accepted on all 46 attempts
(0.776–0.979, mean 0.906).

This project is built in collaboration with [Claude
Code](https://github.com/anthropics/claude-code), and so was this debugging
session — wrong hypotheses and missed predictions are left in as-is.

## 0. Background — this started as a code review

Earlier sessions had covered manifests, normalization constants, licensing,
ONNX↔TFLite conversion, benchmarks, and report wording. `affine_aligner.dart`
had never been opened. This session started with a request to clone the
public repo and explain the actual code, which turned into cross-checking
`AffineAligner`'s 5-point similarity-transform implementation against the
paper's own verifiable numeric example in §III.

At that stage, **just from reading the code, without running anything**, two
candidates were flagged:

- `_svd2x2`'s `vt` sign — the comment correctly says
  `rot(-theta) transposed = rot(theta)`, but the code used `rot(-θ)`
  verbatim.
- `_umeyamaSimilarity` had `srcVar /= n;` but was missing the equivalent
  `/n` on the four covariance terms (`cov00`–`cov11`).

The same review produced a "6 things to check" prediction list (argument
order, whether Σxy pairs correctly with `R=USVᵀ`, a suggestion to fold in a
`det(U)·det(V)` reflection correction, singular-value ordering, missing unit
tests, etc.). Checked against the real code later, **only 2 of those 6
predictions actually hit**, the `det(U)·det(V)` suggestion turned out not to
apply to this implementation at all (see §1), and **the most serious bug
(the `vt` sign) wasn't on that list to begin with.** One more case in this
project of theoretical static analysis not being a substitute for actually
running the code.

## 1. Verifying the static analysis — an independent Python re-implementation

Rather than trust the review outright, `_svd2x2`/`_umeyamaSimilarity` were
ported to Python line-for-line and cross-checked against an independent,
NumPy-standard-SVD-based Umeyama implementation.

**Confirming the `vt` sign bug** — reconstructing `U·diag(s)·Vᵀ` for 5 random
2×2 matrices, the current code failed to reproduce the original matrix every
time (`atol=1e-8`), and flipping the sign to `[vCos, -vSin, vSin, vCos]`
fixed every case.

**Confirming the missing `/n`** — fitting `arcface112Ref` onto itself (an
identity case) produced `sigma=5.0` from the current code, exactly matching
the point count `n=5` (the numerator was never normalized while the
denominator was, inflating the scale by a factor of n).

**Reproducing the paper's example** — with both bugs fixed, the 3-point
example from Umeyama (1991) §III (`src=[(0,0),(1,0),(0,2)]`,
`dst=[(0,0),(-1,0),(0,2)]`) produced `c=0.72111`,
`R=[[0.6,0.4],[-0.4,0.6]]`, `t=(-0.8,0.4)` — matching the paper's published
values exactly, and matching an independent NumPy-standard-SVD
implementation to the decimal place.

**The reflection guard was dead code, but happened to be correct anyway** —
`_svd2x2` always constructs `U` and `Vt` as pure rotations (det=+1), so
`detU·detVt` can never be negative in this implementation; the reflection
branch (`sign`) never fires. And yet, feeding a genuinely mirrored
(left-right flipped) input still produced an `R` with `det=1` (correctly not
treated as a reflection) — this closed-form SVD formula allows the singular
value `s1` to come out negative, and the reflection information is already
baked into that sign. The convention differs from a generic SVD library, so
the earlier session's "fold this into a `det(U)·det(V)` correction" advice
turned out not to apply here.

**Why the existing tests couldn't catch any of this structurally** — the
existing "identity transform" test in
`test/alignment/affine_aligner_test.dart` used a solid-colour image, and a
solid-colour image stays the same solid colour under *any* affine transform
(5x zoom, a 180° flip, doesn't matter). The test is named "identity" but
never actually checks whether the transform is one. Worse, the
`_identityFace()` helper it uses feeds `arcface112Ref` (5 points) as
landmarks, and `_pickFivePoints` duplicates the 4th point as the 5th whenever
`lm.length < 6`, producing `src=[p0,p1,p2,p3,p3]` — not a genuine identity
case to begin with.

## 2. Nailing it down — adding a diagnostic test

Theoretical verification wasn't enough on its own, so a
`@visibleForTesting` hook (`umeyamaSimilarityForTest`) was added to
`AffineAligner` to let tests call the private solver directly, along with
this diagnostic:

```dart
test('DIAG: ref → ref must be identity', () {
  final m = umeyamaSimilarityForTest(arcface112Ref, arcface112Ref);
  expect(m[0], closeTo(1, 1e-6));
  expect(m[4], closeTo(1, 1e-6));
});
```

Actual `flutter test` output (before the fix):

```
a=-4.9996839220786224 b=-0.056219919143706894 tx=340.1815074057739
c=0.056219919143706894 d=-4.9996839220786224 ty=428.2321675657798
```

This matched the Python-ported prediction to the decimal place — the `5` in
the magnitude was the point count `n=5` (missing covariance `/n`), and the
negative sign was the `vt` flip. After the two-line fix, the same test:

```
a=0.9999999999999999 b=0.0 tx=7.105427357601002e-15
c=0.0 d=0.9999999999999999 ty=1.4210854715202004e-14
```

An identity matrix down to floating-point noise. The full test suite (80
tests) passed with no regressions.

**Fix:**
```dart
// _svd2x2
- final vt = [vCos, vSin, -vSin, vCos];
+ final vt = [vCos, -vSin, vSin, vCos];

// _umeyamaSimilarity
  srcVar /= n;
+ cov00 /= n; cov01 /= n; cov10 /= n; cov11 /= n;
```

The `detU/detVt/sign/scaledS` dead code was left untouched this time (see
§1 — it's always harmlessly bypassed in this implementation, so touching it
now would only widen the verification surface for no benefit).

## 3. Reproducing it on a real device — both math bugs fixed, crops still wrong

Running `flutter run --dart-define=FACEKIT_VERBOSE_DEBUG=true` on a real
Pixel 7, enrolling a face, and pulling both the `[AffineAligner] DEBUG
MATRIX` log and the base64-chunked crop dump. Independently recomputing the
logged matrix with a NumPy-standard Umeyama implementation matched **to the
decimal place** — confirming the fixed code was genuinely running on-device.
The problem was what came next.

Reconstructing the crop from its base64 chunks into a PNG showed a face
tilted diagonally, sitting toward the bottom of the frame, with what looked
like window blinds filling much of the background. Pulling the rotation
angle out of the matrix: all 6 detections fell in the 116°–173° range, with
the sign flipping once. Scale came out `sigma≈0.05–0.12` — the algorithm had
concluded the detected 5-point spread was 8–20x larger than the reference
layout.

**Ruled out camera shake / motion blur.** The obvious question was "did you
shake the phone too fast?" The data didn't support it. Across all 6 frames,
the two points assumed to be eyes (indices 0, 1) were separated by only
0.6–16px horizontally (dx) but consistently 107–140px vertically (dy). Shake
would produce a different, random pattern per frame; getting the exact same
"narrow horizontally, wide vertically" shape 6 times in a row pointed at the
code, not the hand holding the phone.

## 4. The camera-rotation-formula bug

**Ruled out "was the phone held sideways?" too.** The user confirmed holding
it upright, selfie-style, the entire time — never tested landscape. And the
2026-07-24 postmortem had already fixed a camera-rotation bug once; checking
git history showed the rotation-related code in `image_converter.dart` /
`main.dart` **hadn't changed at all** since that fix (`aa525e6`). That same
postmortem documented verifying an upright crop under the exact same
front-camera default. The same code, under the same conditions, had been
correct then and looked wrong now — more likely a missed edge case than a
regression.

A temporary debug print was added to `_cameraQuarterTurns()` and measured on
device:

```
lensDirection=CameraLensDirection.front sensorOrientation=270 degrees=90 turns=1
```

`rotateFaceImage()`'s rotation logic itself checked out exactly against
NumPy's `rot90(k=-1)` (90° clockwise), so it wasn't at fault. The bug was in
which rotation amount got computed.

**The correct turn count was determined empirically.** The actual output
landmark coordinates under `turns=1` were inverse-transformed back through
`rotateFaceImage`'s math to recover the raw (pre-rotation, 720×480
landscape) coordinates, then all 4 candidate values (`turns=0,1,2,3`) were
applied forward to see which produced normal face geometry:

```
turns=0: dx=105.6 dy=9.6   (horizontal, ok)  <- but nose above eyes, mouth above nose (upside down)
turns=1: dx=9.6  dy=105.6  (vertical, wrong — current code)
turns=2: dx=105.6 dy=9.6   (horizontal, ok)  <- nose below eyes, mouth below nose (correct)
turns=3: dx=9.6  dy=105.6  (vertical, wrong)
```

Only `turns=2` produced both a horizontal eye axis *and* the correct
up/down ordering (nose below eyes, mouth below nose). The front-camera
formula was fixed accordingly:

```dart
// example/lib/main.dart, _cameraQuarterTurns()
- (360 - sensorOrientation) % 360
+ (450 - sensorOrientation) % 360
```

With `sensorOrientation=270`, this flips `degrees=90 (turns=1)` to
`degrees=180 (turns=2)`. This value wasn't re-derived from camera theory —
it's **empirically calibrated to this specific device and lens** (noted in
the comment). A textbook ML Kit-style formula would predict `turns=3` here;
the measured data said `turns=2` instead — possibly because the Flutter
`camera` plugin, or this device's firmware, doesn't behave exactly like the
raw Camera2 `SENSOR_ORIENTATION` characteristic assumes, though the exact
reason was never root-caused. **The back-camera branch was left untouched
and unverified** — there was no way to check it, since the back camera is
never used in practice (left as a follow-up, see §6).

Re-verifying: all 9 frames now showed a horizontal eye axis (dx=87–126px,
dy=9–24px). The camera-rotation problem was fixed — but the alignment
matrix's rotation angle was still sitting at 42°–90°.

## 5. The landmark-correspondence bug — swapped eye order plus a forced 5th point

`_pickFivePoints` took the first 4 of BlazeFace's 6 keypoints (eye, eye,
nose, mouth) as-is and synthesized a 5th from the ear midpoint:

```dart
static List<Point> _pickFivePoints(List<Point> lm) {
  final fifth = lm.length >= 6
      ? Point((lm[4].x + lm[5].x) / 2, (lm[4].y + lm[5].y) / 2)
      : lm[3];
  return [lm[0], lm[1], lm[2], lm[3], fifth];
}
```

Two separate things were wrong here.

**(1) The eye indices were reversed.** `blazeface_decoder.dart:21`'s comment
reads "Keypoint order: rightEye, leftEye, nose, mouth, rightEar, leftEar",
while `affine_aligner.dart:63-64`'s comment said the opposite ("leftEye[0],
rightEye[1]"). This was checked against real detected landmarks: fitting the
4-point eye+nose+mouth set from 9 real device frames against the 4-point
reference (eyes + nose + mouth-corner midpoint), **using the eye indices
as-is gave rotations of 42°–90° and an erratic scale of 0.07–0.13**, while
**swapping indices 0 and 1 dropped rotation to 13°–24° (a natural head-tilt
range) and stabilized scale to 0.30–0.42.** `blazeface_decoder.dart`'s
ordering was confirmed correct; `affine_aligner.dart`'s assumption had been
wrong.

**(2) Pairing an ear midpoint with a mouth corner was a bad correspondence
to begin with.** BlazeFace only provides a single (centre) mouth landmark,
while the ArcFace reference has two separate mouth-corner points. `lm[3]`
(mouth centre) was being forced onto the left mouth corner, and the ear
midpoint onto the right mouth corner. Ears sit near eye level — nowhere near
a mouth corner — and this bad correspondence dragged the least-squares fit
in a different direction on every frame, which is exactly why the rotation
angle bounced around 42°–90° instead of sitting at a fixed offset.

**Fix:** dropped from 5 points to 4 (eyes + nose + mouth-centre), and
collapsed the reference to match (the two mouth corners averaged into one
point).

```dart
// swap eye order, drop the ears
static List<Point> _pickFourPoints(List<Point> lm) {
  if (lm.length < 5) throw ArgumentError('Need ≥5 landmarks, got ${lm.length}');
  return [lm[1], lm[0], lm[2], lm[3]];
}

// collapse the 5-point reference to 4 (mouth-corner midpoint)
static List<Point> _fourPointReference(List<Point> ref5) {
  final mouthMid = Point((ref5[3].x + ref5[4].x) / 2, (ref5[3].y + ref5[4].y) / 2);
  return [ref5[0], ref5[1], ref5[2], mouthMid];
}
```

## 6. Final verification — the crop confirmed by eye on a real device

With both fixes (camera rotation + landmark correspondence) applied,
re-verified on-device once more. Rotation angles across 7 frames:
`22.8° / 15.2° / 17.0° / 15.2° / 7.8° / 17.1° / 18.5°` — all pointing the
same way, all in a natural range. Reconstructing the base64 crop dump as a
PNG and looking directly at it — **where window blinds used to fill the
frame, this time a bespectacled face filled most of it, with the
eyes-nose-mouth layout in the correct orientation.**

## 7. Converting to formal regression tests

The temporary diagnostic tests were reorganized into permanent regression
coverage (`test/alignment/affine_aligner_test.dart`):

- **Identity case**: identical point sets → identity matrix (the old DIAG
  test, made permanent)
- **Umeyama (1991) §III worked example**: `c=0.72111`, `t=(-0.8,0.4)`, etc.
  checked against the published values within 1e-3
- **Scale recovery**: fitting the ArcFace reference scaled 2.2x and rotated
  12° back onto itself recovers `scale=1/2.2` exactly (within 1e-6)
- **Reflection case**: a horizontally mirrored input still fits as `det(R)=1`
  (not a reflection)
- **Eye-order regression guards (both directions)**: feeding the real
  BlazeFace order (rightEye, leftEye, ...) round-trips to identity, and
  feeding the old (un-swapped) order measurably does *not* — pinning down
  the fix in both directions so a regression can't slip back in unnoticed

A new testing hook, `alignmentMatrixForTest()`, was added so `align()`'s
actual computation path (4-point pick → reference collapse → Umeyama fit)
can be exercised directly without an image or the pixel warp. The full test
suite grew to 85 tests, all passing.

## 8. A new diagnostic tool

`tool/analyze_alignment_log.py` was built from scratch. Given a
`FACEKIT_VERBOSE_DEBUG` log, it:

- reports the `_cameraQuarterTurns` value
- computes rotation angle and scale for every `LANDMARKS`+`DEBUG MATRIX`
  pair
- automatically flags four failure signatures: eyes separated more
  vertically than horizontally (a 90°-class camera-rotation signal), `a≠d`
  (a reflection/SVD-bug signal), `|rotation|>45°`, and scale outside
  `[0.05, 3.0]`
- optionally reconstructs the base64 crop dumps as PNGs via `--crops-out`

Run against a synthetic log reproducing the old broken state, it correctly
flagged both issues (vertically-separated eyes, a 172.6° rotation) and
exited with code 1.

Run against a longer real-world session (26 detections), it flagged 10
(rotation 46°–86°). Investigating further, the flagged detections weren't
scattered randomly — they clustered in short consecutive bursts (#11–13,
#23–25), and the worst case (#12, 83.6°) had its nose landmark sitting
*above* the eye line when inspected directly. That's not something the
alignment math can produce a bug for — it reflects BlazeFace itself
producing poor landmarks under an extreme pose or fast motion in that
moment. Concluded this wasn't a regression, just an inherent limitation of a
lightweight single-stage detector (a pose-sanity gate — reject a frame if
its fitted rotation is too large — would be a new defensive feature, not a
bug fix, and was left out of scope; see §6 of the open items below).

## 9. Re-measuring similarity and benchmark timing

**Similarity** — one real-device enroll followed by 46 identify attempts
across varied pose/distance. Result: **all 46 `accepted=true`**,
`min=0.776 max=0.979 mean=0.906 median=0.904`, a margin of at least `0.376`
above the `0.40` threshold. Comparable to or wider than the 2026-07-24
postmortem's baseline (0.81–0.94, 15 attempts, max 0.979 this time) —
**fixing the alignment bugs did not hurt matching quality.**

**Benchmark** — re-measured Pixel 7 CPU AuraFace timing: detection 45.2ms,
embedding 1332.1ms, full frame 1378.5ms. Somewhat higher than the two runs
already recorded in the docs (1135.3ms, 1238.1ms), but detection time (a
code path unrelated to alignment) moved by a similar amount too — read as
**device-condition noise (thermal, battery, background load)**, not a
regression, since today's fix only touches a 5–6-point SVD computation plus
a 112×112 warp, computationally negligible next to a >1-second ResNet100
embedding pass.

## 10. Summary — how the four bugs related to each other

```
Bug 1: _svd2x2's vt sign flipped        ┐
Bug 2: missing covariance /n            ┤ found via code review (static analysis),
                                         │ confirmed via Python re-implementation + an on-device unit test
                                         ┘

  Both fixed -> real-device crops still show wrong rotation/scale
       |
       v
Bug 3: front-camera rotation formula (should be turns=2, computed turns=1)
  -> confirmed by inverse-transforming landmark coordinates and comparing all 4 turns candidates

  Fixed -> eye axis normalized, but rotation angle still 42°-90°
       |
       v
Bug 4: reversed eye indices + a forced ear-midpoint-to-mouth-corner correspondence
  -> reduced to 4 points (eyes+nose+mouth-centre) + swapped eye indices

  Fixed -> crop confirmed correct by eye, rotation angle settles to 7.8°-24.3° (a natural range)
       |
       v
Final verification: 46/46 similarity accepted, no benchmark regression
```

All four were bugs at completely different layers (SVD math / camera
hardware configuration / landmark semantics), and because they were stacked,
fixing each one in turn produced "better, but still wrong" three times over.
The 2026-07-24 postmortem's lesson ("leave room for two causes to be
stacked") held up again here — four layers deep this time.

## 11. What this reinforced

- **The static code review caught a real bug, but its prediction list
  wasn't reliable on its own.** Only 2 of 6 predictions actually landed, and
  the most serious bug (the `vt` sign) wasn't on the list at all. It was the
  kind of thing "the comment and structure look right, so the behavior
  probably is too" can't catch — it only showed up once `U·diag(s)·Vᵀ` was
  actually multiplied out and compared against the input.
- **A worked numeric example from the paper made diagnosis dramatically
  cheaper.** Without a verifiable answer like `c=0.72111, t=(-0.8,0.4)`,
  there would have been no cheap way to ask "is this SVD correct?" The
  "identical point sets → identity matrix" property used repeatedly in §6
  was the same kind of thing — an oracle that needs neither the paper nor a
  reference implementation.
- **Not taking user-side hypotheses ("did you shake it?", "was it
  sideways?") at face value, and checking them against the data instead,
  saved time.** The same geometric pattern (dx≪dy) reproducing across all 6
  frames was itself evidence against a random cause like camera shake.
- **Checking git history quickly separated "regression" from "missed edge
  case."** The single fact that the rotation-related code hadn't changed
  since the 2026-07-24 fix was enough to drop the "the code broke again"
  hypothesis and narrow straight to "there was a condition that fix never
  covered."
- **The rotation angle's own instability — bouncing frame to frame instead
  of sitting at a fixed offset — was itself diagnostic.** A camera-rotation
  (hardware-configuration) bug should produce roughly the same angle on
  every frame; seeing it scattered across 42°–90° instead was the signal
  that the landmark correspondence itself was drifting differently on every
  frame, which turned out to be exactly the case.
- **Reconstructing and actually looking at the crop image was decisive
  again.** The matrix numbers alone were enough to tell the rotation was
  off, but "there are window blinds filling the frame" only became obvious
  from the actual image — the same conclusion as §8 of the 2026-07-24
  postmortem.

## Appendix: tooling left behind

- `kFacekitVerboseDebug` in `lib/src/core/debug_flags.dart` — reused as-is
  from the 2026-07-24 postmortem.
- `AffineAligner.umeyamaSimilarityForTest` / `alignmentMatrixForTest` —
  `@visibleForTesting` hooks that let tests exercise the private solver
  directly, without an image or the pixel warp.
- `tool/analyze_alignment_log.py` — a new script that, given a verbose log,
  reports the camera rotation value, computes per-frame rotation/scale,
  auto-flags anomalies, and reconstructs crop PNGs, all in one pass.

## Left open

- **The back-camera rotation formula is still unverified.** `turns=2` is an
  empirical value for this device's *front* camera; the back-camera branch
  was left on its old formula (`sensorOrientation % 360`) with no way to
  check it (the app is never used with the back camera in practice).
- **BlazeFace's landmark quality degrading under extreme poses** was found
  but deliberately left unfixed. A rotation-based pose gate ("skip a frame
  for matching if its fitted rotation is too large") would be a new
  defensive feature, not a bug fix, and was kept out of scope here.
- Both of the above, plus the already-deferred AuraFace impostor-pair
  threshold verification, are being rolled into a GitHub issue.
