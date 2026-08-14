🇰🇷 [한국어 원문](../../KR/postmortem/2026-08-14-camera-orientation-and-auto-zoom.md)

---

# The screen always looked fine — the ML pipeline was quietly using a different rotation

**Date:** 2026-08-14
**Trigger:** While validating the two-pass refinement from
[2026-08-13-landmark-jitter-frame-fill.md](2026-08-13-landmark-jitter-frame-fill.md)
on a real device, the user happened to notice a one-time status message
telling them to hold the camera in landscape to enroll, and actually did —
and enrollment/identification suddenly became dramatically more stable,
even though refinement barely triggered at all (1 out of 191 frames)
(similarity 0.69–0.873, 17/17 accepted, versus prior sessions swinging
0.13–0.87 with frequent false rejections). **Conclusion (spoiler):** The
cause wasn't refinement at all — the camera preview (what the user sees)
and the ML pipeline (the frames actually used for recognition) were using
**two different rotation code paths**. The preview follows the live device
orientation sensor and always looks correct; the ML path used a single
fixed value calibrated for "held portrait" only. The user had no way to
notice a mismatch, since the screen always looked normal — and how often
this mismatch actually occurred in practice still hasn't been verified.
That discovery led to two auto-zoom bugs (oscillation, an inverted
direction sign), a wrong "it's just slow" hypothesis, the real reason
detection confidence flickers near threshold, and finally a decision-free
centred-square-crop fix. **By the end, both orientations tested landed at
74.6% / 90% accepted, with refinement succeeding on a real device for the
first time (§13).**

This project was built in collaboration with [Claude
Code](https://github.com/anthropics/claude-code), and so was this session.

## 1. Starting point — actually following the "hold it landscape" hint

`example/lib/main.dart` had a status message shown once, right after
initialization finishes: `'준비 완료 — 카메라를 가로로 들고 "등록"을
눌러보세요.'` ("Ready — hold the camera in landscape and press
'Register'"). Every prior session had used a stand, held portrait. The user
noticed this message and actually held the phone in landscape by hand —
and enrollment/identification got dramatically better at both near and far
distances.

## 2. First hypothesis (wrong) — "this session just happened to be steadier"

The first instinct was "maybe this session had less physical movement" —
until the user pointed out the obvious contradiction: switching from a
*stand* (stationary) to *hand-held* (landscape) should mean *more* shake,
not less, yet the results improved. The "steadiness" theory couldn't
explain that.

## 3. What the log actually showed — not refinement, the face was just bigger to begin with

Pulling the face-to-frame-width ratio out of the landscape session's log
(`log1.txt`):

| Metric | Landscape hold (log1.txt) | Prior sessions |
|---|---|---|
| Mean face fraction | **41.8%** | Swinging in the 15–30% range |
| Samples in 30–59% | 152 of 191 | Wild swings near the 20% boundary |
| Refinement triggers | **1** of 191 | Frequent |
| Similarity | **0.69–0.873**, mean 0.833 | Swinging 0.13–0.87 |
| identify() outcome | **17/17 accepted** | Frequent false rejections |

Refinement had barely run at all, yet the results were dramatically
better — meaning the cause wasn't the software crop logic, it was that
**the face was simply being captured much larger to begin with**.

## 4. Why landscape makes the face bigger — there were two separate rotation paths

Re-reading `example/lib/face_overlay.dart`'s header comment surfaced the
key sentence: `CameraController.startImageStream` delivers frames in the
sensor's **raw** orientation, while the `CameraPreview` widget applies its
own separate rotation driven by **`CameraValue.deviceOrientation`** (the
live device orientation sensor) to render the screen —
`quarterTurnsForOrientation()` is that function.

`main.dart`'s `_cameraQuarterTurns()` (which rotates the frame handed to
the ML pipeline), by contrast, was computed purely from the camera
hardware's **fixed** `sensorOrientation` — a static value assuming "the
device is always held portrait" — and never read the live orientation
sensor at all.

In other words: the preview (what the user sees) auto-corrects to always
look right; the ML path (what actually drives recognition) kept using a
fixed rotation that's only correct for "held portrait." However the user
actually held the phone, the screen always looked fine, so there was
**no way to tell that the frame actually feeding recognition might be in
the wrong orientation.**

Given the camera sensor's field of view is inherently wider along one
axis: holding the phone the way the app assumed ("portrait") puts that
wide axis horizontally, capturing a lot of irrelevant background beside
the face (low face-to-frame-width ratio); holding it landscape instead
puts the wide axis vertically, leaving a much narrower horizontal field of
view that the face fills far more of. The "face-to-frame ratio" instability
this whole investigation has been chasing was, in this session at least,
purely a function of which way the phone happened to be held.

## 5. The fix — make the ML path follow the live orientation sensor too

Changed `_cameraQuarterTurns()` from a single fixed value to the sum of
the **already-verified fixed calibration (for portraitUp, measured on a
real Pixel 7)** plus the **same `quarterTurnsForOrientation(liveOrientation)`
delta the preview already trusted**:

```dart
final turns = (_baseQuarterTurns + orientationDelta) % 4;
```

`_baseQuarterTurns` preserves the existing fixed calculation exactly (so
the portraitUp case is 100% unchanged — no regression there), and
`orientationDelta` reuses a function the preview overlay already relied
on, rather than deriving a new rotation formula from scratch — deriving
sign/direction from theory alone, with no device to test against, would
have been an unverifiable gamble.

**One side effect found along the way**: since the ML image is now
already orientation-corrected (upright), the overlay painter applying
*another* `quarterTurnsForOrientation` rotation on top of it would double-rotate
the box — fixed by changing that call site to `quarterTurns: 0`. (Why this
never broke visibly before: `quarterTurnsForOrientation` returns 0 for
`portraitUp`, so the double rotation was silently "0 + 0" in that one case
— the overlay box may well have been misaligned in other orientations all
along, just never noticed.)

**Important — this fix is unverified on a real device.** The rotation's
sign (clockwise vs counter-clockwise) can't be confirmed from code alone
without hardware, so the most conservative available approach was taken
(reusing an already-measured, already-trusted function) — but all 4
orientations (portraitUp/Down, landscapeLeft/Right) still need to be
checked on-device, including confirming portraitUp is unchanged.

## 6. Also added — camera-level auto-zoom

This also clarified a hard limit of the software-crop refinement: the
camera already downsamples to 720×480 via `ResolutionPreset.medium` before
any of our code runs, so no amount of clever post-capture cropping can
recover detail the camera driver already discarded. `CameraController.
setZoomLevel()`, by contrast, narrows the **capture itself** against the
sensor's native resolution — a genuine resolution gain, not a re-crop of
already-lossy data.

Added a feedback loop (`_maybeAdjustZoom`) that nudges zoom toward keeping
the face fraction in a 30–60% band, with a 400ms cooldown and a fixed step
size so it settles instead of hunting every frame. When zoom is already
maxed out and the face is still too small (user too far away even at max
zoom), a persistent `_framingHint` ("please come a bit closer to the
camera") is shown — also fixing the fact that the old "hold it landscape"
message lived in `_status`, a single field every other message
immediately overwrites, making it effectively invisible after the first
frame.

## 7. Timeline — cause and effect

```
Question: refinement barely ever triggered, so why did landscape dramatically improve results?
   └→ First hypothesis (refuted by the user): "this session was just steadier" — contradicts stand→hand-held meaning MORE shake, not less
   └→ Log analysis: mean face-to-frame-width ratio genuinely jumped from a swinging 15-30% to a stable 41.8%
   └→ Re-reading face_overlay.dart: the preview follows live deviceOrientation; the ML path (_cameraQuarterTurns) used a fixed value only
   └→ Hypothesis: the screen always looks normal regardless of how the phone is actually held, but the ML frame can be silently wrong once the "held portrait" assumption breaks
   └→ Fix: make the ML path apply the same live orientation delta (quarterTurnsForOrientation) the preview already used
        └→ Side effect found: the overlay was now double-rotating — fixed with quarterTurns: 0
   └→ Also added: camera-level auto-zoom + a persistent framing hint
   └→ Still needed: verify all 4 orientations on a real device
```

## 8. Lessons learned

- **"The screen looks fine" is not evidence that the processing pipeline
  is fine.** When the preview and the ML path compute rotation via
  separate code, what the user sees as feedback and what the model
  actually receives can diverge completely — and that divergence is
  invisible from the screen alone. The safest fix is making both paths
  share the same source of truth (live deviceOrientation).
- **When a hypothesis gets refuted by the user's direct pushback, that
  pushback is itself the next clue.** "Less movement" collapsed the
  moment the user pointed out that switching from a stand to hand-held
  should mean more shake, not less — and that single correction is what
  redirected the investigation toward the real cause.
- **Post-capture software cropping and camera-level zoom are
  fundamentally different.** A capture-stage resolution cap (like
  `ResolutionPreset`) is a hard ceiling no amount of clever downstream
  cropping can get past. When chasing an image-quality problem, check
  *which stage* the information was already lost at before optimizing
  anything downstream of it.
- **A status hint that lives in a single field every other message
  overwrites is effectively invisible.** The correct guidance had been
  flashing on screen for exactly one frame, session after session — an
  important persistent hint needs its own state, separate from anything
  that gets overwritten.
- **A rotation formula that's never run on real hardware should be
  labeled "pending verification," not "plausible."** Reusing an
  already-trusted function (the preview's `quarterTurnsForOrientation`)
  minimized risk, but until all 4 orientations are checked on-device,
  both this document and the code comments stay marked as "hypothesis +
  best conservative implementation," not "done."

## 9. First real-device pass — auto-zoom was oscillating

The first real-device run (`log2.txt`, 2026-08-14 16:30) produced "keeps
recognizing then losing it, over and over." The user asked whether
switching back to BlazeFace's 4-point path plus a digital zoom would be
more stable instead — but the log showed the cause wasn't the detector at
all, it was **the auto-zoom control loop itself oscillating**:

```
zoom=1.20 -> faceFraction=0.679 (68%, above the 30-60% target)
zoom=0.90 -> faceFraction=0.249 (25%, below target)
zoom=1.20 -> faceFraction=0.719 (72%, above again)
zoom=0.90 -> faceFraction=0.270 (27%, below again)
```

It never once landed inside the target band — every correction overshot
to the opposite side. That oscillating stretch overlapped almost exactly
with where `accepted=false` clustered; a calmer stretch elsewhere in the
same log (where zoom wasn't moving) had 8 consecutive accepted matches —
recognition was only stable when zoom was stable.

Two causes compounded: (1) the 400ms cooldown was shorter than this
device's actual zoom-settle latency, so the next frame's faceFraction
reading still reflected the *previous*, not-yet-applied zoom level,
making each correction react to stale data and overshoot; (2)
`getMinZoomLevel()` returned below 1.0 (0.8958...), and on many devices
the region near 1.0 crosses into a physical **ultra-wide lens switch** — a
discontinuous FOV jump, not smooth zoom. That's why a 33% zoom-level
increase (0.90->1.20) produced a >2x jump in faceFraction: it crossed that
seam.

**Fixed the zoom control loop instead of switching detectors** — this
oscillation happens upstream of the detector (in the camera capture
itself), so any detector would have hit the same problem:

- Clamp `_minZoom` to never go below 1.0 (avoids the lens-switch seam)
- Cooldown 400ms -> 900ms (longer than the observed settle lag)
- Step 0.3 -> 0.15 (smaller residual overshoot)
- Require 3+ consecutive frames out of range in the *same* direction
  before reacting (a single noisy frame can't retrigger it)

This is also still unverified on a real device — the next test needs to
confirm zoom actually settles now.

## 10. Second real-device pass — the zoom direction sign was inverted

Re-running with `log3.txt` (2026-08-14 16:48), the user reported that
moving *closer* to the camera made it zoom *in further*. The hysteresis
logic added in §9 to stop oscillation had the direction sign backwards:

```dart
final direction = faceFraction < _targetFaceFractionLow ? -1   // too small -> -1 (zoom-out direction)
    : faceFraction > _targetFaceFractionHigh ? 1 : 0;            // too big -> +1 (zoom-in direction)
```

Even though the face was already well above the 30-60% target (66-95%)
the whole session, zoom never once came back down — it climbed from 1.15
to 3.40 monotonically, with one frame reaching 95.5% face-width (nearly
the entire frame). Fixed by flipping the ternary (too small -> +1/zoom
in, too big -> -1/zoom out). The §9 `_minZoom` clamp to 1.0 was confirmed
working in this same log (`range 1.0-10.0`) — this was a separate bug
introduced by the same refactor, not a recurrence of the earlier one.

The same session also clarified liveness: of 238 logged EAR readings,
only 2 (0.8%) dropped below the 0.2 "closed" threshold, and `blinkCount`
only reached 1 thirteen times — the EAR signal itself looked fine, the
user simply wasn't blinking much during this particular test. Combined
with 37 `face lost` events (each discarding any in-progress blink), this
made liveness and recognition-stability issues hard to tell apart. Added
a temporary `_kLivenessEnabled = false` compile-time flag to gate the
blink check out entirely (every frame is treated as already `passed`,
and `MediaPipeFaceLandmarker.detectLandmarks` isn't even called) while
recognition stability is verified in isolation — meant to be flipped back
on afterward.

## 11. Timing logs ruled out "just slow," found the real cause

With liveness off, re-ran on-device (`log4.txt`) to check whether "fails
to detect while still, works after moving" was caused by identify()'s
redundant detect() call slowing frame processing down. It wasn't — most
`onFrame` calls took 43-99ms (median 55ms), and `pipeline.identify()`
was never even called in this session (confirming this particular run
had no real recognition data at all).

The actual cause showed up in the `face found` transition scores instead:
0.450-0.532, right against the detection threshold
(`score_threshold=0.45`). At that margin, ordinary frame-to-frame sensor
noise alone is enough to flip the score above/below threshold, producing
the found/lost/found/lost flicker — the same class of problem as
BlazeFace's threshold-flicker from the 2026-08-12 postmortem, just
triggered by distance instead. This also clarified the chicken-and-egg
problem: auto-zoom needs to make a decision exactly where detection is
least stable to make one.

## 12. A more fundamental fix — crop the first pass to a centred square

At the user's suggestion, changed the *first* pass (not just refinement,
the bbox-dependent second pass) to crop to a **centred square** before
detection (`centerSquareCropRegion`, `lib/src/detection/yunet_decoder.dart`).
For a 720×480 frame:

- Before: 4.5x squeeze on x, 3x on y (asymmetric — the face gets slightly
  squashed sideways)
- After: a centred 480×480 crop squeezes both axes uniformly by 3x, no
  distortion
- For a face at 15% of the frame, the model's effective face width goes
  from 24px to 36px — a 50% gain

Why this is more fundamentally sound than refinement: **it depends on no
prior detection at all.** Refinement crops around the first pass's own
bbox, which is exactly imprecise when the face is small — and checking
every real-device trigger logged so far (5 total), **all of them fell
back**; refinement has never once actually succeeded on a real device.
A centred crop needs no decision, so it can't inherit that failure mode.
Falls back to the whole uncropped frame if nothing's found in the crop
(e.g. an off-centre face).

Added a persistent framing guide to the example app
(`example/lib/face_overlay.dart`, `main.dart`): a dashed box, sized to
match that crop region (with an 85% margin), drawn on the camera preview
at all times — regardless of whether a face is currently detected — with
a "fit your face inside the box" label, so the user has something to aim
for even before the first detection.

Also added the refinement crop region and pass-1 bbox to the existing
fallback debug log, so the next real-device test can show directly
whether the crop is actually missing the face.

## 13. Final real-device verification — both orientations, refinement's first success

**Portrait (`log5.txt`, 2026-08-14 18:49).** First real-device run with the
centred crop + guide box in place:

- `CENTERCROP hit` 201 / `fallback` 143 (58% hit rate) — fallbacks
  clustered early in the session (before the user had settled into the
  guide box), hits were far more common afterward.
- **Refinement succeeded on a real device for the first time** — triggered
  twice, **applied both times** (0 fallbacks, versus 5 triggers / 5
  fallbacks previously):
  ```
  REFINE applied: score 0.457 -> 0.759, face fraction 18.2% -> 37.3%
  ```
  This confirms the hypothesis: a more accurate first-pass detection (now
  via the centred crop) gives refinement a bbox worth cropping around.
- 71 `match:` calls, **53/71 (74.6%) accepted**. Most rejections clustered
  just under the 0.211 threshold (0.13-0.21), not wildly off. Mean
  similarity 0.279, stdev 0.104.

**Landscape (`log6.txt`, 2026-08-14 19:04, `landscapeRight`).** Orientation
handling confirmed working correctly:
`liveOrientation=DeviceOrientation.landscapeRight orientationDelta=1
turns=3` (different from portrait's turns=2). Results were even better
than portrait:

| Metric | Portrait (log5) | Landscape (log6) |
|---|---|---|
| Mean similarity | 0.279 | **0.489** |
| identify accept rate | 74.6% (53/71) | **90% (18/20)** |
| REFINE | 2/2 succeeded | 1/1 succeeded |
| CENTERCROP hit rate | 58% | 32.7% (yet better results) |

Landscape being better matches §3's earlier finding — real face-width
fraction (from TEMP DEBUG LANDMARKS) averaged 36.5% (min 28.9%), much
larger and more stable than portrait (the same physical FOV-alignment
effect).

Interesting wrinkle: the centred-crop hit rate was *lower* in landscape
(32.7% vs 58%) despite better final results. Cause: landscape rotates the
`FaceImage` to a portrait-shaped `480×720`, and the centred crop keeps
only the middle 480 of that 720-tall axis, cropping top/bottom. If the
face isn't dead-centre on that axis, it falls outside the crop and
fallback fires more often. But the fallback (retry on the whole frame)
covers for it — since the face is already large (28-86%) in the whole
frame regardless, the fallback pass finds it easily anyway. **The safety
net worked exactly as designed.** A vertical-centring tweak for tall
frames is possible but deprioritized, since fallback is already
compensating well.

## 14. Full recap — what was done / learned / achieved this session

**What was done (chronological)**
1. Started validating the two-pass refinement from
   [2026-08-13-landmark-jitter-frame-fill.md](2026-08-13-landmark-jitter-frame-fill.md)
   on a real device.
2. The user actually followed the "hold it landscape" hint, surfacing the
   camera orientation bug (§1-5) → fixed the ML path to follow live
   orientation.
3. Added camera-level auto-zoom (§6) → found and fixed an oscillation bug
   on-device (§9) → found and fixed an inverted direction sign (§10).
4. Directly tested the "is it just slow?" hypothesis with timing logs →
   refuted it, found the real cause was detection confidence flickering
   near threshold (§11).
5. Temporarily disabled liveness to isolate recognition-stability testing
   (end of §10).
6. At the user's suggestion, fixed the first pass fundamentally with a
   centred square crop, added an on-screen guide box (§12).
7. Re-verified on-device in both portrait and landscape; refinement
   succeeded for the first time (§13).

**What was learned (key insights)**
- Small face = few pixels = vulnerable to sensor noise — both landmark
  jitter and detection-confidence flicker come from this one mechanism
  (the 2026-08-13 doc and §11).
- "The screen looks fine" is not evidence the ML pipeline is fine — the
  preview and ML path being separate code paths was this session's
  biggest single finding (§4).
- Which way the phone is held changes the camera's effective field of
  view, so the same person at the same distance can have very different
  face-to-frame ratios (§3-4, reconfirmed in §13).
- Adaptive logic that needs a decision (auto-zoom, bbox-based refinement)
  struggles exactly where its own decision signal is unstable — a
  chicken-and-egg problem (§11-12). Decision-free geometric fixes
  (centred crop) sidestep it entirely.
- Software cropping and camera-level zoom are fundamentally different —
  information already discarded at capture can't be recovered downstream
  no matter how cleverly it's cropped (§6).
- "It's an already-verified pattern, just reuse it" is a dangerous
  assumption — reusing refinement's fallback pattern almost happened
  before discovering it had never actually succeeded on a real device
  (§12, found because the user asked the right question).

**What was achieved (deliverables)**
- Code: `lib/src/detection/yunet_detector.dart` (3-stage detection:
  centred crop → whole-frame fallback → bbox-based refinement),
  `yunet_decoder.dart` (pure geometry functions), `example/lib/main.dart`
  (orientation handling, auto-zoom, guide box, diagnostic logs),
  `blink_liveness_detector.dart`/`face_landmarker.dart` (diagnostic logs).
- Tests: unit tests added for every new pure function.
- Docs: this postmortem (14 sections) plus the 2026-08-13 landmark-jitter
  postmortem, both KR/EN.
- Final real-device results: 74.6% (53/71) accepted in portrait → 90%
  (18/20) in landscape, refinement's first real success, mean similarity
  moving from the 0.28s to the 0.49s.

## What's left

**Closed by §13's verification:**
- ~~Verify the zoom-direction fix~~ — `AutoZoom` confirmed working
  correctly in log5/6.
- ~~Verify the centred-crop change~~ — hit/fallback behaved correctly in
  both portrait and landscape, fallback acted as the intended safety net.
- ~~Refinement's 100% fallback rate~~ — 2/2 and 1/1 succeeded once the
  centred crop landed. (The precise mechanism — "imprecise pass-1 bbox →
  crop genuinely misses the face" — was never directly proven via
  coordinate comparison, only indirectly confirmed by the outcome
  improving.)
- ~~Verify portraitUp / landscapeRight~~ — both confirmed, landscape
  outperformed.

**Still open:**
- **2 orientations still unverified**: `portraitDown`, `landscapeLeft` —
  no real-device test yet; the overlay box's visual alignment with the
  real face hasn't been confirmed in either.
- No log has yet shown `_framingHint` actually appearing (zoom maxed out,
  face still too small) — needs a test that deliberately stays far enough
  away to hit the zoom ceiling.
- **Re-enable liveness**: `_kLivenessEnabled = false` is a temporary
  debugging flag — now that recognition stability is reasonably well
  confirmed, flip it back to `true` and run an integrated test, and
  separately re-assess whether the 500ms `face lost` grace period still
  resets too eagerly.
- Whether it's worth tweaking the centred crop's vertical centring for
  tall (landscape-held) frames — low priority given fallback already
  compensates well.

## Appendix: what changed

- `example/lib/main.dart` — `_cameraQuarterTurns()` now uses
  `_baseQuarterTurns + quarterTurnsForOrientation(liveOrientation)`
  instead of a fixed calculation. `FaceOverlayPainter`'s call site set to
  `quarterTurns: 0` (prevents double-rotation). Added the auto-zoom
  feedback loop (`_maybeAdjustZoom`) and a persistent framing hint
  (`_framingHint`). The initial status message no longer instructs
  landscape holding specifically (orientation now adapts automatically).
  Fixed the oscillation found in the first real-device pass (log2.txt):
  `_minZoom` clamped to 1.0, cooldown 900ms, step 0.15, 3-consecutive-frame
  condition. Fixed the direction-sign inversion found in the second pass
  (log3.txt). Added `_kLivenessEnabled` (currently `false`) to isolate
  recognition-stability testing from liveness. Added `[Timing]` diagnostic
  logs for frame processing time. Added a persistent framing guide box
  (`_guideRegion`) with a label, always drawn on the camera preview.
- `lib/src/detection/yunet_decoder.dart` — added the pure
  `centerSquareCropRegion` function (with 3 tests).
- `lib/src/detection/yunet_detector.dart` — `detect()`'s first pass now
  runs through `_detectFirstPass` (centred square crop, falling back to
  the whole frame if nothing's found there). Added the refinement crop
  region and pass-1 bbox to the fallback debug log.
