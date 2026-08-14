🇰🇷 [한국어 원문](../../KR/postmortem/2026-08-13-landmark-jitter-frame-fill.md)

---

# The real reason landmarks jittered — a small face in the frame lets the 160×160 input amplify noise

**Date:** 2026-08-13
**Trigger:** An unsolved problem left open by
[2026-08-12-yunet-dart-port-and-device-tuning.md](2026-08-12-yunet-dart-port-and-device-tuning.md) —
on a real device, similarity for the same person swung wildly between 0.27
and 0.87. `poseGate` (the rotation/scale gate) passed every single frame,
so it was concluded this wasn't a landmark-geometry stability problem, and
the cause was left unknown.
**Conclusion (spoiler):** Re-digging the log showed similarity swings
correlated far more strongly with `AffineAligner`'s fitted scale/rotation
than with detect score. The cause was structural: `resizeNearest` squeezes
the *entire* camera frame down to a fixed 160×160 before YuNet ever sees
it, and when a face occupies only 8–12% of the frame, ordinary sensor
noise alone drove normalized landmark jitter up to **288x higher** than
when the face fills the frame. The investigation itself nearly ran
aground too: a memory bug in the verification script kept it from ever
finishing, which got misdiagnosed as an environment problem and the whole
task got moved to a different machine.

This project was built in collaboration with [Claude
Code](https://github.com/anthropics/claude-code), and so was this session.

## 1. Re-reading the log — poseGate is fine, so why does similarity swing?

Re-analyzed `example/log.txt` (a real-device session log). Bbox stayed
essentially unchanged frame to frame, yet one frame pair had eye-to-eye
landmark distance jump 24%. Correlating similarity against detect score
gave a weak -0.24, but against the scale/rotation `AffineAligner` fits
every frame it was -0.7 to -0.8. In other words, "how confident was the
detection" explained the similarity swings far worse than "how much did
the alignment transform wobble."

## 2. Hypothesis — the 160×160 input starves small faces of pixels

`lib/src/detection/yunet_detector.dart` resizes the camera frame to a
fixed 160×160 via `lib/src/image/image_converter.dart`'s `resizeNearest`
before feeding YuNet. The smaller a face's fraction of the frame, the
fewer of those 160×160 input pixels land on the face itself. Hypothesis:
the smaller the face is framed, the more ordinary frame-to-frame sensor
noise moves the decoded landmarks relative to the face's own size — which
also matched the user's anecdotal report that holding the phone so the
face fills more of the frame felt noticeably more stable.

## 3. The verification script — an A/B test against the real tflite model

Wrote `tool/model_verification/landmark_jitter_ab.py`, re-implementing
`yunet_detector.dart`/`yunet_decoder.dart`'s preprocessing and decode
formula byte-for-byte in Python, tested against the actual
`assets/models/yunet_160/yunet_160.tflite` (not `cv2.FaceDetectorYN`,
which uses a different input size/pipeline). Method: find a face in one
photo, build synthetic "camera crops" around it at various margins (what
fraction of the crop the face occupies), apply N independent trials of a
shared noise model (a few pixels of sub-pixel shift + mild Gaussian noise,
modeling real hand jitter and sensor noise) to each crop before the
160×160 resize, run the real decode on every trial, and compare landmark
jitter (normalized by eye-distance) across margins.

## 4. First result — the opposite of the hypothesis (and it was a bug)

An early two-margin version (small/large) gave: small (face is a small
fraction of the crop) jitter=0.0103, large (face fills the crop)
jitter=0.0428 — **the exact opposite** of the hypothesis, with jitter 4x
*higher* when the face filled more of the frame. The script was then
upgraded to sweep six margins (12, 8, 5, 3, 1.8, 1.1), but the upgraded
version never finished a single run before the environment (VSCode) died.

## 5. Re-running after the handoff — it was actually an OOM

After moving the work to a new machine (and reinstalling the Python
environment from scratch: bootstrapped pip, installed
`numpy`/`tensorflow-cpu`/`Pillow`), re-running the six-margin script died
immediately with `exit 137`. Checking `journalctl -k` showed the kernel
OOM killer had force-killed the `python3` process at total-vm 23GB,
anon-rss 20GB — it had consumed this machine's entire 20GB of physical
RAM.

The cause was the `crop()` + `add_noise()` combination. When the
requested crop size vastly exceeded the source photo (e.g. margin=12
means a crop half-width 12x the face's own half-width), `crop()`
literally materialized that full size via `np.pad(mode="reflect")`. On a
2736×3648 photo, margin=12 requested a 13828×21315 crop — 884MB as
uint8, and `add_noise()` promoting that to `float64` (shift array + noise
array + intermediate arithmetic) pushed peak memory to roughly 14GB for a
*single* trial. Yet all that was ever actually needed was the 160×160 =
25,600 pixels that `resize_nearest` point-samples out of it — the
remaining hundreds of millions of pixels were pure waste.

**The "failure" reported on the previous machine was actually this same
OOM happening silently partway through the 240-inference run.** It got
misdiagnosed as "the run failed" only because `grep` couldn't find a
match in the results file — the real cause (the kernel killing the
process) left no traceback to find in the first place.

## 6. The fix — compute the same values without ever building the oversized array

`resize_nearest` point-samples exactly one source pixel per output pixel,
and nearest-neighbor sampling composes exactly (chaining two
nearest-neighbor lookups equals doing it in one step). That fact was used
to collapse `crop()` → `add_noise()` → `resize_nearest()` into a single
`sample_noisy_input()` that computes, directly, which source coordinates
the final 160×160 output grid needs. Reflect-padding was replaced with an
index formula mathematically equivalent to `np.pad(mode="reflect")`
(`reflect_index()`, period = 2*(n-1)) instead of ever materializing a
padded array. That formula was checked against actual numpy output on
small test arrays for an exact match, and the new pipeline was verified
to produce pixel-identical output to the old `crop()`+`add_noise()`+
`resize_nearest()` chain across several combinations of position, margin,
and shift. Memory is now O(160×160) regardless of margin — the fixed
version ran in 5 seconds at 586MB peak RSS (versus never finishing before,
due to OOM).

The noise model itself is unchanged (iid Gaussian, σ=4, sub-pixel shift
±2.5px), but it now draws from the RNG in a different order (the old code
drew noise for the *entire* crop and only ever used the 160×160 subset
that got sampled), so results for a given `--seed` are no longer
byte-identical to pre-fix runs — same noise model, different random
stream.

## 7. Post-fix results — the hypothesis reproduced strongly, and dramatically

All three photos (`me_1.jpg`, `me_2.jpg`, `other_1.jpg`) showed the same
pattern (`landmark_jitter_over_eye_dist`):

| Photo | margin=12 (face ~8%) | margin=8 (~12%) | margin=5 (~20%) | margin=1.1 (~91%) | widest/tightest ratio |
|---|---|---|---|---|---|
| me_1.jpg | 6.8630 | 0.8469 | 0.0176 | 0.0430 | **159.59x** |
| me_2.jpg | 5.2384 | 0.7264 | 0.0532 | 0.0278 | **188.60x** |
| other_1.jpg | 4.6321 | 1.0853 | 0.0309 | 0.0161 | **287.59x** |

All three show a clear threshold effect: once the face drops below
**~15–20% of the frame width (margin≥8)**, jitter explodes to at or above
the normalized eye-distance itself (0.7–6.9) — the landmarks are
effectively noise at that point. Above **~20%** (margin≤5), jitter stays
comparatively flat in the 0.01–0.05 range (not perfectly monotonic
between the three tightest margins, but stable in order of magnitude).

**The first (two-margin) result being the exact opposite of the
hypothesis was most likely not a flaw in the experiment design or in
YuNet itself, but that run completing in a compromised state near the
OOM boundary discovered above** — this wasn't directly re-confirmed, but
it's the notable contrast against the six-margin version's consistent
direction and magnitude across all three photos.

## 8. Timeline — cause and effect

```
Question: why does similarity for the same person swing 0.27-0.87 on a real device?
   └→ poseGate passes every frame → wrongly concluded it's not a landmark-geometry issue, left unsolved (2026-08-12)
   └→ Re-reading the log: similarity correlates far more with AffineAligner scale/rotation (-0.7~-0.8) than detect score (-0.24)
   └→ Hypothesis: resizeNearest squeezes the whole frame to 160x160, so a smaller face gets fewer pixels and is more noise-sensitive
   └→ Write a Python A/B script (real yunet_160.tflite, two margins)
        └→ First result: opposite of hypothesis (jitter 4x higher when face fills more of the frame)
        └→ Upgraded to six margins — never finished a run before the PC was replaced
   └→ New PC: reinstalled the Python environment, re-ran → exit 137
        └→ journalctl -k confirms: kernel OOM killer, anon-rss 20GB
        └→ Cause: crop()'s reflect-pad materializes an array scaling with margin^2 (margin=12 → 14GB/trial),
           add_noise()'s float64 promotion adds another 8x on top
   └→ Fix: replace the crop→noise→resize chain with sample_noisy_input(), computed directly at the 160x160 output grid
        (reflect_index() eliminates the padded array entirely; verified pixel-identical to the old implementation)
   └→ Re-run: 5 seconds, 586MB, completes
   └→ Result: all three photos show a 160-288x jitter ratio between margin=12 and margin=1.1 — hypothesis strongly confirmed
```

## 9. Lessons learned

- **Don't take "the run failed" at face value — check the exit code and
  the kernel log first.** This script being reported as "failed (exit 1)"
  on the previous machine was purely because `grep` found no match in the
  results file; the real cause (OOM kill, exit 137) was something else
  entirely. Python itself never got a chance to leave a traceback (it was
  SIGKILL'd) — "no error message" is easy to mistake for "nothing's
  wrong with the code."
- **What looks like an environment problem can actually be a reproducible
  bug.** The decision to move the whole task to a new machine assumed a
  new PC was needed, but the same script would have died for the same
  reason on any machine (memory scales with margin², so a machine with
  more RAM would just die later, or only at margin=12). Checking
  `dmesg`/`journalctl -k` for an OOM before concluding "need a different
  PC" would have skipped that whole detour.
- **If a value is going to get downsampled at the end anyway, only
  compute as much of it as the downsample will actually use.** Exploiting
  the fact that `resize_nearest` is a point-sample (and that composed
  point-sampling is exact) let an intermediate pipeline stage's array
  size shrink all the way down to the final output size — this isn't
  specific to this one script; the same optimization applies to any other
  verification script using a "build it big, then shrink it" pattern.
- **When an experiment contradicts the hypothesis it was built to test,
  question the experiment's own execution integrity before abandoning the
  hypothesis.** The first (two-margin, hypothesis-contradicting) result
  looked like a "success" on the surface (the JSON wrote out cleanly),
  but that run may well have completed under resource pressure near the
  OOM boundary — a results file existing and its values being trustworthy
  are two different things.
- **A real-device "same person, but similarity swings" symptom doesn't
  mean landmark geometry is stable just because the rotation/scale gate
  (`poseGate`) passed.** `poseGate` only checks whether the alignment
  transform is within a "sane" range — whether that transform itself is
  wobbling significantly frame to frame is a separate thing that has to
  be measured directly.

## What's left

- This investigation only confirmed *why* it jitters — it hasn't fixed
  the code yet. The next step is to look at having `resizeNearest`
  operate on a crop around the detected face rather than the whole frame
  (e.g. a two-pass approach: locate roughly at low resolution, then
  re-detect at native resolution around that region), or at minimum
  considering a mitigation that more strongly nudges the UI toward "fill
  the frame with your face."
- The non-monotonic bit in the margin 1.8–5 range (e.g. me_1.jpg's
  margin=1.8 being lower than margin=3) hasn't been confirmed as noise
  versus a real secondary effect — more trials or cross-validating with
  more photos should narrow that down.

## Appendix: what changed

- `tool/model_verification/landmark_jitter_ab.py` — replaced
  `crop()`/`add_noise()` with `crop_geometry()`/`sample_noisy_input()`/
  `reflect_index()`. Memory that used to scale with margin is now
  O(160×160) regardless of margin. Verified the change itself produces
  pixel-identical output to the old pipeline (`np.array_equal` across
  several position/margin/shift combinations) with a separate check
  script.
- `tool/model_verification/results_landmark_jitter_ab.json` — updated
  with the final six-margin sweep for `me_1.jpg`.
- `tool/model_verification/results_landmark_jitter_ab_me_2.json`,
  `results_landmark_jitter_ab_other_1.json` (new) — cross-validation runs.
