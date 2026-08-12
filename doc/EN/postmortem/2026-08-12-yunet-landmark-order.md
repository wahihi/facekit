🇰🇷 [한국어 원문](../../KR/postmortem/2026-08-12-yunet-landmark-order.md)

---

# The 30 minutes EER read 48% — hitting the same left/right bug a second time

**Date:** 2026-08-12
**Trigger:** "BlazeFace only gives one mouth point out of its 6 landmarks,
so `AffineAligner` compromises ArcFace's true 5-point reference (which
wants 2 mouth corners) down to 4 points. Wouldn't switching to a detector
that natively gives 5 points remove that compromise entirely?"
**Conclusion (spoiler):** the direction was right and the improvement was
substantial (AuraFace clean EER 10.0% → 2.5%). But getting there involved
(1) ruling out a strong candidate (SCRFD) on licensing, and (2) hitting
**the exact same class of bug as the BlazeFace eye-order bug from the
2026-08-03 postmortem** while validating the remaining candidate (YuNet)
in Python — an EER of 48%, essentially random.

This project was built in collaboration with [Claude
Code](https://github.com/anthropics/claude-code), and so was this
debugging session.

## 0. Why this investigation wasn't a shot in the dark

Earlier in the same session, `compare_with_alignment.py` (using
insightface's SCRFD) had already measured "what happens to EER when real
alignment is applied" — ArcFace 8.5%→3.0%, AuraFace 10.0%→6.5%. So the
direction ("swapping the detector helps") was already backed by data; this
investigation was about narrowing down to a detector that's actually
adoptable.

## 1. Candidate research — SCRFD ruled out on licensing

Both SCRFD (InsightFace) and YuNet (OpenCV Zoo/libfacedetection) natively
support 5-point output. SCRFD's `detection/scrfd/LICENSE` is Apache 2.0,
but that alone wasn't enough to conclude the weights were free to use —
InsightFace's top-level README explicitly separates code and model
licensing:

> "The code of InsightFace is released under the MIT License... **The
> training data containing the annotation (and the models trained with
> these data) are available for non-commercial research purposes only.**"

`buffalo_l` (already treated as research-only in this project) is named as
a concrete example, so SCRFD should be assumed to fall under the same
blanket clause. Independently, SCRFD's training data, WIDER FACE, was
confirmed to be CC BY-NC-ND (non-commercial, no derivatives) licensed —
two independent lines of evidence both converging on "non-commercial," so
SCRFD was ruled out for good.

YuNet's distributor (the OpenCV Zoo README) explicitly labels the model
itself MIT, and `CLAUDE.md:15` already said "Detection (BlazeFace
Apache2.0 / YuNet MIT): may be bundled" — this project appears to have
anticipated YuNet as an alternative from the start.

## 2. First validation attempt — EER 48-49%, essentially random

A new script, `tool/model_verification/compare_with_yunet.py`, ran YuNet
detection via `cv2.FaceDetectorYN` (built into OpenCV 5.0.0), fed the 5
resulting points into `insightface.utils.face_align.norm_crop()` (the same
function used for the SCRFD validation, same `arcface_dst` reference), and
reused the existing embedder classes (ArcFace, AuraFace) over the same 200
LFW pairs.

Result: `ArcFace genuine mean=0.7788 impostor mean=0.7485` — genuine and
impostor were barely distinguishable. EER 48% (clean), 49% (AuraFace
clean) — worse than the worst BlazeFace condition seen anywhere in this
session. **A sign something was fundamentally wrong.**

## 3. Debugging — clean detections, but a black crop

Instead of reasoning it out theoretically, the earlier postmortems'
lesson applied again: **look at the actual data.**

1. **Checked raw detections**: pulled 5 samples and printed bbox, the 5
   points, and score directly. All normal — score 0.92-0.95, eye spacing
   37-45px, sane eye-nose-mouth geometry.
2. **Visualized the crop**: actually rendered an aligned 112×112 crop as a
   PNG — **a tiny face surrounded by solid black.** Something the numbers
   alone hadn't revealed was obvious the instant it was an image (the
   2026-07-24 and 2026-08-03 postmortems both landed on the same lesson —
   this is the third confirmation).
3. **Computed the transform directly**: the fitted scale was 0.125 — about
   1/7 of the ~0.83-0.91 expected from eye spacing alone. **Fed the same 5
   points independently into both insightface's `norm_crop` and facekit's
   own Umeyama solver (already ported to Python earlier in this
   conversation)** — both gave the identical 0.125, ruling out a library
   bug and pointing at the input coordinates themselves (the left/right
   correspondence).

## 4. Root cause confirmed — the opposite mapping fixed it instantly

YuNet's output order is `right_eye, left_eye, nose, right_mouth,
left_mouth`. The first attempt **relabeled** these into ArcFace's
convention (`left_eye, right_eye, nose, left_mouth, right_mouth`) — the
"obviously correct" move, since the names differ. That was wrong:

```python
# Relabeled (mapped left_eye to dst[0]) — wrong
scale, rot = 0.1256, 4.81°

# YuNet's raw order, unchanged — correct
scale, rot = 0.9061, 1.46°
```

**YuNet's own `right_eye`/`left_eye` labels turned out to be the exact
opposite handedness from what facekit assumed** — precisely the same
mistake as the **BlazeFace eye-order bug** already hit once in the
2026-08-03 postmortem, made again this time via the opposite kind of
hasty judgment ("the names are different, so of course I need to relabel
them").

## 5. Re-measured after the fix — better than SCRFD

| Method | ArcFace EER (clean/degraded) | AuraFace EER (clean/degraded) |
|---|---|---|
| Resize only (no alignment) | 8.5% / 25.0% | 10.0% / 28.0% |
| SCRFD-aligned (ruled out on license) | 3.0% / 5.0% | 6.5% / 8.0% |
| **YuNet-aligned (after the fix)** | **2.5% / 4.0%** | **2.5% / 6.0%** |

Better than SCRFD, and **ArcFace and AuraFace's clean EER landed at
exactly the same 2.5%** — strong evidence that the "AuraFace is the
weakest of the three" pattern that ran through this whole session wasn't
a limit of the model itself, but AuraFace being unusually sensitive to
alignment quality specifically (BlazeFace's 4-point compromise).

## 6. Summary — the causal chain

```
Question: what if BlazeFace's 4-point compromise were replaced with a native 5-point detector?
   -> Direction already confirmed via existing SCRFD data (EER improved)
   -> SCRFD itself ruled out by license research (InsightFace policy + WIDER FACE dataset, both non-commercial)
   -> First YuNet Python validation: EER 48% (complete collapse)
        -> Raw detections checked: normal
        -> Crop visualized: tiny face on solid black -> first visual evidence something was wrong
        -> Transform computed directly (cross-checked via 2 independent implementations): confirmed 1/7 scale collapse
        -> Tested the opposite left/right mapping: instantly normalized (scale 0.906, rotation 1.46 deg)
   -> Re-measured: ArcFace 2.5%, AuraFace 2.5% - better than SCRFD, and the two models tied
```

## 7. Lessons learned

- **"Don't trust a model author's own landmark names without verifying
  them" was confirmed for the second time.** With BlazeFace, the mistake
  was assuming the names must be different without checking. This time it
  was the opposite: hastily "correcting" by relabeling because the names
  *looked* different. Both share the same root cause — **trusting a label
  instead of verifying actual coordinates.**
- **When the numbers look wrong, looking at the actual image is still the
  fastest shortcut.** A scale of 0.125 alone didn't immediately reveal
  what was wrong; opening the crop PNG made "tiny and solid black"
  obvious instantly. All three postmortems in this project have now
  landed on this same conclusion.
- **Cross-checking with two independent implementations paid off again.**
  Running both insightface's `norm_crop` and facekit's own Umeyama solver
  in Python and getting the identical wrong answer (0.125) ruled out "it's
  a third-party library bug" immediately, letting the investigation narrow
  straight to "the input data is wrong" instead of wasting time chasing
  the wrong lead.
- **A mistake already made once with BlazeFace (unverified landmark
  name/order) can repeat itself with a different detector just as
  easily.** When this gets ported to real Dart code, the left/right
  mapping confirmed here needs to be pinned down not just in a code
  comment but in a regression test too.

## Appendix: tools left behind

- `tool/model_verification/compare_with_yunet.py` — re-measures LFW EER
  with YuNet (`cv2.FaceDetectorYN`)-based alignment. Reuses the same
  embedder classes as `compare_with_alignment.py` (the SCRFD version), so
  results are directly comparable with only the detector changed.
- `tool/model_verification/yunet.onnx` — the original YuNet model fetched
  from OpenCV Zoo (`face_detection_yunet_2023mar.onnx`).
