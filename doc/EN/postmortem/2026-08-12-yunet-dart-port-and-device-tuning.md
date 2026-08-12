🇰🇷 [한국어 원문](../../KR/postmortem/2026-08-12-yunet-dart-port-and-device-tuning.md)

---

# Three threshold changes to find the real cause — porting YuNet to Dart and tuning it on-device

**Date:** 2026-08-12
**Trigger:** Earlier the same day,
[2026-08-12-yunet-landmark-order.md](2026-08-12-yunet-landmark-order.md)
finished the Python-side validation (AuraFace clean EER 2.5% with YuNet
alignment, better than SCRFD). The next step was clear — port it into the
actual facekit Dart SDK, fully replace BlazeFace, and confirm it on a real
device (Pixel 7).
**Conclusion (spoiler):** The port itself went smoothly — the decode
formula was verified byte-for-byte against `cv2.FaceDetectorYN`'s actual
output before porting. But once it hit the real device, the detection
threshold had to be changed **three times** (0.6→0.3→0.45), and that
process surfaced something more important than the port itself: **AuraFace's
matching threshold (0.30) was still measured under the old "BlazeFace-era,
alignment-free resize" methodology**. A fresh, YuNet-alignment-based
remeasurement already existed in this repo (`results_yunet.json`,
threshold=0.211) — it just hadn't been applied yet, buried under the
detector-swap work.

This project was built in collaboration with [Claude
Code](https://github.com/anthropics/claude-code), and so was this session.

## 1. Dart port — matching the decode formula to cv2's output byte-for-byte

There was no ready-made reference anywhere for decoding YuNet's raw ONNX
output (12 tensors: `cls_8/16/32`, `obj_8/16/32`, `bbox_8/16/32`,
`kps_8/16/32`) — `cv2.FaceDetectorYN` handles all of it internally in C++,
so the actual grid-offset/exp-based bbox/keypoint decode formula is hidden
behind the library. Before porting, it was verified in Python first:

1. Fed the same 160×160 resized image into both `cv2.FaceDetectorYN` and
   the raw ONNX model, treating cv2's output as ground truth.
2. Found the actual postprocessing code in
   `opencv/modules/objdetect/src/face_detect.cpp` to confirm the formula:
   `cx=(col+bbox[0])*stride`, `w=exp(bbox[2])*stride`,
   `score=sqrt(clamp(cls,0,1)*clamp(obj,0,1))`.
3. Decoding the raw tensors manually with this formula matched cv2's
   output **exactly, down to the decimal, on score, bbox, and all 5
   keypoints** (see `tool/model_verification/`'s verification script).

Those verified raw values (real tensor values from a specific grid cell)
were embedded directly into
`test/detection/yunet_decoder_test.dart`'s regression test — so the next
time someone touches this decoder, it's checked against "does it match the
real measured values exactly," not "does the number look plausible." The
order of the 12 output tensors was also pinned down beforehand: the
onnx2tf conversion was empirically confirmed to scramble the graph's
output order, so `YuNetDetector` resolves each tensor's role at runtime
from its shape (grid size, channel count) instead of a hardcoded index
(`lib/src/detection/yunet_detector.dart`). It also turned out the two
score tensors (cls, obj) don't need to be told apart — `sqrt(cls*obj)` is
symmetric, so the result is the same either way.

## 2. AffineAligner — added a native 5-point mode without touching the existing 4-point path

BlazeFace only ever gave a single mouth point out of its 6 landmarks, so
`AffineAligner` compressed ArcFace's true 5-point reference (2 mouth
corners) down to 4 points. YuNet gives 5 points natively, so that
compromise is no longer needed — but simply switching
`AffineAligner.align()` to a 5-point-only path would have silently changed
the behavior of existing test fixtures with `landmarks.length==5` (ones
deliberately padded with a dummy 5th point to exercise the 4-point path,
e.g. `_identityFace()`). Instead of auto-dispatching on landmark count, an
explicit `nativeFivePoint` flag was added so the caller declares which
path to use (`AffineAligner.arcface112(nativeFivePoint: true)`). It
defaults to `false`, so the existing BlazeFace path and its tests are
untouched.

When `nativeFivePoint: true`, YuNet's raw output order is used **as-is**
against `arcface112Ref` — as the earlier postmortem already established,
"the names differ, so it must need relabeling" was the wrong instinct.
This was pinned down not just in a code comment but in a regression test
too (`test/alignment/affine_aligner_test.dart`'s `"correcting" by swapping
the two eye points does NOT round-trip to identity` test) — directly
acting on the earlier postmortem's own "this needs to be a regression test
next time" lesson.

## 3. First device run — an empty log (not a code bug)

The first `flutter run --dart-define=FACEKIT_VERBOSE_DEBUG=true | tee
log.test` on the Pixel 7 hit "Lost connection to device" right after the
Dart VM Service connection was established. The log file was only 44
lines (2.4KB) — not a single `debugPrint` line had made it through. Since
the app was visibly still running on screen (liveness text, box overlay),
this was judged to be just the USB/adb debug bridge dropping, not an app
crash, and the run was retried. Habit worth keeping: **check `wc -l`
before diving into a log's content** — it immediately tells apart "the
code is broken" from "nothing was ever logged in the first place."

## 4. Second device run (threshold 0.6) — flicker keeps liveness from ever finishing

YuNet's manifest `score_threshold` had been set to a provisional 0.6
without on-device validation (already known to need retuning per
CLAUDE.md's rule against carrying over untuned numbers). The retried log
showed:

```
[onFrame] face found, score=0.686
[onFrame] face lost (>500ms, resetting liveness)
[onFrame] face found, score=0.624
[onFrame] face lost (>500ms, resetting liveness)
...
```

Every `found` score clustered at 0.608–0.711 — right at the edge just
above the 0.6 threshold. `found` was always followed a few frames later by
`lost`, and liveness never left `pending`. Enrollment was never even
attempted.

## 5. Third device run (threshold 0.3) — flicker fixed, but two new problems appeared

Lowering the threshold to 0.3 eliminated the flicker entirely — enroll
succeeded, and identify kept running automatically frame after frame. But
a closer read of the log revealed two new issues:

- **Edge-of-frame false positives**: bboxes appeared clipped exactly at
  the image boundary (`bbox=(564.6,0.0)-(720.0,123.1)`, landmark
  coordinates also clipped at 720) — apparently background/corner content
  being misdetected as a "face." At one point `detect: 2 face(s)` showed
  up.
- **Wild similarity swings for the same person**: `match: id=나
  similarity=...` bounced between 0.147 and 0.826 frame to frame.
  `poseGate=ok` was logged every single time (rotation -9°~+2°, all
  sane), so this wasn't a landmark-instability issue — yet several frames
  were **genuinely the enrolled user, wrongly rejected** against
  AuraFace's `matching.threshold=0.30` (`similarity=0.147, 0.149, 0.186,
  0.205, 0.233, 0.257`, among others).

## 6. The real cause found here — a stale matching threshold

While investigating the similarity spread, it turned out
`example/assets/models/auraface/manifest.json`'s
`matching.threshold=0.30` had been **measured on 2026-08-11 under the
"BlazeFace 4-point compression, alignment-free resize" methodology** (EER
10.0%). But YuNet now provides real 5-point alignment, and this repo
already had a fresh remeasurement against exactly that —
`tool/model_verification/results_yunet.json` (produced by
`compare_with_yunet.py` during the earlier
2026-08-12-yunet-landmark-order.md work):

```json
"auraface": {
  "threshold_clean_eer": 0.21119210124015808,
  "eer_clean": 0.025
}
```

In other words: **swapping the detector meant every downstream threshold
that depended on its output — including the matching threshold — needed
re-review too, and that got missed.** The remeasurement already existed;
it just hadn't been applied, and that gap made it all the way to the
device.

## 7. Fourth device run (yunet 0.45 + auraface 0.211) — both fixed

Two values changed together: `score_threshold` 0.3→0.45 (suppress
edge-of-frame false positives) and `matching.threshold` 0.30→0.211 (adopt
the YuNet-alignment-based remeasurement). Retest results:

- Clipped-edge bboxes and simultaneous `2 face(s)` detections —
  **gone entirely.**
- All 26 identify calls were `id=나, accepted=true` — **not a single
  false rejection.** Similarity itself still spread widely (0.271–0.871),
  but every value now safely cleared the new threshold.

## 8. Impostor test — a photo of a stranger on a monitor

Having lowered the threshold (0.30→0.211), the opposite direction
(mistaking a stranger for the enrolled user) needed checking too. A photo
of Elon Musk was displayed on a monitor and pointed at with the camera:

- **The real face** (a large bbox filling most of the frame): correctly
  matched `id=나` 3 times (similarity 0.359/0.380/0.506), plus one false
  rejection (0.196 — the same already-known similarity-variance issue).
- **The monitor photo** (a small bbox in a corner, roughly 93×62px, fitted
  matrix scale a≈1.65, clearly different geometry from the others):
  correctly rejected as `id=null` twice (similarity 0.123/0.147). One
  frame even caught both simultaneously as `detect: 2 face(s)`, and the
  monitor photo — which had the higher detection score that frame — was
  still correctly rejected at the matching stage.

No false acceptance of the stranger ever occurred.

## 9. Summary — the causal chain

```
Question: YuNet is Python-validated (EER 2.5%) — port it to Dart and confirm on-device?
   -> Decode formula verified byte-for-byte against cv2.FaceDetectorYN, ported, pinned via regression test
   -> Added nativeFivePoint mode to AffineAligner (existing 4-point path left untouched)
   -> Device run 1: USB connection dropped, log 44 lines (empty) -> not a code issue, retried
   -> Device run 2 (threshold 0.6): scores clustered right above threshold -> flicker, liveness stuck pending
   -> Device run 3 (threshold 0.3): flicker fixed, but edge false positives + genuine matches wrongly rejected (similarity 0.147-0.30)
        -> Investigated: reconfirmed AuraFace's matching.threshold (0.30) was measured under the old alignment-free-resize methodology
        -> Found an existing YuNet-alignment remeasurement (0.211, EER 2.5%) already in the repo, not yet applied
   -> Device run 4 (yunet 0.45 + auraface 0.211): false positives gone, identify 26/26 correctly accepted
   -> Impostor test (stranger's photo on a monitor): 2/2 correctly rejected, no false acceptance
```

## 10. Lessons learned

- **Swapping a detector means every downstream threshold that depends on
  its output — including the matching threshold — is back on the table
  for review.** This time the remeasurement already existed in the repo,
  but staying focused on "swap the detector" as the scope of work meant it
  only surfaced during on-device debugging. Changing one pipeline stage
  means re-checking every other stage's assumptions about that stage's
  output, not just the stage you touched directly.
- **"The flicker is gone" is not, by itself, grounds to finalize a
  threshold change.** The same log needs to be checked for edge-of-frame
  false positives too, or a newly introduced failure mode (background
  false detections, this time) goes unnoticed.
- **An empty log from a dropped connection and a log full of real errors
  both look like "it didn't work," but the causes are completely
  different.** Checking the line count (`wc -l`) before digging into
  content saves time chasing the wrong kind of problem.
- **Static-photo benchmarks (LFW) are necessary but not sufficient.** Both
  the edge-of-frame false positives and the frame-to-frame similarity
  swings only showed up in real device video — the same conclusion this
  project's earlier postmortems have already reached, holding again here.
- **`poseGate=ok` never once tripped across the entire device session, and
  that's a meaningful signal on its own.** Under BlazeFace, 24–42% of
  real-device frames used to get flagged for bad rotation
  (2026-08-03 postmortem); with YuNet, it never happened once in this
  entire session — clear evidence the landmarks are geometrically much
  more stable now. That means the remaining similarity swing (0.27–0.87)
  is likely not a rotation/scale problem but something else (lighting,
  real-time frame noise, etc.) — a newly confirmed but still-open
  question, left unresolved here.

## Appendix: what changed

- `lib/src/detection/yunet_decoder.dart`, `yunet_detector.dart` (new) —
  pure decode function + `YuNetDetector`. The decode formula's real,
  cv2-verified values are embedded directly in the regression test
  (`test/detection/yunet_decoder_test.dart`).
- `lib/src/alignment/affine_aligner.dart` — added `nativeFivePoint` mode
  (defaults to `false`; the existing 4-point path and its tests are
  untouched).
- `assets/models/yunet_160/` — the YuNet MIT model (fixed 160×160 input,
  bundleable) + manifest.json. `score_threshold` finalized at 0.45 after
  on-device tuning.
- `example/assets/models/auraface/manifest.json` — `matching.threshold`
  0.30→0.211 (the YuNet-alignment-based remeasurement).
- `example/lib/main.dart` — default detector switched from
  `BlazeFaceDetector` to `YuNetDetector`. `BlazeFaceDetector`'s own code
  is left in place as a fallback.
- `test/detection/yunet_decoder_test.dart`,
  `test/detection/yunet_smoke_test.dart`,
  `test/alignment/affine_aligner_test.dart` (native 5-point regression
  tests added).
