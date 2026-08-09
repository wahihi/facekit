🇰🇷 [한국어 원문](../../KR/postmortem/2026-08-09-auraface-normalization.md)

---

# The day a stranger got recognized as me — the cause wasn't alignment, it was normalization

**Date:** 2026-08-09
**Symptom:** After enrolling a stranger's face and starting live identify,
pointing the camera at that stranger scored a ~0.9 similarity — and then
pointing it at a completely different person (the account owner) scored
the same ~0.9. AuraFace was effectively unable to tell two different
people apart.
**Conclusion (spoiler):** completely unrelated to the alignment bugs fixed
in the 2026-08-03 postmortem, **a single setting in the AuraFace
manifest (`input.normalize`) was wrong.** It was feeding raw, unnormalized
0-255 pixels in, which didn't match the value the official `insightface`
package auto-selects for this exact model graph
(`(pixel-127.5)/127.5`). The interesting part: this wrong value was
itself **a misdiagnosis tangled up with the alignment bugs from the
2026-07-24 postmortem** — after alignment was fixed on 2026-08-03, nobody
re-validated the normalization setting, so it silently stayed wrong for
over a month.

This project was built in collaboration with [Claude
Code](https://github.com/anthropics/claude-code), and so was this
debugging session — the wrong hypotheses and several failed real-device
log-capture attempts are left in as-is.

## 0. Reproducing the symptom

After enrolling a stranger's face and turning on "start live identify",
pointing the camera at that same stranger scored roughly 0.9 similarity.
Pointing it at a completely different person (the enrolling account's own
owner) scored roughly the same 0.9 — a sign that something in the
enroll/identify pipeline wasn't distinguishing the two people at all.

## 1. First suspect: the live-identify logic itself?

The `_onFrame` flow in `main.dart` (detect → liveness → `pipeline.identify`)
was re-reviewed, but every frame computed a fresh embedding from the
actual live image — no bug reusing a cached image was found. The
liveness (blink-check) gate itself got in the way of testing, though —
a photo held up to a laptop screen can't blink — so a
`_kSkipLivenessForDebug` flag was added temporarily to bypass the gate and
let identify fire immediately even against a static photo (reverted
afterward, see section 4).

## 2. Second suspect: an alignment regression

A real-device log captured via
`flutter run --dart-define=FACEKIT_VERBOSE_DEBUG=true` and analyzed with
`tool/analyze_alignment_log.py` showed 13 of 46 detections (28%) flagged
with rotation angles between 55° and 170° — exactly the failure signature
the 2026-08-03 postmortem had verified as fixed (BlazeFace's landmark
quality wobbling under extreme pose, a limitation that document's own
"left unresolved" section had already flagged). Reconstructing the aligned
crops from the base64 dumps into PNGs confirmed genuinely bad crops:
faces tilted diagonally with a lot of background (window frames/blinds)
mixed in.

But a decisive counterexample turned up: **two different people scored an
identical 0.83-0.88 match even on clean crops with normal rotation
(20.8°, 31.9°, both rated "ok").** The false accept was happening
independent of alignment quality — alignment wasn't the culprit (or at
least not the main one).

## 3. Control experiment — swapping in ArcFace

Detection and alignment are code that's completely independent of the
embedder, so the same pipeline was re-run with just the embedder swapped
from AuraFace to ArcFace (`buffalo_l`), under the same conditions (enroll
self, then compare self/other). ArcFace **correctly rejected the other
person (similarity 0.391, below the 0.40 threshold) even under a worse
rotation condition (128.1°, FLAGGED)**, and clearly separated the
enrolled user (0.67-0.97). Same detect/align code, same real-device
conditions, only the embedder changed — and the outcome flipped. That's
strong evidence the problem was **in the AuraFace embedder itself, not
detection/alignment.**

## 4. Is the AuraFace model itself weak? — cross-checking against the official implementation

AuraFace (glintr100) is architecturally the same family as ArcFace
(ResNet100, Additive Angular Margin Loss), so there was no inherent reason
for it to have weaker discriminative power. Searching fal.ai's official
README and InsightFace's `ArcFaceONNX` source
(`model_zoo/arcface_onnx.py`) online turned up something important: the
preprocessing isn't a fixed spec at all — it's **auto-detected by
inspecting the first 8 node names of the ONNX graph**:

```python
if find_sub and find_mul:      # normalization is already baked into the graph (MXNet exporter pattern)
    input_mean, input_std = 0.0, 1.0
else:                            # no baked-in normalization -> apply the standard values
    input_mean, input_std = 127.5, 127.5
```

A new script, `tool/model_verification/verify_auraface_official.py` (uses
the official `insightface` package's own detection + alignment +
embedding end-to-end — none of facekit's Dart code is involved), loaded
the actual `glintr100.onnx` and logged this:

```
find model: ./models/auraface/glintr100.onnx recognition ['None', 3, 112, 112] 127.5 127.5
```

**127.5, 127.5** — the standard ArcFace normalization. Testing with this
value on 2 photos of one person + 1 of someone else gave a dramatic split:
genuine 0.7737 vs impostor 0.0257/-0.0266. **The AuraFace model itself
wasn't weak at all.**

## 5. The gap between facekit and the official implementation — manifest normalization

Checking `example/assets/models/auraface/manifest.json`:

```json
"normalize": { "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0] }
```

No normalization at all (raw 0-255 pixels passed straight through) — the
exact opposite of the official value (127.5/127.5).

## 6. How the wrong value got there — the causal link to the 2026-07-24 postmortem

Looking back at the 2026-07-24 postmortem's record: that day the team
concluded "raw, unnormalized pixels must be fed in — self-similarity was
0.93 raw vs. 0.15-0.20 with standard normalization" and adopted the raw
value. But **at that point in time, the alignment pipeline still had
unfixed bugs (reversed eye-left/right indices, an SVD sign bug, a missing
covariance `/n` normalization)** — those were only fixed a month later, on
2026-08-03. In other words, the "aligned crops" being compared on 7/24
were already mis-rotated; feeding those bad crops through *correct*
normalization produced low similarity, which got misdiagnosed as a
normalization problem and "fixed" by switching to raw pixels — papering
over the symptom without touching the real cause. After the alignment
bugs were actually fixed on 8/3, nobody went back to re-validate the
normalization setting, so the raw value silently persisted.

## 7. Fix and real-device re-verification

`input.normalize` was corrected to the standard values:

```json
"normalize": { "mean": [127.5, 127.5, 127.5], "std": [127.5, 127.5, 127.5] }
```

The fix was re-tested on a real device: enroll self, then re-test against
a photo of Elon Musk (re-photographed off a monitor screen).
`_debugDumpAlignedFace` in `lib/src/pipeline/face_pipeline.dart` was
temporarily changed to dump the aligned crop on *every* call instead of
just once per app run (reverted afterward), to actually see what was
being fed to the model. Several log-capture attempts failed because
quitting with `q` raced the final identify() call's base64 dump and
truncated it mid-chunk; after enough retries, four clean Elon Musk crops
were finally captured: similarity **0.060-0.094**, all clearly rejected.
The enrolled user's own face still cleared the threshold.

## 8. Deriving a real threshold from 200 LFW pairs

The quick 3-photo test (genuine 0.77 vs impostor ≈0.03) used too small a
sample to rule out a fluke, so a formal re-measurement was run with the
exact same methodology already used for ArcFace/AdaFace (a
`--auraface-tflite` option added to `compare_arcface_adaface.py`, the same
200 LFW pairs, seed=42, resize-only preprocessing with no 5-point
alignment):

| Model | Condition | Genuine mean | Impostor mean | EER | EER threshold | Accuracy at that threshold |
|---|---|---|---|---|---|---|
| AuraFace | clean | 0.4803 | 0.1952 | 10.0% | 0.300 | 90.0% |
| AuraFace | degraded | 0.3560 | 0.2402 | 28.0% | 0.274 | (68.5% if the clean threshold is reused) |

Roughly on par with ArcFace (8.5% EER) and clearly behind AdaFace (2.0%
EER) — the weakest discriminator of the three, but a world away from the
original symptom of "can't tell people apart at all."
`matching.threshold` was updated from the unvalidated placeholder (0.40)
to the measured value (0.30). A real-device re-check (own face only,
threshold 0.30) accepted 8 of 11 frames (73%) — lower than the LFW
accuracy (90%), likely because a hand-held selfie (glasses glare, angle,
indoor lighting) is a harsher real-world condition than a curated LFW
photo. Whether to loosen this threshold further is a separate
usability-vs-security call.

## 9. Summary — the causal chain

```
7/24: AuraFace normalization diagnosed while an alignment bug was present
   -> mis-rotated crop + correct normalization = low similarity, misdiagnosed
   -> "fixed" by switching to raw pixels — symptom gone, root cause untouched

8/3: all four alignment bugs fixed (a separate investigation, unrelated to 7/24)
   -> only genuine-pair similarity re-validated (0.776-0.979) — normalization untouched
   -> the raw-pixel setting silently remained

8/9: an impostor false-accept reproduced in real use
   -> alignment regression suspected -> ruled out via a control experiment (ArcFace)
   -> cross-checked against the official insightface package -> confirmed the
      normalization value was wrong
   -> corrected to the standard value (127.5/127.5) -> confirmed on-device
      (0.06-0.09) and formally confirmed via a 200-pair LFW re-measurement
      (threshold 0.30)
```

## 10. Lessons learned

- **Don't only suspect values that "look like they need retuning," like a
  threshold.** A setting like `input.normalize` that looks like it "can
  only have one correct value" can still get stuck wrong if a different
  bug (here, alignment) was tangled up with it at the moment the value was
  first chosen. Re-validating a value later also means re-questioning
  whatever assumption was true (or wasn't) back when it was set — here,
  "the crop is actually upright."
- **Swapping the control group is what splits a bug into layers.** Placing
  "an embedder (ArcFace) that correctly rejects an impostor even under bad
  rotation" next to "an embedder (AuraFace) that fails to filter an
  impostor even under good rotation" side by side immediately showed that
  rotation (the shared code) wasn't the culprit — the embedder (the part
  that changed) was. Keeping the *same* control constant was decisive in
  the 7/24 and 8/3 postmortems; this time, *changing* the control was what
  was decisive.
- **When a model itself is suspect, go all the way down to the official
  reference implementation, not just a third-party SDK.** Reading
  facekit's own adapter code or manifest wasn't enough to be sure what
  "the standard convention" even was — only going down into InsightFace's
  actual source (the graph-node-name auto-detection logic) produced the
  real answer.
- **Capturing the one frame you actually want, on a real device, is
  harder than it sounds.** If the timing of quitting with `q` races the
  last identify() call's base64 dump, the chunks get truncated and the
  crop can't be reconstructed — it took several attempts to learn that you
  have to hold the target scene in frame and then wait a good while
  (5+ seconds) before quitting. It sounds trivial, but it delayed getting
  the decisive evidence (the Elon Musk crop) three separate times.

## Appendix: tools left behind

- `tool/model_verification/verify_auraface_official.py` — a new script for
  a quick genuine/impostor similarity check via the official `insightface`
  package. Left in place for getting a fast read from a handful of photos
  when running the full LFW measurement every time is overkill.
- `--auraface-tflite` added to
  `tool/model_verification/compare_arcface_adaface.py` — lets the same
  200-pair LFW methodology already used for ArcFace/AdaFace be applied to
  AuraFace too. Each model's flag is independent, so any single one can be
  run alone.
- The per-frame crop dump in `lib/src/pipeline/face_pipeline.dart`
  (temporarily changed from "once per label" to "every identify() call",
  specifically to see an impostor frame directly) was reverted back once
  the investigation was done — leaving it on by default isn't appropriate
  since a long live session would balloon the log by ~37KB per frame.
