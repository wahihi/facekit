🇰🇷 [한국어 원문](../../KR/postmortem/2026-07-24-auraface-alignment.md)

---

# The day AuraFace's similarity stayed low — two unrelated bugs stacked on top of each other

**Date:** 2026-07-24
**Symptom:** AuraFace was wired up as the default embedding model, and even
for a genuine same-person match, cosine similarity kept bouncing around
0.14–0.37 — below the 0.40 threshold more often than not.
**Conclusion (spoiler):** Two completely unrelated bugs were stacked on top
of each other. One was a camera-rotation problem that had nothing to do with
the model; the other was a normalization-constant problem specific to this
one model. Fixing either one alone produced "somewhat better, but still
wrong" — only fixing both restored things to normal (0.81–0.94).

This is an unfiltered account of that process. There was no shortcut to the
answer — it took several wrong hypotheses to actually reach the real causes.
This project is built in collaboration with [Claude
Code](https://github.com/anthropics/claude-code), and so was this debugging
session — which is why the wrong turns are left in, not edited out.

## 0. Background

facekit is designed so the embedding model can be swapped via a BYOM (Bring
Your Own Model) structure. ArcFace (`buffalo_l`/`w600k_r50`) had been the
example app's default, but its non-commercial research license meant it
could never ship in a release build. So **AuraFace** (fal.ai, glintr100-based,
Apache 2.0) — commercially redistributable — was chosen as the new default.

Writing the manifest, adding an `'auraface'` case to `adapterForFamily()`,
double-checking the license tag, and a `tool/fetch_models.sh` script to pull
the weight from a GitHub Release — all of that was routine work. The trouble
started once enroll → identify was actually tried on a real device.

## 1. First wrong turn: "Is it channel order?"

Starting with RGB gave a low similarity. Based on a `cv2_image[:, :, ::-1]`
(BGR↔RGB flip) line in AuraFace's README example, it got switched to BGR.
That made things *worse* (0.31 → 0.25).

Digging back into InsightFace's own reference implementation
(`model_zoo/arcface_onnx.py`) and its `swapRB=True` convention later, it
turned out that flip in the README was actually "convert the BGR that cv2
reads by default into RGB right before feeding the model." In other words,
RGB was correct from the start, and switching to BGR was a step in the wrong
direction. **Channel order was never the culprit.**

## 2. Second wrong turn: "Is the model conversion broken?"

The next suspect was the `.onnx → .tflite` (fp16) conversion pipeline —
converted directly with `onnx2tf`, so a conversion bug couldn't be ruled out.
The same random input was fed to both the original `.onnx` and the converted
`.tflite` (fp32/fp16), and the outputs compared:

```
onnx vs tflite(fp32): cosine ≈ 0.9999999
onnx vs tflite(fp16): cosine ≈ 0.9999988
```

The conversion was essentially perfect. Input/output tensor shapes
(`[1,112,112,3]` → `[1,512]`) checked out too, as did whether the original
ONNX graph had any baked-in preprocessing — the very first op was a plain
`Conv`, so no normalization was hiding inside the graph. **The conversion was
innocent too.**

## 3. Third wrong turn: "Is the raw embedding weird?"

Next came a check for whether tensors were simply being misread — logging
the embedding vector's own stats (`min`/`max`/`L2 norm`). `l2Norm` came out
in the 10–15 range, `min`/`max` around ±1.3–2.5 — a completely healthy
range. No NaNs, nothing collapsing to zero, nothing saturating. **The
pipeline plumbing was fine too.**

## 4. Seeing the real problem — pulling the aligned crop off the device

At this point three suspects had been cleared and similarity was still low.
So the approach changed from "reason about it theoretically" to "just look
at what image is actually being fed to the model." Getting the aligned
112×112 crop off the device alone took three failed attempts:

1. **Clipboard + base64**: a 37KB string simply got truncated somewhere in
   the process of pasting it into chat.
2. **Write to external storage + `adb pull`**: Android's scoped storage
   blocked the app from even creating its own `Android/data/<pkg>/` folder
   (`Permission denied`) — that specifically requires going through the
   Context API, which plain `dart:io` can't reach.
3. **Local HTTP server + `adb forward`**: `HttpServer.bind` failed with
   `SocketException: Operation not permitted (errno=1)` — device policy was
   blocking server-socket creation outright.

In the end, only the crudest method survived: **chop the base64 into short
pieces, print each as its own `debugPrint` line, and reconstruct it from a
plain copy-paste of the terminal log.** Once that crop was reconstructed as
a PNG and actually looked at — it wasn't a normal face at all. It was
**either flipped 180° upside down, or sheared diagonally into a mess.**

## 5. Turns out it wasn't a model problem after all (though that wasn't obvious right away)

Seeing it "flip 180°" first raised suspicion that the sign computation in
the 5-point alignment fit (Umeyama/SVD) was numerically unstable. But while
trying to verify that hypothesis, a mistake happened — landmark numbers and
crop images from two *different* sessions got mixed up and compared as if
they were the same event. After catching and fixing that mistake, the code
was changed to log the actual computed affine matrix inside
`AffineAligner.align()`, paired exactly with the crop dump from the same
call.

Looking at the real data gathered that way (24+ calls) — **the matrix
computation itself was completely stable.** Rotation angles were smoothly
distributed, and the reflection sign never flipped unexpectedly. The
alignment math wasn't the culprit.

So why were flipped crops coming out? The **raw landmark coordinates
supplied by the camera were already rotated** before alignment ever saw
them. To check, the embedder was temporarily swapped back to ArcFace
`buffalo_l` (normally blocked from release builds by its license) and
re-tested under the exact same camera conditions — **the exact same
pattern** showed up. In other words, this wasn't AuraFace-specific at all;
it was a **problem in the shared detect/align pipeline, independent of
which model was plugged in.**

The root cause, once found, was simple: `_toFaceImage()` (the YUV→RGB
conversion) used the raw buffer straight from the camera sensor, and
**there had never been any `sensorOrientation`-based rotation compensation
in the code at all.** This gap existed from the very start of the project —
it just happened not to surface back when the app defaulted to the back
camera held landscape. Switching the default to the front camera recently
(so users could watch liveness pass on their own screen, held portrait like
a selfie) is what finally exposed it.

**Fix:** add `rotateFaceImage()` (a pure rotation function), and have
`_toFaceImage()` compute the needed rotation from the active camera's
`CameraDescription.sensorOrientation` and lens facing (front/back), applying
it before the frame ever reaches detection.

**Result:** re-verified with ArcFace `buffalo_l` — similarity jumped from
0.14–0.37 to **0.43–0.69**. Clear progress.

## 6. But AuraFace was still low

With the rotation bug fixed, AuraFace was tested again — and still sat at
0.06–0.40. **The rotation bug was real and now fixed, but AuraFace had a
completely separate second problem left.**

This time, the aligned crops were first confirmed to look correct (an
upright, recognizable face). So those two real crops (same person) were fed
into the offline verification harness, and normalization combinations were
swept systematically once more:

```
(pixel-127.5)/127.5, RGB    -> self-similarity 0.148
(pixel-127.5)/127.5, BGR    -> self-similarity 0.169
scaled to 0-1, RGB          -> self-similarity 0.169
scaled to 0-1, BGR          -> self-similarity 0.204
no normalization (raw 0-255), RGB -> self-similarity 0.9325  <- *
no normalization (raw 0-255), BGR -> self-similarity 0.9280
```

**This model expected raw, unnormalized 0–255 pixel input, not the standard
ArcFace `(pixel-127.5)/127.5` convention.** Either the normalization got
absorbed into the graph somewhere during the `glintr100.onnx` → fp16 tflite
conversion, or this particular fal.ai release simply uses a different
convention to begin with.

**Fix:** change `normalize` in `manifest.json` to `mean=[0,0,0],
std=[1,1,1]`.

**Result:** re-tested on-device — similarity **0.81–0.94**, all 15 attempts
`accepted=true`.

## 7. Summary — how the two bugs related to each other

```
No camera rotation compensation (model-independent, a pre-existing gap)
   -> first exposed in real use by switching the default to the front camera
   -> re-verified with ArcFace: 0.14-0.37 -> 0.43-0.69 (after fix)

AuraFace-specific normalization bug (specific to this one model conversion)
   -> AuraFace stayed at 0.06-0.40 even after the rotation fix
   -> re-verified with real crops: raw (unnormalized) 0.93 vs normalized 0.15-0.20

Both fixed -> 0.81-0.94, back to normal
```

Because both bugs existed at the same time, fixing either one alone produced
a "partially better, but still wrong" result. If the first (rotation) fix
had been mistaken for "done," the second problem would never have been
found — and conversely, if the rotation problem hadn't been found first,
fixing the normalization would still have left similarity oddly low, and the
conclusion would have stalled at "something's still wrong here."

## 8. What this reinforced

- **Ruling things out by reasoning (channel order, conversion fidelity) did
  genuinely rule them out** — but reasoning alone never reached the answer.
  Every real turning point came from **pulling the actual data out and
  looking at it directly** (the aligned crop image, the real affine matrix,
  the real landmark coordinates).
- **Getting data off the device was its own separate obstacle.** Clipboard,
  external storage, and a local server were each blocked for a different
  reason, and in the end only the crudest method (chopping a log into text
  and reassembling it) survived all the way through. The fancier approach
  isn't always the one that works.
- **Leave room for two causes to be stacked.** If fixing something produces
  "better, but still off," that doesn't necessarily mean the fix was wrong —
  it can mean there's another problem still sitting underneath.
- **Keeping the same control group (arcface_buffalo_l) throughout was
  decisive.** Proving the rotation bug was model-independent, and proving
  the normalization bug was AuraFace-specific, both came down to checking
  "does this reproduce with a different model under the exact same
  conditions?"

## Appendix: the debug tooling left behind

The diagnostic logging built during this investigation (base64 aligned-crop
dumps, landmark logging, affine-matrix logging, raw embedding stats) wasn't
deleted once the bugs were fixed — it's gated behind
`kFacekitVerboseDebug` in `lib/src/core/debug_flags.dart` instead. It's off
by default and doesn't ship in a normal build; flip it on for one session
with:

```bash
flutter run --dart-define=FACEKIT_VERBOSE_DEBUG=true
```
