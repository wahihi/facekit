# Model verification scripts

Developer-only Python tooling used to derive the `matching.threshold` values
in the embedder manifests (`assets/models/*/manifest.json`). Not part of the
shipped Dart SDK, not run in CI, and doesn't touch any model weights or
datasets bundled in this repo — per the BYOM policy in CLAUDE.md, you supply
your own locally-converted weights.

## compare_arcface_adaface.py

Computes genuine/impostor cosine-similarity distributions for the real
ArcFace, AdaFace, and AuraFace weights on a clean and a simulated-low-quality
LFW subset, and the EER-based threshold each implies. See
`doc/KR/adaface_verification.md` for the ArcFace/AdaFace methodology and
results this produced; AuraFace support (`--auraface-tflite`) was added
2026-08-09 to give AuraFace a real measured threshold instead of the
placeholder 0.40 — see `doc/KR/postmortem/` for why that was overdue (the
manifest's normalization was wrong until this same date, which made
AuraFace unable to tell two different people apart on a real device).

Each model flag is independent — pass just the one(s) you want. All three
share `--pairs-parquet`.

Inputs you need to provide yourself:
- `w600k_r50.tflite` (`--arcface-tflite`) — see `assets/models/arcface_buffalo_l/manifest.json` license.note for sourcing.
- `adaface_ir101_webface12m.onnx` **or** `.tflite` (`--adaface-onnx` / `--adaface-tflite`, pick one) — see `assets/models/adaface_ir101_webface12m/manifest.json` license.note. Export via the official [mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace) checkpoint + `torch.onnx.export` (their `net.build_model('ir_101')`, `forward()` returns `(feature, norm)`), then optionally convert to `.tflite` via onnx2tf.
- `auraface_r100_fp16.tflite` (`--auraface-tflite`) — already present at `example/assets/models/auraface/auraface_r100_fp16.tflite` if you've run `tool/fetch_models.sh` (or `verify_auraface_official.py`, which fetches the `.onnx` separately to `models/auraface/`); point `--auraface-tflite` at whichever `.tflite` copy you have.
- A pairs parquet matching the schema of the `pairs` config of [huggingface.co/datasets/logasja/lfw](https://huggingface.co/datasets/logasja/lfw) (columns: `pair`, `img_0`, `img_1`) — fetch via `https://huggingface.co/api/datasets/logasja/lfw/parquet/pairs/test/0.parquet`.

```
pip install tensorflow onnxruntime pandas pyarrow pillow scikit-learn numpy

# AuraFace only:
python compare_arcface_adaface.py \
  --auraface-tflite ../../example/assets/models/auraface/auraface_r100_fp16.tflite \
  --pairs-parquet /path/to/lfw_pairs_test.parquet \
  --out results_auraface.json

# all three at once:
python compare_arcface_adaface.py \
  --arcface-tflite /path/to/w600k_r50.tflite \
  --adaface-onnx /path/to/adaface_ir101_webface12m.onnx \
  --auraface-tflite ../../example/assets/models/auraface/auraface_r100_fp16.tflite \
  --pairs-parquet /path/to/lfw_pairs_test.parquet
```

Note: AdaFace's IR-101 backbone is heavy enough that onnxruntime's default
`ORT_ENABLE_ALL` graph-optimization pass can take a very long time (we saw
it effectively hang for over an hour) on a memory-constrained machine —
this script already sets `ORT_ENABLE_BASIC` to avoid that.

`results_2026-06-25.json` was produced with `--adaface-onnx`; `results_tflite_2026-06-26.json`
re-ran the same 200 pairs with `--adaface-tflite` against the actual converted `.tflite` and
matched to within float32 rounding noise — confirming the `.onnx`-based threshold/EER numbers
still hold for the format the Dart SDK actually loads.

After a run produces `threshold_clean_eer` / `eer_clean` for AuraFace,
transplant those numbers into `example/assets/models/auraface/manifest.json`'s
`matching.threshold` / `threshold_note`, replacing the current "임시값"
placeholder — same as was already done for ArcFace/AdaFace.

## verify_auraface_official.py

A real-device facekit session (AuraFace embedder) scored an enrolled user
and an unrelated person's photo almost identically (~0.84-0.95 cosine
similarity for both) — AuraFace wasn't discriminating between them. Before
concluding the checkpoint itself is weak, this script reproduces the same
genuine-vs-impostor comparison through the *official* `insightface` Python
package end-to-end (its own SCRFD detector + its own alignment + its own
ONNX embedder) — none of facekit's Dart code is involved, so it isolates
"is the model weak" from "does facekit's own pipeline have a bug".

```
pip install insightface onnxruntime opencv-python-headless huggingface_hub numpy
python verify_auraface_official.py \
  --image me_1.jpg me --image me_2.jpg me --image other_1.jpg other
```

Downloads `fal/AuraFace-v1` (~260MB) to `./models/auraface` on first run
(same source as the app's `tool/fetch_models.sh`), unless `--model-dir`
points at an existing snapshot. Needs plain photo files, not aligned crops —
the official SCRFD detector + aligner run on the whole image. Prints every
pairwise cosine similarity plus a genuine/impostor separation verdict: if
the official pipeline *also* can't separate the two identities, that
points at the checkpoint itself rather than a facekit bug — see
`doc/KR/postmortem/` for the investigation this follows on from.

## compare_with_alignment.py

`compare_arcface_adaface.py`'s LFW measurement only resizes each 250x250
funneled crop to 112x112 — every manifest's `threshold_note` says so
explicitly ("5점 랜드마크 정렬 미적용"). This script re-runs the same
genuine/impostor comparison with real face detection + 5-point alignment
applied first, to check how much of the measured accuracy gap (AuraFace's
10.0% EER vs. ArcFace's 8.5% and AdaFace's 2.0%) is actually attributable
to skipping alignment rather than the embedding models themselves.

It doesn't port facekit's own BlazeFace decoder + Umeyama solver
(`lib/src/detection/blazeface_*.dart`, `lib/src/alignment/affine_aligner.dart`)
to Python line-for-line — instead it reuses the official `insightface`
package's SCRFD detector + `face_align.norm_crop()` (same mechanism as
`verify_auraface_official.py`). Its reference points are byte-identical to
facekit's own `arcface112Ref`, so this measures "the same target geometry,
via a different but standard detector+solver" — a good proxy for "does
alignment help", not a substitute for facekit's exact code path if a
number that has to withstand scrutiny (e.g. a paper or product spec) is
needed later.

Shares `compare_arcface_adaface.py`'s embedder classes directly (imports
them), so the *only* variable that changes between the two scripts'
results is resize-only vs. detect+align preprocessing.

```
pip install insightface onnxruntime opencv-python-headless huggingface_hub \
  tensorflow pandas pyarrow pillow scikit-learn numpy
python compare_with_alignment.py \
  --auraface-tflite ../../example/assets/models/auraface/auraface_r100_fp16.tflite \
  --arcface-tflite /path/to/w600k_r50.tflite \
  --adaface-onnx /path/to/adaface_ir101_webface12m.onnx \
  --pairs-parquet lfw_pairs_test.parquet \
  --out results_with_alignment.json
```

Each image is detected+aligned exactly once and the result reused across
all requested models (not re-detected per model). If SCRFD finds no face
in a given LFW crop it falls back to a plain resize for that one image
(logged as a count) rather than dropping the pair — LFW funneled crops are
mostly clean frontal faces so this should be rare.

## compare_with_yunet.py

SCRFD's pretrained weights turned out to be non-commercial-only (see
`doc/KR/postmortem/2026-08-12-yunet-landmark-order.md`) — this re-runs the
same alignment-comparison methodology with **YuNet**
(OpenCV Zoo/libfacedetection, MIT-licensed, already named as bundleable in
`CLAUDE.md`) instead. Detection uses `cv2.FaceDetectorYN` (built into
opencv-python); alignment still goes through
`insightface.utils.face_align.norm_crop()` with the same `arcface_dst`
reference, so results are directly comparable to
`compare_with_alignment.py`'s SCRFD numbers. Shares
`compare_arcface_adaface.py`'s embedder classes, same as the other two
scripts.

```
pip install opencv-python insightface tensorflow onnxruntime pandas pyarrow pillow scikit-learn numpy
curl -sL -o yunet.onnx "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
python compare_with_yunet.py \
  --yunet-onnx yunet.onnx \
  --auraface-tflite ../../example/assets/models/auraface/auraface_r100_fp16.tflite \
  --arcface-tflite /path/to/w600k_r50.tflite \
  --pairs-parquet lfw_pairs_test.parquet \
  --out results_yunet.json
```

**Landmark order pitfall (read before touching the alignment code):**
`cv2.FaceDetectorYN`'s output row is `[x,y,w,h, x_re,y_re, x_le,y_le,
x_nt,y_nt, x_rmc,y_rmc, x_lmc,y_lmc, score]` — but YuNet's own
`right_eye`/`left_eye`/`right_mouth`/`left_mouth` naming is the *opposite*
handedness from facekit's `arcface_dst` convention. Feed the raw order
straight into `norm_crop` with **no relabeling** (`best[4:14].reshape(5,
2)`, exactly as this script does) — "correcting" it to match ArcFace's
named order by swapping left/right produces a fitted scale of ~0.125
(should be ~0.9), i.e. a tiny face on a mostly-black crop, and collapses
EER to ~48% (random). This is the same class of bug as the BlazeFace
eye-order fix in `lib/src/alignment/affine_aligner.dart` — verify against
actual pixel positions, never trust a landmark name as-is.
