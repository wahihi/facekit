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
