# Model verification scripts

Developer-only Python tooling used to derive the `matching.threshold` values
in the embedder manifests (`assets/models/*/manifest.json`). Not part of the
shipped Dart SDK, not run in CI, and doesn't touch any model weights or
datasets bundled in this repo — per the BYOM policy in CLAUDE.md, you supply
your own locally-converted weights.

## compare_arcface_adaface.py

Computes genuine/impostor cosine-similarity distributions for the real
ArcFace and AdaFace weights on a clean and a simulated-low-quality LFW
subset, and the EER-based threshold each implies. See
`doc/KR/adaface_verification.md` for the methodology and results this
produced.

Inputs you need to provide yourself:
- `w600k_r50.tflite` — see `assets/models/arcface_buffalo_l/manifest.json` license.note for sourcing.
- `adaface_ir101_webface12m.onnx` **or** `.tflite` (`--adaface-onnx` / `--adaface-tflite`, pick one) — see `assets/models/adaface_ir101_webface12m/manifest.json` license.note. Export via the official [mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace) checkpoint + `torch.onnx.export` (their `net.build_model('ir_101')`, `forward()` returns `(feature, norm)`), then optionally convert to `.tflite` via onnx2tf.
- A pairs parquet matching the schema of the `pairs` config of [huggingface.co/datasets/logasja/lfw](https://huggingface.co/datasets/logasja/lfw) (columns: `pair`, `img_0`, `img_1`) — fetch via `https://huggingface.co/api/datasets/logasja/lfw/parquet/pairs/test/0.parquet`.

```
pip install tensorflow onnxruntime pandas pyarrow pillow scikit-learn numpy
python compare_arcface_adaface.py \
  --arcface-tflite /path/to/w600k_r50.tflite \
  --adaface-onnx /path/to/adaface_ir101_webface12m.onnx \
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
