"""Genuine/impostor cosine-similarity comparison between the real ArcFace
(buffalo_l/w600k_r50) and AdaFace (IR-101/WebFace12M) weights, on a clean
and a simulated-low-quality condition. Produces the numbers behind the
`matching.threshold` / `threshold_note` fields in both models' manifest.json
and the writeup in doc/KR/adaface_verification.md.

This is a one-off developer tool, not part of the shipped SDK: per the
BYOM policy in CLAUDE.md, neither model's weights are bundled in this repo.
You must supply your own locally-converted copies (see each manifest's
license.note for where to source them).

Usage:
    pip install tensorflow onnxruntime pandas pyarrow pillow scikit-learn numpy
    python compare_arcface_adaface.py \
        --arcface-tflite /path/to/w600k_r50.tflite \
        --adaface-onnx /path/to/adaface_ir101_webface12m.onnx \
        --pairs-parquet /path/to/lfw_pairs_test.parquet

`--pairs-parquet` expects the schema of the `pairs` config of the
huggingface.co/datasets/logasja/lfw dataset: columns `pair` (1=same person,
0=different), `img_0`, `img_1` (each a dict with an image-bytes field).
"""
import argparse
import io
import json
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import roc_curve

N_PER_CLASS = 100  # 100 genuine + 100 impostor pairs
SEED = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class ArcfaceTflite:
    """Real arcface_buffalo_l weights via the .tflite shipped in example/assets."""

    def __init__(self, path):
        self.interp = tf.lite.Interpreter(model_path=path)
        self.interp.allocate_tensors()
        self.in_idx = self.interp.get_input_details()[0]["index"]
        self.out_idx = self.interp.get_output_details()[0]["index"]

    def embed(self, rgb_112):
        # manifest: color=RGB, mean/std=127.5 -> (x-127.5)/127.5
        x = (rgb_112.astype(np.float32) - 127.5) / 127.5
        x = x[np.newaxis, :, :, :]  # NHWC
        self.interp.set_tensor(self.in_idx, x)
        self.interp.invoke()
        return self.interp.get_tensor(self.out_idx)[0]


class AdafaceOnnx:
    """Real AdaFace IR-101/WebFace12M checkpoint, exported to ONNX (BYOM)."""

    def __init__(self, path):
        opts = ort.SessionOptions()
        # ORT_ENABLE_ALL's extra graph-opt passes are themselves slow on this
        # 245MB IR-101 graph under memory pressure; basic opts is much faster
        # for a one-off local validation run and doesn't change the output.
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opts.intra_op_num_threads = 4
        log("creating AdaFace onnxruntime session...")
        t0 = time.time()
        self.sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        log(f"AdaFace session ready in {time.time() - t0:.1f}s")
        self.input_name = self.sess.get_inputs()[0].name

    def embed(self, rgb_112):
        # manifest: color=BGR, mean/std=127.5 -> (x_bgr-127.5)/127.5
        # (matches official AdaFace to_input(): ((rgb[...,::-1]/255)-.5)/.5)
        bgr = rgb_112[:, :, ::-1]
        x = (bgr.astype(np.float32) - 127.5) / 127.5
        x = np.transpose(x, (2, 0, 1))[np.newaxis, :, :, :]  # NCHW (onnx export convention)
        feature, _norm = self.sess.run(None, {self.input_name: x})  # 'norm' output is unused (training-time quality signal)
        return feature[0]


class AdafaceTflite:
    """Real AdaFace IR-101/WebFace12M checkpoint, converted to .tflite (BYOM) —
    same artifact the Dart SDK's TfliteFaceEmbedder loads, used here to confirm
    the .onnx-based comparison above still holds after TFLite conversion."""

    def __init__(self, path):
        self.interp = tf.lite.Interpreter(model_path=path)
        self.interp.allocate_tensors()
        self.in_idx = self.interp.get_input_details()[0]["index"]
        self.out_idx = self.interp.get_output_details()[0]["index"]  # 'feature'; 'norm' output unused

    def embed(self, rgb_112):
        # manifest: color=BGR, mean/std=127.5 -> (x_bgr-127.5)/127.5 (NHWC, matches the .onnx export's NCHW layout transposed)
        bgr = rgb_112[:, :, ::-1]
        x = (bgr.astype(np.float32) - 127.5) / 127.5
        x = x[np.newaxis, :, :, :]  # NHWC
        self.interp.set_tensor(self.in_idx, x)
        self.interp.invoke()
        return self.interp.get_tensor(self.out_idx)[0]


def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


def degrade(img112):
    """Simulate a low-quality / distant-camera capture: heavy downsample then upsample back."""
    small = img112.resize((24, 24), Image.BILINEAR)
    return small.resize((112, 112), Image.BILINEAR)


def eer_threshold(genuine_scores, impostor_scores):
    y = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    s = np.concatenate([genuine_scores, impostor_scores])
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float(thr[idx]), float(fpr[idx]), float(fnr[idx])


def accuracy_at(genuine_scores, impostor_scores, threshold):
    tp = np.sum(np.array(genuine_scores) >= threshold)
    tn = np.sum(np.array(impostor_scores) < threshold)
    total = len(genuine_scores) + len(impostor_scores)
    return (tp + tn) / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arcface-tflite", required=True)
    p.add_argument("--adaface-onnx")
    p.add_argument("--adaface-tflite")
    p.add_argument("--pairs-parquet", required=True)
    p.add_argument("--n-per-class", type=int, default=N_PER_CLASS)
    p.add_argument("--out", default="results.json")
    args = p.parse_args()
    if not args.adaface_onnx and not args.adaface_tflite:
        p.error("one of --adaface-onnx or --adaface-tflite is required")

    df = pd.read_parquet(args.pairs_parquet)
    genuine = df[df["pair"] == 1].sample(n=args.n_per_class, random_state=SEED).reset_index(drop=True)
    impostor = df[df["pair"] == 0].sample(n=args.n_per_class, random_state=SEED).reset_index(drop=True)
    pairs_df = pd.concat([genuine.assign(label=1), impostor.assign(label=0)]).reset_index(drop=True)

    log("loading ArcFace tflite...")
    arcface = ArcfaceTflite(args.arcface_tflite)
    log("ArcFace ready")
    adaface = AdafaceTflite(args.adaface_tflite) if args.adaface_tflite else AdafaceOnnx(args.adaface_onnx)

    def load112(cell):
        return Image.open(io.BytesIO(cell["bytes"])).convert("RGB").resize((112, 112), Image.BILINEAR)

    results = {}
    for model_name, model in [("arcface", arcface), ("adaface", adaface)]:
        for condition in ["clean", "degraded"]:
            t0 = time.time()
            gscores, iscores = [], []
            for n, (_, row) in enumerate(pairs_df.iterrows()):
                img0 = load112(row["img_0"])
                img1 = load112(row["img_1"])
                if condition == "degraded":
                    img0 = degrade(img0)
                    img1 = degrade(img1)
                sim = cosine(model.embed(np.array(img0)), model.embed(np.array(img1)))
                (gscores if row["label"] == 1 else iscores).append(sim)
                if (n + 1) % 25 == 0:
                    log(f"{model_name}/{condition}: {n + 1}/{len(pairs_df)} pairs ({time.time() - t0:.1f}s elapsed)")
            results[(model_name, condition)] = (gscores, iscores)
            log(f"{model_name}/{condition}: genuine mean={np.mean(gscores):.4f} "
                f"impostor mean={np.mean(iscores):.4f} n={len(gscores)}/{len(iscores)} ({time.time() - t0:.1f}s total)")

    summary = {}
    for model_name in ["arcface", "adaface"]:
        gscores_clean, iscores_clean = results[(model_name, "clean")]
        thr, fpr, fnr = eer_threshold(gscores_clean, iscores_clean)
        gscores_deg, iscores_deg = results[(model_name, "degraded")]
        thr_deg, fpr_deg, fnr_deg = eer_threshold(gscores_deg, iscores_deg)
        summary[model_name] = {
            "threshold_clean_eer": thr,
            "eer_clean": (fpr + fnr) / 2,
            "accuracy_clean_at_threshold": accuracy_at(gscores_clean, iscores_clean, thr),
            "accuracy_degraded_at_clean_threshold": accuracy_at(gscores_deg, iscores_deg, thr),
            "threshold_degraded_eer": thr_deg,
            "eer_degraded": (fpr_deg + fnr_deg) / 2,
        }
        log(f"{model_name}: {json.dumps(summary[model_name], indent=2)}")

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
