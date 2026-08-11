"""Re-runs the ArcFace/AdaFace/AuraFace genuine-vs-impostor LFW comparison
with real face detection + 5-point alignment applied — unlike
compare_arcface_adaface.py, which only resizes the 250x250 LFW funneled
crop straight to 112x112 and explicitly notes in every manifest's
threshold_note that alignment was NOT applied.

This is "option B" from the alignment-inclusion follow-up: rather than
porting facekit's own BlazeFace anchor-decode + Umeyama solver
(lib/src/detection/blazeface_*.dart, lib/src/alignment/affine_aligner.dart)
to Python line-for-line ("option A", not done here), this reuses the
official `insightface` package's own detector (SCRFD) + its
`face_align.norm_crop()` alignment helper — already proven working in
verify_auraface_official.py. Its reference points (`arcface_dst` =
[[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],[41.5493,92.3655],
[70.7299,92.2041]]) are byte-for-byte identical to facekit's own
`arcface112Ref` in affine_aligner.dart, so this measures "the same target
geometry, via a different (but standard) detector+solver" rather than
"facekit's exact detect/align code path". Good enough to answer "does
alignment help, and does it change AuraFace's relative ranking" — not a
substitute for a from-scratch BlazeFace/Umeyama port if a defend-under-
scrutiny number is needed later.

Reuses the same embedder classes (and their manifest-matched
preprocessing) from compare_arcface_adaface.py, so the only variable that
changes between the two scripts' results is resize-only vs. detect+align
preprocessing — everything else (LFW pairs, seed, degraded-condition
proxy, EER methodology) is identical.

Usage:
    pip install insightface onnxruntime opencv-python-headless huggingface_hub \
        tensorflow pandas pyarrow pillow scikit-learn numpy
    python compare_with_alignment.py \
        --auraface-tflite ../../example/assets/models/auraface/auraface_r100_fp16.tflite \
        --arcface-tflite /path/to/w600k_r50.tflite \
        --adaface-onnx /path/to/adaface_ir101_webface12m.onnx \
        --pairs-parquet lfw_pairs_test.parquet \
        --out results_with_alignment.json

`--model-dir` (default models/auraface, auto-downloaded if missing) points
at a local `fal/AuraFace-v1` snapshot purely for its SCRFD detector — same
mechanism as verify_auraface_official.py, unrelated to which embedder
flags you pass.
"""
import argparse
import io
import json
import os
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_curve

from compare_arcface_adaface import (
    AdafaceOnnx,
    AdafaceTflite,
    ArcfaceTflite,
    AurafaceTflite,
    N_PER_CLASS,
    SEED,
    accuracy_at,
    cosine,
    eer_threshold,
    log,
)


def ensure_model(model_dir):
    if os.path.isdir(model_dir) and os.listdir(model_dir):
        log(f"using existing detector snapshot at {model_dir}")
        return
    log(f"downloading fal/AuraFace-v1 to {model_dir} (first run only, ~260MB; "
        f"only the detector is used here)...")
    from huggingface_hub import snapshot_download
    snapshot_download("fal/AuraFace-v1", local_dir=model_dir)


def build_face_app(model_dir):
    from insightface.app import FaceAnalysis
    # Same root/name resolution as verify_auraface_official.py: insightface
    # resolves models at <root>/models/<name>/ and skips its own download
    # iff that exact path already exists.
    model_dir = os.path.normpath(model_dir)
    name = os.path.basename(model_dir)
    models_parent = os.path.dirname(model_dir)
    if os.path.basename(models_parent) != "models":
        raise SystemExit(
            f"--model-dir must be a '.../models/<name>' path (e.g. models/auraface); got {model_dir!r}"
        )
    root = os.path.dirname(models_parent) or "."
    # allowed_modules=['detection'] skips loading landmark_3d_68/2d_106/
    # genderage/recognition — only SCRFD is needed here, and skipping the
    # rest avoids loading glintr100 a second time (compare_with_alignment
    # already loads it separately via --auraface-tflite if requested).
    app = FaceAnalysis(name=name, providers=["CPUExecutionProvider"], root=root,
                        allowed_modules=["detection"])
    app.prepare(ctx_id=0, det_size=(320, 320))
    return app


def align_face(app, pil_rgb):
    """Detect the highest-confidence face in a PIL RGB image and return a
    112x112 RGB numpy array aligned to the standard ArcFace 5-point
    reference. Returns None if no face was detected (caller decides the
    fallback)."""
    from insightface.utils import face_align
    rgb = np.array(pil_rgb)
    bgr = rgb[:, :, ::-1]  # SCRFD is normally driven with cv2.imread-style BGR
    faces = app.get(bgr)
    if not faces:
        return None
    faces.sort(key=lambda f: f.det_score, reverse=True)
    # cv2.warpAffine (inside norm_crop) doesn't care about channel order —
    # warping the RGB array directly with the BGR-detected keypoints (pixel
    # coordinates, channel-order-independent) yields an RGB output straight
    # away, matching what the embedder .embed() methods expect.
    return face_align.norm_crop(rgb, faces[0].kps, image_size=112)


def degrade(img112_rgb):
    im = Image.fromarray(img112_rgb)
    small = im.resize((24, 24), Image.BILINEAR)
    return np.array(small.resize((112, 112), Image.BILINEAR))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arcface-tflite")
    p.add_argument("--adaface-onnx")
    p.add_argument("--adaface-tflite")
    p.add_argument("--auraface-tflite")
    p.add_argument("--pairs-parquet", required=True)
    p.add_argument("--n-per-class", type=int, default=N_PER_CLASS)
    p.add_argument("--model-dir", default="models/auraface",
                    help="local fal/AuraFace-v1 snapshot dir, used only for its SCRFD detector")
    p.add_argument("--out", default="results_with_alignment.json")
    args = p.parse_args()
    if args.adaface_onnx and args.adaface_tflite:
        p.error("pass only one of --adaface-onnx or --adaface-tflite")

    models = []
    if args.arcface_tflite:
        log("loading ArcFace tflite...")
        models.append(("arcface", ArcfaceTflite(args.arcface_tflite)))
    if args.adaface_onnx or args.adaface_tflite:
        log("loading AdaFace...")
        models.append(("adaface", AdafaceTflite(args.adaface_tflite) if args.adaface_tflite else AdafaceOnnx(args.adaface_onnx)))
    if args.auraface_tflite:
        log("loading AuraFace tflite...")
        models.append(("auraface", AurafaceTflite(args.auraface_tflite)))
    if not models:
        p.error("pass at least one of --arcface-tflite / --adaface-onnx / --adaface-tflite / --auraface-tflite")

    ensure_model(args.model_dir)
    face_app = build_face_app(args.model_dir)

    df = pd.read_parquet(args.pairs_parquet)
    genuine = df[df["pair"] == 1].sample(n=args.n_per_class, random_state=SEED).reset_index(drop=True)
    impostor = df[df["pair"] == 0].sample(n=args.n_per_class, random_state=SEED).reset_index(drop=True)
    pairs_df = pd.concat([genuine.assign(label=1), impostor.assign(label=0)]).reset_index(drop=True)

    # Pass 1: detect+align every image exactly once (shared across all
    # models below), falling back to a plain resize if SCRFD finds no face
    # — LFW funneled crops are mostly clean frontal faces, so this should
    # be rare, but a pair shouldn't just vanish from the sample over it.
    log(f"aligning {len(pairs_df) * 2} images...")
    aligned = {}
    fallback_count = 0
    t0 = time.time()
    for n, (_, row) in enumerate(pairs_df.iterrows()):
        for col in ("img_0", "img_1"):
            pil = Image.open(io.BytesIO(row[col]["bytes"])).convert("RGB")
            crop = align_face(face_app, pil)
            if crop is None:
                fallback_count += 1
                crop = np.array(pil.resize((112, 112), Image.BILINEAR))
            aligned[(n, col)] = crop
        if (n + 1) % 25 == 0:
            log(f"align: {n + 1}/{len(pairs_df)} pairs ({time.time() - t0:.1f}s elapsed)")
    log(f"alignment done: {fallback_count}/{len(pairs_df) * 2} images had no SCRFD "
        f"detection and fell back to plain resize ({time.time() - t0:.1f}s total)")

    results = {}
    for model_name, model in models:
        for condition in ["clean", "degraded"]:
            t0 = time.time()
            gscores, iscores = [], []
            for n, (_, row) in enumerate(pairs_df.iterrows()):
                img0 = aligned[(n, "img_0")]
                img1 = aligned[(n, "img_1")]
                if condition == "degraded":
                    img0 = degrade(img0)
                    img1 = degrade(img1)
                sim = cosine(model.embed(img0), model.embed(img1))
                (gscores if row["label"] == 1 else iscores).append(sim)
            results[(model_name, condition)] = (gscores, iscores)
            log(f"{model_name}/{condition}: genuine mean={np.mean(gscores):.4f} "
                f"impostor mean={np.mean(iscores):.4f} n={len(gscores)}/{len(iscores)} ({time.time() - t0:.1f}s)")

    summary = {"fallback_to_resize": fallback_count, "total_images": len(pairs_df) * 2}
    for model_name, _ in models:
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
