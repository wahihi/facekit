"""Re-runs the LFW genuine/impostor comparison with YuNet (OpenCV Zoo,
MIT-licensed, native 5-point landmarks) for detection+alignment, instead of
compare_with_alignment.py's insightface/SCRFD (whose pretrained weights
turned out to be non-commercial-only — see doc/KR/adaface_verification.md
and the SCRFD license investigation this follows on from).

Unlike BlazeFace (6 keypoints but only 1 mouth point, forcing facekit's
4-point Umeyama compromise in AffineAligner), YuNet natively outputs the
full ArcFace-style 5 points (2 eyes, nose, 2 *separate* mouth corners) —
this checks whether that alone (with facekit's other code unchanged)
recovers a similar accuracy gain to what SCRFD showed, using a detector
that's actually safe to bundle.

Detection/landmarks come from OpenCV's own cv2.FaceDetectorYN (no
hand-written decode — reuses the same reasoning as using the official
`insightface` package for SCRFD: trust the reference implementation for
the detector, keep the embedder classes identical to isolate the one
variable that's changing). Alignment reuses insightface's
face_align.norm_crop() with YuNet's landmarks reordered into ArcFace's
convention (YuNet gives right_eye,left_eye,nose,right_mouth,left_mouth;
ArcFace's arcface_dst expects left_eye,right_eye,nose,left_mouth,
right_mouth) — same target reference points as compare_with_alignment.py,
so results are directly comparable.

Usage:
    pip install opencv-python insightface tensorflow onnxruntime pandas pyarrow pillow scikit-learn numpy
    python compare_with_yunet.py \
        --yunet-onnx yunet.onnx \
        --auraface-tflite ../../example/assets/models/auraface/auraface_r100_fp16.tflite \
        --arcface-tflite ../../example/assets/models/arcface_buffalo_l/w600k_r50.tflite \
        --pairs-parquet lfw_pairs_test.parquet \
        --out results_yunet.json
"""
import argparse
import io
import json
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image

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


def align_face(detector, pil_rgb):
    """Detect the highest-confidence face in a PIL RGB image with YuNet and
    return a 112x112 RGB numpy array aligned to the standard ArcFace 5-point
    reference. Returns None if no face was detected."""
    from insightface.utils import face_align

    rgb = np.array(pil_rgb)
    bgr = rgb[:, :, ::-1]  # cv2.FaceDetectorYN expects BGR, like cv2.imread
    detector.setInputSize((bgr.shape[1], bgr.shape[0]))
    _, faces = detector.detect(bgr)
    if faces is None or len(faces) == 0:
        return None
    # faces: Nx15 [x,y,w,h, x_re,y_re, x_le,y_le, x_nt,y_nt, x_rmc,y_rmc, x_lmc,y_lmc, score]
    #
    # TEMP DEBUG-derived fix: YuNet's own "right_eye"/"left_eye" naming turns
    # out to be the *opposite* handedness from facekit's arcface_dst array
    # (same class of bug as the BlazeFace eye-order fix documented in
    # doc/KR/postmortem/2026-08-03-affine-aligner-alignment.md — a model
    # author's own landmark names aren't safe to trust without verifying
    # against the actual pixel positions). Verified directly: feeding
    # YuNet's raw (right_eye,left_eye,nose,right_mouth,left_mouth) order
    # straight into face_align.norm_crop (no relabeling) fits a sane
    # scale/rotation (~0.9, ~a few degrees) on real LFW crops; the
    # "corrected" swap that seemed obviously right by name produced a
    # degenerate ~0.125 scale on every image, which is what silently wrecked
    # the first full run's EER (~48%, i.e. no discrimination at all).
    best = faces[np.argmax(faces[:, -1])]
    kps = best[4:14].reshape(5, 2).astype(np.float32)
    return face_align.norm_crop(rgb, kps, image_size=112)


def degrade(img112_rgb):
    im = Image.fromarray(img112_rgb)
    small = im.resize((24, 24), Image.BILINEAR)
    return np.array(small.resize((112, 112), Image.BILINEAR))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yunet-onnx", required=True)
    p.add_argument("--arcface-tflite")
    p.add_argument("--adaface-onnx")
    p.add_argument("--adaface-tflite")
    p.add_argument("--auraface-tflite")
    p.add_argument("--pairs-parquet", required=True)
    p.add_argument("--n-per-class", type=int, default=N_PER_CLASS)
    p.add_argument("--score-threshold", type=float, default=0.6)
    p.add_argument("--out", default="results_yunet.json")
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

    log("loading YuNet...")
    detector = cv2.FaceDetectorYN_create(
        args.yunet_onnx, "", (320, 320),
        score_threshold=args.score_threshold, nms_threshold=0.3, top_k=5000,
    )

    df = pd.read_parquet(args.pairs_parquet)
    genuine = df[df["pair"] == 1].sample(n=args.n_per_class, random_state=SEED).reset_index(drop=True)
    impostor = df[df["pair"] == 0].sample(n=args.n_per_class, random_state=SEED).reset_index(drop=True)
    pairs_df = pd.concat([genuine.assign(label=1), impostor.assign(label=0)]).reset_index(drop=True)

    log(f"aligning {len(pairs_df) * 2} images with YuNet...")
    aligned = {}
    fallback_count = 0
    t0 = time.time()
    for n, (_, row) in enumerate(pairs_df.iterrows()):
        for col in ("img_0", "img_1"):
            pil = Image.open(io.BytesIO(row[col]["bytes"])).convert("RGB")
            crop = align_face(detector, pil)
            if crop is None:
                fallback_count += 1
                crop = np.array(pil.resize((112, 112), Image.BILINEAR))
            aligned[(n, col)] = crop
        if (n + 1) % 25 == 0:
            log(f"align: {n + 1}/{len(pairs_df)} pairs ({time.time() - t0:.1f}s elapsed)")
    log(f"alignment done: {fallback_count}/{len(pairs_df) * 2} images had no YuNet "
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
