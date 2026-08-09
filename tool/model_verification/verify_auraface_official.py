"""Genuine/impostor cosine-similarity check for AuraFace using the *official*
insightface Python package end-to-end (its own SCRFD detector + its own
5-point alignment + its own ONNX embedder) — not a single line of facekit's
Dart pipeline is involved.

Why this script exists: a real-device facekit session (AuraFace embedder,
AffineAligner alignment) showed the enrolled user and an unrelated photo
(different person) scoring almost the same cosine similarity (~0.84-0.95
for both) — i.e. AuraFace wasn't discriminating between them. Before
concluding "this model/checkpoint is just weak", we need to rule out a
facekit-side bug (detection, alignment, TFLite conversion, adapter
preprocessing) by reproducing the same genuine-vs-impostor comparison
through the *reference* implementation the model's own README documents.
See doc/KR/postmortem/ for the debugging history this follows on from.

Two outcomes, and what each means:
  - Official pipeline ALSO fails to separate genuine/impostor pairs
    -> the AuraFace-v1 (glintr100) checkpoint itself has weak discriminative
       power in these conditions; not a facekit bug. Pick another embedder
       for production use.
  - Official pipeline clearly separates them (genuine notably higher than
    impostor)
    -> something in facekit's own pipeline still differs from the official
       one (detection/alignment quality, TFLite conversion, adapter
       preprocessing) and is worth chasing further.

This is a one-off developer tool, not part of the shipped Dart SDK. Not run
in CI. Doesn't touch any weights bundled in this repo — per the BYOM policy
in CLAUDE.md you supply your own (this script will download AuraFace-v1's
public weights from Hugging Face on first run, same source facekit's own
tool/fetch_models.sh uses).

Usage:
    pip install insightface onnxruntime opencv-python-headless huggingface_hub numpy

    # at least 2 photos of the enrolled identity + 1+ of a different person;
    # plain image files (jpg/png), any size — insightface detects+crops itself.
    python verify_auraface_official.py \
        --image me_1.jpg me \
        --image me_2.jpg me \
        --image other_1.jpg other

    # first run only: downloads ~260MB to ./models/auraface (skips if present)
    # pass --model-dir to reuse an existing fal/AuraFace-v1 snapshot instead.
"""
import argparse
import itertools
import os
import sys

import cv2
import numpy as np


def log(msg):
    print(f"[verify_auraface_official] {msg}", flush=True)


def ensure_model(model_dir):
    if os.path.isdir(model_dir) and os.listdir(model_dir):
        log(f"using existing model snapshot at {model_dir}")
        return
    log(f"downloading fal/AuraFace-v1 to {model_dir} (first run only, ~260MB)...")
    from huggingface_hub import snapshot_download
    snapshot_download("fal/AuraFace-v1", local_dir=model_dir)


def load_face_app(model_dir):
    from insightface.app import FaceAnalysis
    # insightface's own storage.py resolves models at <root>/models/<name>/
    # and *skips* its own auto-download from insightface's GitHub releases
    # iff that exact path already exists (os.path.exists check, no content
    # check) — so root here must be the directory that *contains* "models/",
    # not model_dir's immediate parent. model_dir is expected to look like
    # ".../models/auraface"; root ends up two levels up.
    model_dir = os.path.normpath(model_dir)
    name = os.path.basename(model_dir)
    models_parent = os.path.dirname(model_dir)
    if os.path.basename(models_parent) != "models":
        raise SystemExit(
            f"--model-dir must be a '.../models/<name>' path (e.g. models/auraface) "
            f"so insightface's root/name convention can find it; got {model_dir!r}"
        )
    root = os.path.dirname(models_parent) or "."
    app = FaceAnalysis(name=name, providers=["CPUExecutionProvider"], root=root)
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def embed_image(app, path):
    img = cv2.imread(path)  # BGR, as insightface's own pipeline expects
    if img is None:
        raise SystemExit(f"could not read image: {path}")
    faces = app.get(img)
    if not faces:
        raise SystemExit(
            f"no face detected in {path} — official SCRFD detector found "
            "nothing; try a clearer/more frontal photo"
        )
    if len(faces) > 1:
        log(f"WARNING: {len(faces)} faces detected in {path}, using the highest-confidence one")
    faces.sort(key=lambda f: f.det_score, reverse=True)
    return faces[0].normed_embedding  # already L2-normalised


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--image", nargs=2, action="append", metavar=("PATH", "LABEL"), required=True,
        help="an image file + an identity label (repeat for each photo; "
             "same label = same person, for genuine vs impostor grouping)",
    )
    parser.add_argument(
        "--model-dir", default="models/auraface",
        help="local AuraFace-v1 snapshot dir (default: ./models/auraface, "
             "auto-downloaded from Hugging Face if missing)",
    )
    args = parser.parse_args()

    ensure_model(args.model_dir)
    app = load_face_app(args.model_dir)

    log(f"embedding {len(args.image)} image(s)...")
    entries = []  # (path, label, embedding)
    for path, label in args.image:
        emb = embed_image(app, path)
        entries.append((path, label, emb))
        log(f"  {path} [{label}] -> {emb.shape[0]}-dim, l2norm={np.linalg.norm(emb):.4f}")

    print()
    print(f"{'pair':<45} {'kind':<10} {'cosine sim':>10}")
    print("-" * 68)
    genuine, impostor = [], []
    for (p1, l1, e1), (p2, l2, e2) in itertools.combinations(entries, 2):
        sim = float(np.dot(e1, e2))  # both already L2-normalised -> dot == cosine
        kind = "genuine" if l1 == l2 else "impostor"
        (genuine if kind == "genuine" else impostor).append(sim)
        pair_label = f"{os.path.basename(p1)} vs {os.path.basename(p2)}"
        print(f"{pair_label:<45} {kind:<10} {sim:>10.4f}")

    print()
    if genuine:
        print(f"genuine  pairs: n={len(genuine)}  min={min(genuine):.4f}  max={max(genuine):.4f}  mean={sum(genuine)/len(genuine):.4f}")
    if impostor:
        print(f"impostor pairs: n={len(impostor)}  min={min(impostor):.4f}  max={max(impostor):.4f}  mean={sum(impostor)/len(impostor):.4f}")

    if genuine and impostor:
        gap = min(genuine) - max(impostor)
        print()
        if gap > 0.1:
            print(f"-> clear separation (genuine min - impostor max = {gap:+.4f}): "
                  "official pipeline DOES discriminate. facekit likely still has "
                  "a bug elsewhere (detection/alignment/conversion) — keep digging there.")
        elif gap > 0:
            print(f"-> thin separation (genuine min - impostor max = {gap:+.4f}): "
                  "borderline; more photos per identity would firm this up.")
        else:
            print(f"-> OVERLAP (genuine min - impostor max = {gap:+.4f}, ranges cross): "
                  "official reference pipeline ALSO fails to separate these identities. "
                  "This points to the AuraFace-v1 checkpoint itself, not a facekit bug.")


if __name__ == "__main__":
    sys.exit(main())
