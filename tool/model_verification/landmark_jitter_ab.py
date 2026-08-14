"""A/B test: does landmark jitter shrink when the face fills more of the
160x160 YuNet input?

Motivated by doc/KR/postmortem 2026-08-13 investigation into unstable
detection/recognition after the BlazeFace->YuNet switch. Log analysis of a
real device session showed similarity swings correlate strongly with the
AffineAligner's fitted scale/rotation (corr -0.7~-0.8), not with detect
score (-0.24) -- and one frame pair had bbox essentially unchanged while
eye-to-eye landmark distance jumped 24%. Hypothesis: lib/src/image/
image_converter.dart's resizeNearest squeezes the *whole* camera frame down
to a fixed 160x160 (lib/src/detection/yunet_detector.dart), so a face that's
only a small fraction of the frame gets very few input pixels -- ordinary
per-frame camera sensor noise then moves the decoded landmarks by a large
fraction of the face's own size. The user also reported anecdotally that
holding the phone so the face fills more of the frame (e.g. facing an LCD
screen close up) made liveness/recognition noticeably more stable.

This script re-implements yunet_detector.dart's preprocessing and
yunet_decoder.dart's decode formula byte-for-byte in Python (same resolved
stride/channel logic, same score=sqrt(cls*obj), same grid-offset bbox/kps
formula) against the actual bundled assets/models/yunet_160/yunet_160.tflite
-- not cv2.FaceDetectorYN, which uses a different input size/pipeline.

Method: take one real photo, find one face box in it (via YuNet itself, at
generous framing). Build two synthetic "camera crops" around that face --
SMALL (wide margin, face is a small fraction of the crop, like a
distant/typical phone framing) and LARGE (tight margin, face fills most of
the crop, like the user's close-to-LCD case) -- each downstream-resized to
160x160 by the same resizeNearest squeeze the real pipeline uses. Then apply
N independent trials of a shared noise model (a few pixels of random
sub-pixel translation + mild Gaussian sensor noise) to each crop *before*
the 160x160 resize, run the real decode on every trial, and compare
landmark jitter (std-dev in original-photo pixels, normalised by the face's
own eye-distance so SMALL/LARGE are comparable) between the two framings.

Usage:
    python landmark_jitter_ab.py --image me_1.jpg \
        --tflite ../../assets/models/yunet_160/yunet_160.tflite
"""
import argparse
import json

import numpy as np
import tensorflow as tf
from PIL import Image

_STRIDES = (8, 16, 32)


def log(msg):
    print(f"[landmark_jitter_ab] {msg}")


# ── Faithful port of yunet_detector.dart / yunet_decoder.dart ──────────────

def resize_nearest(rgb, target_w, target_h):
    """Matches lib/src/image/image_converter.dart resizeNearest exactly:
    per-axis floor(index * scale), clamped -- no interpolation."""
    src_h, src_w = rgb.shape[:2]
    x_scale = src_w / target_w
    y_scale = src_h / target_h
    rows = np.clip((np.arange(target_h) * y_scale).astype(np.int64), 0, src_h - 1)
    cols = np.clip((np.arange(target_w) * x_scale).astype(np.int64), 0, src_w - 1)
    return rgb[rows][:, cols]


def resolve_stride_io(interpreter, input_size):
    """Matches yunet_detector.dart _resolveStrideIo: pick role/stride from
    each output tensor's own shape, not a hardcoded index order."""
    details = interpreter.get_output_details()
    by_stride = {s: {"scores": []} for s in _STRIDES}
    for d in details:
        _, n, c = d["shape"]
        stride = next(s for s in _STRIDES if n == (input_size // s) ** 2)
        if c == 4:
            by_stride[stride]["bbox"] = d["index"]
        elif c == 10:
            by_stride[stride]["kps"] = d["index"]
        elif c == 1:
            by_stride[stride]["scores"].append(d["index"])
        else:
            raise ValueError(f"unexpected channel count {c}")
    return by_stride


def decode_stride(outputs, stride, input_size, score_threshold):
    grid = input_size // stride
    cls = outputs["scores"][0].reshape(grid * grid)
    obj = outputs["scores"][1].reshape(grid * grid)
    bbox = outputs["bbox"].reshape(grid * grid, 4)
    kps = outputs["kps"].reshape(grid * grid, 10)

    score = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
    keep = np.where(score >= score_threshold)[0]
    if len(keep) == 0:
        return []

    rows, cols = np.divmod(keep, grid)  # row-major: flat = row*grid + col
    out = []
    max_coord = float(input_size)
    for idx, row, col in zip(keep, rows, cols):
        b = bbox[idx]
        cx, cy = (col + b[0]) * stride, (row + b[1]) * stride
        w, h = np.exp(b[2]) * stride, np.exp(b[3]) * stride
        box = np.clip([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 0, max_coord)
        k = kps[idx]
        pts = np.array([
            [np.clip((col + k[2 * p]) * stride, 0, max_coord),
             np.clip((row + k[2 * p + 1]) * stride, 0, max_coord)]
            for p in range(5)
        ])
        out.append({"score": float(score[idx]), "box": box, "pts": pts})
    return out


def nms(dets, iou_threshold, max_faces):
    dets = sorted(dets, key=lambda d: -d["score"])
    kept = []
    for d in dets:
        if len(kept) >= max_faces:
            break
        if all(_iou(d["box"], k["box"]) < iou_threshold for k in kept):
            kept.append(d)
    return kept


def _iou(a, b):
    inter_l, inter_t = max(a[0], b[0]), max(a[1], b[1])
    inter_r, inter_b = min(a[2], b[2]), min(a[3], b[3])
    inter_w, inter_h = max(0.0, inter_r - inter_l), max(0.0, inter_b - inter_t)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


class YuNetPy:
    """Same contract as YuNetDetector.detect: RGB image in, best face's
    (score, box, landmarks) out, in the *input* image's own pixel space."""

    def __init__(self, tflite_path, score_threshold=0.45, iou_threshold=0.3, max_faces=5):
        self.interp = tf.lite.Interpreter(model_path=tflite_path)
        self.interp.allocate_tensors()
        in_shape = self.interp.get_input_details()[0]["shape"]
        self.input_size = int(in_shape[1])
        assert in_shape[1] == in_shape[2], "expected square input"
        self.stride_io = resolve_stride_io(self.interp, self.input_size)
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_faces = max_faces
        self.input_index = self.interp.get_input_details()[0]["index"]

    def detect(self, rgb):
        h, w = rgb.shape[:2]
        resized = resize_nearest(rgb, self.input_size, self.input_size)
        bgr = resized[:, :, ::-1].astype(np.float32)  # swapToBgr, raw 0-255, no normalise
        tensor = bgr[np.newaxis, ...]
        self.interp.set_tensor(self.input_index, tensor)
        self.interp.invoke()

        candidates = []
        for stride, io in self.stride_io.items():
            outputs = {
                "bbox": self.interp.get_tensor(io["bbox"]),
                "kps": self.interp.get_tensor(io["kps"]),
                "scores": [self.interp.get_tensor(i) for i in io["scores"]],
            }
            candidates += decode_stride(outputs, stride, self.input_size, self.score_threshold)

        kept = nms(candidates, self.iou_threshold, self.max_faces)
        if not kept:
            return None
        best = max(kept, key=lambda d: d["score"])
        sx, sy = w / self.input_size, h / self.input_size
        return {
            "score": best["score"],
            "box": best["box"] * [sx, sy, sx, sy],
            "pts": best["pts"] * [sx, sy],
        }


# ── A/B harness ──────────────────────────────────────────────────────────

def reflect_index(idx, n):
    """Vectorised equivalent of np.pad(mode='reflect') index lookup: maps
    an arbitrarily out-of-[0, n-1] index (positive or negative) to its
    reflected in-bounds index, without ever materialising the padded
    array. Verified to match np.pad(mode='reflect') exactly."""
    period = 2 * (n - 1)
    idx = np.mod(idx, period)
    return np.where(idx >= n, period - idx, idx)


def crop_geometry(cx, cy, half_w, half_h):
    """Same rounding as the old crop(): the crop's top-left origin and
    size in the source photo's coordinate frame, [cx-half_w, cx+half_w] x
    [cy-half_h, cy+half_h] -- but as plain numbers, no pixels touched."""
    l, t = int(round(cx - half_w)), int(round(cy - half_h))
    r, b = int(round(cx + half_w)), int(round(cy + half_h))
    return l, t, r - l, b - t


def sample_noisy_input(rgb, l, t, crop_w, crop_h, target_size, rng, max_shift_px=2.5, noise_sigma=4.0):
    """Equivalent to the old crop() -> add_noise() -> resize_nearest(target_size)
    pipeline, but computed directly at the target_size x target_size output
    grid instead of materialising the full (possibly reflect-padded) crop
    first. resize_nearest only ever reads target_size^2 points out of the
    crop, and nearest-neighbour composes exactly (each output pixel is a
    single point-sample all the way back to the source photo), so this is
    the same computation, not an approximation -- just O(target_size^2)
    instead of O(crop_w * crop_h).

    This matters because crop_w/crop_h scale with margin^2 against a
    reflect-tiled *virtual* canvas: on a 2736x3648 photo, margin=12 asks
    for a 13828x21315 crop (~884MB uint8, ~14GB once add_noise's old
    float64 promotion kicked in) despite only 25600 of those pixels ever
    reaching the detector. That's what was OOM-killing this script before
    it could even print its first log line.

    Note: draws noise from a differently-shaped rng call than the old
    add_noise() (target_size^2 draws vs crop_w*crop_h draws), so runs are
    no longer bit-identical to pre-fix output for the same --seed -- the
    noise model itself (iid gaussian, sigma=noise_sigma, uniform sub-pixel
    shift in [-max_shift_px, max_shift_px]) is unchanged.
    """
    h, w = rgb.shape[:2]
    x_scale, y_scale = crop_w / target_size, crop_h / target_size
    local_rows = np.clip((np.arange(target_size) * y_scale).astype(np.int64), 0, crop_h - 1)
    local_cols = np.clip((np.arange(target_size) * x_scale).astype(np.int64), 0, crop_w - 1)

    dx, dy = rng.uniform(-max_shift_px, max_shift_px, size=2)
    shifted_rows = np.clip(local_rows + dy, 0, crop_h - 1).astype(np.int64)
    shifted_cols = np.clip(local_cols + dx, 0, crop_w - 1).astype(np.int64)

    global_rows = reflect_index(t + shifted_rows, h)
    global_cols = reflect_index(l + shifted_cols, w)
    base_pixels = rgb[global_rows][:, global_cols]

    noise = rng.normal(0, noise_sigma, size=base_pixels.shape)
    return np.clip(base_pixels.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def run_trials(detector, rgb, l, t, crop_w, crop_h, n_trials, seed):
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_trials):
        noisy = sample_noisy_input(rgb, l, t, crop_w, crop_h, detector.input_size, rng)
        det = detector.detect(noisy)
        if det is not None:
            results.append(det)
    return results


def summarize(label, dets, n_trials=None):
    if len(dets) < 2:
        log(f"{label}: too few successful detections ({len(dets)}) to measure jitter")
        return None
    pts = np.stack([d["pts"] for d in dets])  # (n, 5, 2)
    scores = np.array([d["score"] for d in dets])
    eye_dist = np.linalg.norm(pts[:, 0] - pts[:, 1], axis=1)
    mean_eye = eye_dist.mean()
    # per-point std-dev, normalised by this framing's own mean eye-distance
    # so different framings (different absolute face scales) are comparable.
    per_point_std = pts.std(axis=0)  # (5, 2)
    norm_jitter = per_point_std / mean_eye
    n_str = f"{len(dets)}/{n_trials}" if n_trials is not None else str(len(dets))
    log(
        f"{label}: n={n_str} detected, "
        f"score={scores.mean():.3f}+-{scores.std():.3f}, "
        f"eye_dist={mean_eye:.1f}px (cv={eye_dist.std()/mean_eye:.3f}), "
        f"mean landmark jitter/eye_dist={norm_jitter.mean():.4f}"
    )
    return {
        "n": len(dets),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "eye_dist_mean": float(mean_eye),
        "eye_dist_cv": float(eye_dist.std() / mean_eye),
        "landmark_jitter_over_eye_dist": float(norm_jitter.mean()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True)
    p.add_argument("--tflite", default="../../assets/models/yunet_160/yunet_160.tflite")
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--margins", default="12,8,5,3,1.8,1.1", help=(
        "comma-separated crop half-width multiples of the seed face bbox "
        "half-width, largest (face=small fraction of frame) first. Kept "
        ">=1.1 so no margin ever crops into the detected face itself -- "
        "that would confound 'small input resolution' with 'clipped face'."
    ))
    p.add_argument("--out", default="results_landmark_jitter_ab.json")
    args = p.parse_args()
    margins = [float(m) for m in args.margins.split(",")]
    if any(m < 1.0 for m in margins):
        raise SystemExit("--margins values must all be >= 1.0 to avoid clipping the face")

    detector = YuNetPy(args.tflite)

    pil = Image.open(args.image).convert("RGB")
    rgb = np.array(pil)
    log(f"loaded {args.image} {rgb.shape[1]}x{rgb.shape[0]}")

    first = detector.detect(rgb)
    if first is None:
        raise SystemExit("no face found on the full image at all -- pick a different photo")
    l, t, r, b = first["box"]
    cx, cy = (l + r) / 2, (t + b) / 2
    half_w, half_h = (r - l) / 2, (b - t) / 2
    log(f"seed detection: score={first['score']:.3f} box=({l:.0f},{t:.0f})-({r:.0f},{b:.0f})")

    result = {"seed_box_frac_of_source_width": float((r - l) / rgb.shape[1]), "points": []}
    for margin in margins:
        cl, ct, crop_w, crop_h = crop_geometry(cx, cy, half_w * margin, half_h * margin)
        face_frac_pct = 100 / margin
        label = f"margin={margin:g} (face ~{face_frac_pct:.0f}% of {crop_w}x{crop_h} crop width)"
        dets = run_trials(detector, rgb, cl, ct, crop_w, crop_h, args.n_trials, args.seed)
        summary = summarize(label, dets, n_trials=args.n_trials)
        if summary:
            summary["margin"] = margin
            summary["face_pct_of_crop_width"] = face_frac_pct
            result["points"].append(summary)

    if len(result["points"]) >= 2:
        widest = result["points"][0]  # margins sorted largest->smallest by convention
        tightest = result["points"][-1]
        ratio = widest["landmark_jitter_over_eye_dist"] / tightest["landmark_jitter_over_eye_dist"]
        log(f"widest framing (margin={widest['margin']:g}) jitter is {ratio:.2f}x "
            f"tightest framing (margin={tightest['margin']:g}) jitter")
        result["widest_over_tightest_jitter_ratio"] = float(ratio)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
