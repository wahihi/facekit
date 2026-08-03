"""Parses a `flutter run --dart-define=FACEKIT_VERBOSE_DEBUG=true` device log
and checks the alignment pipeline's own diagnostic output for the failure
signatures found during the 2026-08 AffineAligner investigation (see
doc/KR/postmortem/): a wrong camera-rotation quarter-turn, a spurious
40-180 degree rotation from a landmark-correspondence mismatch, or an
implausible scale from a too-small/too-large sampled crop window.

This is a one-off developer tool, not part of the shipped SDK — it only
reads `[FacePipeline]`/`[AffineAligner]`/`[TEMP DEBUG]` debugPrint lines that
only exist when `kFacekitVerboseDebug` is on (see lib/src/core/debug_flags.dart).

Usage:
    cd example
    stdbuf -oL -eL flutter run --dart-define=FACEKIT_VERBOSE_DEBUG=true 2>&1 \\
        | tee ../auraface_conv/log
    # after enrolling/identifying a face and quitting with q or Ctrl+C:
    python3 ../tool/analyze_alignment_log.py ../auraface_conv/log

Optional: pass --crops-out DIR to also reconstruct the aligned-face crop
dumps (enroll/identify) as PNGs (requires `pip install pillow`).
"""
import argparse
import base64
import math
import re
import sys

QUARTER_TURNS_RE = re.compile(
    r'_cameraQuarterTurns: lensDirection=(\S+) sensorOrientation=(-?\d+) '
    r'degrees=(-?\d+) turns=(-?\d+)'
)
LANDMARKS_RE = re.compile(
    r'TEMP DEBUG LANDMARKS (\w+) imageSize=(\d+)x(\d+) '
    r'bbox=\(([-\d.]+),([-\d.]+)\)-\(([-\d.]+),([-\d.]+)\) '
    r'score=([\d.]+) pts=(.+)'
)
POINT_RE = re.compile(r'\d+:\(([-\d.]+),([-\d.]+)\)')
MATRIX_RE = re.compile(
    r'DEBUG MATRIX #(\d+) a=(-?[\d.]+) b=(-?[\d.]+) tx=(-?[\d.]+) '
    r'c=(-?[\d.]+) d=(-?[\d.]+) ty=(-?[\d.]+)'
)
CHUNK_RE = re.compile(r'TEMP DEBUG CHUNK (\w+) (\d+)/(\d+) (\S+)')

# Heuristic thresholds — see doc/KR/postmortem/2026-07-24-auraface-alignment.md
# and the 2026-08 follow-up investigation for what "normal" looked like on a
# real Pixel 7 (rotation ~7-24 deg for a hand-held selfie, scale ~0.3-0.44).
MAX_SANE_ROTATION_DEG = 45.0
MIN_SANE_SCALE = 0.05
MAX_SANE_SCALE = 3.0
EYE_AXIS_RATIO = 2.0  # dy must be at least this much smaller than dx


def parse_log(path):
    quarter_turns = None
    detections = []  # list of dicts: label, image_size, pts, matrix
    pending_landmarks = None

    with open(path, 'r', errors='ignore') as f:
        for line in f:
            m = QUARTER_TURNS_RE.search(line)
            if m:
                quarter_turns = dict(
                    lens=m.group(1), sensor_orientation=int(m.group(2)),
                    degrees=int(m.group(3)), turns=int(m.group(4)),
                )
                continue

            m = LANDMARKS_RE.search(line)
            if m:
                pts = [(float(x), float(y)) for x, y in POINT_RE.findall(m.group(9))]
                pending_landmarks = dict(
                    label=m.group(1), width=int(m.group(2)), height=int(m.group(3)),
                    score=float(m.group(8)), pts=pts,
                )
                continue

            m = MATRIX_RE.search(line)
            if m and pending_landmarks is not None:
                a, b, tx, c, d, ty = (float(m.group(i)) for i in (2, 3, 4, 5, 6, 7))
                pending_landmarks['matrix'] = dict(a=a, b=b, tx=tx, c=c, d=d, ty=ty)
                detections.append(pending_landmarks)
                pending_landmarks = None

    return quarter_turns, detections


def analyze(quarter_turns, detections):
    issues = []

    if quarter_turns is None:
        issues.append(
            "WARN: no '_cameraQuarterTurns' debug line found — was the app "
            "enrolled/identified at least once, and is the temp debug print "
            "from example/lib/main.dart still present?"
        )
    else:
        print(f"camera: lens={quarter_turns['lens']} "
              f"sensorOrientation={quarter_turns['sensor_orientation']} "
              f"-> turns={quarter_turns['turns']} "
              f"({quarter_turns['degrees']}°)")

    if not detections:
        issues.append(
            "WARN: no LANDMARKS+DEBUG MATRIX pairs found — nothing to check. "
            "Did you enroll/identify a face before quitting flutter run?"
        )
        return issues

    print(f"\n{len(detections)} detection(s):")
    for i, det in enumerate(detections, 1):
        m = det['matrix']
        a, c = m['a'], m['c']
        angle = math.degrees(math.atan2(c, a))
        scale = math.hypot(a, c)

        row = (f"  #{i} [{det['label']}] rotation={angle:7.1f}°  scale={scale:.4f}")

        det_issues = []
        if len(det['pts']) >= 2:
            (x0, y0), (x1, y1) = det['pts'][0], det['pts'][1]
            dx, dy = abs(x0 - x1), abs(y0 - y1)
            if dy > 0 and dx < dy / EYE_AXIS_RATIO:
                det_issues.append(
                    f"eye pair separated more vertically (dx={dx:.1f}, dy={dy:.1f}) "
                    "than horizontally — looks like a 90°-class camera "
                    "rotation bug, not a normal face"
                )
        if abs(m['a'] - m['d']) > 1e-3:
            det_issues.append(
                f"a ({m['a']:.4f}) != d ({m['d']:.4f}) — matrix isn't a pure "
                "rotation+uniform-scale; check for a reflection or SVD bug"
            )
        if abs(angle) > MAX_SANE_ROTATION_DEG:
            det_issues.append(
                f"|rotation|={abs(angle):.1f}° exceeds the "
                f"{MAX_SANE_ROTATION_DEG}° sanity bound for a hand-held "
                "selfie — likely a landmark-order or camera-rotation bug"
            )
        if not (MIN_SANE_SCALE <= scale <= MAX_SANE_SCALE):
            det_issues.append(
                f"scale={scale:.4f} is outside the [{MIN_SANE_SCALE}, "
                f"{MAX_SANE_SCALE}] sanity range — the sampled crop window "
                "is probably far too small/large"
            )

        if det_issues:
            print(row + "  [FLAGGED]")
            for msg in det_issues:
                print(f"      - {msg}")
            issues.extend(f"#{i} [{det['label']}]: {msg}" for msg in det_issues)
        else:
            print(row + "  [ok]")

    return issues


def decode_crops(log_path, out_prefix):
    try:
        from PIL import Image
    except ImportError:
        print("\n(skipping crop reconstruction: `pip install pillow` to enable)")
        return

    chunks = {}
    totals = {}
    with open(log_path, 'r', errors='ignore') as f:
        for line in f:
            m = CHUNK_RE.search(line)
            if not m:
                continue
            label, idx, total, b64 = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
            chunks.setdefault(label, {})[idx] = b64
            totals[label] = total

    if not chunks:
        return

    print()
    for label, parts in chunks.items():
        total = totals[label]
        missing = [i for i in range(1, total + 1) if i not in parts]
        if missing:
            print(f"[{label}] WARNING: missing {len(missing)}/{total} chunks, skipping")
            continue
        raw = base64.b64decode(''.join(parts[i] for i in range(1, total + 1)))
        side = int(round(math.sqrt(len(raw) / 3)))
        if side * side * 3 != len(raw):
            print(f"[{label}] {len(raw)} bytes isn't a square RGB image, skipping")
            continue
        out_path = f"{out_prefix}_{label}.png"
        Image.frombytes('RGB', (side, side), raw).save(out_path)
        print(f"[{label}] crop reconstructed -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log_path', help='path to the saved flutter run log')
    parser.add_argument(
        '--crops-out', default=None,
        help='if set, also reconstruct aligned-face crop dumps as '
             '<CROPS_OUT>_enroll.png / <CROPS_OUT>_identify.png',
    )
    args = parser.parse_args()

    quarter_turns, detections = parse_log(args.log_path)
    issues = analyze(quarter_turns, detections)

    if args.crops_out:
        decode_crops(args.log_path, args.crops_out)

    print()
    if issues:
        print(f"RESULT: {len(issues)} issue(s) flagged.")
        sys.exit(1)
    else:
        print("RESULT: all detections look sane.")
        sys.exit(0)


if __name__ == '__main__':
    main()
