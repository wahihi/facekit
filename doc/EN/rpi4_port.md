🇰🇷 [한국어 원문](../KR/rpi4_port.md)

---

# Raspberry Pi 4 port (`feature/rpi4`) — 2026-10-22

Branched from `feature/linux-port` (Q6A, Radxa Dragon QCS6490). The goal was to run the same
headless real-time recognition (`facekit_cli`, `facekit_realtime`) on a Raspberry Pi 4 with 64-bit
Raspberry Pi OS and a Logitech C270 (the same USB camera used on the Q6A), and to decide from
measurements whether AuraFace can be used as is.

No library or CLI code changes (the Pi 4 is aarch64 too). Only the build/run environment differs,
and it is pinned down by two scripts in `tool/rpi4/`: `setup.sh` (environment setup) and
`bench.sh` (measurement).

## How to run

```bash
git clone --branch feature/rpi4 https://github.com/wahihi/facekit.git && cd facekit
./tool/rpi4/setup.sh          # once. The TFLite C build takes ~1 hour. Re-runs skip finished steps
source tool/rpi4/env.sh
./tool/rpi4/bench.sh          # golden vector + bench (1/4 threads) + 30s live camera
```

## Measured results (2026-10-22)

Environment: Raspberry Pi 4 Model B Rev 1.5, 3.7GB RAM, 4× Cortex-A72 @1.8GHz (governor ondemand),
Debian 13 (trixie), kernel 6.18.50+rpt-rpi-v8, Dart 3.13.6, TFLite C v2.21.0-rc0 (built on the Pi),
YuNet160 + AuraFace R100 fp16, XNNPACK CPU, n=30 median.

| Item | Pi 4 | Q6A (reference) |
|---|---|---|
| Golden vector (cosine ≥ 0.999 vs. Android) | **PASS** | PASS |
| 1 thread: detect / align / embed / total | 84.8 / 3.3 / 3089.8 / **3182.8ms** | 18.2 / 1.1 / 701.7 / 721.2ms |
| 4 threads: detect / align / embed / total | 81.9 / 3.2 / 1169.8 / **1254.3ms** | 24.0 / 1.1 / 325.5 / 349.8ms |
| Live camera 30s (C270, 640×480 YUYV, 4 threads) | 21 frames processed, 21 matched, 0 rejected, 0 corrupt | — |
| Memory growth (30s) | +0.6MB | — |
| Temperature / throttling | 53.6°C / 0x0 | — |

`dropped_frames` (638) counts camera frames skipped while a frame was being processed — normal by
design (only the latest frame is processed).

## Decision

- **Functionality: pass.** Golden vector matches, live-camera recognition works, no memory leak.
- **Speed: AuraFace R100 is not suitable for real-time use on the Pi 4.** 1254ms at 4 threads
  (≈0.8fps) exceeds the Q6A bar (300ms range = viable product / 800ms or more = another path
  needed). About 3.6× slower than the Q6A.
- **The Pi 4 port ends here.** The Pi 4 was never a business target (RK3588-class is), so no
  lightweight embedder is added just for the Pi 4.
- Lightweight embedders such as SFace (128D, Apache 2.0) will be evaluated separately as part of
  low-end device work (Galaxy A, RK3588), reusing the Pi 4 as the slowest reference board.

## Issues hit this time

- `tool/model_verification/me_1.jpg` is gitignored, so it is missing from a fresh clone and the
  bench/golden test failed with "image file not found". The same file is tracked as
  `example/assets/golden/me_1.jpg`, so `setup.sh` now copies it when missing.
- GitHub rejects password authentication for git push from the Pi, so an SSH key (ed25519) was
  set up.

## Remaining verification

The same two items left open at the Q6A port completion (`linux_port_scope.md`, 2026-10-21) are
also unverified on the Pi 4.

1. **Rejecting other people and no-face frames**: this 30-second run matched the enrolled person
   in all 21 frames. Whether a different person yields `accepted=false`, and whether no
   identification is produced when no face is present, was not checked.
2. **Clean shutdown on SIGINT (Ctrl+C)**: runs were ended only via `--seconds`; the Ctrl+C path
   was not checked on the device.

These don't affect the decision to end the Pi 4 port, so they remain as follow-up verification
shared by all Linux targets, together with the Q6A.
