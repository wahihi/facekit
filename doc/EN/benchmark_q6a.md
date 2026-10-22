🇰🇷 [한국어 원문](../KR/benchmark_q6a.md)

---

# Q6A (Radxa Dragon, QCS6490) real-device benchmark — 2026-09-03

Measured with the `--bench` option of `packages/facekit_cli`. Input is
`tool/model_verification/me_1.jpg`, detector YuNet (`yunet_160`), embedder AuraFace
(`auraface_r100_fp16`), both on CPU (XNNPACK), n=30 (plus 3 separate warmup runs).
`libtensorflowlite_c.so` was built for arm64 on the board (recipe in
[linux_port_scope.md](linux_port_scope.md)).

## Results

| Threads | Core pinning | detect | align | embed | **total** |
|---|---|---|---|---|---|
| 1 | `taskset -c 4-7` | 18.2ms | 1.1ms | 701.7ms | **721.2ms** |
| 4 | `taskset -c 4-7` (4× A78) | 24.0ms | 1.1ms | 325.5ms | **349.8ms** |

(Values are medians. Core mapping per `lscpu -e`: CPU 0-3 = Cortex-A55, CPU 4-7 = Cortex-A78
— CPU7 is the 2.7GHz "prime" core, 4-6 run at 2.4GHz.)

## Decision — no NPU

The bar set during the earlier NPU discussion: in the 300ms range a CPU-only product is viable;
at 800ms or more the NPU becomes mandatory. Single-threaded 721ms sat in the gray zone, closer to
the NPU side, but **simply running the AuraFace ResNet100 embedding multi-threaded on the Q6A's
four A78 cores brought it down to 349.8ms** — no investment beyond changing one software setting.
That comfortably clears the commercial target of "perceived latency under 0.5s".

**Conclusion: there is no case for investing in a QNN/INT8/ORT stack right now.** Committing to
two unfamiliar stacks (ONNX conversion + QNN EP) for weeks before measuring anything was exactly
the trap to avoid; now the measurement is in, and the answer is "not needed". The NPU path isn't
closed for good — if it ever becomes genuinely necessary (e.g. several faces at once, a heavier
model), it gets re-evaluated, and that decision will also be grounded in measurements.

## Side observations

- **detect gets slightly slower with more threads** (18.2ms → 24.0ms). YuNet's input is a small
  160×160, so thread synchronization overhead appears to outweigh the parallel gain. Negligible
  next to the embedding improvement (701ms → 325ms), so it's left as is for now — tuning detector
  and embedder thread counts separately could save a few more ms later (low priority).
- `align` (AffineAligner, pure math) is ~1.1ms regardless of thread count — negligible, as
  expected.

## How to reproduce

```bash
cd packages/facekit_cli
taskset -c 4-7 \
  env FACEKIT_TFLITE_C_LIBRARY=<path to arm64 libtensorflowlite_c.so> \
  dart run bin/facekit_cli.dart \
  --image ../../tool/model_verification/me_1.jpg \
  --detector ../../assets/models/yunet_160 \
  --embedder ../../example/assets/models/auraface \
  --bench 30 --num-threads 4
```
