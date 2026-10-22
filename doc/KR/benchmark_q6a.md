🇺🇸 [English version](../EN/benchmark_q6a.md)

---

# Q6A(Radxa Dragon, QCS6490) 실측 벤치마크 — 2026-09-03

`packages/facekit_cli`의 `--bench` 옵션으로 실측. 입력은 `tool/model_verification/me_1.jpg`,
검출기 YuNet(`yunet_160`), 임베더 AuraFace(`auraface_r100_fp16`), 둘 다 CPU(XNNPACK), n=30
(워밍업 3회 별도). `libtensorflowlite_c.so`는 arm64로 직접 빌드(레시피는
[doc/KR/linux_port_scope.md](linux_port_scope.md) 참고).

## 결과

| 스레드 수 | 코어 고정 | detect | align | embed | **total** |
|---|---|---|---|---|---|
| 1 | `taskset -c 4-7` | 18.2ms | 1.1ms | 701.7ms | **721.2ms** |
| 4 | `taskset -c 4-7` (A78 4개) | 24.0ms | 1.1ms | 325.5ms | **349.8ms** |

(수치는 median. `lscpu -e`로 확인한 코어 매핑: CPU 0-3 = Cortex-A55, CPU 4-7 = Cortex-A78
— CPU7이 2.7GHz "프라임" 코어, 4-6은 2.4GHz.)

## 판단 — NPU 없이 간다

이전 NPU 논의에서 잡아둔 기준: 300ms대면 CPU만으로 제품 가능, 800ms 이상이면 NPU 필수.
싱글스레드 721ms는 그 애매한 구간에서 NPU 쪽에 가까웠지만, **AuraFace ResNet100 임베딩을
Q6A의 A78 코어 4개에 그냥 멀티스레드로 돌리는 것만으로 349.8ms까지 떨어졌다** — 소프트웨어
설정 하나 바꾼 것 외에 아무 투자 없이. 상용 기준으로 삼았던 "체감 지연 0.5초 이내"를 여유 있게
통과한다.

**결론: 지금 시점엔 QNN/INT8/ORT 스택에 투자할 근거가 없다.** 몇 주짜리 낯선 스택 두 개
(ONNX 변환 + QNN EP)를 실측 없이 먼저 붙이는 게 바로 그 "35번 패턴"이었는데, 이제 실측이
나왔고 결론은 "필요 없다"다. NPU 경로는 완전히 닫는 게 아니라 — 나중에 정말 필요해지면
(예: 여러 얼굴 동시 처리, 더 무거운 모델) 다시 판단하되, 그 근거도 실측으로 잡는다.

## 부가 관찰

- **detect는 멀티스레드에서 오히려 살짝 느려짐**(18.2ms → 24.0ms). YuNet 입력이 160×160으로
  작아서, 스레드 동기화 오버헤드가 병렬화 이득보다 큰 것으로 보임. 임베딩(701ms→325ms 개선)에
  비하면 무시할 수준이라 지금은 그냥 둠 — 나중에 검출기/임베더 스레드 수를 따로 튜닝하면 몇 ms
  더 아낄 여지는 있음(우선순위 낮음).
- `align`(AffineAligner, 순수 수학 연산)은 스레드 수와 무관하게 ~1.1ms — 예상대로 무시 가능한
  비용.

## 재현 방법

```bash
cd packages/facekit_cli
taskset -c 4-7 \
  env FACEKIT_TFLITE_C_LIBRARY=<arm64 libtensorflowlite_c.so 경로> \
  dart run bin/facekit_cli.dart \
  --image ../../tool/model_verification/me_1.jpg \
  --detector ../../assets/models/yunet_160 \
  --embedder ../../example/assets/models/auraface \
  --bench 30 --num-threads 4
```
