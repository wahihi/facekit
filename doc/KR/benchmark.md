# 벤치마크 (2026-06-26)

facekit 파이프라인의 추론 속도·모델 크기·정확도를 측정한 기록.
정확도(EER) 측정 방법론과 원자료는 [adaface_verification.md](adaface_verification.md),
[tool/model_verification/](../../tool/model_verification/)를 참고.

## 측정 환경

이 머신(Linux VM, CPU 2코어, RAM 27GB)에서 TFLite Python 인터프리터(XNNPACK 위임,
싱글스레드)로 측정했다. **모바일 실기기(Pixel 7 등) 수치가 아니다** — NNAPI/GPU
delegate나 모바일 SoC의 실제 처리량과는 다르며, 이 표는 모델 간 상대 비교와
"동작은 한다"는 사실 확인용이다. 실기기 수치는 별도로 측정해 갱신해야 한다.

## 모델 크기 (.tflite, float32)

| 모델 | 역할 | 입력 | 출력 | 크기 |
|---|---|---|---|---|
| BlazeFace short-range | 검출 | 128×128 RGB | 896 anchor × (bbox+16, score) | 0.2 MB |
| ArcFace (buffalo_l/w600k_r50) | 임베딩 | 112×112 RGB | 512-d | 166 MB |
| AdaFace (IR-101/WebFace12M) | 임베딩 | 112×112 BGR | 512-d (+ norm 출력 1개) | 249 MB |

## 추론 시간 (단일 추론, n=50, warmup 5회 제외)

| 모델 | 평균 | p50 | p95 |
|---|---|---|---|
| BlazeFace short-range (검출) | 0.87 ms | 0.81 ms | 1.23 ms |
| ArcFace w600k_r50 (임베딩) | 109.3 ms | 108.5 ms | 115.1 ms |
| AdaFace IR-101/WebFace12M (임베딩) | 203.6 ms | 202.1 ms | 215.5 ms |

검출은 가벼운 모델이라 거의 비용이 없고, 임베딩(특히 AdaFace의 IR-101 백본)이
파이프라인 지연시간의 대부분을 차지한다. AdaFace가 ArcFace(R100)보다 느린 건
백본 자체가 더 깊기 때문 — [adaface_verification.md](adaface_verification.md)의
정확도 이득과 맞바꾸는 트레이드오프다.

## 정확도 (LFW 200쌍, EER 기준 — 상세는 adaface_verification.md)

| 모델 | clean EER | 저화질 EER | 비고 |
|---|---|---|---|
| ArcFace w600k_r50 | 8.5% | 25.0% | 임계값이 화질에 따라 0.26→0.33으로 크게 이동 |
| AdaFace IR-101/WebFace12M | 2.0% | 14.0% | 임계값이 0.21→0.22로 거의 안 움직임 (고정 임계값에 유리) |

## 한계 (이 섹션, VM 측정)

- 이 머신은 모바일 기기가 아니다 — 절대 ms 수치를 모바일 SoC 성능으로 해석하지 말 것.
- n=50, 단일 입력(랜덤 텐서) 반복 — 콜드스타트(모델 로드 시간)는 제외했고, 배치=1만 측정.
- BlazeFace는 더미 텐서 1장 기준이며 실제 카메라 프레임의 디코딩·리사이즈·NMS 비용은
  포함하지 않음(Dart 쪽 `yuv420ToFaceImage`/`resizeNearest` 등 전처리 비용 별도).

## 실기기 (Pixel 7, 2026-06-27)

example 앱(`example/lib/benchmark.dart`)에 내장한 벤치마크 버튼으로 측정. **`flutter build apk
--profile`** 빌드로 측정했다 — `--debug`는 비최적화 JIT라 Dart 쪽 전후처리(앵커 디코드/NMS/이미지
리사이즈) 비용이 부풀려져 신뢰할 수 없고(실측 결과 debug와 거의 동일하게 나와 오히려 이게
"JIT 오버헤드가 원인이 아니다"를 보여줌), `--release`는 `arcface_buffalo_l`이
`redistributable:false`라 `assertLoadable()`이 로드를 막아 실행조차 안 된다. profile은
`kReleaseMode==false`라 가드에 안 걸리면서 AOT 최적화는 release와 동일해 측정에 적합하다.

측정 대상은 카메라 실시간 프레임(검출 → `AffineAligner` 정렬 → 임베딩), 임베딩은
`FacePipeline`이 실제로 쓰는 것과 동일하게 `Isolate.run()`을 통해 매회 새 isolate에서 실행했다
(아이솔레이트 스폰 비용까지 포함한 게 실제 프레임당 비용에 더 가깝다는 판단). n=30, 워밍업 5회
제외, 카메라에 잡힌 동일 프레임 1장을 고정해 반복.

| 모드 | 검출(BlazeFace) | 임베딩(ArcFace buffalo_l) | 전체 1프레임 |
|---|---|---|---|
| CPU (기본값, 가속 미사용) | 평균 65.1ms / p50 66.4ms / p95 79.0ms | 평균 729.4ms / p50 736.2ms / p95 846.7ms | 평균 795.6ms / p50 798.5ms / p95 920.3ms |
| NNAPI (`useNnApiForAndroid=true`) | 평균 76.6ms / p50 76.3ms / p95 101.3ms | 평균 876.2ms / p50 890.5ms / p95 959.5ms | 평균 954.4ms / p50 965.4ms / p95 1065.1ms |

**NNAPI가 CPU보다 오히려 느렸다.** 두 모델 다 float32 그래프인데, NNAPI는 보통 int8 양자화
모델을 가속하는 데 최적화돼 있어 float32는 Pixel 7의 NNAPI 벤더 드라이버에서 가속을 못 받고
CPU로 폴백하면서 **NNAPI 디스패치/IPC 오버헤드만 추가로 얹히는** 경우가 흔하다. BlazeFace처럼
원래도 작은 모델(0.2MB)일수록 그 오버헤드 비중이 더 크게 드러난다(검출도 65→77ms로 느려짐).
이 결과는 `TfliteRunner`의 `useNnApi` 기본값을 `false`로 둔 현재 코드의 선택이 맞았음을
실측으로 확인해준 것이기도 하다 — 가속을 끄는 게 아니라 "이 모델 조합에서는 가속이 손해"라는
근거가 생긴 것.

VM 수치(위 섹션) 대비 실기기 CPU 임베딩(729ms)이 VM(109ms)보다 6배 느린 것도 일관된 원인이다 —
VM 쪽은 Python TFLite가 XNNPACK 위임을 켠 채 측정했지만, 이 SDK의 `TfliteRunner`는
`InterpreterOptions`를 전혀 설정하지 않아(`tflite_runner.dart`) XNNPACK도, 멀티스레드도 켜지지
않은 기본 CPU 레퍼런스 커널로만 돈다. 즉 현재 실기기 수치는 "가속 전무" 상태의 정직한 하한선이며,
XNNPACK 위임을 켜는 건 NNAPI와 별개로 시도해볼 만한 다음 최적화 후보다.

### 한계 (실기기 섹션)

- Pixel 7 1대만 측정. 갤럭시 A(저가형) 수치는 아직 없음 — 측정되면 이 표에 행을 추가할 것.
- n=30, 카메라로 잡은 임의의 한 프레임을 반복 입력으로 고정 — 조명/각도가 다른 여러 프레임에
  대한 분산은 측정하지 않음.
- 검출 시간에는 Dart 쪽 전처리(리사이즈)·후처리(앵커 디코드+NMS)가 포함돼 있어, 네이티브 추론
  자체보다 이 쪽이 비용을 더 많이 차지할 가능성이 있다(코드 검토 결과 `prepareInputTensor`/
  `zeroTensor`가 boxed `List<List<...>>>`를 매 프레임 새로 할당 — `Float32List` 등 typed
  buffer로 바꾸면 줄어들 여지가 있으나 별도 작업으로 남겨둠).
