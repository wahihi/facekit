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

## 한계

- 이 머신은 모바일 기기가 아니다 — 절대 ms 수치를 모바일 SoC 성능으로 해석하지 말 것.
- n=50, 단일 입력(랜덤 텐서) 반복 — 콜드스타트(모델 로드 시간)는 제외했고, 배치=1만 측정.
- BlazeFace는 더미 텐서 1장 기준이며 실제 카메라 프레임의 디코딩·리사이즈·NMS 비용은
  포함하지 않음(Dart 쪽 `yuv420ToFaceImage`/`resizeNearest` 등 전처리 비용 별도).
