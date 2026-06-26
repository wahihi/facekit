# AdaFace 실모델 검증 (2026-06-25)

ArcFace(`arcface_buffalo_l`/w600k_r50)와 AdaFace(`adaface_ir101_webface12m`)
실가중치로 임베딩을 뽑아 동일인/타인 쌍 코사인 유사도를 비교하고, 그 결과로
두 매니페스트의 `matching.threshold`를 갱신했다. 스크립트는
[tool/model_verification/compare_arcface_adaface.py](../../tool/model_verification/compare_arcface_adaface.py).

## 방법

- **데이터셋**: [huggingface.co/datasets/logasja/lfw](https://huggingface.co/datasets/logasja/lfw)
  `pairs/test` split (LFW deepfunneled, 250×250). 동일인 100쌍 + 타인 100쌍 (seed=42)을
  무작위 샘플.
- **전처리**: 250×250 → 112×112 리사이즈만 적용 (5점 랜드마크 정렬 미적용 — LFW funneled
  crop은 이미 대략 중앙정렬되어 있으나, 실제 SDK의 `AffineAligner` 출력과는 다르다).
- **저화질 조건**: 112×112 → 24×24 다운샘플 → 112×112 업샘플로 저해상도/원거리 촬영을
  모사 (단일한 proxy이며, 블러·압축·포즈 변화 등은 포함하지 않음).
- **모델 실행**: ArcFace는 `example/assets/models/arcface_buffalo_l/w600k_r50.tflite`를
  TFLite Interpreter로, AdaFace는 체크포인트→ONNX 변환본을 onnxruntime으로 직접 실행
  (TFLite 변환은 이 머신에서 OOM으로 실패 — 아래 "TFLite 변환 미해결" 참고). 두 모델 모두
  실제 공개 가중치이며, 가중치 파일 자체는 라이선스상 리포에 동봉하지 않음(BYOM).
- **AdaFace 출력**: ONNX 그래프가 `feature`(L2-정규화된 512차원)와 `norm`(정규화 전
  L2 norm 스칼라, AdaFace 학습 시 적응적 마진에 쓰이는 품질 신호) 두 개를 출력한다.
  매칭에는 `feature`만 사용한다. **이 두-출력 구조 때문에 Dart SDK의 `TfliteFaceEmbedder.embed()`가
  단일 출력만 가정하던 부분이 실제로 깨지는 것을 확인**했고, 별도로 수정했다 (아래 참고).
- **임계값**: 코사인 유사도 ROC에서 FPR=FNR이 되는 EER 지점을 임계값으로 채택.

## 결과

| 모델 | 조건 | genuine 평균 | impostor 평균 | EER | EER 임계값 | 해당 임계값 정확도 |
|---|---|---|---|---|---|---|
| ArcFace | clean | 0.4428 | 0.1134 | 8.5% | 0.263 | 91.5% |
| ArcFace | degraded | 0.4392 | 0.2542 | 25.0% | 0.333 | (clean 임계값 적용 시 70.0%) |
| AdaFace | clean | 0.4617 | 0.0527 | 2.0% | 0.211 | 98.0% |
| AdaFace | degraded | 0.3938 | 0.1145 | 14.0% | 0.223 | (clean 임계값 적용 시 83.5%) |

**핵심 발견**: 이번 소규모 측정에서 AdaFace가 ArcFace보다 (1) clean 조건에서도 더 낮은
EER(2.0% vs 8.5%)을 보였고, (2) 저화질 조건에서 EER 악화 폭이 더 작았으며(8.5%→14.0%,
+5.5pt vs ArcFace의 8.5%→25.0%, +16.5pt), (3) clean 기준 임계값을 그대로 저화질에 적용했을 때
정확도 하락도 더 적었다(91.5%→83.5%, -8pt vs ArcFace의 91.5%→70.0%, -21.5pt). 특히
**최적 임계값 자체가 화질에 따라 거의 안 움직인다는 점**(AdaFace 0.211→0.223, +0.012 vs
ArcFace 0.263→0.333, +0.070)은 고정 임계값 하나로 다양한 캡처 품질에 대응해야 하는
모바일 앱 시나리오에서 실질적으로 중요하다.

## 한계 (정밀 논문급 측정 아님)

- 표본 100+100쌍으로 작음 — 신뢰구간이 넓다.
- 5점 랜드마크 정렬을 적용하지 않은 채(리사이즈만) 측정 — 절대 수치는 실제 파이프라인보다
  낮게 나올 가능성이 있다. 단, 두 모델에 동일 조건을 적용했으므로 상대 비교는 유효하다고 본다.
- "저화질"은 단일 proxy(24px 다운/업샘플)일 뿐, 실제 저조도·블러·압축·측면 포즈 등은
  다루지 않았다.
- AdaFace는 `.onnx`로 실행했고, 실제 SDK가 쓰는 `.tflite` 포맷으로는 아직 끝까지
  검증하지 못했다(아래).

## TFLite 변환 완료 및 e2e 검증 (2026-06-26 추가)

더 큰 RAM(27GB)의 머신에서 `adaface_ir101_webface12m.onnx` → `.tflite` 변환(onnx2tf)을
재시도해 성공했다 (이전 머신의 OOM은 RAM 부족이 원인이었음이 확인됨; 변환 자체는
onnx2tf/numpy/tf_keras 버전 호환성 이슈 몇 개를 패치하면 가벼운 작업이었다). ArcFace도
동일 머신에서 `w600k_r50.onnx`(InsightFace buffalo_l 공개 미러)를 `.tflite`로 새로
변환했다.

변환된 `.tflite`를 인터프리터로 직접 열어 확인한 결과, AdaFace 그래프는 예상대로
**출력 2개**(`feature` [1,512], `norm` [1,1])를 가졌다 — `TfliteFaceEmbedder.embed()`의
다중 출력 처리 수정이 실제로 필요했음을 다시 한번 확인.

**Dart SDK e2e 검증**: `test/embedding/face_embedder_smoke_test.dart`의 ArcFace/AdaFace
테스트와 `test/pipeline/face_pipeline_smoke_test.dart`를 실제 `.tflite` 파일로 실행해
모두 통과했다(`TfliteFaceEmbedder.fromFile` → 어댑터 선택 → 추론 → 512차원 L2-정규화
임베딩까지 전 과정). 이전까지는 어댑터 로직을 Python(onnxruntime)으로 재구현해 간접
검증한 것이었는데, 이번엔 **Dart 어댑터 코드 자체가 실가중치로 크래시 없이 동작**함을
직접 증명했다. (테스트 파일의 모델 경로는 특정 개발자 계정에 하드코딩되어 있던 것을
`Platform.environment['HOME']` 기준으로 일반화했다.)

**ONNX ↔ TFLite 결과 일치 확인**: `compare_arcface_adaface.py`에 `--adaface-tflite` 옵션을
추가해 동일 LFW 200쌍에 대해 `.tflite`로 재측정한 결과(`results_tflite_2026-06-26.json`),
AdaFace의 EER/임계값이 `.onnx` 기준 측정치(`results_2026-06-25.json`)와 5~6번째
소수점 수준의 차이(float32 연산 경로 차이로 인한 잡음)만 보이며 거의 완전히 일치했다
(clean EER 2.0%/임계값 0.2106 vs 0.2106, degraded EER 14.0%/임계값 0.2234 vs 0.2234).
즉 위 "결과" 섹션의 결론과 매니페스트 threshold 값은 실제 배포 포맷(`.tflite`)에서도
그대로 유효하다.

## Dart SDK 수정: 다중 출력 임베더 버그

`TfliteFaceEmbedder.embed()`가 출력 텐서 1개만 가정하고 있었는데(`_runner.run(input, output)`),
AdaFace처럼 출력이 2개(`feature`, `norm`)인 모델을 실제로 로드하면 tflite_flutter의
`runForMultipleInputs`가 채워지지 않은 두 번째 출력 슬롯에서 null 단언 실패로 무조건
크래시한다. `lib/src/inference/tflite_runner.dart`에 `zeroTensor()` 헬퍼를 추가하고
`lib/src/embedding/face_embedder.dart`의 `embed()`가 출력 0번(임베딩)만 사용하고 나머지
출력엔 빈 버퍼를 채워 무시하도록 고쳤다. 단위 테스트:
`test/inference/tflite_runner_test.dart`.
