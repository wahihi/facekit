🇺🇸 [English version](../EN/adaface_verification.md)

---

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

## AuraFace EER 측정 (2026-08-09 추가)

기본 임베딩 모델인 AuraFace(glintr100)는 그동안 동일인 쌍 유사도만 확인됐고(7/24, 8/3
포스트모템 참고) 타인 쌍은 측정된 적이 없었다 — `matching.threshold`도 실측 없는
placeholder(0.40)였다. 실기기에서 등록한 사람과 무관한 사람의 사진을 비췄을 때 거의
같은 유사도(둘 다 0.9대)가 나오는 문제가 실제로 재현됐고, 원인을 추적한 결과
`manifest.json`의 `input.normalize`가 잘못된 값(정규화 없이 원본 0~255 픽셀 그대로)으로
설정돼 있었던 것으로 드러났다. 공식 `insightface` 파이썬 패키지로 같은 `glintr100.onnx`를
로드해보면(`tool/model_verification/verify_auraface_official.py`), `ArcFaceONNX`가 그래프를
보고 자동으로 판별하는 정규화 값은 `input_mean=127.5, input_std=127.5`(표준 ArcFace
컨벤션)였다 — raw pixel이 아니었다. 잘못된 raw 값이 처음 채택된 건 그 값을 정한 시점
(7/24)에 정렬 파이프라인에 아직 안 고쳐진 버그(눈 좌우 순서 반전, SVD 부호 오류 등,
8/3에야 수정)가 있어서 입력 크롭 자체가 잘못 회전돼 있었기 때문으로 보인다 — 그 잘못된
크롭에 표준 정규화를 걸었더니 낮은 유사도가 나온 것을 정규화 문제로 오진단했고, 8/3에
정렬 버그를 고친 뒤에도 아무도 정규화 설정을 재검증하지 않아 raw 값이 그대로 남아있었다.

`input.normalize`를 표준값(`mean`/`std` 127.5)으로 고친 뒤, ArcFace/AdaFace와 동일한
방법론(`compare_arcface_adaface.py --auraface-tflite`, 같은 LFW 200쌍, seed=42, 5점
정렬 미적용 리사이즈만)으로 재측정했다:

| 모델 | 조건 | genuine 평균 | impostor 평균 | EER | EER 임계값 | 해당 임계값 정확도 |
|---|---|---|---|---|---|---|
| AuraFace | clean | 0.4803 | 0.1952 | 10.0% | 0.300 | 90.0% |
| AuraFace | degraded | 0.3560 | 0.2402 | 28.0% | 0.274 | (clean 임계값 적용 시 68.5%) |

ArcFace(EER 8.5%)와 비슷한 수준이고 AdaFace(EER 2.0%)보다는 뚜렷이 낮다 — AuraFace가
셋 중 가장 약한 판별력을 보였다. 실기기 즉석 테스트(본인 사진 2장 + 타인 사진 1장,
`verify_auraface_official.py`)에서는 genuine 0.77 vs impostor ≈0.03으로 극적으로
갈렸었는데, 그건 표본 3장짜리 우연한 결과였고 200쌍 통계가 훨씬 신뢰할 수 있는
수치다. `matching.threshold`를 0.40(무근거 placeholder)에서 0.30(EER 기준
실측값)으로 갱신했다.

**교훈**: threshold뿐 아니라 `input.normalize` 같은 "정답이 하나뿐인 것처럼 보이는"
설정값도, 그 값을 처음 정했을 때 다른 버그(이번엔 정렬)가 같이 섞여 있었다면 틀린
채로 굳어질 수 있다. 값 하나를 재검증할 땐 그 값이 결정된 시점에 다른 전제(여기선
"크롭이 똑바로 정렬돼 있다")가 참이었는지도 같이 의심해야 한다.

## 정렬 포함 재측정 및 실기기 기반 threshold 조정 (2026-08-11 추가)

위 AuraFace EER(10.0%)은 ArcFace/AdaFace와 같은 방법론 — **250x250 LFW funneled
crop을 112x112로 리사이즈만 하고 5점 정렬은 적용 안 함**(모든 `threshold_note`에
명시) — 으로 측정한 값이다. facekit의 실제 파이프라인은 검출+5점 정렬을 거치므로,
정렬을 적용하면 수치가 달라지는지 확인했다.

facekit 자체의 BlazeFace 디코더+Umeyama 솔버를 파이썬으로 이식하는 대신(작업량이
커서 별도 과제로 미룸), 공식 `insightface` 패키지의 SCRFD 검출기 +
`face_align.norm_crop()`을 재사용했다(`tool/model_verification/compare_with_alignment.py`).
`norm_crop`의 기준점(`arcface_dst`)이 facekit의 `arcface112Ref`와 소수점까지
동일해서, "같은 목표 기하, 다른(하지만 표준적인) 검출기+솔버"로 측정하는 셈이다.

| 모델 | 조건 | genuine 평균 | impostor 평균 | EER | EER 임계값 | 해당 임계값 정확도 |
|---|---|---|---|---|---|---|
| ArcFace | clean (정렬 적용) | 0.6550 | 0.0108 | 3.0% | 0.156 | 97.0% |
| ArcFace | degraded (정렬 적용) | 0.5234 | 0.0186 | 5.0% | 0.127 | (clean 임계값 적용 시 96.0%) |
| AuraFace | clean (정렬 적용) | 0.5926 | 0.0577 | 6.5% | 0.137 | 93.5% |
| AuraFace | degraded (정렬 적용) | 0.4205 | 0.1009 | 8.0% | 0.215 | (clean 임계값 적용 시 82.0%) |

정렬 미적용 대비 두 모델 다 큰 폭으로 개선됐다(ArcFace clean EER 8.5%→3.0%,
degraded 25%→5%; AuraFace clean 10.0%→6.5%, degraded 28%→8%) — "정렬 미적용 수치는
실제 파이프라인보다 비관적"이라는 위 "한계" 섹션의 예상이 그대로 확인됐다. 다만
**상대 순위는 그대로다** — AuraFace는 정렬을 적용해도 ArcFace보다 여전히 약하다
(clean EER 기준 약 2.2배 나쁨). 즉 정렬 미적용이 AuraFace만 불리하게 만든 게
아니라, 두 모델에 거의 비례해서 영향을 준 것으로 보인다.

**매니페스트 threshold는 이 EER 임계값(0.137)으로 바로 교체하지 않았다.** SCRFD는
facekit이 실제로 쓰는 BlazeFace-short 검출기보다 랜드마크가 안정적인 것으로
보이는데, BlazeFace는 실기기에서 프레임의 28~42%가 회전 이상으로 플래그될 만큼
불안정하다는 게 이미 확인돼 있다(`doc/KR/postmortem/2026-08-03-affine-aligner-alignment.md`).
그래서 이 0.137은 facekit 실사용 조건보다 낙관적인 하한일 가능성이 있다. 대신
**실기기에서 실제로 관찰된 값**을 근거로 삼았다: `input.normalize` 수정 후 실기기
테스트에서 본인 유사도 11회 중 3회(0.285/0.288/0.291)가 기존 threshold(0.30) 바로
밑에서 거부됐고, 같은 세션의 타인(사진) 4회는 0.060~0.094였다. `matching.threshold`를
**0.30 → 0.25**로 낮춰서 그 3회를 전부 통과시키면서도 실측 타인 최댓값(0.094)과
0.156의 여유를 유지하도록 조정했다.

## threshold 미세조정을 포기하고 포즈 게이트로 전환 (2026-08-11 계속)

0.25로 낮춘 뒤 실기기에서 두 번 더 검증했다(임포스터 대상: 일론 머스크 73회,
젠슨 황 60회). 표본이 늘수록 genuine/impostor 간 실측 간격이 계속 좁아지는
추세가 나타났다:

| 표본 크기 | genuine/impostor 간격 |
|---|---|
| 4개 | 0.156 |
| 77개 | 0.036 |
| 137개 (누적) | 0.022 |

threshold 0.25/0.27은 매번 그 시점 데이터에선 "통과"했지만(예: 0.27은 impostor
최댓값 0.249와 0.021 차이, genuine 최솟값 0.271과는 겨우 0.001 차이), 표본을
늘릴 때마다 여백이 계속 줄어드는 패턴 자체가 "이 값이 안전하다"가 아니라
"아직 최악의 경우를 다 못 만났다"는 신호였다. threshold를 실기기 관찰값에 맞춰
계속 미세조정하는 방식은 밑빠진 독에 물 붓기라고 판단했다.

**근본 원인으로 다시 초점을 옮겼다.** genuine 유사도가 프레임마다 크게
흔들리는 것(0.27~0.59) 자체가, 위에서 이미 확인된 BlazeFace 랜드마크
불안정성(회전 이상 24~42%)의 직접적인 결과로 보였다. 그래서 threshold를 계속
낮추는 대신, **정렬 품질이 나쁜 프레임을 아예 매칭에서 제외하는 포즈 게이트를
`AffineAligner`에 추가했다**(`lib/src/alignment/affine_aligner.dart`) — 피팅된
변환 행렬의 회전각이 ±45°를 넘거나 스케일이 `[0.05, 3.0]` 범위를 벗어나면
`align()`이 `AlignedFace` 대신 `null`을 반환하고, 파이프라인은 이를 "얼굴 미검출"과
동일하게 취급한다(경계값은 `tool/analyze_alignment_log.py`가 이미 쓰던 것과
동일 — 같은 저장소, 같은 실기기 조사에서 독립적으로 도출된 값). 순수 함수
`_poseWithinBounds`로 분리해 `poseWithinBoundsForTest`로 단위 테스트했고
(`test/alignment/affine_aligner_test.dart`), `FaceAligner.align()`의 반환
타입이 `AlignedFace?`로 바뀌어 `FacePipeline`/`SyncFacePipeline`/벤치마크
전부 null 케이스를 처리하도록 갱신했다.

포즈 게이트가 근본 원인(정렬 불안정)을 직접 다루므로, `matching.threshold`는
**0.30(LFW 200쌍 EER 기준값)으로 다시 고정**했다 — 실기기 관찰값 기반 미세조정은
여기서 멈춘다. 포즈 게이트를 실기기에서 재검증하는 건 다음 과제로 남긴다.

## 포즈 게이트 실기기 검증 + 역광 발견 (2026-08-11 계속)

포즈 게이트를 켠 상태로 실기기 재검증(본인 vs 마이크 타이슨, `FACEKIT_VERBOSE_DEBUG=true`)을
했다:

```
정렬 시도 107회 중  통과 48회(45%) / 거부 59회(55%)
거부 사유: 59건 전부 회전 위반(스케일 위반은 0건), 회전각 45.5°~153.4°
통과분 결과 — 본인 6회 전부 승인(0.318~0.430), 마이크 타이슨 40회 전부 거부(0.078~0.270)
간격(gap) = 0.048  (게이트 없이 측정했던 0.022보다 개선)
```

게이트는 설계대로 정확히 작동했다(회전 위반만 걸러냄, 스케일 오탐 없음) — 하지만
**정렬 시도의 55%가 그냥 버려진다**는 건 실사용 체감상 "화면에 얼굴이 뻔히 보이는데
얼굴 없음으로 뜨는" 경우가 절반 넘는다는 뜻이라 우려스러운 수치였다.

**원인을 찾다가 역광이라는 환경 변수를 발견했다.** 테스트 당시 광원이 피험자 머리
뒤쪽에 있었는데, 이 상태에서 카메라 자동노출이 밝은 배경에 맞춰지면서 얼굴이
상대적으로 노출부족(실루엣에 가까움) 상태가 됐고, 이게 랜드마크 검출 정밀도를
떨어뜨려 회전 오차를 키우는 것으로 추정된다 — 직접 광원을 몸으로 가려서 얼굴
노출을 정상화하자 체감 인식률이 확실히 올라가는 것을 실기기에서 확인했다(정량
로그는 아직 못 남김). 즉 지금까지 "BlazeFace 랜드마크가 원래 불안정하다"고 다뤄온
것 중 상당 부분이 순수 알고리즘 한계가 아니라 **촬영 환경(역광) 문제**로 설명될
가능성이 있다 — 위 55%라는 거부율도 역광이라는 최악 조건에서 나온 수치일 수
있다는 뜻이다.

**결론**: 포즈 게이트는 채택한다(회전 오탐을 정확히 걸러내는 것 자체는
검증됐고, 통과분의 genuine/impostor 간격도 실제로 개선됐다). 회전 임계값(45°)을
더 조정할지는 역광을 피한 조건에서 재측정한 뒤 판단하는 게 맞다고 보고, 정량
재측정과 함께 설치 문서에 "역광을 피하라"는 촬영 가이드를 추가하는 걸
후속 과제로 남긴다.
