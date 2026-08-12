🇺🇸 [English version](../../EN/postmortem/2026-08-12-yunet-landmark-order.md)

---

# EER 48%가 나왔던 30분 — 같은 좌우 버그를 두 번째로 밟은 이야기

**날짜:** 2026-08-12
**계기:** "BlazeFace는 랜드마크 6점 중 입을 1점만 줘서 `AffineAligner`가 ArcFace의
진짜 5점 기준(입꼬리 2점 요구)을 4점으로 타협하고 있는데, 애초에 5점을 네이티브로
주는 detector로 바꾸면 이 타협이 필요 없어지지 않을까?"라는 질문에서 출발.
**결론(스포일러):** 맞는 방향이었고 실제로 크게 개선됐다(AuraFace clean EER
10.0%→2.5%). 다만 가는 길에 (1) 유력 후보 하나(SCRFD)가 라이선스 문제로 탈락했고,
(2) 남은 후보(YuNet)를 파이썬으로 검증하다가 **2026-08-03 포스트모템의 BlazeFace
눈 순서 버그와 정확히 같은 종류의 실수**를 또 저질러서 EER 48%(사실상 랜덤)를
봤다.

이 프로젝트는 [Claude Code](https://github.com/anthropics/claude-code)와
협업해서 개발했고, 이 디버깅도 마찬가지다.

## 0. 이 조사가 근거 없는 게 아니었던 이유

이미 같은 세션에서 `compare_with_alignment.py`(insightface SCRFD 사용)로
"정렬을 실제로 적용하면 EER이 어떻게 바뀌는가"를 측정해둔 상태였다 —
ArcFace 8.5%→3.0%, AuraFace 10.0%→6.5%. 즉 "detector를 바꾸면 개선된다"는
방향성 자체는 이미 데이터로 확인돼 있었고, 이번 조사는 "그럼 실제로 채택
가능한 detector가 있는가"를 좁히는 작업이었다.

## 1. 후보 조사 — SCRFD는 라이선스에서 탈락

SCRFD(InsightFace)와 YuNet(OpenCV Zoo/libfacedetection) 둘 다 5점 네이티브
출력을 지원한다. SCRFD의 `detection/scrfd/LICENSE`는 Apache 2.0인데, 이것만
보고 "가중치도 자유롭다"고 판단하면 안 됐다 — InsightFace 저장소 최상위
README가 코드와 모델 라이선스를 명시적으로 분리해서 이렇게 적어뒀다:

> "The code of InsightFace is released under the MIT License... **The
> training data containing the annotation (and the models trained with
> these data) are available for non-commercial research purposes only.**"

`buffalo_l`(이 프로젝트가 이미 연구용으로 취급 중인 모델)이 구체적 예시로
언급돼 있어서, SCRFD도 같은 포괄 조항에 걸린다고 봐야 했다. 독립적으로
확인해보니 SCRFD의 학습 데이터인 WIDER FACE 자체도 CC BY-NC-ND(비상업+2차
저작물 금지)였다 — 근거 두 갈래가 모두 "비상업"으로 수렴해서 SCRFD는
채택 불가로 확정했다.

YuNet은 배포처(OpenCV Zoo README)가 모델 자체를 명시적으로 MIT라고 적어뒀고,
`CLAUDE.md:15`에 "검출(BlazeFace Apache2.0 / YuNet MIT): 동봉 가능"이라고 이미
적혀 있었다 — 이 프로젝트가 처음부터 YuNet을 대안으로 염두에 두고 있었던
것으로 보인다.

## 2. 1차 검증 — EER 48~49%, 사실상 랜덤

`tool/model_verification/compare_with_yunet.py`를 새로 작성했다.
`cv2.FaceDetectorYN`(OpenCV 5.0.0에 내장)으로 YuNet 검출을 돌리고, 얻은 5점을
`insightface.utils.face_align.norm_crop()`(SCRFD 검증 때 썼던 것과 같은 함수,
`arcface_dst` 기준점도 동일)에 넣어 정렬한 뒤, 기존 임베더 클래스(ArcFace,
AuraFace)를 그대로 재사용해서 같은 LFW 200쌍을 돌렸다.

결과: `ArcFace genuine mean=0.7788 impostor mean=0.7485` — genuine과 impostor가
거의 구분이 안 됐다. EER 48%(clean), 49%(AuraFace clean)로, 이 세션에서 봤던
가장 나쁜 BlazeFace 조건보다도 훨씬 나빴다. **뭔가 근본적으로 잘못됐다는 신호였다.**

## 3. 디버깅 — 검출값은 멀쩡한데 크롭이 새까맣다

이론으로 추론하지 않고, 이전 포스트모템들의 교훈대로 **실제 데이터를 눈으로
봤다.**

1. **raw 검출값 확인**: 5장을 뽑아서 bbox, 5점 좌표, score를 직접 찍어봤다.
   전부 정상이었다 — score 0.92~0.95, 눈 사이 간격 37~45px, 눈-코-입 배치가
   기하학적으로 말이 됐다.
2. **크롭 시각화**: 정렬된 112×112 크롭을 실제로 PNG로 만들어 봤다 — **작은
   얼굴이 새까만 배경에 둘러싸여** 있었다. 숫자만 봐서는 몰랐던 게 이미지로
   보자마자 바로 드러났다(7/24, 8/3 포스트모템에서도 똑같이 "실제 이미지를
   보는 게 결정적이었다"는 교훈이 나왔었는데, 세 번째로 재확인).
3. **변환행렬 직접 계산**: 추정된 스케일이 0.125였다 — 눈 간격만으로 어림잡은
   기대값(~0.83~0.91)의 약 1/7. **insightface의 `norm_crop`과, 이전 답변에서
   이미 파이썬으로 포팅해뒀던 facekit 자체 Umeyama 구현 양쪽에 같은 5점을
   넣어 독립적으로 재계산**했는데, 둘 다 똑같이 0.125가 나왔다 — 라이브러리
   버그가 아니라 **입력 좌표 자체(좌우 대응관계)가 잘못됐다**는 뜻이었다.

## 4. 원인 확정 — 반대로 매핑해보니 즉시 정상화

YuNet 출력 순서는 `right_eye, left_eye, nose, right_mouth, left_mouth`이고,
처음엔 이걸 ArcFace 컨벤션(`left_eye, right_eye, nose, left_mouth,
right_mouth`)에 맞춰 **재배열해서** 넣었다 — "이름이 다르니 순서를 맞춰야
한다"는 상식적인 판단이었다. 그런데 이게 틀렸다:

```python
# 재배열함 (leftEye를 dst[0]에 매핑) — 틀림
scale, rot = 0.1256, 4.81°

# YuNet 원본 순서 그대로(재배열 없이) — 맞음
scale, rot = 0.9061, 1.46°
```

**YuNet 제작자가 붙인 `right_eye`/`left_eye` 이름표 자체가, facekit이 가정한
좌우 규약과 정반대**였던 것이다 — 2026-08-03 포스트모템에서 이미 한 번
겪었던 **BlazeFace 눈 순서 버그**와 정확히 같은 실수를, 이번엔 "당연히
이름대로 맞춰야지"라는 반대 방향의 성급한 판단으로 다시 저질렀다.

## 5. 수정 후 재측정 — SCRFD보다도 좋은 결과

| 방법 | ArcFace EER(clean/degraded) | AuraFace EER(clean/degraded) |
|---|---|---|
| 리사이즈만(정렬 미적용) | 8.5% / 25.0% | 10.0% / 28.0% |
| SCRFD 정렬(라이선스 문제로 폐기) | 3.0% / 5.0% | 6.5% / 8.0% |
| **YuNet 정렬(수정 후)** | **2.5% / 4.0%** | **2.5% / 6.0%** |

SCRFD보다 좋고, **ArcFace와 AuraFace의 clean EER이 정확히 2.5%로 같아졌다** —
이 세션 내내 반복됐던 "AuraFace가 셋 중 제일 약하다"는 패턴이, 모델 자체의
한계가 아니라 **BlazeFace의 4점 타협에 AuraFace가 유독 더 민감했던 것**일
가능성을 강하게 시사한다.

## 6. 정리 — 인과관계

```
질문: BlazeFace 4점 타협 대신 5점 네이티브 detector를 쓰면?
   └→ 이미 확보된 SCRFD 데이터로 방향성 확인(EER 개선 확인됨)
   └→ SCRFD 자체는 라이선스 조사로 배제(InsightFace 정책 + WIDER FACE 데이터셋 둘 다 비상업)
   └→ YuNet 파이썬 검증 1차: EER 48% (완전 붕괴)
        └→ raw 검출값 확인: 정상
        └→ 크롭 시각화: 작고 새까만 배경 → 뭔가 잘못됐다는 첫 시각적 증거
        └→ 변환행렬 직접 계산(2개 독립 구현으로 대조): 스케일 1/7 붕괴 확인
        └→ 좌우 반대 매핑 테스트: 즉시 정상화(스케일 0.906, 회전 1.46°)
   └→ 재측정: ArcFace 2.5%, AuraFace 2.5% — SCRFD보다 좋고 두 모델이 동률
```

## 7. 배운 것

- **"모델 제작자가 붙인 랜드마크 이름을 검증 없이 믿지 말 것"이 이번에
  두 번째로 확인됐다.** BlazeFace 때는 "당연히 다르겠지"라고 의심 없이
  가정한 게 문제였고, 이번엔 반대로 "이름이 다르니 당연히 맞춰 재배열해야
  한다"는 성급한 "상식적 수정"이 문제였다 — 결국 양쪽 다 **실제 좌표를
  찍어서 검증하지 않고 이름표만 믿은 게** 공통 원인이다.
- **숫자가 이상하면 이미지로 눈으로 확인하는 게 여전히 가장 빠른 지름길이다.**
  스케일 0.125라는 숫자만 봐서는 "뭐가 문제인지" 바로 안 잡히는데, 크롭
  PNG를 열어보는 순간 "작고 새까맣다"는 게 바로 보였다. 세 번째 포스트모템
  전부 같은 결론에 도달했다.
- **독립적으로 두 번 구현해서 대조하는 습관이 이번에도 유효했다.**
  insightface의 `norm_crop`과 facekit 자체 Umeyama를 각각 파이썬으로 돌려서
  같은 결과(0.125)가 나오는 걸 확인했기 때문에, "제3자 라이브러리 버그"라는
  잘못된 방향으로 시간을 낭비하지 않고 바로 "입력 데이터 문제"로 좁힐 수
  있었다.
- **BlazeFace로 이미 한 번 겪은 실수 유형(랜드마크 이름/순서 오검증)은
  detector를 바꿔도 그대로 반복될 수 있다.** 다음에 이걸 실제 Dart로
  이식할 땐, 이번에 확정한 좌우 매핑을 코드 주석뿐 아니라 회귀 테스트로도
  못박아둬야 한다.

## 부록: 남겨둔 도구

- `tool/model_verification/compare_with_yunet.py` — YuNet(`cv2.FaceDetectorYN`)
  기반 정렬로 LFW EER을 재측정하는 스크립트. `compare_with_alignment.py`(SCRFD
  버전)와 같은 임베더 클래스를 재사용해서, 검출기 하나만 바뀐 결과를 직접
  비교할 수 있다.
- `tool/model_verification/yunet.onnx` — OpenCV Zoo에서 받은 원본 YuNet 모델
  (`face_detection_yunet_2023mar.onnx`).
