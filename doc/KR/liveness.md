# 기본 라이브니스: 깜빡임 검출 (2026-06-26)

facekit Free 티어에 추가된 최소 라이브니스 — "정지된 사진을 들이대면 통과 안 된다"는
가장 기본적인 방어선이다. `LivenessDetector` 인터페이스([lib/src/core/contracts.dart](../../lib/src/core/contracts.dart))는
Free/Pro 공통이며, 이번에 Free용 구현체 `BlinkLivenessDetector`를 추가했다.

## 왜 새 모델이 필요했나

기존 검출기(BlazeFace)는 눈을 **점 1개**로만 준다(`DetectedFace.landmarks`의 6-keypoint 중 하나).
EAR(Eye Aspect Ratio) 기반 깜빡임 검출에는 눈꺼풀 위/아래 윤곽 좌표가 필요한데, 점 1개로는
원천적으로 계산할 수 없다. 그래서 검출 다음 단계로 **MediaPipe Face Landmarker(478점 페이스
메시)** 를 새로 추가했다.

- 출처: `face_landmarker.task`(공식 MediaPipe 배포본, Apache 2.0)을 `unzip`해서 그 안의
  `face_landmarks_detector.tflite`만 추출 — 같은 압축 안에 들어있는 얼굴 검출기·블렌드셰입
  모델은 이미 BlazeFace로 검출을 하고 있고 블렌드셰입은 안 쓰므로 제외했다.
- BlazeFace와 같은 프로젝트·같은 라이선스(Apache 2.0, 상업적 사용 가능)라서 **BYOM 없이
  동봉** — `assets/models/face_landmark_478/`, `pubspec.yaml`의 `flutter.assets`에 등록.
- 입력 256×256 RGB, `(pixel-0)/255` 정규화. 출력은 478×3(x,y,z) 랜드마크 텐서 1개 +
  얼굴 신뢰도 스칼라(미사용, AdaFace 때 만든 `zeroTensor`로 무시).

## 파이프라인

```
DetectedFace (BlazeFace box)
  → MediaPipeFaceLandmarker.detectLandmarks()   [lib/src/landmark/face_landmarker.dart]
      박스를 마진 25% 포함 정사각형으로 크롭 → 256×256 리사이즈 → 추론
      → 478점을 크롭 좌표계에서 원본 이미지 좌표계로 역변환
  → FaceLandmarks (478개 Point3D)
  → BlinkLivenessDetector.update(landmarks, timestampMs)  [lib/src/liveness/blink_liveness_detector.dart]
      좌/우 눈 6점씩 뽑아 eyeAspectRatio() 계산(core/math.dart) → 평균 EAR
      → 임계값(기본 0.2) 아래로 떨어졌다가 1초 내 다시 올라오면 "깜빡임 1회"
      → 관찰 윈도우(기본 4초) 안에 1회 이상 깜빡이면 LivenessState.passed
```

`eyeAspectRatio()`는 Soukupová & Čech(2016)의 공개 공식이고, 눈 6점 인덱스(478점 토폴로지
중 우안 `[33,159,158,133,153,145]`, 좌안 `[362,380,374,263,386,385]`)는 공개된 표준 인덱싱이다
([Pushtogithub23/Eye-Blink-Detection-using-MediaPipe-and-OpenCV](https://github.com/Pushtogithub23/Eye-Blink-Detection-using-MediaPipe-and-OpenCV) 등 여러 공개 구현이 동일하게 사용).

## 임계값 — 이번엔 실측이 아니라 "공개 권장값 1단계"

ArcFace/AdaFace 때는 LFW 200쌍으로 직접 EER을 측정해 임계값을 새로 도출했다
([adaface_verification.md](adaface_verification.md)). 이번 EAR 임계값(0.2)은 **그 수준의 실측을
못 했다** — 실제 깜빡이는 얼굴 영상 표본이 이 환경에 없기 때문이다. Soukupová & Čech 논문이
제시하는 공개 권장값을 시작점으로 채택했을 뿐이다. CLAUDE.md 6항("임계값은 새로 튜닝")은
이전 직장 코드 같은 비공개 출처의 값을 그대로 들고 오는 걸 금지하는 것이고, 이건 공개 논문의
상수이므로 위반은 아니지만 — **검증되지 않은 값이라는 한계는 명확히 남는다.** 실기기로
실제 깜빡임 영상을 찍어서 재측정하는 게 다음 단계다.

## 한계 (정직하게)

- **정지 사진 방어는 된다, 그 이상은 아니다.** 평평한 사진은 EAR이 절대 변하지 않으므로
  통과 못 한다. 하지만:
  - 눈 부분에 구멍을 뚫은 사진/마스크는 방어 못 한다(실제 눈이 비치면 EAR 자체가 진짜로 변함).
  - 깜빡이는 얼굴을 녹화한 동영상 재생 공격은 방어 못 한다(영상 속 EAR도 실제로 변함).
  - 이런 공격에 대한 방어는 Pro 티어의 정교한 멀티신호 라이브니스 몫으로 남겨둔다(같은
    `LivenessDetector` 인터페이스 뒤에 더 강한 구현체를 꽂는 구조).
- 크롭 마진(25%), 최대 깜빡임 지속시간(1000ms), 관찰 윈도우(4초) 모두 임의로 정한 시작값 —
  실측 데이터 없음.
- `face_landmarker_smoke_test.dart`는 합성 회색 이미지로 모델이 크래시 없이 478점을 뱉는지만
  확인한다. 실제 얼굴 사진으로 랜드마크 좌표가 진짜 눈/입 위치에 맞게 찍히는지는 — 이 환경에
  카메라/실기기가 없어 **시각적으로 확인하지 못했다** (UI 오버레이 작업과 같은 한계).
- `FacePipeline`이나 example 앱에는 아직 연결하지 않았다 — 이번 범위는 엔진(랜드마커+EAR+
  상태머신)까지. 카메라 프레임으로 실제 동작시키는 건 UI 오버레이 작업과 함께 묶어서
  다음 단계로 진행하는 게 자연스럽다.

## 테스트

- `test/core/math_test.dart` — `eyeAspectRatio` 순수 함수 (뜬눈/감은눈 합성 좌표)
- `test/image/image_converter_test.dart` — `cropFaceImage` 경계 clamp
- `test/landmark/face_landmarker_smoke_test.dart` — 실제 동봉 모델 로드 + 478점 출력 확인
  (BYOM 아니므로 항상 실행, graceful skip 없음)
- `test/liveness/blink_liveness_detector_test.dart` — 합성 EAR 시퀀스로 상태머신 검증
