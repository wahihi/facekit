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

## example 앱 통합 (2026-06-27 추가)

`MediaPipeFaceLandmarker` + `BlinkLivenessDetector`를 `example/lib/main.dart`에 연결했다.
이제 카메라 프레임마다(등록/인식 버튼 상태와 무관하게) 항상 detect→landmark→liveness가
돌고, 결과가 `CameraPreview` 위 `FaceOverlayPainter`([example/lib/face_overlay.dart](../../example/lib/face_overlay.dart))로
그려진다:

- 얼굴 박스 + 좌우 눈 6점씩(EAR에 쓰는 점) 오버레이.
- 라이브니스 `pending`이면 노란 박스 + "눈을 깜빡여주세요", `passed`이면 초록 박스.
- **`enroll`/`identify` 둘 다 라이브니스 `passed`가 될 때까지 매칭을 미룬다** — 정지 사진을
  들이대면 깜빡임이 절대 발생하지 않으므로 매처(matcher)까지 도달하지 못한다. 이게 "사진
  대면 → 차단" 데모 시나리오의 실제 동작 경로다.
- 인식 성공 시 이름+유사도가 박스 라벨에도 표시된다(상태 텍스트와 별개로).

라이브러리 공개 API(`lib/facekit.dart`)에 `MediaPipeFaceLandmarker`/`BlinkLivenessDetector`
export를 추가했다 — 이전엔 둘 다 패키지 외부(example 포함)에서 import할 수 없었다.

### 좌표 변환 — 가장 까다로웠던 부분

`startImageStream`이 주는 `CameraImage`는 **원본 센서 방향**(보통 가로로 긴 1920×1080류)
그대로다. `CameraPreview` 위젯은 `controller.value.deviceOrientation` 기준으로 네이티브
프리뷰를 `RotatedBox`로 추가 보정해 보여준다(camera 0.11.4의 `CameraPreview._getQuarterTurns()`
로직, Android만 해당 — iOS 네이티브 프리뷰는 이미 올바른 방향). 따라서 박스/랜드마크 좌표를
그대로 그리면 디바이스를 기본 방향(`portraitUp`)이 아닌 자세로 들었을 때 어긋난다.

`face_overlay.dart`의 `mapImagePointToPreview()`가 camera 패키지와 **동일한 회전 테이블을
재구현**해 좌표를 변환한다(추측이 아니라 camera 패키지 소스에서 직접 옮긴 것). 전면 카메라는
좌우 미러링(셀카 컨벤션)도 적용한다.

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
  카메라/실기기가 없어 **시각적으로 확인하지 못했다**.
- **좌표 변환 로직도 같은 이유로 실기기 시각 검증을 못 했다.** `mapImagePointToPreview()`의
  회전/스케일 수식 자체는 단위 테스트(`example/test/face_overlay_test.dart`, 모서리 점 매핑
  11개 케이스)로 검증했고 camera 패키지 소스의 회전 테이블을 그대로 따랐지만, 전면 카메라
  미러링 방향이나 `deviceOrientation` 콜백이 실제 디바이스에서 기대대로 들어오는지는 카메라/
  실기기가 없어 확인하지 못했다. 다음 단계에서 실기기로 미러링 방향과 회전 보정을 눈으로
  확인하는 게 필요하다.
- 매 프레임 detect+landmark가 항상 돌게 바뀌어서(이전엔 인식/등록 중에만 detect) CPU 사용량이
  늘었다 — 모바일 실기기에서 프레임 드롭 여부는 미측정(`doc/KR/benchmark.md`도 실기기 측정이
  아님, 같은 한계).

## 테스트

- `test/core/math_test.dart` — `eyeAspectRatio` 순수 함수 (뜬눈/감은눈 합성 좌표)
- `test/image/image_converter_test.dart` — `cropFaceImage` 경계 clamp
- `test/landmark/face_landmarker_smoke_test.dart` — 실제 동봉 모델 로드 + 478점 출력 확인
  (BYOM 아니므로 항상 실행, graceful skip 없음)
- `test/liveness/blink_liveness_detector_test.dart` — 합성 EAR 시퀀스로 상태머신 검증
- `example/test/face_overlay_test.dart` — 좌표 회전/스케일/미러링 순수 함수 테스트(실기기 없이
  검증 가능한 유일한 부분)
