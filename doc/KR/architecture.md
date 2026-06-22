# facekit 아키텍처 및 인식 파이프라인

facekit은 Flutter용 클린룸(clean-room) 온디바이스 얼굴인식 SDK입니다. 이 문서는
코드베이스가 어떻게 구성돼 있고, 카메라 원본 프레임이 "이 사람은 누구다"라는
답으로 바뀌기까지 정확히 어떤 단계를 거치는지 설명합니다.

## 1. 설계 원칙

- **클린룸 구현.** 모든 모듈의 소스 주석에는 근거가 된 공개 모델/논문/스펙을
  명시합니다(BlazeFace 논문, MediaPipe, ArcFace 정렬 기준 등). 독점·상용 코드는
  참조하지 않습니다.
- **순수한 core.** `lib/src/core/`는 `package:flutter/*`나 `dart:ui`를 절대
  import하지 않습니다 — Flutter 없이도 실행·테스트 가능한 순수 Dart입니다.
- **구현보다 인터페이스.** 파이프라인의 모든 단계는 추상 인터페이스
  (`FaceDetector`, `FaceAligner`, `FaceEmbedder`, `FaceMatcher`)로 정의됩니다.
  BlazeFace·ArcFace 등 구체 구현은 이 인터페이스 뒤에서 교체 가능하고,
  테스트에서는 이 자리에 가짜(stub) 구현을 주입합니다.
- **BYOM 라이선스 모델.** 검출(BlazeFace, Apache 2.0)은 패키지에 동봉됩니다.
  반면 임베딩 모델은 동봉하지 않습니다 — 연구용으로만 라이선스된 가중치라서,
  개발자가 직접 가져와서 끼우는 구조(BYOM, Bring Your Own Model)입니다. 이를
  통해 SDK 자체는 비상업 라이선스 리스크에서 자유롭습니다.

## 2. 계층 구조

의존성은 단방향으로만 흐릅니다 — 하위 계층은 절대 상위 계층을 import하지
않습니다.

```
UI / example 앱
        │
        ▼
   pipeline/            (오케스트레이션, isolate 디스패치)
        │
        ▼
 detection · alignment · embedding · matching   (관심사별 Dart 파일)
        │
        ▼
   inference/           (TFLite 연동, manifest 파싱)
        │
        ▼
     core/              (순수 데이터 모델 + 수학 + 인터페이스 — Flutter 의존 0)
```

`core/`는 그 아래에 아무것도 두지 않습니다 — `dart:typed_data`, `dart:math`
외에는 어떤 것에도 의존하지 않습니다.

## 3. 핵심 데이터 모델 (`lib/src/core/models.dart`)

| 타입 | 역할 |
|---|---|
| `FaceImage` | 디코딩된 RGB888 이미지(카메라 프레임이든 갤러리 사진이든 출처 무관) |
| `Rect`, `Point` | 픽셀 좌표계 기하 기본형 |
| `DetectedFace` | 검출 결과 하나: bounding box + 6개 랜드마크 + score, **입력 `FaceImage`의 픽셀 좌표 기준** |
| `AlignedFace` | 정사각형(112×112 또는 160×160)으로 정렬·크롭된 RGB 패치 — 임베더 입력 직전 형태 |
| `Embedding` | L2 정규화된 `Float32List` 벡터 |
| `Enrollment` | 갤러리 항목 하나: `id` + `Embedding` + 메타데이터 |
| `MatchResult` | `matchedId`(nullable) + `similarity` + `accepted` |

모든 수학 함수(`lib/src/core/math.dart`의 `l2Normalize`, `cosineSimilarity`,
`l2Norm`)는 부수효과 없는 순수 함수입니다 — I/O 없음, 상태 변경 없음, Flutter
의존 없음.

## 4. 인식 파이프라인, 단계별로

`FacePipeline`(`lib/src/pipeline/face_pipeline.dart`)이 오케스트레이터입니다.
`enroll()`과 `identify()` 둘 다 동일한 4단계 체인을 거치고, `identify()`만
마지막에 갤러리 매칭 단계가 추가됩니다.

```
FaceImage
   │  (1) 검출(DETECT) — BlazeFaceDetector
   ▼
DetectedFace  (bbox + 6 랜드마크, 픽셀 좌표)
   │  (2) 정렬(ALIGN) — AffineAligner
   ▼
AlignedFace  (112×112 또는 160×160 RGB, 정면화된 얼굴)
   │  (3) 임베딩(EMBED) — TfliteFaceEmbedder (Isolate 내부에서 실행)
   ▼
Embedding  (512차원, L2 정규화됨)
   │  (4) 매칭(MATCH) — CosineMatcher          [identify()에서만]
   ▼
MatchResult  (matchedId, similarity, accepted)
```

### 1단계 — 검출 (`lib/src/detection/`)

`BlazeFaceDetector`는 구글의 **BlazeFace short-range** 모델(Apache 2.0,
`assets/models/blazeface_short/`에 동봉됨)을 감쌉니다:

1. 입력 `FaceImage`를 128×128로 리사이즈하고 `[-1, 1]`로 정규화합니다.
2. TFLite 추론으로 raw 텐서 두 개가 나옵니다: `[1, 896, 16]` regressor(박스 +
   6개 키포인트, 각 2값)와 `[1, 896, 1]` classificator 점수.
3. `blazeface_anchors.dart`가 896개 SSD 앵커 박스를 한 번만 생성해 캐싱하고,
   `blazeface_decoder.dart`가 그 앵커들을 기준으로 raw 텐서를 디코딩하며,
   점수에 sigmoid를 적용하고 `scoreThreshold`로 걸러낸 뒤 NMS(`iouThreshold`)
   를 돌립니다 — 이 전부는 앵커 좌표계인 **정규화된 `[0,1]` 좌표**로
   이뤄집니다.
4. 이후 `denormalizeDetections()`가 모든 박스/랜드마크를 *원본* 이미지의
   width/height 기준으로 스케일링합니다. 그래서 호출자에게 반환되는
   `DetectedFace` 목록은 128×128 모델 입력이 아니라 **입력 `FaceImage`와 같은
   픽셀 좌표계**를 갖게 됩니다. 이 변환 단계가 있기 때문에 다음 단계인
   정렬기는 검출기 내부 해상도를 전혀 몰라도 됩니다.

### 2단계 — 정렬 (`lib/src/alignment/affine_aligner.dart`)

얼굴은 임의의 크기·회전·위치로 찍힙니다. `AffineAligner`는 **5점 유사변환**
(Umeyama의 1991년 폐형(closed-form) 최소제곱법 — 회전 + 등방 스케일 +
이동만, 전단(shear) 없음)으로 이를 보정합니다:

1. BlazeFace의 6개 랜드마크 중 5개(좌안, 우안, 코, 입, 그리고 두 귀 점의
   중점으로 다섯 번째 점을 대체)를 정형화된 기준 좌표 — 예를 들어 ArcFace
   112×112 표준 기준 좌표인 `arcface112Ref` — 와 매칭합니다.
2. 2×2 유사변환 행렬을 해석적(analytic) 2×2 SVD로 폐형 계산합니다(2×2
   한정이라 외부 선형대수 라이브러리가 필요 없습니다).
3. 역변환을 통해 이미지를 이중선형(bilinear) 보간으로 재샘플링해서 정사각형
   `AlignedFace`를 만듭니다(ArcFace/AdaFace/MobileFaceNet은 112×112, FaceNet은
   160×160).

임베딩 모델 계열마다 기대하는 기준 좌표가 다르기 때문에, `AffineAligner`는
기준점+출력크기로 파라미터화됩니다(`AffineAligner.arcface112()`,
`AffineAligner.facenet160()`).

### 3단계 — 임베딩 (`lib/src/embedding/`)

이 단계만 의도적으로 고정된 단일 모델이 아닙니다 — manifest와 어댑터로
구동되도록 만들어서, 파이프라인 코드를 건드리지 않고도 새 임베딩 계열을 추가할
수 있습니다:

- **`ModelManifest`**(`lib/src/inference/model_manifest.dart`)는 모델의 입출력
  계약에 대한 단일 진실 공급원입니다: 입력 크기/색상/정규화 방식, 출력 차원,
  정렬 기준, 매칭 임계값, 라이선스 메타데이터까지 담습니다. 순수 Dart이고,
  `.tflite` 파일 옆의 `manifest.json`에서 파싱됩니다.
- **`EmbedderAdapter`**는 계열별 전/후처리 전략입니다:
  - `ArcfaceAdapter` — `arcface`, `adaface`, `mobilefacenet` 세 계열을
    처리하며, 셋 다 112×112 입력과 manifest에서 그대로 가져온 고정
    `(pixel - mean) / std` 정규화를 공유합니다.
  - `FacenetAdapter` — FaceNet은 명시적인 "예외"입니다: 160×160 입력이고,
    고정 상수 대신 **이미지마다 자기 자신의 평균/표준편차로 prewhiten**합니다
    (`(x - mean(x)) / max(std(x), 1/sqrt(N))`, davidsandberg/facenet의 표준
    전처리 방식). Flutter 의존이 전혀 없는 순수 Dart라 `dart test`로 바로
    단위 테스트가 가능합니다.
- **`TfliteFaceEmbedder`**가 이를 하나로 묶습니다: `.tflite`+manifest를
  로드하고, `adapterForFamily(manifest.family)`로 어댑터를 선택하고,
  `FaceEmbedder` 인터페이스(`embed(AlignedFace) → Future<Embedding>`)를
  구현합니다.
- `manifest.output.l2Normalize`가 true일 때마다 후처리에서 `l2Normalize`를
  적용합니다 — 이 덕분에 `CosineMatcher`는 모델 계열과 무관하게 모든
  임베딩을 단위벡터로 취급할 수 있습니다.

**이 단계가 `Isolate` 안에서 실행되는 이유:** `FacePipeline._embedInIsolate`는
`Isolate.run(() => embedder.embed(face))`로 임베딩 호출을 감쌉니다. TFLite
추론은 파이프라인에서 가장 무거운 단계라서, UI isolate 밖에서 돌려야 추론 중에
앱 프레임률이 떨어지지 않습니다. 이 방식이 FFI 기반 `Interpreter` 객체를
캡처한 클로저를 isolate 경계 너머로 넘겨도 실제로 정상 동작하는지는 별도로
검증했습니다(6절 참고).

### 4단계 — 매칭 (`lib/src/matching/cosine_matcher.dart`)

`CosineMatcher`는 순수하고 상태 없는 함수입니다: probe 임베딩과 갤러리의 모든
`Enrollment` 사이 코사인 유사도를 계산해 최고점을 고르고, `threshold`를 넘을
때만 채택합니다. 임계값은 호출부에 하드코딩하지 않고
`CosineMatcher.fromManifest(manifest)`로 `manifest.matching.threshold`를 읽어
모델과 함께 따라다니게 합니다.

## 5. 라이선스 등급과 BYOM 모델

`ModelManifest.license`는 4가지 등급 중 하나를 가집니다:

| 등급 | 의미 | 재배포 가능 |
|---|---|---|
| `bundled` | facekit 패키지에 동봉(예: BlazeFace, Apache 2.0) | 가능 |
| `research` | 평가/PoC 전용(예: 비상업 데이터로 학습된 ArcFace buffalo_l) | 불가 |
| `byom` | 개발자가 직접 상업 라이선스 모델을 공급 | 해당 없음 |
| `licensed` | 특정 모델에 대해 상업 라이선스를 구매한 상태 | 가능(해당 라이선스 범위 내) |

`ModelManifest.assertLoadable({required bool isReleaseBuild})`는
재배포 불가능한 모델이 release 빌드(`kReleaseMode`)에서 로드되면 예외를
던집니다. 모든 로더(`BlazeFaceDetector.fromAsset/fromFile`,
`TfliteFaceEmbedder.fromAsset/fromFile`)가 `validate()` 직후 이를 호출하므로,
Demo/research 모델은 로컬 개발·테스트(debug/profile 빌드)에서는 자유롭게 쓸 수
있지만 누군가 release APK로 내보내려는 순간 바로 차단됩니다.

실제로는 이런 의미입니다: 개발 중 사용한 ArcFace `buffalo_l` 모델
(`assets/models/arcface_buffalo_l/manifest.json`)은 저장소에 **커밋되지
않습니다** — manifest만 커밋됩니다. 실제 `.tflite` 가중치는 개발자가 각자
로컬에서 받아서 변환하고(7절 참고), `.gitignore`
(`**/assets/models/*/*.tflite`, 동봉된 BlazeFace 모델만 예외 처리)로
제외됩니다.

## 6. 동시성 모델

- `BlazeFaceDetector.detect()`는 현재 호출 isolate에서 동기적으로 실행됩니다
  (얼굴 하나에 대한 검출은 임베딩에 비해 상대적으로 가볍습니다).
- `TfliteFaceEmbedder.embed()`는 `FacePipeline`이 `Isolate.run()`을 통해
  디스패치하므로, (더 무거운) 임베딩 추론이 UI isolate의 프레임 렌더링을
  막지 않습니다.
- 이는 통합 테스트(`test/pipeline/face_pipeline_smoke_test.dart`)로 끝까지
  검증했습니다 — 실제로 생성된 isolate 안에서 **진짜** ArcFace TFLite
  인터프리터를 실행하고, 반환된 임베딩이 수치적으로 올바른지 확인합니다
  (동일 입력에 대해 서로 다른 isolate 생성 2회에서 self-similarity가
  ≈ 1.0으로 일치).

## 7. 임베딩 모델을 로컬에서 직접 돌리는 법

임베딩 가중치는 동봉되지 않기 때문에, 실제로 동작하는 데모를 보려면:

1. 모델 export본을 구합니다(예: InsightFace `buffalo_l` →
   `w600k_r50.onnx`, WebFace600K로 학습된 ArcFace ResNet50 임베딩 모델).
2. ONNX → TensorFlow → TFLite로 변환합니다(`onnx2tf`), 입력
   `[1,112,112,3]`/출력 `[1,512]`의 float32 `.tflite`를 만듭니다.
3. 변환이 정확한지 수치로 검증합니다(같은 입력에 대해 ONNX와 TFLite 출력을
   비교 — 코사인 유사도가 ≈ 1.0이어야 함).
4. `.tflite`를 그 `manifest.json` 옆, 즉 `assets/models/arcface_buffalo_l/`
   (라이브러리 테스트용) 또는
   `example/assets/models/arcface_buffalo_l/`(example 앱용)에 둡니다.

## 8. Example 앱

`example/`은 실제 카메라 피드로 전체 파이프라인을 돌려보는 최소한의 Flutter
앱입니다: BlazeFace는 facekit 패키지에 동봉된 자산(`packages/facekit/assets/...`)
에서, ArcFace는 example 앱 자체의 로컬 자산에서 불러와 `FacePipeline`으로
엮은 뒤, 현재 카메라 프레임을 이름으로 **등록**하는 기능과 메모리상의
갤러리를 대상으로 계속 **인식**하는 기능, 그리고 전면/후면 카메라 전환
버튼을 제공합니다. 임베딩 모델이 연구용 라이선스이기 때문에 이 앱은 항상
debug/profile 빌드로만 만들어야 하며, release 빌드를 시도하면 5절에서 설명한
`assertLoadable` 가드에 걸립니다.
