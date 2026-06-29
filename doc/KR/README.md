🇺🇸 [English README](../../README.md)

---

# facekit

**Flutter용 온디바이스(On-device) 얼굴인식 SDK.** 공개된 모델·논문·공식 문서만 참고해
새로 작성한 클린룸(clean-room) 구현입니다 — 사내 코드나 독점 라이브러리를 참조·복사하지
않았습니다.

- 코드: **Apache License 2.0** ([LICENSE](../../LICENSE))
- 임베딩 모델: **BYOM(Bring Your Own Model)** — 아래 [라이선스 / 모델 정책](#라이선스--모델-정책-byom) 참고

---

## 무엇을 하는 SDK인가

카메라(또는 갤러리 이미지) → 얼굴 검출 → 정렬 → 임베딩 → 매칭까지, 얼굴인식 파이프라인
전 단계를 단말기 안에서(네트워크 호출 없이) 처리합니다.

- **검출**: BlazeFace (MediaPipe, Apache 2.0) — SDK에 동봉, 별도 다운로드 불필요
- **임베딩**: 모델 교체 가능 구조 — ArcFace / AdaFace / MobileFaceNet / FaceNet 어댑터
  내장. 실제 가중치는 BYOM(직접 준비)
- **매칭**: 코사인 유사도 기반, manifest의 임계값으로 정/오답 판정
- **라이브니스(Free)**: 눈 깜빡임(Blink, EAR) 검출 — 정지된 사진을 들이대면 통과하지 못함
- **온디바이스 전용**: 임베딩 등 무거운 추론은 별도 isolate에서 실행해 UI를 막지 않음

자세한 설계는 [doc/EN/architecture.md](../EN/architecture.md) /
[doc/KR/architecture.md](architecture.md)에 있습니다.

## 빠른 시작

```dart
import 'package:facekit/facekit.dart';

// 1) 검출기 — SDK에 동봉된 모델이라 바로 로드 가능
final detectorManifest = ModelManifest.fromJsonString(
  await rootBundle.loadString('packages/facekit/assets/models/blazeface_short/manifest.json'),
);
final detector = await BlazeFaceDetector.fromAsset(
  tfliteAssetPath: 'packages/facekit/assets/models/blazeface_short/face_detection_short_range.tflite',
  manifest: detectorManifest,
);

// 2) 임베딩기 — BYOM: 직접 구한 .tflite 가중치 경로를 지정 (아래 라이선스 섹션 참고)
final embedderManifest = ModelManifest.fromJsonString(
  await rootBundle.loadString('assets/models/arcface_buffalo_l/manifest.json'),
);
final embedder = await TfliteFaceEmbedder.fromAsset(
  tfliteAssetPath: 'assets/models/arcface_buffalo_l/w600k_r50.tflite',
  manifest: embedderManifest,
);

// 3) 파이프라인 구성 후 등록/인식
final pipeline = FacePipeline(
  detector: detector,
  aligner: AffineAligner.arcface112(),
  embedder: embedder,
  matcher: CosineMatcher.fromManifest(embedderManifest),
);

final embedding = await pipeline.enroll(faceImage);            // 등록
final result = await pipeline.identify(faceImage, gallery);     // 인식 → MatchResult?
```

카메라 연동, 박스 오버레이, 라이브니스, 벤치마크 버튼까지 포함된 전체 예제는
[example/](../../example/)에 있습니다.

## 벤치마크

Pixel 7 실기기에서 example 앱 내장 벤치마크 버튼으로 측정 (n=30, profile 빌드).
방법론·VM 비교·정확도(EER) 표 등 전체 내용은 [doc/KR/benchmark.md](benchmark.md) 참고.

| 모드 | 검출(BlazeFace) | 임베딩(ArcFace buffalo_l) | 전체 1프레임 |
|---|---|---|---|
| CPU (기본값) | 평균 65.1ms | 평균 729.4ms | 평균 795.6ms |
| NNAPI | 평균 76.6ms | 평균 876.2ms | 평균 954.4ms |

실측 결과 이 모델 조합(float32)에서는 NNAPI가 CPU보다 오히려 느려서, 기본값은 CPU
고정으로 유지하고 있습니다 — 자세한 원인 분석은 위 문서에 정리했습니다.

정확도(LFW 200쌍 기준 EER)는 ArcFace 8.5% / AdaFace 2.0%이며, 저화질 조건에서는
AdaFace가 더 강건합니다(자세한 수치는 [doc/KR/adaface_verification.md](adaface_verification.md)).

## 라이브니스 / Free·Pro 경계

이 저장소(Free)에는 **눈 깜빡임(Blink) 기반 라이브니스만** 들어있습니다 — 정지된 사진이나
화면 캡처를 들이대면 EAR(눈 개폐 비율)이 변하지 않아 통과하지 못합니다. 단, 눈 부분에
구멍을 낸 마스크나 동영상 재생 공격까지는 방어하지 못합니다 — 이런 멀티신호 방어는 별도
구현체(Pro)가 같은 `LivenessDetector` 인터페이스 뒤에 들어가는 구조이며, **이 저장소에는
포함되어 있지 않습니다.** 자세한 한계는 [doc/KR/liveness.md](liveness.md) 참고.

## 라이선스 / 모델 정책 (BYOM)

직접 작성한 코드 전체는 **Apache License 2.0**입니다 ([LICENSE](../../LICENSE)).

이 SDK는 임베딩 모델 가중치를 **동봉하지 않습니다.** `assets/models/*/manifest.json`마다
`license.tier`/`license.redistributable` 필드로 라이선스를 명시하고, `.gitignore`로
재배포 불가 가중치 파일을 저장소에서 제외합니다:

| 모델 | 역할 | 라이선스 | 동봉 여부 |
|---|---|---|---|
| BlazeFace short-range | 검출 | Apache 2.0 (MediaPipe) | ✅ 동봉 |
| MediaPipe Face Landmarker (478점) | 라이브니스용 랜드마크 | Apache 2.0 (MediaPipe) | ✅ 동봉 |
| ArcFace (buffalo_l / w600k_r50) | 임베딩 | 비상업 연구용 (InsightFace) | ❌ BYOM |
| AdaFace (IR-101 / WebFace12M) | 임베딩 | 비상업 연구용 (mk-minchul/AdaFace) | ❌ BYOM |
| FaceNet512 | 임베딩 | 비상업 연구용 | ❌ BYOM |
| MobileFaceNet | 임베딩 | 배포본에 따라 다름(확인 필요) | ❌ BYOM |

BYOM 모델은 각 `manifest.json`의 `license.source`에 적힌 원본 저장소에서 직접 받아
해당 폴더에 배치해야 합니다. `ModelManifest.assertLoadable()`이 `redistributable:false`
모델은 **release 빌드에서 로드 자체를 막아** 라이선스 위반 가능성을 코드 레벨에서
차단합니다([lib/src/inference/model_manifest.dart](../../lib/src/inference/model_manifest.dart)).

## 사용한 오픈소스

| 항목 | 라이선스 | 출처 |
|---|---|---|
| BlazeFace short-range | Apache 2.0 | https://github.com/google/mediapipe |
| MediaPipe Face Landmarker | Apache 2.0 | https://github.com/google/mediapipe |
| tflite_flutter | Apache 2.0 | https://pub.dev/packages/tflite_flutter |
| camera (Flutter plugin) | BSD-3-Clause | https://pub.dev/packages/camera |

BYOM 대상 임베딩 모델(ArcFace/AdaFace/FaceNet/MobileFaceNet)의 출처는 위
[라이선스 / 모델 정책](#라이선스--모델-정책-byom) 표와 각 `manifest.json`을 참고하세요.

## 디렉터리 구조

```
lib/src/
  core/        순수 Dart 데이터 모델·수학·인터페이스 (Flutter 의존성 없음)
  inference/   TFLite 연동, manifest 파싱/라이선스 가드
  image/       카메라 프레임(YUV420) → RGB 변환, 리사이즈/크롭
  detection/   BlazeFace
  alignment/   5점 어파인 정렬
  embedding/   임베딩 어댑터(ArcFace/AdaFace/FaceNet) + 매니페스트 기반 로더
  matching/    코사인 매처
  landmark/    MediaPipe Face Landmarker (478점)
  liveness/    눈 깜빡임 라이브니스
  pipeline/    검출→정렬→임베딩→매칭 오케스트레이션 (isolate 디스패치)
example/       카메라 연동 + 박스 오버레이 + 라이브니스 + 벤치마크 버튼이 있는 데모 앱
doc/EN, doc/KR 설계 문서, 검증 기록, 벤치마크
```

의존 방향은 항상 단방향입니다: `UI/example → pipeline → detection/alignment/embedding/
matching → inference → core`. `core/`는 어떤 상위 레이어에도, Flutter에도 의존하지
않습니다.
