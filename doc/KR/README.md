🇺🇸 [English README](../../README.md)

---

# facekit

[![test](https://github.com/wahihi/facekit/actions/workflows/test.yml/badge.svg)](https://github.com/wahihi/facekit/actions/workflows/test.yml)

**Flutter용 온디바이스(On-device) 얼굴인식 SDK.** 공개된 모델·논문·공식 문서만 참고해
새로 작성한 클린룸(clean-room) 구현입니다 — 사내 코드나 독점 라이브러리를 참조·복사하지
않았습니다.

- 코드: **Apache License 2.0** ([LICENSE](../../LICENSE))
- 임베딩: 상업 사용 가능한 **AuraFace(Apache 2.0)를 기본으로 동봉**하며, 매니페스트
  교체로 다른 모델도 연결 가능(BYOM 구조 유지) — 아래 [라이선스 / 모델 정책](#라이선스--모델-정책-byom) 참고

---

## 무엇을 하는 SDK인가

카메라(또는 갤러리 이미지) → 얼굴 검출 → 정렬 → 임베딩 → 매칭까지, 얼굴인식 파이프라인
전 단계를 단말기 안에서(네트워크 호출 없이) 처리합니다.

- **검출**: BlazeFace (MediaPipe, Apache 2.0) — SDK에 동봉, 별도 다운로드 불필요
- **임베딩**: 모델 교체 가능 구조 — 기본값으로 **AuraFace**(Apache 2.0) 동봉, 그 외
  ArcFace / AdaFace / MobileFaceNet / FaceNet 어댑터도 내장(가중치는 BYOM, 직접 준비)
- **매칭**: 코사인 유사도 기반, manifest의 임계값으로 정/오답 판정
- **라이브니스(Free)**: 눈 깜빡임(Blink, EAR) 검출 — 정지된 사진을 들이대면 통과하지 못함
- **온디바이스 전용**: 임베딩 등 무거운 추론은 별도 isolate에서 실행해 UI를 막지 않음

자세한 설계는 [doc/EN/architecture.md](../EN/architecture.md) /
[doc/KR/architecture.md](architecture.md)에 있습니다.

## 빠른 시작

Flutter/Android 개발 환경이 전혀 없는 상태라면, OS/하드웨어 요구사항부터 SDK 설치,
BYOM 모델 배치, 빌드, 기기 설치까지 처음부터 따라할 수 있는
[설치 가이드](installation.md)를 먼저 보세요.


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

// 2) 임베딩기 — AuraFace가 기본 동봉(Apache 2.0, tool/fetch_models.sh로 받음).
//    다른 BYOM 모델을 쓰려면 경로만 바꾸면 됨 (아래 라이선스 섹션 참고)
final embedderManifest = ModelManifest.fromJsonString(
  await rootBundle.loadString('assets/models/auraface/manifest.json'),
);
final embedder = await TfliteFaceEmbedder.fromAsset(
  tfliteAssetPath: 'assets/models/auraface/auraface_r100_fp16.tflite',
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

실기기에서 example 앱 내장 벤치마크 버튼으로 측정 (n=30). 방법론·VM 비교·
정확도(EER) 표 등 전체 내용은 [doc/KR/benchmark.md](benchmark.md)(또는
[영문](../EN/benchmark.md)) 참고.

**AuraFace(기본 임베딩 모델)** — Pixel 7, `--release` 빌드
([GitHub Releases](https://github.com/wahihi/facekit/releases)에 올라온 것과
동일한 APK), 재설치 후 독립적으로 2회 측정:

| 기기 | 모드 | 검출(BlazeFace) | 임베딩(AuraFace) | 전체 1프레임 |
|---|---|---|---|---|
| Pixel 7 | CPU (기본값) | 평균 51~52ms | 평균 1083~1186ms | 평균 1135~1238ms |
| Pixel 7 | NNAPI | 평균 50~61ms | 평균 1106~1205ms | 평균 1156~1267ms |
| Galaxy S25 (SM-S931N) | CPU (기본값) | 평균 56.3ms | 평균 598.8ms | 평균 655.7ms |
| Galaxy S25 (SM-S931N) | NNAPI | 평균 46.3ms | 평균 480.2ms | 평균 527.2ms |

Galaxy S25 행은 측정자 본인 기기가 아니라 빌린 기기라 **1회 측정**이며
재측정이 불가능합니다 — 특히 NNAPI가 더 빠르게 나온 결과는 확정값이 아니라
참고치로만 보세요(자세한 단서는 위 문서 참고). Pixel 7의 두 측정 사이 9~10%
정도 편차(검출은 안정적)는 발열·백그라운드 프로세스·재설치 직후 콜드 상태 등
실기기 벤치마크에서 흔한 변동으로 보입니다. AuraFace(ResNet100)는 아래
ArcFace buffalo_l(ResNet50)보다 느린데, 배율은 기기마다 달라(전체 1프레임
기준 Pixel 7 약 1.5배, Galaxy S25 약 2.2배) 더 깊은 백본을 처리하는 효율이
칩마다 다르기 때문으로 보입니다.

**ArcFace buffalo_l (BYOM 예시, 기본값 아님)** — `--profile` 빌드(ArcFace는
연구용 라이선스라 `--release`에서는 로드 자체가 막힘, 아래 라이선스 섹션 참고):

| 기기 | 모드 | 검출(BlazeFace) | 임베딩(ArcFace buffalo_l) | 전체 1프레임 |
|---|---|---|---|---|
| Pixel 7 | CPU (기본값) | 평균 65.1ms | 평균 729.4ms | 평균 795.6ms |
| Pixel 7 | NNAPI | 평균 76.6ms | 평균 876.2ms | 평균 954.4ms |
| Galaxy S25 (SM-S931N) | CPU (기본값) | 평균 48.0ms | 평균 256.6ms | 평균 305.0ms |
| Galaxy S25 (SM-S931N) | NNAPI | 평균 44.7ms | 평균 244.4ms | 평균 289.5ms |

두 모델 다 기본값은 CPU입니다. Pixel 7에서는 두 모델 모두 NNAPI 전체 평균이
CPU보다 근소하게 느리고(ArcFace 쪽이 더 큼), p95는 NNAPI가 항상 더 나쁩니다.
Galaxy S25(Snapdragon 8 Elite)에서는 ArcFace 기준 NNAPI가 평균은 ~15ms 더
빠르지만 p95 분산이 훨씬 큽니다(386ms vs 314ms) — 결국 어느 조합이든 기본값 CPU
유지가 안전합니다. 위 두 표는 빌드 모드 자체가 다르니(이유는 각 문서 참고),
둘 사이 배율은 엄밀한 비교가 아니라 참고치로만 보세요.

정확도(LFW 200쌍 기준 EER)는 ArcFace 8.5% / AdaFace 2.0%이며, 저화질 조건에서는
AdaFace가 더 강건합니다(자세한 수치는 [doc/KR/adaface_verification.md](adaface_verification.md)).
AuraFace는 아직 EER 측정이 없습니다(지금까지는 동일인 쌍 유사도만 확인 — 남겨둔
임계값 재튜닝 이슈 참고).

## 라이브니스 / Free·Pro 경계

이 저장소(Free)에는 **눈 깜빡임(Blink) 기반 라이브니스만** 들어있습니다 — 정지된 사진이나
화면 캡처를 들이대면 EAR(눈 개폐 비율)이 변하지 않아 통과하지 못합니다. 단, 눈 부분에
구멍을 낸 마스크나 동영상 재생 공격까지는 방어하지 못합니다 — 이런 멀티신호 방어는 별도
구현체(Pro)가 같은 `LivenessDetector` 인터페이스 뒤에 들어가는 구조이며, **이 저장소에는
포함되어 있지 않습니다.** 자세한 한계는 [doc/KR/liveness.md](liveness.md) 참고.

## 라이선스 / 모델 정책 (BYOM)

직접 작성한 코드 전체는 **Apache License 2.0**입니다 ([LICENSE](../../LICENSE)).

이 SDK는 **상업 사용 가능한 AuraFace(Apache 2.0)를 기본 임베딩 모델로 제공합니다** —
가중치 자체는 (라이선스가 아니라 용량 문제로) `tool/fetch_models.sh`가 빌드/개발
시점에 받아오고 커밋되진 않습니다. 그 아래의 매니페스트 기반 어댑터 구조는 여전히
BYOM을 지원해서, `manifest.json` + `.tflite`만 놓으면 다른 임베딩 모델로도 교체할
수 있습니다. `assets/models/*/manifest.json`마다 `license.tier`/`license.redistributable`
필드로 라이선스를 명시하고, `.gitignore`로 재배포 불가 가중치 파일을 저장소에서
제외합니다:

| 모델 | 역할 | 라이선스 | 동봉 여부 |
|---|---|---|---|
| BlazeFace short-range | 검출 | Apache 2.0 (MediaPipe) | ✅ 동봉 |
| MediaPipe Face Landmarker (478점) | 라이브니스용 랜드마크 | Apache 2.0 (MediaPipe) | ✅ 동봉 |
| AuraFace (glintr100 / ResNet100) | 임베딩 (기본값) | Apache 2.0 (fal.ai) | ✅ 동봉 (`tool/fetch_models.sh`로 받음) |
| ArcFace (buffalo_l / w600k_r50) | 임베딩 (BYOM 예시) | 비상업 연구용 (InsightFace) | ❌ BYOM |
| AdaFace (IR-101 / WebFace12M) | 임베딩 | 비상업 연구용 (mk-minchul/AdaFace) | ❌ BYOM |
| FaceNet512 | 임베딩 | 비상업 연구용 | ❌ BYOM |
| MobileFaceNet | 임베딩 | 배포본에 따라 다름(확인 필요) | ❌ BYOM |

BYOM 모델은 각 `manifest.json`의 `license.source`에 적힌 원본 저장소에서 직접 받아
해당 폴더에 배치해야 합니다 — 전체 절차는 설치 가이드의
[부록 A](installation.md#부록-a--다른-임베딩-모델-쓰기-byom) 참고. `ModelManifest.assertLoadable()`이
`redistributable:false` 모델은 **release 빌드에서 로드 자체를 막아** 라이선스 위반
가능성을 코드 레벨에서 차단합니다([lib/src/inference/model_manifest.dart](../../lib/src/inference/model_manifest.dart)).

## 사용한 오픈소스

| 항목 | 라이선스 | 출처 |
|---|---|---|
| BlazeFace short-range | Apache 2.0 | https://github.com/google/mediapipe |
| MediaPipe Face Landmarker | Apache 2.0 | https://github.com/google/mediapipe |
| tflite_flutter | Apache 2.0 | https://pub.dev/packages/tflite_flutter |
| camera (Flutter plugin) | BSD-3-Clause | https://pub.dev/packages/camera |

BYOM 대상 임베딩 모델(ArcFace/AdaFace/FaceNet/MobileFaceNet)의 출처는 위
[라이선스 / 모델 정책](#라이선스--모델-정책-byom) 표와 각 `manifest.json`을 참고하세요.

## AI 활용 안내

이 프로젝트는 **Claude Code**(Anthropic)와 협업하여 개발되었습니다 — 초기 아키텍처
결정부터 구현, 디버깅, 문서화까지 전 과정에서 AI의 도움을 받았고, 모든 작업은
개발자 본인의 지시와 검토 하에 진행되었습니다. 그 협업이 실제로 어떤 모습인지
(엉뚱한 길로 샜던 부분까지 포함해서) 있는 그대로 보여주는 디버깅 기록을
[doc/KR/postmortem/](postmortem/)에 남겨두었습니다.

2026-07-24 이후 커밋에는 `Co-Authored-By: Claude` 트레일러가 붙습니다. 그
이전 커밋들은 이 공개 방침이 생기기 전에 만들어졌지만, 개발 방식 자체는
동일했습니다.

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
