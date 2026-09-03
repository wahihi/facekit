# Linux(Q6A) 헤드리스 포팅 — S1 의존 경계 실측 (2026-08-26)

`facekit_cli`(Flutter 없이 `dart run`으로 도는 헤드리스 인식 프로세스) 도입 전, "A안 — Dart 유지"가
실제로 며칠짜리 작업인지 측정하기 위한 1일 조사. 방법과 결론만 남긴다 — 실제 리팩터는 S2에서.

## 방법

1. `grep -rn "package:flutter\|dart:ui" lib/src/` — 계층별 Flutter 의존 스캔
2. `tool/probe/` — `facekit`을 `path:` 의존하는 순수 Dart 콘솔 패키지를 만들어 `dart pub get` 시도

**주의(이 조사 자체에서 나온 함정)**: 이 개발 VM의 `dart`는 `~/flutter/bin/dart`, 즉 Flutter 번들
Dart SDK다. 여기서 `pub get`을 돌리면 `flutter: sdk: flutter` 의존성이 조용히 해결돼버려서
"된다"는 거짓 양성이 나온다. Q6A 보드엔 Flutter 없이 독립 Dart SDK만 있으므로, 검증하려면 이 VM에도
별도로 독립 Dart SDK(`~/tools/dart-standalone/`)를 받아 그걸로 재현해야 한다.

## 발견 1 — 진짜 관문은 파일 단위가 아니라 `pubspec.yaml` 단위

```
$ ~/tools/dart-standalone/dart-sdk/bin/dart pub get   # 독립 Dart SDK, Flutter 없음
Resolving dependencies...
Because probe depends on facekit from path which requires the Flutter SDK,
version solving failed.
Flutter users should use `flutter pub` instead of `dart pub`.
```

`facekit/pubspec.yaml`이 루트에서 `dependencies: flutter: sdk: flutter`를 선언하고 있어서, **어떤
파일을 import하려 하든** pub 의존성 해석 단계에서 즉시 막힌다. `tflite_flutter` 쪽도 마찬가지로
자체 `pubspec.yaml`에 `dependencies: flutter: sdk: flutter`가 있어(`environment.flutter: '>=1.10.0'`),
설령 facekit 쪽에서 그 선언을 뺀다 해도 `tflite_flutter`를 의존성 그래프에 남겨두는 한 똑같이 막힌다.

**즉 "`debugPrint` 몇 개 바꾸면 된다" 수준이 아니라, 패키지를 쪼개야 한다.**

## 발견 2 — `core/`는 이미 완전히 깨끗함

`lib/src/core/models.dart`: `dart:typed_data` 외 의존성 0, `Rect`/`Point`/`Point3D` 전부 자체 정의
(dart:ui 아님). `grep`에서 `dart:ui` 매치 0건. R1 규칙 위반 없음 — core는 지금 그대로 재사용 가능.

## 발견 3 — 카메라 결합은 라이브러리 안에 없음

`lib/src/` 전체에서 `package:camera` / `CameraImage` 참조 0건. 카메라 결합은 전부 `example/`(Flutter
데모 앱)에 있다. 즉 S5(V4L2 캡처)는 라이브러리 코드를 안 건드리고 순수히 새 계층 추가로 끝난다.

## 발견 4 — `package:flutter/foundation.dart` 사용 7개 파일, 전부 대체 가능

| 파일 | 쓰는 심볼 |
|---|---|
| `alignment/affine_aligner.dart` | `debugPrint`, `visibleForTesting` |
| `detection/blazeface_detector.dart` | `kReleaseMode` |
| `detection/yunet_detector.dart` | `debugPrint`, `kReleaseMode` |
| `embedding/face_embedder.dart` | `kReleaseMode`, `debugPrint` |
| `landmark/face_landmarker.dart` | `debugPrint`, `kReleaseMode` |
| `liveness/blink_liveness_detector.dart` | `debugPrint` |
| `pipeline/face_pipeline.dart` | `debugPrint`, `kDebugMode` |

넷 다 Flutter 엔진과 무관한 순수 유틸이다:
- `kReleaseMode`/`kDebugMode`는 `foundation.dart` 내부에서도 `const bool.fromEnvironment('dart.vm.product')` 3줄로 구현돼 있다 — 그대로 복사하면 끝.
- `visibleForTesting`은 `package:meta/meta.dart`(순수 Dart 패키지)가 원본이고 `foundation.dart`는 그걸 재노출만 한다 — import만 `package:meta/meta.dart`로 바꾸면 끝.
- `debugPrint`는 그냥 `print`로 대체하거나, 버퍼링 없는 자체 구현으로 교체.

`lib/src/core/debug_flags.dart`가 이미 존재하지만 이건 별개(verbose 로깅 on/off 스위치)라 이 문제와
무관 — 새 shim은 여기 옆에 추가하면 된다.

## 발견 5 — `tflite_flutter` 결합은 이미 한 파일로 격리돼 있고, 재사용 가능한 순수 Dart FFI 코드가 그 안에 있음

`inference/tflite_runner.dart`가 이미 "Keeps tflite_flutter confined to this file" 원칙으로 짜여
있고, `TfliteRunner.fromFile()`(파일시스템 경로 로드)도 이미 존재한다 — 애초에 비-Flutter 컨텍스트를
어느 정도 염두에 두고 설계돼 있었다.

`tflite_flutter` 0.12.1 자체 소스(Apache 2.0)를 열어보면:

```
lib/src/bindings/tensorflow_lite_bindings_generated.dart  ← ffigen 산출물, Flutter 의존 없음
lib/src/bindings/bindings.dart                            ← 위와 동일
lib/src/ffi/helper.dart                                   ← 위와 동일
lib/src/interpreter.dart, tensor.dart                      ← 여기부터 flutter import 시작
```

즉 `libtensorflowlite_c.so`를 부르는 순수 Dart FFI 바인딩이 **이미 작성돼 있고 Apache 2.0**이라,
S2에서 ffigen으로 300줄을 새로 뽑는 대신 이 바인딩 파일들을 그대로 가져와(출처 명시) 얇은
`TfliteFfiRunner` 래퍼만 새로 짜면 된다. 예상보다 S2 작업량이 줄어드는 요인.

## 결론 — A안은 유효하지만 계획서의 "인터페이스 하나 추가"보다 범위가 크다

원래 S2 설명("비용 사실상 0, 인터페이스 뒤에 구현체 하나")은 **패키지 분리가 필요 없다는 잘못된
전제** 위에 있었다. 실측 결과 실제로 필요한 건:

1. **패키지 스플릿.** `core/`, `contracts.dart`, `alignment/`, `matching/`, `detection/`(YuNet),
   `embedding/`(어댑터), `landmark/`, `liveness/`, `pipeline/`을 `flutter: sdk: flutter`도
   `tflite_flutter`도 의존하지 않는 별도 pubspec(예: 같은 저장소 안 `packages/facekit_core/`)으로
   옮긴다. 기존 `facekit` 패키지는 그 위에 얇게 남아 `TfliteRunner`(Flutter/tflite_flutter 구현체)와
   Flutter 전용 export만 제공 — **`example/`(Flutter 데모 앱)는 무수정.**
2. `debugPrint`/`kReleaseMode` 등 7개 파일의 `package:flutter/foundation.dart` import를 순수 Dart
   shim으로 교체 (발견 4).
3. `tflite_flutter`의 기존 FFI 바인딩 파일을 이식해 `TfliteFfiRunner` 작성 (발견 5 — 원래 계획보다 쉬움).
4. `contracts.dart`에 `InferenceRunner` 추상 인터페이스를 추가해 `TfliteRunner`/`TfliteFfiRunner`
   양쪽이 구현하도록 — 나중에 QNN 러너를 붙일 자리를 지금 열어둔다(비용 0, 기존 결정 유지).

**작업량 추정 갱신**: 패키지 스플릿이 추가되면서 S2가 원래 추산(2~3일)보다 다소 늘어나지만, 바인딩
재사용(발견 5)으로 일부 상쇄된다 — 여전히 2~4일 범위로 본다. 전체 2~3주 추정은 유지.

## 커밋

이 문서 1개. `dart:ui` 제거 리팩터는 필요 없었음(발견 2). `tool/probe/`는 조사에 쓴 최소 재현
픽스처로 남겨둔다(`pubspec.yaml`, `main.dart`만 — 생성 산출물은 `.gitignore` 처리됨).

## S2 완료 (2026-08-26)

발견 1(패키지 스플릿 필요)에 따라 `packages/facekit_core/`(순수 Dart, `flutter:`도
`tflite_flutter`도 미의존)를 실제로 분리했다. `core/alignment/detection/embedding/
landmark/liveness/matching/pipeline/image/inference`(단, `tflite_runner.dart` 제외)가
전부 이동했고, 기존 `facekit` 패키지는 그 위에 얇게 남아 `TfliteRunner`(Flutter/
tflite_flutter 구현체)와 에셋 로더(`lib/src/asset_loaders.dart`)만 제공한다.
`example/`은 5개 호출부(`X.fromAsset(...)` → `loadXFromAsset(...)`)만 기계적으로
바뀌었을 뿐 동작은 무수정.

발견 5(재사용 가능한 FFI 바인딩)에 따라 `TfliteFfiRunner`(`packages/facekit_core/lib/src/
inference/tflite_ffi_runner.dart`)를 작성 — `tflite_flutter`의 ffigen 산출물
(`bindings/tensorflow_lite_bindings_generated.dart`, Apache 2.0, 출처는
`third_party/tflite_flutter/NOTICE.md`)을 그대로 이식하고, 그 위에 float32 전용 텐서
마샬링 레이어를 새로 짰다. `core/contracts.dart`에 `InferenceRunner` 추상 인터페이스를
추가해 `TfliteRunner`/`TfliteFfiRunner` 둘 다 구현하도록 — 나중에 QNN 러너를 붙일 자리.

**검증 결과**:
- `dart analyze`(독립 Dart SDK, `~/tools/dart-standalone/`)로 `facekit_core` 무오류 —
  Flutter 의존 0을 실측으로 재확인.
- `dart test`(같은 독립 SDK)로 110개 중 106개 통과. 실패 4개(`blazeface_smoke_test`,
  `yunet_smoke_test`×2, `face_landmarker_smoke_test`)는 리팩터 이전과 **동일한 4개**이고
  원인도 동일(이 환경에 `libtensorflowlite_c.so`가 없음) — 회귀 아님, CI의 기존
  제외 목록과 같은 성격의 gap(`yunet_smoke_test.dart`는 CI 제외 목록에 빠져 있던 걸
  같이 고쳤다 — `.github/workflows/test.yml`).
- `flutter analyze`/`flutter test`(outer) 무오류, `flutter build apk --debug`
  (example) 성공 — 안드로이드 경로 무회귀.

## x86_64용 libtensorflowlite_c.so 빌드 성공 (2026-08-26, 이어서)

CMake로 직접 빌드해서 실제로 구했다. 공식 가이드(developers.google.com/edge/litert/build/cmake)
그대로 시도했다가 한 번 실패했는데, 원인과 해결책을 남겨둔다 — arm64(Q6A)에서도 똑같이 겪을
문제라 재발 방지 차원.

**1차 시도 실패 원인**: `git clone`(플래그 없음, 즉 master HEAD)으로 받은 뒤 cmake configure를
돌렸더니, `tensorflow/lite/CMakeLists.txt`가 **내부적으로 또 한 번** `FetchContent`로 tensorflow를
따로 받는다(`GIT_TAG v2.21.0-rc0`로 고정돼 있음) — 결과적으로 내 클론(master HEAD)과 내부
재클론(v2.21.0-rc0)이 서로 다른 커밋이 되고, 두 트리가 뒤섞이면서 `mmap_handle.cc`/`.h` 간
선언 불일치로 컴파일 에러(`no declaration matches ...`)가 났다. TF 자체 버그가 아니라 **버전
불일치**였다.

**해결**: 처음부터 `git clone --depth 1 --branch v2.21.0-rc0`로 CMakeLists.txt가 기대하는
바로 그 태그를 받고, `cmake -DTENSORFLOW_SOURCE_DIR=<내 클론 경로> ...`로 내부 재클론 자체를
스킵시켰다. 이후 정상 빌드됨 — sudo 없는 이 VM이라 cmake도 Kitware 공식 바이너리를
`~/tools/cmake`에 따로 설치해야 했다(apt 불가).

**검증**: `FACEKIT_TFLITE_C_LIBRARY=<so 경로> dart test`로 `facekit_core` 전체 재실행 —
**114개 중 114개 통과**(BYOM 모델 3개는 정상 skip). 이전엔 실패하던 4개(`blazeface_smoke_test`,
`yunet_smoke_test`×2, `face_landmarker_smoke_test`)가 전부 실제 추론까지 통과했고,
`face_pipeline_smoke_test.dart`(Isolate.run()이 FFI 포인터를 격리된 isolate로 잘 넘기는지
검증하는 테스트)도 통과 — `TfliteFfiRunner`가 진짜로 동작한다는 뜻.

**다음(Q6A/arm64 때 참고)**: 같은 레시피(태그 고정 + `TENSORFLOW_SOURCE_DIR` 오버라이드)를
그대로 쓰면 된다. 빌드 산출물(`libtensorflowlite_c.so`, 8.8MB)은 저장소에 커밋하지 않음 —
플랫폼별 바이너리라 BYOM/네이티브 아티팩트 취급, 로컬에만 둔다(`~/tools/tflite_c_build/
tflite_build/`).

## S3 — facekit_cli 착수 (2026-08-26, 이어서)

`packages/facekit_cli/`(`facekit_core` path 의존, `package:image`로 JPEG 디코드,
`package:args`로 CLI 인자 파싱) 작성:

```
dart run bin/facekit_cli.dart --image a.jpg \
  --detector assets/models/yunet_160 --embedder example/assets/models/auraface
→ {"faces":[{"bbox":[...],"score":..,"embedding":[512개],"ms":{"detect":..,"align":..,"embed":..}}]}
```

`tool/model_verification/`에 이미 있던 실사진(`me_1.jpg`, `me_2.jpg`, `other_1.jpg` — 프로젝트
초기부터 모델 검증에 써온, 저장소에 커밋된 사진)으로 세 장 다 돌려서 자기 일관성을 확인했다:

| 비교 | 코사인 유사도 | 판정 (threshold 0.211) |
|---|---|---|
| me_1 vs me_2 (동일 인물) | **0.745** | 통과(같은 사람으로 정확히 인식) |
| me_1 vs other_1 (타인) | 0.026 | 정확히 거부 |
| me_2 vs other_1 (타인) | 0.0006 | 정확히 거부 |

YuNet 검출(bbox score 0.92~0.95) + AuraFace 임베딩이 실제로 올바르게 동작한다는 걸
안드로이드 비교 없이도 확인할 수 있는 강력한 정황 증거다.

**다만 이건 원래 S3 DoD(안드로이드 대비 코사인 ≥0.999)는 아직 아니다.** 지금 확인한 건
"같은 사람/다른 사람을 올바르게 구분하는가"고, 원래 계획이 잡으려던 건 "리눅스와 안드로이드가
같은 입력에 대해 플랫폼 차이 없이 같은 숫자를 내는가"(회전 보정 누락, 채널 순서 버그 같은
플랫폼별 버그를 잡기 위한 것). 이건 안드로이드 실기기에서 이 세 사진(또는 아무 사진)을 같은
파이프라인으로 돌린 실제 임베딩값이 있어야 비교할 수 있어서, 이 세션 안에서는 못 끝낸다 —
종화님 Pixel 7에서 한 번 돌려주셔야 하는 부분.

## S3 DoD 통과 (2026-09-03)

`example/`에 임시 "골든 벡터 덤프" 버튼을 추가(`_dumpGoldenVector`, `assets/golden/me_1.jpg` —
`tool/model_verification/me_1.jpg`와 바이트 동일) — Pixel 7 실기기에서 눌러서 JSON을 클립보드로
바로 복사해 받았다(adb/logcat 불필요). 같은 사진을 리눅스 `facekit_cli`로도 돌려서 비교:

| | 값 |
|---|---|
| 코사인 유사도 | **0.99999991** |
| bbox 차이 | 소수점 4자리 수준(부동소수점 반올림 오차) |
| score | 완전히 동일 (0.9324665440) |

기준(≥0.999)을 크게 상회 — **S3 DoD 통과.** 안드로이드와 리눅스가 회전·채널 순서·정규화 순서
전부 일치한다는 뜻.

이 안드로이드 임베딩을 `packages/facekit_cli/test/fixtures/me_1_android_golden.json`에 커밋해서
`packages/facekit_cli/test/golden_vector_test.dart` 회귀 테스트로 고정했다 — 이후 파이프라인을
건드릴 때마다 리눅스 쪽이 이 실기기 값에서 벗어나지 않는지 계속 검증된다. CI에는 다른
모델-로딩 테스트들과 같은 이유(이 환경에 `libtensorflowlite_c.so` 없음)로 실행 단계는 아직
안 넣었고, `pub get`/`dart analyze`만 넣어뒀다.

`bin/facekit_cli.dart`의 로직을 `lib/facekit_cli.dart`(`runDetectAlignEmbed`)로 뽑아서
바이너리와 테스트가 같은 코드를 쓰도록 리팩터 — CLI 자체 동작은 무변경.

**S3 완료.** 다음은 S4(Q6A/arm64 이동).

## S4 완료 (2026-09-03)

코드를 tar.gz + USB로 보드에 옮김(이 환경에서 보드로 직접 scp/ping이 안 됨 — VirtualBox NAT
어댑터가 인터넷은 되지만 같은 물리 LAN의 다른 기기로는 못 나감). 보드에서 x86_64 때와 같은
레시피(태그 `v2.21.0-rc0` 고정 + `TENSORFLOW_SOURCE_DIR` 오버라이드)로 arm64용
`libtensorflowlite_c.so` 빌드 — 중간에 `g++`가 없어서 한 번 막혔다가(`sudo apt install
build-essential`로 해결) 이후 성공.

- **DoD #1 (골든 벡터 arm64 재검증)**: `dart test` 전체 통과 — arm64에서도 안드로이드 실기기
  값과 일치.
- **DoD #2 (실측 벤치마크)**: [doc/KR/benchmark_q6a.md](benchmark_q6a.md) — 싱글스레드
  721ms → A78 4코어 멀티스레드 350ms. **결론: NPU 없이 CPU만으로 상용 기준(체감 지연
  0.5초 이내) 통과, QNN/INT8 투자 근거 없음.**

이 발견에 맞춰 `YuNetDetector.fromFile`/`TfliteFaceEmbedder.fromFile`/`TfliteFfiRunner`에
`numThreads` 옵션과 `facekit_cli --num-threads`를 추가(기본값 1, 기존 동작 무변경).

**S4 완료.** 다음은 S5(V4L2 카메라 캡처).
