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

**아직 안 한 것**: x86_64/arm64용 `libtensorflowlite_c.so` 자체를 구하거나 빌드하는
일 — tflite_flutter 공식 문서 자체가 "리눅스는 프리빌드 없음, Bazel/CMake로 직접
빌드"라고 명시한다. 원래 로드맵의 S4("arm64 이동")에서 어차피 필요했던 작업이라 이번
스코프에서 의도적으로 미뤘다. 지금 코드는 그 라이브러리가 없어도 컴파일·타입체크는
전부 통과하고, 실제 추론만 그 시점까지 막혀 있다.
