🇺🇸 [English version](../EN/installation.md)

---

# 설치 가이드 (처음부터 따라하기)

facekit을 처음 받아서 example 앱을 실기기에서 실행해보기까지, 필요한 모든 단계를
순서대로 정리한 문서입니다. Flutter/Android 개발 환경이 전혀 없는 새 PC를 기준으로
작성했고, 실제로 이 환경(Ubuntu 22.04, Flutter 3.44.2)에서 검증한 명령만 적었습니다.

> facekit 자체는 **라이브러리(패키지)**라서 단독으로 실행되는 화면이 없습니다.
> 실행 가능한 앱은 [example/](../../example/) 폴더에 따로 있습니다 — 저장소
> 루트에서 `flutter run`을 하면 `Target file "lib/main.dart" not found` 에러가
> 나는 게 정상이니, 이 문서를 따라 `example/` 안에서 실행하세요.

## 0. 시스템 요구사항

| 항목 | 요구사항 |
|---|---|
| OS | Linux / macOS / Windows (Flutter가 지원하는 OS) — 이 문서는 **Ubuntu 22.04 (Linux)** 기준 |
| CPU | x86_64 또는 ARM64, 64비트 필수 |
| RAM | 8GB 이상 권장 (Android 에뮬레이터까지 쓰려면 16GB 권장) |
| 디스크 여유 공간 | **10GB 이상** — Flutter SDK(~2.3GB) + Android SDK(~3GB) + 빌드 캐시 + 모델 파일 |
| 빌드 도구 | JDK **17** (Android Gradle Plugin이 요구), Android SDK (compileSdk **36**), Flutter **3.44.2** stable, Dart **3.12.2** |
| 테스트 대상 기기 | Android 실기기 (minSdk **24** = Android 7.0 이상) — USB 디버깅 가능한 기기 권장 |

> iOS는 이 환경에서 검증하지 않았습니다(빌드/실행 모두 미확인). 이 문서는 Android만
> 다룹니다.

## 1. 사전 설치 — Flutter SDK

이미 Flutter가 설치되어 있다면 이 단계는 건너뛰고 2번으로 가세요.

```bash
# 원하는 위치에 압축 해제 (예: ~/development)
cd ~
git clone https://github.com/flutter/flutter.git -b stable development/flutter
# 또는 https://docs.flutter.dev/get-started/install 에서 압축본 다운로드
```

`~/.bashrc` (zsh면 `~/.zshrc`)에 PATH를 추가합니다:

```bash
export PATH="$HOME/development/flutter/bin:$PATH"
```

새 터미널을 열거나 `source ~/.bashrc`로 적용한 뒤 확인:

```bash
flutter --version
# Flutter 3.44.2 • channel stable ... 가 출력되면 OK
```

## 2. Android SDK / JDK 설치

Android Studio를 설치하면 SDK가 같이 깔리지만, SDK만 따로 깔아도 됩니다.

1. **JDK 17** 설치 (Temurin 권장): https://adoptium.net 에서 받거나 패키지 매니저로:
   ```bash
   # Ubuntu 예시
   sudo apt install openjdk-17-jdk
   ```
2. **Android SDK cmdline-tools** 설치 — Android Studio를 설치했다면 `~/Android/Sdk`에
   이미 있습니다. 없다면 https://developer.android.com/studio#command-tools 에서
   cmdline-tools만 받아서 `~/Android/Sdk/cmdline-tools/latest/`에 풉니다.
3. `~/.bashrc`에 추가:
   ```bash
   export JAVA_HOME="$HOME/development/jdk17"          # 본인 JDK 17 설치 경로로
   export ANDROID_HOME="$HOME/Android/Sdk"
   export ANDROID_SDK_ROOT="$HOME/Android/Sdk"
   export PATH="$JAVA_HOME/bin:$ANDROID_HOME/platform-tools:$ANDROID_HOME/cmdline-tools/latest/bin:$PATH"
   ```
   적용 후:
   ```bash
   "$JAVA_HOME/bin/java" -version   # openjdk 17.x 가 나와야 함
   ```
4. 필요한 SDK 패키지 설치 및 라이선스 동의:
   ```bash
   sdkmanager "platform-tools" "platforms;android-36" "build-tools;36.0.0"
   flutter doctor --android-licenses   # 전부 y
   ```
5. 점검:
   ```bash
   flutter doctor -v
   ```
   `[✓] Android toolchain`, `[✓] Connected device` 등이 보이면 됩니다. `Linux toolchain`
   항목(데스크톱 빌드용 clang/ninja/gtk3)은 **Android만 쓸 거면 무시해도 됩니다.**

## 3. 기기 준비 — USB 디버깅

`flutter run`으로 실기기에 바로 설치하려면 미리 연결해둬야 합니다.

1. 폰에서 **설정 → 휴대전화 정보(기기 정보)** → **빌드 번호**를 7번 연속 탭
2. **설정 → 시스템 → 개발자 옵션** → **USB 디버깅** 켜기
3. PC와 USB로 연결 → 폰에 뜨는 **"USB 디버깅을 허용하시겠습니까?"** → **이 컴퓨터에서
   항상 허용** 체크 후 허용
4. PC에서 인식 확인:
   ```bash
   adb devices -l
   ```
   기기 시리얼이 `device` 상태로 보이면 정상입니다.

   - **`no permissions (missing udev rules?)`로 나오는 경우 (Linux 전용 문제)** —
     Google 기기(vendor ID `18d1`)에 대한 udev 권한 규칙이 없는 경우입니다:
     ```bash
     echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="18d1", MODE="0666", GROUP="plugdev"' \
       | sudo tee /etc/udev/rules.d/51-android.rules
     sudo udevadm control --reload-rules
     sudo udevadm trigger
     ```
     이후 USB 케이블을 뽑고 다시 꽂은 뒤 폰의 허용 팝업을 다시 확인하세요. (본인 계정이
     `plugdev` 그룹에 속해 있어야 합니다 — `groups` 명령으로 확인.)
   - **`unauthorized`로 나오는 경우** — 폰 화면의 USB 디버깅 허용 팝업을 아직 못 누른
     상태입니다. 케이블을 재연결하고 팝업을 확인하세요.

## 4. 빠른 시작 (기본 경로)

여기까지(0~3번) 환경만 갖춰져 있으면, **아래 네 단계만으로 등록·인식까지 바로
동작합니다** — 임베딩 모델을 직접 구하거나 변환할 필요가 없습니다. 기본 임베딩
모델인 **AuraFace**(Apache 2.0, 상업적 재배포 가능)를 GitHub Release에서 자동으로
받아오기 때문입니다.

```bash
# 1. 클론
git clone https://github.com/wahihi/facekit.git
cd facekit

# 2. 의존성 설치 (example 앱 기준)
cd example
flutter pub get

# 3. 임베딩 모델 다운로드 (AuraFace, ~130MB, 최초 1회만)
bash ../tool/fetch_models.sh

# 4. 실행
flutter run
```

`tool/fetch_models.sh`는 GitHub Release에서 `.tflite` 가중치를 받아 SHA256으로
무결성을 검증한 뒤 `example/assets/models/auraface/`에 둡니다. 이미 파일이 있으면
아무것도 하지 않고 즉시 종료하므로, 몇 번을 다시 실행해도 안전합니다.

`flutter run`이 뜨면 카메라 화면이 나오고, **"등록" 버튼을 눌러 이름을 등록한 뒤
"실시간 인식 시작"을 누르면 바로 재인식이 동작합니다.** 다른 임베딩 모델(ArcFace 등)을
쓰고 싶은 경우나 매니페스트 구조를 더 알고 싶은 경우는 [부록 A](#부록-a--다른-임베딩-모델-쓰기-byom)를,
동봉된 모델들의 라이선스는 [부록 B](#부록-b--라이선스)를 참고하세요.

## 5. 로그 보기 (동작 과정 확인)

```bash
flutter logs
# 또는
adb logcat -s flutter:*
```

파이프라인 핵심 단계(검출/정렬/임베딩/매칭/라이브니스)는 `debugPrint`로 로그를
남기도록 되어 있습니다(`kDebugMode`에서만 동작, release 빌드에는 안 찍힘). 예:
```
[onFrame] face found, score=0.656
[onFrame] liveness: LivenessState.passed
[FacePipeline] detect: 1 face(s)
[FacePipeline] embed: 512-dim vector
[FacePipeline] match: id=나 similarity=0.873 accepted=true
```

더 자세한 파이프라인 내부 로그(정렬 행렬, raw 임베딩 통계 등)가 필요하면
`flutter run --dart-define=FACEKIT_VERBOSE_DEBUG=true`로 켤 수 있습니다 — 평소엔
꺼져 있어 빌드에 전혀 안 들어갑니다.

## 6. 테스트 실행

```bash
flutter test               # 루트(facekit 패키지) 단위 테스트 — 별도로 루트에서 flutter pub get 필요
cd example && flutter test # example 앱 위젯 테스트
```

## 7. 빌드 모드 참고 (release/profile을 쓰고 싶다면)

`flutter run`은 기본적으로 디버그 모드입니다. 성능을 보거나 배포용 APK가
필요하면:

```bash
flutter build apk --debug      # 디버그용 — 가장 빠르게 빌드, 최적화 없음
flutter build apk --profile    # 성능 측정용 — AOT 최적화 적용, 디버그 도구는 일부 제한
flutter build apk --release    # 배포용 — 최대 최적화
```

빌드 결과물 위치는 모드 공통으로:
```
example/build/app/outputs/flutter-apk/app-{debug,profile,release}.apk
```

**기본 임베딩 모델(AuraFace)은 `bundled`/`redistributable: true`라 `--release`
빌드에서도 정상적으로 로드됩니다.** 다만 [부록 A](#부록-a--다른-임베딩-모델-쓰기-byom)를 따라
`arcface_buffalo_l` 같은 연구용(research-tier) 모델로 바꾸면, `ModelManifest.assertLoadable()`이
**release 빌드에서는 코드 레벨로 모델 로드를 차단**합니다(라이선스 보호 목적 —
자세한 동작은 [부록 B](#부록-b--라이선스) 참고). 그 경우 성능을 보려면 `--profile`을
쓰세요 — AOT 최적화는 release와 동일하면서 이 가드에 걸리지 않습니다. 자세한 이유는
[benchmark.md](benchmark.md)에 정리되어 있습니다.

APK를 직접 설치하려면:
```bash
adb install -r build/app/outputs/flutter-apk/app-profile.apk
```
USB 없이 같은 Wi-Fi로 설치하고 싶다면, PC에서 임시 다운로드 서버를 띄우고 폰
브라우저로 받는 방법도 있습니다:
```bash
cd build/app/outputs/flutter-apk
python3 -m http.server 8765 --bind 0.0.0.0
# 폰 브라우저에서 http://<PC의 LAN IP>:8765/app-profile.apk 접속 후 다운로드
```

## 8. 자주 만나는 문제

| 증상 | 원인 / 해결 |
|---|---|
| `Target file "lib/main.dart" not found` | 저장소 루트에서 `flutter run`을 실행했음 — `cd example` 후 실행 |
| 웹(`flutter run -d chrome`)에서 `dart:ffi` 에러 | `tflite_flutter`가 `dart:ffi`를 쓰는데 웹은 이를 지원하지 않음 — 이 SDK는 웹 미지원, Android(또는 iOS/데스크톱)로 실행 |
| 초기화 실패: ... "모델 파일이 없습니다" (앱 첫 화면) | `tool/fetch_models.sh`를 안 돌렸을 가능성이 높음 — 4번(빠른 시작) 참고 |
| BYOM 모델로 바꾼 뒤 `--release` 빌드에서 모델 로드 실패 | 위 7번 참고 — research-tier 모델의 라이선스 가드. `--profile`이나 `--debug` 사용 |
| `adb devices`에 `no permissions` | 3번의 udev 규칙 추가 |
| `adb devices`에 `unauthorized` | 폰 화면의 USB 디버깅 허용 팝업을 못 누른 상태 — 케이블 재연결 후 팝업 확인 |
| 앱 첫 실행 시 `CameraException(Disposed CameraController...)` 로그 한 줄 | `adb install -r`로 실행 중인 앱 위에 덮어 설치할 때 나는 일회성 잡음으로 보임(이전 프로세스 정리 타이밍) — 이후 정상 동작하면 무시 가능, 완전 종료 후 재실행해도 반복되면 별도 확인 필요 |
| Gradle 빌드 중 "Kotlin Gradle Plugin (KGP)" 경고 | `camera_android_camerax` 플러그인 관련 경고, 현재 빌드를 막지 않으므로 무시 가능 |
| `flutter`/`adb` 명령을 못 찾음 | PATH 설정이 현재 셸에 안 로드됨 — `source ~/.bashrc` 또는 새 터미널 열기 |

---

## 부록 A — 다른 임베딩 모델 쓰기 (BYOM)

기본값(AuraFace)이 아닌 다른 임베딩 모델(ArcFace, AdaFace 등)을 직접 구해서 쓰고
싶은 경우입니다. facekit은 임베딩 모델을 매니페스트 기반으로 갈아끼우는 BYOM
구조라, **모델 자체를 리포에 동봉하지 않고** manifest.json + `.tflite`만 정해진
경로에 놓으면 됩니다.

### A.1 예시 — ArcFace(buffalo_l) 준비하기

이미 리포에 `example/assets/models/arcface_buffalo_l/manifest.json`이 있고
BYOM 예시로 유지되고 있습니다. `.tflite` 가중치만 직접 채우면 됩니다.

```bash
ls example/assets/models/arcface_buffalo_l/
# manifest.json만 보이고 .tflite는 없는 게 정상 (.gitignore로 제외됨)
```

1. https://github.com/deepinsight/insightface 에서 `buffalo_l` 모델 팩의
   `w600k_r50.onnx`(또는 동등한 ArcFace R100 가중치)를 받습니다.
   - 비상업 연구용 라이선스입니다 — 상업적으로 쓰면 안 됩니다 (자세한 내용은
     `example/assets/models/arcface_buffalo_l/manifest.json`의 `license` 필드).
2. ONNX를 TFLite로 변환합니다:
   ```bash
   # TODO: 변환 명령어를 여기에 채우세요 (onnx2tf / onnx-tf 등 — 모델·툴 버전에 따라 다름)

   ```
3. 변환된 파일을 정확히 이 경로/이름으로 둡니다:
   ```
   example/assets/models/arcface_buffalo_l/w600k_r50.tflite
   ```
4. `example/lib/main.dart` 상단의 상수를 바꿔 활성 모델을 전환합니다:
   ```dart
   const _embedderDir = 'assets/models/arcface_buffalo_l';
   const _embedderFile = 'w600k_r50.tflite';
   ```
5. 이 파일이 없는 채로 앱을 실행하면 `_setup()`에서 모델 로드가 실패해 화면에
   "초기화 실패: ..." 메시지가 뜹니다 — 정상적인 동작이니 당황하지 마세요.

### A.2 manifest.json 스키마

`ModelManifest.fromJson()`([lib/src/inference/model_manifest.dart](../../lib/src/inference/model_manifest.dart))이
파싱하는 필드입니다:

```jsonc
{
  "name": "모델 식별용 이름",
  "family": "arcface",              // 아래 A.3의 어댑터 선택 키
  "file": "가중치_파일명.tflite",     // manifest.json과 같은 폴더 기준 상대경로
  "input": {
    "width": 112, "height": 112,     // 모델이 기대하는 정사각 입력 크기
    "color": "RGB",                  // "RGB" | "BGR"
    "layout": "NHWC",
    "normalize": {
      "mean": [127.5, 127.5, 127.5], // 채널별 (pixel - mean) / std
      "std": [127.5, 127.5, 127.5]
    }
  },
  "output": {
    "dim": 512,                      // 임베딩 차원
    "l2_normalize": true             // 후처리에서 L2 정규화를 적용할지
  },
  "alignment": {
    "type": "five_point_affine",
    "reference": "arcface_112"       // affine_aligner.dart의 기준 좌표 세트
  },
  "matching": {
    "metric": "cosine",
    "threshold": 0.40,               // CosineMatcher.fromManifest()가 읽는 값
    "threshold_note": "실측 근거를 여기 남길 것"
  },
  "license": {
    "tier": "bundled",               // "bundled" | "research" | "byom" | "licensed"
    "redistributable": false,        // release 빌드에서 로드 가능 여부 (부록 B 참고)
    "source": "출처/원본 저장소",
    "note": "라이선스 조건 설명"
  }
}
```

`input`/`output`/`alignment` 값은 모델마다 다르므로 원본 모델의 학습·전처리
컨벤션을 반드시 확인하고 채우세요 — 예를 들어 AdaFace는 `color: "BGR"`, FaceNet은
112가 아니라 160 입력에 정규화도 다른 방식(prewhiten)을 씁니다.

### A.3 새 family 등록하기 (어댑터 연결)

`family` 값은 `lib/src/embedding/face_embedder.dart`의 `adapterForFamily()`가
어떤 전처리/후처리 어댑터를 쓸지 고르는 키입니다:

```dart
EmbedderAdapter adapterForFamily(String family) {
  switch (family) {
    case 'arcface':
    case 'adaface':
    case 'mobilefacenet':
    case 'auraface':
      return const ArcfaceAdapter();   // 112x112, (pixel-mean)/std 공통 컨벤션
    case 'facenet':
      return const FacenetAdapter();   // 160x160, per-image prewhiten
    default:
      throw ArgumentError('No embedder adapter registered for family "$family"');
  }
}
```

- 기존 `ArcfaceAdapter`/`FacenetAdapter`와 같은 입력 컨벤션을 쓰는 모델이면 —
  AuraFace가 그랬던 것처럼 — **새 `case`만 추가하고 기존 어댑터를 재사용**하면
  됩니다.
- 완전히 다른 전처리(예: 다른 정규화 공식, 추가 출력 텐서 처리 등)가 필요하면
  `lib/src/embedding/adapters/embedder_adapter.dart`의 `EmbedderAdapter`
  인터페이스(`preprocess`/`postprocess`)를 구현하는 새 어댑터를 만들고, 여기
  `switch`에 등록하세요.

## 부록 B — 라이선스

### 동봉된 모델

| 모델 | 라이선스 | 출처 |
|---|---|---|
| BlazeFace short-range (검출) | Apache 2.0 | https://github.com/google/mediapipe |
| MediaPipe Face Landmarker (478점, 라이브니스용) | Apache 2.0 | https://github.com/google/mediapipe |
| AuraFace (glintr100/ResNet100, 기본 임베딩) | Apache 2.0 | [fal/AuraFace-v1](https://huggingface.co/fal/AuraFace-v1) — `.tflite` 가중치 자체는 용량 때문에 리포에 커밋하지 않고 `tool/fetch_models.sh`로 GitHub Release에서 받음 (라이선스 문제로 뺀 게 아님) |

셋 다 상업적 사용이 가능한 라이선스라 별도 BYOM 절차 없이 그대로 동작합니다.
`arcface_buffalo_l` 등 부록 A에서 다루는 추가 모델들은 대부분 비상업 연구용
라이선스라 직접 받아와야 합니다.

### `assertLoadable()` 가드가 차단하는 것

각 모델의 `manifest.json`에는 `license.tier`(`bundled`/`research`/`byom`/`licensed`)와
`license.redistributable`(불리언)이 있습니다.
[`ModelManifest.assertLoadable()`](../../lib/src/inference/model_manifest.dart)이
매 모델 로드 시점(`TfliteFaceEmbedder.fromAsset`/`fromFile`)에 이렇게 검사합니다:

```dart
void assertLoadable({required bool isReleaseBuild}) {
  if (isReleaseBuild && !license.redistributable) {
    throw StateError(
      'manifest "$name": license.redistributable=false (tier=${license.tier.name}, '
      'source=${license.source}) — this model must not be loaded in a release build',
    );
  }
}
```

즉 **`redistributable: false`인 모델은 release 빌드(`kReleaseMode == true`)에서
로드 자체가 코드 레벨로 막힙니다** — 문서로만 "상업 배포 금지"라고 적어두는 게
아니라, 실수로 그런 모델을 넣은 채 배포 빌드를 만들어도 앱이 그 시점에 에러를
던지도록 강제합니다. `--debug`/`--profile` 빌드는 이 가드에 걸리지 않으므로,
연구용 모델로 개발·성능 측정은 계속할 수 있습니다.

`ModelManifest.validate()`도 함께 확인합니다 — `tier: "research"`이면서
`redistributable: true`인 모순된 조합은 애초에 매니페스트 파싱 단계에서
거부됩니다.

---

더 깊은 내용(아키텍처, 벤치마크 방법론)은 [architecture.md](architecture.md),
[benchmark.md](benchmark.md), [liveness.md](liveness.md)를 참고하세요.
