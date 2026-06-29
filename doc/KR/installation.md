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

## 3. 저장소 클론

```bash
git clone https://github.com/wahihi/facekit.git
cd facekit
```

## 4. 의존성 설치

facekit은 패키지(루트)와 example 앱, 두 개의 독립된 Flutter 프로젝트가 한 저장소에
있습니다 — **둘 다** 의존성을 받아야 합니다.

```bash
flutter pub get            # 루트(facekit 패키지 본체)
cd example
flutter pub get            # example 앱
cd ..
```

## 5. 임베딩 모델 준비 (BYOM) — 건너뛰면 앱이 못 뜹니다

facekit은 검출 모델(BlazeFace)과 라이브니스용 랜드마크 모델만 동봉하고, **임베딩 모델은
저장소에 들어있지 않습니다** (라이선스 정책 — [README.md](../../README.md#license--model-policy-byom)
참고). example 앱은 기본적으로 ArcFace(`buffalo_l`)를 쓰도록 되어 있는데, 이 가중치를
직접 받아서 넣어줘야 합니다.

```bash
ls example/assets/models/arcface_buffalo_l/
# manifest.json만 보이고 .tflite는 없는 게 정상 (.gitignore로 제외됨)
```

1. https://github.com/deepinsight/insightface 에서 `buffalo_l` 모델 팩의
   `w600k_r50.onnx`(또는 동등한 ArcFace R100 가중치)를 받습니다.
   - 비상업 연구용 라이선스입니다 — 상업적으로 쓰면 안 됩니다 (자세한 내용은
     `example/assets/models/arcface_buffalo_l/manifest.json`의 `license` 필드).
2. ONNX를 TFLite로 변환합니다 (`onnx2tf`, `onnx-tf` 등 — 변환 방법은 모델/툴 버전에
   따라 다르므로 이 문서에서는 다루지 않습니다).
3. 변환된 파일을 정확히 이 경로/이름으로 둡니다:
   ```
   example/assets/models/arcface_buffalo_l/w600k_r50.tflite
   ```
4. 이 파일이 없는 채로 앱을 실행하면 `_setup()`에서 모델 로드가 실패해 화면에
   "초기화 실패: ..." 메시지가 뜹니다 — 정상적인 동작이니 당황하지 마세요.

> AdaFace 등 다른 임베딩 모델로 바꾸고 싶다면 `lib/src/embedding/adapters/` 와
> [doc/EN/architecture.md](architecture.md)를 참고하세요. 모델 자체를 안 쓰고 코드
> 구조만 보고 싶다면 이 단계는 생략하고 6번으로 가도 되지만, 그 경우 example 앱은
> 카메라 프리뷰까지는 뜨고 등록/인식 단계에서 동작하지 않습니다.

## 6. 빌드

`example/` 디렉터리에서 실행합니다.

```bash
cd example
```

### 6-1. 개발 중 — 기기에 바로 실행 (가장 많이 쓰는 명령)

```bash
flutter run
```

여러 기기가 연결돼 있으면 `flutter devices`로 확인 후 `-d <기기ID>`로 지정합니다.
이 명령은 디버그 모드로 빌드해 바로 실행하고, hot reload(`r` 키)도 됩니다.

### 6-2. APK만 빌드 (설치 파일만 필요할 때)

```bash
flutter build apk --debug      # 디버그용 — 가장 빠르게 빌드, 최적화 없음
flutter build apk --profile    # 성능 측정용 — AOT 최적화 적용, 디버그 도구는 일부 제한
flutter build apk --release    # 배포용 — 최대 최적화
```

빌드 결과물 위치는 모드 공통으로:
```
example/build/app/outputs/flutter-apk/app-{debug,profile,release}.apk
```

> ⚠️ **`--release`는 이 예제에서 실행 시점에 실패합니다.** ArcFace 데모 모델의
> manifest가 `redistributable: false`라서, `ModelManifest.assertLoadable()`이
> **release 빌드에서는 코드 레벨로 모델 로드를 차단**합니다(라이선스 보호 목적,
> [lib/src/inference/model_manifest.dart](../../lib/src/inference/model_manifest.dart)).
> 성능을 보려면 `--profile`을 쓰세요 — AOT 최적화는 release와 동일하면서 이 가드에
> 걸리지 않습니다. 자세한 이유는 [benchmark.md](benchmark.md)에 정리되어 있습니다.

## 7. 기기에 설치하기

### 7-1. 휴대폰에서 개발자 옵션 / USB 디버깅 켜기

1. **설정 → 휴대전화 정보(기기 정보)** → **빌드 번호**를 7번 연속 탭
2. **설정 → 시스템 → 개발자 옵션** → **USB 디버깅** 켜기
3. PC와 USB로 연결 → 폰에 뜨는 **"USB 디버깅을 허용하시겠습니까?"** → **이 컴퓨터에서
   항상 허용** 체크 후 허용

### 7-2. PC에서 인식 확인

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

### 7-3. APK 설치

USB로 연결된 상태라면:
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

## 8. 로그 보기 (동작 과정 확인)

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

## 9. 테스트 실행

```bash
flutter test               # 루트(facekit 패키지) 단위 테스트
cd example && flutter test # example 앱 위젯 테스트
```

## 10. 자주 만나는 문제

| 증상 | 원인 / 해결 |
|---|---|
| `Target file "lib/main.dart" not found` | 저장소 루트에서 `flutter run`을 실행했음 — `cd example` 후 실행 |
| 웹(`flutter run -d chrome`)에서 `dart:ffi` 에러 | `tflite_flutter`가 `dart:ffi`를 쓰는데 웹은 이를 지원하지 않음 — 이 SDK는 웹 미지원, Android(또는 iOS/데스크톱)로 실행 |
| `--release` 빌드에서 모델 로드 실패 | 위 6-2 참고 — BYOM 데모 모델의 라이선스 가드. `--profile`이나 `--debug` 사용 |
| 초기화 실패: ... (앱 첫 화면) | 5번(BYOM 모델 준비)을 안 했을 가능성이 높음 — `.tflite` 파일 위치 확인 |
| `adb devices`에 `no permissions` | 7-2의 udev 규칙 추가 |
| `adb devices`에 `unauthorized` | 폰 화면의 USB 디버깅 허용 팝업을 못 누른 상태 — 케이블 재연결 후 팝업 확인 |
| 앱 첫 실행 시 `CameraException(Disposed CameraController...)` 로그 한 줄 | `adb install -r`로 실행 중인 앱 위에 덮어 설치할 때 나는 일회성 잡음으로 보임(이전 프로세스 정리 타이밍) — 이후 정상 동작하면 무시 가능, 완전 종료 후 재실행해도 반복되면 별도 확인 필요 |
| Gradle 빌드 중 "Kotlin Gradle Plugin (KGP)" 경고 | `camera_android_camerax` 플러그인 관련 경고, 현재 빌드를 막지 않으므로 무시 가능 |
| `flutter`/`adb` 명령을 못 찾음 | PATH 설정이 현재 셸에 안 로드됨 — `source ~/.bashrc` 또는 새 터미널 열기 |

---

더 깊은 내용(아키텍처, 모델 교체 방법, 벤치마크 방법론)은 [architecture.md](architecture.md),
[benchmark.md](benchmark.md), [liveness.md](liveness.md)를 참고하세요.
