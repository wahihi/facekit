> **완료 (2026-08-13, 새 PC):** 이 조사는 끝났다. 결과와 인과관계는
> [doc/KR/postmortem/2026-08-13-landmark-jitter-frame-fill.md](../../doc/KR/postmortem/2026-08-13-landmark-jitter-frame-fill.md)
> 참고. 요약: 개선판 스크립트가 완주 못 했던 진짜 이유는 환경 문제가
> 아니라 `crop()`+`add_noise()`의 OOM 버그(margin^2 비례 메모리, margin=12
> 에서 시행당 ~14GB)였다. `sample_noisy_input()`으로 고쳐서(메모리
> O(160×160) 고정, 기존 구현과 픽셀 단위 동일 출력 검증됨) 6단계 sweep을
> 완주시켰고, 사진 3장 모두에서 가설이 강하게 재현됐다(얼굴이 프레임의
> ~8%일 때 jitter가 ~91%일 때보다 160~288배 큼). 아래는 그 당시 남긴
> 원본 인계 노트, 기록용으로 보존.

# 랜드마크 jitter 조사 인계 노트 (2026-08-13, PC 교체로 중단)

## 배경
2026-08-12 포스트모템(YuNet Dart 포팅,
`doc/KR/postmortem/2026-08-12-yunet-dart-port-and-device-tuning.md`)에서
**미해결로 남긴 문제**: 실기기에서 같은 사람인데 유사도가 0.27~0.87로 넓게
흔들리는 현상. 회전/스케일(`poseGate`)은 매 프레임 정상이었기 때문에 원인
미상으로 남겨뒀었음.

## 이번 세션에서 한 일

1. **로그 분석**: `example/log.txt`(실기기 세션 로그, TEMP DEBUG LANDMARKS
   포함)를 분석해서, 유사도 흔들림이 detect score(-0.24)보다
   AffineAligner가 fit한 scale/rotation과 훨씬 강하게 상관(-0.7~-0.8)된다는
   걸 발견. bbox는 거의 그대로인데 눈-눈 랜드마크 거리가 24% 뛴 프레임 쌍도
   하나 확인.
2. **가설 수립**: `lib/src/image/image_converter.dart`의 `resizeNearest`가
   카메라 프레임 전체를 고정 160×160으로 욱여넣기 때문에(
   `lib/src/detection/yunet_detector.dart`), 얼굴이 프레임에서 차지하는
   비율이 작을수록 얼굴에 할당되는 입력 픽셀 수가 적어져서, 평범한 프레임
   간 센서 노이즈가 랜드마크를 얼굴 크기 대비 크게 흔든다는 가설. (사용자가
   "휴대폰을 얼굴이 프레임을 꽉 채우도록 들면 더 안정적이더라"라고
   경험적으로 보고한 것과 부합)
3. **검증 스크립트 작성**: `tool/model_verification/landmark_jitter_ab.py`
   — `yunet_detector.dart`/`yunet_decoder.dart`의 전처리·디코드 공식을
   파이썬으로 바이트 단위로 재구현, 실제
   `assets/models/yunet_160/yunet_160.tflite`로 테스트
   (`cv2.FaceDetectorYN`이 아님 — 입력 크기/파이프라인이 다름). 사진
   한 장에서 얼굴을 찾고, 그 주변으로 여러 margin(얼굴이 crop에서 차지하는
   비율)의 "가상 카메라 crop"을 만든 뒤, 각 crop에 노이즈(subpixel shift +
   Gaussian) 를 N회 독립 적용해서 160×160으로 리사이즈·디코드하고,
   margin별 랜드마크 jitter(눈 간 거리로 정규화)를 비교.
4. **1차 결과** (`tool/model_verification/results_landmark_jitter_ab.json`,
   10:40 생성) — 당시엔 margin 2단계(small/large)짜리 구식 버전으로 실행한
   결과:
   - small(얼굴이 crop의 작은 비율) jitter=0.0103
   - large(얼굴이 crop을 꽉 채움) jitter=0.0428
   - **가설과 정반대**: 얼굴이 프레임을 더 채울수록 오히려 jitter가 4배
     더 컸음.
5. **스크립트 개선** (10:42, json보다 나중) — margin을 2단계에서 6단계
   (12,8,5,3,1.8,1.1)로 촘촘하게 나누고, 넓은 margin이 원본 사진 밖으로
   나가는 경우를 위한 reflect-padding 로직 추가. **이 개선판으로는 아직
   한 번도 끝까지 실행하지 못한 채 VSC가 꺼짐.**

## 이번 인계 시도에서 확인한 것 (다른 PC 필요 이유)

- 개선판 스크립트를 재실행 시도 → 240회 TFLite 추론(margin 6 × trial 40)이
  120초 타임아웃을 넘겨 백그라운드로 전환 → "실패(exit 1)"로 보고됐지만,
  실제로는 `grep`이 매치를 못 찾아서 난 가짜 실패 신호였음.
- 확인 결과: `results_landmark_jitter_ab.json`은 갱신 안 됐고(여전히 10:40
  버전), 돌던 python 프로세스도 없었음 → **실행이 끝까지 완주하지 못하고
  중간에 죽음.** 에러 메시지/트레이스백도 안 남아서 원인 불명(리소스
  문제로 추정, OOM 여부 등은 확인 못 함).
- 축소판(`--n-trials 3 --margins 12,1.1`)으로 재현 시도하던 중, Bash 툴
  자체가 `Tool permission request failed: AbortError: Stream closed`를
  반복적으로 던지기 시작 → **환경/호스트 연결 문제로 판단, 여기서 막힘.**

## 파일 상태 (git 미추적 — PC 옮길 때 반드시 같이 옮길 것)

| 파일 | 상태 |
|---|---|
| `example/log.txt` | 실기기 세션 로그. 조사의 원본 데이터. |
| `tool/model_verification/landmark_jitter_ab.py` | 개선판(margin 6단계 sweep). **아직 실행 검증 안 됨.** |
| `tool/model_verification/results_landmark_jitter_ab.json` | 구식 2단계(small/large) 결과. 개선판 실행 후 덮어써야 함. |

이 3개는 `git add` 안 된 untracked 파일이라 그냥 `git clone`/`git pull`로는
새 PC에 안 딸려간다. 수동 복사하거나 WIP 커밋으로 옮길 것.

## 다음 PC에서 이어서 할 일

1. 위 3개 파일을 새 PC로 옮긴다.
2. 아래 명령을 **끝까지 완주시켜서** 6단계 sweep 결과를 확보한다 (이번
   PC에서는 완주 자체가 안 됐으므로, 타임아웃을 넉넉히 잡고 완주 여부부터
   확인할 것):
   ```
   cd tool/model_verification
   python3 landmark_jitter_ab.py --image me_1.jpg \
       --tflite ../../assets/models/yunet_160/yunet_160.tflite
   ```
3. sweep 결과로 "얼굴이 프레임을 채울수록 jitter가 더 크다"는 1차 결과
   (가설과 반대)가 재현되는지 확인한다. 재현되면:
   - `crop()`/`add_noise()` 순서나 margin 정의에 버그가 없는지 재점검.
     특히: `resizeNearest`는 nearest-neighbor라 노이즈를 평균으로 지워주지
     않으므로, 넓은 margin(많이 다운샘플링)일수록 오히려 노이즈가 sparse
     subsampling으로 인해 덜 반영될 수 있다 — 이 방향의 설명이 맞는지
     확인 필요.
   - `me_2.jpg`, `other_1.jpg`로도 같은 sweep을 돌려서 사진 한 장짜리
     우연이 아닌지 교차검증.
4. 결론이 서면 `doc/KR/postmortem/2026-08-13-...md` (+ `doc/EN/` 대응판)
   형식으로 정리한다 — 기존 포스트모템들과 같은 컨벤션(README 링크,
   인과관계 다이어그램, "배운 것" 섹션).
