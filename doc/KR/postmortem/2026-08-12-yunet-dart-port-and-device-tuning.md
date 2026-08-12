🇺🇸 [English version](../../EN/postmortem/2026-08-12-yunet-dart-port-and-device-tuning.md)

---

# threshold를 세 번 바꾸고 나서 찾은 진짜 원인 — YuNet Dart 포팅과 실기기 튜닝

**날짜:** 2026-08-12
**계기:** 같은 날 앞서 나온
[2026-08-12-yunet-landmark-order.md](2026-08-12-yunet-landmark-order.md)에서
파이썬 검증까지 끝났다 (YuNet 정렬 기준 AuraFace clean EER 2.5%, SCRFD보다도
좋음). 다음 단계는 명확했다 — 실제 facekit Dart SDK에 포팅하고, BlazeFace를
완전히 대체해서 실기기(Pixel 7)에서 확인하는 것.
**결론(스포일러):** 포팅 자체는 사고 없이 끝났다 — 디코드 공식을
`cv2.FaceDetectorYN`의 실제 출력과 바이트 단위로 대조해서 검증한 뒤
포팅했다. 그런데 실기기에 올리자 검출 threshold를 **세 번**(0.6→0.3→0.45)
고쳐야 했고, 그 과정에서 포팅 자체보다 더 중요한 걸 발견했다: **AuraFace의
매칭 threshold(0.30)가 여전히 "BlazeFace 시절, 정렬 없는 리사이즈" 기준으로
측정된 값이었다**는 것. 이 리포에는 이미 YuNet 정렬 기준 재측정치가
있었는데(`results_yunet.json`, threshold=0.211), 디텍터를 바꾸는 작업에
가려서 반영이 안 돼 있었다.

이 프로젝트는 [Claude Code](https://github.com/anthropics/claude-code)와
협업해서 개발했고, 이 세션도 마찬가지다.

## 1. Dart 포팅 — 디코드 공식을 cv2 기준과 바이트 단위로 맞추기

YuNet의 raw ONNX 출력(`cls_8/16/32`, `obj_8/16/32`, `bbox_8/16/32`,
`kps_8/16/32`, 총 12개 텐서)을 직접 디코드하는 코드는 어디에도 정리된 형태로
없었다 — `cv2.FaceDetectorYN`이 C++ 내부에서 다 처리해버리기 때문에, 실제
grid-offset·exp 기반 bbox·keypoint 디코드 공식은 라이브러리 뒤에 숨어 있다.
포팅에 들어가기 전에 파이썬에서 먼저 검증했다:

1. 같은 160×160 리사이즈 이미지를 `cv2.FaceDetectorYN`과 raw ONNX 양쪽에
   동시에 먹여서, cv2의 출력을 신뢰 가능한 정답으로 삼았다.
2. `opencv/modules/objdetect/src/face_detect.cpp`의 실제 후처리 코드를
   찾아서 공식을 확인: `cx=(col+bbox[0])*stride`, `w=exp(bbox[2])*stride`,
   `score=sqrt(clamp(cls,0,1)*clamp(obj,0,1))`.
3. 이 공식대로 raw 텐서를 직접 디코드한 결과가 cv2의 출력과 **score, bbox,
   5개 keypoint 전부 소수점까지 정확히 일치**했다 (`tool/model_verification/`
   의 검증 스크립트).

이렇게 검증된 raw 값(특정 grid cell의 실제 텐서 값)을 그대로
`test/detection/yunet_decoder_test.dart`의 회귀 테스트에 박아넣었다 — 다음에
누가 이 디코더를 건드리면 "숫자가 그럴듯해 보이는지"가 아니라 "실측값과
정확히 같은지"로 검증되게 하기 위해서다. 출력 텐서 12개의 순서도 미리
확인해뒀다: onnx2tf 변환 과정에서 그래프 출력 순서가 뒤섞이는 걸 실측으로
확인했고 ([YuNetDetector]가 텐서 shape(격자 크기·채널 수)로 런타임에 역할을
재해석하도록 처리, `lib/src/detection/yunet_detector.dart`), cls/obj 두
스코어 텐서는 어느 쪽이 어느 쪽인지 구분할 필요가 없다는 것도 확인했다 —
`sqrt(cls*obj)`는 어차피 대칭이라 순서가 바뀌어도 결과가 같다.

## 2. AffineAligner — 기존 4점 경로는 안 건드리고 5점 네이티브 모드만 추가

BlazeFace는 랜드마크 6점 중 입을 1점만 줘서, `AffineAligner`가 ArcFace의
진짜 5점 기준(입꼬리 2점)을 4점으로 압축해서 썼다. YuNet은 5점을 그대로
주므로 이 압축이 필요 없다 — 하지만 기존 `AffineAligner.align()`을 그냥
5점 전용으로 바꿔버리면, `landmarks.length==5`인 기존 테스트 픽스처들(4점
경로를 테스트하려고 일부러 5번째 점을 더미로 채운 것들, 예:
`_identityFace()`)의 동작이 조용히 바뀌어버린다. 그래서 랜드마크 개수로
자동 분기하는 대신, `nativeFivePoint`라는 명시적 플래그를 추가해서 호출자가
어느 경로를 쓸지 선언하게 했다 (`AffineAligner.arcface112(nativeFivePoint:
true)`). 기본값은 `false`라서 기존 BlazeFace 경로/테스트는 전부 그대로다.

`nativeFivePoint: true`일 때는 YuNet의 원본 출력 순서를 **그대로**
`arcface112Ref`에 정렬해서 쓴다 — 앞선 포스트모템에서 이미 확인했듯,
"이름이 다르니 재배열해야 한다"는 판단이 오히려 틀렸기 때문이다. 이걸
코드 주석뿐 아니라 회귀 테스트로도 못박아뒀다
(`test/alignment/affine_aligner_test.dart`의
`"correcting" by swapping the two eye points does NOT round-trip to
identity` 테스트) — 앞선 포스트모템의 "다음엔 회귀 테스트로도 남겨야
한다"는 교훈을 바로 실천한 것.

## 3. 실기기 1차 — 로그가 비어있다 (코드 문제 아님)

`flutter run --dart-define=FACEKIT_VERBOSE_DEBUG=true | tee log.test`로
Pixel 7에 처음 올렸을 때, "Lost connection to device"가 Dart VM Service
연결 직후 바로 떴다. 로그 파일이 44줄(2.4KB)밖에 안 됐다 — `debugPrint`가
단 한 줄도 안 찍힌 채 끊긴 것. 화면에서는 앱이 정상적으로 돌고 있는 게
보였으므로(라이브니스 문구, 사각형 오버레이), USB/adb 디버그 브리지만
끊긴 것으로 판단하고 재시도했다. 여기서 얻은 습관적 교훈: **로그 내용을
분석하기 전에 `wc -l`부터 확인**하면, "코드가 잘못됨"과 "애초에 아무것도
안 찍힘"을 즉시 구분할 수 있다.

## 4. 실기기 2차 (threshold 0.6) — flicker로 라이브니스가 못 끝남

YuNet 매니페스트의 `score_threshold`는 실기기 검증 없이 0.6으로
잠정적으로 넣어둔 값이었다 (CLAUDE.md 규칙상 새로 튜닝해야 하는 값이라는
걸 이미 인지하고 있었다). 재시도한 로그에서:

```
[onFrame] face found, score=0.686
[onFrame] face lost (>500ms, resetting liveness)
[onFrame] face found, score=0.624
[onFrame] face lost (>500ms, resetting liveness)
...
```

`found` 스코어가 전부 0.608~0.711 — threshold(0.6) 바로 위 경계에 몰려
있었다. `found` 몇 프레임 뒤엔 항상 `lost`가 뒤따랐고, 라이브니스는
`pending`에서 한 번도 못 벗어났다. 등록 자체가 시도조차 안 됐다.

## 5. 실기기 3차 (threshold 0.3) — flicker는 고쳤는데 새 문제가 생겼다

threshold를 0.3으로 낮추자 flicker는 완전히 사라졌다 — enroll 성공, 이후
identify가 프레임마다 자동으로 돌았다. 하지만 로그를 자세히 보니 두 가지
문제가 새로 보였다:

- **가장자리 오검출**: bbox가 이미지 경계에 정확히 클리핑된 경우가
  나타났다 (`bbox=(564.6,0.0)-(720.0,123.1)`, 랜드마크 좌표도 720에
  클리핑됨) — 배경/모서리를 "얼굴"로 잘못 인식한 것으로 보인다. 나중엔
  `detect: 2 face(s)`까지 나왔다.
- **같은 사람인데 유사도가 요동침**: `match: id=나 similarity=...`가
  프레임마다 0.147~0.826 사이를 오갔다. `poseGate=ok`가 매번 찍혔으므로
  (회전 -9°~+2°, 전부 정상) 랜드마크가 흔들리는 문제는 아니었는데도,
  AuraFace의 `matching.threshold=0.30` 기준으로 여러 프레임이 **진짜
  본인인데 거부**됐다(`similarity=0.147, 0.149, 0.186, 0.205, 0.233,
  0.257` 등 다수).

## 6. 여기서 발견한 진짜 원인 — 매칭 threshold가 낡은 값이었다

유사도가 왜 이렇게 넓게 흩어지는지 조사하다가,
`example/assets/models/auraface/manifest.json`의 `matching.threshold=0.30`
이 **"BlazeFace 4점 압축, 정렬 없는 리사이즈" 기준으로 2026-08-11에
측정된 값**(EER 10.0%)이라는 걸 다시 확인했다. 그런데 지금은 YuNet이
진짜 5점 정렬을 제공하고 있고, 이 리포엔 이미 그 기준으로 재측정한 결과가
있었다 — `tool/model_verification/results_yunet.json`
(`compare_with_yunet.py`로 생성, 2026-08-12-yunet-landmark-order.md 작업
중 만들어짐):

```json
"auraface": {
  "threshold_clean_eer": 0.21119210124015808,
  "eer_clean": 0.025
}
```

즉 **디텍터를 바꾸는 작업을 하면서, 그 디텍터의 출력에 의존하는 다른
threshold(매칭 threshold)까지 같이 재검토해야 한다는 걸 깜빡했던
것**이다. 이미 측정까지 끝나 있었는데 반영이 안 된 상태로 실기기까지
올라간 셈이다.

## 7. 실기기 4차 (yunet 0.45 + auraface 0.211) — 둘 다 해결

두 값을 동시에 바꿨다: `score_threshold` 0.3→0.45(가장자리 오검출 억제),
`matching.threshold` 0.30→0.211(YuNet 정렬 기준 재측정치 채택). 재시도
결과:

- 가장자리 클리핑 bbox, `2 face(s)` 동시 검출 — **완전히 사라짐**.
- identify 26번 전부 `id=나, accepted=true` — **단 한 번도 오거부 없음**.
  유사도 자체는 여전히 0.271~0.871로 넓게 퍼져 있었지만, 새 threshold
  기준으로는 전부 안전하게 통과.

## 8. Impostor 테스트 — 모니터 속 타인 사진

threshold를 낮췄으니(0.30→0.211) 반대 방향(타인을 본인으로 잘못
인식)도 확인이 필요했다. 모니터에 일론 머스크 사진을 띄우고 카메라로
비췄다:

- **본인 얼굴**(화면을 거의 채우는 큰 bbox): `id=나` 3회 정상 인식
  (유사도 0.359/0.380/0.506) + 1회 오거부(0.196, 이미 알고 있던 유사도
  변동폭 문제).
- **모니터 사진**(구석에 작게 잡히는 bbox, 약 93×62px, 변환행렬
  스케일 a≈1.65로 다른 검출들과 확연히 다름): `id=null` 2회 정상 거부
  (유사도 0.123/0.147). 한 프레임에서는 본인 얼굴과 동시에
  `detect: 2 face(s)`로 잡혔는데도, 스코어가 더 높았던 모니터 사진 쪽이
  정확히 거부됐다.

타인 오인식은 한 번도 없었다.

## 9. 정리 — 인과관계

```
질문: 파이썬 검증(EER 2.5%)까지 끝난 YuNet을 Dart로 옮기고 실기기에서 확인하면?
   └→ 디코드 공식을 cv2.FaceDetectorYN 기준과 바이트 단위로 대조 후 포팅, 회귀 테스트로 고정
   └→ AffineAligner에 nativeFivePoint 모드 추가 (기존 4점 경로는 그대로 둠)
   └→ 실기기 1차: USB 연결 끊김, 로그 44줄(비어있음) — 코드 문제 아님, 재시도
   └→ 실기기 2차 (threshold 0.6): score가 threshold 바로 위에 몰려 flicker, 라이브니스 pending에서 정체
   └→ 실기기 3차 (threshold 0.3): flicker 해결됐지만 가장자리 오검출 + 진짜 매치 오거부(유사도 0.147~0.30)
        └→ 조사: AuraFace matching.threshold(0.30)가 "정렬 없는 리사이즈" 시절 측정값이었음을 재확인
        └→ 이미 리포에 있던 YuNet 정렬 기준 재측정치(0.211, EER 2.5%) 발견 — 반영 누락 상태였음
   └→ 실기기 4차 (yunet 0.45 + auraface 0.211): 오검출 사라짐, identify 26/26 정상 인식
   └→ impostor 테스트 (모니터 속 타인 사진): 2/2 정상 거부, 오인식 없음
```

## 10. 배운 것

- **디텍터를 바꾸면, 그 디텍터의 출력에 의존하는 하위 threshold(매칭
  threshold 포함)도 전부 재검토 대상이다.** 이번엔 재측정치가 이미 리포에
  있었는데도, "디텍터 교체"라는 작업 범위에만 집중하다가 실기기 디버깅
  도중에야 발견했다. 파이프라인의 한 단계를 바꾸는 작업은, 그 단계의
  출력을 소비하는 모든 다른 단계의 가정도 같이 점검해야 한다.
- **"flicker가 없어졌다"는 성공 신호 하나만 보고 threshold 변경을
  확정하면 안 된다.** 같은 로그를 가장자리 오검출 여부까지 같이 봐야
  새로 생긴 실패 모드(이번엔 배경 오검출)를 놓치지 않는다.
- **로그가 비어있는 경우(연결 끊김)와 코드가 실제로 잘못돼서 에러가
  찍히는 경우는 겉보기엔 둘 다 "잘 안 됨"으로 보이지만 원인이 완전히
  다르다.** 로그 내용을 파고들기 전에 `wc -l`로 줄 수부터 확인하는 게
  삽질을 줄인다.
- **정적 사진 기반 벤치마크(LFW)는 필요하지만 충분하지 않다.** 가장자리
  오검출과 프레임 간 유사도 흔들림 둘 다 실기기 영상에서만 드러났다 —
  이 프로젝트의 이전 포스트모템들에서 반복된 결론이 이번에도 유효했다.
- **`poseGate=ok`가 실기기 세션 내내 한 번도 안 걸린 건 그 자체로
  의미 있는 신호다.** BlazeFace 시절엔 실기기 프레임의 24~42%가 회전
  이상으로 걸렸는데(2026-08-03 포스트모템), YuNet은 이번 세션 전체에서
  단 한 번도 안 걸렸다 — 랜드마크 기하학적 안정성은 확실히 개선됐다는
  뜻이고, 남은 유사도 흔들림(0.27~0.87)은 회전/스케일 문제가 아니라 다른
  원인(조명, 실시간 프레임 노이즈 등)일 가능성이 높다는 게 이번에 새로
  확인된 부분 — 아직 못 푼 문제로 남겨둔다.

## 부록: 바뀐 것들

- `lib/src/detection/yunet_decoder.dart`, `yunet_detector.dart` (신규) —
  순수 디코드 함수 + `YuNetDetector`. 디코드 공식은 cv2 기준 실측값을
  회귀 테스트(`test/detection/yunet_decoder_test.dart`)에 그대로 박아둠.
- `lib/src/alignment/affine_aligner.dart` — `nativeFivePoint` 모드 추가
  (기본값 `false`, 기존 4점 경로/테스트 전부 그대로).
- `assets/models/yunet_160/` — YuNet MIT 모델(160×160 고정 입력, 동봉
  가능) + manifest.json. `score_threshold`는 실기기 튜닝 끝에 0.45로
  확정.
- `example/assets/models/auraface/manifest.json` — `matching.threshold`
  0.30→0.211 (YuNet 정렬 기준 재측정치).
- `example/lib/main.dart` — 기본 디텍터를 `BlazeFaceDetector`에서
  `YuNetDetector`로 교체. `BlazeFaceDetector` 코드 자체는 폴백용으로
  그대로 남겨둠.
- `test/detection/yunet_decoder_test.dart`,
  `test/detection/yunet_smoke_test.dart`,
  `test/alignment/affine_aligner_test.dart`(네이티브 5점 회귀 테스트
  추가).
