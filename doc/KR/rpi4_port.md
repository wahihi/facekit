🇺🇸 [English version](../EN/rpi4_port.md)

---

# Raspberry Pi 4 포팅 (`feature/rpi4`) — 2026-10-22

`feature/linux-port`(Q6A, Radxa Dragon QCS6490)에서 분기. 같은 헤드리스 실시간 인식
(`facekit_cli`, `facekit_realtime`)을 Raspberry Pi 4 + 64-bit Raspberry Pi OS + Logitech C270
(Q6A와 같은 USB 카메라)에서 돌리고, AuraFace를 그대로 쓸 수 있는지 실측으로 판단했다.

라이브러리·CLI 코드 변경 없음(Pi4도 aarch64). 바뀐 건 빌드/실행 환경뿐이고 `tool/rpi4/`의
`setup.sh`(환경 구축), `bench.sh`(실측)로 고정했다.

## 실행 방법

```bash
git clone --branch feature/rpi4 https://github.com/wahihi/facekit.git && cd facekit
./tool/rpi4/setup.sh          # 1회. TFLite C 빌드 ~1시간. 재실행 시 완료 단계는 건너뜀
source tool/rpi4/env.sh
./tool/rpi4/bench.sh          # 골든 벡터 + bench(스레드 1/4) + 실카메라 30초
```

## 실측 결과 (2026-10-22)

환경: Raspberry Pi 4 Model B Rev 1.5, RAM 3.7GB, Cortex-A72 ×4 @1.8GHz (governor ondemand),
Debian 13 (trixie), kernel 6.18.50+rpt-rpi-v8, Dart 3.13.6, TFLite C v2.21.0-rc0 (Pi에서 빌드),
YuNet160 + AuraFace R100 fp16, XNNPACK CPU, n=30 median.

| 항목 | Pi 4 | Q6A (참고) |
|---|---|---|
| 골든 벡터 (안드로이드 대비 cosine ≥ 0.999) | **PASS** | PASS |
| 1스레드: detect / align / embed / total | 84.8 / 3.3 / 3089.8 / **3182.8ms** | 18.2 / 1.1 / 701.7 / 721.2ms |
| 4스레드: detect / align / embed / total | 81.9 / 3.2 / 1169.8 / **1254.3ms** | 24.0 / 1.1 / 325.5 / 349.8ms |
| 실카메라 30초 (C270, 640×480 YUYV, 4스레드) | 처리 21프레임, 매칭 21, 불일치 0, 손상 0 | — |
| 메모리 증가 (30초) | +0.6MB | — |
| 온도 / 스로틀 | 53.6°C / 0x0 | — |

`dropped_frames`(638)는 처리 중 들어온 카메라 프레임을 건너뛴 수로, 설계상 정상(최신 프레임만 처리).

## 판단

- **기능: 통과.** 골든 벡터 일치, 실카메라 인식 동작, 메모리 누수 없음.
- **속도: AuraFace R100은 Pi4 실시간 용도에 부적합.** 4스레드 1254ms(≈0.8fps)로 Q6A 기준선
  (300ms대 제품 가능 / 800ms 이상 다른 경로 필요)을 넘는다. Q6A 대비 약 3.6배 느림.
- **Pi4 포팅은 여기서 종료.** Pi4는 원래 사업 타깃(RK3588급)이 아니므로, Pi4만을 위해 경량
  임베더를 붙이지 않는다.
- SFace(128D, Apache 2.0) 등 경량 임베더는 저사양 대응(갤럭시 A·RK3588) 작업의 일부로 별도
  판단하며, 그때 Pi4를 가장 느린 기준 보드로 재사용한다.

## 이번에 겪은 문제

- `tool/model_verification/me_1.jpg`가 `.gitignore` 대상이라 clone에 없어서 bench/골든 테스트가
  "이미지 파일을 찾을 수 없음"으로 실패. 같은 파일이 `example/assets/golden/me_1.jpg`로 추적되고
  있어, `setup.sh`가 없으면 복사하도록 수정.
- Pi의 git push는 GitHub 비밀번호 인증이 막혀 있어 SSH 키(ed25519)로 설정.

## 남은 검증

Q6A 포팅 완료 시 남긴 항목(`linux_port_scope.md` 2026-10-21)과 같은 두 가지는 Pi4에서도 미확인이다.

1. **타인 및 얼굴 없음 상황의 식별 거부**: 이번 30초 실행은 등록 인물로 21건 모두 매칭됐다.
   다른 사람 앞에서 `accepted=false`, 얼굴이 없을 때 식별 결과가 나오지 않는지는 확인하지 않았다.
2. **SIGINT(Ctrl+C) 정상 종료**: `--seconds`로만 종료했고 Ctrl+C 경로는 실기기에서 확인하지 않았다.

Pi4 포팅을 종료하는 판단에는 영향이 없으므로, Q6A와 함께 리눅스 공통 후속 검증으로 둔다.
