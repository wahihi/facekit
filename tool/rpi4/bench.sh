#!/usr/bin/env bash
# Raspberry Pi 4 실측 — AuraFace 를 그대로 쓸지, SFace 등 경량 임베더가 필요한지 판단할 숫자를 만든다.
# Raspberry Pi 4 measurement — produces the numbers for "keep AuraFace or add a lighter embedder".
#
# 하는 일 (Q6A 의 S4/S5b/S6 DoD 와 같은 항목):
#   1. 시스템 정보 (보드, 커널, RAM, CPU governor, 온도, 스로틀 상태)
#   2. 정확성: packages/facekit_cli 골든 벡터 테스트 (arm64 에서도 안드로이드 값과 일치하는지)
#   3. 속도: facekit_cli --bench 30, 스레드 1 / 4 (detect / align / embed / total ms)
#   4. 실카메라: facekit_realtime 30초 (--no-camera 로 생략 가능)
# 결과는 tool/rpi4/results/<시각>/ 에 저장되고, 마지막에 summary.txt 를 출력한다.
#
# Usage:
#   source tool/rpi4/env.sh
#   ./tool/rpi4/bench.sh                    # 카메라 자동 탐색
#   ./tool/rpi4/bench.sh --device /dev/video0
#   ./tool/rpi4/bench.sh --no-camera
#   ./tool/rpi4/bench.sh --seconds 300      # 실카메라 장시간

set -uo pipefail   # -e 없음: 한 단계가 실패해도 나머지 측정은 계속하고 summary 에 기록

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cli_dir="${repo_root}/packages/facekit_cli"

device=""
camera=1
seconds=30
n_bench=30
while (( $# )); do
  case "$1" in
    --device)    device="$2"; shift 2 ;;
    --no-camera) camera=0; shift ;;
    --seconds)   seconds="$2"; shift 2 ;;
    --bench)     n_bench="$2"; shift 2 ;;
    -h|--help)   sed -n 2,20p "$0"; exit 0 ;;
    *) echo "알 수 없는 인자: $1" >&2; exit 64 ;;
  esac
done

if [[ -f "${script_dir}/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${script_dir}/env.sh"
fi
command -v dart >/dev/null || { echo "dart 없음 — 먼저 ./tool/rpi4/setup.sh" >&2; exit 1; }
[[ -f "${FACEKIT_TFLITE_C_LIBRARY:-}" ]] || { echo "FACEKIT_TFLITE_C_LIBRARY 가 없음 — 먼저 ./tool/rpi4/setup.sh" >&2; exit 1; }

out="${script_dir}/results/$(date +%Y%m%d-%H%M%S)"
mkdir -p "${out}"
summary="${out}/summary.txt"
: > "${summary}"
say() { echo "$*" | tee -a "${summary}"; }

detector="${repo_root}/assets/models/yunet_160"
embedder="${repo_root}/example/assets/models/auraface"
image="${repo_root}/tool/model_verification/me_1.jpg"

thermal() {
  local t thr=""
  t="$(awk '{printf "%.1f", $1/1000}' /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo '?')"
  command -v vcgencmd >/dev/null && thr="$(vcgencmd get_throttled 2>/dev/null)"
  echo "temp=${t}C ${thr}"
}

# --- 1. 시스템 정보 ---------------------------------------------------------
say "=== facekit rpi4 bench  $(date '+%F %T') ==="
say "board:    $(tr -d '\0' < /proc/device-tree/model 2>/dev/null)"
say "kernel:   $(uname -srm)"
say "os:       $(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-?}")"
say "ram:      $(awk '/MemTotal/ {printf "%.1f GB", $2/1048576}' /proc/meminfo)"
say "cpu:      $(nproc) cores, max $(( $(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null || echo 0) / 1000 )) MHz, governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
say "dart:     $(dart --version 2>&1)"
say "git:      $(git -C "${repo_root}" rev-parse --abbrev-ref HEAD) @ $(git -C "${repo_root}" rev-parse --short HEAD)"
say "thermal:  $(thermal)"
say ""

# --- 2. 정확성 (골든 벡터) --------------------------------------------------
say "--- [2] golden vector test (packages/facekit_cli) ---"
if (cd "${cli_dir}" && dart test test/golden_vector_test.dart) > "${out}/golden_test.log" 2>&1; then
  say "PASS  ($(tail -1 "${out}/golden_test.log"))"
else
  say "FAIL  — ${out}/golden_test.log 확인"
fi
say ""

# --- 3. 속도 (스레드 1 / 4) -------------------------------------------------
say "--- [3] facekit_cli --bench ${n_bench}  (YuNet160 + AuraFace R100 fp16, XNNPACK CPU) ---"
for t in 1 4; do
  json="${out}/bench_t${t}.json"
  if (cd "${cli_dir}" && dart run bin/facekit_cli.dart \
        --image "${image}" --detector "${detector}" --embedder "${embedder}" \
        --bench "${n_bench}" --num-threads "${t}") > "${json}" 2> "${out}/bench_t${t}.stderr"; then
    line="$(python3 - "${json}" <<'PY' 2>/dev/null
import json, sys
b = json.load(open(sys.argv[1]))["bench"]
print("  ".join(f"{k}={b[k]['median_ms']:.1f}ms" for k in ("detect", "align", "embed", "total")))
PY
)"
    say "threads=${t}:  ${line:-$(cat "${json}")}   [$(thermal)]"
  else
    say "threads=${t}:  FAIL — ${out}/bench_t${t}.stderr 확인"
  fi
done
say ""

# --- 4. 실카메라 ------------------------------------------------------------
if (( camera )); then
  say "--- [4] facekit_realtime ${seconds}s (num-threads 4) ---"
  if [[ -z "${device}" ]] && command -v v4l2-ctl >/dev/null; then
    # Pi 내장 코덱/ISP 노드(bcm2835-*, rpi-hevc, pispbe, unicam)를 제외한 첫 캡처 장치 = USB 카메라
    device="$(v4l2-ctl --list-devices 2>/dev/null | awk '
      /^[^ \t]/ { skip = ($0 ~ /bcm2835|rpi-hevc|pispbe|unicam|rp1-cfe/) ; next }
      !skip && /\/dev\/video/ { gsub(/[ \t]/, ""); print; exit }')"
  fi
  if [[ -z "${device}" ]]; then
    say "카메라 장치를 못 찾음 — --device /dev/videoN 으로 지정 (v4l2-ctl --list-devices 참고)"
  else
    say "device:   ${device}"
    v4l2-ctl -d "${device}" --list-formats-ext > "${out}/camera_formats.txt" 2>&1 || true
    (cd "${cli_dir}" && dart run bin/facekit_realtime.dart \
        --device "${device}" --width 640 --height 480 \
        --detector "${detector}" --embedder "${embedder}" \
        --enroll-image "${image}" --seconds "${seconds}" --num-threads 4) \
        > "${out}/realtime_frames.jsonl" 2> "${out}/realtime.stderr"
    rc=$?
    say "exit=${rc}   [$(thermal)]"
    say "stats:    $(grep -E '^\{' "${out}/realtime.stderr" | tail -1)"
  fi
  say ""
fi

say "결과 디렉토리: ${out}"
echo
echo "↑ 위 summary 전체를 복사해서 붙여주세요. (${summary})"
