#!/usr/bin/env bash
# facekit_cli 를 Raspberry Pi 4 (64-bit Raspberry Pi OS) 에서 돌리기 위한 1회성 셋업.
# One-time setup to run facekit_cli on a Raspberry Pi 4 (64-bit Raspberry Pi OS).
#
# Q6A 포팅(doc/KR/linux_port_scope.md S4)과 같은 레시피를 Pi4에 맞춰 자동화한 것:
#   1. apt 의존성 (build-essential, cmake, git, curl, unzip, v4l-utils)
#   2. 독립 Dart SDK (arm64, Flutter 없음) -> ~/tools/dart-sdk
#   3. libtensorflowlite_c.so (arm64) — 태그 v2.21.0-rc0 고정 + TENSORFLOW_SOURCE_DIR 오버라이드
#      -> ~/tools/tflite_c_build/tflite_build/libtensorflowlite_c.so
#   4. packages/facekit_cli/native/libv4l2_capture.so
#   5. AuraFace 가중치 (tool/fetch_models.sh)
#   6. dart pub get (facekit_core, facekit_cli)
#   7. tool/rpi4/env.sh 생성 (PATH, FACEKIT_TFLITE_C_LIBRARY)
#
# 각 단계는 이미 끝나 있으면 건너뛴다 — 중간에 실패해도 다시 실행하면 이어서 진행.
# Each step is skipped if already done, so re-running resumes after a failure.
#
# Usage:  ./tool/rpi4/setup.sh            (저장소 루트에서 / from repo root)
#         FACEKIT_BUILD_JOBS=2 ./tool/rpi4/setup.sh   (빌드 병렬도 수동 지정)

set -euo pipefail

readonly TF_TAG="v2.21.0-rc0"        # tensorflow/lite/CMakeLists.txt 내부 FetchContent 태그와 일치해야 함
readonly DART_MIN="3.12.2"           # packages/facekit_cli/pubspec.yaml environment.sdk
readonly DART_URL="https://storage.googleapis.com/dart-archive/channels/stable/release/latest/sdk/dartsdk-linux-arm64-release.zip"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
tools_dir="${HOME}/tools"
dart_dir="${tools_dir}/dart-sdk"
tf_root="${tools_dir}/tflite_c_build"
tf_src="${tf_root}/tensorflow_src"
tf_build="${tf_root}/tflite_build"
tf_lib="${tf_build}/libtensorflowlite_c.so"

log()  { echo -e "\n[rpi4-setup] $*"; }
fail() { echo "[rpi4-setup] 오류 / error: $*" >&2; exit 1; }

# --- 0. 플랫폼 확인 ---------------------------------------------------------
arch="$(uname -m)"
[[ "${arch}" == "aarch64" ]] || fail "aarch64 가 아님 (uname -m = ${arch}). 64-bit Raspberry Pi OS 가 필요함."
model="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
mem_kb="$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
mem_gb=$(( (mem_kb + 524288) / 1048576 ))
log "보드: ${model} / RAM ~${mem_gb}GB / 코어 $(nproc)개"

# --- 1. apt 의존성 ----------------------------------------------------------
need_pkgs=()
for p in build-essential cmake git curl unzip v4l-utils; do
  dpkg -s "$p" >/dev/null 2>&1 || need_pkgs+=("$p")
done
if (( ${#need_pkgs[@]} )); then
  log "apt 설치: ${need_pkgs[*]} (sudo 비밀번호를 물을 수 있음)"
  sudo apt-get update
  sudo apt-get install -y "${need_pkgs[@]}"
else
  log "apt 의존성 이미 설치됨 — 건너뜀"
fi

# --- 2. 독립 Dart SDK -------------------------------------------------------
dart_ok() {
  local bin="$1" v
  [[ -x "$bin" ]] || return 1
  v="$("$bin" --version 2>&1 | sed -n 's/.*version: \([0-9][0-9.]*\).*/\1/p')"
  [[ -n "$v" ]] && [[ "$(printf '%s\n%s\n' "$DART_MIN" "$v" | sort -V | head -1)" == "$DART_MIN" ]]
}
if dart_ok "${dart_dir}/bin/dart"; then
  log "Dart SDK 이미 있음: $("${dart_dir}/bin/dart" --version 2>&1)"
else
  log "독립 Dart SDK (arm64) 설치 -> ${dart_dir}"
  mkdir -p "${tools_dir}"
  tmp_zip="$(mktemp --suffix=.zip)"
  curl -fL --retry 3 -o "${tmp_zip}" "${DART_URL}"
  rm -rf "${dart_dir}"
  unzip -q "${tmp_zip}" -d "${tools_dir}"
  rm -f "${tmp_zip}"
  dart_ok "${dart_dir}/bin/dart" || fail "설치된 Dart 버전이 ${DART_MIN} 미만"
  log "설치됨: $("${dart_dir}/bin/dart" --version 2>&1)"
fi
export PATH="${dart_dir}/bin:${PATH}"

# --- 3. libtensorflowlite_c.so (arm64) --------------------------------------
if [[ -f "${tf_lib}" ]]; then
  log "libtensorflowlite_c.so 이미 있음 — 건너뜀 (${tf_lib})"
else
  mkdir -p "${tf_root}"
  if [[ ! -d "${tf_src}/.git" ]]; then
    log "TensorFlow ${TF_TAG} 얕은 클론 (수백 MB, 시간 걸림)"
    git clone --depth 1 --branch "${TF_TAG}" https://github.com/tensorflow/tensorflow.git "${tf_src}"
  fi
  # Pi4 4GB 이하에서 -j4 는 C++ 컴파일 중 OOM 위험 -> RAM 1.5GB 당 1잡으로 제한
  jobs="${FACEKIT_BUILD_JOBS:-}"
  if [[ -z "${jobs}" ]]; then
    jobs=$(( mem_kb / 1572864 ))
    (( jobs < 1 )) && jobs=1
    (( jobs > $(nproc) )) && jobs=$(nproc)
  fi
  log "CMake configure + build (-j${jobs}). Pi4 에서 1시간 이상 걸릴 수 있음 — 방열 확인."
  mkdir -p "${tf_build}"
  cmake -S "${tf_src}/tensorflow/lite/c" -B "${tf_build}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DTENSORFLOW_SOURCE_DIR="${tf_src}"
  cmake --build "${tf_build}" -j "${jobs}"
  [[ -f "${tf_lib}" ]] || fail "빌드는 끝났지만 ${tf_lib} 가 없음"
  log "빌드됨: ${tf_lib} ($(du -h "${tf_lib}" | cut -f1))"
fi

# --- 4. libv4l2_capture.so --------------------------------------------------
v4l2_lib="${repo_root}/packages/facekit_cli/native/libv4l2_capture.so"
if [[ -f "${v4l2_lib}" && "${v4l2_lib}" -nt "${repo_root}/packages/facekit_cli/native/v4l2_capture.c" ]]; then
  log "libv4l2_capture.so 최신 — 건너뜀"
else
  log "libv4l2_capture.so 빌드"
  bash "${repo_root}/packages/facekit_cli/native/build.sh"
fi

# --- 5. AuraFace 가중치 -----------------------------------------------------
log "AuraFace 가중치 확인/다운로드"
bash "${repo_root}/tool/fetch_models.sh"

# --- 6. dart pub get --------------------------------------------------------
for pkg in facekit_core facekit_cli; do
  log "dart pub get (${pkg})"
  (cd "${repo_root}/packages/${pkg}" && dart pub get)
done

# --- 7. env.sh --------------------------------------------------------------
env_file="${script_dir}/env.sh"
cat > "${env_file}" <<EOF
# tool/rpi4/setup.sh 가 생성 — 'source tool/rpi4/env.sh' 로 사용 (커밋하지 않음)
export PATH="${dart_dir}/bin:\${PATH}"
export FACEKIT_TFLITE_C_LIBRARY="${tf_lib}"
export FACEKIT_V4L2_LIBRARY="${v4l2_lib}"
EOF
log "완료. 다음:  source tool/rpi4/env.sh && ./tool/rpi4/bench.sh"
