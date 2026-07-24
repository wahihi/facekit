#!/usr/bin/env bash
# Downloads BYOM/Demo embedding model weights that are not committed to the repo
# (see .gitignore) from a GitHub Release, and verifies their integrity.
# 리포에 커밋되지 않는 BYOM/Demo 임베딩 모델 가중치를 GitHub Release에서 받아
# SHA256으로 무결성을 검증한다.
#
# Requires execute permission / 실행 권한 필요:
#   chmod +x tool/fetch_models.sh
#   ./tool/fetch_models.sh

set -euo pipefail

# TODO: fill in before shipping / 배포 전 채울 것.
readonly RELEASE_URL="https://github.com/wahihi/facekit/releases/download/v0.1.0-models/auraface_r100_fp16.tflite"
readonly EXPECTED_SHA256="6c5737440ddfddf06b349401e6befbb19c8d3c6c562718acc785d8fb41447d29"

readonly DEST_PATH="example/assets/models/auraface/auraface_r100_fp16.tflite"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
dest="${repo_root}/${DEST_PATH}"

if [[ -f "${dest}" ]]; then
  echo "[fetch_models] 이미 존재함, 건너뜀 / already present, skipping: ${DEST_PATH}"
  exit 0
fi

if [[ "${RELEASE_URL}" == "<RELEASE_URL>" ]]; then
  echo "[fetch_models] 오류: RELEASE_URL이 채워지지 않았습니다. 스크립트 상단의 RELEASE_URL을 실제 GitHub Release 자산 URL로 채우세요." >&2
  echo "[fetch_models] error: RELEASE_URL is still a placeholder. Fill in the actual GitHub Release asset URL at the top of this script." >&2
  exit 1
fi

if [[ "${EXPECTED_SHA256}" == "<SHA256>" ]]; then
  echo "[fetch_models] 오류: EXPECTED_SHA256이 채워지지 않았습니다. 스크립트 상단의 EXPECTED_SHA256을 실제 해시로 채우세요." >&2
  echo "[fetch_models] error: EXPECTED_SHA256 is still a placeholder. Fill in the actual expected hash at the top of this script." >&2
  exit 1
fi

mkdir -p "$(dirname "${dest}")"

tmp_file="$(mktemp)"
cleanup() { rm -f "${tmp_file}"; }
trap cleanup EXIT

echo "[fetch_models] 다운로드 중 / downloading: ${RELEASE_URL}"
if ! curl -fL --retry 3 -o "${tmp_file}" "${RELEASE_URL}"; then
  echo "[fetch_models] 오류: 다운로드 실패. RELEASE_URL이 유효한지, 네트워크 연결을 확인하세요: ${RELEASE_URL}" >&2
  echo "[fetch_models] error: download failed. Check that RELEASE_URL is valid and the network is reachable: ${RELEASE_URL}" >&2
  exit 1
fi

actual_sha256="$(sha256sum "${tmp_file}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${EXPECTED_SHA256}" ]]; then
  echo "[fetch_models] 오류: SHA256 불일치. 파일이 손상되었거나 변조되었을 수 있습니다." >&2
  echo "[fetch_models]   기대값 / expected: ${EXPECTED_SHA256}" >&2
  echo "[fetch_models]   실제값 / actual:   ${actual_sha256}" >&2
  echo "[fetch_models] error: SHA256 mismatch. The file may be corrupted or tampered with." >&2
  exit 1
fi

mv "${tmp_file}" "${dest}"
trap - EXIT

echo "[fetch_models] 완료 / done: ${DEST_PATH} (sha256=${actual_sha256})"
