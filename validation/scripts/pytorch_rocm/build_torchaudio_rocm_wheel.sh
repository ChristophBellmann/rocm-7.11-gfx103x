#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-build}"
LEGACY_WORKSPACE_DIR="${WORKSPACE_DIR:-}"
RELEASE_ROOT="${RELEASE_ROOT:-${LEGACY_WORKSPACE_DIR:-${ROOT}/validation/workspace/cache}}"
WORKSPACE_DIR="${RELEASE_ROOT}"
ROCM_PREFIX="${ROCM_PREFIX:-/opt/rocm}"
TORCH_WHEEL_PATH="${TORCH_WHEEL_PATH:-}"
TORCHAUDIO_REF="${TORCHAUDIO_REF:-v2.11.0-rc2}"
SRC_DIR="${SRC_DIR:-${WORKSPACE_DIR}/git/torchaudio}"
WHEEL_DIR="${WHEEL_DIR:-${WORKSPACE_DIR}/wheels/pytorch_rocm711}"
BUILD_VENV_DIR="${BUILD_VENV_DIR:-${WORKSPACE_DIR}/venvs/torchaudio}"
BUILD_INFO_PATH="${BUILD_INFO_PATH:-${WHEEL_DIR}/TORCHAUDIO_BUILD_INFO.json}"

usage() {
  cat <<'USAGE'
Usage: build_torchaudio_rocm_wheel.sh [options]

Builds a torchaudio wheel against the promoted/current custom PyTorch wheel.

Defaults:
  - torch wheel: /opt/rocm/wheels/pytorch_rocm711/torch-current.whl
                 else newest wheel in validation/workspace/cache/wheels/pytorch_rocm711
                 else newest wheel in ./dist
  - ROCm prefix: /opt/rocm, else <repo>/<build-dir>/dist/rocm
  - source dir : validation/workspace/cache/git/torchaudio
  - wheel dir  : validation/workspace/cache/wheels/pytorch_rocm711
  - build venv : validation/workspace/cache/venvs/torchaudio

Options:
  --torch-wheel <path>  Explicit torch wheel
  --rocm-prefix <dir>   ROCm prefix for import/smoke
  --build-dir <dir>     Fallback in-tree build dir (default: build)
  --ref <git-ref>       torchaudio git ref/tag
  --src-dir <dir>       Source checkout dir
  --wheel-dir <dir>     Output wheel dir
  --build-venv <dir>    Build venv dir
  -h, --help            Show help
USAGE
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "'$1' not found"
}

auto_find_torch_wheel() {
  if [[ -f "${ROCM_PREFIX}/wheels/pytorch_rocm711/torch-current.whl" ]]; then
    echo "${ROCM_PREFIX}/wheels/pytorch_rocm711/torch-current.whl"
    return 0
  fi
  ls -1t \
    "${WORKSPACE_DIR}/wheels/pytorch_rocm711"/torch-*.whl \
    "${ROOT}/dist"/torch-*.whl \
    2>/dev/null | head -n 1 || true
}

choose_rocm_prefix() {
  if [[ -d "${ROCM_PREFIX}" ]]; then
    echo "${ROCM_PREFIX}"
    return 0
  fi
  local fallback="${ROOT}/${BUILD_DIR}/dist/rocm"
  if [[ -d "${fallback}" ]]; then
    echo "${fallback}"
    return 0
  fi
  die "ROCm prefix not found: '${ROCM_PREFIX}' and fallback missing: '${fallback}'"
}

sync_source_tree() {
  local src_dir="$1"
  local ref="$2"
  if [[ ! -d "${src_dir}/.git" ]]; then
    rm -rf "${src_dir}"
    git clone https://github.com/pytorch/audio.git "${src_dir}"
  fi
  git -C "${src_dir}" fetch --tags origin
  git -C "${src_dir}" checkout --force "${ref}"
  git -C "${src_dir}" clean -fdx
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --torch-wheel)
      TORCH_WHEEL_PATH="${2:-}"
      shift 2
      ;;
    --rocm-prefix)
      ROCM_PREFIX="${2:-}"
      shift 2
      ;;
    --build-dir)
      BUILD_DIR="${2:-}"
      shift 2
      ;;
    --ref)
      TORCHAUDIO_REF="${2:-}"
      shift 2
      ;;
    --src-dir)
      SRC_DIR="${2:-}"
      shift 2
      ;;
    --wheel-dir)
      WHEEL_DIR="${2:-}"
      shift 2
      ;;
    --build-venv)
      BUILD_VENV_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown arg: $1 (use --help)"
      ;;
  esac
done

need_cmd git
need_cmd python3

if [[ -z "${TORCH_WHEEL_PATH}" ]]; then
  TORCH_WHEEL_PATH="$(auto_find_torch_wheel || true)"
fi
[[ -n "${TORCH_WHEEL_PATH}" ]] || die "No torch wheel found. Promote/build PyTorch first."
[[ -f "${TORCH_WHEEL_PATH}" ]] || die "Torch wheel not found: ${TORCH_WHEEL_PATH}"
if [[ -L "${TORCH_WHEEL_PATH}" ]]; then
  TORCH_WHEEL_PATH="$(readlink -f "${TORCH_WHEEL_PATH}")"
fi

rocm_use="$(choose_rocm_prefix)"

echo "== torchaudio ROCm wheel build =="
echo "torch wheel : ${TORCH_WHEEL_PATH}"
echo "ROCm prefix : ${rocm_use}"
echo "ref         : ${TORCHAUDIO_REF}"
echo "src dir     : ${SRC_DIR}"
echo "wheel dir   : ${WHEEL_DIR}"
echo "build venv  : ${BUILD_VENV_DIR}"
echo ""

mkdir -p "${WHEEL_DIR}" "$(dirname "${BUILD_VENV_DIR}")"

if [[ ! -x "${BUILD_VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${BUILD_VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${BUILD_VENV_DIR}/bin/activate"

python -m pip install -U pip setuptools wheel >/dev/null
python -m pip install -U typing-extensions filelock fsspec jinja2 networkx sympy >/dev/null
python -m pip install --no-deps --force-reinstall "${TORCH_WHEEL_PATH}" >/dev/null

sync_source_tree "${SRC_DIR}" "${TORCHAUDIO_REF}"

export ROCM_PATH="${rocm_use}"
export HIP_PATH="${HIP_PATH:-$ROCM_PATH}"
export HSA_PATH="${HSA_PATH:-$ROCM_PATH}"
export PATH="$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:${PATH:-}"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/lib/llvm/lib:$ROCM_PATH/lib/host-math/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:$ROCM_PATH/llvm/lib:${LD_LIBRARY_PATH:-}"
if [[ -f "$ROCM_PATH/lib/llvm/lib/libomp.so" ]]; then
  export LD_PRELOAD="$ROCM_PATH/lib/llvm/lib/libomp.so${LD_PRELOAD:+:${LD_PRELOAD}}"
fi
export USE_ROCM=1
export USE_CUDA=0
export BUILD_RNNT=0
export BUILD_CUDA_CTC_DECODER=0

rm -f "${WHEEL_DIR}"/torchaudio-*.whl

echo "==> building wheel"
(
  cd "${SRC_DIR}"
  python setup.py bdist_wheel
)

TORCHAUDIO_WHEEL="$(ls -1t "${SRC_DIR}"/dist/torchaudio-*.whl 2>/dev/null | head -n1 || true)"
[[ -n "${TORCHAUDIO_WHEEL}" ]] || die "torchaudio wheel not produced in ${SRC_DIR}/dist"
cp -f "${TORCHAUDIO_WHEEL}" "${WHEEL_DIR}/"
TORCHAUDIO_WHEEL="${WHEEL_DIR}/$(basename "${TORCHAUDIO_WHEEL}")"

echo "==> verifying import"
python -m pip install --no-deps --force-reinstall "${TORCHAUDIO_WHEEL}" >/dev/null
(
  cd /tmp
  python - <<'PY'
import torch
import torchaudio

print("torch      :", torch.__version__)
print("hip        :", getattr(torch.version, "hip", None))
print("torchaudio :", torchaudio.__version__)
PY
)

python3 - <<'PY' "${BUILD_INFO_PATH}" "${TORCHAUDIO_WHEEL}" "${TORCHAUDIO_REF}" "${TORCH_WHEEL_PATH}" "${rocm_use}"
import json
import os
import sys
from datetime import datetime, timezone

out_path, wheel_path, ref, torch_wheel, rocm_prefix = sys.argv[1:6]
payload = {
    "built_at_utc": datetime.now(timezone.utc).isoformat(),
    "torchaudio_wheel": os.path.basename(wheel_path),
    "torchaudio_ref": ref,
    "torch_wheel": os.path.basename(torch_wheel),
    "rocm_prefix": rocm_prefix,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
print(out_path)
PY

echo ""
echo "Built wheel:"
ls -lh "${TORCHAUDIO_WHEEL}"
echo "Build info:"
cat "${BUILD_INFO_PATH}"
