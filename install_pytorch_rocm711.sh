#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILD_DIR="${BUILD_DIR:-build-stage2}"
ROCM_PREFIX="${ROCM_PREFIX:-/opt/rocm}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/torch-rocm711}"
WHEEL_PATH="${WHEEL_PATH:-}"
ASSUME_YES=0
DO_SMOKE=1

usage() {
  cat <<'EOF'
Usage: install_pytorch_rocm711.sh [options]

Installs the custom-built PyTorch (built against this repo's ROCm 7.11 dist) into a Python venv.

Default:
  - venv:   ~/.venvs/torch-rocm711
  - wheel:  /opt/rocm/wheels/pytorch_rocm711/torch-*.whl (if present), else validation cache (latest by mtime)
  - ROCm:   /opt/rocm if present, else <repo>/<build-dir>/dist/rocm

Options:
  --venv <dir>        Venv dir (default: ~/.venvs/torch-rocm711)
  --wheel <path>      Torch wheel to install (default: auto-discover in validation cache)
  --rocm-prefix <dir> ROCm prefix to use for the smoke test (default: /opt/rocm, fallback: in-tree dist)
  --build-dir <dir>   In-tree build dir for fallback ROCm prefix (default: build-stage2)
  --no-smoke          Install only (skip GPU smoke test)
  -y, --yes           Do not prompt
  -h, --help          Show help

Notes:
  - This installs only 'torch'. If you need 'torchvision'/'torchaudio', add them separately.
  - The wheel is CPython-version-specific (e.g. cp312). If your python differs, rebuild the wheel.
  - For a fresh wheel build (very heavy):
      python3 validation/scripts/validate.py --profile pytorch_rocm711_source --build-dirs build-stage2 --yes --power --log
EOF
}

confirm() {
  local msg="$1"
  if (( ASSUME_YES )); then
    return 0
  fi
  read -r -p "${msg} [Y/n] " ans
  case "${ans}" in
    ""|Y|y|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

need_cmd() {
  local exe="$1"
  command -v "${exe}" >/dev/null 2>&1 || die "'${exe}' not found"
}

auto_find_wheel() {
  ls -1t \
    "${ROCM_PREFIX}/wheels/pytorch_rocm711"/torch-*.whl \
    "${ROOT}/validation/workspace/cache/wheels/pytorch_rocm711"/torch-*.whl \
    "${ROOT}/validation/workspace/cache/git/pytorch_rocm711/dist"/torch-*.whl \
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_DIR="${2:-}"
      shift 2
      ;;
    --wheel)
      WHEEL_PATH="${2:-}"
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
    --no-smoke)
      DO_SMOKE=0
      shift
      ;;
    -y|--yes)
      ASSUME_YES=1
      shift
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

need_cmd python3

if [[ -z "${WHEEL_PATH}" ]]; then
  WHEEL_PATH="$(auto_find_wheel || true)"
fi
[[ -n "${WHEEL_PATH}" ]] || die "No torch wheel found. Build it first (see --help)."
[[ -f "${WHEEL_PATH}" ]] || die "Wheel not found: ${WHEEL_PATH}"

rocm_use="$(choose_rocm_prefix)"

echo "== PyTorch (ROCm 7.11) install =="
echo "wheel   : ${WHEEL_PATH}"
echo "venv    : ${VENV_DIR}"
echo "ROCm    : ${rocm_use}"
echo ""

if ! confirm "Proceed with venv install?"; then
  echo "Aborted."
  exit 0
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "==> creating venv"
  mkdir -p "$(dirname "${VENV_DIR}")"
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "==> pip install torch wheel"
python -m pip install -U pip setuptools wheel >/dev/null
python -m pip install --upgrade --force-reinstall "${WHEEL_PATH}"

if (( DO_SMOKE )); then
  echo ""
  echo "==> GPU smoke test (ROCm)"
  export ROCM_PATH="${rocm_use}"
  export HIP_PATH="${HIP_PATH:-$ROCM_PATH}"
  export HSA_PATH="${HSA_PATH:-$ROCM_PATH}"
  export PATH="$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:${PATH:-}"
  export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/lib/llvm/lib:$ROCM_PATH/lib/host-math/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:$ROCM_PATH/llvm/lib:${LD_LIBRARY_PATH:-}"
  if [[ -f "$ROCM_PATH/lib/llvm/lib/libomp.so" ]]; then
    export LD_PRELOAD="$ROCM_PATH/lib/llvm/lib/libomp.so${LD_PRELOAD:+:${LD_PRELOAD}}"
  fi
  export USE_ROCM_HIPBLASLT="${USE_ROCM_HIPBLASLT:-0}"
  if [[ -z "${HIP_DEVICE_LIB_PATH:-}" ]]; then
    if [[ -d "$ROCM_PATH/lib/llvm/amdgcn/bitcode" ]]; then
      export HIP_DEVICE_LIB_PATH="$ROCM_PATH/lib/llvm/amdgcn/bitcode"
    elif [[ -d "$ROCM_PATH/amdgcn/bitcode" ]]; then
      export HIP_DEVICE_LIB_PATH="$ROCM_PATH/amdgcn/bitcode"
    fi
  fi

  python - <<'PY'
import os, time
import torch

print(f"torch                 : {torch.__version__}")
print(f"torch.version.rocm    : {torch.version.rocm}")
print(f"torch.version.hip     : {torch.version.hip}")

ok = torch.cuda.is_available()
print(f"torch.cuda.is_available: {ok}")
if not ok:
    raise SystemExit("ERROR: CUDA/HIP backend not available. Check /dev/kfd permissions and ROCm env.")

name = torch.cuda.get_device_name(0)
print(f"device                : {name}")

def find_loaded_hip_lib() -> str:
    try:
        with open("/proc/self/maps", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "libamdhip64.so" in line:
                    return line.split()[-1]
    except Exception:
        pass
    return ""

hip_lib = find_loaded_hip_lib()
if hip_lib:
    print(f"hip_lib               : {hip_lib}")

dev = torch.device("cuda")
dtype = torch.float16
n = 4096
iters = 50

a = torch.randn((n, n), device=dev, dtype=dtype)
b = torch.randn((n, n), device=dev, dtype=dtype)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(iters):
    c = a @ b
torch.cuda.synchronize()
dt = time.time() - t0
print(f"matmul fp16           : n={n} iters={iters} wall={dt:.3f}s it/s={iters/dt:.2f}")
PY
fi

echo ""
echo "Install complete."
echo "Activate:"
echo "  source \"${VENV_DIR}/bin/activate\""
