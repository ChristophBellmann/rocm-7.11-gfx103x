#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-build}"
LEGACY_WORKSPACE_DIR="${WORKSPACE_DIR:-}"
RELEASE_ROOT="${RELEASE_ROOT:-${LEGACY_WORKSPACE_DIR:-${ROOT}/validation/workspace/cache}}"
WORKSPACE_DIR="${RELEASE_ROOT}"
ROCM_PREFIX="${ROCM_PREFIX:-/opt/rocm}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/torch-rocm711}"
WHEEL_PATH="${WHEEL_PATH:-}"
TORCHCODEC_WHEEL_PATH="${TORCHCODEC_WHEEL_PATH:-}"
TORCHAUDIO_WHEEL_PATH="${TORCHAUDIO_WHEEL_PATH:-}"
ASSUME_YES=0
DO_SMOKE=1
INSTALL_TORCHCODEC=1
INSTALL_TORCHAUDIO=1

usage() {
  cat <<'USAGE'
Usage: install_pytorch_rocm_wheel_to_venv.sh [options]

Installs the promoted custom PyTorch wheel into a Python venv.
If available, also installs the matching promoted torchcodec and torchaudio companion wheels.

Default:
  - venv:   ~/.venvs/torch-rocm711
  - wheel:  /opt/rocm/wheels/pytorch_rocm711/torch-current.whl if present,
            else the newest torch-*.whl from /opt/rocm, validation/workspace/cache, or ./dist
  - ROCm:   /opt/rocm if present, else <repo>/<build-dir>/dist/rocm

Options:
  --venv <dir>        Venv dir (default: ~/.venvs/torch-rocm711)
  --wheel <path>      Torch wheel to install (default: auto-discover)
  --torchcodec-wheel <path>
                      Companion torchcodec wheel to install (default: auto-discover next to torch wheel)
  --torchaudio-wheel <path>
                      Companion torchaudio wheel to install (default: auto-discover next to torch wheel)
  --rocm-prefix <dir> ROCm prefix to use for the smoke test (default: /opt/rocm, fallback: in-tree dist)
  --build-dir <dir>   In-tree build dir for fallback ROCm prefix (default: build)
  --no-torchcodec     Install only torch (skip companion torchcodec wheel)
  --no-torchaudio     Install only torch (skip companion torchaudio wheel)
  --no-smoke          Install only (skip GPU smoke test)
  -y, --yes           Do not prompt
  -h, --help          Show help
USAGE
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

write_rocm_runtime_wrappers() {
  local venv_dir="$1"
  local rocm_use="$2"
  local activate_script="${venv_dir}/bin/activate_rocm_pytorch.sh"
  local python_wrapper="${venv_dir}/bin/python-rocm"

  cat >"${activate_script}" <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export ROCM_PATH="__ROCM_USE__"
export HIP_PATH="${HIP_PATH:-$ROCM_PATH}"
export HSA_PATH="${HSA_PATH:-$ROCM_PATH}"
export PATH="$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$VENV_DIR/bin:${PATH:-}"
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

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
SCRIPT
  sed -i "s|__ROCM_USE__|${rocm_use}|g" "${activate_script}"
  chmod +x "${activate_script}"

  cat >"${python_wrapper}" <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/activate_rocm_pytorch.sh"
exec "${VIRTUAL_ENV}/bin/python" "$@"
SCRIPT
  chmod +x "${python_wrapper}"
}

auto_find_wheel() {
  if [[ -f "${ROCM_PREFIX}/wheels/pytorch_rocm711/torch-current.whl" ]]; then
    echo "${ROCM_PREFIX}/wheels/pytorch_rocm711/torch-current.whl"
    return 0
  fi
  ls -1t \
    "${ROCM_PREFIX}/wheels/pytorch_rocm711"/torch-*.whl \
    "${WORKSPACE_DIR}/wheels/pytorch_rocm711"/torch-*.whl \
    "${ROOT}/dist"/torch-*.whl \
    2>/dev/null | head -n 1 || true
}

auto_find_torchcodec_wheel() {
  if [[ -f "${ROCM_PREFIX}/wheels/pytorch_rocm711/torchcodec-current.whl" ]]; then
    echo "${ROCM_PREFIX}/wheels/pytorch_rocm711/torchcodec-current.whl"
    return 0
  fi
  ls -1t \
    "${ROCM_PREFIX}/wheels/pytorch_rocm711"/torchcodec-*.whl \
    "${WORKSPACE_DIR}/wheels/pytorch_rocm711"/torchcodec-*.whl \
    2>/dev/null | head -n 1 || true
}

auto_find_torchaudio_wheel() {
  if [[ -f "${ROCM_PREFIX}/wheels/pytorch_rocm711/torchaudio-current.whl" ]]; then
    echo "${ROCM_PREFIX}/wheels/pytorch_rocm711/torchaudio-current.whl"
    return 0
  fi
  ls -1t \
    "${ROCM_PREFIX}/wheels/pytorch_rocm711"/torchaudio-*.whl \
    "${WORKSPACE_DIR}/wheels/pytorch_rocm711"/torchaudio-*.whl \
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
    --torchcodec-wheel)
      TORCHCODEC_WHEEL_PATH="${2:-}"
      shift 2
      ;;
    --torchaudio-wheel)
      TORCHAUDIO_WHEEL_PATH="${2:-}"
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
    --no-torchcodec)
      INSTALL_TORCHCODEC=0
      shift
      ;;
    --no-torchaudio)
      INSTALL_TORCHAUDIO=0
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
[[ -n "${WHEEL_PATH}" ]] || die "No torch wheel found. Build/promote it first."
[[ -f "${WHEEL_PATH}" ]] || die "Wheel not found: ${WHEEL_PATH}"
if [[ -L "${WHEEL_PATH}" ]]; then
  WHEEL_PATH="$(readlink -f "${WHEEL_PATH}")"
fi

if (( INSTALL_TORCHCODEC )) && [[ -z "${TORCHCODEC_WHEEL_PATH}" ]]; then
  TORCHCODEC_WHEEL_PATH="$(auto_find_torchcodec_wheel || true)"
fi
if [[ -n "${TORCHCODEC_WHEEL_PATH}" && -L "${TORCHCODEC_WHEEL_PATH}" ]]; then
  TORCHCODEC_WHEEL_PATH="$(readlink -f "${TORCHCODEC_WHEEL_PATH}")"
fi
if [[ -n "${TORCHCODEC_WHEEL_PATH}" && ! -f "${TORCHCODEC_WHEEL_PATH}" ]]; then
  die "torchcodec wheel not found: ${TORCHCODEC_WHEEL_PATH}"
fi

if (( INSTALL_TORCHAUDIO )) && [[ -z "${TORCHAUDIO_WHEEL_PATH}" ]]; then
  TORCHAUDIO_WHEEL_PATH="$(auto_find_torchaudio_wheel || true)"
fi
if [[ -n "${TORCHAUDIO_WHEEL_PATH}" && -L "${TORCHAUDIO_WHEEL_PATH}" ]]; then
  TORCHAUDIO_WHEEL_PATH="$(readlink -f "${TORCHAUDIO_WHEEL_PATH}")"
fi
if [[ -n "${TORCHAUDIO_WHEEL_PATH}" && ! -f "${TORCHAUDIO_WHEEL_PATH}" ]]; then
  die "torchaudio wheel not found: ${TORCHAUDIO_WHEEL_PATH}"
fi

rocm_use="$(choose_rocm_prefix)"

echo "== PyTorch (ROCm 7.11) install =="
echo "wheel     : ${WHEEL_PATH}"
if (( INSTALL_TORCHCODEC )); then
  echo "torchcodec: ${TORCHCODEC_WHEEL_PATH:-<not found; skipped>}"
else
  echo "torchcodec: <disabled>"
fi
if (( INSTALL_TORCHAUDIO )); then
  echo "torchaudio: ${TORCHAUDIO_WHEEL_PATH:-<not found; skipped>}"
else
  echo "torchaudio: <disabled>"
fi
echo "venv      : ${VENV_DIR}"
echo "ROCm      : ${rocm_use}"
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

if (( INSTALL_TORCHCODEC )) && [[ -n "${TORCHCODEC_WHEEL_PATH}" ]]; then
  echo "==> pip install torchcodec wheel"
  python -m pip install --no-deps --upgrade --force-reinstall "${TORCHCODEC_WHEEL_PATH}"
fi

if (( INSTALL_TORCHAUDIO )) && [[ -n "${TORCHAUDIO_WHEEL_PATH}" ]]; then
  echo "==> pip install torchaudio wheel"
  python -m pip install --no-deps --upgrade --force-reinstall "${TORCHAUDIO_WHEEL_PATH}"
fi

write_rocm_runtime_wrappers "${VENV_DIR}" "${rocm_use}"

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

  if (( INSTALL_TORCHCODEC )) && [[ -n "${TORCHCODEC_WHEEL_PATH}" ]]; then
    export THEROCK_TORCHCODEC_EXPECTED=1
  else
    export THEROCK_TORCHCODEC_EXPECTED=0
  fi

  if (( INSTALL_TORCHAUDIO )) && [[ -n "${TORCHAUDIO_WHEEL_PATH}" ]]; then
    export THEROCK_TORCHAUDIO_EXPECTED=1
  else
    export THEROCK_TORCHAUDIO_EXPECTED=0
  fi

  python - <<'PY'
import os, time
import torch

print(f"torch                 : {torch.__version__}")
print(f"torch.version.rocm    : {torch.version.rocm}")
print(f"torch.version.hip     : {torch.version.hip}")

try:
    import torchcodec
    print(f"torchcodec            : {getattr(torchcodec, '__file__', 'unknown')}")
except Exception as e:
    if os.environ.get("THEROCK_TORCHCODEC_EXPECTED") == "1":
        print(f"torchcodec            : import failed: {e!r}")
        raise
    print("torchcodec            : not installed")

try:
    import torchaudio
    print(f"torchaudio            : {getattr(torchaudio, '__version__', 'unknown')}")
except Exception as e:
    if os.environ.get("THEROCK_TORCHAUDIO_EXPECTED") == "1":
        print(f"torchaudio            : import failed: {e!r}")
        raise
    print("torchaudio            : not installed")

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
echo "  source \"${VENV_DIR}/bin/activate_rocm_pytorch.sh\""
echo "Or run:"
echo "  \"${VENV_DIR}/bin/python-rocm\" -c 'import torch; print(torch.cuda.is_available())'"
