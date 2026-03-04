#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORK_ROOT="${WORK_ROOT:-${ROOT}/validation/workspace/builds/onnxruntime_rocm}"
ORT_SRC_DIR="${ORT_SRC_DIR:-${WORK_ROOT}/onnxruntime}"
BUILD_DIR="${BUILD_DIR:-${WORK_ROOT}/build-gfx1031-tlsfix-wheel}"
LOG_FILE="${LOG_FILE:-${WORK_ROOT}/ort_build_live.log}"
WHEEL_OUT_DIR="${WHEEL_OUT_DIR:-${ROOT}/validation/workspace/cache/wheels/onnxruntime_rocm711}"

ORT_REPO_URL="${ORT_REPO_URL:-https://github.com/microsoft/onnxruntime.git}"
ORT_REF="${ORT_REF:-main}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
ROCM_VERSION="${ROCM_VERSION:-7.11.0}"
HIP_ARCH="${HIP_ARCH:-gfx1031}"
HIP_PLATFORM="${HIP_PLATFORM:-amd}"
USE_MIGRAPHX="${USE_MIGRAPHX:-0}"
MIGRAPHX_HOME="${MIGRAPHX_HOME:-${ROCM_PATH}}"
PARALLEL="${PARALLEL:-$(nproc)}"
DO_UPDATE="${DO_UPDATE:-1}"

TLS_C_FLAGS="${TLS_C_FLAGS:--ftls-model=global-dynamic}"
TLS_CXX_FLAGS="${TLS_CXX_FLAGS:--ftls-model=global-dynamic}"
TLS_LINK_FLAGS="${TLS_LINK_FLAGS:--Wl,--no-as-needed}"

if [[ "${WORK_ROOT}" != /* ]]; then
  WORK_ROOT="${ROOT}/${WORK_ROOT}"
fi
if [[ "${ORT_SRC_DIR}" != /* ]]; then
  ORT_SRC_DIR="${ROOT}/${ORT_SRC_DIR}"
fi
if [[ "${BUILD_DIR}" != /* ]]; then
  BUILD_DIR="${ROOT}/${BUILD_DIR}"
fi
if [[ "${LOG_FILE}" != /* ]]; then
  LOG_FILE="${ROOT}/${LOG_FILE}"
fi
if [[ "${WHEEL_OUT_DIR}" != /* ]]; then
  WHEEL_OUT_DIR="${ROOT}/${WHEEL_OUT_DIR}"
fi
if [[ "${ROCM_PATH}" != /* ]]; then
  ROCM_PATH="${ROOT}/${ROCM_PATH}"
fi
if [[ "${MIGRAPHX_HOME}" != /* ]]; then
  MIGRAPHX_HOME="${ROOT}/${MIGRAPHX_HOME}"
fi

# Avoid stale CMake cache when toggling MIGraphX EP.
DEFAULT_BUILD_DIR="${WORK_ROOT}/build-gfx1031-tlsfix-wheel"
if [[ "${USE_MIGRAPHX}" == "1" && "${BUILD_DIR}" == "${DEFAULT_BUILD_DIR}" ]]; then
  BUILD_DIR="${WORK_ROOT}/build-gfx1031-tlsfix-wheel-migraphx"
fi

mkdir -p "${WORK_ROOT}" "${WHEEL_OUT_DIR}" "$(dirname "${LOG_FILE}")"

# Support both layouts:
# 1) WORK_ROOT/onnxruntime (default)
# 2) WORK_ROOT itself is the ORT git checkout
if [[ ! -d "${ORT_SRC_DIR}/.git" && -d "${WORK_ROOT}/.git" ]]; then
  ORT_SRC_DIR="${WORK_ROOT}"
fi

verify_wheel_tls() {
  local wheel_path="$1"
  local tmp_dir so_path
  tmp_dir="$(mktemp -d)"
  cleanup() { rm -rf "${tmp_dir}"; }
  trap cleanup RETURN

  "${PYTHON_BIN}" -c 'import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); n=[x for x in z.namelist() if x.endswith("onnxruntime/capi/libonnxruntime_providers_rocm.so")][0]; z.extract(n, sys.argv[2])' "${wheel_path}" "${tmp_dir}"
  so_path="$(find "${tmp_dir}" -name libonnxruntime_providers_rocm.so | head -n1 || true)"
  if [[ -z "${so_path}" || ! -f "${so_path}" ]]; then
    echo "Could not extract provider .so from wheel: ${wheel_path}" >&2
    return 2
  fi

  if readelf -dW "${so_path}" | grep -q "STATIC_TLS"; then
    echo "TLS check FAILED: provider contains STATIC_TLS" >&2
    return 3
  fi

  if readelf -rW "${so_path}" | grep -Eq "_ZSt11__once_call|_ZSt15__once_callable|R_X86_64_TPOFF64"; then
    echo "TLS check FAILED: provider still has once/TPOFF TLS relocations" >&2
    return 4
  fi

  echo "TLS check OK (no STATIC_TLS / no once-call TLS relocations)."
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found/executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${ORT_SRC_DIR}/.git" ]]; then
  echo "[ORT] Clone ${ORT_REPO_URL} -> ${ORT_SRC_DIR}"
  git clone --recursive "${ORT_REPO_URL}" "${ORT_SRC_DIR}"
fi

cd "${ORT_SRC_DIR}"

# Ensure origin heads are fetchable (some cached clones may have tag-only refspecs).
if ! git config --get-all remote.origin.fetch | grep -q 'refs/heads/\*'; then
  git config --add remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*'
fi

if [[ "${DO_UPDATE}" == "1" ]]; then
  echo "[ORT] Fetch updates"
  git fetch origin --tags --prune
  git fetch upstream --tags --prune || true
fi

echo "[ORT] Checkout ${ORT_REF}"
if git show-ref --verify --quiet "refs/remotes/origin/${ORT_REF}"; then
  git checkout -B "${ORT_REF}" "origin/${ORT_REF}"
else
  git checkout "${ORT_REF}"
fi
if [[ "${DO_UPDATE}" == "1" ]]; then
  git pull --ff-only || true
  git submodule sync --recursive
  git submodule update --init --recursive
fi

export HIP_PLATFORM

echo "== ONNX Runtime ROCm wheel build =="
echo "ROOT=${ROOT}"
echo "WORK_ROOT=${WORK_ROOT}"
echo "ORT_SRC_DIR=${ORT_SRC_DIR}"
echo "BUILD_DIR=${BUILD_DIR}"
echo "WHEEL_OUT_DIR=${WHEEL_OUT_DIR}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "ROCM_PATH=${ROCM_PATH}"
echo "ROCM_VERSION=${ROCM_VERSION}"
echo "HIP_ARCH=${HIP_ARCH}"
echo "HIP_PLATFORM=${HIP_PLATFORM}"
echo "USE_MIGRAPHX=${USE_MIGRAPHX}"
echo "MIGRAPHX_HOME=${MIGRAPHX_HOME}"
echo "PARALLEL=${PARALLEL}"
echo "DO_UPDATE=${DO_UPDATE}"
echo "LOG_FILE=${LOG_FILE}"

BUILD_ARGS=()
if [[ "${DO_UPDATE}" == "1" ]]; then
  BUILD_ARGS+=(--update)
fi
if [[ "${USE_MIGRAPHX}" == "1" ]]; then
  if [[ "${DO_UPDATE}" != "1" ]]; then
    # Ensure CMake configure picks up MIGraphX flags even when repo update is disabled.
    BUILD_ARGS+=(--update)
  fi
  BUILD_ARGS+=(--use_migraphx --migraphx_home "${MIGRAPHX_HOME}")
fi

# Needed for pybind build (onnxruntime/python/numpy_helper.h).
"${PYTHON_BIN}" -m pip install -q "numpy<2"

# Reconfigure when cache is missing/stale (e.g. missing NumPy include detection).
CMAKE_CACHE="${BUILD_DIR}/Release/CMakeCache.txt"
if [[ ! -f "${CMAKE_CACHE}" ]] || ! grep -q '^Python_NumPy_INCLUDE_DIR:' "${CMAKE_CACHE}" 2>/dev/null; then
  if [[ " ${BUILD_ARGS[*]} " != *" --update "* ]]; then
    BUILD_ARGS+=(--update)
  fi
fi

"${PYTHON_BIN}" tools/ci_build/build.py \
  --build_dir "${BUILD_DIR}" \
  --config Release \
  "${BUILD_ARGS[@]}" \
  --build \
  --build_wheel \
  --enable_pybind \
  --skip_tests \
  --parallel "${PARALLEL}" \
  --use_rocm \
  --rocm_home "${ROCM_PATH}" \
  --rocm_version "${ROCM_VERSION}" \
  --cmake_extra_defines \
    "CMAKE_HIP_ARCHITECTURES=${HIP_ARCH}" \
    "onnxruntime_USE_COMPOSABLE_KERNEL=OFF" \
    "onnxruntime_BUILD_UNIT_TESTS=OFF" \
    "onnxruntime_DISABLE_CONTRIB_OPS=ON" \
    "CMAKE_C_FLAGS=${TLS_C_FLAGS}" \
    "CMAKE_CXX_FLAGS=${TLS_CXX_FLAGS}" \
    "CMAKE_SHARED_LINKER_FLAGS=${TLS_LINK_FLAGS}" \
  2>&1 | tee "${LOG_FILE}"

WHEEL_PATH="$(find "${BUILD_DIR}/Release/dist" -maxdepth 1 -type f -name 'onnxruntime_rocm-*.whl' | head -n1 || true)"
if [[ -z "${WHEEL_PATH}" ]]; then
  echo "No wheel found under ${BUILD_DIR}/Release/dist" >&2
  exit 2
fi

verify_wheel_tls "${WHEEL_PATH}"

cp -f "${WHEEL_PATH}" "${WHEEL_OUT_DIR}/"

echo
echo "Built wheel:"
echo "${WHEEL_PATH}"
echo "Copied to:"
ls -lh "${WHEEL_OUT_DIR}"/onnxruntime_rocm-*.whl
