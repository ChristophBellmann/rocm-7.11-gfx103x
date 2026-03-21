#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
ORT_REPO_URL="${ORT_REPO_URL:-https://github.com/ChristophBellmann/rocm-7.11-onnxruntime-gfx103x.git}"
ORT_REF="${ORT_REF:-christoph/gfx1031-buildfixes}"
ONNXRUNTIME_ROCM_REPO="${ONNXRUNTIME_ROCM_REPO:-${ROOT}/validation/workspace/cache/git/onnxruntime_rocm711}"
WORK_ROOT="${WORK_ROOT:-${ROOT}/validation/workspace/builds/onnxruntime_rocm}"
BUILD_DIR="${BUILD_DIR:-${WORK_ROOT}/build-gfx1031-tlsfix-wheel}"
LOG_FILE="${LOG_FILE:-${WORK_ROOT}/ort_build_live.log}"
WHEEL_OUT_DIR="${WHEEL_OUT_DIR:-${ROOT}/validation/workspace/cache/wheels/onnxruntime_rocm711}"
ROCM_PATH="${ROCM_PATH:-${ROOT}/build-stage2/dist/rocm}"
ORT_USE_CCACHE="${ORT_USE_CCACHE:-1}"

setup_ccache() {
  if [[ -x "${ROOT}/.local/bin/ccache" ]]; then
    PATH="${ROOT}/.local/bin:${PATH}"
  fi
  if [[ -x "${ROOT}/build_tools/setup_ccache.py" ]]; then
    eval "$(python3 "${ROOT}/build_tools/setup_ccache.py" --init)"
  fi
  export CCACHE_SLOPPINESS="${CCACHE_SLOPPINESS:-include_file_ctime}"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<USAGE
TheRock integration wrapper for the external ORT release helper.

Environment:
  ORT_REPO_URL=${ORT_REPO_URL}
  ORT_REF=${ORT_REF}
  ONNXRUNTIME_ROCM_REPO=${ONNXRUNTIME_ROCM_REPO}
  WORK_ROOT=${WORK_ROOT}
  WHEEL_OUT_DIR=${WHEEL_OUT_DIR}
  ROCM_PATH=${ROCM_PATH}
  ORT_USE_CCACHE=${ORT_USE_CCACHE}
USAGE
  exit 0
fi

for var_name in ONNXRUNTIME_ROCM_REPO WORK_ROOT BUILD_DIR LOG_FILE WHEEL_OUT_DIR ROCM_PATH; do
  var_value="${!var_name}"
  if [[ "${var_value}" != /* ]]; then
    printf -v "${var_name}" '%s/%s' "${ROOT}" "${var_value}"
  fi
done

if [[ -n "${ROCM_PATH:-}" && "${ROCM_PATH}" != /* ]]; then
  ROCM_PATH="${ROOT}/${ROCM_PATH}"
  export ROCM_PATH
fi
if [[ -n "${MIGRAPHX_HOME:-}" && "${MIGRAPHX_HOME}" != /* ]]; then
  MIGRAPHX_HOME="${ROOT}/${MIGRAPHX_HOME}"
  export MIGRAPHX_HOME
fi

if [[ "${ORT_USE_CCACHE}" == "1" ]]; then
  setup_ccache
  if command -v ccache >/dev/null 2>&1; then
    export ORT_CMAKE_C_COMPILER_LAUNCHER="${ORT_CMAKE_C_COMPILER_LAUNCHER:-ccache}"
    export ORT_CMAKE_CXX_COMPILER_LAUNCHER="${ORT_CMAKE_CXX_COMPILER_LAUNCHER:-ccache}"
  fi
fi

if [[ ! -e "${ONNXRUNTIME_ROCM_REPO}/.git" ]]; then
  mkdir -p "$(dirname "${ONNXRUNTIME_ROCM_REPO}")"
  git clone --branch "${ORT_REF}" --single-branch "${ORT_REPO_URL}" "${ONNXRUNTIME_ROCM_REPO}"
fi

HELPER="${ONNXRUNTIME_ROCM_REPO}/tools/rocm_release/build_onnxruntime_rocm_wheel.sh"
if [[ ! -x "${HELPER}" ]]; then
  echo "ERROR: missing ORT release helper: ${HELPER}" >&2
  echo "Expected repo: ${ONNXRUNTIME_ROCM_REPO}" >&2
  exit 1
fi

export WORK_ROOT BUILD_DIR LOG_FILE WHEEL_OUT_DIR
exec bash "${HELPER}" "$@"
