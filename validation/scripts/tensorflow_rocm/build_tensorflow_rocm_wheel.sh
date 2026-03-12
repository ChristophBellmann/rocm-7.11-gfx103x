#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
DEFAULT_TF_REPO_DIR="${ROOT}/validation/workspace/cache/git/tensorflow_rocm711"
LEGACY_TF_REPO_DIR="${ROOT}/validation/workspace/builds/tensorflow_rocm/tensorflow"
if [[ -z "${TF_REPO_DIR:-}" ]]; then
  if [[ -e "${DEFAULT_TF_REPO_DIR}/.git" ]] || [[ ! -e "${LEGACY_TF_REPO_DIR}/.git" ]]; then
    TF_REPO_DIR="${DEFAULT_TF_REPO_DIR}"
  else
    TF_REPO_DIR="${LEGACY_TF_REPO_DIR}"
  fi
fi
WORK_ROOT="${WORK_ROOT:-${ROOT}/validation/workspace/builds/tensorflow_rocm}"
IN_TREE_ROCM_PATH="${ROOT}/build-stage2/dist/rocm"
DEFAULT_ROCM_PATH="${IN_TREE_ROCM_PATH}"
if [[ ! -d "${DEFAULT_ROCM_PATH}" ]]; then
  DEFAULT_ROCM_PATH="/opt/rocm"
fi
REQUESTED_ROCM_PATH="${ROCM_PATH:-}"
TF_USE_SYSTEM_ROCM="${TF_USE_SYSTEM_ROCM:-0}"
ROCM_PATH="${REQUESTED_ROCM_PATH:-${DEFAULT_ROCM_PATH}}"
if [[ -d "${IN_TREE_ROCM_PATH}" ]]; then
  if [[ -z "${REQUESTED_ROCM_PATH}" ]] || \
     [[ "${REQUESTED_ROCM_PATH}" == /opt/rocm* && "${TF_USE_SYSTEM_ROCM}" != "1" ]]; then
    ROCM_PATH="${IN_TREE_ROCM_PATH}"
  fi
fi

TF_REPO_URL="${TF_REPO_URL:-https://github.com/ChristophBellmann/rocm-7.11-tensorflow-gfx103x.git}"
TF_REF="${TF_REF:-christoph/gfx1031-buildfixes}"
WHEEL_OUT_DIR="${WHEEL_OUT_DIR:-${ROOT}/validation/workspace/cache/wheels/tensorflow_rocm_custom}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT}/validation/workspace/cache/xdg}"
BAZELISK_HOME="${BAZELISK_HOME:-${ROOT}/validation/workspace/cache/bazelisk}"
BAZEL_OUTPUT_USER_ROOT="${BAZEL_OUTPUT_USER_ROOT:-${ROOT}/validation/workspace/cache/bazel/output_user_root}"
CCACHE_DIR="${CCACHE_DIR:-${ROOT}/validation/workspace/cache/ccache}"
DO_UPDATE="${DO_UPDATE:-1}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<USAGE
TheRock integration wrapper for the external TensorFlow ROCm release helper.

Environment:
  TF_REPO_DIR=${TF_REPO_DIR}
  TF_REPO_URL=${TF_REPO_URL}
  TF_REF=${TF_REF}
  WORK_ROOT=${WORK_ROOT}
  WHEEL_OUT_DIR=${WHEEL_OUT_DIR}
USAGE
  exit 0
fi

for var_name in TF_REPO_DIR WORK_ROOT WHEEL_OUT_DIR XDG_CACHE_HOME BAZELISK_HOME BAZEL_OUTPUT_USER_ROOT CCACHE_DIR; do
  var_value="${!var_name}"
  if [[ "${var_value}" != /* ]]; then
    printf -v "${var_name}" '%s/%s' "${ROOT}" "${var_value}"
  fi
done

if [[ ! -d "${ROCM_PATH}" ]]; then
  echo "ERROR: ROCM_PATH does not exist: ${ROCM_PATH}" >&2
  exit 1
fi

mkdir -p "${WORK_ROOT}" "${WHEEL_OUT_DIR}" "${XDG_CACHE_HOME}" "${BAZELISK_HOME}" "${BAZEL_OUTPUT_USER_ROOT}" "${CCACHE_DIR}"
mkdir -p "$(dirname "${TF_REPO_DIR}")"
if [[ ! -d "${TF_REPO_DIR}/.git" ]]; then
  git clone "${TF_REPO_URL}" "${TF_REPO_DIR}"
fi

HELPER="${TF_REPO_DIR}/tools/rocm_release/build_tensorflow_rocm_wheel.sh"
if [[ ! -x "${HELPER}" ]]; then
  echo "ERROR: TensorFlow repo is missing ${HELPER}" >&2
  exit 1
fi

cd "${TF_REPO_DIR}"
repo_dirty=0
if [[ -n "$(git status --porcelain)" ]]; then
  repo_dirty=1
  echo "WARNING: TensorFlow repo working tree is dirty; skipping automatic checkout/update to avoid clobbering local changes."
fi

if [[ "${DO_UPDATE}" == "1" && "${repo_dirty}" == "0" ]]; then
  git fetch --tags --all
  git checkout "${TF_REF}"
  git pull --ff-only || true
else
  echo "Using existing TensorFlow checkout: $(git rev-parse --short HEAD)"
fi

export ROCM_PATH
export WORK_ROOT
export WHEEL_OUT_DIR
export XDG_CACHE_HOME
export BAZELISK_HOME
export BAZEL_OUTPUT_USER_ROOT
export CCACHE_DIR

exec bash "${HELPER}"
