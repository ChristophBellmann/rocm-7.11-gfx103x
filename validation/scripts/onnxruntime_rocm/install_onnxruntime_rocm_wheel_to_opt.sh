#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
ONNXRUNTIME_ROCM_REPO="${ONNXRUNTIME_ROCM_REPO:-${ROOT}/validation/workspace/cache/git/onnxruntime_rocm711}"
DEFAULT_SRC_DIR="${SRC_DIR:-${ROOT}/validation/workspace/cache/wheels/onnxruntime_rocm711}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<USAGE
TheRock integration wrapper for the external ORT promote helper.

Environment:
  ONNXRUNTIME_ROCM_REPO=${ONNXRUNTIME_ROCM_REPO}
  SRC_DIR=${DEFAULT_SRC_DIR}
USAGE
  exit 0
fi

if [[ "${ONNXRUNTIME_ROCM_REPO}" != /* ]]; then
  ONNXRUNTIME_ROCM_REPO="${ROOT}/${ONNXRUNTIME_ROCM_REPO}"
fi

HELPER="${ONNXRUNTIME_ROCM_REPO}/tools/rocm_release/install_onnxruntime_rocm_wheel_to_opt.sh"
if [[ ! -x "${HELPER}" ]]; then
  echo "ERROR: missing ORT release helper: ${HELPER}" >&2
  echo "Expected repo: ${ONNXRUNTIME_ROCM_REPO}" >&2
  exit 1
fi

if [[ "${DEFAULT_SRC_DIR}" != /* ]]; then
  DEFAULT_SRC_DIR="${ROOT}/${DEFAULT_SRC_DIR}"
fi

find_latest_wheel() {
  ls -1t "${DEFAULT_SRC_DIR}"/onnxruntime_rocm-*.whl 2>/dev/null | head -n1 || true
}

ARGS=("$@")
if [[ "${#ARGS[@]}" -eq 0 ]]; then
  latest="$(find_latest_wheel)"
  if [[ -n "${latest}" ]]; then
    ARGS=("${latest}")
  fi
fi

exec env SRC_DIR="${DEFAULT_SRC_DIR}" bash "${HELPER}" "${ARGS[@]}"
