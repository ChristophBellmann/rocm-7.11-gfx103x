#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
TF_REPO_DIR="${TF_REPO_DIR:-${ROOT}/validation/workspace/builds/tensorflow_rocm/tensorflow}"
DEFAULT_SRC_DIR="${SRC_DIR:-${ROOT}/validation/workspace/cache/wheels/tensorflow_rocm_custom}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<USAGE
TheRock integration wrapper for the external TensorFlow ROCm promote helper.

Environment:
  TF_REPO_DIR=${TF_REPO_DIR}
  SRC_DIR=${DEFAULT_SRC_DIR}
USAGE
  exit 0
fi

for var_name in TF_REPO_DIR DEFAULT_SRC_DIR; do
  var_value="${!var_name}"
  if [[ "${var_value}" != /* ]]; then
    printf -v "${var_name}" '%s/%s' "${ROOT}" "${var_value}"
  fi
done

HELPER="${TF_REPO_DIR}/tools/rocm_release/install_tensorflow_rocm_wheel_to_opt.sh"
if [[ ! -x "${HELPER}" ]]; then
  echo "ERROR: missing TensorFlow release helper: ${HELPER}" >&2
  exit 1
fi

find_latest_wheel() {
  ls -1t "${DEFAULT_SRC_DIR}"/tensorflow*.whl 2>/dev/null | head -n1 || true
}

ARGS=("$@")
if [[ "${#ARGS[@]}" -eq 0 ]]; then
  latest="$(find_latest_wheel)"
  if [[ -n "${latest}" ]]; then
    ARGS=("${latest}")
  fi
fi

exec env SRC_DIR="${DEFAULT_SRC_DIR}" bash "${HELPER}" "${ARGS[@]}"
