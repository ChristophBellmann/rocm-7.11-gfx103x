#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
ONNXRUNTIME_ROCM_REPO="${ONNXRUNTIME_ROCM_REPO:-${ROOT}/validation/workspace/cache/git/onnxruntime_rocm711}"
BUILD_DIR="${BUILD_DIR:-${ROOT}/validation/workspace/builds/onnxruntime_rocm/build-gfx1031-tlsfix-wheel}"
VENV_PROVIDER_SO="${VENV_PROVIDER_SO:-${ROOT}/validation/workspace/envs/py/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime_providers_rocm.so}"
WHEEL_PATH="${WHEEL_PATH:-}"
SOURCE_FILE="${SOURCE_FILE:-}"

usage() {
  cat <<USAGE
Usage:
  $0 [--source REL_OR_ABS_PATH] [--wheel PATH]

Checks:
  1. optional Source -> object freshness inside the ORT ROCm provider build
  2. Release/libonnxruntime_providers_rocm.so exists
  3. active validation-venv provider .so matches Release/libonnxruntime_providers_rocm.so
  4. optional wheel-staged provider .so matches Release/libonnxruntime_providers_rocm.so

Environment:
  ONNXRUNTIME_ROCM_REPO=${ONNXRUNTIME_ROCM_REPO}
  BUILD_DIR=${BUILD_DIR}
  VENV_PROVIDER_SO=${VENV_PROVIDER_SO}
  WHEEL_PATH=<optional wheel to verify>
  SOURCE_FILE=<optional source file to verify>
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE_FILE="${2:-}"
      shift 2
      ;;
    --wheel)
      WHEEL_PATH="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

for var_name in ONNXRUNTIME_ROCM_REPO BUILD_DIR VENV_PROVIDER_SO; do
  var_value="${!var_name}"
  if [[ "${var_value}" != /* ]]; then
    printf -v "${var_name}" '%s/%s' "${ROOT}" "${var_value}"
  fi
done

if [[ -n "${WHEEL_PATH}" && "${WHEEL_PATH}" != /* ]]; then
  WHEEL_PATH="${ROOT}/${WHEEL_PATH}"
fi

RELEASE_DIR="${BUILD_DIR}/Release"
RELEASE_PROVIDER_SO="${RELEASE_DIR}/libonnxruntime_providers_rocm.so"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

note() {
  echo "$*"
}

sha256_of() {
  sha256sum "$1" | awk '{print $1}'
}

verify_release_and_venv_provider() {
  [[ -f "${RELEASE_PROVIDER_SO}" ]] || fail "missing Release provider: ${RELEASE_PROVIDER_SO}"
  [[ -f "${VENV_PROVIDER_SO}" ]] || fail "missing active venv provider: ${VENV_PROVIDER_SO}"

  local release_sha venv_sha
  release_sha="$(sha256_of "${RELEASE_PROVIDER_SO}")"
  venv_sha="$(sha256_of "${VENV_PROVIDER_SO}")"
  note "Release provider: ${RELEASE_PROVIDER_SO}"
  note "Active venv provider: ${VENV_PROVIDER_SO}"
  note "  release sha256: ${release_sha}"
  note "  venv    sha256: ${venv_sha}"
  [[ "${release_sha}" == "${venv_sha}" ]] || fail "active venv provider does not match Release/libonnxruntime_providers_rocm.so"
  note "Provider sync OK: active venv provider matches Release/libonnxruntime_providers_rocm.so"
}

verify_source_object_freshness() {
  local source_arg="$1"
  local source_abs
  if [[ -z "${source_arg}" ]]; then
    return 0
  fi
  if [[ "${source_arg}" == /* ]]; then
    source_abs="${source_arg}"
  else
    source_abs="${ONNXRUNTIME_ROCM_REPO}/${source_arg}"
  fi
  [[ -f "${source_abs}" ]] || fail "source file not found: ${source_abs}"

  local object_path="${RELEASE_DIR}/CMakeFiles/onnxruntime_providers_rocm.dir/${source_abs}.o"
  [[ -f "${object_path}" ]] || fail "provider object not found for source: ${object_path}"

  local source_mtime object_mtime
  source_mtime="$(stat -c '%Y' "${source_abs}")"
  object_mtime="$(stat -c '%Y' "${object_path}")"
  note "Source file: ${source_abs}"
  note "Object file: ${object_path}"
  note "  source mtime: $(stat -c '%y' "${source_abs}")"
  note "  object mtime: $(stat -c '%y' "${object_path}")"
  if (( object_mtime < source_mtime )); then
    fail "provider object is older than source; incremental build is stale"
  fi
  note "Source/object freshness OK"
}

verify_wheel_provider() {
  local wheel_path="$1"
  [[ -n "${wheel_path}" ]] || return 0
  [[ -f "${wheel_path}" ]] || fail "wheel not found: ${wheel_path}"
  [[ -f "${RELEASE_PROVIDER_SO}" ]] || fail "missing Release provider: ${RELEASE_PROVIDER_SO}"

  local tmp_dir
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "${tmp_dir}"' EXIT

  python3 - <<'PY' "${wheel_path}" "${tmp_dir}"
import sys, zipfile
wheel, out_dir = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(wheel) as zf:
    names = [n for n in zf.namelist() if n.endswith("onnxruntime/capi/libonnxruntime_providers_rocm.so")]
    if not names:
        raise SystemExit("no onnxruntime/capi/libonnxruntime_providers_rocm.so in wheel")
    zf.extract(names[0], out_dir)
PY

  local wheel_provider
  wheel_provider="$(find "${tmp_dir}" -name libonnxruntime_providers_rocm.so | head -n1 || true)"
  [[ -n "${wheel_provider}" && -f "${wheel_provider}" ]] || fail "could not extract provider .so from wheel: ${wheel_path}"

  local release_sha wheel_sha
  release_sha="$(sha256_of "${RELEASE_PROVIDER_SO}")"
  wheel_sha="$(sha256_of "${wheel_provider}")"
  note "Wheel provider: ${wheel_provider}"
  note "  release sha256: ${release_sha}"
  note "  wheel   sha256: ${wheel_sha}"
  [[ "${release_sha}" == "${wheel_sha}" ]] || fail "wheel-staged provider does not match Release/libonnxruntime_providers_rocm.so"
  note "Wheel/provider sync OK"
}

verify_source_object_freshness "${SOURCE_FILE}"
verify_release_and_venv_provider
verify_wheel_provider "${WHEEL_PATH}"
