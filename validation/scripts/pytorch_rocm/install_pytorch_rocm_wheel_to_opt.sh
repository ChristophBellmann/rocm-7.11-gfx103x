#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
SRC_DIR_PRIMARY="${SRC_DIR_PRIMARY:-${ROOT}/validation/workspace/cache/wheels/pytorch_rocm711}"
SRC_DIR_FALLBACK="${SRC_DIR_FALLBACK:-${ROOT}/validation/workspace/cache/git/pytorch_rocm711/dist}"
DEST_DIR="${DEST_DIR:-/opt/rocm/wheels/pytorch_rocm711}"
TS="$(date +%Y%m%d_%H%M%S)"
BACKUP_ROOT="${BACKUP_ROOT:-${ROOT}/build-stage2/install-backups/${TS}/pytorch_wheels}"

if [[ "${EUID}" -eq 0 ]] || [[ -w "${DEST_DIR}" ]] || [[ -w "$(dirname "${DEST_DIR}")" ]]; then
  SUDO=()
else
  SUDO=(sudo)
fi

usage() {
  echo "Usage:"
  echo "  $0 [path/to/torch-*.whl]"
  echo "  $0 --restore [torch-*.whl]"
}

find_latest_wheel() {
  ls -1t \
    "${SRC_DIR_PRIMARY}"/torch-*.whl \
    "${SRC_DIR_FALLBACK}"/torch-*.whl \
    2>/dev/null | head -n1 || true
}

extract_member() {
  local wheel_path="$1"
  local suffix="$2"
  local out_dir="$3"
  python3 - <<'PY' "${wheel_path}" "${suffix}" "${out_dir}"
import sys
import zipfile

wheel, suffix, out_dir = sys.argv[1:4]
with zipfile.ZipFile(wheel) as zf:
    matches = [name for name in zf.namelist() if name.endswith(suffix)]
    if not matches:
        raise SystemExit(1)
    zf.extract(matches[0], out_dir)
    print(matches[0])
PY
}

verify_wheel_runtime_contract() {
  local wheel_path="$1"
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  local cpu_rel hip_rel cpu_so hip_so
  cpu_rel="$(extract_member "${wheel_path}" "torch/lib/libtorch_cpu.so" "${tmp_dir}")" || {
    echo "Could not extract libtorch_cpu.so from ${wheel_path}" >&2
    rm -rf "${tmp_dir}"
    return 2
  }
  hip_rel="$(extract_member "${wheel_path}" "torch/lib/libtorch_hip.so" "${tmp_dir}")" || {
    echo "Could not extract libtorch_hip.so from ${wheel_path}" >&2
    rm -rf "${tmp_dir}"
    return 3
  }
  cpu_so="${tmp_dir}/${cpu_rel}"
  hip_so="${tmp_dir}/${hip_rel}"

  if readelf -dW "${cpu_so}" | rg -q "NEEDED.*libomp"; then
    echo "Runtime check OK: libtorch_cpu.so depends on libomp."
  elif nm -D --undefined-only "${cpu_so}" | rg -q "__kmpc_"; then
    echo "Runtime note: libtorch_cpu.so leaves __kmpc_* unresolved." >&2
    echo "              Use the ROCm runtime env so /opt/rocm/lib/llvm/lib/libomp.so is on the loader path or preloaded." >&2
  else
    echo "Runtime note: libtorch_cpu.so has no explicit libomp dependency." >&2
  fi

  if readelf -dW "${hip_so}" | rg -q "NEEDED.*libhipblaslt"; then
    echo "Runtime check FAILED: libtorch_hip.so still depends on libhipblaslt" >&2
    rm -rf "${tmp_dir}"
    return 4
  fi

  echo "Runtime check OK: libtorch_hip.so has no libhipblaslt dependency."
  rm -rf "${tmp_dir}"
}

restore_latest_backup() {
  local wheel_name="$1"
  local target="${DEST_DIR}/${wheel_name}"
  local latest
  latest="$(ls -1t "${target}".bak_* 2>/dev/null | head -n1 || true)"
  if [[ -z "${latest}" ]]; then
    echo "No backup found for ${target}" >&2
    exit 1
  fi
  echo "Restoring backup:"
  echo "  from: ${latest}"
  echo "  to:   ${target}"
  "${SUDO[@]}" cp -f "${latest}" "${target}"
  "${SUDO[@]}" sha256sum "${target}"
  ls -lh "${target}"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--restore" ]]; then
  if [[ -n "${2:-}" ]]; then
    WHEEL_NAME="$(basename "${2}")"
  else
    LATEST="$(find_latest_wheel)"
    if [[ -z "${LATEST}" ]]; then
      echo "No wheel found in ${SRC_DIR_PRIMARY} or ${SRC_DIR_FALLBACK}" >&2
      exit 1
    fi
    WHEEL_NAME="$(basename "${LATEST}")"
  fi
  restore_latest_backup "${WHEEL_NAME}"
  exit 0
fi

SRC_WHEEL="${1:-}"
if [[ -z "${SRC_WHEEL}" ]]; then
  SRC_WHEEL="$(find_latest_wheel)"
fi
if [[ -z "${SRC_WHEEL}" || ! -f "${SRC_WHEEL}" ]]; then
  echo "Source wheel not found: ${SRC_WHEEL:-<empty>}" >&2
  echo "Checked ${SRC_DIR_PRIMARY} and ${SRC_DIR_FALLBACK}" >&2
  usage >&2
  exit 1
fi

verify_wheel_runtime_contract "${SRC_WHEEL}"

DEST_WHEEL="${DEST_DIR}/$(basename "${SRC_WHEEL}")"

echo "Source: ${SRC_WHEEL}"
echo "Target: ${DEST_WHEEL}"
echo "Backup root: ${BACKUP_ROOT}"

"${SUDO[@]}" mkdir -p "${DEST_DIR}"
mkdir -p "${BACKUP_ROOT}"

if [[ -d "${DEST_DIR}" ]]; then
  echo "Directory backup: ${BACKUP_ROOT}/$(basename "${DEST_DIR}")"
  "${SUDO[@]}" rsync -a "${DEST_DIR}/" "${BACKUP_ROOT}/$(basename "${DEST_DIR}")/"
fi

if [[ -f "${DEST_WHEEL}" ]]; then
  BACKUP="${DEST_WHEEL}.bak_${TS}"
  echo "Backup: ${BACKUP}"
  "${SUDO[@]}" cp -f "${DEST_WHEEL}" "${BACKUP}"
fi

"${SUDO[@]}" cp -f "${SRC_WHEEL}" "${DEST_WHEEL}"

echo
echo "SHA256:"
sha256sum "${SRC_WHEEL}"
"${SUDO[@]}" sha256sum "${DEST_WHEEL}"

echo
echo "Installed wheel:"
ls -lh "${DEST_WHEEL}"
