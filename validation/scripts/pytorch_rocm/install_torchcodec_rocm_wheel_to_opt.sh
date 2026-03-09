#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
SRC_DIR_PRIMARY="${SRC_DIR_PRIMARY:-${ROOT}/validation/workspace/cache/wheels/pytorch_rocm711}"
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
  echo "  $0 [path/to/torchcodec-*.whl]"
  echo "  $0 --restore [torchcodec-*.whl]"
}

find_latest_wheel() {
  ls -1t "${SRC_DIR_PRIMARY}"/torchcodec-*.whl 2>/dev/null | head -n1 || true
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
  "${SUDO[@]}" ln -sfn "${wheel_name}" "${DEST_DIR}/torchcodec-current.whl"
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
      echo "No torchcodec wheel found in ${SRC_DIR_PRIMARY}" >&2
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
  echo "Checked ${SRC_DIR_PRIMARY}" >&2
  usage >&2
  exit 1
fi

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
"${SUDO[@]}" ln -sfn "$(basename "${DEST_WHEEL}")" "${DEST_DIR}/torchcodec-current.whl"

echo
echo "SHA256:"
sha256sum "${SRC_WHEEL}"
"${SUDO[@]}" sha256sum "${DEST_WHEEL}"

echo
echo "Installed wheel:"
ls -lh "${DEST_WHEEL}"
ls -lh "${DEST_DIR}/torchcodec-current.whl"

