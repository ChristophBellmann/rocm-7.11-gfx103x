#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
LEGACY_WORKSPACE_DIR="${WORKSPACE_DIR:-}"
RELEASE_ROOT="${RELEASE_ROOT:-${LEGACY_WORKSPACE_DIR:-${ROOT}/validation/workspace/cache}}"
WORKSPACE_DIR="${RELEASE_ROOT}"
SRC_DIR_PRIMARY="${SRC_DIR_PRIMARY:-${WORKSPACE_DIR}/wheels/pytorch_rocm711}"
SRC_DIR_FALLBACK="${SRC_DIR_FALLBACK:-${ROOT}/dist}"
DEST_DIR="${DEST_DIR:-/opt/rocm/wheels/pytorch_rocm711}"
TS="$(date +%Y%m%d_%H%M%S)"
BACKUP_ROOT="${BACKUP_ROOT:-${WORKSPACE_DIR}/install-backups/${TS}/pytorch_wheels}"

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

sanitize_wheel_rpaths_for_system_rocm() {
  local src_wheel="$1"
  local out_wheel="$2"
  local tmp_dir unpack_dir
  tmp_dir="$(mktemp -d)"
  unpack_dir="${tmp_dir}/wheel"
  mkdir -p "${unpack_dir}"

  python3 - <<'PY' "${src_wheel}" "${unpack_dir}"
import sys
import zipfile

wheel, out_dir = sys.argv[1:3]
with zipfile.ZipFile(wheel) as zf:
    zf.extractall(out_dir)
PY

  local torch_c
  torch_c="$(find "${unpack_dir}/torch" -maxdepth 1 -type f -name '_C*.so' | head -n1 || true)"
  if [[ -n "${torch_c}" ]]; then
    patchelf --force-rpath --set-rpath '$ORIGIN:$ORIGIN/lib:/opt/rocm/lib:/opt/rocm/lib/llvm/lib' "${torch_c}"
  fi

  while IFS= read -r sofile; do
    patchelf --force-rpath --set-rpath '$ORIGIN:/opt/rocm/lib:/opt/rocm/lib/llvm/lib' "${sofile}"
  done < <(find "${unpack_dir}/torch/lib" -maxdepth 1 -type f \( -name '*.so' -o -name '*.so.*' \) | sort)

  local cpu_so="${unpack_dir}/torch/lib/libtorch_cpu.so"
  if [[ -f "${cpu_so}" ]]; then
    if readelf -dW "${cpu_so}" | rg -q "NEEDED.*libgomp\\.so\\.1"; then
      patchelf --replace-needed libgomp.so.1 libomp.so "${cpu_so}"
    fi
    if nm -D --undefined-only "${cpu_so}" | rg -q "__kmpc_" \
      && ! readelf -dW "${cpu_so}" | rg -q "NEEDED.*libomp"; then
      patchelf --add-needed libomp.so "${cpu_so}"
    fi
  fi

  local hip_so="${unpack_dir}/torch/lib/libtorch_hip.so"
  if [[ -f "${hip_so}" ]] \
    && nm -D --undefined-only "${hip_so}" | rg -q "rsmi_init" \
    && ! readelf -dW "${hip_so}" | rg -q "NEEDED.*librocm_smi64"; then
    patchelf --add-needed librocm_smi64.so.1 "${hip_so}"
  fi

  python3 - <<'PY' "${unpack_dir}" "${out_wheel}"
import os
import sys
import zipfile

src_dir, out_wheel = sys.argv[1:3]
with zipfile.ZipFile(out_wheel, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(src_dir):
        for name in files:
            full = os.path.join(root, name)
            rel = os.path.relpath(full, src_dir)
            zf.write(full, rel)
PY

  rm -rf "${tmp_dir}"
}

verify_wheel_system_rpath_contract() {
  local wheel_path="$1"
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  python3 - <<'PY' "${wheel_path}" "${tmp_dir}"
import sys
import zipfile

wheel, out_dir = sys.argv[1:3]
with zipfile.ZipFile(wheel) as zf:
    zf.extractall(out_dir)
PY

  local bad=0
  while IFS= read -r sofile; do
    local rp
    rp="$(patchelf --print-rpath "${sofile}" 2>/dev/null || true)"
    echo "RPATH $(basename "${sofile}") => ${rp}"
    if [[ "${rp}" == *"/build-stage2/dist/rocm/lib"* ]]; then
      echo "System wheel check FAILED: stale in-tree RPATH remains in ${sofile}" >&2
      bad=1
    fi
  done < <(find "${tmp_dir}/torch" -type f \( -name '*.so' -o -name '*.so.*' \) | sort)

  rm -rf "${tmp_dir}"
  [[ "${bad}" -eq 0 ]]
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
    echo "Runtime check FAILED: libtorch_cpu.so leaves __kmpc_* unresolved without libomp." >&2
    rm -rf "${tmp_dir}"
    return 5
  else
    echo "Runtime note: libtorch_cpu.so has no OpenMP runtime dependency."
  fi

  if nm -D --undefined-only "${hip_so}" | rg -q "rsmi_init" \
    && ! readelf -dW "${hip_so}" | rg -q "NEEDED.*librocm_smi64"; then
    echo "Runtime check FAILED: libtorch_hip.so references rsmi_init without librocm_smi64." >&2
    rm -rf "${tmp_dir}"
    return 6
  fi
  echo "Runtime check OK: libtorch_hip.so declares required SMI runtime if needed."

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
  "${SUDO[@]}" ln -sfn "${wheel_name}" "${DEST_DIR}/torch-current.whl"
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

PATCHED_WHEEL="$(mktemp --suffix=.whl)"
sanitize_wheel_rpaths_for_system_rocm "${SRC_WHEEL}" "${PATCHED_WHEEL}"
verify_wheel_system_rpath_contract "${PATCHED_WHEEL}"
verify_wheel_runtime_contract "${PATCHED_WHEEL}"

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

"${SUDO[@]}" cp -f "${PATCHED_WHEEL}" "${DEST_WHEEL}"
"${SUDO[@]}" ln -sfn "$(basename "${DEST_WHEEL}")" "${DEST_DIR}/torch-current.whl"

echo
echo "SHA256:"
echo "Original:"
sha256sum "${SRC_WHEEL}"
echo "Installed copy:"
"${SUDO[@]}" sha256sum "${DEST_WHEEL}"
"${SUDO[@]}" rm -f "${PATCHED_WHEEL}"

echo
echo "Installed wheel:"
ls -lh "${DEST_WHEEL}"
ls -lh "${DEST_DIR}/torch-current.whl"
