#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults (overridden by config YAML, env, or CLI flags)
CONFIG_FILE="${CONFIG_FILE:-${ROOT}/config_gfx1031.yaml}"
LOG_FILE="${LOG_FILE:-${ROOT}/build.log}"
BUILD_DIR="${BUILD_DIR:-}"
STAGE="${STAGE:-}"
STAGE1_BUILD_DIR="${STAGE1_BUILD_DIR:-}"
MEM_HIGH="${MEM_HIGH:-}"
MEM_MAX="${MEM_MAX:-}"
PRESERVE_LD_LIBRARY_PATH="${PRESERVE_LD_LIBRARY_PATH:-}"
JOBS="${JOBS:-}"
AUTO_FETCH_SOURCES="${AUTO_FETCH_SOURCES:-}"
AUTO_APPLY_PATCHES="${AUTO_APPLY_PATCHES:-}"
PATCH_TAG="${PATCH_TAG:-}"
PATCH_COMPILER_PROJECTS="${PATCH_COMPILER_PROJECTS:-}"
DETACH=0
WAIT_LOCK=0

# Track whether the user explicitly selected a stage/build dir via environment
# variables before we load config defaults.
ENV_BUILD_DIR_SET=0
ENV_STAGE_SET=0
if [[ -n "${BUILD_DIR}" ]]; then ENV_BUILD_DIR_SET=1; fi
if [[ -n "${STAGE}" ]]; then ENV_STAGE_SET=1; fi

# Fallback defaults if config YAML doesn't set them.
DEFAULT_THEROCK_AMDGPU_TARGETS="gfx1031"
DEFAULT_BUILD_DIR="build"
DEFAULT_STAGE="1"
DEFAULT_STAGE1_BUILD_DIR="build-stage1"
DEFAULT_MEM_HIGH="28G"
DEFAULT_MEM_MAX="31G"
DEFAULT_PRESERVE_LD_LIBRARY_PATH="0"
DEFAULT_AUTO_FETCH_SOURCES="true"
DEFAULT_AUTO_APPLY_PATCHES="true"
DEFAULT_PATCH_TAG="amd-mainline"
DEFAULT_PATCH_COMPILER_PROJECTS="spirv-llvm-translator"

# Feature defaults (config YAML should normally set these; env overrides always win).
DEFAULT_ENABLE_COMPILER="true"
DEFAULT_ENABLE_CORE_RUNTIME="true"
DEFAULT_ENABLE_HIP_RUNTIME="true"
DEFAULT_ENABLE_OCL_RUNTIME="false"
DEFAULT_ENABLE_HIPIFY="true"
DEFAULT_ENABLE_BLAS="true"
DEFAULT_ENABLE_PRIM="true"
DEFAULT_ENABLE_RAND="true"
DEFAULT_ENABLE_FFT="true"
DEFAULT_ENABLE_SPARSE="true"
DEFAULT_ENABLE_SOLVER="true"
DEFAULT_ENABLE_HIPBLASLT="false"
DEFAULT_ENABLE_HIPSPARSELT="false"
DEFAULT_ENABLE_MIOPEN="true"
DEFAULT_ENABLE_HIPDNN="true"
DEFAULT_ENABLE_COMPOSABLE_KERNEL="true"
DEFAULT_ENABLE_RCCL="true"
DEFAULT_ENABLE_ROCWMMA="false"
DEFAULT_ENABLE_PROFILER="true"
DEFAULT_ENABLE_DC_TOOLS="false"
DEFAULT_ENABLE_BUILD_TESTING="false"
DEFAULT_ENABLE_ROCPROFSYS="false"
DEFAULT_ENABLE_BENCHMARKS="false"

usage() {
  cat <<'EOF_USAGE'
Usage: build_gfx1031.sh <command> [options] [args...]

Purpose:
  Reproducible TheRock builds for gfx1031 with:
  - venv + ccache setup
  - systemd memory limits (MemoryHigh/MemoryMax)
  - per-build-dir locking (prevents concurrent build corruption)
  - strict in-tree ROCm/HIP roots (avoid /opt/rocm mixing)

Commands:
  configure         Top-level CMake configure (Stage-1/Stage-2 supported)
  configure-sub     (Re)configure specific subprojects only (<name>+configure)
  bootstrap         Build early sysdeps (+dist) and verify outputs
  build             Build (Stage-1 in build-stage1 defaults to toolchain; Stage-2 defaults to full graph; or pass ninja targets)
  rebuild           Expunge + rebuild specific subprojects
  expunge           Expunge specific subprojects (no rebuild)
  rocprofiler-gcc   Phase-2: build rocprofiler-systems with GCC in a separate build dir

Default behavior (no --stage*/--build-dir and no explicit targets):
  configure         Stage-1 configure; if Stage-1 toolchain exists, also Stage-2 configure
  build             Stage-1 bootstrap+build; then Stage-2 configure+bootstrap+build

Quick start (fresh clone):
  ./build_gfx1031.sh configure
  ./build_gfx1031.sh build
  ./test_gfx1031.sh
  python3 validation/scripts/validate.py

Shared options:
  --config <file>          Config file (default: ./config_gfx1031.yaml)
  --stage1                Use BUILD_DIR=build-stage1 and STAGE=1
  --stage2                Use BUILD_DIR=build-stage2 and STAGE=2 (uses Stage-1 toolchain)
  --build-dir <dir>       Override build directory (default: build)
  --stage1-build-dir <d>  Stage-1 build dir used by --stage2 (default: build-stage1)
  --wait                  Wait for an in-progress build lock (per BUILD_DIR)
  -j, --jobs <n>          Ninja parallelism (optional)
  --detach                Run build in background via systemd-run (build/configure-sub only)
  -h, --help              Show this help

Configure options:
  --all                   Run configure for both Stage-1 and Stage-2 (still requires separate bootstrap/build)
  --clean                 Remove BUILD_DIR before configuring (default)
  --no-clean              Do not remove BUILD_DIR before configuring
  --no-check-clean        Skip "build dir must be empty" check
  -- <extra cmake args>   Extra args forwarded to top-level cmake

Notes:
  - Stage-2 configure requires the Stage-1 toolchain at:
      <stage1_build_dir>/compiler/amd-llvm/dist/lib/llvm/bin/{clang,clang++,lld}
  - The script does not auto-fall back to system hipcc for CMake HIP projects.
    If ./install/bin/hipcc exists, it is used for CMake-HIP-language subprojects.

Environment:
  CONFIG_FILE, LOG_FILE, BUILD_DIR, STAGE, STAGE1_BUILD_DIR
  MEM_HIGH / MEM_MAX, PRESERVE_LD_LIBRARY_PATH, JOBS
  ENABLE_* / ENABLE_BENCHMARKS and THEROCK_AMDGPU_TARGETS (defaults from config YAML)
EOF_USAGE
}

bool_on_off() {
  local v="$1"
  if [[ "${v}" == "true" ]]; then echo "ON"; else echo "OFF"; fi
}

require_cmd() {
  local exe="$1"
  local hint="$2"
  if ! command -v "${exe}" >/dev/null 2>&1; then
    echo "${exe} not found; ${hint}" >&2
    exit 1
  fi
}

ensure_venv() {
  local req_hash marker_file current_hash needs_sync=0
  if [[ ! -f "${ROOT}/.venv/bin/activate" ]]; then
    echo "Creating .venv (python3 -m venv .venv && pip install -r requirements.txt)..." | tee -a "${LOG_FILE}"
    python3 -m venv "${ROOT}/.venv"
    needs_sync=1
  fi

  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"

  marker_file="${ROOT}/.venv/.requirements.sha256"
  req_hash="$(sha256sum "${ROOT}/requirements.txt" | awk '{print $1}')"
  if [[ -f "${marker_file}" ]]; then
    current_hash="$(<"${marker_file}")"
  else
    current_hash=""
  fi
  if [[ "${current_hash}" != "${req_hash}" ]]; then
    needs_sync=1
  fi

  # Guard against manual package removal/corruption while requirements hash stays unchanged.
  if ! python3 - <<'PY' >/dev/null 2>&1
import importlib
importlib.import_module("yaml")
importlib.import_module("CppHeaderParser")
PY
  then
    needs_sync=1
  fi

  if (( needs_sync )); then
    echo "Syncing .venv requirements from requirements.txt..." | tee -a "${LOG_FILE}"
    pip install --upgrade pip
    pip install -r "${ROOT}/requirements.txt"
    printf '%s\n' "${req_hash}" > "${marker_file}"
  fi
}

setup_ccache() {
  if [[ -x "${ROOT}/.local/bin/ccache" ]]; then
    PATH="${ROOT}/.local/bin:${PATH}"
  fi
  if [[ -x "${ROOT}/build_tools/setup_ccache.py" ]]; then
    eval "$(python3 "${ROOT}/build_tools/setup_ccache.py" --init)"
  fi
  export CCACHE_SLOPPINESS="${CCACHE_SLOPPINESS:-include_file_ctime}"
  require_cmd ccache "install it or run setup_ccache.py as in README."
}

compute_sysdeps_ld_library_path() {
  local -a sysdeps_libs=(
    "${ROOT}/${BUILD_DIR}/dist/rocm/lib/rocm_sysdeps/lib"
    "${ROOT}/${BUILD_DIR}/third-party/sysdeps/linux/zstd/build/dist/lib/rocm_sysdeps/lib"
    "${ROOT}/${BUILD_DIR}/third-party/sysdeps/linux/zstd/build/stage/lib/rocm_sysdeps/lib"
    "${ROOT}/${BUILD_DIR}/third-party/sysdeps/linux/zstd/build/build/b"
    "${ROOT}/${BUILD_DIR}/third-party/sysdeps/linux/zlib/build/dist/lib/rocm_sysdeps/lib"
    "${ROOT}/${BUILD_DIR}/third-party/sysdeps/linux/zlib/build/stage/lib/rocm_sysdeps/lib"
    "${ROOT}/${BUILD_DIR}/third-party/sysdeps/linux/zlib/build/build/b"
    "${ROOT}/${BUILD_DIR}/third-party/sysdeps/linux/bzip2/build/dist/lib/rocm_sysdeps/lib"
    "${ROOT}/${BUILD_DIR}/third-party/sysdeps/linux/bzip2/build/stage/lib/rocm_sysdeps/lib"
    "${ROOT}/${BUILD_DIR}/third-party/sysdeps/linux/liblzma/build/dist/lib/rocm_sysdeps/lib"
    "${ROOT}/${BUILD_DIR}/third-party/sysdeps/linux/liblzma/build/stage/lib/rocm_sysdeps/lib"
    "${ROOT}/${BUILD_DIR}/third-party/sysdeps/linux/elfutils/build/dist/lib/rocm_sysdeps/lib"
    "${ROOT}/${BUILD_DIR}/third-party/sysdeps/linux/elfutils/build/stage/lib/rocm_sysdeps/lib"
    "${ROOT}/${BUILD_DIR}/third-party/sysdeps/linux/libdrm/build/dist/lib/rocm_sysdeps/lib"
    "${ROOT}/${BUILD_DIR}/third-party/sysdeps/linux/libdrm/build/stage/lib/rocm_sysdeps/lib"
    "${ROOT}/${BUILD_DIR}/third-party/sysdeps/linux/numactl/build/dist/lib/rocm_sysdeps/lib"
    "${ROOT}/${BUILD_DIR}/third-party/sysdeps/linux/numactl/build/stage/lib/rocm_sysdeps/lib"
  )
  local ldpath=""
  local p
  for p in "${sysdeps_libs[@]}"; do
    [[ -d "$p" ]] && ldpath="${ldpath:+$ldpath:}$p"
  done
  echo "${ldpath}"
}

run_cmd() {
  local cmdline="$1"
  local ldpath
  ldpath="$(compute_sysdeps_ld_library_path)"
  local ld_export="export LD_LIBRARY_PATH=\"${ldpath}\""
  if [[ "${PRESERVE_LD_LIBRARY_PATH}" == "1" ]]; then
    ld_export="export LD_LIBRARY_PATH=\"${ldpath:+$ldpath:}\${LD_LIBRARY_PATH}\""
  fi
  local jobs_arg=""
  if [[ -n "${JOBS}" ]]; then
    jobs_arg="-j ${JOBS}"
  fi
  if (( DETACH )); then
    local unit="therock-gfx1031-${BUILD_DIR}-${cmd}"
    systemd-run --user --no-block --quiet --collect --unit "${unit}" --property=Restart=no \
      --property="MemoryHigh=${MEM_HIGH}" --property="MemoryMax=${MEM_MAX}" \
      --property=MemoryAccounting=yes --property=CPUAccounting=yes \
      bash -lc "cd \"${ROOT}\" && source \"${ROOT}/.venv/bin/activate\" && ${ld_export} && ${cmdline} ${jobs_arg} >> \"${LOG_FILE}\" 2>&1"
    echo "Started as user unit: ${unit}.service (logs: ${LOG_FILE})"
  else
    systemd-run --user --scope -p "MemoryHigh=${MEM_HIGH}" -p "MemoryMax=${MEM_MAX}" \
      bash -lc "cd \"${ROOT}\" && source \"${ROOT}/.venv/bin/activate\" && ${ld_export} && ${cmdline} ${jobs_arg}" 2>&1 | tee -a "${LOG_FILE}"
  fi
}

run_cmd_array() {
  local -a cmdline=("$@")
  local escaped
  printf -v escaped '%q ' "${cmdline[@]}"
  run_cmd "${escaped}"
}

load_config_yaml() {
  local cfg="$1"
  if [[ ! -f "${cfg}" ]]; then
    echo "Config file not found: ${cfg}" >&2
    exit 1
  fi
  python3 - "$cfg" <<'PY'
import os, sys, shlex
try:
    import yaml
except Exception as e:
    print(f"ERROR: PyYAML not available: {e}", file=sys.stderr)
    sys.exit(1)

cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}

def get(d, *path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def emit(name, value):
    # Don't override if already set in env (user override)
    if os.environ.get(name):
        return
    if value is None:
        return
    if isinstance(value, bool):
        value = "true" if value else "false"
    else:
        value = str(value)
    print(f'export {name}={shlex.quote(value)}')

emit("THEROCK_AMDGPU_TARGETS", get(data, "amdgpu_targets"))

emit("MEM_HIGH", get(data, "memory", "high"))
emit("MEM_MAX", get(data, "memory", "max"))

emit("STAGE", get(data, "build", "stage"))
emit("BUILD_DIR", get(data, "build", "build_dir"))
emit("STAGE1_BUILD_DIR", get(data, "build", "stage1_build_dir"))
emit("JOBS", get(data, "build", "jobs"))
emit("PRESERVE_LD_LIBRARY_PATH", get(data, "build", "preserve_ld_library_path"))
emit("AUTO_FETCH_SOURCES", get(data, "build", "auto_fetch_sources"))
emit("ENABLE_BENCHMARKS", get(data, "build", "benchmarks"))

emit("AUTO_APPLY_PATCHES", get(data, "patches", "auto_apply"))
emit("PATCH_TAG", get(data, "patches", "tag"))
compiler_projects = get(data, "patches", "compiler_projects", default=[]) or []
if not os.environ.get("PATCH_COMPILER_PROJECTS"):
    if isinstance(compiler_projects, list) and compiler_projects:
        print(f'export PATCH_COMPILER_PROJECTS={shlex.quote(chr(31).join(str(x) for x in compiler_projects))}')
    else:
        print('export PATCH_COMPILER_PROJECTS=""')

features = get(data, "features", default={}) or {}
mapping = {
    "ENABLE_COMPILER": "enable_compiler",
    "ENABLE_CORE_RUNTIME": "enable_core_runtime",
    "ENABLE_HIP_RUNTIME": "enable_hip_runtime",
    "ENABLE_OCL_RUNTIME": "enable_ocl_runtime",
    "ENABLE_HIPIFY": "enable_hipify",
    "ENABLE_BLAS": "enable_blas",
    "ENABLE_PRIM": "enable_prim",
    "ENABLE_RAND": "enable_rand",
    "ENABLE_FFT": "enable_fft",
    "ENABLE_SPARSE": "enable_sparse",
    "ENABLE_SOLVER": "enable_solver",
    "ENABLE_HIPBLASLT": "enable_hipblaslt",
    "ENABLE_HIPSPARSELT": "enable_hipsparselt",
    "ENABLE_MIOPEN": "enable_miopen",
    "ENABLE_HIPDNN": "enable_hipdnn",
    "ENABLE_COMPOSABLE_KERNEL": "enable_composable_kernel",
    "ENABLE_RCCL": "enable_rccl",
    "ENABLE_ROCWMMA": "enable_rocwmma",
    "ENABLE_PROFILER": "enable_profiler",
    "ENABLE_DC_TOOLS": "enable_dc_tools",
    "ENABLE_BUILD_TESTING": "enable_build_testing",
    "ENABLE_ROCPROFSYS": "enable_rocprofsys",
}
for env_name, key in mapping.items():
    emit(env_name, features.get(key))

extra = get(data, "extra_cmake_args", default=[]) or []
if os.environ.get("THEROCK_EXTRA_CMAKE_ARGS"):
    sys.exit(0)
if isinstance(extra, list) and extra:
    # Join with ASCII unit separator to avoid shell quoting issues; bash splits later.
    print(f'export THEROCK_EXTRA_CMAKE_ARGS={shlex.quote(chr(31).join(str(x) for x in extra))}')
else:
    print('export THEROCK_EXTRA_CMAKE_ARGS=""')
PY
}

cmd="${1:-}"
if [[ -z "${cmd}" || "${cmd}" == "-h" || "${cmd}" == "--help" ]]; then
  usage
  exit 0
fi
shift || true

# Parse shared flags first.
DO_CLEAN=1
CHECK_CLEAN=1
EXTRA_CMAKE_ARGS=()
SUBPROJECTS=()
CONFIGURE_ALL=0
BUILD_ALL=0
USER_SELECTED_BUILD_DIR=0
USER_SELECTED_STAGE=0

# First pass: allow --config anywhere.
argv=("$@")
for ((i=0; i<${#argv[@]}; i++)); do
  if [[ "${argv[$i]}" == "--config" ]]; then
    CONFIG_FILE="${argv[$((i+1))]:-}"
    break
  fi
done

require_cmd python3 "install python3 and python3-venv."
ensure_venv

# Load config defaults (from venv python) before processing CLI overrides.
if [[ -n "${CONFIG_FILE}" ]]; then
  eval "$(load_config_yaml "${CONFIG_FILE}")"
fi

# Apply post-YAML fallbacks (only if still unset).
THEROCK_AMDGPU_TARGETS="${THEROCK_AMDGPU_TARGETS:-${DEFAULT_THEROCK_AMDGPU_TARGETS}}"
BUILD_DIR="${BUILD_DIR:-${DEFAULT_BUILD_DIR}}"
STAGE="${STAGE:-${DEFAULT_STAGE}}"
STAGE1_BUILD_DIR="${STAGE1_BUILD_DIR:-${DEFAULT_STAGE1_BUILD_DIR}}"
MEM_HIGH="${MEM_HIGH:-${DEFAULT_MEM_HIGH}}"
MEM_MAX="${MEM_MAX:-${DEFAULT_MEM_MAX}}"
PRESERVE_LD_LIBRARY_PATH="${PRESERVE_LD_LIBRARY_PATH:-${DEFAULT_PRESERVE_LD_LIBRARY_PATH}}"
AUTO_FETCH_SOURCES="${AUTO_FETCH_SOURCES:-${DEFAULT_AUTO_FETCH_SOURCES}}"
AUTO_APPLY_PATCHES="${AUTO_APPLY_PATCHES:-${DEFAULT_AUTO_APPLY_PATCHES}}"
PATCH_TAG="${PATCH_TAG:-${DEFAULT_PATCH_TAG}}"
PATCH_COMPILER_PROJECTS="${PATCH_COMPILER_PROJECTS:-${DEFAULT_PATCH_COMPILER_PROJECTS}}"

ENABLE_COMPILER="${ENABLE_COMPILER:-${DEFAULT_ENABLE_COMPILER}}"
ENABLE_CORE_RUNTIME="${ENABLE_CORE_RUNTIME:-${DEFAULT_ENABLE_CORE_RUNTIME}}"
ENABLE_HIP_RUNTIME="${ENABLE_HIP_RUNTIME:-${DEFAULT_ENABLE_HIP_RUNTIME}}"
ENABLE_OCL_RUNTIME="${ENABLE_OCL_RUNTIME:-${DEFAULT_ENABLE_OCL_RUNTIME}}"
ENABLE_HIPIFY="${ENABLE_HIPIFY:-${DEFAULT_ENABLE_HIPIFY}}"
ENABLE_BLAS="${ENABLE_BLAS:-${DEFAULT_ENABLE_BLAS}}"
ENABLE_PRIM="${ENABLE_PRIM:-${DEFAULT_ENABLE_PRIM}}"
ENABLE_RAND="${ENABLE_RAND:-${DEFAULT_ENABLE_RAND}}"
ENABLE_FFT="${ENABLE_FFT:-${DEFAULT_ENABLE_FFT}}"
ENABLE_SPARSE="${ENABLE_SPARSE:-${DEFAULT_ENABLE_SPARSE}}"
ENABLE_SOLVER="${ENABLE_SOLVER:-${DEFAULT_ENABLE_SOLVER}}"
ENABLE_HIPBLASLT="${ENABLE_HIPBLASLT:-${DEFAULT_ENABLE_HIPBLASLT}}"
ENABLE_HIPSPARSELT="${ENABLE_HIPSPARSELT:-${DEFAULT_ENABLE_HIPSPARSELT}}"
ENABLE_MIOPEN="${ENABLE_MIOPEN:-${DEFAULT_ENABLE_MIOPEN}}"
ENABLE_HIPDNN="${ENABLE_HIPDNN:-${DEFAULT_ENABLE_HIPDNN}}"
ENABLE_COMPOSABLE_KERNEL="${ENABLE_COMPOSABLE_KERNEL:-${DEFAULT_ENABLE_COMPOSABLE_KERNEL}}"
ENABLE_RCCL="${ENABLE_RCCL:-${DEFAULT_ENABLE_RCCL}}"
ENABLE_ROCWMMA="${ENABLE_ROCWMMA:-${DEFAULT_ENABLE_ROCWMMA}}"
ENABLE_PROFILER="${ENABLE_PROFILER:-${DEFAULT_ENABLE_PROFILER}}"
ENABLE_DC_TOOLS="${ENABLE_DC_TOOLS:-${DEFAULT_ENABLE_DC_TOOLS}}"
ENABLE_BUILD_TESTING="${ENABLE_BUILD_TESTING:-${DEFAULT_ENABLE_BUILD_TESTING}}"
ENABLE_ROCPROFSYS="${ENABLE_ROCPROFSYS:-${DEFAULT_ENABLE_ROCPROFSYS}}"
ENABLE_BENCHMARKS="${ENABLE_BENCHMARKS:-${DEFAULT_ENABLE_BENCHMARKS}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_FILE="${2:-}"
      shift 2
      ;;
    --stage1)
      BUILD_DIR="build-stage1"
      STAGE=1
      USER_SELECTED_BUILD_DIR=1
      USER_SELECTED_STAGE=1
      shift
      ;;
    --stage2)
      BUILD_DIR="build-stage2"
      STAGE=2
      USER_SELECTED_BUILD_DIR=1
      USER_SELECTED_STAGE=1
      shift
      ;;
    --build-dir)
      BUILD_DIR="${2:-}"
      USER_SELECTED_BUILD_DIR=1
      shift 2
      ;;
    --stage1-build-dir)
      STAGE1_BUILD_DIR="${2:-}"
      shift 2
      ;;
    --detach)
      DETACH=1
      shift
      ;;
    --wait)
      WAIT_LOCK=1
      shift
      ;;
    -j|--jobs)
      JOBS="${2:-}"
      shift 2
      ;;
    --clean)
      DO_CLEAN=1
      shift
      ;;
    --no-clean)
      DO_CLEAN=0
      shift
      ;;
    --no-check-clean)
      CHECK_CLEAN=0
      shift
      ;;
    --all)
      if [[ "${cmd}" != "configure" ]]; then
        echo "Option --all is only valid for: ./build_gfx1031.sh configure" >&2
        exit 2
      fi
      CONFIGURE_ALL=1
      shift
      ;;
    --)
      shift
      if [[ "${cmd}" == "configure" ]]; then
        EXTRA_CMAKE_ARGS+=("$@")
      else
        SUBPROJECTS+=("$@")
      fi
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      SUBPROJECTS+=("$1")
      shift
      ;;
  esac
done

# Default behavior in this fork:
# - `configure` without explicit stage/build-dir configures Stage-1 and then,
#   if the Stage-1 toolchain already exists, also configures Stage-2.
# - `build` without explicit stage/build-dir builds everything needed for tests
#   and validation: Stage-1 bootstrap+build, then Stage-2 configure+bootstrap+build.
if [[ ${ENV_BUILD_DIR_SET} -eq 0 && ${ENV_STAGE_SET} -eq 0 && ${USER_SELECTED_BUILD_DIR} -eq 0 && ${USER_SELECTED_STAGE} -eq 0 ]]; then
  if [[ "${cmd}" == "configure" ]]; then
    CONFIGURE_ALL=1
  elif [[ "${cmd}" == "build" && ${#SUBPROJECTS[@]} -eq 0 ]]; then
    BUILD_ALL=1
  fi
fi

stage1_toolchain_ok() {
  local stage1_llvm_bin="${ROOT}/${STAGE1_BUILD_DIR}/compiler/amd-llvm/dist/lib/llvm/bin"
  [[ -x "${stage1_llvm_bin}/clang" && -x "${stage1_llvm_bin}/clang++" && -x "${stage1_llvm_bin}/lld" ]]
}

# Special helper: configure both Stage-1 and Stage-2 in sequence.
# This only configures; it does not bootstrap or build.
if [[ "${cmd}" == "configure" && ${CONFIGURE_ALL} -eq 1 ]]; then
  cfg_args=(--config "${CONFIG_FILE}")
  if (( DO_CLEAN )); then cfg_args+=(--clean); else cfg_args+=(--no-clean); fi
  if (( CHECK_CLEAN == 0 )); then cfg_args+=(--no-check-clean); fi
  if [[ -n "${STAGE1_BUILD_DIR}" ]]; then cfg_args+=(--stage1-build-dir "${STAGE1_BUILD_DIR}"); fi

  echo "Configuring Stage-1..." | tee -a "${LOG_FILE}"
  if (( ${#EXTRA_CMAKE_ARGS[@]} > 0 )); then
    "${ROOT}/build_gfx1031.sh" configure --stage1 "${cfg_args[@]}" -- "${EXTRA_CMAKE_ARGS[@]}"
  else
    "${ROOT}/build_gfx1031.sh" configure --stage1 "${cfg_args[@]}"
  fi

  if ! stage1_toolchain_ok; then
    echo "" | tee -a "${LOG_FILE}"
    echo "Stage-1 toolchain not built yet; skipping Stage-2 configure for now." | tee -a "${LOG_FILE}"
    echo "Next:" | tee -a "${LOG_FILE}"
    echo "  ./build_gfx1031.sh bootstrap --stage1 && ./build_gfx1031.sh build --stage1" | tee -a "${LOG_FILE}"
    echo "  ./build_gfx1031.sh configure --stage2 && ./build_gfx1031.sh bootstrap --stage2 && ./build_gfx1031.sh build --stage2" | tee -a "${LOG_FILE}"
    exit 0
  fi

  echo "" | tee -a "${LOG_FILE}"
  echo "Configuring Stage-2..." | tee -a "${LOG_FILE}"
  if (( ${#EXTRA_CMAKE_ARGS[@]} > 0 )); then
    "${ROOT}/build_gfx1031.sh" configure --stage2 "${cfg_args[@]}" -- "${EXTRA_CMAKE_ARGS[@]}"
  else
    "${ROOT}/build_gfx1031.sh" configure --stage2 "${cfg_args[@]}"
  fi

  echo "" | tee -a "${LOG_FILE}"
  echo "Configure-all complete. Next:" | tee -a "${LOG_FILE}"
  echo "  ./build_gfx1031.sh bootstrap --stage1 && ./build_gfx1031.sh build --stage1" | tee -a "${LOG_FILE}"
  echo "  ./build_gfx1031.sh bootstrap --stage2 && ./build_gfx1031.sh build --stage2" | tee -a "${LOG_FILE}"
  exit 0
fi

if [[ "${cmd}" == "build" && ${BUILD_ALL} -eq 1 ]]; then
  echo "Building full pipeline (Stage-1 -> Stage-2)..." | tee -a "${LOG_FILE}"

  # Stage-1: configure if needed, then bootstrap + build.
  if [[ ! -f "${ROOT}/build-stage1/build.ninja" ]]; then
    "${ROOT}/build_gfx1031.sh" configure --stage1 --config "${CONFIG_FILE}"
  fi
  if [[ ! -f "${ROOT}/build-stage1/${BOOTSTRAP_OK_MARKER_NAME}" ]]; then
    "${ROOT}/build_gfx1031.sh" bootstrap --stage1 --config "${CONFIG_FILE}"
  fi
  "${ROOT}/build_gfx1031.sh" build --stage1 --config "${CONFIG_FILE}"

  if ! stage1_toolchain_ok; then
    echo "ERROR: Stage-1 toolchain missing after Stage-1 build; cannot proceed to Stage-2 configure." >&2
    exit 1
  fi

  # Stage-2: configure now that Stage-1 toolchain exists, then bootstrap + build.
  if [[ ! -f "${ROOT}/build-stage2/build.ninja" ]]; then
    "${ROOT}/build_gfx1031.sh" configure --stage2 --config "${CONFIG_FILE}"
  fi
  if [[ ! -f "${ROOT}/build-stage2/${BOOTSTRAP_OK_MARKER_NAME}" ]]; then
    "${ROOT}/build_gfx1031.sh" bootstrap --stage2 --config "${CONFIG_FILE}"
  fi
  "${ROOT}/build_gfx1031.sh" build --stage2 --config "${CONFIG_FILE}"

  echo "Full build complete. Next:" | tee -a "${LOG_FILE}"
  echo "  ./test_gfx1031.sh --stage2" | tee -a "${LOG_FILE}"
  echo "  python3 validation/scripts/validate.py" | tee -a "${LOG_FILE}"
  exit 0
fi

# Per-builddir lock (prevents concurrent runs touching the same BUILD_DIR).
mkdir -p "${ROOT}/.locks"
LOCK_FILE="${ROOT}/.locks/therock_${BUILD_DIR}.lock"
LOCK_FD=200
exec {LOCK_FD}>"${LOCK_FILE}"
if (( WAIT_LOCK )); then
  flock "${LOCK_FD}"
else
  if ! flock -n "${LOCK_FD}"; then
    echo "Another build is already running for BUILD_DIR='${BUILD_DIR}' (lock: ${LOCK_FILE})." >&2
    echo "Use: ./build_gfx1031.sh ${cmd} --build-dir ${BUILD_DIR} --wait" >&2
    exit 3
  fi
fi

# Setup prerequisites (ccache) for all commands.
setup_ccache
require_cmd ninja "install it before building."
require_cmd cmake "install CMake (system /usr/bin/cmake recommended)."

bool_is_true() {
  local v="${1:-}"
  [[ "${v}" == "1" || "${v}" == "true" || "${v}" == "True" || "${v}" == "TRUE" || "${v}" == "yes" || "${v}" == "YES" || "${v}" == "on" || "${v}" == "ON" ]]
}

have_submodule_content() {
  local path="$1"
  # Submodules have a .git file/dir. Also check for "not empty" to avoid
  # corner cases where a directory exists but wasn't initialized.
  if [[ -d "${ROOT}/${path}" && -n "$(ls -A "${ROOT}/${path}" 2>/dev/null)" ]]; then
    [[ -e "${ROOT}/${path}/.git" ]]
    return $?
  fi
  return 1
}

ensure_sources() {
  # If this is a fresh clone without initialized submodules, bootstrap them.
  local need_update=0
  if ! have_submodule_content "rocm-libraries" || ! have_submodule_content "rocm-systems" || ! have_submodule_content "compiler/amd-llvm"; then
    need_update=1
  fi

  if (( need_update == 0 )); then
    return 0
  fi

  if ! bool_is_true "${AUTO_FETCH_SOURCES}"; then
    echo "Missing sources/submodules. Enable AUTO_FETCH_SOURCES=true or run:" >&2
    echo "  python3 ./build_tools/fetch_sources.py" >&2
    exit 1
  fi

  echo "Sources/submodules missing (fresh clone). Running fetch_sources.py..." | tee -a "${LOG_FILE}"
  python3 "${ROOT}/build_tools/fetch_sources.py" \
    --update-submodules --no-remote \
    --no-apply-patches \
    --include-system-projects \
    --include-compilers \
    --include-rocm-libraries \
    --include-rocm-systems \
    --include-ml-frameworks \
    --include-rocm-media | tee -a "${LOG_FILE}"

  if bool_is_true "${AUTO_APPLY_PATCHES}" && [[ -n "${PATCH_TAG:-}" ]]; then
    # Apply only the patches needed for this fork's workflow by default.
    if [[ -n "${PATCH_COMPILER_PROJECTS:-}" ]]; then
      local -a projs=()
      IFS=$'\x1f' read -r -a projs <<<"${PATCH_COMPILER_PROJECTS}"
      if [[ ${#projs[@]} -gt 0 ]]; then
        echo "Applying compiler patches (tag=${PATCH_TAG}): ${projs[*]}" | tee -a "${LOG_FILE}"
        python3 "${ROOT}/build_tools/fetch_sources.py" \
          --no-update-submodules --no-remote \
          --apply-patches --patch-tag "${PATCH_TAG}" \
          --no-include-system-projects \
          --include-compilers --no-include-rocm-libraries --no-include-rocm-systems --no-include-ml-frameworks --no-include-rocm-media \
          --compiler-projects "${projs[@]}" | tee -a "${LOG_FILE}"
      fi
    fi
  fi
}

ensure_sources

if [[ "${cmd}" != "configure" && "${cmd}" != "rocprofiler-gcc" ]]; then
  if [[ ! -f "${ROOT}/${BUILD_DIR}/build.ninja" ]]; then
    echo "Missing ${BUILD_DIR}/build.ninja; run: ./build_gfx1031.sh configure --build-dir ${BUILD_DIR}" >&2
    exit 1
  fi
fi

bootstrap_targets=(
  "rocm-cmake+dist"
  "therock-zlib+dist"
  "therock-zstd+dist"
  "therock-numactl+dist"
  "therock-elfutils+dist"
  "therock-host-blas+dist"
  "therock-fmt+dist"
  "therock-spdlog+dist"
  "therock-yaml-cpp+dist"
  "therock-nlohmann-json+dist"
  "therock-eigen+dist"
  "therock-FunctionalPlus+dist"
)

stage1_default_targets=(
  "amd-llvm+dist"
  "hip-clr+dist"
)

BOOTSTRAP_OK_MARKER_NAME=".therock_bootstrap.ok"

verify_bootstrap() {
  local -a expect_paths=(
    "${ROOT}/${BUILD_DIR}/base/rocm-cmake/dist/share/rocmcmakebuildtools/cmake"
    "${ROOT}/${BUILD_DIR}/base/rocm-cmake/dist/share/rocm/cmake"
    "${ROOT}/${BUILD_DIR}/third-party/sysdeps/linux/zlib/build/dist/lib/rocm_sysdeps/lib/cmake/ZLIB/zlib-config.cmake"
    "${ROOT}/${BUILD_DIR}/third-party/sysdeps/linux/zlib/build/dist/lib/rocm_sysdeps/lib/librocm_sysdeps_z.so.1"
    "${ROOT}/${BUILD_DIR}/third-party/sysdeps/linux/zstd/build/dist/lib/rocm_sysdeps/lib/cmake/zstd/zstdConfig.cmake"
    "${ROOT}/${BUILD_DIR}/third-party/sysdeps/linux/zstd/build/dist/lib/rocm_sysdeps/lib/librocm_sysdeps_zstd.so.1"
    "${ROOT}/${BUILD_DIR}/third-party/sysdeps/linux/numactl/build/dist/lib/rocm_sysdeps/lib/cmake/NUMA/numa-config.cmake"
    "${ROOT}/${BUILD_DIR}/third-party/sysdeps/linux/elfutils/build/dist/lib/rocm_sysdeps/lib/cmake/LibElf/libelf-config.cmake"
    "${ROOT}/${BUILD_DIR}/third-party/host-blas/dist/lib/host-math/lib/cmake/OpenBLAS/OpenBLASConfig.cmake"
  )
  local missing=0
  local p
  for p in "${expect_paths[@]}"; do
    if [[ ! -e "${p}" ]]; then
      echo "MISSING: ${p}" | tee -a "${LOG_FILE}"
      missing=1
    else
      echo "OK: ${p}" | tee -a "${LOG_FILE}"
    fi
  done
  return "${missing}"
}

require_bootstrap_ok() {
  local marker="${ROOT}/${BUILD_DIR}/${BOOTSTRAP_OK_MARKER_NAME}"
  if [[ ! -f "${marker}" ]]; then
    echo "Bootstrap not verified for BUILD_DIR='${BUILD_DIR}' (missing ${marker})." >&2
    echo "Run: ./build_gfx1031.sh bootstrap --build-dir ${BUILD_DIR}" >&2
    exit 2
  fi
}

configure_top() {
  local targets="${THEROCK_AMDGPU_TARGETS:-gfx1031}"
  local build_path="${ROOT}/${BUILD_DIR}"

  if [[ -f "${LOG_FILE}" ]]; then
    local ts
    ts="$(date +%Y%m%d-%H%M%S)"
    mv "${LOG_FILE}" "${LOG_FILE}.bak-${ts}"
  fi

  if (( DO_CLEAN )); then
    rm -rf "${build_path}"
  fi
  if (( CHECK_CLEAN )) && [[ -d "${build_path}" ]] && [[ -n "$(ls -A "${build_path}" 2>/dev/null)" ]]; then
    echo "${BUILD_DIR}/ is not clean. Use --clean or --no-check-clean." >&2
    exit 1
  fi

  # Host compiler selection. Use absolute paths so cmake --regenerate-during-build
  # does not depend on PATH.
  local c_compiler=""
  local cxx_compiler=""
  local linker=""
  local ar=""
  local ranlib=""
  local nm=""

  if [[ "${STAGE}" == "2" ]]; then
    local stage1_llvm_bin="${ROOT}/${STAGE1_BUILD_DIR}/compiler/amd-llvm/dist/lib/llvm/bin"
    if [[ ! -x "${stage1_llvm_bin}/clang" || ! -x "${stage1_llvm_bin}/clang++" || ! -x "${stage1_llvm_bin}/lld" ]]; then
      echo "STAGE=2 requires Stage-1 toolchain in ${stage1_llvm_bin} (missing clang/clang++/lld)." >&2
      exit 1
    fi
    c_compiler="${stage1_llvm_bin}/clang"
    cxx_compiler="${stage1_llvm_bin}/clang++"
    linker="${stage1_llvm_bin}/lld"
    ar="${stage1_llvm_bin}/llvm-ar"
    ranlib="${stage1_llvm_bin}/llvm-ranlib"
    nm="${stage1_llvm_bin}/llvm-nm"
  else
    # Prefer explicit llvm-18 if present.
    if [[ -x "/usr/lib/llvm-18/bin/clang" && -x "/usr/lib/llvm-18/bin/clang++" ]]; then
      c_compiler="/usr/lib/llvm-18/bin/clang"
      cxx_compiler="/usr/lib/llvm-18/bin/clang++"
    else
      c_compiler="$(command -v clang || true)"
      cxx_compiler="$(command -v clang++ || true)"
    fi
  fi

  if [[ -z "${c_compiler}" || -z "${cxx_compiler}" ]]; then
    echo "clang/clang++ not found; install clang-18 (or provide clang in PATH)." >&2
    exit 1
  fi

  # HIP compiler selection for CMake HIP-language projects:
  # Prefer in-tree toolchain from ./install if present. Do NOT auto-fall back to system hipcc.
  local hip_compiler=""
  local rocm_prefix="${ROOT}/install"
  if [[ -x "${rocm_prefix}/bin/hipcc" ]]; then
    hip_compiler="${rocm_prefix}/bin/hipcc"
    echo "Using in-tree hipcc for CMake HIP projects: ${hip_compiler}" | tee -a "${LOG_FILE}"
  else
    echo "INFO: ${rocm_prefix}/bin/hipcc not found yet (expected on first bootstrap). Leaving CMAKE_HIP_COMPILER unset; TheRock HIP subprojects use COMPILER_TOOLCHAIN=amd-hip internally." | tee -a "${LOG_FILE}"
  fi

  local -a cmake_args=(
    "-DTHEROCK_AMDGPU_TARGETS=${targets}"
    "-DTHEROCK_DIST_AMDGPU_TARGETS=${targets}"
    "-DTHEROCK_DIST_AMDGPU_FAMILIES=${targets}"
    "-DDEFAULT_ROCM_PATH=${build_path}/core/clr/dist"
    "-DROCM_PATH=${build_path}/core/clr/dist"
    "-DROCM_DIR=${build_path}/core/clr/dist"
    "-DROCM_ROOT=${build_path}/core/clr/dist"
    "-DHIP_ROOT_DIR=${build_path}/core/clr/dist"
    "-DHIP_DIR=${build_path}/core/clr/dist"
    "-DHIP_PATH=${build_path}/core/clr/dist"
    "-DTHEROCK_ENABLE_ALL=OFF"
    "-DTHEROCK_ENABLE_COMPILER=$(bool_on_off "${ENABLE_COMPILER}")"
    "-DTHEROCK_ENABLE_CORE_RUNTIME=$(bool_on_off "${ENABLE_CORE_RUNTIME}")"
    "-DTHEROCK_ENABLE_HIP_RUNTIME=$(bool_on_off "${ENABLE_HIP_RUNTIME}")"
    "-DTHEROCK_ENABLE_OCL_RUNTIME=$(bool_on_off "${ENABLE_OCL_RUNTIME}")"
    "-DTHEROCK_ENABLE_HIPIFY=$(bool_on_off "${ENABLE_HIPIFY}")"
    "-DTHEROCK_ENABLE_BLAS=$(bool_on_off "${ENABLE_BLAS}")"
    "-DTHEROCK_ENABLE_PRIM=$(bool_on_off "${ENABLE_PRIM}")"
    "-DTHEROCK_ENABLE_RAND=$(bool_on_off "${ENABLE_RAND}")"
    "-DTHEROCK_ENABLE_FFT=$(bool_on_off "${ENABLE_FFT}")"
    "-DTHEROCK_ENABLE_SPARSE=$(bool_on_off "${ENABLE_SPARSE}")"
    "-DTHEROCK_ENABLE_SOLVER=$(bool_on_off "${ENABLE_SOLVER}")"
    "-DTHEROCK_ENABLE_HIPBLASLT=$(bool_on_off "${ENABLE_HIPBLASLT}")"
    "-DTHEROCK_ENABLE_HIPSPARSELT=$(bool_on_off "${ENABLE_HIPSPARSELT}")"
    "-DTHEROCK_ENABLE_MIOPEN=$(bool_on_off "${ENABLE_MIOPEN}")"
    "-DTHEROCK_ENABLE_HIPDNN=$(bool_on_off "${ENABLE_HIPDNN}")"
    "-DTHEROCK_ENABLE_COMPOSABLE_KERNEL=$(bool_on_off "${ENABLE_COMPOSABLE_KERNEL}")"
    "-DTHEROCK_ENABLE_RCCL=$(bool_on_off "${ENABLE_RCCL}")"
    "-DTHEROCK_ENABLE_ROCWMMA=$(bool_on_off "${ENABLE_ROCWMMA}")"
    "-DTHEROCK_ENABLE_PROFILER=$(bool_on_off "${ENABLE_PROFILER}")"
    "-DTHEROCK_ENABLE_ROCPROFSYS=$(bool_on_off "${ENABLE_ROCPROFSYS}")"
    "-DTHEROCK_ENABLE_DC_TOOLS=$(bool_on_off "${ENABLE_DC_TOOLS}")"
    "-DBUILD_TESTING=$(bool_on_off "${ENABLE_BUILD_TESTING}")"
    "-DTHEROCK_BUILD_BENCHMARKS=$(bool_on_off "${ENABLE_BENCHMARKS}")"
    "-DTHEROCK_MIOPEN_USE_COMPOSABLE_KERNEL=$(bool_on_off "${ENABLE_COMPOSABLE_KERNEL}")"
    "-DCMAKE_C_FLAGS="
    "-DCMAKE_CXX_FLAGS="
    "-DCMAKE_C_COMPILER:FILEPATH=${c_compiler}"
    "-DCMAKE_CXX_COMPILER:FILEPATH=${cxx_compiler}"
    "-DCMAKE_C_COMPILER_LAUNCHER=ccache"
    "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
  )
  local venv_python="${ROOT}/.venv/bin/python3"
  if [[ -x "${venv_python}" ]]; then
    cmake_args+=("-DPython3_EXECUTABLE:FILEPATH=${venv_python}")
    cmake_args+=("-DPYTHON_EXECUTABLE:FILEPATH=${venv_python}")
  fi
  if [[ -n "${linker}" ]]; then cmake_args+=("-DCMAKE_LINKER:FILEPATH=${linker}"); fi
  if [[ -x "${ar}" ]]; then cmake_args+=("-DCMAKE_AR:FILEPATH=${ar}"); fi
  if [[ -x "${ranlib}" ]]; then cmake_args+=("-DCMAKE_RANLIB:FILEPATH=${ranlib}"); fi
  if [[ -x "${nm}" ]]; then cmake_args+=("-DCMAKE_NM:FILEPATH=${nm}"); fi
  if [[ -n "${hip_compiler}" ]]; then cmake_args+=("-DCMAKE_HIP_COMPILER:FILEPATH=${hip_compiler}"); fi
  # Extra args from config (unit separator split) + CLI.
  if [[ -n "${THEROCK_EXTRA_CMAKE_ARGS:-}" ]]; then
    IFS=$'\x1f' read -r -a _cfg_extra <<<"${THEROCK_EXTRA_CMAKE_ARGS}"
    cmake_args+=("${_cfg_extra[@]}")
  fi
  cmake_args+=("${EXTRA_CMAKE_ARGS[@]}")

  systemd-run --user --scope -p "MemoryHigh=${MEM_HIGH}" -p "MemoryMax=${MEM_MAX}" \
    bash -lc "cd \"${ROOT}\" && source \"${ROOT}/.venv/bin/activate\" && cmake -B \"${BUILD_DIR}\" -GNinja . ${cmake_args[*]}" 2>&1 | tee -a "${LOG_FILE}"

  echo "Configure complete. Next:" | tee -a "${LOG_FILE}"
  echo "  ./build_gfx1031.sh bootstrap --build-dir ${BUILD_DIR}" | tee -a "${LOG_FILE}"
  echo "  ./build_gfx1031.sh build --build-dir ${BUILD_DIR} [--detach]" | tee -a "${LOG_FILE}"
}

case "${cmd}" in
  configure)
    configure_top
    ;;
  configure-sub)
    if [[ ${#SUBPROJECTS[@]} -eq 0 ]]; then
      echo "configure-sub requires subproject names (e.g. roctracer rocPRIM rocprofiler-sdk)." >&2
      exit 2
    fi
    for t in "${SUBPROJECTS[@]}"; do
      run_cmd_array ninja -C "${BUILD_DIR}" "${t}+configure"
    done
    ;;
  bootstrap)
    echo "Bootstrapping ${#bootstrap_targets[@]} targets in ${BUILD_DIR}..." | tee -a "${LOG_FILE}"
    rm -f "${ROOT}/${BUILD_DIR}/${BOOTSTRAP_OK_MARKER_NAME}" || true
    for t in "${bootstrap_targets[@]}"; do
      echo "==> ${t}" | tee -a "${LOG_FILE}"
      run_cmd_array ninja -C "${BUILD_DIR}" "${t}"
    done
    echo "Verifying expected bootstrap artifacts..." | tee -a "${LOG_FILE}"
    if ! verify_bootstrap; then
      echo "Bootstrap incomplete (missing artifacts). See ${LOG_FILE}." >&2
      exit 1
    fi
    touch "${ROOT}/${BUILD_DIR}/${BOOTSTRAP_OK_MARKER_NAME}"
    echo "Bootstrap complete. Next: ./build_gfx1031.sh build --build-dir ${BUILD_DIR}" | tee -a "${LOG_FILE}"
    ;;
  build)
    if (( DETACH )) && [[ "${LOG_FILE}" == "${ROOT}/build.log" ]]; then
      LOG_FILE="${ROOT}/${BUILD_DIR}.log"
    fi
    # Enforce bootstrap for Stage-1 toolchain builds so we don't get "half-configured"
    # failures from missing sysdeps/dist CMake config files.
    if [[ "${STAGE}" == "1" && "${BUILD_DIR}" == "${STAGE1_BUILD_DIR}" ]]; then
      require_bootstrap_ok
    fi
    if [[ ${#SUBPROJECTS[@]} -gt 0 ]]; then
      run_cmd_array ninja -C "${BUILD_DIR}" "${SUBPROJECTS[@]}"
    elif [[ "${STAGE}" == "1" && "${BUILD_DIR}" == "${STAGE1_BUILD_DIR}" ]]; then
      run_cmd_array ninja -C "${BUILD_DIR}" "${stage1_default_targets[@]}"
    else
      run_cmd_array ninja -C "${BUILD_DIR}"
    fi
    ;;
  expunge)
    if [[ ${#SUBPROJECTS[@]} -eq 0 ]]; then
      echo "expunge requires subproject names (e.g. amd-llvm hip-clr rocBLAS rocRAND)." >&2
      exit 2
    fi
    for t in "${SUBPROJECTS[@]}"; do
      run_cmd_array ninja -C "${BUILD_DIR}" "${t}+expunge"
    done
    ;;
  rebuild)
    if [[ ${#SUBPROJECTS[@]} -eq 0 ]]; then
      echo "rebuild requires subproject names (e.g. amd-llvm hip-clr rocBLAS rocRAND)." >&2
      exit 2
    fi
    for t in "${SUBPROJECTS[@]}"; do
      run_cmd_array ninja -C "${BUILD_DIR}" "${t}+expunge"
      run_cmd_array ninja -C "${BUILD_DIR}" "${t}"
    done
    ;;
  rocprofiler-gcc)
    # Phase-2: rocprofiler-systems with GNU compilers (Dyninst requirement).
    build_dir="${ROOT}/build-rocprofiler-gcc"
    install_prefix="${ROOT}/install-rocprofiler-gcc"
    require_cmd gcc "install build-essential."
    require_cmd g++ "install build-essential."
    mkdir -p "${build_dir}"
    systemd-run --user --scope -p "MemoryHigh=${MEM_HIGH}" -p "MemoryMax=${MEM_MAX}" \
      bash -lc "cd \"${ROOT}\" && cmake -S . -B \"${build_dir}\" -GNinja \
        -DTHEROCK_ENABLE_ROCPROFSYS=ON \
        -DTHEROCK_DISABLE_GNU_CHECK=ON \
        -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_INSTALL_PREFIX=\"${install_prefix}\"" 2>&1 | tee -a "${LOG_FILE}"
    systemd-run --user --scope -p "MemoryHigh=${MEM_HIGH}" -p "MemoryMax=${MEM_MAX}" \
      bash -lc "cd \"${build_dir}\" && ninja && ninja install" 2>&1 | tee -a "${LOG_FILE}"
    echo "rocprofiler-systems installed to: ${install_prefix}" | tee -a "${LOG_FILE}"
    ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    usage >&2
    exit 2
    ;;
esac
