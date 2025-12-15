#!/usr/bin/env bash
# TheRock gfx1031 low-memory builder
# Optimized for Christoph’s 31 GiB machine and the gfx1031 ROCm stack.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
VENV_ACTIVATE="${VENV_ACTIVATE:-$SCRIPT_DIR/.venv/bin/activate}"
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "ERROR: virtualenv activation script not found at $VENV_ACTIVATE"
    exit 1
fi
source "$VENV_ACTIVATE"

# Build banner for visibility
cat <<'EOF'
========================================
TheRock gfx1031 Low-Memory Build Helper
========================================
   Target : gfx1031 (AMD Radeon RX 6700 XT)
   Memory : 32 505 856 KB per job (≈31 GiB profile)
   Jobs   : defaults to 4 / load limit 4
   Env    : loads the in-tree ROCm/HIP tree before building
EOF

# Ensure in-tree HIP/ROCm build artifacts are discoverable by subprojects.
ROCM_BUILD_DIR="$SCRIPT_DIR/build/core/clr/dist"
export ROCM_PATH="$ROCM_BUILD_DIR"
export HIP_PATH="$ROCM_BUILD_DIR"
export HIP_DEVICE_LIB_PATH="$ROCM_BUILD_DIR/lib/llvm/amdgcn/bitcode"
export PATH="$ROCM_BUILD_DIR/bin:$ROCM_BUILD_DIR/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_BUILD_DIR/lib:$ROCM_BUILD_DIR/lib64:$ROCM_BUILD_DIR/lib/llvm/lib:$LD_LIBRARY_PATH"

LIBDRM_STAGE_DIR="$SCRIPT_DIR/build/third-party/sysdeps/linux/libdrm/build/stage/lib/rocm_sysdeps"
LIBDRM_INCLUDE="$LIBDRM_STAGE_DIR/include"
LIBDRM_LIB="$LIBDRM_STAGE_DIR/lib"

if [ -d "$LIBDRM_INCLUDE" ]; then
    export CPATH="$LIBDRM_INCLUDE${CPATH:+:$CPATH}"
    export C_INCLUDE_PATH="$LIBDRM_INCLUDE${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
    export CPLUS_INCLUDE_PATH="$LIBDRM_INCLUDE${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
fi

if [ -d "$LIBDRM_LIB" ]; then
    export LD_LIBRARY_PATH="$LIBDRM_LIB:$LD_LIBRARY_PATH"
fi

# Set memory limits per process (in KB); default to ~31GB machine profile (~8GiB per build job).
DEFAULT_PER_PROCESS_MEMORY_KB=32505856
BUILD_MEM_LIMIT_USED="${BUILD_MEM_LIMIT_KB:-$DEFAULT_PER_PROCESS_MEMORY_KB}"
ulimit -v "$BUILD_MEM_LIMIT_USED" \
  || echo "Warning: Could not set per-process virtual memory limit."

# Force single-threaded linking for LLVM to reduce memory spikes
export LLVM_PARALLEL_LINK_JOBS=1

# Default to 4 jobs and a load cap of 4 for the low-memory profile.
DEFAULT_JOBS=4
DEFAULT_LOAD_LIMIT=4
JOBS="${BUILD_JOBS:-$DEFAULT_JOBS}"
LOAD_LIMIT="${BUILD_LOAD_LIMIT:-$DEFAULT_LOAD_LIMIT}"

echo "BUILD_MEM_LIMIT_KB=${BUILD_MEM_LIMIT_USED} BUILD_JOBS=${JOBS} BUILD_LOAD_LIMIT=${LOAD_LIMIT}"
echo "Building with -j$JOBS, load limit $LOAD_LIMIT, and OOM protections enabled..."
mkdir -p build/math-libs/BLAS/{rocSPARSE,hipSPARSE}/build/clients/matrices
nice -n 10 cmake --build build -j"$JOBS" -- -l"$LOAD_LIMIT" "$@"
