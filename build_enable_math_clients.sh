#!/usr/bin/env bash
# Configure TheRock with the expanded math clients so `rocblas-bench`, `rocfft-rider`, and friends appear in the staging tree.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV="$SCRIPT_DIR/.venv/bin/activate"
if [ ! -f "$VENV" ]; then
  echo "virtualenv not found at $VENV—run 'python3 -m venv .venv && source .venv/bin/activate' first"
  exit 1
fi
source "$VENV"

cat <<'EOF'
=============================================
Enable TheRock math clients
=============================================
targets: gfx1031
ninja: keep 4 jobs/load-limited 4 (low-memory profile)
clients: rocblas-bench, rocfft-rider, rocSPARSE(*), rocBLAS/hipBLAS benchmarks
EOF

BUILD_MEM_LIMIT_KB="${BUILD_MEM_LIMIT_KB:-32505856}"
BUILD_JOBS="${BUILD_JOBS:-4}"
BUILD_LOAD_LIMIT="${BUILD_LOAD_LIMIT:-4}"

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

cmake -B build -GNinja . \
  -DTHEROCK_AMDGPU_TARGETS=gfx1031 \
  -DTHEROCK_ENABLE_ALL=OFF \
  -DTHEROCK_ENABLE_BLAS=ON \
  -DTHEROCK_ENABLE_FFT=ON \
  -DTHEROCK_ENABLE_PRIM=ON \
  -DTHEROCK_ENABLE_RAND=ON \
  -DTHEROCK_ENABLE_SOLVER=ON \
  -DTHEROCK_ENABLE_SPARSE=ON \
  -DTHEROCK_ENABLE_MIOPEN=ON \
  -DTHEROCK_ENABLE_MIOPEN_PLUGIN=ON \
  -DTHEROCK_ENABLE_HIPDNN=ON \
  -DTHEROCK_BUILD_TESTING=ON \
  -DBUILD_CLIENTS_OPENMP=ON \
  -DBUILD_CLIENTS_BENCHMARKS=ON \
  -DBUILD_CLIENTS_SAMPLES=ON \
  -DBUILD_TESTING=OFF \
  -DTHEROCK_ENABLE_HOST_BLAS=ON \
  -DTHEROCK_ENABLE_HOST_SUITE_SPARSE=ON

echo "Re-run the low-memory builder:"
echo "  BUILD_MEM_LIMIT_KB=${BUILD_MEM_LIMIT_KB} BUILD_JOBS=${BUILD_JOBS} BUILD_LOAD_LIMIT=${BUILD_LOAD_LIMIT} ./build_low_memory.sh"
