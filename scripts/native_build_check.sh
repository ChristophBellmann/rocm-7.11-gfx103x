#!/usr/bin/env bash
# Quick sanity check for the gfx1031 native build tree documented in BUILD_SUCCESS_NATIVE_GFX1031.md.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/build/dist/rocm"
CORE="$ROOT/build/core/clr/dist"

if [ ! -d "$DIST" ]; then
  echo "ERROR: expected build/dist/rocm under $ROOT"
  exit 1
fi

export ROCM_PATH="$DIST"
export HIP_PATH="$DIST"
export HIP_DEVICE_LIB_PATH="$DIST/lib/llvm/amdgcn/bitcode"
export PATH="$DIST/bin:$CORE/bin:$CORE/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$DIST/lib:$DIST/lib64:$CORE/lib:$CORE/lib/llvm/lib:$LD_LIBRARY_PATH"

echo "Validating gfx1031 native build (see BUILD_SUCCESS_NATIVE_GFX1031.md)"
echo "- rocminfo"
set +o pipefail
rocminfo | head -n 5
set -o pipefail
echo "- rocm-smi --showproductname"
rocm-smi --showproductname
echo "- hipconfig --version"
hipconfig --version

if [ -x "$DIST/bin/rocblas-bench" ]; then
echo "- rocblas-bench (axpy, n=4)"
set +o errexit
if "$DIST/bin/rocblas-bench" -f axpy -r f32_r -n 4 -i 1 >/tmp/rocblas-axpy.log 2>&1; then
  echo "rocblas-bench sample succeeded"
else
  echo "rocblas-bench axpy failed; see /tmp/rocblas-axpy.log"
fi
set -o errexit
else
  echo "rocblas-bench missing; rerun build_enable_math_clients.sh + build_low_memory.sh"
fi

echo "Native gfx1031 build looks configured as summarized in BUILD_SUCCESS_NATIVE_GFX1031.md"
