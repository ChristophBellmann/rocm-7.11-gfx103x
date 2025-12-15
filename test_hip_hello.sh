#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_PATH="$ROOT_DIR/build/dist/rocm"
HIPCC="$ROCM_PATH/bin/hipcc"
BITCODE_DIR="$ROCM_PATH/lib/llvm/amdgcn/bitcode"
SOURCE="$ROOT_DIR/hello.cpp"
OUT_BIN="$ROOT_DIR/hello_hip_test"

if [[ ! -d "$ROCM_PATH" ]]; then
  echo "ROCm build not found at $ROCM_PATH" >&2
  exit 1
fi

if [[ ! -x "$HIPCC" ]]; then
  echo "hipcc compiler not found at $HIPCC" >&2
  exit 1
fi

if [[ ! -d "$BITCODE_DIR" ]]; then
  echo "ROCm device bitcode directory missing at $BITCODE_DIR" >&2
  exit 1
fi

if [[ ! -f "$SOURCE" ]]; then
  cat <<'EOF' > "$SOURCE"
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void hello_kernel()
{
    printf("Hello from GPU thread %d\n", hipThreadIdx_x);
}

int main()
{
    hipLaunchKernelGGL(hello_kernel, dim3(1), dim3(32), 0, 0);
    hipDeviceSynchronize();
    printf("Hello from host\n");
    return 0;
}
EOF
fi

export ROCM_PATH="$ROCM_PATH"
export HIP_PATH="$ROCM_PATH"
export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"

echo "Compiling $SOURCE with device libraries from $BITCODE_DIR"
"$HIPCC" --rocm-device-lib-path="$BITCODE_DIR" "$SOURCE" -o "$OUT_BIN"

echo "Running $OUT_BIN"
"$OUT_BIN"
