#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_PATH="$ROOT_DIR/build/dist/rocm"
HIPCC="$ROCM_PATH/bin/hipcc"
BITCODE_DIR="$ROCM_PATH/lib/llvm/amdgcn/bitcode"
SOURCE="$ROOT_DIR/hello.cpp"
BIN="$ROOT_DIR/hello_power_bench"
ORIGINAL_PATH="$PATH"
ORIGINAL_LD="${LD_LIBRARY_PATH:-}"

usage() {
  cat <<'EOF'
Usage: run_hip_workload.sh [duration_seconds]

Runs the hello kernel in a tight loop for the specified duration (default 20 seconds)
without emitting any stdout so you can monitor GPU usage from another terminal.
EOF
  exit "$1"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage 0
fi

DEFAULT_DURATION=20
DURATION="${1:-$DEFAULT_DURATION}"
if ! [[ "$DURATION" =~ ^[0-9]+$ ]]; then
  echo "duration must be an integer number of seconds" >&2
  usage 1
fi

if [[ ! -d "$ROCM_PATH" ]]; then
  echo "ROCm build tree missing at $ROCM_PATH" >&2
  exit 1
fi

if [[ ! -x "$HIPCC" ]]; then
  echo "hipcc compiler missing at $HIPCC" >&2
  exit 1
fi

set_env() {
  export ROCM_PATH="$ROCM_PATH"
  export HIP_PATH="$ROCM_PATH"
  export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$ORIGINAL_PATH"
  export LD_LIBRARY_PATH="$ROCM_PATH/lib${ORIGINAL_LD:+:$ORIGINAL_LD}"
}

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
    (void) hipDeviceSynchronize();
    printf("Hello from host\n");
    return 0;
}
EOF
fi

if [[ ! -f "$BIN" || "$SOURCE" -nt "$BIN" ]]; then
  set_env
  "$HIPCC" --rocm-device-lib-path="$BITCODE_DIR" "$SOURCE" -o "$BIN"
fi

set_env

END=$((SECONDS + DURATION))
while ((SECONDS < END)); do
  "$BIN" > /dev/null 2>&1 || true
done
