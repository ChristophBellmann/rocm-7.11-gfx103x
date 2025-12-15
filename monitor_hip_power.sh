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

if [[ ! -d "$ROCM_PATH" ]]; then
  echo "ROCm build tree missing at $ROCM_PATH" >&2
  exit 1
fi

if [[ ! -x "$HIPCC" ]]; then
  echo "hipcc compiler missing at $HIPCC" >&2
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
    (void) hipDeviceSynchronize();
    printf("Hello from host\n");
    return 0;
}
EOF
fi

set_env() {
  export ROCM_PATH="$ROCM_PATH"
  export HIP_PATH="$ROCM_PATH"
  export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$ORIGINAL_PATH"
  export LD_LIBRARY_PATH="$ROCM_PATH/lib${ORIGINAL_LD:+:$ORIGINAL_LD}"
}

if [[ ! -f "$BIN" || "$SOURCE" -nt "$BIN" ]]; then
  set_env
  "$HIPCC" --rocm-device-lib-path="$BITCODE_DIR" "$SOURCE" -o "$BIN"
fi

set_env

ROCM_SMI="$ROCM_PATH/bin/rocm-smi"

run_hip_loop() {
  while true; do
    "$BIN" >/tmp/hip_power_run.log 2>&1 || true
  done
}

run_hip_loop &
HIP_PID=$!
trap 'kill "$HIP_PID" >/dev/null 2>&1 || true' EXIT

SAMPLES=5
for ((i = 1; i <= SAMPLES; i++)); do
  printf "\n==== rocm-smi sample %d ====\n" "$i"
  "$ROCM_SMI" --showtemp --showclocks --showproductname || true
  sleep 1
done

kill "$HIP_PID" >/dev/null 2>&1 || true
wait "$HIP_PID" >/dev/null 2>&1 || true
echo
echo "Measured $SAMPLES rocm-smi snapshots while $BIN looped in the background."
