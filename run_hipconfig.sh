#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_PATH="$ROOT_DIR/build/dist/rocm"
HIPCONFIG="$ROCM_PATH/bin/hipconfig"

if [[ ! -d "$ROCM_PATH" ]]; then
  echo "ROCm build tree required at $ROCM_PATH" >&2
  exit 1
fi

if [[ ! -x "$HIPCONFIG" ]]; then
  echo "hipconfig missing at $HIPCONFIG" >&2
  exit 1
fi

export ROCM_PATH="$ROCM_PATH"
export HIP_PATH="$ROCM_PATH"
export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"

exec "$HIPCONFIG" --full
