#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/build/dist/rocm"
CORE="$ROOT/build/core/clr/dist"

if [ ! -x "$DIST/bin/rocm-smi" ]; then
  echo "rocm-smi not found under $DIST/bin"
  exit 1
fi

sudo \
  env \
    ROCM_PATH="$DIST" \
    HIP_PATH="$DIST" \
    HIP_DEVICE_LIB_PATH="$DIST/lib/llvm/amdgcn/bitcode" \
    PATH="$DIST/bin:$CORE/lib/llvm/bin:$PATH" \
    LD_LIBRARY_PATH="$DIST/lib:$DIST/lib64:$CORE/lib:$CORE/lib/llvm/lib:${LD_LIBRARY_PATH:-}" \
    "$DIST/bin/rocm-smi" --gpureset -d 0

echo "rocm-smi --reset completed (build tree) on $(hostname)"
