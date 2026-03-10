#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTORCH_REPO="${ROOT}/validation/workspace/cache/git/pytorch_rocm711"
PYTORCH_REPO="${PYTORCH_ROCM_REPO:-${DEFAULT_PYTORCH_REPO}}"
TARGET="${PYTORCH_REPO}/tools/rocm_release/install_pytorch_rocm_wheel_to_venv.sh"

if [[ ! -x "${TARGET}" ]]; then
  echo "Missing target helper: ${TARGET}" >&2
  echo "Set PYTORCH_ROCM_REPO to your rocm-7.11-pytorch-gfx103x checkout." >&2
  exit 1
fi

echo "NOTE: ${BASH_SOURCE[0]} is kept as a compatibility wrapper."
echo "      Standard project install helper: tools/rocm_release/install_pytorch_rocm_wheel_to_venv.sh"
exec "${TARGET}" "$@"
