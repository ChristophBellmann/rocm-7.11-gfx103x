#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${ROOT}/validation/scripts/pytorch_rocm/install_pytorch_rocm_wheel_to_venv.sh"

if [[ ! -x "${TARGET}" ]]; then
  echo "Missing target helper: ${TARGET}" >&2
  exit 1
fi

echo "NOTE: ${BASH_SOURCE[0]} is kept as a compatibility wrapper."
echo "      Standard project install helper: validation/scripts/pytorch_rocm/install_pytorch_rocm_wheel_to_venv.sh"
exec "${TARGET}" "$@"
