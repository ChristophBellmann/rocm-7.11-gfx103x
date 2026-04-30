#!/usr/bin/env bash
set -euo pipefail

cat <<'NOTE'
This validation-side helper is deprecated.

The canonical ROCm/PyTorch venv installer now lives in:
  rocm-7.11-pytorch-gfx103x/tools/rocm_release/create_rocm_venv.sh

Reason:
  The old helper encoded an install-order workaround. The canonical helper now
  owns the deterministic order, generates constraints, writes python-rocm and
  pip-rocm wrappers, and patches venv activation so ROCm loader environment is
  active before Python starts.

Use from the PyTorch fork:
  bash tools/rocm_release/create_rocm_venv.sh \
    --venv .venv \
    --rocm-prefix /opt/rocm \
    -y

For app packages:
  .venv/bin/pip-rocm install openai-whisper

This file intentionally no longer reimplements that logic, to avoid divergent
venv behavior between TheRock validation and the PyTorch wheel packaging repo.
NOTE

exit 2
