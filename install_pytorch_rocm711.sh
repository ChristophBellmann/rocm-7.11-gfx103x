#!/usr/bin/env bash
set -euo pipefail

cat <<'NOTE'
This top-level helper is deprecated.

The canonical ROCm/PyTorch venv installer now lives in:
  rocm-7.11-pytorch-gfx103x/tools/rocm_release/create_rocm_venv.sh

Use:
  cd /path/to/rocm-7.11-pytorch-gfx103x
  bash tools/rocm_release/create_rocm_venv.sh \
    --venv .venv \
    --rocm-prefix /opt/rocm \
    -y

For application packages:
  .venv/bin/pip-rocm install openai-whisper

Reason:
  The deterministic venv flow must live with the custom PyTorch wheel family.
  TheRock validates promoted artifacts; it does not own project venv creation.
NOTE

exit 2
