#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec "${ROOT}/validation/scripts/onnxruntime_rocm/systemd_onnxruntime_rocm_build.sh" start "$@"
