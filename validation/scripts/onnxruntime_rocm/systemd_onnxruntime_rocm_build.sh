#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORK_ROOT="${WORK_ROOT:-${ROOT}/validation/workspace/builds/onnxruntime_rocm}"
UNIT="${UNIT:-onnxruntime-rocm-bootstrap.service}"
LOG_FILE="${LOG_FILE:-${WORK_ROOT}/build_start.log}"
INTERVAL_SEC="${INTERVAL_SEC:-30}"

usage() {
  cat <<USAGE
Usage:
  $0 start
  $0 monitor [extra args passed to monitor_gfx1031.sh]

Environment:
  UNIT=${UNIT}
  WORK_ROOT=${WORK_ROOT}
  LOG_FILE=${LOG_FILE}
  INTERVAL_SEC=${INTERVAL_SEC}
USAGE
}

cmd="${1:-}"
case "${cmd}" in
  start)
    shift
    if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
      usage
      exit 0
    fi
    mkdir -p "$(dirname "${LOG_FILE}")"
    build_cmd="cd '${ROOT}' && bash '${ROOT}/validation/scripts/onnxruntime_rocm/build_onnxruntime_rocm_wheel.sh' >> '${LOG_FILE}' 2>&1"
    systemd-run --user --unit "${UNIT}" --collect /usr/bin/bash -lc "${build_cmd}"
    echo "Started unit: ${UNIT}"
    echo "Log file: ${LOG_FILE}"
    echo "Check: systemctl --user status ${UNIT} --no-pager"
    ;;
  monitor)
    shift
    if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
      usage
      exit 0
    fi
    UNIT="${UNIT}" LOG_FILE="${LOG_FILE}" INTERVAL_SEC="${INTERVAL_SEC}" \
      "${ROOT}/monitor_gfx1031.sh" "$@"
    ;;
  --help|-h|"")
    usage
    exit 0
    ;;
  *)
    echo "Unknown subcommand: ${cmd}" >&2
    usage >&2
    exit 2
    ;;
esac
