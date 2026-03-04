#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORK_ROOT="${WORK_ROOT:-${ROOT}/validation/workspace/builds/tensorflow_rocm}"
TF_DIR="${TF_DIR:-${WORK_ROOT}/tensorflow}"
UNIT="${UNIT:-tensorflow-rocm-wheel-build.service}"
LOG_FILE="${LOG_FILE:-${WORK_ROOT}/tf_build_live.log}"
BAZELISK="${BAZELISK:-${WORK_ROOT}/bin/bazelisk}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT}/validation/workspace/cache/xdg}"
BAZELISK_HOME="${BAZELISK_HOME:-${ROOT}/validation/workspace/cache/bazelisk}"
BAZEL_OUTPUT_USER_ROOT="${BAZEL_OUTPUT_USER_ROOT:-${ROOT}/validation/workspace/cache/bazel/output_user_root}"

mkdir -p "$(dirname "${LOG_FILE}")" "${XDG_CACHE_HOME}" "${BAZELISK_HOME}" "${BAZEL_OUTPUT_USER_ROOT}"

CMD='set -o pipefail; export LIBRARY_PATH="'"${WORK_ROOT}"'/syslibs:${LIBRARY_PATH}"; export LD_LIBRARY_PATH="'"${WORK_ROOT}"'/syslibs:${LD_LIBRARY_PATH}"; export XDG_CACHE_HOME="'"${XDG_CACHE_HOME}"'"; export BAZELISK_HOME="'"${BAZELISK_HOME}"'"; "'"${BAZELISK}"'" --output_user_root="'"${BAZEL_OUTPUT_USER_ROOT}"'" build --config=opt --config=rocm --verbose_failures //tensorflow/tools/pip_package:wheel 2>&1 | tee -a "'"${LOG_FILE}"'"; exit ${PIPESTATUS[0]}'

systemd-run --user --unit "${UNIT}" --collect --property=WorkingDirectory="${TF_DIR}" /usr/bin/bash -lc "${CMD}"

echo "Started unit: ${UNIT}"
echo "Log file: ${LOG_FILE}"
echo "Check: systemctl --user status ${UNIT} --no-pager"
