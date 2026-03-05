#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORK_ROOT="${WORK_ROOT:-${ROOT}/validation/workspace/builds/tensorflow_rocm}"
TF_SRC_DIR="${TF_SRC_DIR:-${WORK_ROOT}/tensorflow}"
SYSLIBS_DIR="${SYSLIBS_DIR:-${WORK_ROOT}/syslibs}"
VENV_DIR="${VENV_DIR:-${WORK_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
JOBS="${JOBS:-$(nproc)}"
TF_REPO_URL="${TF_REPO_URL:-https://github.com/tensorflow/tensorflow.git}"
TF_REF="${TF_REF:-v2.20.0}"
BAZEL_BIN_DIR="${BAZEL_BIN_DIR:-${WORK_ROOT}/bin}"
BAZELISK="${BAZELISK:-${BAZEL_BIN_DIR}/bazelisk}"
WHEEL_OUT_DIR="${WHEEL_OUT_DIR:-${ROOT}/validation/workspace/cache/wheels/tensorflow_rocm_custom}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT}/validation/workspace/cache/xdg}"
BAZELISK_HOME="${BAZELISK_HOME:-${ROOT}/validation/workspace/cache/bazelisk}"
BAZEL_OUTPUT_USER_ROOT="${BAZEL_OUTPUT_USER_ROOT:-${ROOT}/validation/workspace/cache/bazel/output_user_root}"
CCACHE_DIR="${CCACHE_DIR:-${ROOT}/validation/workspace/cache/ccache}"
BUILD_START_LOG="${BUILD_START_LOG:-${WORK_ROOT}/build_start.log}"
DO_UPDATE="${DO_UPDATE:-1}"
BAZEL_VERBOSE_FAILURES="${BAZEL_VERBOSE_FAILURES:-1}"

if [[ "${WORK_ROOT}" != /* ]]; then
  WORK_ROOT="${ROOT}/${WORK_ROOT}"
fi
if [[ "${TF_SRC_DIR}" != /* ]]; then
  TF_SRC_DIR="${ROOT}/${TF_SRC_DIR}"
fi
if [[ "${VENV_DIR}" != /* ]]; then
  VENV_DIR="${ROOT}/${VENV_DIR}"
fi
if [[ "${SYSLIBS_DIR}" != /* ]]; then
  SYSLIBS_DIR="${ROOT}/${SYSLIBS_DIR}"
fi
if [[ "${BAZEL_BIN_DIR}" != /* ]]; then
  BAZEL_BIN_DIR="${ROOT}/${BAZEL_BIN_DIR}"
fi
if [[ "${WHEEL_OUT_DIR}" != /* ]]; then
  WHEEL_OUT_DIR="${ROOT}/${WHEEL_OUT_DIR}"
fi
if [[ "${XDG_CACHE_HOME}" != /* ]]; then
  XDG_CACHE_HOME="${ROOT}/${XDG_CACHE_HOME}"
fi
if [[ "${BAZELISK_HOME}" != /* ]]; then
  BAZELISK_HOME="${ROOT}/${BAZELISK_HOME}"
fi
if [[ "${BAZEL_OUTPUT_USER_ROOT}" != /* ]]; then
  BAZEL_OUTPUT_USER_ROOT="${ROOT}/${BAZEL_OUTPUT_USER_ROOT}"
fi
if [[ "${CCACHE_DIR}" != /* ]]; then
  CCACHE_DIR="${ROOT}/${CCACHE_DIR}"
fi
if [[ "${BUILD_START_LOG}" != /* ]]; then
  BUILD_START_LOG="${ROOT}/${BUILD_START_LOG}"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${VENV_DIR}/bin/python"
elif [[ "${PYTHON_BIN}" != /* ]]; then
  PYTHON_BIN="${ROOT}/${PYTHON_BIN}"
fi

mkdir -p "${WORK_ROOT}" "${BAZEL_BIN_DIR}" "${WHEEL_OUT_DIR}" \
  "${SYSLIBS_DIR}" \
  "${XDG_CACHE_HOME}" "${BAZELISK_HOME}" "${BAZEL_OUTPUT_USER_ROOT}" "${CCACHE_DIR}"
mkdir -p "$(dirname "${BUILD_START_LOG}")"
: > "${BUILD_START_LOG}"
# Keep a stable live log for `less +F .../build_start.log` regardless of caller.
exec > >(tee -a "${BUILD_START_LOG}") 2>&1
export XDG_CACHE_HOME
export BAZELISK_HOME
export CCACHE_DIR
# Make libnuma discoverable for lld-based link steps even without libnuma-dev.
if [[ -e "/lib/x86_64-linux-gnu/libnuma.so.1" ]]; then
  ln -sfn /lib/x86_64-linux-gnu/libnuma.so.1 "${SYSLIBS_DIR}/libnuma.so"
  ln -sfn /lib/x86_64-linux-gnu/libnuma.so.1 "${SYSLIBS_DIR}/libnuma.so.1"
fi
export LIBRARY_PATH="${SYSLIBS_DIR}:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export CCACHE_BASEDIR="${TF_SRC_DIR}"
export CCACHE_COMPILERCHECK=content

# Require ROCm LLVM toolchain for ROCm builds; do not fallback to system clang.
# Prefer in-tree TheRock stage-2 toolchain when present.
if [[ -x "${ROOT}/build-stage2/dist/rocm/llvm/bin/clang" && -x "${ROOT}/build-stage2/dist/rocm/llvm/bin/clang++" ]]; then
  DEFAULT_REAL_CLANG="${ROOT}/build-stage2/dist/rocm/llvm/bin/clang"
  DEFAULT_REAL_CLANGXX="${ROOT}/build-stage2/dist/rocm/llvm/bin/clang++"
else
  DEFAULT_REAL_CLANG="${ROCM_PATH}/llvm/bin/clang"
  DEFAULT_REAL_CLANGXX="${ROCM_PATH}/llvm/bin/clang++"
fi
REAL_CLANG="${REAL_CLANG:-${DEFAULT_REAL_CLANG}}"
REAL_CLANGXX="${REAL_CLANGXX:-${DEFAULT_REAL_CLANGXX}}"
if [[ ! -x "${REAL_CLANG}" || ! -x "${REAL_CLANGXX}" ]]; then
  echo "ERROR: Required ROCm clang toolchain not found/executable." >&2
  echo "Expected: ${ROCM_PATH}/llvm/bin/clang and clang++" >&2
  echo "Set ROCM_PATH correctly or provide REAL_CLANG/REAL_CLANGXX explicitly." >&2
  exit 1
fi
CCACHE_CLANG_WRAPPER="${BAZEL_BIN_DIR}/clang_ccache_wrapper.sh"
CCACHE_CLANGXX_WRAPPER="${BAZEL_BIN_DIR}/clangxx_ccache_wrapper.sh"
cat > "${CCACHE_CLANG_WRAPPER}" <<EOF
#!/usr/bin/env bash
exec /usr/bin/ccache "${REAL_CLANG}" "\$@"
EOF
cat > "${CCACHE_CLANGXX_WRAPPER}" <<EOF
#!/usr/bin/env bash
exec /usr/bin/ccache "${REAL_CLANGXX}" "\$@"
EOF
chmod +x "${CCACHE_CLANG_WRAPPER}" "${CCACHE_CLANGXX_WRAPPER}"
export CC="${CCACHE_CLANG_WRAPPER}"
export CXX="${CCACHE_CLANGXX_WRAPPER}"

patch_rocm_crosstool_builtin_includes() {
  local clang_resource_dir crosstool_build real_clang_dir
  clang_resource_dir="$("${REAL_CLANG}" --print-resource-dir 2>/dev/null || true)"
  real_clang_dir="$(cd "$(dirname "${REAL_CLANG}")" && pwd)"
  crosstool_build="${TF_SRC_DIR}/bazel-tensorflow/external/local_config_rocm/crosstool/BUILD"

  if [[ ! -f "${crosstool_build}" ]]; then
    return 0
  fi

  "${PYTHON_BIN}" - "${crosstool_build}" "${clang_resource_dir}" "${real_clang_dir}" "${ROCM_PATH}" <<'PY'
import pathlib
import re
import sys
import glob
import os

p = pathlib.Path(sys.argv[1])
clang_resource_dir = sys.argv[2]
real_clang_dir = sys.argv[3]
rocm_path = sys.argv[4] if len(sys.argv) > 4 else ""
s = p.read_text()

m = re.search(r'cxx_builtin_include_directories\s*=\s*\[(.*?)\]\s*,', s, re.S)
if not m:
    print("WARNING: cxx_builtin_include_directories block not found in local_config_rocm/crosstool/BUILD")
    raise SystemExit(0)

block = m.group(1)
add_candidates = []
if clang_resource_dir:
    add_candidates.append(os.path.join(clang_resource_dir, "include"))
# Resolve include dir via compiler binary layout as fallback.
for d in glob.glob(os.path.join(real_clang_dir, "..", "lib", "clang", "*", "include")):
    add_candidates.append(os.path.realpath(d))
    add_candidates.append(os.path.abspath(d))
if rocm_path:
    for pat in (
        os.path.join(rocm_path, "llvm", "lib", "clang", "*", "include"),
        os.path.join(rocm_path, "lib", "llvm", "lib", "clang", "*", "include"),
    ):
        for d in glob.glob(pat):
            add_candidates.append(os.path.realpath(d))
            add_candidates.append(os.path.abspath(d))

added = []
for include_dir in add_candidates:
    if not os.path.isdir(include_dir):
        continue
    needle = f'"{include_dir}"'
    if needle in block:
        continue
    block = block.rstrip() + (", " if block.strip() else "") + needle
    added.append(include_dir)

new_block = block
s = s[:m.start(1)] + new_block + s[m.end(1):]
p.write_text(s)
if added:
    print("Patched cxx builtin include dirs:")
    for d in added:
        print(f"  - {d}")
PY
}

purge_bazel_local_config_rocm() {
  # Force Bazel repo reconfiguration so stale ROCm/Git metadata is not reused
  # across branch switches (e.g. r2.20-rocm-enhanced -> christoph/gfx1031-buildfixes).
  bazelisk --output_user_root="${BAZEL_OUTPUT_USER_ROOT}" shutdown >/dev/null 2>&1 || true
  find "${BAZEL_OUTPUT_USER_ROOT}" -type d -path "*/external/local_config_rocm" -prune -exec rm -rf {} + 2>/dev/null || true
  find "${BAZEL_OUTPUT_USER_ROOT}" -type d -path "*/external/local_config_git" -prune -exec rm -rf {} + 2>/dev/null || true
  rm -rf "${TF_SRC_DIR}/bazel-tensorflow/external/local_config_rocm" 2>/dev/null || true
  rm -rf "${TF_SRC_DIR}/bazel-tensorflow/external/local_config_git" 2>/dev/null || true
}

force_disable_generated_rocm_hipblaslt() {
  "${PYTHON_BIN}" - "${BAZEL_OUTPUT_USER_ROOT}" "${TF_SRC_DIR}" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
tf_src = pathlib.Path(sys.argv[2])
patched = []

files = list(root.glob("**/external/local_config_rocm/rocm/rocm_config/rocm_config.h"))
files += list(root.glob("**/external/local_config_rocm/rocm/build_defs.bzl"))
files += [tf_src / "bazel-tensorflow" / "external" / "local_config_rocm" / "rocm" / "rocm_config" / "rocm_config.h"]
files += [tf_src / "bazel-tensorflow" / "external" / "local_config_rocm" / "rocm" / "build_defs.bzl"]

for p in files:
    if not p.exists() or not p.is_file():
        continue
    s = p.read_text(encoding="utf-8")
    n = s
    if p.name == "rocm_config.h":
        n = re.sub(r"#define\s+TF_HIPBLASLT\s+1", "#define TF_HIPBLASLT 0", n)
    elif p.name == "build_defs.bzl":
        n = n.replace("if_rocm_hipblaslt(if_true, if_false = []):\n    return if_true", "if_rocm_hipblaslt(if_true, if_false = []):\n    return if_false")
    if n != s:
        p.write_text(n, encoding="utf-8")
        patched.append(str(p))

if patched:
    print("Patched generated local_config_rocm files:")
    for p in patched:
        print(f"  - {p}")
PY
}

# Ensure Python tooling exists before any in-script patch helpers run.
if [[ ! -x "${PYTHON_BIN}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

if [[ ! -x "${BAZELISK}" ]]; then
  curl -L -o "${BAZELISK}" https://github.com/bazelbuild/bazelisk/releases/download/v1.23.0/bazelisk-linux-amd64
  chmod +x "${BAZELISK}"
fi
ln -sf "${BAZELISK}" "${BAZEL_BIN_DIR}/bazel"
export PATH="${BAZEL_BIN_DIR}:${PATH}"

if [[ ! -d "${TF_SRC_DIR}/.git" ]]; then
  git clone "${TF_REPO_URL}" "${TF_SRC_DIR}"
fi

cd "${TF_SRC_DIR}"
if [[ "${DO_UPDATE}" == "1" ]]; then
  git fetch --tags --all
fi
# Reset tracked file modifications from previous patch runs before switching tags.
git restore --worktree --staged . || true
git checkout "${TF_REF}"
if [[ "${DO_UPDATE}" == "1" ]]; then
  git pull --ff-only || true
fi
[[ -f third_party/gpus/rocm_configure.bzl ]] && git restore --worktree --staged third_party/gpus/rocm_configure.bzl || true
[[ -f third_party/xla/third_party/gpus/rocm_configure.bzl ]] && git restore --worktree --staged third_party/xla/third_party/gpus/rocm_configure.bzl || true

# ROCm tensorflow-upstream carries its own ROCm adaptations.
# Use a minimal build path and skip downstream patch blocks in this script.
ORIGIN_URL="$(git remote get-url origin 2>/dev/null || true)"
IS_CHRISTOPH_TF_FORK=0
if [[ "${TF_REPO_URL}" == *"rocm-7.11-tensorflow-gfx103x"* ]] || \
   [[ "${ORIGIN_URL}" == *"rocm-7.11-tensorflow-gfx103x"* ]]; then
  IS_CHRISTOPH_TF_FORK=1
fi
if [[ "${TF_REPO_URL}" == *"ROCm/tensorflow-upstream"* ]] || \
   [[ "${IS_CHRISTOPH_TF_FORK}" == "1" ]] || \
   [[ "${ORIGIN_URL}" == *"ROCm/tensorflow-upstream"* ]] || \
   [[ "${IS_CHRISTOPH_TF_FORK}" == "1" ]]; then
  # Some ROCm crosstool wrappers invoke `python` via /usr/bin/env.
  ln -sf "${PYTHON_BIN}" "${BAZEL_BIN_DIR}/python"
  "${PYTHON_BIN}" -m pip install -U pip setuptools wheel numpy
  "${PYTHON_BIN}" -m pip install -U keras_preprocessing packaging requests opt_einsum six

  if [[ "${IS_CHRISTOPH_TF_FORK}" == "1" ]]; then
    echo "Using ${TF_REPO_URL}@${TF_REF} with persisted gfx1031/ROCm patchset (no runtime source patching)."
  else
    # Disable hipBLASLt already at ROCm configure-time so TF_HIPBLASLT is 0 and
    # if_rocm_hipblaslt() branches stay off in generated local_config_rocm.
    "${PYTHON_BIN}" - <<'PY' "${TF_SRC_DIR}"
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
cands = [
    root / "third_party" / "xla" / "third_party" / "gpus" / "rocm" / "rocm_configure.bzl",
    root / "third_party" / "gpus" / "rocm_configure.bzl",
]
p = next((c for c in cands if c.exists()), None)
if p is not None:
    s = p.read_text(encoding="utf-8")
    s = re.sub(
        r'have_hipblaslt\s*=\s*"1"\s+if\s+rocm_libs\["hipblaslt"\]\s*!=\s*None\s+else\s+"0"',
        'have_hipblaslt = "0"',
        s,
    )
    s = s.replace(
        '"%{rocm_hipblaslt}": "True" if rocm_libs["hipblaslt"] != None else "False",',
        '"%{rocm_hipblaslt}": "False",',
    )
    p.write_text(s, encoding="utf-8")
PY

    # ROCm 7.11 headers can expose FlatBuffers v25 first in include resolution.
    # TensorFlow generated schema headers are version-pinned to v24 and fail with
    # a static_assert otherwise. Allow v24 (expected) and v25 (ROCm toolchain env).
    "${PYTHON_BIN}" - <<'PY' "${TF_SRC_DIR}"
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
headers = [
    "tensorflow/compiler/mlir/lite/schema/schema_generated.h",
    "tensorflow/compiler/mlir/lite/schema/conversion_metadata_generated.h",
    "tensorflow/lite/acceleration/configuration/configuration_generated.h",
    "tensorflow/lite/delegates/gpu/cl/compiled_program_cache_generated.h",
    "tensorflow/lite/delegates/gpu/cl/serialization_generated.h",
    "tensorflow/lite/delegates/gpu/common/task/tflite_serialization_base_generated.h",
    "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h",
    "tensorflow/lite/delegates/gpu/common/gpu_model_generated.h",
]

pat = re.compile(
    r"static_assert\(\s*FLATBUFFERS_VERSION_MAJOR == 24\s*&&\s*"
    r"FLATBUFFERS_VERSION_MINOR == (\d+)\s*&&\s*"
    r"FLATBUFFERS_VERSION_REVISION == (\d+),\s*"
    r"\"Non-compatible flatbuffers version included\"\);",
    re.M,
)

for rel in headers:
    p = root / rel
    if not p.exists():
        continue
    s = p.read_text(encoding="utf-8")
    if "FLATBUFFERS_VERSION_MAJOR == 25" in s:
        continue
    s2, n = pat.subn(
        lambda m: (
            "static_assert((FLATBUFFERS_VERSION_MAJOR == 24 &&\n"
            f"              FLATBUFFERS_VERSION_MINOR == {m.group(1)} &&\n"
            f"              FLATBUFFERS_VERSION_REVISION == {m.group(2)}) ||\n"
            "              (FLATBUFFERS_VERSION_MAJOR == 25),\n"
            '             "Non-compatible flatbuffers version included");'
        ),
        s,
        count=1,
    )
    if n:
        p.write_text(s2, encoding="utf-8")
PY

    # Enable native gfx1031 acceptance (RX 6700 XT) without HSA override.
    "${PYTHON_BIN}" - <<'PY' "${TF_SRC_DIR}"
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
p = root / "third_party/xla/xla/stream_executor/device_description.h"
s = p.read_text(encoding="utf-8")

if '"gfx1031"' not in s:
    s = s.replace(
        '"gfx1030",                        // RX68xx / RX69xx\n',
        '"gfx1030", "gfx1031",             // RX68xx / RX69xx / RX6700XT\n',
        1,
    )
    s = s.replace(
        'bool gfx10_rx68xx() const { return gfx_version() == "gfx1030"; }\n',
        'bool gfx10_rx68xx() const { return gfx_version() == "gfx1030" || gfx_version() == "gfx1031"; }\n',
        1,
    )
    s = s.replace(
        'bool gfx10_rx69xx() const { return gfx_version() == "gfx1030"; }\n',
        'bool gfx10_rx69xx() const { return gfx_version() == "gfx1030" || gfx_version() == "gfx1031"; }\n',
        1,
    )

p.write_text(s, encoding="utf-8")
PY

    # Allow disabling hipBLASLt initialization via env flag. This is useful on
    # gfx1031 compatibility runs where hipblasLtCreate can crash inside COMGR.
    "${PYTHON_BIN}" - <<'PY' "${TF_SRC_DIR}"
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
p = root / "third_party/xla/xla/stream_executor/rocm/rocm_blas.cc"
s = p.read_text(encoding="utf-8")

if "#include <cstdlib>" not in s:
    s = s.replace("#include <cstdint>\n", "#include <cstdint>\n#include <cstdlib>\n", 1)

needle = """#if TF_HIPBLASLT
  if (!blas_lt_.Init().ok()) {
    LOG(ERROR) << "Failed to initialize hipblasLt";
    return false;
  }
#endif
"""
repl = """#if TF_HIPBLASLT
  const char* disable_hipblaslt = std::getenv("TF_ROCM_DISABLE_HIPBLASLT_INIT");
  const bool skip_hipblaslt = disable_hipblaslt != nullptr &&
                              disable_hipblaslt[0] != '\\0' &&
                              disable_hipblaslt[0] != '0';
  if (skip_hipblaslt) {
    LOG(WARNING) << "Skipping hipBLASLt initialization due to TF_ROCM_DISABLE_HIPBLASLT_INIT="
                 << disable_hipblaslt;
  } else if (!blas_lt_.Init().ok()) {
    LOG(ERROR) << "Failed to initialize hipblasLt";
    return false;
  }
#endif
"""
if needle in s:
    s = s.replace(needle, repl, 1)

p.write_text(s, encoding="utf-8")
PY
  fi

  export TF_NEED_ROCM=1
  export TF_NEED_CUDA=0
  export TF_NEED_TENSORRT=0
  export TF_NEED_CLANG=0
  export TF_ROCM_CLANG=1
  export CLANG_COMPILER_PATH="${CCACHE_CLANG_WRAPPER}"
  export TF_ENABLE_XLA=1
  export TF_ROCM_AMDGPU_TARGETS="gfx1031"
  export ROCM_PATH
  export HIP_DEVICE_LIB_PATH="${ROCM_PATH}/lib/llvm/amdgcn/bitcode"
  export PYTHON_BIN_PATH="${PYTHON_BIN}"
  export CC_OPT_FLAGS="-O3"
  export TF_SET_ANDROID_WORKSPACE=0

  if [[ ! -d "${ROCM_PATH}/amdgcn" && -d "${ROCM_PATH}/lib/llvm/amdgcn" ]]; then
    ln -s "${ROCM_PATH}/lib/llvm/amdgcn" "${ROCM_PATH}/amdgcn"
  fi

  purge_bazel_local_config_rocm
  set +e
  yes "" | ./configure
  cfg_rc=${PIPESTATUS[1]:-1}
  set -e
  if [[ "${cfg_rc}" -ne 0 && "${cfg_rc}" -ne 141 ]]; then
    echo "TensorFlow configure failed with exit code ${cfg_rc}"
    exit "${cfg_rc}"
  fi
  patch_rocm_crosstool_builtin_includes
  force_disable_generated_rocm_hipblaslt

  bazelisk --output_user_root="${BAZEL_OUTPUT_USER_ROOT}" build \
    --config=opt \
    --config=rocm \
    $( [[ "${BAZEL_VERBOSE_FAILURES}" == "1" ]] && echo "--verbose_failures" ) \
    --jobs="${JOBS}" \
    --repo_env=TF_ROCM_CLANG=1 \
    --repo_env=CLANG_COMPILER_PATH="${CCACHE_CLANG_WRAPPER}" \
    --action_env=PATH="${PATH}" \
    --action_env=CLANG_COMPILER_PATH="${CCACHE_CLANG_WRAPPER}" \
    --action_env=LIBRARY_PATH="${LIBRARY_PATH}" \
    --action_env=HIP_DEVICE_LIB_PATH="${HIP_DEVICE_LIB_PATH}" \
    --action_env=CCACHE_DIR="${CCACHE_DIR}" \
    --action_env=CCACHE_BASEDIR="${CCACHE_BASEDIR}" \
    --action_env=CCACHE_COMPILERCHECK="${CCACHE_COMPILERCHECK}" \
    --linkopt=-L"${SYSLIBS_DIR}" \
    --host_linkopt=-L"${SYSLIBS_DIR}" \
    //tensorflow/tools/pip_package:wheel
  WHEEL_HELPER="./bazel-bin/tensorflow/tools/pip_package/wheel"
  WHEEL_HOUSE="${TF_SRC_DIR}/bazel-bin/tensorflow/tools/pip_package/wheel_house"
  if [[ -x "${WHEEL_HELPER}" ]]; then
    "${WHEEL_HELPER}" \
      --output-name tensorflow_rocm_custom \
      --project-name tensorflow-rocm-custom \
      --output-dir "${WHEEL_OUT_DIR}"
  else
    shopt -s nullglob
    wheels=( "${WHEEL_HOUSE}"/tensorflow-*.whl )
    shopt -u nullglob
    if [[ "${#wheels[@]}" -eq 0 ]]; then
      echo "ERROR: No TensorFlow wheel found in ${WHEEL_HOUSE} and helper ${WHEEL_HELPER} is missing." >&2
      exit 1
    fi
    mkdir -p "${WHEEL_OUT_DIR}"
    cp -f "${wheels[@]}" "${WHEEL_OUT_DIR}/"
  fi

  echo "Done. Wheel(s):"
  ls -lh "${WHEEL_OUT_DIR}"/*.whl
  exit 0
fi

# TensorFlow ROCm configure needs local adjustments for our ROCm clang 22 stack.
if [[ -f "${TF_SRC_DIR}/third_party/gpus/rocm_configure.bzl" ]]; then
  ROCM_CFG="${TF_SRC_DIR}/third_party/gpus/rocm_configure.bzl"
else
  ROCM_CFG="${TF_SRC_DIR}/third_party/xla/third_party/gpus/rocm_configure.bzl"
fi
"${PYTHON_BIN}" - <<'PY' "${ROCM_CFG}"
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
lines = p.read_text().splitlines(keepends=True)

start = next(i for i, l in enumerate(lines) if l.startswith("def _rocm_include_path("))
end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("def ")), len(lines))
body = lines[start:end]

def ensure_after(match_substr: str, to_add: list[str]) -> None:
    idx = next((i for i, l in enumerate(body) if match_substr in l), None)
    if idx is None:
        return
    offset = 1
    for entry in to_add:
        if not any(entry.strip() == l.strip() for l in body):
            body.insert(idx + offset, entry)
            offset += 1

# Ensure LLVM headers are available (fixes llvm/ADT/ArrayRef.h not found).
ensure_after(
    "rocm_toolkit_path =",
    ['    inc_dirs.append(rocm_toolkit_path + "/lib/llvm/include")\n'],
)

# Ensure clang 21/22 resource include paths are known.
ensure_after(
    'inc_dirs.append(rocm_toolkit_path + "/lib/llvm/lib/clang/20/include")',
    [
        '        inc_dirs.append(rocm_toolkit_path + "/lib/llvm/lib/clang/21/include")\n',
        '        inc_dirs.append(rocm_toolkit_path + "/lib/llvm/lib/clang/22/include")\n',
    ],
)

lines[start:end] = body
p.write_text("".join(lines))
PY

# ROCm 7.x ships newer rocPRIM APIs; legacy specializations in gpu_prim.h fail.
# Guard that block for older ROCm only.
if [[ -f "${TF_SRC_DIR}/third_party/xla/xla/service/gpu/gpu_prim.h" ]]; then
  GPU_PRIM_H="${TF_SRC_DIR}/third_party/xla/xla/service/gpu/gpu_prim.h"
else
  GPU_PRIM_H="${TF_SRC_DIR}/tensorflow/core/kernels/gpu_prim.h"
fi
"${PYTHON_BIN}" - <<'PY' "${GPU_PRIM_H}"
import pathlib
import re
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()

if "TF_ROCM_VERSION < 70000" not in s:
    s = s.replace(
        "namespace rocprim {\nnamespace detail {\n",
        "#if (TF_ROCM_VERSION < 70000)\nnamespace rocprim {\nnamespace detail {\n",
        1,
    )
    s = s.replace(
        "};  // namespace rocprim\n",
        "};  // namespace rocprim\n#endif  // TF_ROCM_VERSION < 70000\n",
        1,
    )
    p.write_text(s)
PY

# Keep TMA metadata type complete in StreamExecutor headers for TF 2.20.
SE_H="${TF_SRC_DIR}/third_party/xla/xla/stream_executor/stream_executor.h"
CUDA_EXEC_H="${TF_SRC_DIR}/third_party/xla/xla/stream_executor/cuda/cuda_executor.h"
CUDA_EXEC_CC="${TF_SRC_DIR}/third_party/xla/xla/stream_executor/cuda/cuda_executor.cc"
"${PYTHON_BIN}" - <<'PY' "${SE_H}" "${CUDA_EXEC_H}" "${CUDA_EXEC_CC}"
import pathlib
import sys

se_h = pathlib.Path(sys.argv[1])

s = se_h.read_text()
if '#include "xla/stream_executor/gpu/tma_metadata.h"\n' not in s:
    s = s.replace(
        '#include "xla/stream_executor/stream.h"\n',
        '#include "xla/stream_executor/stream.h"\n#include "xla/stream_executor/gpu/tma_metadata.h"\n',
        1,
    )
s = s.replace("namespace gpu { class TmaDescriptor; }\n\n", "")
se_h.write_text(s)
PY

# hipBLASLt compatibility: some ROCm builds do not export hipblasStatusToString.
HIPBLASLT_WRAP_H="${TF_SRC_DIR}/third_party/xla/xla/stream_executor/rocm/hipblaslt_wrapper.h"
"${PYTHON_BIN}" - <<'PY' "${HIPBLASLT_WRAP_H}"
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()
s = s.replace("  __macro(hipblasStatusToString)\n", "")
p.write_text(s)
PY

# hipBLASLt pointer mode API changed naming (hipblasLtPointerMode_t).
HIP_BLAS_LT_H="${TF_SRC_DIR}/third_party/xla/xla/stream_executor/rocm/hip_blas_lt.h"
HIP_BLAS_LT_CC="${TF_SRC_DIR}/third_party/xla/xla/stream_executor/rocm/hip_blas_lt.cc"
ROCM_BLAS_CC="${TF_SRC_DIR}/third_party/xla/xla/stream_executor/rocm/rocm_blas.cc"
"${PYTHON_BIN}" - <<'PY' "${HIP_BLAS_LT_H}" "${HIP_BLAS_LT_CC}"
import pathlib
import sys

h = pathlib.Path(sys.argv[1])
cc = pathlib.Path(sys.argv[2])

hs = h.read_text()
hs = hs.replace("hipblasPointerMode_t pointer_mode() const {", "hipblasLtPointerMode_t pointer_mode() const {")
hs = hs.replace("return HIPBLAS_POINTER_MODE_HOST;", "return HIPBLASLT_POINTER_MODE_HOST;")
h.write_text(hs)

cs = cc.read_text()
cs = cs.replace("HIPBLAS_POINTER_MODE_DEVICE", "HIPBLASLT_POINTER_MODE_DEVICE")
cc.write_text(cs)
PY

# Allow disabling hipBLASLt initialization via env flag. This is useful on
# gfx1031 compatibility runs where hipblasLtCreate can crash inside COMGR.
"${PYTHON_BIN}" - <<'PY' "${ROCM_BLAS_CC}"
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()

if "#include <cstdlib>" not in s:
    s = s.replace("#include <cstdint>\n", "#include <cstdint>\n#include <cstdlib>\n", 1)

needle = """#if TF_HIPBLASLT
  if (!blas_lt_.Init().ok()) {
    LOG(ERROR) << "Failed to initialize hipblasLt";
    return false;
  }
#endif
"""
repl = """#if TF_HIPBLASLT
  const char* disable_hipblaslt = std::getenv("TF_ROCM_DISABLE_HIPBLASLT_INIT");
  const bool skip_hipblaslt = disable_hipblaslt != nullptr &&
                              disable_hipblaslt[0] != '\\0' &&
                              disable_hipblaslt[0] != '0';
  if (skip_hipblaslt) {
    LOG(WARNING) << "Skipping hipBLASLt initialization due to TF_ROCM_DISABLE_HIPBLASLT_INIT="
                 << disable_hipblaslt;
  } else if (!blas_lt_.Init().ok()) {
    LOG(ERROR) << "Failed to initialize hipblasLt";
    return false;
  }
#endif
"""
if needle in s:
    s = s.replace(needle, repl, 1)

p.write_text(s)
PY

# Fix strict include checking for grappler:devices in this ROCm setup.
GRAPPLER_BUILD="${TF_SRC_DIR}/tensorflow/core/grappler/BUILD"
GRAPPLER_DEVICES_CC="${TF_SRC_DIR}/tensorflow/core/grappler/devices.cc"
TF_PLATFORM_BUILD="${TF_SRC_DIR}/tensorflow/core/platform/BUILD"
"${PYTHON_BIN}" - <<'PY' "${TF_PLATFORM_BUILD}"
import pathlib
import re
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()

pattern = re.compile(
    r'(tf_cuda_library\(\n\s*name = "stream_executor",[\s\S]*?\n\s*cuda_deps\s*=)([\s\S]*?)(\n\s*features = \["-parse_headers"\],)',
    re.M,
)
new_s, n = pattern.subn(r'\1 [],\3', s, count=1)
if n == 1:
    s = new_s

# For ROCm-only builds, avoid pulling CUDA platform IDs into stream_executor
# targets, otherwise CUDA-only TMA sources get compiled.
s = s.replace('        "@local_xla//xla/stream_executor/cuda:cuda_platform_id",\n', "")

if s != p.read_text():
    p.write_text(s)
PY

TF_GPU_RUNTIME_BUILD="${TF_SRC_DIR}/tensorflow/core/common_runtime/gpu/BUILD"
"${PYTHON_BIN}" - <<'PY' "${TF_GPU_RUNTIME_BUILD}"
import pathlib
import re
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()

pattern = re.compile(
    r'(tf_cuda_library\(\n\s*name = "gpu_runtime_impl",[\s\S]*?\n\s*cuda_deps\s*=\s*)(\[[\s\S]*?\])(\s*,\n\s*defines = if_linux_x86_64)',
    re.M,
)
new_s, n = pattern.subn(r'\1[]\3', s, count=1)
s = new_s if n == 1 else s

# Force ROCm runtime dependency and prevent accidental CUDA runtime pull-in.
s = s.replace('        "@local_xla//xla/stream_executor/cuda:all_runtime",\n', "")

# Keep this build path ROCm-only; CUDA PJRT client drags in CUDA runtime.
s = s.replace('                "@local_xla//xla/pjrt/gpu:se_gpu_pjrt_client",\n', "")

if s != p.read_text():
    p.write_text(s)
PY

TF_PLATFORM_BUILD_CONFIG="${TF_SRC_DIR}/tensorflow/core/platform/build_config.default.bzl"
"${PYTHON_BIN}" - <<'PY' "${TF_PLATFORM_BUILD_CONFIG}"
import pathlib
import re
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()
s = re.sub(
    r"\]\s*\+\s*if_cuda\(\[\n\s*Label\(\"@local_xla//xla/stream_executor:cuda_platform\"\),\n\s*\]\)",
    "]",
    s,
    count=1,
)
if s != p.read_text():
    p.write_text(s)
PY

TF_KERNELS_BUILD="${TF_SRC_DIR}/tensorflow/core/kernels/BUILD"
"${PYTHON_BIN}" - <<'PY' "${TF_KERNELS_BUILD}"
import pathlib
import re
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()

pattern = re.compile(
    r'(tf_kernel_library\(\n\s*name = "concat_lib",[\s\S]*?\n\s*deps = \[)([\s\S]*?)(\n\s*\],\n\s*alwayslink = 0,\n\))',
    re.M,
)

required = [
    '        "//tensorflow/core/common_runtime/gpu:gpu_lib",',
    '        "//tensorflow/core/platform:stream_executor",',
    '        "@local_xla//xla:autotune_results_proto_cc",',
    '        "@local_xla//xla:autotuning_proto_cc",',
    '        "@local_xla//xla/stream_executor:device_description_proto_cc",',
    '        "@local_xla//xla/stream_executor/cuda:cuda_compute_capability_proto_cc",',
    '        "@local_xla//xla/tsl/protobuf:dnn_proto_cc",',
]

def repl(m):
    deps_block = m.group(2)
    for dep in required:
        if dep not in deps_block:
            deps_block += "\n" + dep
    return m.group(1) + deps_block + m.group(3)

new_s, n = pattern.subn(repl, s, count=1)
if n == 1 and new_s != s:
    p.write_text(new_s)
PY

GPU_MANAGED_ALLOCATOR_CC="${TF_SRC_DIR}/tensorflow/core/common_runtime/gpu/gpu_managed_allocator.cc"
"${PYTHON_BIN}" - <<'PY' "${GPU_MANAGED_ALLOCATOR_CC}"
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()
s = s.replace("  void** result = 0;\n", "  void* result = nullptr;\n")
s = s.replace("  CHECK_EQ(hipHostMalloc(&result, num_bytes, 0), 0);\n", "  CHECK_EQ(hipHostMalloc(&result, num_bytes, 0), 0);\n")
s = s.replace("  ptr = reinterpret_cast<void*>(result);\n", "  ptr = result;\n")
p.write_text(s)
PY

TFL_SCHEMA_GENERATED_H="${TF_SRC_DIR}/tensorflow/compiler/mlir/lite/schema/schema_generated.h"
"${PYTHON_BIN}" - <<'PY' "${TFL_SCHEMA_GENERATED_H}"
import pathlib
import re
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()

# Allow newer FlatBuffers versions in this local toolchain.
s = re.sub(
    r"static_assert\(FLATBUFFERS_VERSION_MAJOR == 24 &&",
    "static_assert(FLATBUFFERS_VERSION_MAJOR >= 24 &&",
    s,
    count=1,
)
s = s.replace("FLATBUFFERS_VERSION_MINOR == 3", "FLATBUFFERS_VERSION_MINOR >= 3")
s = s.replace("FLATBUFFERS_VERSION_REVISION == 25", "FLATBUFFERS_VERSION_REVISION >= 0")

p.write_text(s)
PY

CUDA_STATUS_H="${TF_SRC_DIR}/third_party/xla/xla/stream_executor/cuda/cuda_status.h"
CUDA_STATUS_CC="${TF_SRC_DIR}/third_party/xla/xla/stream_executor/cuda/cuda_status.cc"
"${PYTHON_BIN}" - <<'PY' "${CUDA_STATUS_H}" "${CUDA_STATUS_CC}"
import pathlib
import sys

h = pathlib.Path(sys.argv[1])
cc = pathlib.Path(sys.argv[2])

hs = h.read_text()
if "SE_HAS_CUDA_HEADERS" not in hs:
    hs = hs.replace(
        '#include "third_party/gpus/cuda/include/cuda.h"\n#include "third_party/gpus/cuda/include/cuda_runtime_api.h"\n',
        """#if __has_include("third_party/gpus/cuda/include/cuda.h")\n#define SE_HAS_CUDA_HEADERS 1\n#include "third_party/gpus/cuda/include/cuda.h"\n#include "third_party/gpus/cuda/include/cuda_runtime_api.h"\n#else\n#define SE_HAS_CUDA_HEADERS 0\nenum CUresult : int {\n  CUDA_SUCCESS = 0,\n  CUDA_ERROR_OUT_OF_MEMORY = 2,\n  CUDA_ERROR_NOT_FOUND = 500,\n};\nenum cudaError_t : int {\n  cudaSuccess = 0,\n};\n#endif\n""",
        1,
    )
    h.write_text(hs)

cs = cc.read_text()
if "SE_HAS_CUDA_HEADERS" not in cs:
    cs = cs.replace(
        '#include "third_party/gpus/cuda/include/cuda.h"\n#include "third_party/gpus/cuda/include/cuda_runtime_api.h"\n#include "third_party/gpus/cuda/include/driver_types.h"\n',
        """#if __has_include("third_party/gpus/cuda/include/cuda.h")\n#define SE_HAS_CUDA_HEADERS 1\n#include "third_party/gpus/cuda/include/cuda.h"\n#include "third_party/gpus/cuda/include/cuda_runtime_api.h"\n#include "third_party/gpus/cuda/include/driver_types.h"\n#else\n#define SE_HAS_CUDA_HEADERS 0\n#endif\n""",
        1,
    )
    cs = cs.replace(
        "absl::Status ToStatusSlow(CUresult result, absl::string_view detail) {\n"
        "  const char* error_name;\n"
        "  std::string error_detail;\n"
        "  if (cuGetErrorName(result, &error_name)) {\n"
        "    error_detail = absl::StrCat(detail, \": UNKNOWN ERROR (\",\n"
        "                                static_cast<int>(result), \")\");\n"
        "  } else {\n"
        "    const char* error_string;\n"
        "    if (cuGetErrorString(result, &error_string)) {\n"
        "      error_detail = absl::StrCat(detail, \": \", error_name);\n"
        "    } else {\n"
        "      error_detail = absl::StrCat(detail, \": \", error_name, \": \", error_string);\n"
        "    }\n"
        "  }\n"
        "\n"
        "  if (result == CUDA_ERROR_OUT_OF_MEMORY) {\n"
        "    return absl::ResourceExhaustedError(error_detail);\n"
        "  } else if (result == CUDA_ERROR_NOT_FOUND) {\n"
        "    return absl::NotFoundError(error_detail);\n"
        "  } else {\n"
        "    return absl::InternalError(absl::StrCat(\"CUDA error: \", error_detail));\n"
        "  }\n"
        "}\n",
        """absl::Status ToStatusSlow(CUresult result, absl::string_view detail) {\n#if SE_HAS_CUDA_HEADERS\n  const char* error_name;\n  std::string error_detail;\n  if (cuGetErrorName(result, &error_name)) {\n    error_detail = absl::StrCat(detail, ": UNKNOWN ERROR (",\n                                static_cast<int>(result), ")");\n  } else {\n    const char* error_string;\n    if (cuGetErrorString(result, &error_string)) {\n      error_detail = absl::StrCat(detail, ": ", error_name);\n    } else {\n      error_detail = absl::StrCat(detail, ": ", error_name, ": ", error_string);\n    }\n  }\n\n  if (result == CUDA_ERROR_OUT_OF_MEMORY) {\n    return absl::ResourceExhaustedError(error_detail);\n  } else if (result == CUDA_ERROR_NOT_FOUND) {\n    return absl::NotFoundError(error_detail);\n  } else {\n    return absl::InternalError(absl::StrCat("CUDA error: ", error_detail));\n  }\n#else\n  return absl::InternalError(\n      absl::StrCat("CUDA error (stub): ", detail, " code=", result));\n#endif\n}\n""",
        1,
    )
    cs = cs.replace(
        "absl::Status ToStatusSlow(cudaError_t result, absl::string_view detail) {\n"
        "  std::string error_detail(detail);\n"
        "  const char* error_name = cudaGetErrorName(result);\n"
        "  const char* error_string = cudaGetErrorString(result);\n"
        "  if (error_name == nullptr) {\n"
        "    absl::StrAppend(&error_detail, \": UNKNOWN ERROR (\",\n"
        "                    static_cast<int>(result), \")\");\n"
        "  } else {\n"
        "    absl::StrAppend(&error_detail, \": \", error_name);\n"
        "  }\n"
        "\n"
        "  if (error_string != nullptr) {\n"
        "    absl::StrAppend(&error_detail, \": \", error_string);\n"
        "  }\n"
        "\n"
        "  return absl::InternalError(\n"
        "      absl::StrCat(\"CUDA Runtime error: \", error_detail));\n"
        "}\n",
        """absl::Status ToStatusSlow(cudaError_t result, absl::string_view detail) {\n#if SE_HAS_CUDA_HEADERS\n  std::string error_detail(detail);\n  const char* error_name = cudaGetErrorName(result);\n  const char* error_string = cudaGetErrorString(result);\n  if (error_name == nullptr) {\n    absl::StrAppend(&error_detail, ": UNKNOWN ERROR (",\n                    static_cast<int>(result), ")");\n  } else {\n    absl::StrAppend(&error_detail, ": ", error_name);\n  }\n\n  if (error_string != nullptr) {\n    absl::StrAppend(&error_detail, ": ", error_string);\n  }\n\n  return absl::InternalError(\n      absl::StrCat("CUDA Runtime error: ", error_detail));\n#else\n  return absl::InternalError(\n      absl::StrCat("CUDA Runtime error (stub): ", detail, " code=", result));\n#endif\n}\n""",
        1,
    )
    cc.write_text(cs)
PY

CUDA_CONTEXT_H="${TF_SRC_DIR}/third_party/xla/xla/stream_executor/cuda/cuda_context.h"
CUDA_CONTEXT_CC="${TF_SRC_DIR}/third_party/xla/xla/stream_executor/cuda/cuda_context.cc"
"${PYTHON_BIN}" - <<'PY' "${CUDA_CONTEXT_H}" "${CUDA_CONTEXT_CC}"
import pathlib
import sys

h = pathlib.Path(sys.argv[1])
cc = pathlib.Path(sys.argv[2])

hs = h.read_text()
if "__has_include(\"third_party/gpus/cuda/include/cuda.h\")" not in hs:
    hs = hs.replace(
        '#include "third_party/gpus/cuda/include/cuda.h"\n',
        """#if __has_include("third_party/gpus/cuda/include/cuda.h")\n#define SE_HAS_CUDA_HEADERS 1\n#include "third_party/gpus/cuda/include/cuda.h"\n#else\n#define SE_HAS_CUDA_HEADERS 0\ntypedef void* CUcontext;\ntypedef int CUdevice;\n#endif\n""",
        1,
    )
    h.write_text(hs)

cs = cc.read_text()
if "__has_include(\"third_party/gpus/cuda/include/cuda.h\")" not in cs:
    cs = cs.replace(
        '#include "third_party/gpus/cuda/include/cuda.h"\n',
        """#if __has_include("third_party/gpus/cuda/include/cuda.h")\n#define SE_HAS_CUDA_HEADERS 1\n#include "third_party/gpus/cuda/include/cuda.h"\n#else\n#define SE_HAS_CUDA_HEADERS 0\ntypedef void* CUcontext;\ntypedef int CUdevice;\n#endif\n""",
        1,
    )

if "#if SE_HAS_CUDA_HEADERS" not in cs:
    cs = cs.replace(
        "namespace stream_executor::gpu {\n\nnamespace {\n",
        """namespace stream_executor::gpu {\n\n#if !SE_HAS_CUDA_HEADERS\n\nContextMap<CUcontext, CudaContext>* CudaContext::GetContextMap() {\n  static auto* context_map =\n      new ContextMap<CUcontext, CudaContext>([](void*) { return 0; });\n  return context_map;\n}\n\nCudaContext::~CudaContext() { GetContextMap()->Remove(context()); }\n\nabsl::StatusOr<CudaContext*> CudaContext::Create(int device_ordinal,\n                                                 CUdevice) {\n  return GetContextMap()->Add(nullptr, device_ordinal);\n}\n\nvoid CudaContext::SetActive() {}\n\nbool CudaContext::IsActive() const { return false; }\n\nabsl::Status CudaContext::Synchronize() { return absl::OkStatus(); }\n\n#else\n\nnamespace {\n""",
        1,
    )
    cs = cs.replace(
        "}  // namespace stream_executor::gpu\n",
        """#endif\n\n}  // namespace stream_executor::gpu\n""",
        1,
    )
    cc.write_text(cs)
PY

CUDA_EVENT_H="${TF_SRC_DIR}/third_party/xla/xla/stream_executor/cuda/cuda_event.h"
CUDA_EVENT_CC="${TF_SRC_DIR}/third_party/xla/xla/stream_executor/cuda/cuda_event.cc"
"${PYTHON_BIN}" - <<'PY' "${CUDA_EVENT_H}" "${CUDA_EVENT_CC}"
import pathlib
import sys

h = pathlib.Path(sys.argv[1])
cc = pathlib.Path(sys.argv[2])

hs = h.read_text()
if "__has_include(\"third_party/gpus/cuda/include/cuda.h\")" not in hs:
    hs = hs.replace(
        '#include "third_party/gpus/cuda/include/cuda.h"\n',
        """#if __has_include("third_party/gpus/cuda/include/cuda.h")\n#define SE_HAS_CUDA_HEADERS 1\n#include "third_party/gpus/cuda/include/cuda.h"\n#else\n#define SE_HAS_CUDA_HEADERS 0\ntypedef void* CUevent;\n#endif\n""",
        1,
    )
    h.write_text(hs)

cs = cc.read_text()
if "__has_include(\"third_party/gpus/cuda/include/cuda.h\")" not in cs:
    cs = cs.replace(
        '#include "third_party/gpus/cuda/include/cuda.h"\n',
        """#if __has_include("third_party/gpus/cuda/include/cuda.h")\n#define SE_HAS_CUDA_HEADERS 1\n#include "third_party/gpus/cuda/include/cuda.h"\n#else\n#define SE_HAS_CUDA_HEADERS 0\ntypedef void* CUevent;\ntypedef void* CUstream;\n#endif\n""",
        1,
    )
if "#if !SE_HAS_CUDA_HEADERS" not in cs:
    cs = cs.replace(
        "namespace stream_executor {\nnamespace gpu {\nnamespace {\n",
        """namespace stream_executor {\nnamespace gpu {\n\n#if !SE_HAS_CUDA_HEADERS\n\nEvent::Status CudaEvent::PollForStatus() { return Event::Status::kPending; }\n\nabsl::Status CudaEvent::WaitForEventOnExternalStream(std::intptr_t) {\n  return absl::OkStatus();\n}\n\nabsl::StatusOr<CudaEvent> CudaEvent::Create(StreamExecutor* executor,\n                                            bool) {\n  return CudaEvent(executor, nullptr);\n}\n\nCudaEvent::~CudaEvent() = default;\n\nCudaEvent& CudaEvent::operator=(CudaEvent&& other) {\n  if (this == &other) return *this;\n  executor_ = other.executor_;\n  handle_ = other.handle_;\n  other.executor_ = nullptr;\n  other.handle_ = nullptr;\n  return *this;\n}\n\nCudaEvent::CudaEvent(CudaEvent&& other)\n    : executor_(other.executor_), handle_(other.handle_) {\n  other.executor_ = nullptr;\n  other.handle_ = nullptr;\n}\n\n#else\n\nnamespace {\n""",
        1,
    )
    cs = cs.replace(
        "}  // namespace gpu\n}  // namespace stream_executor\n",
        """#endif\n\n}  // namespace gpu\n}  // namespace stream_executor\n""",
        1,
    )
    cc.write_text(cs)
PY

CUDA_KERNEL_H="${TF_SRC_DIR}/third_party/xla/xla/stream_executor/cuda/cuda_kernel.h"
CUDA_KERNEL_CC="${TF_SRC_DIR}/third_party/xla/xla/stream_executor/cuda/cuda_kernel.cc"
"${PYTHON_BIN}" - <<'PY' "${CUDA_KERNEL_H}" "${CUDA_KERNEL_CC}"
import pathlib
import sys

h = pathlib.Path(sys.argv[1])
cc = pathlib.Path(sys.argv[2])

hs = h.read_text()
if "__has_include(\"third_party/gpus/cuda/include/cuda.h\")" not in hs:
    hs = hs.replace(
        '#include "third_party/gpus/cuda/include/cuda.h"\n',
        """#if __has_include("third_party/gpus/cuda/include/cuda.h")\n#define SE_HAS_CUDA_HEADERS 1\n#include "third_party/gpus/cuda/include/cuda.h"\n#else\n#define SE_HAS_CUDA_HEADERS 0\ntypedef void* CUfunction;\n#endif\n""",
        1,
    )
    h.write_text(hs)

cs = cc.read_text()
if "__has_include(\"third_party/gpus/cuda/include/cuda.h\")" not in cs:
    cs = cs.replace(
        '#include "third_party/gpus/cuda/include/cuda.h"\n',
        """#if __has_include("third_party/gpus/cuda/include/cuda.h")\n#define SE_HAS_CUDA_HEADERS 1\n#include "third_party/gpus/cuda/include/cuda.h"\n#else\n#define SE_HAS_CUDA_HEADERS 0\ntypedef void* CUfunction;\n#endif\n""",
        1,
    )
if "#if !SE_HAS_CUDA_HEADERS" not in cs:
    cs = cs.replace(
        "namespace stream_executor {\nnamespace gpu {\n\nnamespace {\n",
        """namespace stream_executor {\nnamespace gpu {\n\n#if !SE_HAS_CUDA_HEADERS\n\nabsl::StatusOr<int32_t> CudaKernel::GetMaxOccupiedBlocksPerCore(\n    ThreadDim, size_t) const {\n  return 0;\n}\n\nabsl::StatusOr<KernelMetadata> CudaKernel::GetKernelMetadata() {\n  return KernelMetadata();\n}\n\nabsl::Status CudaKernel::Launch(const ThreadDim&, const BlockDim&,\n                                const std::optional<ClusterDim>&,\n                                Stream*, const KernelArgs&) {\n  return absl::OkStatus();\n}\n\n#else\n\nnamespace {\n""",
        1,
    )
    cs = cs.replace(
        "}  // namespace gpu\n}  // namespace stream_executor\n",
        """#endif\n\n}  // namespace gpu\n}  // namespace stream_executor\n""",
        1,
    )
    cc.write_text(cs)
PY

CUDA_TIMER_CC="${TF_SRC_DIR}/third_party/xla/xla/stream_executor/cuda/cuda_timer.cc"
"${PYTHON_BIN}" - <<'PY' "${CUDA_TIMER_CC}"
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()

if "__has_include(\"third_party/gpus/cuda/include/cuda.h\")" not in s:
    s = s.replace(
        '#include "third_party/gpus/cuda/include/cuda.h"\n',
        """#if __has_include("third_party/gpus/cuda/include/cuda.h")\n#define SE_HAS_CUDA_HEADERS 1\n#include "third_party/gpus/cuda/include/cuda.h"\n#else\n#define SE_HAS_CUDA_HEADERS 0\n#endif\n""",
        1,
    )

if "#if !SE_HAS_CUDA_HEADERS" not in s:
    s = s.replace(
        "namespace stream_executor::gpu {\n\nnamespace {\n",
        """namespace stream_executor::gpu {\n\n#if !SE_HAS_CUDA_HEADERS\n\nCudaTimer::~CudaTimer() = default;\n\nabsl::StatusOr<absl::Duration> CudaTimer::GetElapsedDuration() {\n  if (is_stopped_) {\n    return absl::FailedPreconditionError(\"Measuring inactive timer\");\n  }\n  is_stopped_ = true;\n  return absl::ZeroDuration();\n}\n\nabsl::StatusOr<CudaTimer> CudaTimer::Create(StreamExecutor* executor,\n                                            Stream* stream,\n                                            TimerType) {\n  TF_ASSIGN_OR_RETURN(CudaEvent start_event,\n                      CudaEvent::Create(executor, /*allow_timing=*/true));\n  TF_ASSIGN_OR_RETURN(CudaEvent stop_event,\n                      CudaEvent::Create(executor, /*allow_timing=*/true));\n  return CudaTimer(executor, std::move(start_event), std::move(stop_event),\n                   stream, GpuSemaphore{});\n}\n\n#else\n\nnamespace {\n""",
        1,
    )
    s = s.replace(
        "}  // namespace stream_executor::gpu\n",
        """#endif\n\n}  // namespace stream_executor::gpu\n""",
        1,
    )

p.write_text(s)
PY

CUDA_COMMAND_BUFFER_H="${TF_SRC_DIR}/third_party/xla/xla/stream_executor/cuda/cuda_command_buffer.h"
CUDA_COMMAND_BUFFER_CC="${TF_SRC_DIR}/third_party/xla/xla/stream_executor/cuda/cuda_command_buffer.cc"
"${PYTHON_BIN}" - <<'PY' "${CUDA_COMMAND_BUFFER_H}" "${CUDA_COMMAND_BUFFER_CC}"
import pathlib
import sys

h = pathlib.Path(sys.argv[1])
cc = pathlib.Path(sys.argv[2])

hs = h.read_text()
if "__has_include(\"third_party/gpus/cuda/include/cuda.h\")" not in hs:
    hs = hs.replace(
        '#include "third_party/gpus/cuda/include/cuda.h"\n',
        """#if __has_include("third_party/gpus/cuda/include/cuda.h")\n#define SE_HAS_CUDA_HEADERS 1\n#include "third_party/gpus/cuda/include/cuda.h"\n#else\n#define SE_HAS_CUDA_HEADERS 0\n#ifndef CUDA_VERSION\n#define CUDA_VERSION 0\n#endif\ntypedef unsigned long long cuuint64_t;\ntypedef void* CUgraph;\ntypedef void* CUgraphExec;\ntypedef void* CUgraphNode;\ntypedef void* CUfunction;\ntypedef void* CUstream;\ntypedef void* CUcontext;\ntypedef unsigned long long CUdeviceptr;\n#endif\n""",
        1,
    )
    h.write_text(hs)

cs = cc.read_text()
if "__has_include(\"third_party/gpus/cuda/include/cuda.h\")" not in cs:
    cs = cs.replace(
        '#include "third_party/gpus/cuda/include/cuda.h"\n',
        """#if __has_include("third_party/gpus/cuda/include/cuda.h")\n#define SE_HAS_CUDA_HEADERS 1\n#include "third_party/gpus/cuda/include/cuda.h"\n#else\n#define SE_HAS_CUDA_HEADERS 0\n#ifndef CUDA_VERSION\n#define CUDA_VERSION 0\n#endif\ntypedef unsigned long long cuuint64_t;\ntypedef void* CUgraph;\ntypedef void* CUgraphExec;\ntypedef void* CUgraphNode;\ntypedef void* CUfunction;\ntypedef void* CUstream;\ntypedef void* CUcontext;\ntypedef unsigned long long CUdeviceptr;\n#endif\n""",
        1,
    )
if "#if !SE_HAS_CUDA_HEADERS" not in cs:
    cs = cs.replace(
        "namespace stream_executor::gpu {\nnamespace {\n",
        """namespace stream_executor::gpu {\n\n#if !SE_HAS_CUDA_HEADERS\n\nusing GraphNodeHandle = GpuCommandBuffer::GraphNodeHandle;\nusing GraphConditionalHandle = GpuCommandBuffer::GraphConditionalHandle;\n\nnamespace {\nabsl::Status CudaUnavailable() {\n  return absl::UnimplementedError("CUDA command buffer requires CUDA headers");\n}\n}  // namespace\n\nabsl::StatusOr<std::unique_ptr<CudaCommandBuffer>> CudaCommandBuffer::Create(\n    Mode, StreamExecutor*, CudaContext*) {\n  return CudaUnavailable();\n}\n\nabsl::StatusOr<GraphNodeHandle> CudaCommandBuffer::CreateSetCaseConditionNode(\n    absl::Span<const GraphConditionalHandle>, DeviceMemory<uint8_t>, bool,\n    int32_t, bool, absl::Span<const GraphNodeHandle>) {\n  return CudaUnavailable();\n}\n\nabsl::Status CudaCommandBuffer::UpdateSetCaseConditionNode(\n    GraphNodeHandle, absl::Span<const GraphConditionalHandle>,\n    DeviceMemory<uint8_t>, bool, int32_t, bool) {\n  return CudaUnavailable();\n}\n\nabsl::StatusOr<GraphNodeHandle> CudaCommandBuffer::CreateSetWhileConditionNode(\n    GraphConditionalHandle, DeviceMemory<bool>,\n    absl::Span<const GraphNodeHandle>) {\n  return CudaUnavailable();\n}\n\nabsl::Status CudaCommandBuffer::UpdateSetWhileConditionNode(\n    GraphNodeHandle, GraphConditionalHandle, DeviceMemory<bool>) {\n  return CudaUnavailable();\n}\n\nabsl::StatusOr<CudaCommandBuffer::NoOpKernel*> CudaCommandBuffer::GetNoOpKernel() {\n  return CudaUnavailable();\n}\n\nabsl::StatusOr<GpuCommandBuffer::GraphConditionalNodeHandle>\nCudaCommandBuffer::CreateConditionalNode(absl::Span<const GraphNodeHandle>,\n                                         GraphConditionalHandle,\n                                         ConditionType) {\n  return CudaUnavailable();\n}\n\nabsl::StatusOr<GraphNodeHandle> CudaCommandBuffer::CreateMemsetNode(\n    absl::Span<const GraphNodeHandle>, DeviceMemoryBase, BitPattern, size_t) {\n  return CudaUnavailable();\n}\n\nabsl::Status CudaCommandBuffer::UpdateMemsetNode(GraphNodeHandle,\n                                                  DeviceMemoryBase,\n                                                  BitPattern, size_t) {\n  return CudaUnavailable();\n}\n\nabsl::StatusOr<GraphNodeHandle> CudaCommandBuffer::CreateMemcpyD2DNode(\n    absl::Span<const GraphNodeHandle>, DeviceMemoryBase, DeviceMemoryBase,\n    uint64_t) {\n  return CudaUnavailable();\n}\n\nabsl::Status CudaCommandBuffer::UpdateMemcpyD2DNode(\n    GraphNodeHandle, DeviceMemoryBase, DeviceMemoryBase, uint64_t) {\n  return CudaUnavailable();\n}\n\nabsl::Status CudaCommandBuffer::PopulateDnnGraphNode(\n    dnn::DnnGraph&, Stream&, absl::Span<DeviceMemoryBase>) {\n  return CudaUnavailable();\n}\n\nabsl::Status CudaCommandBuffer::UpdateDnnGraphNode(\n    dnn::DnnGraph&, Stream&, absl::Span<DeviceMemoryBase>, GraphNodeHandle) {\n  return CudaUnavailable();\n}\n\nabsl::StatusOr<GraphNodeHandle> CudaCommandBuffer::CreateChildNode(\n    absl::Span<const GraphNodeHandle>, const CommandBuffer&) {\n  return CudaUnavailable();\n}\n\nabsl::Status CudaCommandBuffer::UpdateChildNode(GraphNodeHandle,\n                                                 const CommandBuffer&) {\n  return CudaUnavailable();\n}\n\nabsl::StatusOr<GraphNodeHandle> CudaCommandBuffer::CreateKernelNode(\n    absl::Span<const GraphNodeHandle>, StreamPriority, const ThreadDim&,\n    const BlockDim&, const Kernel&, const KernelArgsPackedArrayBase&) {\n  return CudaUnavailable();\n}\n\nabsl::Status CudaCommandBuffer::UpdateKernelNode(\n    GraphNodeHandle, const ThreadDim&, const BlockDim&, const Kernel&,\n    const KernelArgsPackedArrayBase&) {\n  return CudaUnavailable();\n}\n\nabsl::StatusOr<GraphNodeHandle> CudaCommandBuffer::CreateEmptyNode(\n    absl::Span<const GraphNodeHandle>) {\n  return CudaUnavailable();\n}\n\nabsl::Status CudaCommandBuffer::Trace(\n    Stream*, absl::AnyInvocable<absl::Status()> function) {\n  return function();\n}\n\nabsl::Status CudaCommandBuffer::LaunchGraph(Stream*) { return CudaUnavailable(); }\n\nabsl::StatusOr<size_t> CudaCommandBuffer::GetNodeCount() const {\n  return static_cast<size_t>(0);\n}\n\nabsl::Status CudaCommandBuffer::SetPriority(StreamPriority) {\n  return CudaUnavailable();\n}\n\nabsl::Status CudaCommandBuffer::PrepareFinalization() { return absl::OkStatus(); }\n\nabsl::StatusOr<GraphConditionalHandle>\nCudaCommandBuffer::CreateConditionalHandle() {\n  return CudaUnavailable();\n}\n\nabsl::Status CudaCommandBuffer::WriteGraphToDotFile(absl::string_view) {\n  return CudaUnavailable();\n}\n\nabsl::Status CudaCommandBuffer::InstantiateGraph() { return CudaUnavailable(); }\n\nstd::unique_ptr<ScopedUpdateMode> CudaCommandBuffer::ActivateUpdateMode(\n    GpuCommandBuffer*) {\n  return nullptr;\n}\n\nCudaCommandBuffer::~CudaCommandBuffer() = default;\n\nabsl::Status CudaCommandBuffer::CheckCanBeUpdated() {\n  return CudaUnavailable();\n}\n\n#else\n\nnamespace {\n""",
        1,
    )
    cs = cs.replace(
        "}  // namespace stream_executor::gpu\n",
        """#endif\n\n}  // namespace stream_executor::gpu\n""",
        1,
    )
cc.write_text(cs)
PY

CUDA_STREAM_H="${TF_SRC_DIR}/third_party/xla/xla/stream_executor/cuda/cuda_stream.h"
CUDA_STREAM_CC="${TF_SRC_DIR}/third_party/xla/xla/stream_executor/cuda/cuda_stream.cc"
"${PYTHON_BIN}" - <<'PY' "${CUDA_STREAM_H}" "${CUDA_STREAM_CC}"
import pathlib
import sys

h = pathlib.Path(sys.argv[1])
cc = pathlib.Path(sys.argv[2])

hs = h.read_text()
if "__has_include(\"third_party/gpus/cuda/include/cuda.h\")" not in hs:
    hs = hs.replace(
        '#include "third_party/gpus/cuda/include/cuda.h"\n',
        """#if __has_include("third_party/gpus/cuda/include/cuda.h")\n#define SE_HAS_CUDA_HEADERS 1\n#include "third_party/gpus/cuda/include/cuda.h"\n#else\n#define SE_HAS_CUDA_HEADERS 0\ntypedef void* CUstream;\n#endif\n""",
        1,
    )
    h.write_text(hs)

cs = cc.read_text()
if "__has_include(\"third_party/gpus/cuda/include/cuda.h\")" not in cs:
    cs = cs.replace(
        '#include "third_party/gpus/cuda/include/cuda.h"\n',
        """#if __has_include("third_party/gpus/cuda/include/cuda.h")\n#define SE_HAS_CUDA_HEADERS 1\n#include "third_party/gpus/cuda/include/cuda.h"\n#else\n#define SE_HAS_CUDA_HEADERS 0\ntypedef void* CUstream;\ntypedef void* CUevent;\ntypedef unsigned long long CUdeviceptr;\n#endif\n""",
        1,
    )
if "#if !SE_HAS_CUDA_HEADERS" not in cs:
    cs = cs.replace(
        "namespace stream_executor {\nnamespace gpu {\n\nnamespace {\n",
        """namespace stream_executor {\nnamespace gpu {\n\n#if !SE_HAS_CUDA_HEADERS\n\nnamespace {\nabsl::Status CudaUnavailable() {\n  return absl::UnimplementedError("CUDA stream requires CUDA headers");\n}\n}  // namespace\n\nabsl::StatusOr<std::unique_ptr<CudaStream>> CudaStream::Create(\n    StreamExecutor* executor,\n    std::optional<std::variant<StreamPriority, int>> priority) {\n  TF_ASSIGN_OR_RETURN(auto completed_event,\n                      CudaEvent::Create(executor, /*allow_timing=*/false));\n  return std::unique_ptr<CudaStream>(new CudaStream(\n      executor, std::move(completed_event), std::move(priority), nullptr));\n}\n\nabsl::Status CudaStream::WaitFor(Stream*) { return CudaUnavailable(); }\n\nabsl::Status CudaStream::RecordEvent(Event*) { return CudaUnavailable(); }\n\nabsl::Status CudaStream::WaitFor(Event*) { return CudaUnavailable(); }\n\nabsl::Status CudaStream::RecordCompletedEvent() { return absl::OkStatus(); }\n\nCudaStream::~CudaStream() = default;\n\nabsl::Status CudaStream::BlockHostUntilDone() { return absl::OkStatus(); }\n\nabsl::Status CudaStream::Memset32(DeviceMemoryBase*, uint32_t, uint64_t) {\n  return CudaUnavailable();\n}\n\nabsl::Status CudaStream::MemZero(DeviceMemoryBase*, uint64_t) {\n  return CudaUnavailable();\n}\n\nabsl::Status CudaStream::Memcpy(DeviceMemoryBase*, const DeviceMemoryBase&,\n                                uint64_t) {\n  return CudaUnavailable();\n}\n\nabsl::Status CudaStream::Memcpy(DeviceMemoryBase*, const void*, uint64_t) {\n  return CudaUnavailable();\n}\n\nabsl::Status CudaStream::Memcpy(void*, const DeviceMemoryBase&, uint64_t) {\n  return CudaUnavailable();\n}\n\nabsl::Status CudaStream::DoHostCallbackWithStatus(\n    absl::AnyInvocable<absl::Status() &&> callback) {\n  return std::move(callback)();\n}\n\nabsl::Status CudaStream::LaunchKernel(const ThreadDim&, const BlockDim&,\n                                      const std::optional<ClusterDim>&, void*,\n                                      absl::string_view, void**, int64_t) {\n  return CudaUnavailable();\n}\n\nvoid CudaStream::SetName(std::string) {}\n\n}  // namespace gpu\n\n#else\n\nnamespace {\n""",
        1,
    )
    cs = cs.replace(
        "}  // namespace stream_executor\n",
        """#endif\n\n}  // namespace stream_executor\n""",
        1,
    )
cc.write_text(cs)
PY

CUDA_DELAY_KERNEL_CC="${TF_SRC_DIR}/third_party/xla/xla/stream_executor/cuda/delay_kernel_cuda.cu.cc"
"${PYTHON_BIN}" - <<'PY' "${CUDA_DELAY_KERNEL_CC}"
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()

if "__global__ void DelayKernel" in s:
    p.write_text("""/* Copyright 2024 The OpenXLA Authors. */\n\n#include \"xla/stream_executor/cuda/delay_kernel.h\"\n\n#include \"absl/status/status.h\"\n#include \"xla/stream_executor/gpu/gpu_semaphore.h\"\n\nnamespace stream_executor::gpu {\n\nabsl::StatusOr<GpuSemaphore> LaunchDelayKernel(Stream* stream) {\n  return GpuSemaphore::Create(stream->parent());\n}\n\nnamespace delay_kernel {\nvoid* kernel() { return nullptr; }\n}  // namespace delay_kernel\n\n}  // namespace stream_executor::gpu\n""")
PY

CUDA_BLAS_UTILS_H="${TF_SRC_DIR}/third_party/xla/xla/stream_executor/cuda/cuda_blas_utils.h"
CUDA_BLAS_UTILS_CC="${TF_SRC_DIR}/third_party/xla/xla/stream_executor/cuda/cuda_blas_utils.cc"
"${PYTHON_BIN}" - <<'PY' "${CUDA_BLAS_UTILS_H}" "${CUDA_BLAS_UTILS_CC}"
import pathlib
import sys

h = pathlib.Path(sys.argv[1])
cc = pathlib.Path(sys.argv[2])

hs = h.read_text()
if "__has_include(\"third_party/gpus/cuda/include/cublas_v2.h\")" not in hs:
    hs = hs.replace(
        '#include "third_party/gpus/cuda/include/cublas_v2.h"\n#include "third_party/gpus/cuda/include/library_types.h"\n',
        """#if __has_include("third_party/gpus/cuda/include/cublas_v2.h")\n#define SE_HAS_CUBLAS_HEADERS 1\n#include "third_party/gpus/cuda/include/cublas_v2.h"\n#include "third_party/gpus/cuda/include/library_types.h"\n#else\n#define SE_HAS_CUBLAS_HEADERS 0\n#ifndef CUDA_VERSION\n#define CUDA_VERSION 0\n#endif\ntypedef int cublasStatus_t;\nstatic constexpr cublasStatus_t CUBLAS_STATUS_SUCCESS = 0;\ntypedef int cudaDataType_t;\nstatic constexpr cudaDataType_t CUDA_R_16F = 0;\nstatic constexpr cudaDataType_t CUDA_R_16BF = 1;\nstatic constexpr cudaDataType_t CUDA_R_32F = 2;\nstatic constexpr cudaDataType_t CUDA_R_64F = 3;\nstatic constexpr cudaDataType_t CUDA_R_8I = 4;\nstatic constexpr cudaDataType_t CUDA_R_32I = 5;\nstatic constexpr cudaDataType_t CUDA_C_32F = 6;\nstatic constexpr cudaDataType_t CUDA_C_64F = 7;\nstatic constexpr cudaDataType_t CUDA_R_8F_E5M2 = 8;\nstatic constexpr cudaDataType_t CUDA_R_8F_E4M3 = 9;\ntypedef int cublasComputeType_t;\nstatic constexpr cublasComputeType_t CUBLAS_COMPUTE_16F = 0;\nstatic constexpr cublasComputeType_t CUBLAS_COMPUTE_32F = 1;\nstatic constexpr cublasComputeType_t CUBLAS_COMPUTE_64F = 2;\nstatic constexpr cublasComputeType_t CUBLAS_COMPUTE_32I = 3;\nstatic constexpr cublasComputeType_t CUBLAS_COMPUTE_32F_FAST_16F = 4;\nstatic constexpr cublasComputeType_t CUBLAS_COMPUTE_32F_FAST_16BF = 5;\nstatic constexpr cublasComputeType_t CUBLAS_COMPUTE_32F_FAST_TF32 = 6;\ntypedef int cublasOperation_t;\nstatic constexpr cublasOperation_t CUBLAS_OP_N = 0;\nstatic constexpr cublasOperation_t CUBLAS_OP_T = 1;\nstatic constexpr cublasOperation_t CUBLAS_OP_C = 2;\n#endif\n""",
        1,
    )
    h.write_text(hs)

cs = cc.read_text()
if "__has_include(\"third_party/gpus/cuda/include/cublas_v2.h\")" not in cs:
    cs = cs.replace(
        '#include "third_party/gpus/cuda/include/cublas_v2.h"\n#include "third_party/gpus/cuda/include/cuda.h"\n#include "third_party/gpus/cuda/include/library_types.h"\n',
        """#if __has_include("third_party/gpus/cuda/include/cublas_v2.h")\n#define SE_HAS_CUBLAS_HEADERS 1\n#include "third_party/gpus/cuda/include/cublas_v2.h"\n#include "third_party/gpus/cuda/include/cuda.h"\n#include "third_party/gpus/cuda/include/library_types.h"\n#else\n#define SE_HAS_CUBLAS_HEADERS 0\n#endif\n""",
        1,
    )
    cs = cs.replace(
        "const char* ToString(cublasStatus_t status) {\n#if CUDA_VERSION >= 11050  // `GetStatusString` was added in 11.4 update 2.\n  return cublasGetStatusString(status);\n#else\n  return \"cublas error\";\n#endif  // CUDA_VERSION >= 11050\n}\n",
        """const char* ToString(cublasStatus_t status) {\n#if SE_HAS_CUBLAS_HEADERS && CUDA_VERSION >= 11050  // `GetStatusString` was added in 11.4 update 2.\n  return cublasGetStatusString(status);\n#else\n  (void)status;\n  return "cublas error";\n#endif  // SE_HAS_CUBLAS_HEADERS && CUDA_VERSION >= 11050\n}\n""",
        1,
    )
    cc.write_text(cs)
PY

XLA_SERVICE_BUILD="${TF_SRC_DIR}/third_party/xla/xla/service/BUILD"
"${PYTHON_BIN}" - <<'PY' "${XLA_SERVICE_BUILD}"
import pathlib
import re
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()

# ROCm-only: drop CUDA branch from gpu_plugin_without_collectives deps.
s = re.sub(
    r'\)\s*\+\s*if_cuda_is_configured\(\[\n[\s\S]*?\n\s*\]\)\s*\+\s*if_rocm_is_configured\(\[',
    ') + if_rocm_is_configured([',
    s,
    count=1,
)

p.write_text(s)
PY

KERNEL_GEN_GPU_BLOB_PASS="${TF_SRC_DIR}/tensorflow/compiler/mlir/tools/kernel_gen/transforms/gpu_kernel_to_blob_pass.cc"
"${PYTHON_BIN}" - <<'PY' "${KERNEL_GEN_GPU_BLOB_PASS}"
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()

if 'xla/service/gpu/llvm_gpu_backend/amdgpu_backend.h' not in s:
    s = s.replace(
        '#include "xla/stream_executor/gpu/asm_compiler.h"\n',
        '#include "xla/stream_executor/gpu/asm_compiler.h"\n#include "xla/service/gpu/llvm_gpu_backend/amdgpu_backend.h"\n',
        1,
    )

s = s.replace(
    'auto hsaco_or = xla::gpu::amdgpu::CompileToHsaco(\n'
    '          llvm_module_copy.get(),\n'
    '          tensorflow::se::RocmComputeCapability{arch_str}, options,\n'
    '          options.DebugString());\n',
    'stream_executor::GpuComputeCapability gpu_version =\n'
    '          tensorflow::se::RocmComputeCapability{arch_str};\n'
    '      auto hsaco_or = xla::gpu::amdgpu::CompileToHsaco(\n'
    '          llvm_module_copy.get(), gpu_version, options,\n'
    '          options.DebugString());\n',
    1,
)

s = s.replace(
    'images.push_back({arch_str, std::move(hsaco)});\n',
    'images.push_back(tensorflow::se::HsacoImage{arch_str, std::move(hsaco)});\n',
    1,
)

p.write_text(s)
PY

XLA_STREAM_EXECUTOR_BUILD="${TF_SRC_DIR}/third_party/xla/xla/stream_executor/BUILD"
"${PYTHON_BIN}" - <<'PY' "${XLA_STREAM_EXECUTOR_BUILD}"
import pathlib
import re
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()

def ensure_dep(block_name: str, dep_line: str, text: str) -> str:
    pat = re.compile(
        rf'(name = "{re.escape(block_name)}",[\s\S]*?\n\s*(?:deps|protodeps) = \[)([\s\S]*?)(\n\s*\],)',
        re.M,
    )
    m = pat.search(text)
    if not m:
        return text
    body = m.group(2)
    if dep_line not in body:
        body = body + "\n" + dep_line
    return text[:m.start()] + m.group(1) + body + m.group(3) + text[m.end():]

s = ensure_dep(
    "device_description_proto",
    '        "//xla/stream_executor/cuda:cuda_compute_capability_proto",',
    s,
)
s = ensure_dep(
    "device_description",
    '        "//xla/stream_executor/cuda:cuda_compute_capability",',
    s,
)
p.write_text(s)
PY

XLA_CUDA_BUILD="${TF_SRC_DIR}/third_party/xla/xla/stream_executor/cuda/BUILD"
"${PYTHON_BIN}" - <<'PY' "${XLA_CUDA_BUILD}"
import pathlib
import re
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()

# ROCm-only: do not pull CUDA TMA utility target in transitive CUDA executor deps.
s = s.replace('        ":tma_util",\n', "")

# ROCm-only: keep target shape but drop CUDA runtime payload from all_runtime.
s = re.sub(
    r'(cc_library\(\n\s*name = "all_runtime",[\s\S]*?\n\s*deps = )\[[\s\S]*?\],(\n\s*alwayslink = 1,\n\))',
    r'\1[],\2',
    s,
    count=1,
)

p.write_text(s)
PY

XLA_CUDA_BUILD_DEFS="${TF_SRC_DIR}/third_party/xla/xla/tsl/platform/default/cuda_build_defs.bzl"
"${PYTHON_BIN}" - <<'PY' "${XLA_CUDA_BUILD_DEFS}"
import pathlib
import re
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()

# Force ROCm-only behavior in this custom build.
s = re.sub(
    r"def if_cuda_is_configured\(x, no_cuda = \[\]\):\n\s*return _if_cuda_is_configured\(x, no_cuda\)\n",
    "def if_cuda_is_configured(x, no_cuda = []):\n    return no_cuda\n",
    s,
    count=1,
)
s = re.sub(
    r"def is_cuda_configured\(\):\n\s*return _is_cuda_configured\(\)\n",
    "def is_cuda_configured():\n    return False\n",
    s,
    count=1,
)
s = re.sub(
    r"def if_cuda_newer_than\(wanted_ver, if_true, if_false = \[\]\):\n\s*return _if_cuda_newer_than\(wanted_ver, if_true, if_false\)\n",
    "def if_cuda_newer_than(wanted_ver, if_true, if_false = []):\n    return if_false\n",
    s,
    count=1,
)

p.write_text(s)
PY

XLA_GPU_SERVICE_BUILD="${TF_SRC_DIR}/third_party/xla/xla/service/gpu/BUILD"
"${PYTHON_BIN}" - <<'PY' "${XLA_GPU_SERVICE_BUILD}"
import pathlib
import re
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()

pattern = re.compile(
    r'(tf_proto_library\(\n\s*name = "fusion_process_dump_proto",[\s\S]*?\n\s*protodeps = \[)([\s\S]*?)(\n\s*\],\n\))',
    re.M,
)

def repl(m):
    deps = m.group(2)
    add = '        "//xla/stream_executor/cuda:cuda_compute_capability_proto",'
    if add not in deps:
        deps = deps + "\n" + add
    return m.group(1) + deps + m.group(3)

new_s, n = pattern.subn(repl, s, count=1)
if n == 1:
    p.write_text(new_s)
PY

XLA_PJRT_GPU_BUILD="${TF_SRC_DIR}/third_party/xla/xla/pjrt/gpu/BUILD"
"${PYTHON_BIN}" - <<'PY' "${XLA_PJRT_GPU_BUILD}"
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()
s = s.replace('        "//xla/stream_executor/gpu:gpu_cudamallocasync_allocator",\n', "")
s = s.replace('        "@local_config_cuda//cuda:cuda_headers",\n', "")
p.write_text(s)
PY

XLA_SE_GPU_BUILD="${TF_SRC_DIR}/third_party/xla/xla/stream_executor/gpu/BUILD"
"${PYTHON_BIN}" - <<'PY' "${XLA_SE_GPU_BUILD}"
import pathlib
import re
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()
s = s.replace('    srcs = ["gpu_cudamallocasync_allocator.cc"],\n', '    srcs = if_cuda_is_configured(["gpu_cudamallocasync_allocator.cc"]),\n')
s = s.replace('    hdrs = ["gpu_cudamallocasync_allocator.h"],\n', '    hdrs = if_cuda_is_configured(["gpu_cudamallocasync_allocator.h"]),\n')
pat = re.compile(
    r'(cc_library\(\n\s*name = "gpu_cudamallocasync_allocator",[\s\S]*?\n\s*deps = )\[([\s\S]*?)\](,\n\))',
    re.M,
)
m = pat.search(s)
if m:
    s = s[:m.start()] + m.group(1) + "if_cuda([" + m.group(2) + "])" + m.group(3) + s[m.end():]
p.write_text(s)
PY

"${PYTHON_BIN}" - <<'PY' "${GRAPPLER_DEVICES_CC}"
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()
s = s.replace(
    '#include "tensorflow/core/platform/stream_executor.h"\n',
    '#include "tensorflow/core/platform/stream_executor_no_cuda.h"\n',
)
p.write_text(s)
PY

"${PYTHON_BIN}" - <<'PY' "${GRAPPLER_BUILD}"
import pathlib
import re
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()
pattern = re.compile(
    r'(tf_cuda_library\(\n\s*name = "devices",[\s\S]*?\n\s*deps = \[)([\s\S]*?)(\n\s*\],\n\))',
    re.M,
)

replacement_deps = """
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@local_xla//xla:autotune_results_proto_cc",
        "@local_xla//xla:autotuning_proto_cc",
        "@local_xla//xla/stream_executor:stream_executor_h",
        "@local_xla//xla/stream_executor:platform",
        "@local_xla//xla/stream_executor:platform_manager",
        "@local_xla//xla/stream_executor:stream",
        "@local_xla//xla/stream_executor:device_description",
        "@local_xla//xla/stream_executor:device_description_proto_cc",
        "@local_xla//xla/stream_executor/cuda:cuda_platform_id",
        "@local_xla//xla/stream_executor/gpu:gpu_init",
        "@local_xla//xla/stream_executor/rocm:rocm_platform_id",
        "@local_xla//xla/tsl/protobuf:dnn_proto_cc",
        "@local_xla//xla/tsl/platform/default:dso_loader",
"""

def repl(m):
    return m.group(1) + "\n" + replacement_deps.rstrip() + m.group(3)

new_s, n = pattern.subn(repl, s, count=1)
if n == 1 and new_s != s:
    p.write_text(new_s)
PY

AUTOTUNE_MAPS_BUILD="${TF_SRC_DIR}/tensorflow/core/util/autotune_maps/BUILD"
"${PYTHON_BIN}" - <<'PY' "${AUTOTUNE_MAPS_BUILD}"
import pathlib
import re
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()

pat = re.compile(
    r'(tf_cuda_library\(\n\s*name = "conv_parameters",[\s\S]*?\n\s*deps = \[)([\s\S]*?)(\n\s*\],\n\))',
    re.M,
)
m = pat.search(s)
if m:
    deps = m.group(2)
    dep = '        "@local_xla//xla/stream_executor/cuda:cuda_platform_id",'
    if dep not in deps:
        deps = deps + "\n" + dep
        s = s[:m.start()] + m.group(1) + deps + m.group(3) + s[m.end():]
        p.write_text(s)
PY

TF2TRT_BUILD="${TF_SRC_DIR}/tensorflow/compiler/tf2tensorrt/BUILD"
"${PYTHON_BIN}" - <<'PY' "${TF2TRT_BUILD}"
import pathlib
import re
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()

pat = re.compile(
    r'(tf_cuda_library\(\n\s*name = "trt_engine_utils",[\s\S]*?\n\s*deps = \[)([\s\S]*?)(\n\s*\]\s*\+\s*if_tensorrt\(\[":tensorrt_lib"\]\),\n\))',
    re.M,
)
m = pat.search(s)
if m:
    deps = m.group(2)
    dep = '        "@local_xla//xla/stream_executor/cuda:cuda_platform_id",'
    if dep not in deps:
        deps = deps + "\n" + dep
        s = s[:m.start()] + m.group(1) + deps + m.group(3) + s[m.end():]
        p.write_text(s)
PY

TF_CORE_PLATFORM_BUILD="${TF_SRC_DIR}/tensorflow/core/platform/BUILD"
"${PYTHON_BIN}" - <<'PY' "${TF_CORE_PLATFORM_BUILD}"
import pathlib
import re
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()

def ensure_deps_for_target(text: str, target: str, deps_to_add: list[str]) -> str:
    pat = re.compile(
        rf'(name = "{re.escape(target)}",[\s\S]*?\n\s*deps = \[)([\s\S]*?)(\n\s*\](?:\s*\+\s*if_[\s\S]*?)?,\n\))',
        re.M,
    )
    m = pat.search(text)
    if not m:
        return text
    deps = m.group(2)
    for dep in deps_to_add:
        if dep not in deps:
            deps = deps + "\n" + dep
    return text[:m.start()] + m.group(1) + deps + m.group(3) + text[m.end():]

dep_lines = [
    '        "@local_xla//xla/stream_executor/cuda:cuda_platform_id",',
    '        "@local_xla//xla/stream_executor:launch_dim_proto_cc",',
    '        "@local_xla//xla/stream_executor:blas_proto_cc",',
    '        "@local_xla//xla/stream_executor:kernel_spec_proto_cc",',
]
s2 = ensure_deps_for_target(s, "stream_executor", dep_lines)
s2 = ensure_deps_for_target(s2, "stream_executor_no_cuda", dep_lines)
if s2 != s:
    p.write_text(s2)
PY

TF_CORE_KERNELS_BUILD="${TF_SRC_DIR}/tensorflow/core/kernels/BUILD"
"${PYTHON_BIN}" - <<'PY' "${TF_CORE_KERNELS_BUILD}"
import pathlib
import re
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()

pat = re.compile(
    r'(tf_kernel_library\(\n\s*name = "concat_lib",[\s\S]*?\n\s*deps = \[)([\s\S]*?)(\n\s*\],\n\s*alwayslink = 0,\n\))',
    re.M,
)
m = pat.search(s)
if m:
    deps = m.group(2)
    needed = [
        '        "//tensorflow/core/common_runtime/gpu:gpu_lib",',
        '        "//tensorflow/core/platform:stream_executor_no_cuda",',
        '        "@local_xla//xla/stream_executor:launch_dim_proto_cc",',
        '        "@local_xla//xla/stream_executor:blas_proto_cc",',
        '        "@local_xla//xla/stream_executor:kernel_spec_proto_cc",',
    ]
    changed = False
    for dep in needed:
        if dep not in deps:
            deps = deps + "\n" + dep
            changed = True
    if changed:
        s = s[:m.start()] + m.group(1) + deps + m.group(3) + s[m.end():]
        p.write_text(s)

# dynamic_stitch_op pulls gpu_device_array.h -> gpu_event_mgr.h
# and needs explicit gpu runtime dependency under strict include checking.
pat2 = re.compile(
    r'(tf_kernel_library\(\n\s*name = "dynamic_stitch_op",[\s\S]*?\n\s*deps = DYNAMIC_DEPS \+ \[)([\s\S]*?)(\n\s*\],\n\))',
    re.M,
)
m2 = pat2.search(s)
if m2:
    deps2 = m2.group(2)
    dep = '        "//tensorflow/core/common_runtime/gpu:gpu_lib",'
    if dep not in deps2:
        deps2 = deps2 + "\n" + dep
        s = s[:m2.start()] + m2.group(1) + deps2 + m2.group(3) + s[m2.end():]
        p.write_text(s)

# Handle one-line deps form used in TF 2.20:
# deps = DYNAMIC_DEPS + [":loose_headers"],
s = s.replace(
    "    deps = DYNAMIC_DEPS + [\":loose_headers\"],\n",
    "    deps = DYNAMIC_DEPS + [\":loose_headers\", \"//tensorflow/core/common_runtime/gpu:gpu_lib\"],\n",
)
p.write_text(s)

# sync_ops includes stream_executor proto headers; add explicit deps.
pat_sync = re.compile(
    r'(tf_kernel_library\(\n\s*name = "sync_ops",[\s\S]*?\n\s*deps = \[)([\s\S]*?)(\n\s*\],\n\))',
    re.M,
)
m_sync = pat_sync.search(s)
if m_sync:
    deps_sync = m_sync.group(2)
    needed_sync = [
        '        "//tensorflow/core/platform:stream_executor_no_cuda",',
        '        "@local_xla//xla/stream_executor:launch_dim_proto_cc",',
        '        "@local_xla//xla/stream_executor:blas_proto_cc",',
        '        "@local_xla//xla/stream_executor:kernel_spec_proto_cc",',
    ]
    changed_sync = False
    for dep in needed_sync:
        if dep not in deps_sync:
            deps_sync = deps_sync + "\n" + dep
            changed_sync = True
    if changed_sync:
        s = s[:m_sync.start()] + m_sync.group(1) + deps_sync + m_sync.group(3) + s[m_sync.end():]
        p.write_text(s)

# Do not apply broad regex rewrites across all tf_kernel_library blocks here.
# Earlier attempts caused index-shift corruption in BUILD syntax.
# Keep dependency adjustments targeted and explicit.

# Ensure shared math kernel deps include GPU event manager headers.
if 'MATH_DEPS = [\n    ":fill_functor",\n    "//tensorflow/core/common_runtime/gpu:gpu_lib",\n' not in s:
    s2 = s.replace(
        'MATH_DEPS = [\n    ":fill_functor",\n',
        'MATH_DEPS = [\n    ":fill_functor",\n    "//tensorflow/core/common_runtime/gpu:gpu_lib",\n',
        1,
    )
    if s2 != s:
        s = s2
        p.write_text(s)

# If MATH_DEPS already has gpu_lib, remove redundant per-target duplicates.
for needle in [
    '        "//tensorflow/core/common_runtime/gpu:gpu_lib",\n',
]:
    s2 = s.replace(
        '    deps = MATH_DEPS + [\n        "//tensorflow/core/kernels/mlir_generated:cast_op",\n' + needle + '    ],\n',
        '    deps = MATH_DEPS + [\n        "//tensorflow/core/kernels/mlir_generated:cast_op",\n    ],\n',
    )
    s2 = s2.replace(
        '    deps = MATH_DEPS + [\n        "//tensorflow/core:framework_internal",\n' + needle + '    ],\n',
        '    deps = MATH_DEPS + [\n        "//tensorflow/core:framework_internal",\n    ],\n',
    )
    s2 = s2.replace(
        '    deps = MATH_DEPS + [\n        "//tensorflow/core/kernels/mlir_generated:cwise_op",\n        "@com_google_absl//absl/base:prefetch",\n' + needle + '    ],\n',
        '    deps = MATH_DEPS + [\n        "//tensorflow/core/kernels/mlir_generated:cwise_op",\n        "@com_google_absl//absl/base:prefetch",\n    ],\n',
    )
    if s2 != s:
        s = s2
        p.write_text(s)

# ROCm-only build: avoid hard cublas_plugin edges from matmul targets.
s2 = s.replace(
    '    ] + mkl_deps() + if_cuda([\n'
    '        "@local_xla//xla/stream_executor/cuda:cublas_plugin",\n'
    '    ]) + if_rocm_hipblaslt([\n',
    '    ] + mkl_deps() + if_rocm_hipblaslt([\n',
)
s2 = s2.replace(
    '    deps = if_cuda_or_rocm([\n'
    '        "@com_google_absl//absl/container:flat_hash_map",\n'
    '        "@local_xla//xla:status_macros",\n'
    '        "@local_xla//xla:xla_data_proto_cc",\n'
    '        "@local_xla//xla/stream_executor/cuda:cublas_plugin",\n'
    '        "//tensorflow/core:framework",\n'
    '        "//tensorflow/core:lib",\n'
    '        "//tensorflow/core/platform:tensor_float_32_hdr_lib",\n'
    '        "//tensorflow/core/util:env_var",\n'
    '    ]) + if_cuda([\n'
    '        "@local_xla//xla/stream_executor/cuda:cublas_lt_header",\n'
    '    ]) + if_rocm([\n'
    '        "@local_xla//xla/stream_executor/rocm:hipblas_lt_header",\n',
    '    deps = if_cuda_or_rocm([\n'
    '        "@com_google_absl//absl/container:flat_hash_map",\n'
    '        "@local_xla//xla:status_macros",\n'
    '        "@local_xla//xla:xla_data_proto_cc",\n'
    '        "//tensorflow/core:framework",\n'
    '        "//tensorflow/core:lib",\n'
    '        "//tensorflow/core/platform:tensor_float_32_hdr_lib",\n'
    '        "//tensorflow/core/util:env_var",\n'
    '    ]) + if_rocm([\n'
    '        "@local_xla//xla/stream_executor/rocm:rocblas_plugin",\n'
    '    ]) + if_cuda([\n'
    '        "@local_xla//xla/stream_executor/cuda:cublas_plugin",\n'
    '        "@local_xla//xla/stream_executor/cuda:cublas_lt_header",\n'
    '    ]) + if_rocm([\n'
    '        "@local_xla//xla/stream_executor/rocm:hipblas_lt_header",\n',
)
if s2 != s:
    s = s2
    p.write_text(s)

# TensorFlow 2.20 stream_executor API: ROCm LRN backward no longer takes
# a separate nullptr workspace argument before scratch allocator.
LRN_CC = pathlib.Path(sys.argv[1]).parent / "lrn_op.cc"
if LRN_CC.exists():
    lrn = LRN_CC.read_text()
    lrn2 = lrn.replace(
        "output_image_data, input_grads_data, &output_grads_data,\n"
        "        /*workspace_allocator=*/nullptr, &scratch_allocator);",
        "output_image_data, input_grads_data, &output_grads_data,\n"
        "        &scratch_allocator);",
    )
    if lrn2 != lrn:
        LRN_CC.write_text(lrn2)
PY

IMAGE_BUILD="${TF_SRC_DIR}/tensorflow/core/kernels/image/BUILD"
"${PYTHON_BIN}" - <<'PY' "${IMAGE_BUILD}"
import pathlib
import re
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()

pat = re.compile(
    r'(tf_kernel_library\(\n\s*name = "crop_and_resize_op",[\s\S]*?\n\s*\] \+ if_cuda_or_rocm\(\[)([\s\S]*?)(\n\s*\]\),\n\))',
    re.M,
)
m = pat.search(s)
if m:
    deps = m.group(2)
    dep = '        "//tensorflow/core/common_runtime/gpu:gpu_lib",'
    if dep not in deps:
        deps = dep + "\n" + deps
        s = s[:m.start()] + m.group(1) + deps + m.group(3) + s[m.end():]
        p.write_text(s)
PY

CONV_GPU_CC="${TF_SRC_DIR}/tensorflow/core/kernels/conv_ops_gpu.cc"
"${PYTHON_BIN}" - <<'PY' "${CONV_GPU_CC}"
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()

s = s.replace(
    "if (!dnn->GetMIOpenConvolveAlgorithms(\n"
    "            kind, se::dnn::ToDataType<T>::value, stream, input_desc, input_ptr,\n"
    "            filter_desc, filter_ptr, output_desc, output_ptr, conv_desc,\n"
    "            &scratch_allocator, &algorithms)) {",
    "if (!dnn->GetMIOpenConvolveAlgorithms(\n"
    "            kind, se::dnn::ToDataType<T>::value, se::dnn::ToDataType<T>::value,\n"
    "            stream, input_desc, input_ptr, filter_desc, filter_ptr, output_desc,\n"
    "            output_ptr, conv_desc, &scratch_allocator, &algorithms)) {",
)
s = s.replace(
    "      result.set_scratch_bytes(profile_result.scratch_size());",
    "      result.set_scratch_bytes(scratch_allocator.TotalByteSize());",
)
s = s.replace(
    "            se::dnn::AlgorithmConfig(profile_algorithm,\n"
    "                                     miopen_algorithm.scratch_size()),",
    "            se::dnn::AlgorithmConfig(profile_algorithm,\n"
    "                                     scratch_allocator.TotalByteSize()),",
)

p.write_text(s)
PY

DTENSOR_SHAPE_UTILS_CC="${TF_SRC_DIR}/tensorflow/dtensor/mlir/shape_utils.cc"
"${PYTHON_BIN}" - <<'PY' "${DTENSOR_SHAPE_UTILS_CC}"
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()
s = s.replace(
    "ExtractGlobalOutputShape(cast<mlir::OpResult>(input_value.get()))",
    "ExtractGlobalOutputShape(llvm::cast<mlir::OpResult>(input_value.get()))",
)
p.write_text(s)
PY

DTENSOR_SPMD_EXPANSION_CC="${TF_SRC_DIR}/tensorflow/dtensor/mlir/spmd_expansion.cc"
"${PYTHON_BIN}" - <<'PY' "${DTENSOR_SPMD_EXPANSION_CC}"
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()
s = s.replace(
    "            cast<mlir::BlockArgument>(resource).getArgNumber();",
    "            llvm::cast<mlir::BlockArgument>(resource).getArgNumber();",
)
s = s.replace(
    "    if (isa<mlir::TF::ResourceType>(arg_type)) {",
    "    if (llvm::isa<mlir::TF::ResourceType>(arg_type)) {",
)
p.write_text(s)
PY

CORE_UTIL_BUILD="${TF_SRC_DIR}/tensorflow/core/util/BUILD"
"${PYTHON_BIN}" - <<'PY' "${CORE_UTIL_BUILD}"
import pathlib
import re
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()

pat = re.compile(
    r'(tf_kernel_library\(\n\s*name = "rocm_solvers",[\s\S]*?\n\s*deps = \[)([\s\S]*?)(\n\s*\] \+ if_rocm\(\[)',
    re.M,
)
m = pat.search(s)
if m:
    deps = m.group(2)
    needed = [
        '        "//tensorflow/core/common_runtime/gpu:gpu_lib",',
        '        "//tensorflow/core/platform:stream_executor_no_cuda",',
        '        "@local_xla//xla/stream_executor:platform_manager",',
    ]
    changed = False
    for dep in needed:
        if dep not in deps:
            deps = dep + "\n" + deps
            changed = True
    if changed:
        s = s[:m.start()] + m.group(1) + deps + m.group(3) + s[m.end():]
        p.write_text(s)

# In this ROCm-only build, keep cuda_solvers as a no-op stub so transitive
# deps do not pull CUDA cublas/cusolver targets.
s = s.replace(
    'tf_kernel_library(
'
    '    name = "cuda_solvers",
'
    '    srcs = ["cuda_solvers.cc"],
'
    '    hdrs = ["gpu_solvers.h"],
'
    '    compatible_with = [],
'
    '    features = ["-layering_check"],
'
    '    visibility = ["//tensorflow/core/kernels:friends"],
'
    '    deps = [
'
    '        "//tensorflow/core:framework",
'
    '        "//tensorflow/core:lib",
'
    '        "@local_xla//xla/stream_executor/cuda:cublas_plugin",
'
    '        "@local_xla//xla/tsl/cuda:cusolver",
'
    '    ],
'
    ')
',
    'tf_kernel_library(
'
    '    name = "cuda_solvers",
'
    '    srcs = [],
'
    '    hdrs = ["gpu_solvers.h"],
'
    '    compatible_with = [],
'
    '    features = ["-layering_check"],
'
    '    visibility = ["//tensorflow/core/kernels:friends"],
'
    '    deps = [
'
    '        "//tensorflow/core:framework",
'
    '        "//tensorflow/core:lib",
'
    '    ],
'
    ')
',
)
PY

GPU_SOLVERS_H="${TF_SRC_DIR}/tensorflow/core/util/gpu_solvers.h"
"${PYTHON_BIN}" - <<'PY' "${GPU_SOLVERS_H}"
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()

# ROCm 7.x installs rocBLAS headers under include/rocblas/rocblas.h.
s = s.replace(
    '#include "rocm/include/rocblas.h"\n',
    '#include "rocm/include/rocblas/rocblas.h"\n',
)

p.write_text(s)
PY

if [[ ! -x "${PYTHON_BIN}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi
# Some ROCm crosstool wrappers invoke `python` via /usr/bin/env.
# Provide a local `python` shim that points to the selected venv interpreter.
ln -sf "${PYTHON_BIN}" "${BAZEL_BIN_DIR}/python"
"${PYTHON_BIN}" -m pip install -U pip setuptools wheel numpy
"${PYTHON_BIN}" -m pip install -U keras_preprocessing packaging requests opt_einsum six

# Configure TensorFlow ROCm build non-interactively.
export TF_NEED_ROCM=1
export TF_NEED_CUDA=0
export TF_NEED_TENSORRT=0
export TF_NEED_CLANG=0
export TF_ROCM_CLANG=1
export CLANG_COMPILER_PATH="${CCACHE_CLANG_WRAPPER}"
export TF_ENABLE_XLA=1
export TF_ROCM_AMDGPU_TARGETS="gfx1031"
export ROCM_PATH
export HIP_DEVICE_LIB_PATH="${ROCM_PATH}/lib/llvm/amdgcn/bitcode"
export PYTHON_BIN_PATH="${PYTHON_BIN}"
export CC_OPT_FLAGS="-O3"
export TF_SET_ANDROID_WORKSPACE=0

# Some toolchains expect ROCm device bitcode under /opt/rocm/amdgcn/bitcode.
if [[ ! -d "${ROCM_PATH}/amdgcn" && -d "${ROCM_PATH}/lib/llvm/amdgcn" ]]; then
  ln -s "${ROCM_PATH}/lib/llvm/amdgcn" "${ROCM_PATH}/amdgcn"
fi

purge_bazel_local_config_rocm
set +e
yes "" | ./configure
cfg_rc=$?
set -e
if [[ "${cfg_rc}" -ne 0 && "${cfg_rc}" -ne 141 ]]; then
  echo "TensorFlow configure failed with exit code ${cfg_rc}"
  exit "${cfg_rc}"
fi
patch_rocm_crosstool_builtin_includes
force_disable_generated_rocm_hipblaslt

bazelisk --output_user_root="${BAZEL_OUTPUT_USER_ROOT}" build \
  --config=opt \
  --config=rocm \
  $( [[ "${BAZEL_VERBOSE_FAILURES}" == "1" ]] && echo "--verbose_failures" ) \
  --jobs="${JOBS}" \
  --repo_env=TF_ROCM_CLANG=1 \
  --repo_env=CLANG_COMPILER_PATH="${CCACHE_CLANG_WRAPPER}" \
  --action_env=PATH="${PATH}" \
  --action_env=CLANG_COMPILER_PATH="${CCACHE_CLANG_WRAPPER}" \
  --action_env=LIBRARY_PATH="${LIBRARY_PATH}" \
  --action_env=HIP_DEVICE_LIB_PATH="${HIP_DEVICE_LIB_PATH}" \
  --action_env=CCACHE_DIR="${CCACHE_DIR}" \
  --action_env=CCACHE_BASEDIR="${CCACHE_BASEDIR}" \
  --action_env=CCACHE_COMPILERCHECK="${CCACHE_COMPILERCHECK}" \
  --linkopt=-L"${SYSLIBS_DIR}" \
  --host_linkopt=-L"${SYSLIBS_DIR}" \
  //tensorflow/tools/pip_package:wheel
WHEEL_HELPER="./bazel-bin/tensorflow/tools/pip_package/wheel"
WHEEL_HOUSE="${TF_SRC_DIR}/bazel-bin/tensorflow/tools/pip_package/wheel_house"
if [[ -x "${WHEEL_HELPER}" ]]; then
  "${WHEEL_HELPER}" \
    --output-name tensorflow_rocm_custom \
    --project-name tensorflow-rocm-custom \
    --output-dir "${WHEEL_OUT_DIR}"
else
  shopt -s nullglob
  wheels=( "${WHEEL_HOUSE}"/tensorflow-*.whl )
  shopt -u nullglob
  if [[ "${#wheels[@]}" -eq 0 ]]; then
    echo "ERROR: No TensorFlow wheel found in ${WHEEL_HOUSE} and helper ${WHEEL_HELPER} is missing." >&2
    exit 1
  fi
  mkdir -p "${WHEEL_OUT_DIR}"
  cp -f "${wheels[@]}" "${WHEEL_OUT_DIR}/"
fi

echo "Done. Wheel(s):"
ls -lh "${WHEEL_OUT_DIR}"/*.whl
