#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_ID="tf_gpu_crash_matrix_$(date +%Y-%m-%d_%H%M%S)"
OUT_DIR="${ROOT}/validation/workspace/runs/${RUN_ID}"
OUT_LOG="${OUT_DIR}/summary.tsv"
META_LOG="${OUT_DIR}/meta.txt"

TF_VENV="${TF_VENV:-${ROOT}/validation/workspace/envs/tf_bench}"
BUILD_DIR="${BUILD_DIR:-build-stage2}"
IN_TREE_ROCM="${ROOT}/${BUILD_DIR}/dist/rocm"

mkdir -p "${OUT_DIR}"

cat >"${META_LOG}" <<EOF
run_id=${RUN_ID}
root=${ROOT}
tf_venv=${TF_VENV}
build_dir=${BUILD_DIR}
in_tree_rocm=${IN_TREE_ROCM}
timestamp=$(date -Is)
EOF

if [[ ! -x "${TF_VENV}/bin/python" ]]; then
  echo "ERROR: missing venv python: ${TF_VENV}/bin/python" >&2
  exit 1
fi

source "${TF_VENV}/bin/activate"

python -c "import tensorflow as tf; print(tf.__version__)" > "${OUT_DIR}/tensorflow_version.txt" 2>&1 || true
{
  echo "=== rocminfo (/opt/rocm) ==="
  /opt/rocm/bin/rocminfo 2>&1 || true
  echo
  echo "=== rocminfo (in-tree) ==="
  PATH="${IN_TREE_ROCM}/bin:${IN_TREE_ROCM}/llvm/bin:${PATH}" \
  LD_LIBRARY_PATH="${IN_TREE_ROCM}/lib:${IN_TREE_ROCM}/lib64:${IN_TREE_ROCM}/lib/host-math/lib:${IN_TREE_ROCM}/lib/rocm_sysdeps/lib:${IN_TREE_ROCM}/llvm/lib:${LD_LIBRARY_PATH:-}" \
  "${IN_TREE_ROCM}/bin/rocminfo" 2>&1 || true
} > "${OUT_DIR}/rocminfo.txt"

python - <<'PY' > "${OUT_DIR}/tf_libs.txt" 2>&1
import pathlib
import tensorflow as tf

sp = pathlib.Path(tf.__file__).resolve().parent
print("tensorflow_site_package", sp)
for so in ("libtensorflow_framework.so.2", "libtensorflow_cc.so.2"):
    p = (sp / ".." / so).resolve()
    print("candidate", p)
PY

while IFS= read -r cand; do
  so_path="${cand#candidate }"
  if [[ -f "${so_path}" ]]; then
    ldd "${so_path}" > "${OUT_DIR}/ldd_$(basename "${so_path}").txt" 2>&1 || true
  fi
done < <(grep '^candidate ' "${OUT_DIR}/tf_libs.txt" || true)

cat > "${OUT_DIR}/repro.py" <<'PY'
import json
import os
import sys
import tensorflow as tf

def loaded_paths():
    out = set()
    try:
        with open("/proc/self/maps", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 6 and p[5].startswith("/") and ".so" in p[5]:
                    out.add(p[5])
    except Exception as e:
        print("maps_error", type(e).__name__)
    return sorted(out)

def find_lib(paths, base):
    b = base + ".so"
    for p in paths:
        bn = os.path.basename(p)
        if bn == b or bn.startswith(b + "."):
            return p
    return ""

print("tf", tf.__version__, flush=True)
gpus = tf.config.list_physical_devices("GPU")
print("gpus", gpus, flush=True)
paths = loaded_paths()
libs = {}
for lib in ("libamdhip64", "libhsa-runtime64", "libamd_comgr", "libhipblaslt", "librocblas"):
    libs[lib] = find_lib(paths, lib)
print("loaded_libs", json.dumps(libs, sort_keys=True), flush=True)

if "--device-only" in sys.argv:
    if gpus:
        with tf.device("/GPU:0"):
            x = tf.constant([1.0], dtype=tf.float32)
        print("device_only_ok", float(x.numpy()[0]), flush=True)
    else:
        print("device_only_no_gpu", flush=True)
    raise SystemExit(0)

if gpus:
    with tf.device("/GPU:0"):
        a = tf.random.normal([1024, 1024], dtype=tf.float32)
        b = tf.random.normal([1024, 1024], dtype=tf.float32)
        c = tf.matmul(a, b)
    print("matmul_sum", float(tf.reduce_sum(c).numpy()), flush=True)
else:
    print("no_gpu_visible", flush=True)
PY

echo -e "case_id\tstack\thsa_override\tdisable_hipblaslt\tmode\trc" > "${OUT_LOG}"

case_id=0
for stack in system in_tree; do
  for hsa in none 10.3.0 10.3.1; do
    for disable in 0 1; do
      for mode in device_only matmul; do
        case_id=$((case_id+1))
        case_name="$(printf "c%02d_%s_hsa-%s_lt-%s_%s" "${case_id}" "${stack}" "${hsa}" "${disable}" "${mode}")"
        case_log="${OUT_DIR}/${case_name}.log"
        case_bt="${OUT_DIR}/${case_name}.gdb.txt"

        run_env=(
          "PYTHONUNBUFFERED=1"
          "PYTHONFAULTHANDLER=1"
          "TF_CPP_MIN_LOG_LEVEL=1"
          "TF_ROCM_DISABLE_HIPBLASLT_INIT=${disable}"
        )
        if [[ "${hsa}" != "none" ]]; then
          run_env+=("HSA_OVERRIDE_GFX_VERSION=${hsa}")
        fi

        if [[ "${stack}" == "in_tree" ]]; then
          run_env+=(
            "ROCM_PATH=${IN_TREE_ROCM}"
            "HIP_PATH=${IN_TREE_ROCM}"
            "HSA_PATH=${IN_TREE_ROCM}"
            "PATH=${IN_TREE_ROCM}/bin:${IN_TREE_ROCM}/llvm/bin:${PATH}"
            "LD_LIBRARY_PATH=${IN_TREE_ROCM}/lib:${IN_TREE_ROCM}/lib64:${IN_TREE_ROCM}/lib/host-math/lib:${IN_TREE_ROCM}/lib/rocm_sysdeps/lib:${IN_TREE_ROCM}/llvm/lib:${LD_LIBRARY_PATH:-}"
          )
        fi

        py_args=("${OUT_DIR}/repro.py")
        if [[ "${mode}" == "device_only" ]]; then
          py_args+=("--device-only")
        fi

        set +e
        env "${run_env[@]}" python -u -X faulthandler "${py_args[@]}" > "${case_log}" 2>&1
        rc=$?
        set -e

        if [[ "${rc}" -ne 0 ]]; then
          set +e
          env "${run_env[@]}" gdb -q -batch -ex run -ex bt --args python -u "${py_args[@]}" > "${case_bt}" 2>&1
          set -e
        fi

        echo -e "${case_name}\t${stack}\t${hsa}\t${disable}\t${mode}\t${rc}" >> "${OUT_LOG}"
      done
    done
  done
done

echo "wrote: ${OUT_DIR}"
echo "summary: ${OUT_LOG}"
