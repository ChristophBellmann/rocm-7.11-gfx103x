from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any

from core.context import Context
from core.reporting.models import StepResult
from core.rocm_env import deactivated_env
from core.runner import fmt_duration, run_cmd
from steps.shared import append_power, baseline_avg_w, with_power_sampler


def _latest_wheel(wheel_dir: Path) -> Path | None:
    wheels = sorted(wheel_dir.glob("*.whl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return wheels[0] if wheels else None


def _resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = repo_root / p
    return p


def _matmul_script() -> str:
    return r"""
import os
import time
import tensorflow as tf

min_s = float(os.environ.get("ROCM_VALIDATION_TF_MIN_S", "5"))
mnk = int(os.environ.get("ROCM_VALIDATION_TF_MNK", "4096"))
dtype_name = os.environ.get("ROCM_VALIDATION_TF_DTYPE", "float16").strip().lower()
dtype = tf.float16 if dtype_name in ("fp16", "float16", "half") else tf.float32

print("tf_version", getattr(tf, "__version__", ""))
gpus = tf.config.list_physical_devices("GPU")
print("gpu_count", len(gpus))
if not gpus:
    print("GPU_NOT_AVAILABLE")
    raise SystemExit(2)

try:
    import os as _os
    import re as _re

    def _loaded_so_paths():
        out = set()
        with open("/proc/self/maps", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                p = parts[5]
                if p.startswith("/") and ".so" in p:
                    out.add(p)
        return sorted(out)

    def _find_lib(paths, base):
        pat = _re.compile(_re.escape(base) + r"(\..*)?$")
        for p in paths:
            b = _os.path.basename(p)
            if pat.match(b):
                return p
        return ""

    _paths = _loaded_so_paths()
    for _lib in ("libamdhip64.so", "libhsa-runtime64.so", "libhipblaslt.so"):
        _p = _find_lib(_paths, _lib)
        if _p:
            print("loaded", _lib, _p)
except Exception as e:
    print("loaded_libs_error", type(e).__name__)

with tf.device("/GPU:0"):
    a = tf.random.uniform((mnk, mnk), dtype=dtype)
    b = tf.random.uniform((mnk, mnk), dtype=dtype)
    c = tf.matmul(a, b)
    _ = float(tf.reduce_sum(c).numpy())

t0 = time.time()
runs = 0
with tf.device("/GPU:0"):
    while (time.time() - t0) < min_s:
        c = tf.matmul(a, b)
        _ = float(tf.reduce_sum(c).numpy())
        runs += 1
dt = time.time() - t0

flops = runs * 2.0 * mnk * mnk * mnk
print("GPU_OK")
print("op", "C=A*B (dense matmul)")
print("mnk", mnk)
print("dtype", "fp16" if dtype == tf.float16 else "fp32")
print("runs", runs)
print("seconds", dt)
print("tflops_est", flops / dt / 1e12 if dt > 0 else 0.0)
""".strip()


def step_tensorflow_matmul(
    ctx: Context,
    cfg: dict[str, Any],
    build_dir: str,
    rocm_dist: Path,
    env: dict[str, str],
    log: Path | None,
) -> StepResult:
    wl = cfg.get("workloads", {}).get("tensorflow", {}) or {}
    use_in_tree = bool(wl.get("use_in_tree_rocm", False))
    run_env = dict(env if use_in_tree else deactivated_env(env, rocm_dist))

    wheel_out_dir = _resolve_repo_path(
        ctx.repo_root,
        str(
            wl.get(
                "wheel_out_dir",
                ctx.repo_root / "validation" / "workspace" / "cache" / "wheels" / "tensorflow_rocm_custom",
            )
        ),
    )
    wheel = _latest_wheel(wheel_out_dir)
    if wheel is None:
        return StepResult(build_dir, "TensorFlow matmul (GPU)", "SKIP", "0ms", f"no wheel in {wheel_out_dir}")

    # Keep TF ABI/deps stable in the validation venv.
    pip_install = [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "numpy<2", "protobuf<7", str(wheel)]
    r_install = run_cmd(ctx.repo_root, run_env, pip_install, 1800, log)
    if r_install.rc != 0:
        return StepResult(build_dir, "TensorFlow matmul (GPU)", "FAIL", fmt_duration(r_install.dur_ms), f"pip rc={r_install.rc}")

    min_s = float(wl.get("min_bench_s", 5.0) or 5.0)
    mnk = int(wl.get("matmul_mnk", 4096) or 4096)
    dtype = str(wl.get("dtype", "float16") or "float16")
    timeout_s = int(cfg.get("timeouts_s", {}).get("tensorflow_functional", 900))
    run_env["ROCM_VALIDATION_TF_MIN_S"] = str(min_s)
    run_env["ROCM_VALIDATION_TF_MNK"] = str(mnk)
    run_env["ROCM_VALIDATION_TF_DTYPE"] = dtype
    run_env.setdefault("PYTHONUNBUFFERED", "1")
    run_env.setdefault("PYTHONFAULTHANDLER", "1")
    run_env.setdefault("TF_ROCM_DISABLE_HIPBLASLT", str(wl.get("disable_hipblaslt", 1)))
    run_env.setdefault("TF_ROCM_USE_HIPBLASLT", str(wl.get("use_hipblaslt", 0)))
    run_env.setdefault("TF_ROCM_DISABLE_HIPBLASLT_INIT", str(wl.get("disable_hipblaslt_init", 1)))
    hsa_override = str(wl.get("hsa_override_gfx_version", "") or "").strip()
    if hsa_override and "HSA_OVERRIDE_GFX_VERSION" not in run_env:
        run_env["HSA_OVERRIDE_GFX_VERSION"] = hsa_override

    script = _matmul_script()

    def run_one(sampler):
        t0 = time.monotonic()
        r = run_cmd(ctx.repo_root, run_env, [sys.executable, "-u", "-X", "faulthandler", "-c", script], timeout_s, log)
        wall_s = time.monotonic() - t0
        return r, wall_s, sampler

    r, wall_s, sampler = with_power_sampler(cfg, build_dir=build_dir, fn=run_one)
    out = (r.out + "\n" + r.err).strip()
    if r.rc != 0:
        if "GPU_NOT_AVAILABLE" in out:
            return StepResult(build_dir, "TensorFlow matmul (GPU)", "FAIL", fmt_duration(r.dur_ms), "tf GPU not available")
        if r.rc < 0:
            return StepResult(build_dir, "TensorFlow matmul (GPU)", "FAIL", fmt_duration(r.dur_ms), f"signal={-r.rc} (crash)")
        return StepResult(build_dir, "TensorFlow matmul (GPU)", "FAIL", fmt_duration(r.dur_ms), f"rc={r.rc}")
    if "GPU_OK" not in out:
        return StepResult(build_dir, "TensorFlow matmul (GPU)", "FAIL", fmt_duration(r.dur_ms), "GPU validation marker missing")

    tflops = ""
    dtype_out = ""
    op = ""
    tf_ver = ""
    runs = 0.0
    secs = 0.0
    m = re.search(r"^tflops_est\s+([0-9.]+)$", out, re.MULTILINE)
    if m:
        tflops = m.group(1)
    m = re.search(r"^dtype\s+(.+)$", out, re.MULTILINE)
    if m:
        dtype_out = m.group(1).strip()
    m = re.search(r"^op\s+(.+)$", out, re.MULTILINE)
    if m:
        op = m.group(1).strip()
    m = re.search(r"^tf_version\s+(.+)$", out, re.MULTILINE)
    if m:
        tf_ver = m.group(1).strip()
    m = re.search(r"^runs\s+(\S+)$", out, re.MULTILINE)
    if m:
        runs = float(m.group(1))
    m = re.search(r"^seconds\s+(\S+)$", out, re.MULTILINE)
    if m:
        secs = float(m.group(1))

    loaded = {}
    for line in out.splitlines():
        if not line.startswith("loaded "):
            continue
        parts = line.split(" ", 2)
        if len(parts) == 3:
            loaded[parts[1].strip()] = parts[2].strip()

    metric = f"op={op or 'matmul'} shape={mnk}x{mnk}x{mnk} dtype={dtype_out or dtype} wall={wall_s:.2f}s rocm_env={'in-tree' if use_in_tree else 'system'}"
    if tflops:
        metric += f" tflops_est={float(tflops):.2f}"
    if runs > 0 and secs > 0:
        metric += f" it/s={runs/secs:.2f}"
    if tf_ver:
        metric += f" tf={tf_ver}"
    if run_env.get("HSA_OVERRIDE_GFX_VERSION"):
        metric += f" hsa_override={run_env['HSA_OVERRIDE_GFX_VERSION']}"
    hip_lib = loaded.get("libamdhip64.so", "")
    if hip_lib:
        metric += f" hip_lib={hip_lib}"

    req = str(wl.get("require_rocm_prefix", "") or "").strip()
    if req:
        expected = str(rocm_dist) if req == "in-tree" else req.rstrip("/")
        if not hip_lib:
            return StepResult(build_dir, "TensorFlow matmul (GPU)", "FAIL", fmt_duration(r.dur_ms), f"could not determine loaded ROCm runtime lib (libamdhip64) | expected prefix: {expected} | {metric}")
        if not hip_lib.startswith(expected + "/"):
            return StepResult(build_dir, "TensorFlow matmul (GPU)", "FAIL", fmt_duration(r.dur_ms), f"ROCm runtime lib not from expected prefix: {expected} | hip_lib={hip_lib}")

    metric = append_power(metric, sampler, baseline_w=baseline_avg_w(cfg, build_dir))
    return StepResult(build_dir, "TensorFlow matmul (GPU)", "OK", fmt_duration(r.dur_ms), metric)
