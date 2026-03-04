from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from core.context import Context
from core.reporting.models import StepResult
from core.rocm_env import which
from core.runner import fmt_duration, run_cmd
from steps.shared import downloads_enabled


def _latest_wheel(wheel_dir: Path) -> Path | None:
    wheels = sorted(wheel_dir.glob("*.whl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return wheels[0] if wheels else None


def step_onnxruntime_rocm_wheel(
    ctx: Context,
    cfg: dict[str, Any],
    build_dir: str,
    rocm_dist: Path,
    env: dict[str, str],
    log: Path | None,
) -> StepResult:
    if not downloads_enabled(cfg):
        return StepResult(build_dir, "ONNX Runtime (ROCm) wheel build", "SKIP", "0ms", "downloads disabled")
    if which("bash", env) is None:
        return StepResult(build_dir, "ONNX Runtime (ROCm) wheel build", "SKIP", "0ms", "missing bash")
    if which("git", env) is None:
        return StepResult(build_dir, "ONNX Runtime (ROCm) wheel build", "SKIP", "0ms", "missing git")

    repo_root = ctx.repo_root
    script = repo_root / "validation" / "scripts" / "onnxruntime_rocm" / "build_onnxruntime_rocm_wheel.sh"
    if not script.is_file():
        return StepResult(build_dir, "ONNX Runtime (ROCm) wheel build", "FAIL", "0ms", f"missing script: {script}")

    wl = cfg.get("workloads", {}).get("onnxruntime", {}) or {}
    step_env = dict(env)
    step_env["PYTHON_BIN"] = str(sys.executable)
    step_env["ORT_REPO_URL"] = str(wl.get("repo_url", "https://github.com/microsoft/onnxruntime.git"))
    step_env["ORT_REF"] = str(wl.get("ref", "main"))
    step_env["PARALLEL"] = str(wl.get("jobs", os.cpu_count() or 1))
    if wl.get("rocm_path"):
        step_env["ROCM_PATH"] = str(wl.get("rocm_path"))
    if wl.get("rocm_version"):
        step_env["ROCM_VERSION"] = str(wl.get("rocm_version"))
    if wl.get("hip_arch"):
        step_env["HIP_ARCH"] = str(wl.get("hip_arch"))
    step_env["USE_MIGRAPHX"] = "1" if bool(wl.get("use_migraphx", False)) else "0"
    if wl.get("migraphx_home"):
        step_env["MIGRAPHX_HOME"] = str(wl.get("migraphx_home"))
    if wl.get("work_root"):
        step_env["WORK_ROOT"] = str(wl.get("work_root"))
    if wl.get("wheel_out_dir"):
        step_env["WHEEL_OUT_DIR"] = str(wl.get("wheel_out_dir"))
    step_env["DO_UPDATE"] = "1" if bool(wl.get("do_update", True)) else "0"

    timeout_s = int(cfg.get("timeouts_s", {}).get("onnxruntime_build", 21600))
    r = run_cmd(repo_root, step_env, ["bash", str(script)], timeout_s, log)
    if r.rc != 0:
        return StepResult(build_dir, "ONNX Runtime (ROCm) wheel build", "FAIL", fmt_duration(r.dur_ms), f"rc={r.rc}")

    wheel_out_dir = Path(
        str(
            wl.get(
                "wheel_out_dir",
                repo_root / "validation" / "workspace" / "cache" / "wheels" / "onnxruntime_rocm711",
            )
        )
    )
    wheel = _latest_wheel(wheel_out_dir)
    if wheel is None:
        return StepResult(
            build_dir,
            "ONNX Runtime (ROCm) wheel build",
            "FAIL",
            fmt_duration(r.dur_ms),
            f"build finished but no wheel in {wheel_out_dir}",
        )

    size_mb = wheel.stat().st_size / (1024 * 1024)
    metric = f"wheel={wheel.name} size={size_mb:.1f}MB"
    return StepResult(build_dir, "ONNX Runtime (ROCm) wheel build", "OK", fmt_duration(r.dur_ms), metric)
