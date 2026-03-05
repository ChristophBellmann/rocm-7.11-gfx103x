from __future__ import annotations

import os
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


def _resolve_repo_path(repo_root: Path, value: str | os.PathLike[str]) -> str:
    p = Path(value)
    if not p.is_absolute():
        p = repo_root / p
    return str(p)


def step_tensorflow_rocm_wheel(
    ctx: Context,
    cfg: dict[str, Any],
    build_dir: str,
    rocm_dist: Path,
    env: dict[str, str],
    log: Path | None,
) -> StepResult:
    if not downloads_enabled(cfg):
        return StepResult(
            build_dir,
            "TensorFlow (ROCm) wheel build",
            "SKIP",
            "0ms",
            "downloads disabled",
        )
    if which("bash", env) is None:
        return StepResult(build_dir, "TensorFlow (ROCm) wheel build", "SKIP", "0ms", "missing bash")
    if which("git", env) is None:
        return StepResult(build_dir, "TensorFlow (ROCm) wheel build", "SKIP", "0ms", "missing git")

    repo_root = ctx.repo_root
    script = repo_root / "validation" / "scripts" / "tensorflow_rocm" / "build_tensorflow_rocm_wheel.sh"
    if not script.is_file():
        return StepResult(
            build_dir,
            "TensorFlow (ROCm) wheel build",
            "FAIL",
            "0ms",
            f"missing script: {script}",
        )

    wl = cfg.get("workloads", {}).get("tensorflow", {}) or {}
    step_env = dict(env)
    step_env["TF_REPO_URL"] = str(wl.get("repo_url", "https://github.com/tensorflow/tensorflow.git"))
    step_env["TF_REF"] = str(wl.get("ref", "v2.20.0"))
    step_env["JOBS"] = str(wl.get("jobs", os.cpu_count() or 1))
    if wl.get("rocm_path"):
        step_env["ROCM_PATH"] = _resolve_repo_path(repo_root, str(wl.get("rocm_path")))
    if wl.get("work_root"):
        step_env["WORK_ROOT"] = _resolve_repo_path(repo_root, str(wl.get("work_root")))
    if wl.get("wheel_out_dir"):
        step_env["WHEEL_OUT_DIR"] = _resolve_repo_path(repo_root, str(wl.get("wheel_out_dir")))
    step_env["DO_UPDATE"] = "1" if bool(wl.get("do_update", True)) else "0"

    timeout_s = int(cfg.get("timeouts_s", {}).get("tensorflow_build", 43200))
    r = run_cmd(repo_root, step_env, ["bash", str(script)], timeout_s, log)
    if r.rc != 0:
        return StepResult(
            build_dir,
            "TensorFlow (ROCm) wheel build",
            "FAIL",
            fmt_duration(r.dur_ms),
            f"rc={r.rc}",
        )

    wheel_out_dir = Path(
        _resolve_repo_path(
            repo_root,
            str(
                wl.get(
                    "wheel_out_dir",
                    repo_root / "validation" / "workspace" / "cache" / "wheels" / "tensorflow_rocm_custom",
                )
            ),
        )
    )
    wheel = _latest_wheel(wheel_out_dir)
    if wheel is None:
        return StepResult(
            build_dir,
            "TensorFlow (ROCm) wheel build",
            "FAIL",
            fmt_duration(r.dur_ms),
            f"build finished but no wheel in {wheel_out_dir}",
        )

    size_mb = wheel.stat().st_size / (1024 * 1024)
    metric = f"wheel={wheel.name} size={size_mb:.1f}MB"
    return StepResult(build_dir, "TensorFlow (ROCm) wheel build", "OK", fmt_duration(r.dur_ms), metric)
