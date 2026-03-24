from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core.artifacts import write_report_json
from core.context import Context
from core.power import PowerSampler, discover_sensors, format_power_metrics, write_csv
from core.rocm_env import activated_env, deactivated_env, which
from core.tree import detect_build_dirs, detect_default_build_dir, rocm_dist_for_build
from core.reporting.models import StepResult
from core.runner import fmt_duration, run_cmd
from steps.workloads.llama_cpp.docker_env import step_llama_cpp_docker
from steps.workloads.mfem.hip_build import step_mfem_hip
from steps.workloads.ollama.functional import step_ollama
from steps.workloads.open_interpreter.functional import step_open_interpreter
from steps.workloads.petsc.hip_build import step_petsc_hip
from steps.workloads.pytorch.functional import step_pytorch_audio, step_pytorch_video
from steps.workloads.onnxruntime.rocm_wheel import step_onnxruntime_rocm_wheel
from steps.workloads.tensorflow.functional import step_tensorflow_matmul
from steps.workloads.tensorflow.rocm_wheel import step_tensorflow_rocm_wheel
from steps.workloads.whisper.setup import step_whisper


@dataclass(frozen=True)
class Step:
    group: str
    key: str
    name: str
    expected: str
    fn: Callable[[Context, dict[str, Any], str, Path, dict[str, str], Path | None], StepResult]


def _log_path(ctx: Context, build_dir: str, key: str) -> Path | None:
    if ctx.logs_dir is None:
        return None
    return ctx.logs_dir / f"{build_dir}.{key}.log"


def _power_enabled(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("run", {}).get("power_monitor", False))


def _power_csv_path(ctx: Context, build_dir: str, key: str) -> Path | None:
    if ctx.logs_dir is None:
        return None
    return ctx.logs_dir / f"{build_dir}.{key}.power.csv"


def _with_power_sampler(ctx: Context, cfg: dict[str, Any], build_dir: str, key: str, fn):
    if not _power_enabled(cfg):
        return fn(None)
    sensors = discover_sensors()
    if sensors is None:
        return fn(None)
    sampler = PowerSampler(sensors=sensors, interval_s=0.5)
    sampler.start()
    try:
        return fn(sampler)
    finally:
        sampler.stop()
        csvp = _power_csv_path(ctx, build_dir, key)
        if csvp is not None:
            write_csv(csvp, sampler.samples())


def _append_power(metric: str, sampler: PowerSampler | None, *, baseline_avg_w: float | None) -> str:
    if sampler is None:
        return metric
    pm = format_power_metrics(sampler, baseline_avg_w=baseline_avg_w)
    if not pm:
        return metric
    if metric:
        return f"{metric} | {pm}"
    return pm


def _baseline_cache(cfg: dict[str, Any]) -> dict[str, Any]:
    rt = cfg.setdefault("_runtime", {})
    return rt.setdefault("power_baseline", {})


def _get_baseline_avg_w(cfg: dict[str, Any], build_dir: str) -> float | None:
    b = _baseline_cache(cfg).get(build_dir) or {}
    v = b.get("avg_w")
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


def _set_baseline(cfg: dict[str, Any], build_dir: str, *, avg_w: float | None, peak_w: float | None, energy_ws: float | None, gpu: float | None, mem: float | None) -> None:
    _baseline_cache(cfg)[build_dir] = {"avg_w": avg_w, "peak_w": peak_w, "energy_ws": energy_ws, "gpu": gpu, "mem": mem}


def _step_power_idle_baseline(ctx: Context, cfg: dict[str, Any], build_dir: str, rocm_dist: Path, env: dict[str, str], log: Path | None) -> StepResult:
    if not _power_enabled(cfg):
        return StepResult(build_dir, "Power idle baseline", "SKIP", "0ms", "power monitor disabled")
    sensors = discover_sensors()
    if sensors is None:
        return StepResult(build_dir, "Power idle baseline", "SKIP", "0ms", "no amdgpu power sensor found in sysfs")
    sampler = PowerSampler(sensors=sensors, interval_s=0.5)
    sampler.start()
    try:
        # No GPU load: just sleep to measure baseline.
        time.sleep(5.0)
    finally:
        sampler.stop()
        csvp = _power_csv_path(ctx, build_dir, "power_idle_baseline")
        if csvp is not None:
            write_csv(csvp, sampler.samples())
    avg = sampler.avg_power_w()
    peak = sampler.peak_power_w()
    e = sampler.energy_ws()
    gpu = sampler.avg_gpu_busy()
    mem = sampler.avg_mem_busy()
    _set_baseline(cfg, build_dir, avg_w=avg, peak_w=peak, energy_ws=e, gpu=gpu, mem=mem)
    metric = format_power_metrics(sampler)
    warn = []
    if gpu is not None and gpu >= 10.0:
        warn.append(f"gpu%={gpu:.0f} (busy?)")
    if warn:
        metric = (metric + " WARN:" + ";".join(warn)).strip()
    return StepResult(build_dir, "Power idle baseline", "OK", "5.000s", metric)


def _step_rocm_env(ctx: Context, cfg: dict[str, Any], build_dir: str, rocm_dist: Path, env: dict[str, str], log: Path | None) -> StepResult:
    ok = (rocm_dist / "bin").is_dir() and (rocm_dist / "llvm" / "bin").is_dir()
    return StepResult(build_dir, "ROCm env activation", "OK" if ok else "FAIL", "0ms", f"ROCM_PATH={rocm_dist}")


def _step_rocminfo(ctx: Context, cfg: dict[str, Any], build_dir: str, rocm_dist: Path, env: dict[str, str], log: Path | None) -> StepResult:
    t = int(cfg.get("timeouts_s", {}).get("rocminfo", 10))
    if which("rocminfo", env) is None:
        return StepResult(build_dir, "rocminfo", "SKIP", "0ms", "rocminfo not in PATH")
    r = run_cmd(ctx.repo_root, env, ["rocminfo"], t, log)
    return StepResult(build_dir, "rocminfo", "OK" if r.rc == 0 else "FAIL", fmt_duration(r.dur_ms), "" if r.rc == 0 else f"rc={r.rc}")


def _step_hipcc_compile_run(ctx: Context, cfg: dict[str, Any], build_dir: str, rocm_dist: Path, env: dict[str, str], log: Path | None) -> StepResult:
    t = int(cfg.get("timeouts_s", {}).get("hipcc_compile_run", 120))
    hipcc = which("hipcc", env)
    if not hipcc:
        return StepResult(build_dir, "hipcc compile+run", "SKIP", "0ms", "hipcc not in PATH")
    # Stage-1 toolchain builds may have hipcc but not the ROCr runtime/libs needed to execute.
    if which("rocminfo", env) is None:
        return StepResult(build_dir, "hipcc compile+run", "SKIP", "0ms", "runtime not present (rocminfo missing)")

    arch = str(cfg.get("rocm", {}).get("amd_gpu_arch", "gfx1031"))
    with tempfile.TemporaryDirectory(prefix="rocm-validation-hip-") as td:
        td = Path(td)
        src = td / "vadd.cpp"
        exe = td / "vadd"
        src.write_text(
            r"""
#include <hip/hip_runtime.h>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <vector>

__global__ void vadd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  // Choose a size that is big enough to keep the GPU busy for a few seconds
  // (we want sustained load, not just microsecond kernels).
  int n = 1<<25; // 33,554,432 floats (~128MB per vector)
  int iters = 3000;
  size_t bytes = n * sizeof(float);
  std::vector<float> ha(n, 1.0f), hb(n, 2.0f), hc(n, 0.0f);
  float *da=nullptr, *db=nullptr, *dc=nullptr;
  if (hipMalloc(&da, bytes) != hipSuccess ||
      hipMalloc(&db, bytes) != hipSuccess ||
      hipMalloc(&dc, bytes) != hipSuccess) {
    std::fprintf(stderr, "hipMalloc failed (bytes=%zu)\n", bytes);
    return 2;
  }
  hipMemcpy(da, ha.data(), bytes, hipMemcpyHostToDevice);
  hipMemcpy(db, hb.data(), bytes, hipMemcpyHostToDevice);
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  hipDeviceSynchronize();
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; i++) {
    hipLaunchKernelGGL(vadd, dim3(blocks), dim3(threads), 0, 0, da, db, dc, n);
  }
  hipDeviceSynchronize();
  auto t1 = std::chrono::high_resolution_clock::now();
  hipMemcpy(hc.data(), dc, bytes, hipMemcpyDeviceToHost);
  hipFree(da); hipFree(db); hipFree(dc);
  for (int i = 0; i < 10; i++) {
    if (hc[i] != 3.0f) { std::printf("FAIL %d %f\n", i, hc[i]); return 1; }
  }
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  std::printf("OK iters=%d ms=%lld\n", iters, (long long)ms);
  return 0;
}
""".lstrip(),
            encoding="utf-8",
        )
        r1 = run_cmd(ctx.repo_root, env, [hipcc, f"--offload-arch={arch}", str(src), "-O2", "-o", str(exe)], t, log)
        if r1.rc != 0:
            return StepResult(build_dir, "hipcc compile+run", "FAIL", fmt_duration(r1.dur_ms), f"compile rc={r1.rc}")
        if not exe.exists():
            return StepResult(build_dir, "hipcc compile+run", "FAIL", fmt_duration(r1.dur_ms), "compile produced no output executable")
        def run_kernel(sampler: PowerSampler | None):
            r2 = run_cmd(ctx.repo_root, env, [str(exe)], 120, log)
            return r2, sampler

        r2, sampler = _with_power_sampler(ctx, cfg, build_dir, "hipcc_compile_run", run_kernel)
        ok = (r2.rc == 0) and ("OK" in (r2.out + r2.err))
        metric = "" if ok else f"run rc={r2.rc}"
        if ok:
            metric = _append_power(metric, sampler, baseline_avg_w=_get_baseline_avg_w(cfg, build_dir))
        return StepResult(build_dir, "hipcc compile+run", "OK" if ok else "FAIL", fmt_duration(r1.dur_ms + r2.dur_ms), metric)


def _extract_gflops(text: str) -> float | None:
    # Matches rocblas-bench CSV output tail: "<GFLOPS>, <ms>"
    import re

    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*,\s*[0-9]+(?:\.[0-9]+)?\s*$", text.strip(), re.M)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _latest_wheel(wheel_dir: Path) -> Path | None:
    wheels = sorted(wheel_dir.glob("*.whl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return wheels[0] if wheels else None


def _ort_runtime_env(env: dict[str, str], rocm_dist: Path, wl: dict[str, Any]) -> tuple[dict[str, str], bool, str]:
    use_in_tree = bool(wl.get("use_in_tree_rocm", True))
    require_prefix = str(wl.get("require_rocm_prefix", "") or "").strip()
    if use_in_tree:
        return env, True, str(rocm_dist)

    run_env = deactivated_env(env, rocm_dist)
    if require_prefix and require_prefix != "in-tree":
        prefix_path = Path(require_prefix)
        if prefix_path.is_absolute():
            run_env = activated_env(run_env, prefix_path)
            return run_env, False, str(prefix_path)
    return run_env, False, require_prefix or "system"


def _resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = repo_root / p
    return p


def _onnxruntime_tts_model_dir(ctx: Context) -> Path:
    return ctx.repo_root / "validation" / "workspace" / "cache" / "models" / "onnxruntime_tts"


def _ensure_onnxruntime_tts_staged_file(ctx: Context, value: str | Path, *, kind: str) -> Path:
    staged_dir = _onnxruntime_tts_model_dir(ctx)
    path = _resolve_repo_path(ctx.repo_root, value)
    if path.is_symlink():
        raise ValueError(f"{kind} must be copied into {staged_dir}, not symlinked: {path}")
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(staged_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"{kind} must live under {staged_dir}; external paths are not supported: {path}") from exc
    return path


def _staged_onnxruntime_tts_models(ctx: Context) -> list[Path]:
    model_dir = _onnxruntime_tts_model_dir(ctx)
    if not model_dir.is_dir():
        return []
    return sorted(
        (p for p in model_dir.glob("*.onnx") if p.is_file() and not p.is_symlink()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


_ORT_TTS_SHAPE_DIAG_OUTPUTS = [
    "output",
    "/dp/Split_output_0",
    "/Exp_output_0",
    "/Mul_output_0",
    "/Mul_1_output_0",
    "/Ceil_output_0",
    "/Cast_output_0",
    "/CumSum_output_0",
    "/Reshape_1_output_0",
    "/dp/flows.5/Expand_15_output_0",
    "/dp/flows.5/Reshape_16_output_0",
    "/dp/flows.7/Mul_10_output_0",
    "/dp/flows.7/Mul_16_output_0",
]


def _sanitize_artifact_name(text: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return clean.strip("._-") or "case"


def _cfg_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def _candidate_onnxruntime_tts_diag_pythons(ctx: Context, runtime_py: str) -> list[str]:
    candidates = [
        runtime_py,
        str(ctx.workspace_root / "envs" / "py" / "bin" / "python"),
    ]
    seen: set[str] = set()
    out: list[str] = []
    for cand in candidates:
        if cand and cand not in seen and Path(cand).is_file():
            seen.add(cand)
            out.append(cand)
    return out


def _python_supports_onnxruntime_tts_diag(ctx: Context, run_env: dict[str, str], python: str, log: Path | None) -> bool:
    probe = run_cmd(
        ctx.repo_root,
        run_env,
        [
            python,
            "-c",
            "import onnx, onnxruntime as ort; assert 'CPUExecutionProvider' in ort.get_available_providers()",
        ],
        60,
        log,
    )
    return probe.rc == 0


def _resolve_onnxruntime_tts_diag_python(
    ctx: Context,
    run_env: dict[str, str],
    runtime_py: str,
    log: Path | None,
) -> str | None:
    for cand in _candidate_onnxruntime_tts_diag_pythons(ctx, runtime_py):
        if _python_supports_onnxruntime_tts_diag(ctx, run_env, cand, log):
            return cand
    return None


def _shape_diag_scalar(item: dict[str, Any] | None, key: str) -> int | float | None:
    if not isinstance(item, dict):
        return None
    values = item.get(key)
    if not isinstance(values, list) or not values:
        return None
    value = values[0]
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    return None


def _shape_diag_last_dim(item: dict[str, Any] | None, key: str) -> int | None:
    if not isinstance(item, dict):
        return None
    shape = item.get(key)
    if not isinstance(shape, list) or not shape:
        return None
    last = shape[-1]
    return int(last) if isinstance(last, int) else None


def _fmt_shape_diag_number(value: int | float | None) -> str:
    if value is None:
        return "?"
    if isinstance(value, int):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.4g}"


def _summarize_onnxruntime_tts_exact_chain(data: dict[str, Any]) -> str:
    exact = data.get("exact_chain_repro")
    if not isinstance(exact, dict):
        return ""
    boundary_cmp = exact.get("full_boundary_compare") or {}
    mini_cmp = exact.get("mini_cpu_boundary_compare") or {}
    parts = []
    first_boundary = str(boundary_cmp.get("first_mismatch") or "").strip()
    if first_boundary:
        parts.append(f"boundary_first={first_boundary}")
    boundary_count = boundary_cmp.get("mismatch_count")
    if isinstance(boundary_count, int):
        parts.append(f"boundary_mismatch={boundary_count}")
    mini_count = mini_cmp.get("mismatch_count")
    if isinstance(mini_count, int):
        if mini_count == 0:
            parts.append("mini=green")
        else:
            first_mini = str(mini_cmp.get("first_mismatch") or "").strip()
            if first_mini:
                parts.append(f"mini_first={first_mini}")
            else:
                parts.append(f"mini_mismatch={mini_count}")
    return " ".join(parts)


def _summarize_onnxruntime_tts_shape_diag(data: dict[str, Any]) -> str:
    cmp = data.get("comparison") or {}
    mismatches = cmp.get("mismatches") or []
    by_name = {str(item.get("name")): item for item in mismatches if isinstance(item, dict)}
    output = by_name.get("output")
    split0 = by_name.get("/dp/Split_output_0")
    ceil0 = by_name.get("/Ceil_output_0")
    cast0 = by_name.get("/Cast_output_0")
    expand15 = by_name.get("/dp/flows.5/Expand_15_output_0")
    mul10 = by_name.get("/dp/flows.7/Mul_10_output_0")
    mul16 = by_name.get("/dp/flows.7/Mul_16_output_0")
    parts = []
    first_nan = str(cmp.get("first_rocm_nan") or "").strip()
    if first_nan:
        parts.append(f"first_nan={first_nan}")
    out_cpu = _shape_diag_last_dim(output, "cpu_shape")
    out_rocm = _shape_diag_last_dim(output, "rocm_shape")
    if out_cpu is not None or out_rocm is not None:
        parts.append(f"len={_fmt_shape_diag_number(out_cpu)}->{_fmt_shape_diag_number(out_rocm)}")
    cast_cpu = _shape_diag_scalar(cast0, "cpu_sample")
    cast_rocm = _shape_diag_scalar(cast0, "rocm_sample")
    if cast_cpu is not None or cast_rocm is not None:
        parts.append(f"cast={_fmt_shape_diag_number(cast_cpu)}->{_fmt_shape_diag_number(cast_rocm)}")
    split_cpu = _shape_diag_scalar(split0, "cpu_sample")
    split_rocm = _shape_diag_scalar(split0, "rocm_sample")
    if split_cpu is not None or split_rocm is not None:
        parts.append(f"split0={_fmt_shape_diag_number(split_cpu)}->{_fmt_shape_diag_number(split_rocm)}")
    ceil_cpu = _shape_diag_scalar(ceil0, "cpu_sample")
    ceil_rocm = _shape_diag_scalar(ceil0, "rocm_sample")
    if ceil_cpu is not None or ceil_rocm is not None:
        parts.append(f"ceil0={_fmt_shape_diag_number(ceil_cpu)}->{_fmt_shape_diag_number(ceil_rocm)}")
    expand_cpu = _shape_diag_scalar(expand15, "cpu_sample")
    expand_rocm = _shape_diag_scalar(expand15, "rocm_sample")
    if expand_cpu is not None or expand_rocm is not None:
        parts.append(f"flow5_expand15={_fmt_shape_diag_number(expand_cpu)}->{_fmt_shape_diag_number(expand_rocm)}")
    mul10_cpu = _shape_diag_scalar(mul10, "cpu_sample")
    mul10_rocm = _shape_diag_scalar(mul10, "rocm_sample")
    if mul10_cpu is not None or mul10_rocm is not None:
        parts.append(f"flow7_mul10={_fmt_shape_diag_number(mul10_cpu)}->{_fmt_shape_diag_number(mul10_rocm)}")
    mul16_cpu = _shape_diag_scalar(mul16, "cpu_sample")
    mul16_rocm = _shape_diag_scalar(mul16, "rocm_sample")
    if mul16_cpu is not None or mul16_rocm is not None:
        parts.append(f"flow7_mul16={_fmt_shape_diag_number(mul16_cpu)}->{_fmt_shape_diag_number(mul16_rocm)}")
    exact = _summarize_onnxruntime_tts_exact_chain(data)
    if exact:
        parts.append(exact)
    return " ".join(parts)


def _run_onnxruntime_tts_shape_diagnostics(
    ctx: Context,
    wl: dict[str, Any],
    run_env: dict[str, str],
    runtime_py: str,
    model: Path,
    case_file: Path,
    case_label: str,
    log: Path | None,
) -> str:
    diag_py = _resolve_onnxruntime_tts_diag_python(ctx, run_env, runtime_py, log)
    if diag_py is None:
        return "shape_diag=skip(no python with onnx+onnxruntime)"

    timeout_s = max(60, int(wl.get("tts_shape_diag_timeout_s", 240)))
    diag_script = ctx.repo_root / "validation" / "src" / "steps" / "workloads" / "onnxruntime" / "piper_tts_debug.py"
    slug = _sanitize_artifact_name(case_label)
    diag_dir = ctx.run_root / "artifacts"
    diag_dir.mkdir(parents=True, exist_ok=True)
    runs = [("current", False), ("nofast", True)]
    summaries: list[str] = []
    artifacts: list[str] = []
    for name, disable_fast in runs:
        out_json = diag_dir / f"onnxruntime_tts_shape_diag_{slug}_{name}.json"
        exact_dir = diag_dir / f"onnxruntime_tts_shape_diag_{slug}_{name}_exact_chain"
        cmd = [
            diag_py,
            str(diag_script),
            "--model",
            str(model),
            "--case-file",
            str(case_file),
            "--case-label",
            case_label,
            "--graph-mode",
            "full",
            "--opt-level",
            "disable",
            "--exact-chain-repro",
            "dp_shape_cast",
            "--artifact-dir",
            str(exact_dir),
            "--out-json",
            str(out_json),
        ]
        if disable_fast:
            cmd.append("--disable-fast-reduction")
        for output_name in _ORT_TTS_SHAPE_DIAG_OUTPUTS:
            cmd.extend(["--output", output_name])
        result = run_cmd(ctx.repo_root, run_env, cmd, timeout_s, log)
        if result.rc != 0:
            summaries.append(f"{name}=diag_rc{result.rc}")
            continue
        try:
            payload = json.loads(out_json.read_text(encoding="utf-8"))
        except Exception:
            summaries.append(f"{name}=diag_parse_error")
            continue
        summaries.append(f"{name}[{_summarize_onnxruntime_tts_shape_diag(payload)}]")
        artifacts.append(out_json.name)
        exact = payload.get("exact_chain_repro")
        if isinstance(exact, dict):
            report_json = str(exact.get("report_json") or "").strip()
            if report_json:
                report_path = Path(report_json)
                try:
                    rel = report_path.resolve().relative_to(diag_dir.resolve())
                    artifacts.append(str(rel))
                except Exception:
                    artifacts.append(report_path.name)
    if artifacts:
        return f"shape_diag {' '.join(summaries)} artifacts={','.join(artifacts)}"
    return f"shape_diag {' '.join(summaries)}"


def _onnxruntime_tts_case_file(ctx: Context, wl: dict[str, Any], model: Path) -> Path:
    case_file_cfg = str(wl.get("tts_case_file", "") or "").strip()
    if case_file_cfg:
        return _resolve_repo_path(ctx.repo_root, case_file_cfg)
    return ctx.repo_root / "validation" / "fixtures" / "onnxruntime_tts" / f"{model.stem}.real_cases.json"


def _summarize_onnxruntime_tts_flow_probe(data: dict[str, Any]) -> str:
    cmp = data.get("comparison") or {}
    exact = data.get("exact_chain_repro") or {}
    parts: list[str] = []
    preset = str((exact.get("preset") or data.get("preset") or "")).strip()
    if preset:
        parts.append(f"preset={preset}")
    if bool(data.get("freeze_dp_random_zeros", False)):
        parts.append("freeze=1")
    mismatch_count = cmp.get("mismatch_count")
    if isinstance(mismatch_count, int):
        parts.append(f"mismatch={mismatch_count}")
    first = str(cmp.get("first_mismatch") or "").strip()
    if first:
        parts.append(f"first={first}")
    first_nan = str(cmp.get("first_rocm_nan") or "").strip()
    if first_nan:
        parts.append(f"first_nan={first_nan}")
    exact_summary = _summarize_onnxruntime_tts_exact_chain(data)
    if exact_summary:
        parts.append(exact_summary)
    return " ".join(parts)


def _onnxruntime_runtime_python(
    ctx: Context,
    wl: dict[str, Any],
    env: dict[str, str],
    log: Path | None,
) -> tuple[str | None, str | None]:
    venv_dir = _resolve_repo_path(
        ctx.repo_root,
        str(
            wl.get(
                "runtime_venv_dir",
                ctx.repo_root / "validation" / "workspace" / "envs" / "onnxruntime_rocm",
            )
        ),
    )
    py = venv_dir / "bin" / "python"
    if py.exists():
        return str(py), None

    create = run_cmd(ctx.repo_root, env, [sys.executable, "-m", "venv", str(venv_dir)], 600, log)
    if create.rc != 0:
        return None, f"venv create rc={create.rc}"

    bootstrap = run_cmd(
        ctx.repo_root,
        env,
        [str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        1800,
        log,
    )
    if bootstrap.rc != 0:
        return None, f"venv bootstrap rc={bootstrap.rc}"

    return str(py), None


def _install_onnxruntime_runtime_wheel(
    ctx: Context,
    wl: dict[str, Any],
    env: dict[str, str],
    log: Path | None,
    py: str,
    wheel: Path,
) -> tuple[bool, Any, str]:
    install_cmd = [py, "-m", "pip", "install", "-q", "--force-reinstall", "numpy<2", "protobuf<5", str(wheel)]
    r_install = run_cmd(ctx.repo_root, env, install_cmd, 600, log)
    if r_install.rc == 0:
        return True, r_install, py

    # The shared ORT runtime venv can be left in a partially mutated state if a
    # previous pip operation was interrupted. Recreate it once and retry the
    # install instead of treating that stale state as a real stack failure.
    venv_dir = Path(py).resolve().parent.parent
    shutil.rmtree(venv_dir, ignore_errors=True)
    fresh_py, fresh_err = _onnxruntime_runtime_python(ctx, wl, env, log)
    if fresh_py is None:
        return False, r_install, fresh_err or "onnxruntime runtime venv recreate failed"
    r_retry = run_cmd(ctx.repo_root, env, [fresh_py, "-m", "pip", "install", "-q", "--force-reinstall", "numpy<2", "protobuf<5", str(wheel)], 600, log)
    if r_retry.rc != 0:
        return False, r_retry, fresh_py
    return True, r_retry, fresh_py


def _resolve_onnxruntime_infer_inputs(ctx: Context, cfg: dict[str, Any], step_name: str) -> tuple[dict[str, Any], Path, Path] | StepResult:
    wl = cfg.get("workloads", {}).get("onnxruntime", {}) or {}
    wheel_out_dir = Path(
        str(
            wl.get(
                "wheel_out_dir",
                ctx.repo_root / "validation" / "workspace" / "cache" / "wheels" / "onnxruntime_rocm711",
            )
        )
    )
    if not wheel_out_dir.is_absolute():
        wheel_out_dir = ctx.repo_root / wheel_out_dir
    wheel = _latest_wheel(wheel_out_dir)
    if wheel is None:
        return StepResult("", step_name, "FAIL", "0ms", f"no wheel in {wheel_out_dir}")

    model = Path(
        str(
            wl.get(
                "infer_model",
                "validation/workspace/builds/onnxruntime_rocm/winml/test/collateral/models/mnist.onnx",
            )
        )
    )
    if not model.is_absolute():
        model = ctx.repo_root / model
    if not model.is_file():
        return StepResult("", step_name, "FAIL", "0ms", f"missing model: {model}")
    return wl, wheel, model


def _resolve_onnxruntime_tts_inputs(ctx: Context, cfg: dict[str, Any], step_name: str) -> tuple[dict[str, Any], Path, Path, Path] | StepResult:
    wl = cfg.get("workloads", {}).get("onnxruntime", {}) or {}
    wheel_out_dir = Path(
        str(
            wl.get(
                "wheel_out_dir",
                ctx.repo_root / "validation" / "workspace" / "cache" / "wheels" / "onnxruntime_rocm711",
            )
        )
    )
    if not wheel_out_dir.is_absolute():
        wheel_out_dir = ctx.repo_root / wheel_out_dir
    wheel = _latest_wheel(wheel_out_dir)
    if wheel is None:
        return StepResult("", step_name, "FAIL", "0ms", f"no wheel in {wheel_out_dir}")

    model_cfg = str(wl.get("tts_model", "") or "").strip()
    if model_cfg:
        try:
            model = _ensure_onnxruntime_tts_staged_file(ctx, model_cfg, kind="Piper ONNX model")
        except ValueError as exc:
            return StepResult("", step_name, "SKIP", "0ms", str(exc))
    else:
        model_dir = _onnxruntime_tts_model_dir(ctx)
        models = _staged_onnxruntime_tts_models(ctx)
        if not models:
            symlinked = sorted(model_dir.glob("*.onnx"), key=lambda p: p.name) if model_dir.is_dir() else []
            first_symlink = next((p for p in symlinked if p.is_symlink()), None)
            if first_symlink is not None:
                return StepResult(
                    "",
                    step_name,
                    "SKIP",
                    "0ms",
                    f"no local staged Piper ONNX model in {model_dir}; replace symlink with copied file: {first_symlink.name}",
                )
            return StepResult("", step_name, "SKIP", "0ms", f"no staged Piper ONNX model in {model_dir}")
        model = models[0]
    if not model.is_file():
        return StepResult("", step_name, "SKIP", "0ms", f"missing Piper ONNX model: {model}")

    config_cfg = str(wl.get("tts_model_config", "") or "").strip()
    if config_cfg:
        try:
            config_path = _ensure_onnxruntime_tts_staged_file(ctx, config_cfg, kind="Piper config sidecar")
        except ValueError as exc:
            return StepResult("", step_name, "SKIP", "0ms", str(exc))
    else:
        config_path = model.with_suffix(model.suffix + ".json")
        try:
            config_path = _ensure_onnxruntime_tts_staged_file(ctx, config_path, kind="Piper config sidecar")
        except ValueError as exc:
            return StepResult("", step_name, "SKIP", "0ms", str(exc))
    if not config_path.is_file():
        return StepResult("", step_name, "SKIP", "0ms", f"missing Piper config sidecar: {config_path}")
    return wl, wheel, model, config_path


def _load_onnxruntime_tts_cases(ctx: Context, wl: dict[str, Any], model: Path, step_name: str) -> tuple[list[dict[str, Any]], Path] | StepResult:
    cases: list[dict[str, Any]] = []
    case_file_cfg = str(wl.get("tts_case_file", "") or "").strip()
    if case_file_cfg:
        case_file = Path(case_file_cfg)
        if not case_file.is_absolute():
            case_file = ctx.repo_root / case_file
    else:
        case_file = ctx.repo_root / "validation" / "fixtures" / "onnxruntime_tts" / f"{model.stem}.real_cases.json"
    if case_file.is_file():
        try:
            payload = json.loads(case_file.read_text(encoding="utf-8"))
            loaded_cases = payload.get("cases") or []
        except Exception as exc:
            return StepResult("", step_name, "FAIL", "0ms", f"invalid tts_case_file {case_file}: {exc}")
        for case in loaded_cases:
            try:
                ids = [int(v) for v in (case.get("ids") or [])]
                scale_list = [float(v) for v in (case.get("scales") or [])]
            except Exception as exc:
                return StepResult("", step_name, "FAIL", "0ms", f"invalid TTS case in {case_file}: {exc}")
            if not ids:
                return StepResult("", step_name, "FAIL", "0ms", f"invalid TTS case in {case_file}: empty ids")
            if len(scale_list) != 3:
                return StepResult("", step_name, "FAIL", "0ms", f"invalid TTS case in {case_file}: expected 3 floats in scales")
            cases.append(
                {
                    "label": str(case.get("label") or f"ids_len={len(ids)}"),
                    "ids": ids,
                    "scales": scale_list,
                }
            )
    else:
        try:
            phoneme_lengths = [max(1, int(v)) for v in (wl.get("tts_phoneme_lengths") or [32])]
        except Exception as exc:
            return StepResult("", step_name, "FAIL", "0ms", f"invalid tts_phoneme_lengths: {exc}")
        scales_cfg = wl.get("tts_scales") or [[0.667, 0.75, 0.8], [0.667, 1.0, 0.8], [0.667, 1.25, 0.8]]
        for phoneme_len in phoneme_lengths:
            for scales in scales_cfg:
                try:
                    scale_list = [float(v) for v in scales]
                except Exception as exc:
                    return StepResult("", step_name, "FAIL", "0ms", f"invalid tts_scales entry {scales!r}: {exc}")
                if len(scale_list) != 3:
                    return StepResult("", step_name, "FAIL", "0ms", f"invalid tts_scales entry {scales!r}: expected 3 floats")
                cases.append({"label": f"synthetic_len={phoneme_len}", "phoneme_len": phoneme_len, "scales": scale_list})
    if not cases:
        return StepResult("", step_name, "FAIL", "0ms", "no Piper TTS validation cases configured")
    return cases, case_file


def _step_onnxruntime_tts_flow_probe(
    ctx: Context,
    cfg: dict[str, Any],
    build_dir: str,
    rocm_dist: Path,
    env: dict[str, str],
    log: Path | None,
) -> StepResult:
    step_name = "ONNX Runtime Piper TTS Flow Probe (ROCm)"
    resolved = _resolve_onnxruntime_tts_inputs(ctx, cfg, step_name)
    if isinstance(resolved, StepResult):
        return StepResult(build_dir, resolved.name, resolved.status, resolved.duration, resolved.metric)
    wl, wheel, model, _config_path = resolved
    py, py_err = _onnxruntime_runtime_python(ctx, wl, env, log)
    if py is None:
        return StepResult(build_dir, step_name, "FAIL", "0ms", py_err or "onnxruntime runtime venv unavailable")
    run_env, use_in_tree, _runtime_prefix = _ort_runtime_env(env, rocm_dist, wl)

    install_ok, r_install, py_or_err = _install_onnxruntime_runtime_wheel(ctx, wl, run_env, log, py, wheel)
    if not install_ok:
        return StepResult(build_dir, step_name, "FAIL", fmt_duration(r_install.dur_ms), f"pip rc={r_install.rc}")
    py = py_or_err

    diag_py = _resolve_onnxruntime_tts_diag_python(ctx, run_env, py, log)
    if diag_py is None:
        return StepResult(build_dir, step_name, "FAIL", fmt_duration(r_install.dur_ms), "no python with onnx+onnxruntime for flow probe")

    case_file = _onnxruntime_tts_case_file(ctx, wl, model)
    if not case_file.is_file():
        return StepResult(build_dir, step_name, "FAIL", "0ms", f"missing real-case fixture for flow probe: {case_file}")

    case_label = str(wl.get("tts_flow_probe_case_label", "mogli") or "").strip()
    preset = str(wl.get("tts_flow_probe_preset", "dp_flow3_branch") or "").strip()
    graph_mode = str(wl.get("tts_flow_probe_graph_mode", "full") or "full").strip()
    opt_level = str(wl.get("tts_flow_probe_opt_level", "disable") or "disable").strip()
    outputs = _cfg_string_list(wl.get("tts_flow_probe_outputs"))
    node_prefixes = _cfg_string_list(wl.get("tts_flow_probe_node_prefixes"))
    node_names = _cfg_string_list(wl.get("tts_flow_probe_node_names"))
    rocm_provider_options = _cfg_string_list(wl.get("tts_flow_probe_rocm_provider_options"))
    freeze_dp_random_zeros = bool(wl.get("tts_flow_probe_freeze_dp_random_zeros", False))
    extract_per_output = bool(wl.get("tts_flow_probe_extract_per_output", False))
    run_nofast = bool(wl.get("tts_flow_probe_run_nofast", True))
    chunk_size = max(0, int(wl.get("tts_flow_probe_chunk_size", 0)))
    max_report = max(1, int(wl.get("tts_flow_probe_max_report", 30)))
    timeout_s = max(
        60,
        int(
            cfg.get("timeouts_s", {}).get(
                "onnxruntime_tts_flow_probe",
                wl.get("tts_flow_probe_timeout_s", 300),
            )
        ),
    )
    if not preset and not outputs and not node_prefixes and not node_names:
        return StepResult(build_dir, step_name, "FAIL", "0ms", "flow probe needs tts_flow_probe_preset or explicit outputs/node selectors")

    diag_dir = ctx.run_root / "artifacts"
    diag_dir.mkdir(parents=True, exist_ok=True)
    diag_script = ctx.repo_root / "validation" / "src" / "steps" / "workloads" / "onnxruntime" / "piper_tts_debug.py"
    slug = _sanitize_artifact_name(case_label or "case")
    runs = [("current", False)]
    if run_nofast:
        runs.append(("nofast", True))

    summaries: list[str] = []
    artifacts: list[str] = []
    total_dur_ms = r_install.dur_ms
    for name, disable_fast in runs:
        out_json = diag_dir / f"onnxruntime_tts_flow_probe_{slug}_{name}.json"
        exact_dir = diag_dir / f"onnxruntime_tts_flow_probe_{slug}_{name}_exact_chain"
        cmd = [
            diag_py,
            str(diag_script),
            "--model",
            str(model),
            "--case-file",
            str(case_file),
            "--graph-mode",
            graph_mode,
            "--opt-level",
            opt_level,
            "--chunk-size",
            str(chunk_size),
            "--max-report",
            str(max_report),
            "--out-json",
            str(out_json),
        ]
        if case_label:
            cmd.extend(["--case-label", case_label])
        if preset:
            cmd.extend(["--exact-chain-repro", preset, "--artifact-dir", str(exact_dir)])
        if disable_fast:
            cmd.append("--disable-fast-reduction")
        if freeze_dp_random_zeros:
            cmd.append("--freeze-dp-random-zeros")
        if extract_per_output:
            cmd.append("--extract-per-output")
        for output_name in outputs:
            cmd.extend(["--output", output_name])
        for node_prefix in node_prefixes:
            cmd.extend(["--node-prefix", node_prefix])
        for node_name in node_names:
            cmd.extend(["--node-name", node_name])
        for item in rocm_provider_options:
            cmd.extend(["--rocm-provider-option", item])
        result = run_cmd(ctx.repo_root, run_env, cmd, timeout_s, log)
        total_dur_ms += result.dur_ms
        if result.rc != 0:
            return StepResult(
                build_dir,
                step_name,
                "FAIL",
                fmt_duration(total_dur_ms),
                f"flow_probe {name} rc={result.rc}",
            )
        try:
            payload = json.loads(out_json.read_text(encoding="utf-8"))
        except Exception as exc:
            return StepResult(
                build_dir,
                step_name,
                "FAIL",
                fmt_duration(total_dur_ms),
                f"flow_probe {name} parse_error={exc}",
            )
        summaries.append(f"{name}[{_summarize_onnxruntime_tts_flow_probe(payload)}]")
        artifacts.append(out_json.name)
        exact = payload.get("exact_chain_repro")
        if isinstance(exact, dict):
            report_json = str(exact.get("report_json") or "").strip()
            if report_json:
                report_path = Path(report_json)
                try:
                    rel = report_path.resolve().relative_to(diag_dir.resolve())
                    artifacts.append(str(rel))
                except Exception:
                    artifacts.append(report_path.name)

    req = str(wl.get("require_rocm_prefix", "") or "").strip()
    if req:
        rocm_hint = ""
        for artifact in artifacts:
            if not artifact.endswith(".json"):
                continue
            try:
                payload = json.loads((diag_dir / artifact).read_text(encoding="utf-8"))
            except Exception:
                continue
            rocm_hint = str(payload.get("hip_lib") or "").strip()
            if rocm_hint:
                break
        expected = str(rocm_dist) if req == "in-tree" else req.rstrip("/")
        if rocm_hint and not rocm_hint.startswith(expected + "/"):
            return StepResult(
                build_dir,
                step_name,
                "FAIL",
                fmt_duration(total_dur_ms),
                f"ROCm runtime lib not from expected prefix: {expected}; hip_lib={rocm_hint}",
            )

    metric = " ".join(summaries)
    if artifacts:
        metric += f" artifacts={','.join(artifacts)}"
    metric += f" rocm_env={'in-tree' if use_in_tree else 'system'}"
    return StepResult(build_dir, step_name, "OK", fmt_duration(total_dur_ms), metric)


def _step_onnxruntime_infer(ctx: Context, cfg: dict[str, Any], build_dir: str, rocm_dist: Path, env: dict[str, str], log: Path | None) -> StepResult:
    step_name = "ONNX Runtime inference (ROCm)"
    resolved = _resolve_onnxruntime_infer_inputs(ctx, cfg, step_name)
    if isinstance(resolved, StepResult):
        return StepResult(build_dir, resolved.name, resolved.status, resolved.duration, resolved.metric)
    wl, wheel, model = resolved
    py, py_err = _onnxruntime_runtime_python(ctx, wl, env, log)
    if py is None:
        return StepResult(build_dir, step_name, "FAIL", "0ms", py_err or "onnxruntime runtime venv unavailable")
    run_env, use_in_tree, _runtime_prefix = _ort_runtime_env(env, rocm_dist, wl)

    # Keep ORT import ABI-stable in this venv for custom wheel tests.
    install_ok, r_install, py_or_err = _install_onnxruntime_runtime_wheel(ctx, wl, run_env, log, py, wheel)
    if not install_ok:
        return StepResult(
            build_dir,
            step_name,
            "FAIL",
            fmt_duration(r_install.dur_ms),
            f"pip rc={r_install.rc}",
        )
    py = py_or_err

    warmup = int(wl.get("infer_warmup", 50))
    iters = int(wl.get("infer_iters", 1500))
    timeout_s = int(cfg.get("timeouts_s", {}).get("onnxruntime_infer", 900))

    with tempfile.TemporaryDirectory(prefix="rocm-validation-ort-infer-") as td:
        tdp = Path(td)
        script = tdp / "ort_infer.py"
        profile_prefix = tdp / "onnxruntime_profile"
        script.write_text(
            (
                "import json\n"
                "import time\n"
                "import numpy as np\n"
                "import os\n"
                "import onnxruntime as ort\n"
                f"model = r'''{model}'''\n"
                f"profile_prefix = r'''{profile_prefix}'''\n"
                f"warmup = {warmup}\n"
                f"iters = {iters}\n"
                "providers = ['ROCMExecutionProvider', 'CPUExecutionProvider']\n"
                "so = ort.SessionOptions()\n"
                "so.enable_profiling = True\n"
                "so.profile_file_prefix = profile_prefix\n"
                "so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL\n"
                "sess = ort.InferenceSession(model, sess_options=so, providers=providers)\n"
                "sess_providers = sess.get_providers()\n"
                "if 'ROCMExecutionProvider' not in sess_providers:\n"
                "    raise RuntimeError(f'ROCMExecutionProvider missing from session providers: {sess_providers}')\n"
                "inp = sess.get_inputs()[0]\n"
                "shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]\n"
                "dtype = np.float32\n"
                "if inp.type == 'tensor(float16)': dtype = np.float16\n"
                "elif inp.type == 'tensor(double)': dtype = np.float64\n"
                "elif inp.type == 'tensor(int64)': dtype = np.int64\n"
                "elif inp.type == 'tensor(int32)': dtype = np.int32\n"
                "x = np.random.rand(*shape).astype(dtype) if np.issubdtype(dtype, np.floating) else np.random.randint(0, 10, size=shape, dtype=dtype)\n"
                "feed = {inp.name: x}\n"
                "for _ in range(warmup):\n"
                "    sess.run(None, feed)\n"
                "t0 = time.perf_counter()\n"
                "for _ in range(iters):\n"
                "    out = sess.run(None, feed)\n"
                "t1 = time.perf_counter()\n"
                "profile_path = sess.end_profiling()\n"
                "provider_events = 0\n"
                "with open(profile_path, 'r', encoding='utf-8') as f:\n"
                "    for ev in json.load(f):\n"
                "        p = (ev.get('args') or {}).get('provider')\n"
                "        if p == 'ROCMExecutionProvider':\n"
                "            provider_events += 1\n"
                "dt = t1 - t0\n"
                "res = {\n"
                "  'avg_ms': (dt * 1000.0) / iters,\n"
                "  'iters_per_s': iters / dt,\n"
                "  'iters': iters,\n"
                "  'provider_events_rocm': provider_events,\n"
                "  'session_providers': sess_providers,\n"
                "  'model': model,\n"
                "}\n"
                "try:\n"
                "  with open('/proc/self/maps', 'r', encoding='utf-8', errors='ignore') as f:\n"
                "    libs = sorted({line.strip().split()[5] for line in f if len(line.strip().split()) >= 6 and line.strip().split()[5].startswith('/') and '.so' in line.strip().split()[5]})\n"
                "  for p in libs:\n"
                "    b = os.path.basename(p)\n"
                "    if b == 'libamdhip64.so' or b.startswith('libamdhip64.so.'):\n"
                "      res['hip_lib'] = p\n"
                "      break\n"
                "except Exception:\n"
                "  pass\n"
                "print('ORT_RESULT_JSON=' + json.dumps(res, sort_keys=True))\n"
            ),
            encoding="utf-8",
        )

        def run_infer(sampler: PowerSampler | None):
            r = run_cmd(ctx.repo_root, run_env, [py, str(script)], timeout_s, log)
            return r, sampler

        r_infer, sampler = _with_power_sampler(ctx, cfg, build_dir, "onnxruntime_infer", run_infer)
        if r_infer.rc != 0:
            return StepResult(
                build_dir,
                step_name,
                "FAIL",
                fmt_duration(r_install.dur_ms + r_infer.dur_ms),
                f"infer rc={r_infer.rc}",
            )

    m = re.search(r"ORT_RESULT_JSON=(\{.*\})", r_infer.out + "\n" + r_infer.err)
    if not m:
        return StepResult(
            build_dir,
            step_name,
            "FAIL",
            fmt_duration(r_install.dur_ms + r_infer.dur_ms),
            "missing ORT_RESULT_JSON in output",
        )
    data = json.loads(m.group(1))
    if int(data.get("provider_events_rocm", 0)) <= 0:
        return StepResult(
            build_dir,
            step_name,
            "FAIL",
            fmt_duration(r_install.dur_ms + r_infer.dur_ms),
            "no ROCMExecutionProvider events in ORT profile",
        )

    metric = (
        f"iters={int(data.get('iters', iters))} "
        f"avg_ms={float(data.get('avg_ms', 0.0)):.4f} "
        f"iters_per_s={float(data.get('iters_per_s', 0.0)):.2f} "
        f"rocm_events={int(data.get('provider_events_rocm', 0))}"
    )
    hip_lib = str(data.get("hip_lib", "") or "").strip()
    if hip_lib:
        metric += f" hip_lib={hip_lib}"
    metric += f" rocm_env={'in-tree' if use_in_tree else 'system'}"
    req = str(wl.get("require_rocm_prefix", "") or "").strip()
    if req:
        expected = str(rocm_dist) if req == "in-tree" else req.rstrip("/")
        if not hip_lib:
            return StepResult(build_dir, step_name, "FAIL", fmt_duration(r_install.dur_ms + r_infer.dur_ms), f"could not determine loaded ROCm runtime lib (libamdhip64) | expected prefix: {expected} | {metric}")
        if not hip_lib.startswith(expected + "/"):
            return StepResult(build_dir, step_name, "FAIL", fmt_duration(r_install.dur_ms + r_infer.dur_ms), f"ROCm runtime lib not from expected prefix: {expected} | hip_lib={hip_lib}")
    metric = _append_power(metric, sampler, baseline_avg_w=_get_baseline_avg_w(cfg, build_dir))
    return StepResult(
        build_dir,
        step_name,
        "OK",
        fmt_duration(r_install.dur_ms + r_infer.dur_ms),
        metric,
    )


def _step_onnxruntime_tts_infer(ctx: Context, cfg: dict[str, Any], build_dir: str, rocm_dist: Path, env: dict[str, str], log: Path | None) -> StepResult:
    step_name = "ONNX Runtime Piper TTS (ROCm)"
    resolved = _resolve_onnxruntime_tts_inputs(ctx, cfg, step_name)
    if isinstance(resolved, StepResult):
        return StepResult(build_dir, resolved.name, resolved.status, resolved.duration, resolved.metric)
    wl, wheel, model, config_path = resolved
    py, py_err = _onnxruntime_runtime_python(ctx, wl, env, log)
    if py is None:
        return StepResult(build_dir, step_name, "FAIL", "0ms", py_err or "onnxruntime runtime venv unavailable")
    run_env, use_in_tree, _runtime_prefix = _ort_runtime_env(env, rocm_dist, wl)

    install_ok, r_install, py_or_err = _install_onnxruntime_runtime_wheel(ctx, wl, run_env, log, py, wheel)
    if not install_ok:
        return StepResult(build_dir, step_name, "FAIL", fmt_duration(r_install.dur_ms), f"pip rc={r_install.rc}")
    py = py_or_err

    loaded = _load_onnxruntime_tts_cases(ctx, wl, model, step_name)
    if isinstance(loaded, StepResult):
        return StepResult(build_dir, loaded.name, loaded.status, loaded.duration, loaded.metric)
    cases, case_file = loaded

    warmup = max(0, int(wl.get("tts_warmup", 1)))
    iters = max(1, int(wl.get("tts_iters", 4)))
    seed = int(wl.get("tts_seed", 0))
    require_cpu_reference = bool(wl.get("tts_require_cpu_reference", True))
    require_output_match = bool(wl.get("tts_require_output_match", True))
    compare_rtol = float(wl.get("tts_compare_rtol", 1.0e-3))
    compare_atol = float(wl.get("tts_compare_atol", 1.0e-5))
    freeze_randomnormal_like_zeros = bool(wl.get("tts_freeze_randomnormal_like_zeros", False))
    require_zero_miopen_warnings = bool(wl.get("tts_require_zero_miopen_workspace_warnings", False))
    timeout_s = int(cfg.get("timeouts_s", {}).get("onnxruntime_infer", 900))
    debug_dir = ctx.repo_root / "validation" / "workspace" / "debug" / "onnxruntime_tts_profiles"
    debug_dir.mkdir(parents=True, exist_ok=True)
    profile_prefix = debug_dir / f"{build_dir.replace('/', '_')}_onnxruntime_tts_profile"
    freeze_dur_ms = 0.0

    with tempfile.TemporaryDirectory(prefix="rocm-validation-ort-tts-") as td:
        tdp = Path(td)
        runnable_model = model
        if freeze_randomnormal_like_zeros:
            diag_py = _resolve_onnxruntime_tts_diag_python(ctx, run_env, py, log)
            if diag_py is None:
                return StepResult(build_dir, step_name, "FAIL", fmt_duration(r_install.dur_ms), "no python with onnx+onnxruntime to freeze Piper RandomNormalLike nodes")
            diag_script = ctx.repo_root / "validation" / "src" / "steps" / "workloads" / "onnxruntime" / "piper_tts_debug.py"
            runnable_model = tdp / f"{model.stem}.rng_frozen.onnx"
            freeze_cmd = [
                diag_py,
                str(diag_script),
                "--model",
                str(model),
                "--freeze-dp-random-zeros",
                "--write-frozen-model",
                str(runnable_model),
            ]
            r_freeze = run_cmd(ctx.repo_root, run_env, freeze_cmd, 300, log)
            freeze_dur_ms += r_freeze.dur_ms
            if r_freeze.rc != 0:
                return StepResult(build_dir, step_name, "FAIL", fmt_duration(r_install.dur_ms + freeze_dur_ms), f"freeze_randomnormal_like rc={r_freeze.rc}")
        script = tdp / "ort_tts_infer.py"
        script.write_text(
            (
                "import json\n"
                "import os\n"
                "import time\n"
                "from pathlib import Path\n"
                "import numpy as np\n"
                "import onnxruntime as ort\n"
                f"model = r'''{runnable_model}'''\n"
                f"config_path = r'''{config_path}'''\n"
                f"profile_prefix = r'''{profile_prefix}'''\n"
                f"warmup = {warmup}\n"
                f"iters = {iters}\n"
                f"seed = {seed}\n"
                f"cases = {json.dumps(cases, sort_keys=True)}\n"
                f"require_cpu_reference = {str(require_cpu_reference)}\n"
                "cfg = json.loads(Path(config_path).read_text(encoding='utf-8'))\n"
                "vals = sorted({int(v) for arr in (cfg.get('phoneme_id_map') or {}).values() for v in arr})\n"
                "vals = [v for v in vals if v >= 0]\n"
                "if not vals:\n"
                "    raise RuntimeError(f'no phoneme ids in Piper config: {config_path}')\n"
                "base_vals = [v for v in vals if v > 0] or vals\n"
                "def _dtype(type_name):\n"
                "    if type_name == 'tensor(int64)': return np.int64\n"
                "    if type_name == 'tensor(int32)': return np.int32\n"
                "    if type_name == 'tensor(float)': return np.float32\n"
                "    raise RuntimeError(f'unsupported Piper input dtype: {type_name}')\n"
                "def build_feed(sess, case):\n"
                "    token_ids = case.get('ids')\n"
                "    if token_ids is None:\n"
                "        length = int(case['phoneme_len'])\n"
                "        reps = (length + len(base_vals) - 1) // len(base_vals)\n"
                "        token_ids = (base_vals * reps)[:length]\n"
                "    else:\n"
                "        token_ids = [int(v) for v in token_ids]\n"
                "        length = len(token_ids)\n"
                "    seq = np.asarray([token_ids], dtype=np.int64)\n"
                "    lens = np.asarray([length], dtype=np.int64)\n"
                "    scales = np.asarray(case['scales'], dtype=np.float32)\n"
                "    feed = {}\n"
                "    for inp in sess.get_inputs():\n"
                "        dt = _dtype(inp.type)\n"
                "        if inp.name == 'input':\n"
                "            feed[inp.name] = seq.astype(dt, copy=False)\n"
                "        elif inp.name == 'input_lengths':\n"
                "            feed[inp.name] = lens.astype(dt, copy=False)\n"
                "        elif inp.name == 'scales':\n"
                "            feed[inp.name] = scales.astype(dt, copy=False)\n"
                "        elif inp.name in {'sid', 'speaker_id'}:\n"
                "            feed[inp.name] = np.asarray([0], dtype=dt)\n"
                "        else:\n"
                "            raise RuntimeError(f'unsupported Piper input name: {inp.name}')\n"
                "    return feed\n"
                "def provider_event_count(profile_path, provider_name):\n"
                "    with open(profile_path, 'r', encoding='utf-8') as f:\n"
                "        return sum(1 for ev in json.load(f) if ((ev.get('args') or {}).get('provider') == provider_name))\n"
                "def rocm_lib_hint():\n"
                "    try:\n"
                "        with open('/proc/self/maps', 'r', encoding='utf-8', errors='ignore') as f:\n"
                "            libs = sorted({line.strip().split()[5] for line in f if len(line.strip().split()) >= 6 and line.strip().split()[5].startswith('/') and '.so' in line.strip().split()[5]})\n"
                "        preferred = ('libamdhip64.so', 'libMIOpen.so', 'librocblas.so')\n"
                "        for needle in preferred:\n"
                "            for p in libs:\n"
                "                base = os.path.basename(p)\n"
                "                if base == needle or base.startswith(needle + '.'):\n"
                "                    return p\n"
                "    except Exception:\n"
                "        return ''\n"
                "    return ''\n"
                "def run_cases(providers, *, enable_profiling, profile_name):\n"
                "    ort.set_seed(seed)\n"
                "    np.random.seed(seed)\n"
                "    so = ort.SessionOptions()\n"
                "    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL\n"
                "    if enable_profiling:\n"
                "        so.enable_profiling = True\n"
                "        so.profile_file_prefix = profile_name\n"
                "    sess = ort.InferenceSession(model, sess_options=so, providers=providers)\n"
                "    out = {'session_providers': sess.get_providers(), 'case_results': [], 'failures': []}\n"
                "    raw_outputs = {}\n"
                "    t0 = time.perf_counter()\n"
                "    for case in cases:\n"
                "        feed = build_feed(sess, case)\n"
                "        label = str(case.get('label') or f\"len={int(feed['input_lengths'][0])}\")\n"
                "        try:\n"
                "            for _ in range(warmup):\n"
                "                sess.run(None, feed)\n"
                "            last = None\n"
                "            for _ in range(iters):\n"
                "                last = sess.run(None, feed)\n"
                "            arr = np.asarray(last[0], dtype=np.float32) if last else np.asarray([], dtype=np.float32)\n"
                "            shape = list(arr.shape)\n"
                "            out['case_results'].append({'label': label, 'phoneme_len': int(feed['input_lengths'][0]), 'scales': case['scales'], 'output_shape': shape, 'sample': arr.reshape(-1)[:16].tolist(), 'abs_mean': float(np.mean(np.abs(arr))) if arr.size else 0.0, 'mean': float(np.mean(arr)) if arr.size else 0.0, 'std': float(np.std(arr)) if arr.size else 0.0})\n"
                "            raw_outputs[label] = arr\n"
                "        except Exception as exc:\n"
                "            out['failures'].append({'label': label, 'phoneme_len': int(feed['input_lengths'][0]), 'scales': case['scales'], 'error_type': type(exc).__name__, 'error': str(exc)})\n"
                "    dt = time.perf_counter() - t0\n"
                "    out['ok_cases'] = len(out['case_results'])\n"
                "    out['failed_cases'] = len(out['failures'])\n"
                "    out['case_count'] = len(cases)\n"
                "    out['total_case_runs'] = len(cases) * max(iters, 1)\n"
                "    out['avg_case_ms'] = (dt * 1000.0) / max(len(cases), 1)\n"
                "    if enable_profiling:\n"
                "        profile_path = sess.end_profiling()\n"
                "        out['profile_path'] = profile_path\n"
                "        out['provider_events_rocm'] = provider_event_count(profile_path, 'ROCMExecutionProvider')\n"
                "    return out, raw_outputs\n"
                "def compare_case_outputs(cpu_out, cpu_raw, rocm_out, rocm_raw, *, rtol, atol):\n"
                "    comparisons = []\n"
                "    by_label = {str(item.get('label')): item for item in (cpu_out.get('case_results') or [])}\n"
                "    for rocm_case in (rocm_out.get('case_results') or []):\n"
                "        label = str(rocm_case.get('label'))\n"
                "        cpu_case = by_label.get(label)\n"
                "        cpu_arr = cpu_raw.get(label)\n"
                "        rocm_arr = rocm_raw.get(label)\n"
                "        if cpu_case is None or cpu_arr is None or rocm_arr is None:\n"
                "            comparisons.append({'label': label, 'ok': False, 'shape_equal': False, 'allclose': False, 'reason': 'missing_cpu_or_rocm_case'})\n"
                "            continue\n"
                "        shape_equal = list(cpu_arr.shape) == list(rocm_arr.shape)\n"
                "        cmp = {'label': label, 'shape_equal': shape_equal, 'cpu_shape': list(cpu_arr.shape), 'rocm_shape': list(rocm_arr.shape)}\n"
                "        if not shape_equal:\n"
                "            cmp.update({'allclose': False, 'ok': False, 'reason': 'shape_mismatch'})\n"
                "            comparisons.append(cmp)\n"
                "            continue\n"
                "        diff = np.abs(cpu_arr - rocm_arr)\n"
                "        cmp.update({'max_abs_diff': float(np.max(diff)) if diff.size else 0.0, 'mean_abs_diff': float(np.mean(diff)) if diff.size else 0.0, 'allclose': bool(np.allclose(cpu_arr, rocm_arr, rtol=rtol, atol=atol)), 'ok': bool(np.allclose(cpu_arr, rocm_arr, rtol=rtol, atol=atol)), 'reason': 'value_mismatch' if not bool(np.allclose(cpu_arr, rocm_arr, rtol=rtol, atol=atol)) else ''})\n"
                "        comparisons.append(cmp)\n"
                "    return comparisons\n"
                "result = {\n"
                "    'model': model,\n"
                "    'model_config': config_path,\n"
                "}\n"
                "if require_cpu_reference:\n"
                f"    result['cpu'], cpu_raw = run_cases(['CPUExecutionProvider'], enable_profiling=False, profile_name='')\n"
                "else:\n"
                "    cpu_raw = {}\n"
                f"result['rocm'], rocm_raw = run_cases(['ROCMExecutionProvider', 'CPUExecutionProvider'], enable_profiling=True, profile_name=profile_prefix)\n"
                "if require_cpu_reference:\n"
                f"    result['compare'] = compare_case_outputs(result['cpu'], cpu_raw, result['rocm'], rocm_raw, rtol={compare_rtol}, atol={compare_atol})\n"
                "result['hip_lib'] = rocm_lib_hint()\n"
                "print('ORT_TTS_RESULT_JSON=' + json.dumps(result, sort_keys=True))\n"
            ),
            encoding="utf-8",
        )

        def run_tts(sampler: PowerSampler | None):
            r = run_cmd(ctx.repo_root, run_env, [py, str(script)], timeout_s, log)
            return r, sampler

        r_tts, sampler = _with_power_sampler(ctx, cfg, build_dir, "onnxruntime_tts_infer", run_tts)
        total_tts_ms = r_install.dur_ms + freeze_dur_ms + r_tts.dur_ms
        if r_tts.rc != 0:
            return StepResult(build_dir, step_name, "FAIL", fmt_duration(total_tts_ms), f"tts rc={r_tts.rc}")

    combined = r_tts.out + "\n" + r_tts.err
    m = re.search(r"ORT_TTS_RESULT_JSON=(\{.*\})", combined)
    if not m:
        return StepResult(build_dir, step_name, "FAIL", fmt_duration(total_tts_ms), "missing ORT_TTS_RESULT_JSON in output")
    data = json.loads(m.group(1))
    miopen_warning_count = len(re.findall(r"MIOpen\(HIP\): Warning \[IsEnoughWorkspace\]", combined))

    cpu_data = data.get("cpu") if require_cpu_reference else None
    if require_cpu_reference:
        if not isinstance(cpu_data, dict):
            return StepResult(build_dir, step_name, "FAIL", fmt_duration(total_tts_ms), "missing CPU reference data")
        if int(cpu_data.get("failed_cases", 0)) > 0:
            first = (cpu_data.get("failures") or [{}])[0]
            return StepResult(
                build_dir,
                step_name,
                "FAIL",
                fmt_duration(total_tts_ms),
                f"CPU reference failed for {int(cpu_data.get('failed_cases', 0))}/{int(cpu_data.get('case_count', 0))} cases; first={first.get('label') or first.get('phoneme_len')}:{first.get('scales')} {first.get('error_type')} {first.get('error')}",
            )

    rocm_data = data.get("rocm") or {}
    if int(rocm_data.get("provider_events_rocm", 0)) <= 0:
        return StepResult(build_dir, step_name, "FAIL", fmt_duration(total_tts_ms), "no ROCMExecutionProvider events in Piper TTS ORT profile")

    compare_data = data.get("compare") if require_cpu_reference else None
    if require_cpu_reference and require_output_match:
        if not isinstance(compare_data, list):
            return StepResult(build_dir, step_name, "FAIL", fmt_duration(total_tts_ms), "missing CPU-vs-ROCm compare data")
        bad = [item for item in compare_data if not bool(item.get("ok"))]
        if bad:
            first = bad[0]
            detail = (
                f"ROCm Piper TTS semantic mismatch for {len(bad)}/{len(compare_data)} cases; "
                f"first={first.get('label')} reason={first.get('reason')} "
                f"shape_equal={first.get('shape_equal')} cpu_shape={first.get('cpu_shape')} rocm_shape={first.get('rocm_shape')}"
            )
            if "max_abs_diff" in first:
                detail += f" max_abs_diff={float(first.get('max_abs_diff', 0.0)):.6g} mean_abs_diff={float(first.get('mean_abs_diff', 0.0)):.6g}"
            detail += f" rtol={compare_rtol:.3g} atol={compare_atol:.3g}"
            if bool(wl.get("tts_shape_diagnose_on_shape_mismatch", True)) and str(first.get("reason") or "") == "shape_mismatch" and case_file.is_file():
                case_label = str(first.get("label") or "").strip()
                if case_label:
                    diag = _run_onnxruntime_tts_shape_diagnostics(
                        ctx,
                        wl,
                        run_env,
                        py,
                        model,
                        case_file,
                        case_label,
                        log,
                    )
                    if diag:
                        detail += f"; {diag}"
            return StepResult(build_dir, step_name, "FAIL", fmt_duration(total_tts_ms), detail)

    hip_lib = str(data.get("hip_lib", "") or "").strip()
    profile_path = str(rocm_data.get("profile_path", "") or "").strip()
    if int(rocm_data.get("failed_cases", 0)) > 0:
        first = (rocm_data.get("failures") or [{}])[0]
        detail = (
            f"ROCm Piper TTS failed for {int(rocm_data.get('failed_cases', 0))}/{int(rocm_data.get('case_count', 0))} cases; "
            f"first={first.get('label') or first.get('phoneme_len')}:{first.get('scales')} {first.get('error_type')} {first.get('error')}; "
            f"rocm_events={int(rocm_data.get('provider_events_rocm', 0))}; miopen_warn={miopen_warning_count}"
        )
        if profile_path:
            detail += f"; profile={profile_path}"
        return StepResult(
            build_dir,
            step_name,
            "FAIL",
            fmt_duration(total_tts_ms),
            detail,
        )

    if require_zero_miopen_warnings and miopen_warning_count > 0:
        return StepResult(
            build_dir,
            step_name,
            "FAIL",
            fmt_duration(total_tts_ms),
            f"MIOpen emitted workspace warnings: {miopen_warning_count}",
        )

    req = str(wl.get("require_rocm_prefix", "") or "").strip()
    if req:
        expected = str(rocm_dist) if req == "in-tree" else req.rstrip("/")
        if not hip_lib:
            return StepResult(build_dir, step_name, "FAIL", fmt_duration(total_tts_ms), f"could not determine loaded ROCm runtime lib (libamdhip64/libMIOpen/librocblas); expected prefix: {expected}")
        if not hip_lib.startswith(expected + "/"):
            return StepResult(build_dir, step_name, "FAIL", fmt_duration(total_tts_ms), f"ROCm runtime lib not from expected prefix: {expected}; hip_lib={hip_lib}")

    metric = (
        f"cases={int(rocm_data.get('case_count', 0))} "
        f"rocm_ok={int(rocm_data.get('ok_cases', 0))}/{int(rocm_data.get('case_count', 0))} "
        f"avg_case_ms={float(rocm_data.get('avg_case_ms', 0.0)):.2f} "
        f"rocm_events={int(rocm_data.get('provider_events_rocm', 0))} "
        f"miopen_warn={miopen_warning_count}"
    )
    if require_cpu_reference and isinstance(cpu_data, dict):
        metric += f" cpu_ok={int(cpu_data.get('ok_cases', 0))}/{int(cpu_data.get('case_count', 0))}"
    if require_cpu_reference and isinstance(compare_data, list):
        compare_ok = sum(1 for item in compare_data if bool(item.get("ok")))
        metric += f" compare_ok={compare_ok}/{len(compare_data)}"
    if hip_lib:
        metric += f" hip_lib={hip_lib}"
    if profile_path:
        metric += f" profile={profile_path}"
    metric += f" rocm_env={'in-tree' if use_in_tree else 'system'}"
    metric = _append_power(metric, sampler, baseline_avg_w=_get_baseline_avg_w(cfg, build_dir))
    return StepResult(build_dir, step_name, "OK", fmt_duration(total_tts_ms), metric)


def _step_onnxruntime_tts_benchmark(
    ctx: Context,
    cfg: dict[str, Any],
    build_dir: str,
    rocm_dist: Path,
    env: dict[str, str],
    log: Path | None,
) -> StepResult:
    step_name = "ONNX Runtime Piper TTS Benchmark"
    resolved = _resolve_onnxruntime_tts_inputs(ctx, cfg, step_name)
    if isinstance(resolved, StepResult):
        return StepResult(build_dir, resolved.name, resolved.status, resolved.duration, resolved.metric)
    wl, wheel, model, config_path = resolved
    py, py_err = _onnxruntime_runtime_python(ctx, wl, env, log)
    if py is None:
        return StepResult(build_dir, step_name, "FAIL", "0ms", py_err or "onnxruntime runtime venv unavailable")
    run_env, use_in_tree, _runtime_prefix = _ort_runtime_env(env, rocm_dist, wl)

    install_ok, r_install, py_or_err = _install_onnxruntime_runtime_wheel(ctx, wl, run_env, log, py, wheel)
    if not install_ok:
        return StepResult(build_dir, step_name, "FAIL", fmt_duration(r_install.dur_ms), f"pip rc={r_install.rc}")
    py = py_or_err

    loaded = _load_onnxruntime_tts_cases(ctx, wl, model, step_name)
    if isinstance(loaded, StepResult):
        return StepResult(build_dir, loaded.name, loaded.status, loaded.duration, loaded.metric)
    cases, _case_file = loaded

    warmup = max(0, int(wl.get("tts_bench_warmup", 2)))
    iters = max(1, int(wl.get("tts_bench_iters", 20)))
    seed = int(wl.get("tts_seed", 0))
    freeze_randomnormal_like_zeros = bool(wl.get("tts_freeze_randomnormal_like_zeros", False))
    rocm_provider_options = {"miopen_conv_use_max_workspace": "1"}
    bench_env = dict(run_env)
    for item in _cfg_string_list(wl.get("tts_bench_rocm_provider_options")):
        if "=" not in item:
            return StepResult(build_dir, step_name, "FAIL", "0ms", f"invalid tts_bench_rocm_provider_options entry: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            return StepResult(build_dir, step_name, "FAIL", "0ms", f"invalid tts_bench_rocm_provider_options entry: {item!r}")
        rocm_provider_options[key] = value
    for item in _cfg_string_list(wl.get("tts_bench_env")):
        if "=" not in item:
            return StepResult(build_dir, step_name, "FAIL", "0ms", f"invalid tts_bench_env entry: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            return StepResult(build_dir, step_name, "FAIL", "0ms", f"invalid tts_bench_env entry: {item!r}")
        bench_env[key] = value
    timeout_s = max(60, int(cfg.get("timeouts_s", {}).get("onnxruntime_infer", wl.get("tts_bench_timeout_s", 900))))

    artifact_dir = ctx.run_root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "onnxruntime_tts_benchmark.json"
    freeze_dur_ms = 0.0

    with tempfile.TemporaryDirectory(prefix="rocm-validation-ort-tts-bench-") as td:
        tdp = Path(td)
        runnable_model = model
        if freeze_randomnormal_like_zeros:
            diag_py = _resolve_onnxruntime_tts_diag_python(ctx, run_env, py, log)
            if diag_py is None:
                return StepResult(build_dir, step_name, "FAIL", fmt_duration(r_install.dur_ms), "no python with onnx+onnxruntime to freeze Piper RandomNormalLike nodes")
            diag_script = ctx.repo_root / "validation" / "src" / "steps" / "workloads" / "onnxruntime" / "piper_tts_debug.py"
            runnable_model = tdp / f"{model.stem}.rng_frozen.onnx"
            freeze_cmd = [
                diag_py,
                str(diag_script),
                "--model",
                str(model),
                "--freeze-dp-random-zeros",
                "--write-frozen-model",
                str(runnable_model),
            ]
            r_freeze = run_cmd(ctx.repo_root, bench_env, freeze_cmd, 300, log)
            freeze_dur_ms += r_freeze.dur_ms
            if r_freeze.rc != 0:
                return StepResult(build_dir, step_name, "FAIL", fmt_duration(r_install.dur_ms + freeze_dur_ms), f"freeze_randomnormal_like rc={r_freeze.rc}")

        child_script = tdp / "ort_tts_bench_child.py"
        child_script.write_text(
            (
                "import json\n"
                "import sys\n"
                "import os\n"
                "import time\n"
                "from pathlib import Path\n"
                "import numpy as np\n"
                "import onnxruntime as ort\n"
                f"model = r'''{runnable_model}'''\n"
                f"config_path = r'''{config_path}'''\n"
                f"seed = {seed}\n"
                f"cases = {json.dumps(cases, sort_keys=True)}\n"
                f"rocm_provider_options = {json.dumps(rocm_provider_options, sort_keys=True)}\n"
                f"bench_env = {json.dumps({k: bench_env[k] for k in sorted(bench_env) if k not in run_env or run_env.get(k) != bench_env.get(k)}, sort_keys=True)}\n"
                "provider_name = sys.argv[1]\n"
                "case_index = int(sys.argv[2])\n"
                "mode = sys.argv[3]\n"
                "cfg = json.loads(Path(config_path).read_text(encoding='utf-8'))\n"
                "vals = sorted({int(v) for arr in (cfg.get('phoneme_id_map') or {}).values() for v in arr})\n"
                "vals = [v for v in vals if v >= 0]\n"
                "if not vals:\n"
                "    raise RuntimeError(f'no phoneme ids in Piper config: {config_path}')\n"
                "base_vals = [v for v in vals if v > 0] or vals\n"
                "def _dtype(type_name):\n"
                "    if type_name == 'tensor(int64)': return np.int64\n"
                "    if type_name == 'tensor(int32)': return np.int32\n"
                "    if type_name == 'tensor(float)': return np.float32\n"
                "    raise RuntimeError(f'unsupported Piper input dtype: {type_name}')\n"
                "def build_feed(sess, case):\n"
                "    token_ids = case.get('ids')\n"
                "    if token_ids is None:\n"
                "        length = int(case['phoneme_len'])\n"
                "        reps = (length + len(base_vals) - 1) // len(base_vals)\n"
                "        token_ids = (base_vals * reps)[:length]\n"
                "    else:\n"
                "        token_ids = [int(v) for v in token_ids]\n"
                "        length = len(token_ids)\n"
                "    seq = np.asarray([token_ids], dtype=np.int64)\n"
                "    lens = np.asarray([length], dtype=np.int64)\n"
                "    scales = np.asarray(case['scales'], dtype=np.float32)\n"
                "    feed = {}\n"
                "    for inp in sess.get_inputs():\n"
                "        dt = _dtype(inp.type)\n"
                "        if inp.name == 'input':\n"
                "            feed[inp.name] = seq.astype(dt, copy=False)\n"
                "        elif inp.name == 'input_lengths':\n"
                "            feed[inp.name] = lens.astype(dt, copy=False)\n"
                "        elif inp.name == 'scales':\n"
                "            feed[inp.name] = scales.astype(dt, copy=False)\n"
                "        elif inp.name in {'sid', 'speaker_id'}:\n"
                "            feed[inp.name] = np.asarray([0], dtype=dt)\n"
                "        else:\n"
                "            raise RuntimeError(f'unsupported Piper input name: {inp.name}')\n"
                "    return feed\n"
                "def rocm_lib_hint():\n"
                "    try:\n"
                "        with open('/proc/self/maps', 'r', encoding='utf-8', errors='ignore') as f:\n"
                "            libs = sorted({line.strip().split()[5] for line in f if len(line.strip().split()) >= 6 and line.strip().split()[5].startswith('/') and '.so' in line.strip().split()[5]})\n"
                "        preferred = ('libamdhip64.so', 'libMIOpen.so', 'librocblas.so')\n"
                "        for needle in preferred:\n"
                "            for p in libs:\n"
                "                base = os.path.basename(p)\n"
                "                if base == needle or base.startswith(needle + '.'):\n"
                "                    return p\n"
                "    except Exception:\n"
                "        return ''\n"
                "    return ''\n"
                "def bench_case(providers, case):\n"
                "    ort.set_seed(seed)\n"
                "    np.random.seed(seed)\n"
                "    so = ort.SessionOptions()\n"
                "    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL\n"
                "    t0 = time.perf_counter()\n"
                "    sess = ort.InferenceSession(model, sess_options=so, providers=providers)\n"
                "    create_ms = (time.perf_counter() - t0) * 1000.0\n"
                "    feed = build_feed(sess, case)\n"
                "    label = str(case.get('label') or f\"len={int(feed['input_lengths'][0])}\")\n"
                "    result = {\n"
                "        'ok': True,\n"
                "        'provider_name': provider_name,\n"
                "        'mode': mode,\n"
                "        'bench_env': bench_env,\n"
                "        'label': label,\n"
                "        'phoneme_len': int(feed['input_lengths'][0]),\n"
                "        'session_create_ms': create_ms,\n"
                "        'session_providers': sess.get_providers(),\n"
                "        'hip_lib': rocm_lib_hint(),\n"
                "    }\n"
                "    try:\n"
                "        if mode == 'cold':\n"
                "            t1 = time.perf_counter()\n"
                "            last = sess.run(None, feed)\n"
                "            result['run_ms'] = (time.perf_counter() - t1) * 1000.0\n"
                "        elif mode == 'reuse':\n"
                "            sess.run(None, feed)\n"
                "            t1 = time.perf_counter()\n"
                "            last = sess.run(None, feed)\n"
                "            result['run_ms'] = (time.perf_counter() - t1) * 1000.0\n"
                "        else:\n"
                "            raise RuntimeError(f'unsupported mode: {mode}')\n"
                "        arr = np.asarray(last[0], dtype=np.float32) if last else np.asarray([], dtype=np.float32)\n"
                "        result['output_shape'] = list(arr.shape)\n"
                "    except Exception as exc:\n"
                "        result['ok'] = False\n"
                "        result['error_type'] = type(exc).__name__\n"
                "        result['error'] = str(exc)\n"
                "    return result\n"
                "if provider_name == 'cpu':\n"
                "    providers = ['CPUExecutionProvider']\n"
                "elif provider_name == 'rocm':\n"
                "    providers = [('ROCMExecutionProvider', rocm_provider_options), 'CPUExecutionProvider']\n"
                "else:\n"
                "    raise RuntimeError(f'unsupported provider_name: {provider_name}')\n"
                "result = bench_case(providers, cases[case_index])\n"
                "print('ORT_TTS_BENCH_CHILD_JSON=' + json.dumps(result, sort_keys=True))\n"
            ),
            encoding="utf-8",
        )
        def run_child(provider_name: str, case_index: int, mode: str) -> tuple[ProcResult, dict[str, Any] | None]:
            res = run_cmd(ctx.repo_root, bench_env, [py, str(child_script), provider_name, str(case_index), mode], timeout_s, log)
            if res.rc != 0:
                return res, None
            match = re.search(r"ORT_TTS_BENCH_CHILD_JSON=(\{.*\})", res.out + "\n" + res.err)
            if not match:
                return res, None
            return res, json.loads(match.group(1))

        total_ms = r_install.dur_ms + freeze_dur_ms
        cpu_cold_runs: list[dict[str, Any]] = []
        cpu_reuse_runs: list[dict[str, Any]] = []
        rocm_cold_runs: list[dict[str, Any]] = []
        rocm_reuse_runs: list[dict[str, Any]] = []

        for case_index in range(len(cases)):
            res, payload = run_child("cpu", case_index, "cold")
            total_ms += res.dur_ms
            if res.rc != 0 or payload is None or not bool(payload.get("ok")):
                detail = payload.get("error") if isinstance(payload, dict) else f"rc={res.rc}"
                return StepResult(build_dir, step_name, "FAIL", fmt_duration(total_ms), f"cpu cold benchmark failed for case {case_index}: {detail}")
            cpu_cold_runs.append(payload)

            res, payload = run_child("cpu", case_index, "reuse")
            total_ms += res.dur_ms
            if res.rc != 0 or payload is None or not bool(payload.get("ok")):
                detail = payload.get("error") if isinstance(payload, dict) else f"rc={res.rc}"
                return StepResult(build_dir, step_name, "FAIL", fmt_duration(total_ms), f"cpu reuse benchmark failed for case {case_index}: {detail}")
            cpu_reuse_runs.append(payload)

            res, payload = run_child("rocm", case_index, "cold")
            total_ms += res.dur_ms
            if res.rc != 0 or payload is None or not bool(payload.get("ok")):
                detail = payload.get("error") if isinstance(payload, dict) else f"rc={res.rc}"
                return StepResult(build_dir, step_name, "FAIL", fmt_duration(total_ms), f"rocm cold benchmark failed for case {case_index}: {detail}")
            rocm_cold_runs.append(payload)

            res, payload = run_child("rocm", case_index, "reuse")
            total_ms += res.dur_ms
            if res.rc != 0 or payload is None:
                return StepResult(build_dir, step_name, "FAIL", fmt_duration(total_ms), f"rocm reuse benchmark child failed for case {case_index}: rc={res.rc}")
            rocm_reuse_runs.append(payload)

    def _avg(items: list[dict[str, Any]], key: str) -> float:
        vals = [float(item.get(key, 0.0) or 0.0) for item in items if bool(item.get("ok"))]
        return float(sum(vals) / len(vals)) if vals else 0.0

    rocm_providers = (rocm_cold_runs[0].get("session_providers") if rocm_cold_runs else []) or []
    if "ROCMExecutionProvider" not in rocm_providers:
        return StepResult(build_dir, step_name, "FAIL", fmt_duration(total_ms), f"ROCMExecutionProvider missing from benchmark session providers: {rocm_providers}")

    hip_lib = str((rocm_cold_runs[0].get("hip_lib") if rocm_cold_runs else "") or "").strip()
    req = str(wl.get("require_rocm_prefix", "") or "").strip()
    if req:
        expected = str(rocm_dist) if req == "in-tree" else req.rstrip("/")
        if not hip_lib:
            return StepResult(build_dir, step_name, "FAIL", fmt_duration(total_ms), f"could not determine loaded ROCm runtime lib (libamdhip64/libMIOpen/librocblas); expected prefix: {expected}")
        if not hip_lib.startswith(expected + "/"):
            return StepResult(build_dir, step_name, "FAIL", fmt_duration(total_ms), f"ROCm runtime lib not from expected prefix: {expected}; hip_lib={hip_lib}")

    data = {
        "model": str(model),
        "model_config": str(config_path),
        "cpu": {"cold": cpu_cold_runs, "reuse": cpu_reuse_runs},
        "rocm": {"cold": rocm_cold_runs, "reuse": rocm_reuse_runs},
        "summary": {
            "cpu_create_ms": _avg(cpu_cold_runs, "session_create_ms"),
            "cpu_cold_ms": _avg(cpu_cold_runs, "run_ms"),
            "cpu_reuse_ms": _avg(cpu_reuse_runs, "run_ms"),
            "rocm_create_ms": _avg(rocm_cold_runs, "session_create_ms"),
            "rocm_cold_ms": _avg(rocm_cold_runs, "run_ms"),
            "rocm_reuse_ms": _avg(rocm_reuse_runs, "run_ms"),
            "rocm_reuse_ok": sum(1 for item in rocm_reuse_runs if bool(item.get("ok"))),
            "rocm_reuse_fail": sum(1 for item in rocm_reuse_runs if not bool(item.get("ok"))),
        },
        "hip_lib": hip_lib,
        "bench_env": {k: bench_env[k] for k in sorted(bench_env) if k not in run_env or run_env.get(k) != bench_env.get(k)},
        "rocm_provider_options": rocm_provider_options,
    }
    cpu_cold_ms = float(data["summary"]["cpu_cold_ms"])
    rocm_cold_ms = float(data["summary"]["rocm_cold_ms"])
    cpu_reuse_ms = float(data["summary"]["cpu_reuse_ms"])
    rocm_reuse_ms = float(data["summary"]["rocm_reuse_ms"])
    data["summary"]["speedup_cold_rocm_vs_cpu"] = (cpu_cold_ms / rocm_cold_ms) if rocm_cold_ms > 0.0 else None
    data["summary"]["speedup_reuse_rocm_vs_cpu"] = (cpu_reuse_ms / rocm_reuse_ms) if rocm_reuse_ms > 0.0 else None
    artifact_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    metric = (
        f"cases={len(cases)} "
        f"cpu_create_ms={float(data['summary']['cpu_create_ms']):.2f} cpu_cold_ms={cpu_cold_ms:.2f} cpu_reuse_ms={cpu_reuse_ms:.2f} "
        f"rocm_create_ms={float(data['summary']['rocm_create_ms']):.2f} rocm_cold_ms={rocm_cold_ms:.2f}"
    )
    if float(data["summary"]["rocm_reuse_ms"]) > 0.0:
        metric += f" rocm_reuse_ms={float(data['summary']['rocm_reuse_ms']):.2f}"
    metric += f" rocm_reuse_ok={int(data['summary']['rocm_reuse_ok'])}/{len(cases)}"
    speedup_cold = data["summary"].get("speedup_cold_rocm_vs_cpu")
    if speedup_cold is not None:
        metric += f" rocm_vs_cpu_cold={float(speedup_cold):.3f}x"
    speedup_reuse = data["summary"].get("speedup_reuse_rocm_vs_cpu")
    if speedup_reuse is not None:
        metric += f" rocm_vs_cpu_reuse={float(speedup_reuse):.3f}x"
    if hip_lib:
        metric += f" hip_lib={hip_lib}"
    metric += f" artifact={artifact_path.name} rocm_env={'in-tree' if use_in_tree else 'system'}"
    return StepResult(build_dir, step_name, "OK", fmt_duration(total_ms), metric)


def _step_onnxruntime_tts_steady_state_probe(
    ctx: Context,
    cfg: dict[str, Any],
    build_dir: str,
    rocm_dist: Path,
    env: dict[str, str],
    log: Path | None,
) -> StepResult:
    step_name = "ONNX Runtime Piper TTS Steady-State Probe"
    resolved = _resolve_onnxruntime_tts_inputs(ctx, cfg, step_name)
    if isinstance(resolved, StepResult):
        return StepResult(build_dir, resolved.name, resolved.status, resolved.duration, resolved.metric)
    wl, wheel, model, config_path = resolved
    py, py_err = _onnxruntime_runtime_python(ctx, wl, env, log)
    if py is None:
        return StepResult(build_dir, step_name, "FAIL", "0ms", py_err or "onnxruntime runtime venv unavailable")
    run_env, use_in_tree, _runtime_prefix = _ort_runtime_env(env, rocm_dist, wl)

    install_ok, r_install, py_or_err = _install_onnxruntime_runtime_wheel(ctx, wl, run_env, log, py, wheel)
    if not install_ok:
        return StepResult(build_dir, step_name, "FAIL", fmt_duration(r_install.dur_ms), f"pip rc={r_install.rc}")
    py = py_or_err

    loaded = _load_onnxruntime_tts_cases(ctx, wl, model, step_name)
    if isinstance(loaded, StepResult):
        return StepResult(build_dir, loaded.name, loaded.status, loaded.duration, loaded.metric)
    cases, _case_file = loaded
    case_label = str(wl.get("tts_steady_case_label", "mogli") or "").strip()
    case = next((item for item in cases if str(item.get("label") or "").strip() == case_label), None)
    if case is None:
        return StepResult(build_dir, step_name, "FAIL", "0ms", f"unknown tts_steady_case_label: {case_label}")

    seed = int(wl.get("tts_seed", 0))
    iters = max(2, int(wl.get("tts_steady_iters", 6)))
    timeout_s = max(60, int(cfg.get("timeouts_s", {}).get("onnxruntime_infer", wl.get("tts_steady_timeout_s", 900))))
    freeze_randomnormal_like_zeros = bool(wl.get("tts_freeze_randomnormal_like_zeros", False))
    rocm_provider_options = {"miopen_conv_use_max_workspace": "1"}
    steady_env = dict(run_env)
    for item in _cfg_string_list(wl.get("tts_steady_rocm_provider_options")):
        if "=" not in item:
            return StepResult(build_dir, step_name, "FAIL", "0ms", f"invalid tts_steady_rocm_provider_options entry: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            return StepResult(build_dir, step_name, "FAIL", "0ms", f"invalid tts_steady_rocm_provider_options entry: {item!r}")
        rocm_provider_options[key] = value
    for item in _cfg_string_list(wl.get("tts_steady_env")):
        if "=" not in item:
            return StepResult(build_dir, step_name, "FAIL", "0ms", f"invalid tts_steady_env entry: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            return StepResult(build_dir, step_name, "FAIL", "0ms", f"invalid tts_steady_env entry: {item!r}")
        steady_env[key] = value

    artifact_dir = ctx.run_root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "onnxruntime_tts_steady_state_probe.json"
    freeze_dur_ms = 0.0

    with tempfile.TemporaryDirectory(prefix="rocm-validation-ort-tts-steady-") as td:
        tdp = Path(td)
        runnable_model = model
        if freeze_randomnormal_like_zeros:
            diag_py = _resolve_onnxruntime_tts_diag_python(ctx, run_env, py, log)
            if diag_py is None:
                return StepResult(build_dir, step_name, "FAIL", fmt_duration(r_install.dur_ms), "no python with onnx+onnxruntime to freeze Piper RandomNormalLike nodes")
            diag_script = ctx.repo_root / "validation" / "src" / "steps" / "workloads" / "onnxruntime" / "piper_tts_debug.py"
            runnable_model = tdp / f"{model.stem}.rng_frozen.onnx"
            freeze_cmd = [
                diag_py,
                str(diag_script),
                "--model",
                str(model),
                "--freeze-dp-random-zeros",
                "--write-frozen-model",
                str(runnable_model),
            ]
            r_freeze = run_cmd(ctx.repo_root, steady_env, freeze_cmd, 300, log)
            freeze_dur_ms += r_freeze.dur_ms
            if r_freeze.rc != 0:
                return StepResult(build_dir, step_name, "FAIL", fmt_duration(r_install.dur_ms + freeze_dur_ms), f"freeze_randomnormal_like rc={r_freeze.rc}")

        child_script = tdp / "ort_tts_steady_probe_child.py"
        child_script.write_text(
            (
                "import json\n"
                "import os\n"
                "import sys\n"
                "import time\n"
                "from pathlib import Path\n"
                "import numpy as np\n"
                "import onnxruntime as ort\n"
                f"model = r'''{runnable_model}'''\n"
                f"config_path = r'''{config_path}'''\n"
                f"seed = {seed}\n"
                f"iters = {iters}\n"
                f"case = {json.dumps(case, sort_keys=True)}\n"
                f"rocm_provider_options = {json.dumps(rocm_provider_options, sort_keys=True)}\n"
                f"steady_env = {json.dumps({k: steady_env[k] for k in sorted(steady_env) if k not in run_env or run_env.get(k) != steady_env.get(k)}, sort_keys=True)}\n"
                "provider_name = sys.argv[1]\n"
                "def _dtype(type_name):\n"
                "    if type_name == 'tensor(int64)': return np.int64\n"
                "    if type_name == 'tensor(int32)': return np.int32\n"
                "    if type_name == 'tensor(float)': return np.float32\n"
                "    raise RuntimeError(f'unsupported Piper input dtype: {type_name}')\n"
                "def build_feed(sess):\n"
                "    token_ids = [int(v) for v in (case.get('ids') or [])]\n"
                "    seq = np.asarray([token_ids], dtype=np.int64)\n"
                "    lens = np.asarray([len(token_ids)], dtype=np.int64)\n"
                "    scales = np.asarray(case['scales'], dtype=np.float32)\n"
                "    feed = {}\n"
                "    for inp in sess.get_inputs():\n"
                "        dt = _dtype(inp.type)\n"
                "        if inp.name == 'input':\n"
                "            feed[inp.name] = seq.astype(dt, copy=False)\n"
                "        elif inp.name == 'input_lengths':\n"
                "            feed[inp.name] = lens.astype(dt, copy=False)\n"
                "        elif inp.name == 'scales':\n"
                "            feed[inp.name] = scales.astype(dt, copy=False)\n"
                "        elif inp.name in {'sid', 'speaker_id'}:\n"
                "            feed[inp.name] = np.asarray([0], dtype=dt)\n"
                "        else:\n"
                "            raise RuntimeError(f'unsupported Piper input name: {inp.name}')\n"
                "    return feed\n"
                "def rocm_lib_hint():\n"
                "    try:\n"
                "        with open('/proc/self/maps', 'r', encoding='utf-8', errors='ignore') as f:\n"
                "            libs = sorted({line.strip().split()[5] for line in f if len(line.strip().split()) >= 6 and line.strip().split()[5].startswith('/') and '.so' in line.strip().split()[5]})\n"
                "        preferred = ('libamdhip64.so', 'libMIOpen.so', 'librocblas.so')\n"
                "        for needle in preferred:\n"
                "            for p in libs:\n"
                "                base = os.path.basename(p)\n"
                "                if base == needle or base.startswith(needle + '.'):\n"
                "                    return p\n"
                "    except Exception:\n"
                "        return ''\n"
                "    return ''\n"
                "if provider_name == 'cpu':\n"
                "    providers = ['CPUExecutionProvider']\n"
                "elif provider_name == 'rocm':\n"
                "    providers = [('ROCMExecutionProvider', rocm_provider_options), 'CPUExecutionProvider']\n"
                "else:\n"
                "    raise RuntimeError(f'unsupported provider_name: {provider_name}')\n"
                "ort.set_seed(seed)\n"
                "np.random.seed(seed)\n"
                "so = ort.SessionOptions()\n"
                "so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL\n"
                "sess = ort.InferenceSession(model, sess_options=so, providers=providers)\n"
                "feed = build_feed(sess)\n"
                "rows = []\n"
                "for i in range(iters):\n"
                "    try:\n"
                "        t0 = time.perf_counter()\n"
                "        out = sess.run(None, feed)\n"
                "        dt = (time.perf_counter() - t0) * 1000.0\n"
                "        arr = np.asarray(out[0], dtype=np.float32)\n"
                "        rows.append({'iter': i, 'ok': True, 'ms': dt, 'shape': list(arr.shape), 'mean': float(np.mean(arr)) if arr.size else 0.0, 'std': float(np.std(arr)) if arr.size else 0.0})\n"
                "    except Exception as exc:\n"
                "        rows.append({'iter': i, 'ok': False, 'error_type': type(exc).__name__, 'error': str(exc)})\n"
                "        break\n"
                "print('ORT_TTS_STEADY_JSON=' + json.dumps({'provider_name': provider_name, 'session_providers': sess.get_providers(), 'hip_lib': rocm_lib_hint(), 'steady_env': steady_env, 'rows': rows}, sort_keys=True))\n"
            ),
            encoding="utf-8",
        )

        def run_child(provider_name: str) -> tuple[ProcResult, dict[str, Any] | None]:
            res = run_cmd(ctx.repo_root, steady_env, [py, str(child_script), provider_name], timeout_s, log)
            match = re.search(r"ORT_TTS_STEADY_JSON=(\{.*\})", res.out + "\n" + res.err)
            if not match:
                return res, None
            return res, json.loads(match.group(1))

        cpu_run, cpu_data = run_child("cpu")
        total_ms = r_install.dur_ms + freeze_dur_ms + cpu_run.dur_ms
        if cpu_data is None:
            return StepResult(build_dir, step_name, "FAIL", fmt_duration(total_ms), f"cpu steady probe rc={cpu_run.rc}")
        rocm_run, rocm_data = run_child("rocm")
        total_ms += rocm_run.dur_ms
        if rocm_data is None:
            return StepResult(build_dir, step_name, "FAIL", fmt_duration(total_ms), f"rocm steady probe rc={rocm_run.rc}")

    rocm_providers = rocm_data.get("session_providers") or []
    if "ROCMExecutionProvider" not in rocm_providers:
        return StepResult(build_dir, step_name, "FAIL", fmt_duration(total_ms), f"ROCMExecutionProvider missing from steady probe session providers: {rocm_providers}")
    hip_lib = str(rocm_data.get("hip_lib") or "").strip()
    req = str(wl.get("require_rocm_prefix", "") or "").strip()
    if req:
        expected = str(rocm_dist) if req == "in-tree" else req.rstrip("/")
        if not hip_lib:
            return StepResult(build_dir, step_name, "FAIL", fmt_duration(total_ms), f"could not determine loaded ROCm runtime lib (libamdhip64/libMIOpen/librocblas); expected prefix: {expected}")
        if not hip_lib.startswith(expected + "/"):
            return StepResult(build_dir, step_name, "FAIL", fmt_duration(total_ms), f"ROCm runtime lib not from expected prefix: {expected}; hip_lib={hip_lib}")

    data = {
        "model": str(model),
        "model_config": str(config_path),
        "case_label": case_label,
        "iters": iters,
        "cpu": cpu_data,
        "rocm": rocm_data,
        "steady_env": {k: steady_env[k] for k in sorted(steady_env) if k not in run_env or run_env.get(k) != steady_env.get(k)},
    }
    artifact_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def _unique_shapes(payload: dict[str, Any]) -> list[list[int]]:
        seen: list[list[int]] = []
        for row in payload.get("rows") or []:
            shape = [int(v) for v in (row.get("shape") or [])]
            if shape not in seen:
                seen.append(shape)
        return seen

    cpu_shapes = _unique_shapes(cpu_data)
    rocm_shapes = _unique_shapes(rocm_data)
    metric = (
        f"case={case_label} iters={iters} "
        f"cpu_shapes={cpu_shapes} rocm_shapes={rocm_shapes} "
        f"rocm_shape_stable={'yes' if len(rocm_shapes) == 1 else 'no'}"
    )
    if hip_lib:
        metric += f" hip_lib={hip_lib}"
    metric += f" artifact={artifact_path.name} rocm_env={'in-tree' if use_in_tree else 'system'}"
    return StepResult(build_dir, step_name, "OK", fmt_duration(total_ms), metric)


def _step_onnxruntime_migraphx_infer(ctx: Context, cfg: dict[str, Any], build_dir: str, rocm_dist: Path, env: dict[str, str], log: Path | None) -> StepResult:
    step_name = "ONNX Runtime inference (MIGraphX EP)"
    resolved = _resolve_onnxruntime_infer_inputs(ctx, cfg, step_name)
    if isinstance(resolved, StepResult):
        return StepResult(build_dir, resolved.name, resolved.status, resolved.duration, resolved.metric)
    wl, wheel, model = resolved
    py, py_err = _onnxruntime_runtime_python(ctx, wl, env, log)
    if py is None:
        return StepResult(build_dir, step_name, "FAIL", "0ms", py_err or "onnxruntime runtime venv unavailable")
    run_env, use_in_tree, _runtime_prefix = _ort_runtime_env(env, rocm_dist, wl)

    # Keep ORT import ABI-stable in this venv for custom wheel tests.
    install_ok, r_install, py_or_err = _install_onnxruntime_runtime_wheel(ctx, wl, run_env, log, py, wheel)
    if not install_ok:
        return StepResult(build_dir, step_name, "FAIL", fmt_duration(r_install.dur_ms), f"pip rc={r_install.rc}")
    py = py_or_err

    warmup = int(wl.get("migraphx_infer_warmup", wl.get("infer_warmup", 50)))
    iters = int(wl.get("migraphx_infer_iters", wl.get("infer_iters", 1500)))
    timeout_s = int(cfg.get("timeouts_s", {}).get("onnxruntime_migraphx_infer", 900))

    with tempfile.TemporaryDirectory(prefix="rocm-validation-ort-migraphx-infer-") as td:
        tdp = Path(td)
        script = tdp / "ort_migraphx_infer.py"
        profile_prefix = tdp / "onnxruntime_migraphx_profile"
        script.write_text(
            (
                "import json\n"
                "import time\n"
                "import numpy as np\n"
                "import os\n"
                "import onnxruntime as ort\n"
                f"model = r'''{model}'''\n"
                f"profile_prefix = r'''{profile_prefix}'''\n"
                f"warmup = {warmup}\n"
                f"iters = {iters}\n"
                "providers = ['MIGraphXExecutionProvider', 'ROCMExecutionProvider', 'CPUExecutionProvider']\n"
                "so = ort.SessionOptions()\n"
                "so.enable_profiling = True\n"
                "so.profile_file_prefix = profile_prefix\n"
                "so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL\n"
                "sess = ort.InferenceSession(model, sess_options=so, providers=providers)\n"
                "sess_providers = sess.get_providers()\n"
                "if 'MIGraphXExecutionProvider' not in sess_providers:\n"
                "    raise RuntimeError(f'MIGraphXExecutionProvider missing from session providers: {sess_providers}')\n"
                "inp = sess.get_inputs()[0]\n"
                "shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]\n"
                "dtype = np.float32\n"
                "if inp.type == 'tensor(float16)': dtype = np.float16\n"
                "elif inp.type == 'tensor(double)': dtype = np.float64\n"
                "elif inp.type == 'tensor(int64)': dtype = np.int64\n"
                "elif inp.type == 'tensor(int32)': dtype = np.int32\n"
                "x = np.random.rand(*shape).astype(dtype) if np.issubdtype(dtype, np.floating) else np.random.randint(0, 10, size=shape, dtype=dtype)\n"
                "feed = {inp.name: x}\n"
                "for _ in range(warmup):\n"
                "    sess.run(None, feed)\n"
                "t0 = time.perf_counter()\n"
                "for _ in range(iters):\n"
                "    _ = sess.run(None, feed)\n"
                "t1 = time.perf_counter()\n"
                "profile_path = sess.end_profiling()\n"
                "provider_events = 0\n"
                "with open(profile_path, 'r', encoding='utf-8') as f:\n"
                "    for ev in json.load(f):\n"
                "        p = (ev.get('args') or {}).get('provider')\n"
                "        if p == 'MIGraphXExecutionProvider':\n"
                "            provider_events += 1\n"
                "dt = t1 - t0\n"
                "res = {\n"
                "  'avg_ms': (dt * 1000.0) / iters,\n"
                "  'iters_per_s': iters / dt,\n"
                "  'iters': iters,\n"
                "  'provider_events_migraphx': provider_events,\n"
                "  'session_providers': sess_providers,\n"
                "  'model': model,\n"
                "}\n"
                "try:\n"
                "  with open('/proc/self/maps', 'r', encoding='utf-8', errors='ignore') as f:\n"
                "    libs = sorted({line.strip().split()[5] for line in f if len(line.strip().split()) >= 6 and line.strip().split()[5].startswith('/') and '.so' in line.strip().split()[5]})\n"
                "  for p in libs:\n"
                "    b = os.path.basename(p)\n"
                "    if b == 'libamdhip64.so' or b.startswith('libamdhip64.so.'):\n"
                "      res['hip_lib'] = p\n"
                "      break\n"
                "except Exception:\n"
                "  pass\n"
                "print('ORT_RESULT_JSON=' + json.dumps(res, sort_keys=True))\n"
            ),
            encoding="utf-8",
        )

        def run_infer(sampler: PowerSampler | None):
            r = run_cmd(ctx.repo_root, run_env, [py, str(script)], timeout_s, log)
            return r, sampler

        r_infer, sampler = _with_power_sampler(ctx, cfg, build_dir, "onnxruntime_migraphx_infer", run_infer)
        if r_infer.rc != 0:
            return StepResult(build_dir, step_name, "FAIL", fmt_duration(r_install.dur_ms + r_infer.dur_ms), f"infer rc={r_infer.rc}")

    m = re.search(r"ORT_RESULT_JSON=(\{.*\})", r_infer.out + "\n" + r_infer.err)
    if not m:
        return StepResult(build_dir, step_name, "FAIL", fmt_duration(r_install.dur_ms + r_infer.dur_ms), "missing ORT_RESULT_JSON in output")
    data = json.loads(m.group(1))
    if int(data.get("provider_events_migraphx", 0)) <= 0:
        return StepResult(build_dir, step_name, "FAIL", fmt_duration(r_install.dur_ms + r_infer.dur_ms), "no MIGraphXExecutionProvider events in ORT profile")

    metric = (
        f"iters={int(data.get('iters', iters))} "
        f"avg_ms={float(data.get('avg_ms', 0.0)):.4f} "
        f"iters_per_s={float(data.get('iters_per_s', 0.0)):.2f} "
        f"migraphx_events={int(data.get('provider_events_migraphx', 0))}"
    )
    hip_lib = str(data.get("hip_lib", "") or "").strip()
    if hip_lib:
        metric += f" hip_lib={hip_lib}"
    metric += f" rocm_env={'in-tree' if use_in_tree else 'system'}"
    req = str(wl.get("require_rocm_prefix", "") or "").strip()
    if req:
        expected = str(rocm_dist) if req == "in-tree" else req.rstrip("/")
        if not hip_lib:
            return StepResult(build_dir, step_name, "FAIL", fmt_duration(r_install.dur_ms + r_infer.dur_ms), f"could not determine loaded ROCm runtime lib (libamdhip64) | expected prefix: {expected} | {metric}")
        if not hip_lib.startswith(expected + "/"):
            return StepResult(build_dir, step_name, "FAIL", fmt_duration(r_install.dur_ms + r_infer.dur_ms), f"ROCm runtime lib not from expected prefix: {expected} | hip_lib={hip_lib}")
    metric = _append_power(metric, sampler, baseline_avg_w=_get_baseline_avg_w(cfg, build_dir))
    return StepResult(build_dir, step_name, "OK", fmt_duration(r_install.dur_ms + r_infer.dur_ms), metric)


def _step_rocblas(ctx: Context, cfg: dict[str, Any], build_dir: str, rocm_dist: Path, env: dict[str, str], log: Path | None) -> StepResult:
    t = int(cfg.get("timeouts_s", {}).get("rocblas_bench", 60))
    if not which("rocblas-bench", env):
        return StepResult(build_dir, "rocBLAS GEMM f32", "SKIP", "0ms", "rocblas-bench not in PATH")
    # Single long-running bench invocation (avoid "pulses" from repeated process startup).
    m = n = k = 6144
    iters = 80
    cmd = [
        "rocblas-bench",
        "-f",
        "gemm",
        "-r",
        "f32_r",
        "-m",
        str(m),
        "-n",
        str(n),
        "-k",
        str(k),
        "--alpha",
        "1",
        "--beta",
        "0",
        "--iters",
        str(iters),
    ]

    def run_one(sampler: PowerSampler | None):
        r = run_cmd(ctx.repo_root, env, cmd, t, log)
        return r, sampler

    r, sampler = _with_power_sampler(ctx, cfg, build_dir, "rocblas_gemm_f32", run_one)
    if r.rc != 0:
        return StepResult(build_dir, "rocBLAS GEMM f32", "FAIL", fmt_duration(r.dur_ms), f"rc={r.rc}")
    gflops = _extract_gflops(r.out + "\n" + r.err)
    metric = f"m=n=k={m} iters={iters}"
    if gflops is not None:
        metric += f" TFLOPS={gflops/1000.0:.3f} (GFLOPS={gflops:.1f})"
    metric = _append_power(metric, sampler, baseline_avg_w=_get_baseline_avg_w(cfg, build_dir))
    return StepResult(build_dir, "rocBLAS GEMM f32", "OK", fmt_duration(r.dur_ms), metric)


def _step_rocfft(ctx: Context, cfg: dict[str, Any], build_dir: str, rocm_dist: Path, env: dict[str, str], log: Path | None) -> StepResult:
    t = int(cfg.get("timeouts_s", {}).get("rocfft_bench", 30))
    if not which("rocfft-bench", env):
        return StepResult(build_dir, "rocFFT 1024", "SKIP", "0ms", "rocfft-bench not in PATH")
    # Single long-running bench invocation.
    length = 1_048_576
    batch = 16
    ntrial = 800
    cmd = ["rocfft-bench", "--length", str(length), "--precision", "single", "-t", "0", "-b", str(batch), "-N", str(ntrial)]

    def run_one(sampler: PowerSampler | None):
        r = run_cmd(ctx.repo_root, env, cmd, max(t, 120), log)
        return r, sampler

    r, sampler = _with_power_sampler(ctx, cfg, build_dir, "rocfft", run_one)
    if r.rc != 0:
        return StepResult(build_dir, "rocFFT", "FAIL", fmt_duration(r.dur_ms), f"rc={r.rc}")
    metric = f"len={length} batch={batch} ntrial={ntrial}"
    metric = _append_power(metric, sampler, baseline_avg_w=_get_baseline_avg_w(cfg, build_dir))
    return StepResult(build_dir, "rocFFT", "OK", fmt_duration(r.dur_ms), metric)


def _step_rocrand(ctx: Context, cfg: dict[str, Any], build_dir: str, rocm_dist: Path, env: dict[str, str], log: Path | None) -> StepResult:
    t = int(cfg.get("timeouts_s", {}).get("rocrand_bench", 30))
    if not which("benchmark_rocrand_generate", env):
        return StepResult(build_dir, "rocRAND generate", "SKIP", "0ms", "benchmark_rocrand_generate not in PATH")
    # Single long-running bench invocation.
    size = 134_217_728  # 512MB of floats
    trials = 3300
    cmd = [
        "benchmark_rocrand_generate",
        "--size",
        str(size),
        "--trials",
        str(trials),
        "--dis",
        "uniform-float",
        "--engine",
        "philox",
        "--format",
        "csv",
    ]

    def run_one(sampler: PowerSampler | None):
        r = run_cmd(ctx.repo_root, env, cmd, max(t, 120), log)
        return r, sampler

    r, sampler = _with_power_sampler(ctx, cfg, build_dir, "rocrand_generate", run_one)
    if r.rc != 0:
        return StepResult(build_dir, "rocRAND generate", "FAIL", fmt_duration(r.dur_ms), f"rc={r.rc}")
    metric = f"size={size} trials={trials}"
    metric = _append_power(metric, sampler, baseline_avg_w=_get_baseline_avg_w(cfg, build_dir))
    return StepResult(build_dir, "rocRAND generate", "OK", fmt_duration(r.dur_ms), metric)


def _step_miopen_driver(ctx: Context, cfg: dict[str, Any], build_dir: str, rocm_dist: Path, env: dict[str, str], log: Path | None) -> StepResult:
    t = int(cfg.get("timeouts_s", {}).get("miopen_driver", 10))
    drv = which("MIOpenDriver", env) or which("miopen-driver", env)
    if not drv:
        return StepResult(build_dir, "MIOpen driver", "SKIP", "0ms", "MIOpenDriver/miopen-driver not in PATH")
    r = run_cmd(ctx.repo_root, env, [drv, "--version"], t, log)
    return StepResult(build_dir, "MIOpen driver", "OK" if r.rc == 0 else "FAIL", fmt_duration(r.dur_ms), f"driver={Path(drv).name}")


def _step_miopen_smoke(ctx: Context, cfg: dict[str, Any], build_dir: str, rocm_dist: Path, env: dict[str, str], log: Path | None) -> StepResult:
    t = int(cfg.get("timeouts_s", {}).get("miopen_smoke", 240))
    drv = which("MIOpenDriver", env) or which("miopen-driver", env)
    if not drv:
        return StepResult(build_dir, "MIOpen smoke", "SKIP", "0ms", "MIOpenDriver/miopen-driver not in PATH")
    def cmd_for(iters: int) -> list[str]:
        # Use a moderately-sized forward conv with timing enabled and verification disabled.
        # Goal: ~5s of sustained GPU load (adaptive iters).
        return [
            drv,
            "conv",
            "--forw",
            "1",
            "--verify",
            "0",
            "--gpualloc",
            "1",
            "--time",
            "1",
            "--wall",
            "1",
            "--iter",
            str(iters),
            "--batchsize",
            "32",
            "--in_channels",
            "64",
            "--out_channels",
            "64",
            "--in_h",
            "224",
            "--in_w",
            "224",
            "--fil_h",
            "3",
            "--fil_w",
            "3",
            "--pad_h",
            "1",
            "--pad_w",
            "1",
        ]

    # Single long-running invocation. `--iter` impacts runtime (measured on RX 6700 XT).
    iters = 700
    cmd = cmd_for(iters)

    def run_one(sampler: PowerSampler | None):
        r = run_cmd(ctx.repo_root, env, cmd, t, log)
        return r, sampler

    r, sampler = _with_power_sampler(ctx, cfg, build_dir, "miopen_smoke", run_one)
    if r.rc != 0:
        return StepResult(build_dir, "MIOpen smoke", "FAIL", fmt_duration(r.dur_ms), f"rc={r.rc}")
    metric = f"driver={Path(drv).name} iters={iters}"
    metric = _append_power(metric, sampler, baseline_avg_w=_get_baseline_avg_w(cfg, build_dir))
    return StepResult(build_dir, "MIOpen smoke", "OK", fmt_duration(r.dur_ms), metric)


def build_plan(cfg: dict[str, Any], *, doctor_only: bool = False) -> list[Step]:
    steps_cfg = cfg.get("steps", {})
    pytorch_cfg = cfg.get("workloads", {}).get("pytorch", {}) or {}
    plan: list[Step] = []

    def group_enabled(group: str) -> bool:
        return bool(steps_cfg.get(group, True))

    def add(group: str, key: str, name: str, expected: str, fn):
        if doctor_only and group not in {"rocm_sanity"}:
            return
        if not group_enabled(group):
            return
        plan.append(Step(group=group, key=key, name=name, expected=expected, fn=fn))

    add("rocm_sanity", "rocm_env", "ROCm env activation", "typ. <50ms", _step_rocm_env)
    add("rocm_sanity", "power_idle_baseline", "Power idle baseline", "5s (no load)", _step_power_idle_baseline)
    add("rocm_sanity", "rocminfo", "rocminfo", "typ. <1s", _step_rocminfo)
    add("rocm_sanity", "hipcc_compile_run", "hipcc compile+run", "typ. ~5s (sustained)", _step_hipcc_compile_run)

    add("rocm_bench_smoke", "rocblas_gemm_f32", "rocBLAS GEMM f32", "typ. ~5s (continuous)", _step_rocblas)
    add("rocm_bench_smoke", "rocfft", "rocFFT", "typ. ~5s (continuous)", _step_rocfft)
    add("rocm_bench_smoke", "rocrand_generate", "rocRAND generate", "typ. ~5s (continuous)", _step_rocrand)

    add("miopen_smoke", "miopen_driver", "MIOpen driver", "typ. <1s", _step_miopen_driver)
    add("miopen_smoke", "miopen_smoke", "MIOpen smoke", "typ. ~5s (continuous; first run may JIT)", _step_miopen_smoke)

    add("llama_cpp_docker", "llama_cpp_docker", "llama.cpp (docker) smoke", "minutes (pull), ~10-60s bench", step_llama_cpp_docker)
    add("ollama", "ollama", "Ollama (local binary) smoke", "<10s download, <1s version", step_ollama)
    add("open_interpreter", "open_interpreter", "Open Interpreter (pip) smoke", "minutes (pip), <2s help", step_open_interpreter)
    add("whisper", "whisper", "Whisper (python) smoke", "<5s run (if installed)", step_whisper)
    add("mfem_hip", "mfem_hip", "MFEM (HIP) build+run", "minutes (clone/build), <5s run", step_mfem_hip)
    if bool(pytorch_cfg.get("run_audio", True)):
        add("pytorch", "pytorch_audio", "PyTorch (audio) conv", "typ. ~5s (sustained)", step_pytorch_audio)
    if bool(pytorch_cfg.get("run_video", True)):
        add("pytorch", "pytorch_video", "PyTorch (video) conv", "typ. ~5s (sustained)", step_pytorch_video)
    add("petsc_hip", "petsc_hip", "PETSc (HIP) build+solve", "minutes (clone/build), ~5s solve", step_petsc_hip)
    add("onnxruntime_rocm_wheel", "onnxruntime_rocm_wheel", "ONNX Runtime (ROCm) wheel build", "hours (clone/build)", step_onnxruntime_rocm_wheel)
    add("onnxruntime_infer", "onnxruntime_infer", "ONNX Runtime inference (ROCm)", "typ. ~5s (continuous)", _step_onnxruntime_infer)
    add("onnxruntime_tts_infer", "onnxruntime_tts_infer", "ONNX Runtime Piper TTS (ROCm)", "typ. ~5s (continuous; real Piper graph)", _step_onnxruntime_tts_infer)
    add("onnxruntime_tts_benchmark", "onnxruntime_tts_benchmark", "ONNX Runtime Piper TTS Benchmark", "typ. ~10-60s (CPU vs ROCm real-model timing)", _step_onnxruntime_tts_benchmark)
    add("onnxruntime_tts_steady_state_probe", "onnxruntime_tts_steady_state_probe", "ONNX Runtime Piper TTS Steady-State Probe", "typ. ~10-60s (repeated-run real-model diagnosis)", _step_onnxruntime_tts_steady_state_probe)
    add("onnxruntime_tts_flow_probe", "onnxruntime_tts_flow_probe", "ONNX Runtime Piper TTS Flow Probe (ROCm)", "typ. ~10-60s (diagnostic subgraph repro)", _step_onnxruntime_tts_flow_probe)
    add("onnxruntime_migraphx_infer", "onnxruntime_migraphx_infer", "ONNX Runtime inference (MIGraphX EP)", "typ. ~5s (continuous)", _step_onnxruntime_migraphx_infer)
    add("tensorflow_rocm_wheel", "tensorflow_rocm_wheel", "TensorFlow (ROCm) wheel build", "hours (clone/build)", step_tensorflow_rocm_wheel)
    add("tensorflow_functional", "tensorflow_matmul", "TensorFlow matmul (GPU)", "typ. ~5s (sustained)", step_tensorflow_matmul)
    return plan


def run_plan(ctx: Context, cfg: dict[str, Any], plan: list[Step]) -> list[StepResult]:
    build_dirs = cfg.get("run", {}).get("build_dirs") or []
    if not build_dirs:
        if bool(cfg.get("run", {}).get("all_build_dirs", False)):
            build_dirs = detect_build_dirs(ctx.repo_root)
        else:
            build_dirs = [detect_default_build_dir(ctx.repo_root)]

    results: list[StepResult] = []
    report: dict[str, Any] = {"run_id": ctx.run_id, "build_dirs": build_dirs, "results": []}
    for build_dir in build_dirs:
        rocm_dist = rocm_dist_for_build(ctx.repo_root, build_dir)
        env = activated_env(ctx.env_base(), rocm_dist)
        for step in plan:
            log = _log_path(ctx, build_dir, step.key)
            r = step.fn(ctx, cfg, build_dir, rocm_dist, env, log)
            results.append(r)
            report["results"].append(
                {
                    "build_dir": r.build_dir,
                    "group": step.group,
                    "key": step.key,
                    "name": r.name,
                    "status": r.status,
                    "duration": r.duration,
                    "metric": r.metric,
                }
            )
    write_report_json(ctx, report)
    return results
