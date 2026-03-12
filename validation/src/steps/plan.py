from __future__ import annotations

import json
import os
import re
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
        model = Path(model_cfg)
        if not model.is_absolute():
            model = ctx.repo_root / model
    else:
        model_dir = ctx.repo_root / "validation" / "workspace" / "cache" / "models" / "onnxruntime_tts"
        models = sorted(model_dir.glob("*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True) if model_dir.is_dir() else []
        if not models:
            return StepResult("", step_name, "SKIP", "0ms", f"no staged Piper ONNX model in {model_dir}")
        model = models[0]
    if not model.is_file():
        return StepResult("", step_name, "SKIP", "0ms", f"missing Piper ONNX model: {model}")

    config_cfg = str(wl.get("tts_model_config", "") or "").strip()
    if config_cfg:
        config_path = Path(config_cfg)
        if not config_path.is_absolute():
            config_path = ctx.repo_root / config_path
    else:
        config_path = model.with_suffix(model.suffix + ".json")
    if not config_path.is_file():
        return StepResult("", step_name, "SKIP", "0ms", f"missing Piper config sidecar: {config_path}")
    return wl, wheel, model, config_path


def _step_onnxruntime_infer(ctx: Context, cfg: dict[str, Any], build_dir: str, rocm_dist: Path, env: dict[str, str], log: Path | None) -> StepResult:
    py = sys.executable
    step_name = "ONNX Runtime inference (ROCm)"
    resolved = _resolve_onnxruntime_infer_inputs(ctx, cfg, step_name)
    if isinstance(resolved, StepResult):
        return StepResult(build_dir, resolved.name, resolved.status, resolved.duration, resolved.metric)
    wl, wheel, model = resolved
    use_in_tree = bool(wl.get("use_in_tree_rocm", True))
    run_env = env if use_in_tree else deactivated_env(env, rocm_dist)

    # Keep ORT import ABI-stable in this venv for custom wheel tests.
    install_cmd = [py, "-m", "pip", "install", "-q", "--force-reinstall", "numpy<2", "protobuf<5", str(wheel)]
    r_install = run_cmd(ctx.repo_root, run_env, install_cmd, 600, log)
    if r_install.rc != 0:
        return StepResult(
            build_dir,
            step_name,
            "FAIL",
            fmt_duration(r_install.dur_ms),
            f"pip rc={r_install.rc}",
        )

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
    py = sys.executable
    step_name = "ONNX Runtime Piper TTS (ROCm)"
    resolved = _resolve_onnxruntime_tts_inputs(ctx, cfg, step_name)
    if isinstance(resolved, StepResult):
        return StepResult(build_dir, resolved.name, resolved.status, resolved.duration, resolved.metric)
    wl, wheel, model, config_path = resolved
    use_in_tree = bool(wl.get("use_in_tree_rocm", True))
    run_env = env if use_in_tree else deactivated_env(env, rocm_dist)

    install_cmd = [py, "-m", "pip", "install", "-q", "--force-reinstall", "numpy<2", "protobuf<5", str(wheel)]
    r_install = run_cmd(ctx.repo_root, run_env, install_cmd, 600, log)
    if r_install.rc != 0:
        return StepResult(build_dir, step_name, "FAIL", fmt_duration(r_install.dur_ms), f"pip rc={r_install.rc}")

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
            return StepResult(build_dir, step_name, "FAIL", "0ms", f"invalid tts_case_file {case_file}: {exc}")
        for case in loaded_cases:
            try:
                ids = [int(v) for v in (case.get("ids") or [])]
                scale_list = [float(v) for v in (case.get("scales") or [])]
            except Exception as exc:
                return StepResult(build_dir, step_name, "FAIL", "0ms", f"invalid TTS case in {case_file}: {exc}")
            if not ids:
                return StepResult(build_dir, step_name, "FAIL", "0ms", f"invalid TTS case in {case_file}: empty ids")
            if len(scale_list) != 3:
                return StepResult(build_dir, step_name, "FAIL", "0ms", f"invalid TTS case in {case_file}: expected 3 floats in scales")
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
            return StepResult(build_dir, step_name, "FAIL", "0ms", f"invalid tts_phoneme_lengths: {exc}")
        scales_cfg = wl.get("tts_scales") or [[0.667, 0.75, 0.8], [0.667, 1.0, 0.8], [0.667, 1.25, 0.8]]
        for phoneme_len in phoneme_lengths:
            for scales in scales_cfg:
                try:
                    scale_list = [float(v) for v in scales]
                except Exception as exc:
                    return StepResult(build_dir, step_name, "FAIL", "0ms", f"invalid tts_scales entry {scales!r}: {exc}")
                if len(scale_list) != 3:
                    return StepResult(build_dir, step_name, "FAIL", "0ms", f"invalid tts_scales entry {scales!r}: expected 3 floats")
                cases.append({"label": f"synthetic_len={phoneme_len}", "phoneme_len": phoneme_len, "scales": scale_list})
    if not cases:
        return StepResult(build_dir, step_name, "FAIL", "0ms", "no Piper TTS validation cases configured")

    warmup = max(0, int(wl.get("tts_warmup", 1)))
    iters = max(1, int(wl.get("tts_iters", 4)))
    seed = int(wl.get("tts_seed", 0))
    require_cpu_reference = bool(wl.get("tts_require_cpu_reference", True))
    require_zero_miopen_warnings = bool(wl.get("tts_require_zero_miopen_workspace_warnings", False))
    timeout_s = int(cfg.get("timeouts_s", {}).get("onnxruntime_infer", 900))
    debug_dir = ctx.repo_root / "validation" / "workspace" / "debug" / "onnxruntime_tts_profiles"
    debug_dir.mkdir(parents=True, exist_ok=True)
    profile_prefix = debug_dir / f"{build_dir.replace('/', '_')}_onnxruntime_tts_profile"

    with tempfile.TemporaryDirectory(prefix="rocm-validation-ort-tts-") as td:
        tdp = Path(td)
        script = tdp / "ort_tts_infer.py"
        script.write_text(
            (
                "import json\n"
                "import os\n"
                "import time\n"
                "from pathlib import Path\n"
                "import numpy as np\n"
                "import onnxruntime as ort\n"
                f"model = r'''{model}'''\n"
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
                "            shape = list(np.asarray(last[0]).shape) if last else []\n"
                "            out['case_results'].append({'label': label, 'phoneme_len': int(feed['input_lengths'][0]), 'scales': case['scales'], 'output_shape': shape})\n"
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
                "    return out\n"
                "result = {\n"
                "    'model': model,\n"
                "    'model_config': config_path,\n"
                "}\n"
                "if require_cpu_reference:\n"
                "    result['cpu'] = run_cases(['CPUExecutionProvider'], enable_profiling=False, profile_name='')\n"
                "result['rocm'] = run_cases(['ROCMExecutionProvider', 'CPUExecutionProvider'], enable_profiling=True, profile_name=profile_prefix)\n"
                "result['hip_lib'] = rocm_lib_hint()\n"
                "print('ORT_TTS_RESULT_JSON=' + json.dumps(result, sort_keys=True))\n"
            ),
            encoding="utf-8",
        )

        def run_tts(sampler: PowerSampler | None):
            r = run_cmd(ctx.repo_root, run_env, [py, str(script)], timeout_s, log)
            return r, sampler

        r_tts, sampler = _with_power_sampler(ctx, cfg, build_dir, "onnxruntime_tts_infer", run_tts)
        if r_tts.rc != 0:
            return StepResult(build_dir, step_name, "FAIL", fmt_duration(r_install.dur_ms + r_tts.dur_ms), f"tts rc={r_tts.rc}")

    combined = r_tts.out + "\n" + r_tts.err
    m = re.search(r"ORT_TTS_RESULT_JSON=(\{.*\})", combined)
    if not m:
        return StepResult(build_dir, step_name, "FAIL", fmt_duration(r_install.dur_ms + r_tts.dur_ms), "missing ORT_TTS_RESULT_JSON in output")
    data = json.loads(m.group(1))
    miopen_warning_count = len(re.findall(r"MIOpen\(HIP\): Warning \[IsEnoughWorkspace\]", combined))

    cpu_data = data.get("cpu") if require_cpu_reference else None
    if require_cpu_reference:
        if not isinstance(cpu_data, dict):
            return StepResult(build_dir, step_name, "FAIL", fmt_duration(r_install.dur_ms + r_tts.dur_ms), "missing CPU reference data")
        if int(cpu_data.get("failed_cases", 0)) > 0:
            first = (cpu_data.get("failures") or [{}])[0]
            return StepResult(
                build_dir,
                step_name,
                "FAIL",
                fmt_duration(r_install.dur_ms + r_tts.dur_ms),
                f"CPU reference failed for {int(cpu_data.get('failed_cases', 0))}/{int(cpu_data.get('case_count', 0))} cases; first={first.get('label') or first.get('phoneme_len')}:{first.get('scales')} {first.get('error_type')} {first.get('error')}",
            )

    rocm_data = data.get("rocm") or {}
    if int(rocm_data.get("provider_events_rocm", 0)) <= 0:
        return StepResult(build_dir, step_name, "FAIL", fmt_duration(r_install.dur_ms + r_tts.dur_ms), "no ROCMExecutionProvider events in Piper TTS ORT profile")

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
            fmt_duration(r_install.dur_ms + r_tts.dur_ms),
            detail,
        )

    if require_zero_miopen_warnings and miopen_warning_count > 0:
        return StepResult(
            build_dir,
            step_name,
            "FAIL",
            fmt_duration(r_install.dur_ms + r_tts.dur_ms),
            f"MIOpen emitted workspace warnings: {miopen_warning_count}",
        )

    req = str(wl.get("require_rocm_prefix", "") or "").strip()
    if req:
        expected = str(rocm_dist) if req == "in-tree" else req.rstrip("/")
        if not hip_lib:
            return StepResult(build_dir, step_name, "FAIL", fmt_duration(r_install.dur_ms + r_tts.dur_ms), f"could not determine loaded ROCm runtime lib (libamdhip64/libMIOpen/librocblas); expected prefix: {expected}")
        if not hip_lib.startswith(expected + "/"):
            return StepResult(build_dir, step_name, "FAIL", fmt_duration(r_install.dur_ms + r_tts.dur_ms), f"ROCm runtime lib not from expected prefix: {expected}; hip_lib={hip_lib}")

    metric = (
        f"cases={int(rocm_data.get('case_count', 0))} "
        f"rocm_ok={int(rocm_data.get('ok_cases', 0))}/{int(rocm_data.get('case_count', 0))} "
        f"avg_case_ms={float(rocm_data.get('avg_case_ms', 0.0)):.2f} "
        f"rocm_events={int(rocm_data.get('provider_events_rocm', 0))} "
        f"miopen_warn={miopen_warning_count}"
    )
    if require_cpu_reference and isinstance(cpu_data, dict):
        metric += f" cpu_ok={int(cpu_data.get('ok_cases', 0))}/{int(cpu_data.get('case_count', 0))}"
    if hip_lib:
        metric += f" hip_lib={hip_lib}"
    if profile_path:
        metric += f" profile={profile_path}"
    metric += f" rocm_env={'in-tree' if use_in_tree else 'system'}"
    metric = _append_power(metric, sampler, baseline_avg_w=_get_baseline_avg_w(cfg, build_dir))
    return StepResult(build_dir, step_name, "OK", fmt_duration(r_install.dur_ms + r_tts.dur_ms), metric)


def _step_onnxruntime_migraphx_infer(ctx: Context, cfg: dict[str, Any], build_dir: str, rocm_dist: Path, env: dict[str, str], log: Path | None) -> StepResult:
    py = sys.executable
    step_name = "ONNX Runtime inference (MIGraphX EP)"
    resolved = _resolve_onnxruntime_infer_inputs(ctx, cfg, step_name)
    if isinstance(resolved, StepResult):
        return StepResult(build_dir, resolved.name, resolved.status, resolved.duration, resolved.metric)
    wl, wheel, model = resolved
    use_in_tree = bool(wl.get("use_in_tree_rocm", True))
    run_env = env if use_in_tree else deactivated_env(env, rocm_dist)

    # Keep ORT import ABI-stable in this venv for custom wheel tests.
    install_cmd = [py, "-m", "pip", "install", "-q", "--force-reinstall", "numpy<2", "protobuf<5", str(wheel)]
    r_install = run_cmd(ctx.repo_root, run_env, install_cmd, 600, log)
    if r_install.rc != 0:
        return StepResult(build_dir, step_name, "FAIL", fmt_duration(r_install.dur_ms), f"pip rc={r_install.rc}")

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
    add("pytorch", "pytorch_audio", "PyTorch (audio) conv", "typ. ~5s (sustained)", step_pytorch_audio)
    add("pytorch", "pytorch_video", "PyTorch (video) conv", "typ. ~5s (sustained)", step_pytorch_video)
    add("petsc_hip", "petsc_hip", "PETSc (HIP) build+solve", "minutes (clone/build), ~5s solve", step_petsc_hip)
    add("onnxruntime_rocm_wheel", "onnxruntime_rocm_wheel", "ONNX Runtime (ROCm) wheel build", "hours (clone/build)", step_onnxruntime_rocm_wheel)
    add("onnxruntime_infer", "onnxruntime_infer", "ONNX Runtime inference (ROCm)", "typ. ~5s (continuous)", _step_onnxruntime_infer)
    add("onnxruntime_tts_infer", "onnxruntime_tts_infer", "ONNX Runtime Piper TTS (ROCm)", "typ. ~5s (continuous; real Piper graph)", _step_onnxruntime_tts_infer)
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
