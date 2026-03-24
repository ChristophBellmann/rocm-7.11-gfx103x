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
from steps.workloads.pytorch.setup import ensure_pytorch, with_openmp_runtime_env


def _gpu_required_script(kind: str) -> str:
    # NOTE: torch uses "cuda" device strings even on ROCm.
    # We treat missing torch/ROCm as SKIP/FAIL at the step layer.
    return rf"""
import os, time
import torch

min_s=float(os.environ.get("ROCM_VALIDATION_PYTORCH_MIN_S","5"))
kind=os.environ.get("ROCM_VALIDATION_PYTORCH_KIND","audio")
expect_hip=os.environ.get("ROCM_VALIDATION_PYTORCH_EXPECT_HIP_SUBSTR","").strip()
expect_rocm=os.environ.get("ROCM_VALIDATION_PYTORCH_EXPECT_ROCM_SUBSTR","").strip()
need_torchaudio=os.environ.get("ROCM_VALIDATION_PYTORCH_REQUIRE_TORCHAUDIO","0") == "1"
need_torchcodec=os.environ.get("ROCM_VALIDATION_PYTORCH_REQUIRE_TORCHCODEC","0") == "1"

print("torch", getattr(torch, "__version__", ""))
print("torch.cuda.is_available", torch.cuda.is_available())
v=getattr(torch, "version", None)
hip_ver=getattr(v, "hip", None)
rocm_ver=getattr(v, "rocm", None)
print("torch.version.hip", hip_ver)
print("torch.version.rocm", rocm_ver)
if not torch.cuda.is_available():
    print("GPU_NOT_AVAILABLE")
    raise SystemExit(2)
if hip_ver in (None, "", "None"):
    print("GPU_NOT_ROCM")
    raise SystemExit(3)
if expect_hip and (expect_hip not in str(hip_ver)):
    print("HIP_VERSION_MISMATCH", expect_hip, hip_ver)
    raise SystemExit(4)
if expect_rocm and (expect_rocm not in str(rocm_ver)):
    print("ROCM_VERSION_MISMATCH", expect_rocm, rocm_ver)
    raise SystemExit(5)

if need_torchaudio:
    try:
        import torchaudio
        print("torchaudio.version", getattr(torchaudio, "__version__", ""))
    except Exception as e:
        print("TORCHAUDIO_IMPORT_FAILED", repr(e))
        raise SystemExit(6)

if need_torchcodec:
    try:
        import torchcodec
        print("torchcodec.module", getattr(torchcodec, "__file__", ""))
    except Exception as e:
        print("TORCHCODEC_IMPORT_FAILED", repr(e))
        raise SystemExit(7)

dev=torch.device("cuda")
name=torch.cuda.get_device_name(0)
print("device_name", name)

# Record where ROCm runtime libraries are actually loaded from (for "in-tree" vs
# "system" validation).
try:
    import os as _os
    import re as _re

    def _loaded_so_paths():
        out=set()
        with open("/proc/self/maps","r",encoding="utf-8",errors="ignore") as f:
            for line in f:
                parts=line.strip().split()
                if len(parts) < 6:
                    continue
                p=parts[5]
                if p.startswith("/") and ".so" in p:
                    out.add(p)
        return sorted(out)

    def _find_lib(paths, base):
        # Match libfoo.so or libfoo.so.<ver>
        pat=_re.compile(_re.escape(base) + r"(\..*)?$")
        for p in paths:
            b=_os.path.basename(p)
            if pat.match(b):
                return p
        return ""

    _paths=_loaded_so_paths()
    for _lib in ("libamdhip64.so","libhsa-runtime64.so","libhiprtc.so"):
        _p=_find_lib(_paths, _lib)
        if _p:
            print("loaded", _lib, _p)
except Exception as e:
    print("loaded_libs_error", type(e).__name__)

torch.manual_seed(0)

if kind=="audio":
    if need_torchaudio:
        # Real torchaudio GPU path: repeated resample on batched waveforms.
        sr_in=48000; sr_out=16000
        B=32; seconds=8; L=sr_in*seconds
        x=torch.randn(B, L, device=dev, dtype=torch.float32)
        y=torchaudio.functional.resample(x, sr_in, sr_out)
        torch.cuda.synchronize()
        t0=time.time()
        runs=0
        while (time.time()-t0) < min_s:
            y=torchaudio.functional.resample(x, sr_in, sr_out)
            # Keep downstream tensor use to avoid compiler/lazy elision.
            _=torch.mean(torch.abs(y))
            runs += 1
        torch.cuda.synchronize()
        dt=time.time()-t0
        samples = runs * B * L
        print("GPU_OK")
        print("kind", "audio")
        print("audio_backend", "torchaudio_resample")
        print("shape", f"{{B}}x{{L}}")
        print("dtype", "fp32")
        print("runs", runs)
        print("seconds", dt)
        if dt > 0:
            print("samples_per_s", samples/dt)
    else:
        # Fallback path when torchaudio is not required: Conv1d compute.
        B=16; C=64; L=16384
        conv=torch.nn.Conv1d(C, 128, kernel_size=33, padding=16, bias=False).to(dev, dtype=torch.float16).eval()
        x=torch.randn(B, C, L, device=dev, dtype=torch.float16)
        # Warmup
        y=conv(x); torch.cuda.synchronize()
        t0=time.time()
        runs=0
        while (time.time()-t0) < min_s:
            y=conv(x)
            runs += 1
        torch.cuda.synchronize()
        dt=time.time()-t0
        # Rough FLOP estimate for Conv1d (multiply+add):
        # out_len ~= L, out_ch=128, in_ch=64, k=33
        flops = runs * 2.0 * B * 128 * L * 64 * 33
        print("GPU_OK")
        print("kind", "audio")
        print("audio_backend", "conv1d")
        print("shape", f"{{B}}x{{C}}x{{L}}")
        print("dtype", "fp16")
        print("runs", runs)
        print("seconds", dt)
        print("tflops_est", flops/dt/1e12)
else:
    # Conv3d on a small video-like tensor: (B,C,T,H,W)
    B=2; C=32; T=16; H=112; W=112
    conv=torch.nn.Conv3d(C, 64, kernel_size=3, padding=1, bias=False).to(dev, dtype=torch.float16).eval()
    x=torch.randn(B, C, T, H, W, device=dev, dtype=torch.float16)
    y=conv(x); torch.cuda.synchronize()
    t0=time.time()
    runs=0
    while (time.time()-t0) < min_s:
        y=conv(x)
        runs += 1
    torch.cuda.synchronize()
    dt=time.time()-t0
    # Rough FLOP estimate for Conv3d (multiply+add):
    # out_ch=64, in_ch=32, k^3=27, out volume ~= T*H*W
    flops = runs * 2.0 * B * 64 * (T*H*W) * 32 * 27
    print("GPU_OK")
    print("kind", "video")
    print("shape", f"{{B}}x{{C}}x{{T}}x{{H}}x{{W}}")
    print("dtype", "fp16")
    print("runs", runs)
    print("seconds", dt)
    print("tflops_est", flops/dt/1e12)
""".strip()


def _step_pytorch_conv(ctx: Context, cfg: dict[str, Any], build_dir: str, rocm_dist: Path, env: dict[str, str], log: Path | None, *, kind: str) -> StepResult:
    wl = cfg.get("workloads", {}).get("pytorch", {}) or {}
    use_in_tree = bool(wl.get("use_in_tree_rocm", False))
    run_env = env if use_in_tree else deactivated_env(env, rocm_dist)
    run_env = with_openmp_runtime_env(run_env)

    meta = ensure_pytorch(ctx, cfg, run_env, log, rocm_dist=rocm_dist)
    if meta is not None and meta.status != "OK":
        return StepResult(build_dir, f"PyTorch ({kind})", meta.status, meta.duration, meta.metric)

    min_s = float(cfg.get("workloads", {}).get("pytorch", {}).get("min_bench_s", 5.0) or 5.0)
    t = int(cfg.get("timeouts_s", {}).get("pytorch", 900))
    run_env = dict(run_env)
    run_env["ROCM_VALIDATION_PYTORCH_MIN_S"] = str(min_s)
    run_env["ROCM_VALIDATION_PYTORCH_KIND"] = kind
    expect_hip = str(wl.get("expected_hip_substr", "") or "").strip()
    if expect_hip:
        run_env["ROCM_VALIDATION_PYTORCH_EXPECT_HIP_SUBSTR"] = expect_hip
    expect_rocm = str(wl.get("expected_rocm_substr", "") or "").strip()
    if expect_rocm:
        run_env["ROCM_VALIDATION_PYTORCH_EXPECT_ROCM_SUBSTR"] = expect_rocm
    if bool(wl.get("require_torchaudio", False)):
        run_env["ROCM_VALIDATION_PYTORCH_REQUIRE_TORCHAUDIO"] = "1"
    if bool(wl.get("require_torchcodec", False)):
        run_env["ROCM_VALIDATION_PYTORCH_REQUIRE_TORCHCODEC"] = "1"
    # Improve crash diagnostics (some ROCm/runtime mismatches can segfault).
    run_env.setdefault("PYTHONUNBUFFERED", "1")
    run_env.setdefault("PYTHONFAULTHANDLER", "1")
    # Some prebuilt ROCm wheels ship code objects for gfx1030 but not gfx1031.
    # Allow opting into the common compatibility workaround (only if explicitly configured).
    if not use_in_tree:
        arch = str(cfg.get("rocm", {}).get("amd_gpu_arch", "gfx1031"))
        override = str(wl.get("hsa_override_gfx_version", "") or "").strip()
        if override and "HSA_OVERRIDE_GFX_VERSION" not in run_env:
            run_env["HSA_OVERRIDE_GFX_VERSION"] = override

    script = _gpu_required_script(kind)

    def run_one(sampler):
        t0 = time.monotonic()
        r = run_cmd(ctx.repo_root, run_env, [sys.executable, "-u", "-X", "faulthandler", "-c", script], t, log)
        wall_s = time.monotonic() - t0
        return r, wall_s, sampler

    r, wall_s, sampler = with_power_sampler(cfg, build_dir=build_dir, fn=run_one)
    out = (r.out + "\n" + r.err).strip()

    if r.rc != 0:
        if "No module named 'torch'" in out:
            return StepResult(build_dir, f"PyTorch ({kind})", "SKIP", fmt_duration(r.dur_ms), "torch not installed (set workloads.pytorch.auto_install=true or install ROCm torch)")
        if "GPU_NOT_AVAILABLE" in out:
            return StepResult(build_dir, f"PyTorch ({kind})", "FAIL", fmt_duration(r.dur_ms), "torch.cuda.is_available=false (GPU required)")
        if "GPU_NOT_ROCM" in out:
            return StepResult(build_dir, f"PyTorch ({kind})", "FAIL", fmt_duration(r.dur_ms), "torch.version.hip missing (ROCm torch required)")
        if "HIP_VERSION_MISMATCH" in out:
            m = re.search(r"^HIP_VERSION_MISMATCH\s+(\S+)\s+(.+)$", out, re.MULTILINE)
            if m:
                return StepResult(
                    build_dir,
                    f"PyTorch ({kind})",
                    "FAIL",
                    fmt_duration(r.dur_ms),
                    f"torch.version.hip mismatch: have={m.group(2).strip()} expected~={m.group(1).strip()}",
                )
            return StepResult(build_dir, f"PyTorch ({kind})", "FAIL", fmt_duration(r.dur_ms), "torch.version.hip mismatch")
        if "ROCM_VERSION_MISMATCH" in out:
            m = re.search(r"^ROCM_VERSION_MISMATCH\s+(\S+)\s+(.+)$", out, re.MULTILINE)
            if m:
                return StepResult(
                    build_dir,
                    f"PyTorch ({kind})",
                    "FAIL",
                    fmt_duration(r.dur_ms),
                    f"torch.version.rocm mismatch: have={m.group(2).strip()} expected~={m.group(1).strip()}",
                )
            return StepResult(build_dir, f"PyTorch ({kind})", "FAIL", fmt_duration(r.dur_ms), "torch.version.rocm mismatch")
        if "TORCHAUDIO_IMPORT_FAILED" in out:
            m = re.search(r"^TORCHAUDIO_IMPORT_FAILED\s+(.+)$", out, re.MULTILINE)
            detail = m.group(1).strip() if m else "torchaudio import failed"
            return StepResult(build_dir, f"PyTorch ({kind})", "FAIL", fmt_duration(r.dur_ms), detail)
        if "TORCHCODEC_IMPORT_FAILED" in out:
            m = re.search(r"^TORCHCODEC_IMPORT_FAILED\s+(.+)$", out, re.MULTILINE)
            detail = m.group(1).strip() if m else "torchcodec import failed"
            return StepResult(build_dir, f"PyTorch ({kind})", "FAIL", fmt_duration(r.dur_ms), detail)
        if r.rc < 0:
            return StepResult(build_dir, f"PyTorch ({kind})", "FAIL", fmt_duration(r.dur_ms), f"signal={-r.rc} (crash) | try workloads.pytorch.use_in_tree_rocm=false")
        return StepResult(build_dir, f"PyTorch ({kind})", "FAIL", fmt_duration(r.dur_ms), f"rc={r.rc}")

    if "GPU_OK" not in out:
        return StepResult(build_dir, f"PyTorch ({kind})", "FAIL", fmt_duration(r.dur_ms), "GPU validation marker missing")

    # Extract a small metric summary.
    device = ""
    tflops = ""
    m = re.search(r"^device_name\s+(.+)$", out, re.MULTILINE)
    if m:
        device = m.group(1).strip()
    m = re.search(r"^tflops_est\s+([0-9.]+)$", out, re.MULTILINE)
    if m:
        tflops = m.group(1)
    metric = f"device={device} rocm_env={'in-tree' if use_in_tree else 'system'} wall={wall_s:.2f}s"
    if not use_in_tree and run_env.get("HSA_OVERRIDE_GFX_VERSION"):
        metric += f" hsa_override={run_env['HSA_OVERRIDE_GFX_VERSION']}"
    if tflops:
        metric += f" tflops_est={float(tflops):.2f}"
    m = re.search(r"^torch\.version\.rocm\s+(.+)$", out, re.MULTILINE)
    if m:
        rocm_ver = m.group(1).strip()
        if rocm_ver and rocm_ver not in ("None", "null"):
            metric += f" rocm={rocm_ver}"
    m = re.search(r"^torchaudio\.version\s+(.+)$", out, re.MULTILINE)
    if m:
        metric += f" torchaudio={m.group(1).strip()}"
    m = re.search(r"^audio_backend\s+(.+)$", out, re.MULTILINE)
    if m:
        metric += f" audio_backend={m.group(1).strip()}"
    m = re.search(r"^samples_per_s\s+([0-9.]+)$", out, re.MULTILINE)
    if m:
        metric += f" samples_per_s={float(m.group(1)):.0f}"
    m = re.search(r"^torchcodec\.module\s+(.+)$", out, re.MULTILINE)
    if m:
        metric += f" torchcodec=ok"
    m = re.search(r"^shape\s+(.+)$", out, re.MULTILINE)
    if m:
        metric += f" shape={m.group(1).strip()}"
    m = re.search(r"^runs\s+(\S+)$", out, re.MULTILINE)
    runs = float(m.group(1)) if m else 0.0
    m = re.search(r"^seconds\s+(\S+)$", out, re.MULTILINE)
    secs = float(m.group(1)) if m else 0.0
    if runs > 0 and secs > 0:
        metric += f" it/s={runs/secs:.2f}"

    # Determine where ROCm runtime libraries came from.
    loaded = {}
    for line in out.splitlines():
        if not line.startswith("loaded "):
            continue
        parts = line.split(" ", 2)
        if len(parts) == 3:
            loaded[parts[1].strip()] = parts[2].strip()

    hip_lib = loaded.get("libamdhip64.so", "")
    if hip_lib:
        metric += f" hip_lib={hip_lib}"

    req = str(wl.get("require_rocm_prefix", "") or "").strip()
    if req:
        expected = str(rocm_dist) if req == "in-tree" else req.rstrip("/")
        if not hip_lib:
            return StepResult(build_dir, f"PyTorch ({kind})", "FAIL", fmt_duration(r.dur_ms), f"could not determine loaded ROCm runtime lib (libamdhip64) | expected prefix: {expected} | {metric}")
        if not hip_lib.startswith(expected + "/"):
            return StepResult(build_dir, f"PyTorch ({kind})", "FAIL", fmt_duration(r.dur_ms), f"ROCm runtime lib not from expected prefix: {expected} | hip_lib={hip_lib}")

    metric = append_power(metric, sampler, baseline_w=baseline_avg_w(cfg, build_dir))

    # Enforce that GPU use is visible in power/utilization.
    if sampler is not None:
        gpu = sampler.avg_gpu_busy()
        base_w = baseline_avg_w(cfg, build_dir) or 0.0
        avgw = sampler.avg_power_w() or 0.0
        if (gpu is not None and gpu < 10) and (avgw - base_w) < 10:
            return StepResult(build_dir, f"PyTorch ({kind})", "FAIL", fmt_duration(r.dur_ms), f"no clear GPU activity detected | {metric}")

    return StepResult(build_dir, f"PyTorch ({kind})", "OK", fmt_duration(r.dur_ms), metric)


def step_pytorch_audio(ctx: Context, cfg: dict[str, Any], build_dir: str, rocm_dist: Path, env: dict[str, str], log: Path | None) -> StepResult:
    return _step_pytorch_conv(ctx, cfg, build_dir, rocm_dist, env, log, kind="audio")


def step_pytorch_video(ctx: Context, cfg: dict[str, Any], build_dir: str, rocm_dist: Path, env: dict[str, str], log: Path | None) -> StepResult:
    return _step_pytorch_conv(ctx, cfg, build_dir, rocm_dist, env, log, kind="video")
