from __future__ import annotations

import os
import json
import sys
from pathlib import Path
from typing import Any
import glob

from core.context import Context
from core.reporting.models import StepResult
from core.rocm_env import activated_env
from core.runner import fmt_duration, run_cmd
from steps.shared import downloads_enabled


def _as_abs(ctx: Context, p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (ctx.repo_root / pp)


def with_openmp_runtime_env(env: dict[str, str]) -> dict[str, str]:
    """
    Some torch builds may require an OpenMP runtime that provides __kmpc_* symbols.

    On this repo's ROCm 7.11 setup we provide libomp under /opt/rocm. Preloading
    it is the most robust way to satisfy missing __kmpc_* at import time.
    """
    out = dict(env)
    cur = out.get("LD_PRELOAD", "")

    # Prefer ROCM_PATH if present; fall back to system /opt/rocm.
    rocm = (out.get("ROCM_PATH", "") or os.environ.get("ROCM_PATH", "") or "/opt/rocm").strip()
    cand = Path(rocm) / "lib" / "llvm" / "lib" / "libomp.so"
    if cand.is_file():
        if str(cand) not in cur.split(":"):
            out["LD_PRELOAD"] = ":".join([str(cand)] + ([cur] if cur else []))
    return out


def _probe_torch(ctx: Context, env: dict[str, str], log: Path | None) -> tuple[int, str, str, str]:
    env2 = env
    probe = run_cmd(
        ctx.repo_root,
        env2,
        [
            sys.executable,
            "-c",
            "import torch; "
            "print(getattr(torch,'__version__','')); "
            "v=getattr(torch,'version',None); "
            "print(getattr(v,'hip',None) or ''); "
            "print(getattr(v,'rocm',None) or '')",
        ],
        30,
        log,
    )
    if probe.rc != 0:
        out = ((probe.out or "") + "\n" + (probe.err or "")).strip()
        # Retry with libomp preload if we hit missing __kmpc_* symbols.
        if "__kmpc_" in out:
            env3 = with_openmp_runtime_env(env2)
            if env3.get("LD_PRELOAD") != env2.get("LD_PRELOAD"):
                probe = run_cmd(
                    ctx.repo_root,
                    env3,
                    [
                        sys.executable,
                        "-c",
                        "import torch; "
                        "print(getattr(torch,'__version__','')); "
                        "v=getattr(torch,'version',None); "
                        "print(getattr(v,'hip',None) or ''); "
                        "print(getattr(v,'rocm',None) or '')",
                    ],
                    30,
                    log,
                )
                if probe.rc == 0:
                    env2 = env3
                else:
                    return probe.rc, "", "", ""
        return probe.rc, "", "", ""
    lines = (probe.out or "").splitlines()
    ver = lines[0].strip() if len(lines) >= 1 else ""
    hip = lines[1].strip() if len(lines) >= 2 else ""
    rocm = lines[2].strip() if len(lines) >= 3 else ""
    return 0, ver, hip, rocm


def _probe_module(ctx: Context, env: dict[str, str], log: Path | None, module_name: str) -> int:
    probe = run_cmd(
        ctx.repo_root,
        env,
        [sys.executable, "-c", f"import {module_name}"],
        30,
        log,
    )
    return int(probe.rc or 0)


def _latest_wheel_in_dir(wheel_dir: Path, pattern: str) -> Path | None:
    wheels = sorted(wheel_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return wheels[0] if wheels else None


def _resolve_package_spec(ctx: Context, spec: str) -> str:
    marker = " @ file://"
    if marker not in spec:
        return spec
    name, raw_path = spec.split(marker, 1)
    path_text = raw_path.strip()
    if not path_text:
        return spec

    base = ctx.repo_root
    path_for_glob = path_text
    if path_text and not Path(path_text).is_absolute():
        path_for_glob = str(base / path_text)

    resolved_path: str | None = None
    if any(ch in path_for_glob for ch in "*?[]"):
        matches = [Path(p) for p in glob.glob(path_for_glob)]
        if matches:
            matches.sort(key=lambda p: (p.stat().st_mtime, str(p)))
            resolved_path = str(matches[-1].resolve())
    else:
        p = Path(path_for_glob)
        if p.exists():
            resolved_path = str(p.resolve())

    if not resolved_path:
        return spec
    return f"{name}{marker}{resolved_path}"


def _best_effort_uninstall(ctx: Context, env: dict[str, str], packages: list[str], timeout_s: int, log: Path | None) -> None:
    pkgs = [p.strip() for p in packages if str(p).strip()]
    if not pkgs:
        return
    cmd = [sys.executable, "-m", "pip", "uninstall", "-y"] + pkgs
    run_cmd(ctx.repo_root, env, cmd, timeout_s, log)


def _run_logged(
    cwd: Path, env: dict[str, str], cmd: list[str], timeout_s: int | None, log: Path | None
) -> tuple[int, int]:
    """
    Like `run_cmd`, but streams stdout/stderr to the log file to avoid storing
    huge build output in memory (important for PyTorch builds).
    """
    import shlex
    import subprocess
    import time

    start = time.time()
    if log is None:
        r = run_cmd(cwd, env, cmd, timeout_s, None)
        return r.rc, r.dur_ms

    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"$ {shlex.join(cmd)}\n")
        f.flush()
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except FileNotFoundError as e:
            dur_ms = int((time.time() - start) * 1000)
            f.write(f"{e}\n\n")
            return 127, dur_ms
        try:
            proc.wait(timeout=timeout_s)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            rc = 124
        finally:
            f.write("\n")
            f.flush()

    dur_ms = int((time.time() - start) * 1000)
    return int(rc or 0), dur_ms


def _ensure_pytorch_source_build_rocm_sdk(ctx: Context, cfg: dict[str, Any], env: dict[str, str], log: Path | None) -> StepResult:
    wl = cfg.get("workloads", {}).get("pytorch", {}) or {}
    sb = wl.get("source_build", {}) or {}

    if not downloads_enabled(cfg):
        # If torch is already installed, proceed. Otherwise, we cannot build without downloads.
        rc, ver, hip, rocm = _probe_torch(ctx, env, log)
        if rc == 0 and ver:
            extra = f" rocm={rocm}" if rocm else ""
            return StepResult(
                "<meta>",
                "PyTorch setup",
                "OK",
                "0ms",
                f"torch already installed (downloads disabled) torch={ver} hip={hip}{extra}",
            )
        return StepResult("<meta>", "PyTorch setup", "SKIP", "0ms", "downloads disabled (cannot build torch from source)")

    index_url = str(sb.get("index_url", "") or "").strip()
    if not index_url:
        return StepResult("<meta>", "PyTorch setup", "FAIL", "0ms", "source_build.index_url is required")

    rocm_sdk_version = str(sb.get("rocm_sdk_version", "") or "").strip()
    hashtag = str(sb.get("pytorch_repo_hashtag", "nightly") or "nightly").strip()
    gitrepo_origin = str(sb.get("gitrepo_origin", "") or "").strip()

    pytorch_dir = _as_abs(ctx, str(sb.get("pytorch_dir", ctx.git_cache_dir() / "pytorch_rocm711")))
    wheels_dir = _as_abs(ctx, str(sb.get("wheels_dir", ctx.cache_dir() / "wheels" / "pytorch_rocm711")))
    pip_cache_dir = _as_abs(ctx, str(sb.get("pip_cache_dir", ctx.cache_dir() / "pip")))

    update_checkout = bool(sb.get("update_checkout", False))
    depth = int(sb.get("depth", 0) or 0)
    use_ccache = bool(sb.get("use_ccache", True))
    clean = bool(sb.get("clean", False))
    build_torchaudio = bool(sb.get("build_torchaudio", False))
    pytorch_audio_dir = _as_abs(ctx, str(sb.get("pytorch_audio_dir", ctx.git_cache_dir() / "pytorch_audio_rocm711")))

    # Checkout/update sources (cached under validation/workspace).
    if update_checkout or not (pytorch_dir / ".git").exists():
        wheels_dir.mkdir(parents=True, exist_ok=True)
        pip_cache_dir.mkdir(parents=True, exist_ok=True)
        t_checkout = int(cfg.get("timeouts_s", {}).get("pytorch_checkout", 3600))
        checkout_script = ctx.repo_root / "external-builds" / "pytorch" / "pytorch_torch_repo.py"
        cmd = [
            sys.executable,
            str(checkout_script),
            "checkout",
            "--checkout-dir",
            str(pytorch_dir),
            "--repo-hashtag",
            hashtag,
        ]
        if gitrepo_origin:
            cmd += ["--gitrepo-origin", gitrepo_origin]
        if depth > 0:
            cmd += ["--depth", str(depth)]
        r = run_cmd(ctx.repo_root, env, cmd, t_checkout, log)
        if r.rc != 0:
            return StepResult("<meta>", "PyTorch setup", "FAIL", fmt_duration(r.dur_ms), f"checkout rc={r.rc}")
    if build_torchaudio and (update_checkout or not (pytorch_audio_dir / ".git").exists()):
        t_checkout = int(cfg.get("timeouts_s", {}).get("pytorch_checkout", 3600))
        checkout_audio_script = ctx.repo_root / "external-builds" / "pytorch" / "pytorch_audio_repo.py"
        audio_cmd = [
            sys.executable,
            str(checkout_audio_script),
            "checkout",
            "--checkout-dir",
            str(pytorch_audio_dir),
            "--torch-dir",
            str(pytorch_dir),
            "--patchset",
            "rocm-custom",
        ]
        if depth > 0:
            audio_cmd += ["--depth", str(depth)]
        r_audio = run_cmd(ctx.repo_root, env, audio_cmd, t_checkout, log)
        if r_audio.rc != 0:
            return StepResult("<meta>", "PyTorch setup", "FAIL", fmt_duration(r_audio.dur_ms), f"torchaudio checkout rc={r_audio.rc}")

    # Build + install into the current venv via TheRock tooling.
    build_script = ctx.repo_root / "external-builds" / "pytorch" / "build_prod_wheels.py"
    arch = str(cfg.get("rocm", {}).get("amd_gpu_arch", "gfx1031"))
    t_build = int(cfg.get("timeouts_s", {}).get("pytorch_build", 21600))

    cmd = [
        sys.executable,
        str(build_script),
        "build",
        "--install-rocm",
        "--index-url",
        index_url,
        "--output-dir",
        str(wheels_dir),
        "--pip-cache-dir",
        str(pip_cache_dir),
        "--pytorch-dir",
        str(pytorch_dir),
        "--pytorch-rocm-arch",
        arch,
        "--no-build-triton",
        "--no-build-pytorch-vision",
    ]
    if build_torchaudio:
        cmd += ["--pytorch-audio-dir", str(pytorch_audio_dir)]
    else:
        cmd += ["--no-build-pytorch-audio"]
    if rocm_sdk_version:
        cmd += ["--rocm-sdk-version", rocm_sdk_version]
    if use_ccache:
        cmd += ["--use-ccache"]
    if clean:
        cmd += ["--clean"]

    r = run_cmd(ctx.repo_root, env, cmd, t_build, log)
    if r.rc != 0:
        return StepResult("<meta>", "PyTorch setup", "FAIL", fmt_duration(r.dur_ms), f"build rc={r.rc}")

    # Probe the resulting install and store a small build marker for reuse/debug.
    rc, ver, hip, rocm = _probe_torch(ctx, env, log)
    if rc != 0 or not ver:
        return StepResult("<meta>", "PyTorch setup", "FAIL", fmt_duration(r.dur_ms), "torch import failed after source build")
    if build_torchaudio:
        audio_wheel = _latest_wheel_in_dir(wheels_dir, "torchaudio-*.whl")
        if audio_wheel is None:
            return StepResult("<meta>", "PyTorch setup", "FAIL", fmt_duration(r.dur_ms), f"torchaudio build requested but no wheel in {wheels_dir}")
        r_audio_install = run_cmd(
            ctx.repo_root,
            env,
            [sys.executable, "-m", "pip", "install", "-I", "--no-deps", str(audio_wheel)],
            3600,
            log,
        )
        if r_audio_install.rc != 0:
            return StepResult("<meta>", "PyTorch setup", "FAIL", fmt_duration(r_audio_install.dur_ms), f"torchaudio install rc={r_audio_install.rc}")
        if _probe_module(ctx, env, log, "torchaudio") != 0:
            return StepResult("<meta>", "PyTorch setup", "FAIL", fmt_duration(r_audio_install.dur_ms), "torchaudio import failed after source build")

    marker = {
        "index_url": index_url,
        "rocm_sdk_version": rocm_sdk_version,
        "pytorch_repo_hashtag": hashtag,
        "pytorch_dir": str(pytorch_dir),
        "wheels_dir": str(wheels_dir),
        "torch_version": ver,
        "torch_hip_version": hip,
        "torch_rocm_version": rocm,
    }
    try:
        (wheels_dir / "BUILD_INFO.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True), encoding="utf-8"
        )
    except Exception:
        pass

    return StepResult(
        "<meta>",
        "PyTorch setup",
        "OK",
        fmt_duration(r.dur_ms),
        f"source build torch={ver} hip={hip} rocm={rocm}",
    )


def _ensure_pytorch_source_build_in_tree(
    ctx: Context,
    cfg: dict[str, Any],
    env: dict[str, str],
    log: Path | None,
    *,
    rocm_dist: Path,
) -> StepResult:
    wl = cfg.get("workloads", {}).get("pytorch", {}) or {}
    sb = wl.get("source_build", {}) or {}

    if not rocm_dist.is_dir():
        return StepResult("<meta>", "PyTorch setup", "FAIL", "0ms", f"ROCm dist not found: {rocm_dist}")

    if not downloads_enabled(cfg):
        # If torch is already installed, proceed. Otherwise, we cannot build without downloads.
        rc, ver, hip, rocm = _probe_torch(ctx, env, log)
        if rc == 0 and ver:
            extra = f" rocm={rocm}" if rocm else ""
            return StepResult(
                "<meta>",
                "PyTorch setup",
                "OK",
                "0ms",
                f"torch already installed (downloads disabled) torch={ver} hip={hip}{extra}",
            )
        return StepResult("<meta>", "PyTorch setup", "SKIP", "0ms", "downloads disabled (cannot build torch from source)")

    hashtag = str(sb.get("pytorch_repo_hashtag", "nightly") or "nightly").strip()
    gitrepo_origin = str(sb.get("gitrepo_origin", "") or "").strip()
    pytorch_dir = _as_abs(ctx, str(sb.get("pytorch_dir", ctx.git_cache_dir() / "pytorch_in_tree")))
    wheels_dir = _as_abs(ctx, str(sb.get("wheels_dir", ctx.cache_dir() / "wheels" / "pytorch_in_tree")))
    pip_cache_dir = _as_abs(ctx, str(sb.get("pip_cache_dir", ctx.cache_dir() / "pip")))

    update_checkout = bool(sb.get("update_checkout", False))
    depth = int(sb.get("depth", 0) or 0)
    use_ccache = bool(sb.get("use_ccache", True))
    clean = bool(sb.get("clean", False))
    build_torchaudio = bool(sb.get("build_torchaudio", False))
    pytorch_audio_dir = _as_abs(ctx, str(sb.get("pytorch_audio_dir", ctx.git_cache_dir() / "pytorch_audio_rocm711")))

    wheels_dir.mkdir(parents=True, exist_ok=True)
    pip_cache_dir.mkdir(parents=True, exist_ok=True)

    # Checkout/update sources (cached under validation/workspace).
    if update_checkout or not (pytorch_dir / ".git").exists():
        t_checkout = int(cfg.get("timeouts_s", {}).get("pytorch_checkout", 3600))
        checkout_script = ctx.repo_root / "external-builds" / "pytorch" / "pytorch_torch_repo.py"
        cmd = [
            sys.executable,
            str(checkout_script),
            "checkout",
            "--checkout-dir",
            str(pytorch_dir),
            "--repo-hashtag",
            hashtag,
            "--no-patch",
        ]
        if gitrepo_origin:
            cmd += ["--gitrepo-origin", gitrepo_origin]
        if depth > 0:
            cmd += ["--depth", str(depth)]
        rc, dur_ms = _run_logged(ctx.repo_root, env, cmd, t_checkout, log)
        if rc != 0:
            return StepResult("<meta>", "PyTorch setup", "FAIL", fmt_duration(dur_ms), f"checkout rc={rc}")
    if build_torchaudio and (update_checkout or not (pytorch_audio_dir / ".git").exists()):
        t_checkout = int(cfg.get("timeouts_s", {}).get("pytorch_checkout", 3600))
        checkout_audio_script = ctx.repo_root / "external-builds" / "pytorch" / "pytorch_audio_repo.py"
        audio_cmd = [
            sys.executable,
            str(checkout_audio_script),
            "checkout",
            "--checkout-dir",
            str(pytorch_audio_dir),
            "--torch-dir",
            str(pytorch_dir),
            "--patchset",
            "rocm-custom",
        ]
        if depth > 0:
            audio_cmd += ["--depth", str(depth)]
        rc_audio, dur_ms_audio = _run_logged(ctx.repo_root, env, audio_cmd, t_checkout, log)
        if rc_audio != 0:
            return StepResult("<meta>", "PyTorch setup", "FAIL", fmt_duration(dur_ms_audio), f"torchaudio checkout rc={rc_audio}")

    build_env = activated_env(ctx.env_base(), rocm_dist)
    build_env.update(env)
    build_env["ROCM_HOME"] = str(rocm_dist)
    build_env["ROCM_PATH"] = str(rocm_dist)
    build_env["HIP_PATH"] = str(rocm_dist)
    build_env["HSA_PATH"] = str(rocm_dist)
    build_env["CMAKE_PREFIX_PATH"] = str(rocm_dist)
    build_env.setdefault("CC", "clang")
    build_env.setdefault("CXX", "clang++")

    build_env.setdefault("USE_CUDA", "0")
    build_env.setdefault("USE_ROCM", "1")
    build_env.setdefault("USE_ROCM_HIPBLASLT", "0")
    build_env.setdefault("USE_NCCL", "0")
    build_env.setdefault("USE_RCCL", "0")
    build_env.setdefault("USE_MPI", "0")
    build_env.setdefault("USE_NUMA", "0")
    build_env.setdefault("BUILD_TEST", "0")
    build_env.setdefault("USE_NINJA", "1")

    # WORKAROUND: The new HIP CLR build (therock-7.13) doesn't link
    # libamdhip64.so against librocm_smi64.so.  PyTorch's libtorch_hip.so
    # references rsmi_init but can't resolve it through the NEEDED chain.
    # Preload the SMI library so the symbol is available at dlopen time.
    rsmi_lib = rocm_dist / "lib" / "librocm_smi64.so"
    if rsmi_lib.is_file():
        existing = build_env.get("LD_PRELOAD", "")
        build_env["LD_PRELOAD"] = f"{rsmi_lib}:{existing}" if existing else str(rsmi_lib)

    arch = str(cfg.get("rocm", {}).get("amd_gpu_arch", "gfx1031"))
    build_env["PYTORCH_ROCM_ARCH"] = arch

    # PyTorch's ROCm toolchain currently injects `-fclang-abi-compat=17` into
    # HIP compilation units to avoid ABI/mangling mismatches. When the host C++
    # code is built with clang++ (as we do for this repo), we must mirror that
    # flag for the host compilation too, otherwise we can end up with runtime
    # link errors such as:
    #   libtorch_hip.so: undefined symbol: ...TensorBase::const_data_ptr<Half>()
    cxx = build_env.get("CXXFLAGS", "")
    if "-fclang-abi-compat=17" not in cxx.split():
        build_env["CXXFLAGS"] = f"{cxx} -fclang-abi-compat=17".strip()

    # Sysdeps include path (prevents missing libdrm headers in some builds).
    sysdeps_dir = rocm_dist / "lib" / "rocm_sysdeps"
    if sysdeps_dir.is_dir():
        cxx = build_env.get("CXXFLAGS", "")
        build_env["CXXFLAGS"] = (
            f"{cxx} -I{sysdeps_dir / 'include'} -I{rocm_dist / 'include' / 'roctracer'}"
        ).strip()
        ld = build_env.get("LDFLAGS", "")
        build_env["LDFLAGS"] = f"{ld} -L{sysdeps_dir / 'lib'}".strip()
        build_env.setdefault("PKG_CONFIG_PATH", str(sysdeps_dir / "lib" / "pkgconfig"))

    if use_ccache:
        build_env.setdefault("CMAKE_C_COMPILER_LAUNCHER", "ccache")
        build_env.setdefault("CMAKE_CXX_COMPILER_LAUNCHER", "ccache")

    # Uninstall any existing torch to avoid mixing files.
    run_cmd(ctx.repo_root, build_env, [sys.executable, "-m", "pip", "uninstall", "torch", "-y"], 300, log)

    # Install Python requirements for the checked out PyTorch tree.
    t_req = int(cfg.get("timeouts_s", {}).get("pytorch_install", 1800))
    req = pytorch_dir / "requirements.txt"
    if req.is_file():
        cmd = [sys.executable, "-m", "pip", "install"]
        if pip_cache_dir:
            cmd += ["--cache-dir", str(pip_cache_dir)]
        cmd += ["-r", str(req)]
        r = run_cmd(pytorch_dir, build_env, cmd, t_req, log)
        if r.rc != 0:
            return StepResult("<meta>", "PyTorch setup", "FAIL", fmt_duration(r.dur_ms), f"requirements rc={r.rc}")

    # Optional: MKL headers (best-effort).
    run_cmd(ctx.repo_root, build_env, [sys.executable, "-m", "pip", "install", "mkl-static", "mkl-include"], t_req, log)

    if clean:
        import shutil

        for d in (pytorch_dir / "build", pytorch_dir / "dist"):
            if d.is_dir():
                shutil.rmtree(d)

    t_build = int(cfg.get("timeouts_s", {}).get("pytorch_build", 21600))
    rc, dur_ms = _run_logged(pytorch_dir, build_env, [sys.executable, "setup.py", "bdist_wheel"], t_build, log)
    if rc != 0:
        return StepResult("<meta>", "PyTorch setup", "FAIL", fmt_duration(dur_ms), f"build rc={rc}")

    dist_dir = pytorch_dir / "dist"
    wheels = sorted(dist_dir.glob("torch-*.whl"))
    if not wheels:
        return StepResult("<meta>", "PyTorch setup", "FAIL", fmt_duration(dur_ms), f"no torch wheel found under {dist_dir}")
    wheel = wheels[-1]
    try:
        import shutil

        dst = wheels_dir / wheel.name
        shutil.copyfile(wheel, dst)
    except Exception:
        pass

    r2 = run_cmd(ctx.repo_root, build_env, [sys.executable, "-m", "pip", "install", "-I", "--no-deps", str(wheel)], 3600, log)
    if r2.rc != 0:
        return StepResult("<meta>", "PyTorch setup", "FAIL", fmt_duration(r2.dur_ms), f"install rc={r2.rc}")
    extra_dur_ms = 0
    if build_torchaudio:
        audio_env = dict(build_env)
        audio_env.update(
            {
                "USE_ROCM": "1",
                "USE_CUDA": "0",
                "USE_FFMPEG": "0",
                "USE_OPENMP": "1",
                "BUILD_SOX": "0",
            }
        )
        rc_audio_build, dur_ms_audio_build = _run_logged(
            pytorch_audio_dir,
            audio_env,
            [sys.executable, "setup.py", "bdist_wheel"],
            t_build,
            log,
        )
        extra_dur_ms += dur_ms_audio_build
        if rc_audio_build != 0:
            return StepResult("<meta>", "PyTorch setup", "FAIL", fmt_duration(dur_ms + r2.dur_ms + extra_dur_ms), f"torchaudio build rc={rc_audio_build}")
        audio_wheel = _latest_wheel_in_dir(pytorch_audio_dir / "dist", "torchaudio-*.whl")
        if audio_wheel is None:
            return StepResult("<meta>", "PyTorch setup", "FAIL", fmt_duration(dur_ms + r2.dur_ms + extra_dur_ms), f"no torchaudio wheel found under {pytorch_audio_dir / 'dist'}")
        try:
            import shutil

            dst_audio = wheels_dir / audio_wheel.name
            shutil.copyfile(audio_wheel, dst_audio)
        except Exception:
            pass
        r_audio_install = run_cmd(
            ctx.repo_root,
            build_env,
            [sys.executable, "-m", "pip", "install", "-I", "--no-deps", str(audio_wheel)],
            3600,
            log,
        )
        extra_dur_ms += r_audio_install.dur_ms
        if r_audio_install.rc != 0:
            return StepResult("<meta>", "PyTorch setup", "FAIL", fmt_duration(dur_ms + r2.dur_ms + extra_dur_ms), f"torchaudio install rc={r_audio_install.rc}")
        if _probe_module(ctx, build_env, log, "torchaudio") != 0:
            return StepResult("<meta>", "PyTorch setup", "FAIL", fmt_duration(dur_ms + r2.dur_ms + extra_dur_ms), "torchaudio import failed after source build")

    rc2, ver, hip, rocm = _probe_torch(ctx, build_env, log)
    if rc2 != 0 or not ver:
        return StepResult("<meta>", "PyTorch setup", "FAIL", fmt_duration(r2.dur_ms), "torch import failed after in-tree source build")

    marker = {
        "backend": "in-tree",
        "rocm_dist": str(rocm_dist),
        "pytorch_repo_hashtag": hashtag,
        "pytorch_dir": str(pytorch_dir),
        "wheels_dir": str(wheels_dir),
        "wheel": str(wheel),
        "torch_version": ver,
        "torch_hip_version": hip,
        "torch_rocm_version": rocm,
        "torchaudio_built": build_torchaudio,
    }
    try:
        (wheels_dir / "BUILD_INFO.json").write_text(json.dumps(marker, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        pass

    metric = f"in-tree source build torch={ver} hip={hip}"
    if rocm:
        metric += f" rocm={rocm}"
    if build_torchaudio:
        metric += " torchaudio=ok"
    return StepResult("<meta>", "PyTorch setup", "OK", fmt_duration(dur_ms + r2.dur_ms + extra_dur_ms), metric)


def ensure_pytorch(
    ctx: Context, cfg: dict[str, Any], env: dict[str, str], log: Path | None, *, rocm_dist: Path | None = None
) -> StepResult | None:
    """
    Ensure a ROCm-enabled PyTorch is available in the validation venv.

    By default we do NOT auto-install torch because ROCm wheels are large and
    version/index dependent. If you want auto-install, set:
      workloads.pytorch.auto_install: true
    and optionally configure:
      workloads.pytorch.pip_args: [...]
      workloads.pytorch.packages: [...]
    """
    wl = cfg.get("workloads", {}).get("pytorch", {}) or {}
    auto = bool(wl.get("auto_install", False))
    if not auto:
        return None

    # Ensure the env can import torch even if the wheel needs libomp for __kmpc_*.
    env = with_openmp_runtime_env(env)

    sb = wl.get("source_build", {}) or {}
    if bool(sb.get("enabled", False)):
        backend = str(sb.get("backend", "rocm-sdk") or "rocm-sdk").strip().lower()
        if backend == "in-tree":
            if rocm_dist is None:
                return StepResult("<meta>", "PyTorch setup", "FAIL", "0ms", "source_build.backend=in-tree requires rocm_dist")
            # If torch is already installed and matches the requested version
            # constraints, avoid rebuilding (PyTorch source builds can take a
            # long time). Users can force a rebuild by setting
            # `workloads.pytorch.force_reinstall=true`.
            force = bool(wl.get("force_reinstall", False))
            expected_ver = str(wl.get("expected_version_substr", "") or "").strip()
            expected_hip = str(wl.get("expected_hip_substr", "") or "").strip()
            expected_rocm = str(wl.get("expected_rocm_substr", "") or "").strip()
            require_torchaudio = bool(wl.get("require_torchaudio", False))
            if not force:
                probe_rc, installed_ver, installed_hip, installed_rocm = _probe_torch(ctx, env, log)
                if probe_rc == 0 and installed_ver:
                    torchaudio_missing = require_torchaudio and _probe_module(ctx, env, log, "torchaudio") != 0
                    if (
                        not torchaudio_missing
                        and (not expected_ver or expected_ver in installed_ver)
                        and (not expected_hip or expected_hip in installed_hip)
                        and (not expected_rocm or expected_rocm in installed_rocm)
                        and (installed_hip or installed_rocm)
                    ):
                        return None
            return _ensure_pytorch_source_build_in_tree(ctx, cfg, env, log, rocm_dist=rocm_dist)
        return _ensure_pytorch_source_build_rocm_sdk(ctx, cfg, env, log)
    force = bool(wl.get("force_reinstall", False))
    expected = str(wl.get("expected_version_substr", "") or "").strip()
    expected_hip = str(wl.get("expected_hip_substr", "") or "").strip()
    expected_rocm = str(wl.get("expected_rocm_substr", "") or "").strip()

    # If torch is already installed in the validation venv, we can proceed even
    # when downloads are disabled.
    probe_rc, installed_ver, installed_hip, installed_rocm = _probe_torch(ctx, env, log)
    # If torch cannot be imported (rc != 0), treat it as "broken install" and
    # force an overwrite install. Without this, pip may think requirements are
    # satisfied and refuse to fix a corrupted environment.
    need_reinstall = force or (probe_rc != 0)
    if probe_rc == 0 and not force:
        if expected and expected not in installed_ver:
            # Installed torch does not match the requested wheel channel/version tag.
            # If downloads are disabled, proceed but warn (the step may still fail).
            if not downloads_enabled(cfg):
                return StepResult(
                    "<meta>",
                    "PyTorch setup",
                    "OK",
                    "0ms",
                    f"torch already installed but version mismatch: have={installed_ver} expected~={expected} (downloads disabled; proceeding)",
                )
            need_reinstall = True
        if expected_hip and expected_hip not in installed_hip:
            if not downloads_enabled(cfg):
                return StepResult(
                    "<meta>",
                    "PyTorch setup",
                    "OK",
                    "0ms",
                    f"torch already installed but HIP version mismatch: have={installed_hip} expected~={expected_hip} (downloads disabled; proceeding)",
                )
            need_reinstall = True
        if expected_rocm and expected_rocm not in installed_rocm:
            if not downloads_enabled(cfg):
                return StepResult(
                    "<meta>",
                    "PyTorch setup",
                    "OK",
                    "0ms",
                    f"torch already installed but ROCm version mismatch: have={installed_rocm} expected~={expected_rocm} (downloads disabled; proceeding)",
                )
            need_reinstall = True
        else:
            return None

    if not downloads_enabled(cfg):
        return StepResult("<meta>", "PyTorch setup", "SKIP", "0ms", "downloads disabled (cannot install torch)")

    t = int(cfg.get("timeouts_s", {}).get("pytorch_install", 1800))
    uninstall_packages = list(wl.get("uninstall_packages", []) or [])
    if uninstall_packages:
        _best_effort_uninstall(ctx, env, uninstall_packages, t, log)
    pip_args = list(wl.get("pip_args", []) or [])
    packages = list(wl.get("packages", []) or [])
    if not packages:
        packages = ["torch", "torchvision"]
    packages = [_resolve_package_spec(ctx, p) for p in packages]

    # `pip_install` doesn't support extra args; call pip directly for flexibility.
    extra_flags: list[str] = []
    if need_reinstall:
        # Make the result deterministic even if a different torch build is
        # already installed in the validation venv.
        if "--ignore-installed" not in pip_args and "-I" not in pip_args:
            extra_flags += ["-I"]
        if "--no-cache-dir" not in pip_args:
            extra_flags += ["--no-cache-dir"]
    cmd = [sys.executable, "-m", "pip", "install"] + extra_flags + pip_args + packages
    r = run_cmd(ctx.repo_root, env, cmd, t, log)
    if r.rc != 0:
        return StepResult("<meta>", "PyTorch setup", "FAIL", fmt_duration(r.dur_ms), f"pip rc={r.rc}")
    probe_rc, installed_ver, installed_hip, installed_rocm = _probe_torch(ctx, env, log)
    if probe_rc != 0 or not installed_ver:
        return StepResult("<meta>", "PyTorch setup", "FAIL", fmt_duration(r.dur_ms), "torch import failed after pip install")
    metric = f"pip install torch={installed_ver} hip={installed_hip}"
    if installed_rocm:
        metric += f" rocm={installed_rocm}"
    return StepResult("<meta>", "PyTorch setup", "OK", fmt_duration(r.dur_ms), metric)
