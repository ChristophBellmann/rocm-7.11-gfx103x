from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from core.context import Context
from core.runner import CommandResult
from core.reporting.models import StepResult
from core.rocm_env import which
from core.runner import fmt_duration, run_cmd
from steps.workloads.mfem.fetch import ensure_mfem_source
from steps.shared import append_power, baseline_avg_w, with_power_sampler


def step_mfem_hip(ctx: Context, cfg: dict[str, Any], build_dir: str, rocm_dist: Path, env: dict[str, str], log: Path | None) -> StepResult:
    if which("cmake", env) is None or which("ninja", env) is None:
        return StepResult(build_dir, "MFEM (HIP) build+run", "SKIP", "0ms", "missing cmake/ninja (install system packages)")
    if which("hipcc", env) is None:
        return StepResult(build_dir, "MFEM (HIP) build+run", "SKIP", "0ms", "hipcc not in PATH (need Stage-2 dist)")

    src, meta = ensure_mfem_source(ctx, cfg, env, log)
    if meta is not None:
        return StepResult(build_dir, "MFEM (HIP) build+run", meta.status, meta.duration, meta.metric)
    if src is None:
        return StepResult(build_dir, "MFEM (HIP) build+run", "FAIL", "0ms", "MFEM source unavailable")

    t = int(cfg.get("timeouts_s", {}).get("mfem_hip", 3600))
    # Keep MFEM source-of-truth in git cache and use a separate build dir.
    # Older validation states used builds/mfem as a source+build tree, which was redundant.
    bld = ctx.builds_dir() / "mfem_build"
    bld.mkdir(parents=True, exist_ok=True)

    hipcc = which("hipcc", env) or "hipcc"
    clangxx = str(rocm_dist / "llvm" / "bin" / "clang++")
    arch = str(cfg.get("rocm", {}).get("amd_gpu_arch", "gfx1031"))

    cfg_cmd = [
        "cmake",
        "-S",
        str(src),
        "-B",
        str(bld),
        "-G",
        "Ninja",
        "-DMFEM_USE_HIP=YES",
        f"-DHIP_ARCH={arch}",
        # The in-tree HIP CMake package derives HIP_PLATFORM via hipconfig, but some
        # external projects end up with an unset/empty value during early configure.
        # Pin it to keep MFEM reproducible against the in-tree ROCm dist.
        "-DHIP_PLATFORM=amd",
        f"-DCMAKE_CXX_COMPILER={clangxx}",
        f"-DCMAKE_HIP_COMPILER={hipcc}",
    ]
    r1 = run_cmd(ctx.repo_root, env, cfg_cmd, 600, log)
    if r1.rc != 0:
        return StepResult(build_dir, "MFEM (HIP) build+run", "FAIL", fmt_duration(r1.dur_ms), f"cmake rc={r1.rc}")
    r2 = run_cmd(ctx.repo_root, env, ["ninja", "-C", str(bld), "-j4"], t, log)
    if r2.rc != 0:
        return StepResult(build_dir, "MFEM (HIP) build+run", "FAIL", fmt_duration(r1.dur_ms + r2.dur_ms), f"ninja rc={r2.rc}")

    exe = bld / "examples" / "ex1"
    if not exe.exists():
        exe = bld / "bin" / "ex1"
    if not exe.exists():
        # Examples are often excluded from the default "all" target unless explicitly enabled.
        # Build just ex1 to keep this step reproducible and lightweight.
        r2b = run_cmd(ctx.repo_root, env, ["ninja", "-C", str(bld), "-j4", "ex1"], t, log)
        if r2b.rc != 0:
            return StepResult(
                build_dir,
                "MFEM (HIP) build+run",
                "FAIL",
                fmt_duration(r1.dur_ms + r2.dur_ms + r2b.dur_ms),
                f"ninja ex1 rc={r2b.rc}",
            )
        exe = bld / "examples" / "ex1"
        if not exe.exists():
            exe = bld / "bin" / "ex1"
    if not exe.exists():
        return StepResult(build_dir, "MFEM (HIP) build+run", "FAIL", fmt_duration(r1.dur_ms + r2.dur_ms), "MFEM executable not found after build")

    wl = cfg.get("workloads", {}).get("mfem", {}) or {}
    mesh_name = str(wl.get("mesh", "") or "").strip() or "fichera-amr.mesh"
    # Prefer a larger mesh to make GPU load/power more visible, but keep a safe fallback.
    mesh_candidates = [mesh_name, "fichera.mesh", "beam-hex.mesh", "star.mesh"]
    mesh = None
    for name in mesh_candidates:
        m = src / "data" / name
        if m.is_file():
            mesh = m
            break
    if mesh is None:
        return StepResult(build_dir, "MFEM (HIP) build+run", "FAIL", fmt_duration(r1.dur_ms + r2.dur_ms), "MFEM mesh not found")

    # Prefer a realistic FEM-style GPU workload: repeated operator applies for ~min_bench_s.
    # This produces a more sustained/meaningful GPU signal than short example runs.
    min_run_s = float(cfg.get("workloads", {}).get("mfem", {}).get("min_bench_s", 5.0) or 5.0)
    order = int(wl.get("order", 3) or 3)
    refine = int(wl.get("refine", -1) or -1)
    use_pa = int(wl.get("pa", 1) or 1)

    def parse_kv(text: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or " " not in line:
                continue
            k, v = line.split(" ", 1)
            out[k.strip()] = v.strip()
        return out

    if os.environ.get("ROCM_VALIDATION_MFEM_EXAMPLE_ONLY", "") != "1":
        app_src = ctx.builds_dir() / "mfem_apply"
        app_src.mkdir(parents=True, exist_ok=True)
        app_cpp = app_src / "mfem_apply.cpp"
        app_cmake = app_src / "CMakeLists.txt"
        app_bld = ctx.builds_dir() / "mfem_apply_build"
        app_bld.mkdir(parents=True, exist_ok=True)
        app_exe = app_bld / "mfem_apply"

        app_cmake.write_text(
            (
                "cmake_minimum_required(VERSION 3.22)\n"
                "project(mfem_apply CXX)\n"
                "set(CMAKE_CXX_STANDARD 17)\n"
                "set(CMAKE_CXX_STANDARD_REQUIRED ON)\n"
                "find_package(MFEM REQUIRED CONFIG PATHS \"${MFEM_DIR}\" NO_DEFAULT_PATH)\n"
                "add_executable(mfem_apply mfem_apply.cpp)\n"
                "target_link_libraries(mfem_apply PRIVATE mfem)\n"
            ),
            encoding="utf-8",
        )
        app_cpp.write_text(
            r"""
#include "mfem.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include <hip/hip_runtime.h>

static int env_int(const char *k, int defv)
{
  const char *v = std::getenv(k);
  if (!v || !*v) { return defv; }
  try { return std::stoi(v); } catch (...) { return defv; }
}

static double env_double(const char *k, double defv)
{
  const char *v = std::getenv(k);
  if (!v || !*v) { return defv; }
  try { return std::stod(v); } catch (...) { return defv; }
}

int main(int argc, char **argv)
{
  mfem::Device device("hip");
  if (argc < 2)
  {
    std::cerr << "usage: mfem_apply <mesh>\n";
    return 2;
  }

  const std::string mesh_path = argv[1];
  const int order = env_int("ROCM_VALIDATION_MFEM_ORDER", 3);
  const int refine_in = env_int("ROCM_VALIDATION_MFEM_REFINE", -1);
  const int use_pa = env_int("ROCM_VALIDATION_MFEM_PA", 1);
  const double min_s = env_double("ROCM_VALIDATION_MFEM_MIN_S", 5.0);

  mfem::Mesh mesh(mesh_path.c_str(), 1, 1);
  const int dim = mesh.Dimension();
  int refine = refine_in;
  if (refine < 0)
  {
    // Similar to MFEM examples/ex1: refine to a mesh with no more than ~50k elements.
    const double ne = (double)mesh.GetNE();
    const double target = 50000.0;
    if (ne > 0 && ne < target)
    {
      refine = (int)std::floor(std::log(target / ne) / std::log(2.0) / (double)dim);
      if (refine < 0) { refine = 0; }
    }
    else
    {
      refine = 0;
    }
  }
  for (int i = 0; i < refine; ++i) { mesh.UniformRefinement(); }

  mfem::H1_FECollection fec(order, dim);
  mfem::FiniteElementSpace fes(&mesh, &fec);
  mfem::Array<int> ess_tdof_list; // none

  mfem::LinearForm b(&fes);
  mfem::ConstantCoefficient one(1.0);
  b.AddDomainIntegrator(new mfem::DomainLFIntegrator(one));
  b.Assemble();

  mfem::GridFunction x(&fes);
  x = 0.0;

  mfem::BilinearForm a(&fes);
  if (use_pa) { a.SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL); }
  a.AddDomainIntegrator(new mfem::DiffusionIntegrator(one));
  a.Assemble();

  mfem::OperatorPtr A;
  mfem::Vector B, X;
  a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

  mfem::Vector Y(X.Size());
  X.Randomize(1);

  // Warmup
  A->Mult(X, Y);
  hipDeviceSynchronize();

  const auto t0 = std::chrono::steady_clock::now();
  int iters = 0;
  while (true)
  {
    A->Mult(X, Y);
    std::swap(X, Y);
    iters++;
    const auto now = std::chrono::steady_clock::now();
    const double dt = std::chrono::duration<double>(now - t0).count();
    if (dt >= min_s) { break; }
  }
  hipDeviceSynchronize();

  const auto t1 = std::chrono::steady_clock::now();
  const double seconds = std::chrono::duration<double>(t1 - t0).count();
  const double nrm = X.Norml2(); // enforce readback

  std::cout << "GPU_OK\n";
  std::cout << "mesh " << mesh_path << "\n";
  std::cout << "order " << order << "\n";
  std::cout << "refine " << refine << "\n";
  std::cout << "pa " << use_pa << "\n";
  std::cout << "ndofs " << fes.GetTrueVSize() << "\n";
  std::cout << "iters " << iters << "\n";
  std::cout << "seconds " << seconds << "\n";
  std::cout << "norm " << nrm << "\n";
  return 0;
}
""".lstrip(),
            encoding="utf-8",
        )

        r_app_cfg = run_cmd(
            ctx.repo_root,
            env,
            [
                "cmake",
                "-S",
                str(app_src),
                "-B",
                str(app_bld),
                "-G",
                "Ninja",
                f"-DMFEM_DIR={bld}",
                "-DHIP_PLATFORM=amd",
                f"-DHIP_ROOT_DIR={rocm_dist}",
                f"-DHIP_PATH={rocm_dist}",
                f"-DROCM_PATH={rocm_dist}",
                f"-DCMAKE_CXX_COMPILER={clangxx}",
            ],
            120,
            log,
        )
        if r_app_cfg.rc == 0:
            r_app_bld = run_cmd(ctx.repo_root, env, ["ninja", "-C", str(app_bld), "-j4"], 600, log)
            if r_app_bld.rc == 0 and app_exe.exists():
                run_cwd = ctx.builds_dir() / "mfem" / "_run"
                run_cwd.mkdir(parents=True, exist_ok=True)

                def run_apply(sampler):
                    candidates = [
                        (refine, order, use_pa),
                        (max(0, refine - 1), order, use_pa),
                        (max(0, refine - 1), max(1, order - 1), use_pa),
                        (0, 2, use_pa),
                        (0, 2, 0),
                    ]
                    last: CommandResult | None = None
                    for rlv, ordv, pav in candidates:
                        run_env = dict(env)
                        run_env["ROCM_VALIDATION_MFEM_MIN_S"] = str(min_run_s)
                        run_env["ROCM_VALIDATION_MFEM_ORDER"] = str(ordv)
                        run_env["ROCM_VALIDATION_MFEM_REFINE"] = str(rlv)
                        run_env["ROCM_VALIDATION_MFEM_PA"] = str(pav)
                        t0 = time.monotonic()
                        r = run_cmd(run_cwd, run_env, [str(app_exe), str(mesh)], 600, log)
                        wall_s = time.monotonic() - t0
                        last = r
                        if r.rc == 0 and "GPU_OK" in (r.out + "\n" + r.err):
                            return r, wall_s, sampler
                        txt = (r.out + "\n" + r.err).lower()
                        if "out of memory" in txt or "hip error" in txt or "hsa" in txt:
                            continue
                        return r, wall_s, sampler
                    return last or CommandResult(rc=1, out="", err="mfem_apply failed", dur_ms=0), 0.0, sampler

                r3a, wall_sa, sampler_a = with_power_sampler(cfg, build_dir=build_dir, fn=run_apply)
                if r3a.rc == 0 and "GPU_OK" in (r3a.out + "\n" + r3a.err):
                    kv = parse_kv((r3a.out + "\n" + r3a.err).strip())
                    metric = f"mfem_apply mesh={Path(kv.get('mesh', mesh.name)).name} wall={wall_sa:.2f}s"
                    for k in ("order", "refine", "pa", "ndofs", "iters", "seconds"):
                        if k in kv:
                            metric += f" {k}={kv[k]}"
                    try:
                        iters = float(kv.get("iters", "0") or 0)
                        secs = float(kv.get("seconds", "0") or 0)
                        if iters > 0 and secs > 0:
                            metric += f" apply/s={iters/secs:.1f}"
                    except Exception:
                        pass
                    metric = append_power(metric, sampler_a, baseline_w=baseline_avg_w(cfg, build_dir))
                    if sampler_a is not None:
                        gpu = sampler_a.avg_gpu_busy()
                        base_w = baseline_avg_w(cfg, build_dir) or 0.0
                        avgw = sampler_a.avg_power_w() or 0.0
                        if (gpu is not None and gpu < 15) and (avgw - base_w) < 15:
                            return StepResult(build_dir, "MFEM (HIP) build+run", "FAIL", fmt_duration(r1.dur_ms + r2.dur_ms + r3a.dur_ms), f"no GPU activity detected | {metric}")
                    return StepResult(build_dir, "MFEM (HIP) build+run", "OK", fmt_duration(r1.dur_ms + r2.dur_ms + r3a.dur_ms), metric)
    # Detect runtime flags from help, so we can request HIP device when supported.
    h = run_cmd(ctx.repo_root, env, [str(exe), "-h"], 20, log)
    help_txt = (h.out + "\n" + h.err)
    have_device_flag = ("-d " in help_txt) or ("--device" in help_txt)
    have_refine_flag = ("-r " in help_txt) or ("--refine" in help_txt)
    have_order_flag = ("-o " in help_txt) or ("--order" in help_txt)
    have_pa_flag = ("-pa" in help_txt) or ("--partial-assembly" in help_txt)
    have_no_vis_flag = ("-no-vis" in help_txt) or ("--no-visualization" in help_txt)

    def mk_args(refine: int | None, order: int | None) -> list[str]:
        args: list[str] = [str(exe), "-m", str(mesh)]
        if have_refine_flag and refine is not None:
            args += ["-r", str(refine)]
        if have_order_flag and order is not None:
            args += ["-o", str(order)]
        # Partial assembly tends to avoid large sparse matrices and is much more
        # memory-friendly on GPUs while still validating HIP execution.
        if have_pa_flag:
            args += ["-pa"]
        if have_device_flag:
            args += ["-d", "hip"]
        if have_no_vis_flag:
            args += ["-no-vis"]
        return args

    # Prefer a heavier parameter set, but fall back if HIP OOM/other runtime
    # errors occur. We will also repeat runs until we reach min_bench_s to avoid
    # short "pulses" that make GPU/power validation noisy.
    candidates: list[tuple[int | None, int | None]] = []
    if have_refine_flag and have_order_flag:
        candidates = [(2, 3), (1, 3), (1, 2), (0, 2), (None, 1), (None, None)]
    elif have_order_flag:
        candidates = [(None, 3), (None, 2), (None, 1), (None, None)]
    else:
        candidates = [(None, None)]

    min_run_s = float(cfg.get("workloads", {}).get("mfem", {}).get("min_bench_s", 5.0) or 5.0)

    # MFEM examples may write output files (e.g. refined meshes / solutions) into
    # the current working directory. Keep the repo root clean by running in a
    # dedicated workspace folder.
    run_cwd = ctx.builds_dir() / "mfem" / "_run"
    run_cwd.mkdir(parents=True, exist_ok=True)

    def run_one(sampler):
        last_err: CommandResult | None = None
        for refine, order in candidates:
            args = mk_args(refine, order)
            t0 = time.monotonic()
            r = run_cmd(run_cwd, env, args, 600, log)
            wall_s = time.monotonic() - t0
            last_err = r
            if r.rc != 0:
                txt = (r.out + "\n" + r.err).lower()
                # Common HIP runtime failures: retry with smaller params.
                if "out of memory" in txt or "hip error" in txt or "hsa" in txt:
                    continue
                # Otherwise treat as hard failure.
                return r, wall_s, sampler

            # Extend to a sustained run by repeating the same command.
            total_wall_s = wall_s
            total_ms = r.dur_ms
            runs = 1
            last_ok = r
            while total_wall_s < min_run_s:
                t1 = time.monotonic()
                r2 = run_cmd(run_cwd, env, args, 600, log)
                dt = time.monotonic() - t1
                total_wall_s += dt
                total_ms += r2.dur_ms
                runs += 1
                last_ok = r2
                if r2.rc != 0:
                    return r2, total_wall_s, sampler

            # Annotate stdout minimally (for logs/diagnostics).
            out = last_ok.out
            err = last_ok.err
            suffix = f"\nMFEM_VALIDATE exe={exe.name} params: refine={refine} order={order} pa={int(have_pa_flag)} runs={runs}\n"
            out = (out or "") + suffix
            return CommandResult(rc=0, out=out, err=err, dur_ms=total_ms), total_wall_s, sampler

        # No candidate succeeded: return the last error.
        assert last_err is not None
        return last_err, 0.0, sampler

    r3, wall_s, sampler = with_power_sampler(cfg, build_dir=build_dir, fn=run_one)
    metric = f"{exe.name} mesh={mesh.name} wall={wall_s:.2f}s"
    if have_device_flag:
        metric += " device=hip"
    else:
        metric += " device=(flag-missing)"
    metric = append_power(metric, sampler, baseline_w=baseline_avg_w(cfg, build_dir))

    if r3.rc != 0:
        return StepResult(build_dir, "MFEM (HIP) build+run", "FAIL", fmt_duration(r1.dur_ms + r2.dur_ms + r3.dur_ms), f"run rc={r3.rc} | {metric}")

    if not have_device_flag:
        return StepResult(build_dir, "MFEM (HIP) build+run", "FAIL", fmt_duration(r1.dur_ms + r2.dur_ms + r3.dur_ms), f"cannot force HIP device (unknown CLI flags) | {metric}")

    if wall_s < min_run_s:
        metric += f" (short<{min_run_s:.0f}s; increase mfem run size)"

    if sampler is not None:
        gpu = sampler.avg_gpu_busy()
        base_w = baseline_avg_w(cfg, build_dir) or 0.0
        avgw = sampler.avg_power_w() or 0.0
        if (gpu is not None and gpu < 5) and (avgw - base_w) < 5:
            return StepResult(build_dir, "MFEM (HIP) build+run", "FAIL", fmt_duration(r1.dur_ms + r2.dur_ms + r3.dur_ms), f"no GPU activity detected | {metric}")

    return StepResult(build_dir, "MFEM (HIP) build+run", "OK", fmt_duration(r1.dur_ms + r2.dur_ms + r3.dur_ms), metric)
