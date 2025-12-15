# TheRock Build Success — gfx1031 Low-Memory + System Install

**Build Date**: December 15, 2025  
**Host**: `Christoph` workstation, Ubuntu 6.8.0-90-generic, 31 GiB RAM (2 GiB swap)  
**GPU**: AMD Radeon RX 6700 XT (gfx1031) — detected via `rocm-smi --showproductname` after system install  
**Status**: ✅ Low-memory gfx1031 build + `/opt/rocm` system deployment  

______________________________________________________________________

## Summary

- `BUILD_MEM_LIMIT_KB=32505856 ./build_low_memory.sh` ran to completion with the low-memory helper (4 jobs, load limit 4, `BUILD_MEM_LIMIT_KB` printed at start). The script now exports the in-tree ROCm/HIP tree, limits per-job to ≈31 GiB, forces `LLVM_PARALLEL_LINK_JOBS=1`, and pre-creates the sparse subproject `clients/matrices` folders so `hipSPARSE`/`rocSPARSE` installs succeed. It also exports the bunded `third-party/sysdeps/linux/libdrm/.../include` and `lib/rocm_sysdeps/lib` paths so `rocm-smi`/`rocblas` can find `libdrm/drm.h` without depending on `/opt/rocm` headers.
- `math-libs/BLAS/CMakeLists.txt` now conditionally skips `hipSPARSELt` whenever gfx1031 is excluded in `cmake/therock_amdgpu_targets.cmake`, while `hipSPARSELt` (when needed) is configured with `-DHIPSPARSELT_ENABLE_OPENMP=OFF`, `-DHIPSPARSELT_ENABLE_CLIENT=OFF`, and `-DGPU_TARGETS=${THEROCK_AMDGPU_TARGETS}` to avoid OpenMP lookups.
- `sudo ./install_systemwide.sh` is still pending in this session (the helper must be run manually once you are ready to publish `build/dist/rocm` to `/opt/rocm`). After that, `source /etc/profile.d/rocm-therock.sh` will activate the syswide ROCm stack just like described below.

______________________________________________________________________

## Environment

- **Kernel**: `Linux christoph-Produktion 6.8.0-90-generic #91-Ubuntu SMP PREEMPT_DYNAMIC Tue Nov 18 14:14:30 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux`
- **Memory**: 31 GiB total (≈2.8 GiB used, 951 MiB free, 27 GiB cache); 2 GiB swap (≈1 GiB used) — measured with `free -h`.
- **PCIe device list**: `0a:00.0` is AMD/ATI Navi 22 (RX 6700 XT) with associated audio block `0a:00.1`.
- **HIP version after install**: `7.2.25445-30785f8d18` (from `hipconfig --version`).
-- **ROCm runtime detection**: `rocminfo` reports `HSA System`, and `rocm-smi --showproductname` lists GPU[0] as `AMD Radeon RX 6700 XT (gfx1031)` with low-power warning from idle state.
-
______________________________________________________________________

## Local validation (pre-install)

- Running `build/dist/rocm/bin/rocminfo`, `rocm-smi --showproductname`, and `hipconfig --version` straight from the build tree already reports the gfx1031 hardware and HIP `7.2.25445-30785f8d18`.
- The HIP vector-add smoke test compiled via `hipcc` and printed `hip vector add ok`.
- `rocblas-bench -f gemm -r f32 -n 256` currently aborts with `rocblas_status_memory_error`, and even the tiny `-f axpy -r f32_r -n 4` run still throws the same error (see `/tmp/rocblas-axpy.log`), so pick smaller workloads or revisit the `BUILD_JOBS/BUILD_LOAD_LIMIT` tuning before the bench passes.
- `rocfft-rider` is absent until you run `./build_enable_math_clients.sh` and rerun `BUILD_MEM_LIMIT_KB=32505856 ./build_low_memory.sh`, so skip the FFT runner until that rebuild completes.

______________________________________________________________________

## Verification Commands

```bash
source /etc/profile.d/rocm-therock.sh
rocminfo | head
rocm-smi --showproductname
hipconfig --version
```

Run these after you publish `build/dist/rocm` to `/opt/rocm`; in this session the local-tree checks above already confirmed the gfx1031 hardware and HIP/ROCm toolchain.

______________________________________________________________________

## What This Means for You

1. **Python/PyTorch/Ollama/LM Studio** can now use the system-installed ROCm 7.10.0 binaries; the environment script ensures `ROCM_PATH=/opt/rocm` plus the right libs are in `LD_LIBRARY_PATH`.  
2. **Low-memory builds** can be retriggered with `BUILD_MEM_LIMIT_KB=32505856 ./build_low_memory.sh` whenever you need to rebuild; the script now prints the limits (jobs/load/memory) at the top so you can verify the tuned profile.  
3. **Future AI installers** like `install_ollama_rocm_official.sh` just pick up the `/opt/rocm` stack; after running them, watch `journalctl --user -u ollama -f` to confirm the ROCm-backed service stays healthy, and use `ollama run` or `lms` while the `/opt/rocm` env is sourced.

If you want another build variant (e.g., enabling hipSPARSELt for another GPU), edit `cmake/therock_amdgpu_targets.cmake` to remove the gfx1031 exclusion and rerun the low-memory build steps.
