# Build Experience Notes (gfx1031 custom branch)

## Current Configuration

- Branch: `rocm-7.11-gfx103x`
- Configure helper: `./build_gfx1031.sh configure` (reads defaults from `config_gfx1031.yaml`)
- Build helper: `./build_gfx1031.sh bootstrap` then `./build_gfx1031.sh build` (uses `ninja`, not `cmake --build`)
- Targets: `THEROCK_AMDGPU_TARGETS=gfx1031`
- Dist bundle: `THEROCK_DIST_AMDGPU_TARGETS=gfx1031`, `THEROCK_DIST_AMDGPU_FAMILIES=gfx1031`
- Python: `.venv` is auto-created/activated by `build_gfx1031.sh` (for build tools + YAML parsing)
- `ccache` enabled via `eval "$(./build_tools/setup_ccache.py --init)"` inside `build_gfx1031.sh`
- Memory limits (default): `systemd-run --user --scope -p MemoryHigh=28G -p MemoryMax=31G ...`
- Compiler stack used in this repo:
  - Host GCC: `gcc 13.3.0`
  - Host clang: `AMD clang 22.0.0git`
  - ROCm HIP compiler: `/opt/rocm/bin/hipcc` (`HIP version 7.2.53150-1cedb43795`)
  - ROCm clang++: `/opt/rocm/lib/llvm/bin/clang++` (`AMD clang 22.0.0git`)
  - Clang resource includes used by ROCm toolchain: `/opt/rocm/lib/llvm/lib/clang/22/include`

0a. **2026-03-03: TensorFlow ROCm build and clang-22 include handling**
   - TensorFlow `v2.19.0` `third_party/gpus/rocm_configure.bzl` only listed builtin clang include dirs up to version 20 for ROCm.
   - On this system ROCm ships clang resource headers under `.../clang/22/include`.
   - The TensorFlow fork build helper `tools/rocm_release/build_tensorflow_rocm_wheel.sh` patches `rocm_configure.bzl` during build setup to append:
     - `/lib/llvm/lib/clang/21/include`
     - `/lib/llvm/lib/clang/22/include`
   - This keeps the TensorFlow ROCm build path deterministic for this local compiler stack.

0b. **2026-03-07: TensorFlow ROCm runtime fix for ROCm LLVM / COMGR symbol collisions**
   - A freshly built TensorFlow ROCm wheel still crashed on the first GPU op even with hipBLASLt init disabled.
   - Native backtraces showed ROCm COMGR / HIP resolving into TensorFlow's bundled LLVM symbols from `libtensorflow_framework.so.2` (`llvm::Twine`, `llvm::compression`, `llvm::localCache`) instead of ROCm's own `libLLVM.so`.
   - The durable fix is now in the TensorFlow fork helper `tools/rocm_release/build_tensorflow_rocm_wheel.sh`:
     - prefer the in-tree ROCm dist by default,
     - rewrite `.tf_configure.bazelrc` to pin `ROCM_PATH` and `LD_LIBRARY_PATH` to that in-tree dist,
     - postprocess the produced wheel with `patchelf --rename-dynamic-symbols` so TensorFlow's dynamic LLVM exports are renamed across the wheel DSOs and cannot interpose on ROCm COMGR anymore.
   - Functional validation is green again against the in-tree ROCm build:
     - run: `validation/workspace/runs/2026-03-07_161741`
     - TensorFlow matmul: `C = A * B` dense GEMM, `m=n=k=4096`, `dtype=float16`
     - measured `tflops_est = 21.50`
   - loaded HIP runtime: `build-stage2/dist/rocm/lib/libamdhip64.so.7.2.53150-1cedb43795`
   - hipBLASLt remains disabled for this gfx1031 profile (`TF_ROCM_DISABLE_HIPBLASLT=1`, `TF_ROCM_USE_HIPBLASLT=0`, `TF_ROCM_DISABLE_HIPBLASLT_INIT=1`).

0c. **2026-03-10: Framework wheel packaging moved out of `validation/`**
   - TheRock now treats PyTorch, ONNX Runtime, and TensorFlow as external framework builds again.
   - The framework forks own the build/promote helpers under `tools/rocm_release/`.
   - TheRock keeps only thin integration wrappers and validation profiles.
   - Resulting boundary:
     - TheRock repo: custom ROCm build + runtime validation + integration wrappers
     - Framework forks: source patches + wheel build logic + wheel promote helpers
   - This now applies consistently to:
     - `rocm-7.11-pytorch-gfx103x`
     - `rocm-7.11-onnxruntime-gfx103x`
     - `rocm-7.11-tensorflow-gfx103x`

0d. **2026-03-08: TensorFlow functional validation moved to a dedicated venv**
   - The post-build TensorFlow matmul step no longer installs the wheel into the shared validation venv `validation/workspace/envs/py`.
   - It now creates/uses `validation/workspace/envs/tensorflow_rocm/`.
   - Reason: TensorFlow runtime pins (`numpy`, `protobuf`, `grpcio`, etc.) otherwise leak into unrelated validation workloads and create cross-profile package conflicts.

0e. **2026-03-10: Unified custom-wheel lifecycle across framework forks**
   - The operational model is now the same for PyTorch, ONNX Runtime, and TensorFlow:
     1. build repo-local against the custom in-tree ROCm stack,
     2. validate against `<builddir>/dist/rocm`,
     3. promote to `/opt/rocm/wheels/...`,
     4. validate the promoted system state separately via `*_promoted` profiles.
   - TheRock validation now checks both runtime linkage/prefix and workload behavior with stats/power where applicable.
   - Stable consumer entrypoints are the promoted aliases under `/opt/rocm/wheels/...`, e.g.:
     - `torch-current.whl`
     - `onnxruntime-current.whl`
     - `tensorflow-current.whl`
   - Consumer repos are expected to use only the promoted artifacts, not local build trees.

0f. **2026-03-22: ORT/Piper performance triage must follow the validation workflow, not ad-hoc scripts**
   - The `validation/` operating model matters for performance work just as much as for correctness work.
   - Keep exactly one source of truth per run:
     - one wheel source (in-tree or promoted),
     - one ROCm runtime prefix,
     - one staged model copy,
     - one artifact root under `validation/workspace/runs/<run_id>/`.
   - Run functional real-model validation first, then benchmark via a dedicated validation profile.
   - For Piper/ORT on gfx1031, isolated benchmark child processes were necessary to avoid mixing CPU and ROCm session state and to keep run artifacts reproducible.
   - If repeated ROCm runs on the same real model show output-shape drift, treat that as a stability/correctness investigation first; do not summarize it as a plain throughput result.

0g. **2026-03-22: ORT validation reference runs must be serial when they share the runtime venv**
   - `onnxruntime_in_tree_tts`, `onnxruntime_rocm711_promoted_tts`, and the matching benchmark profiles all reuse `validation/workspace/envs/onnxruntime_rocm/`.
   - Those steps run `pip --force-reinstall` inside that shared venv before the workload.
   - Running multiple such profiles in parallel can corrupt the venv transiently and produce false failures such as:
     - pip install `OSError` on temporary dist-info files
     - invalid partial distributions like `~umpy` / `~ympy`
   - Treat only serial reruns as authoritative for ORT reference validation and benchmark baselines.

0h. **2026-03-22: Serial Piper TTS benchmark baseline shows ROCm far slower than CPU on the small real workload**
   - Validation benchmark profiles now exist for both:
     - `onnxruntime_in_tree_tts_benchmark`
     - `onnxruntime_rocm711_promoted_tts_benchmark`
   - Serial reference runs:
     - in-tree: `validation/workspace/runs/2026-03-22_203627`
     - promoted: `validation/workspace/runs/2026-03-22_203450`
   - The two states agree closely:
     - CPU create about `0.72-0.74 s`
     - CPU cold about `34-40 ms`
     - CPU second run about `31-33 ms`
     - ROCm create about `0.86-0.88 s`
     - ROCm cold about `13.2-13.4 s`
     - ROCm second run about `4.0 s`
   - This means the slowdown is not just session creation overhead; the real ROCm Piper inference path is much slower than CPU for these two short real cases.
   - The benchmark artifacts also show a repeated-run shape drift on ROCm:
     - `mogli`: cold `[1,1,1,11008]` -> second run `[1,1,1,10752]`
     - `hey mogli`: cold `[1,1,1,14080]` -> second run `[1,1,1,12800]`
   - Because the same drift appears both in-tree and promoted, the next diagnosis target is a ROCm steady-state/runtime behavior issue, not a promote-only packaging mismatch.

0i. **2026-03-22: Steady-state ROCm Piper issue reproduces as a repeated-run runtime problem, not only as a slow benchmark**
   - Dedicated validation profiles now exist:
     - `onnxruntime_in_tree_tts_steady_state`
     - `onnxruntime_rocm711_promoted_tts_steady_state`
   - Serial probe runs:
     - in-tree: `validation/workspace/runs/2026-03-22_204403`
     - promoted: `validation/workspace/runs/2026-03-22_204446`
   - Case `mogli`, 6 runs on the same session:
     - CPU remains stable for all iterations at shape `[1,1,1,11008]`
     - ROCm reproduces a repeated-run failure pattern under both prefixes
   - Current strongest local reproduction:
     - iter 0: cold, correct shape
     - iter 1: still correct
     - iter 2: same shape but changed output statistics
     - iter 3+: either `Reshape_1` runtime failure or a shape jump in the duration path
   - Follow-up ad-hoc matrix on the same frozen model:
     - `ORT_DISABLE_ALL` delays the drift but does not eliminate it
     - `ORT_ENABLE_BASIC` and `ORT_ENABLE_ALL` both still drift
     - `ORT_ROCM_DISABLE_FAST_REDUCTION=1` also does not eliminate the repeated-run drift
   - Interpretation:
     - this is not just promote-path packaging
     - not just graph optimization level
     - not just the known fast-reduction bug family
     - current best hypothesis remains a ROCm EP steady-state / lifetime / execution-order issue on the real Piper graph

0j. **2026-03-22: Repeat-trace on the top duration path shows the steady-state jump occurs before `Reshape_1`**
   - The internal helper `validation/src/steps/workloads/onnxruntime/piper_tts_debug.py` now supports repeated runs on the same session for selected outputs.
   - Repeated ROCm trace on staged Lessac case `mogli` with:
     - `/dp/Split_output_0`
     - `/Exp_output_0`
     - `/Ceil_output_0`
     - `/Cast_output_0`
     - `/CumSum_output_0`
   - Trace artifact:
     - `validation/workspace/debug/piper_steady_repeat_cumsum_trace.json`
   - Current finding:
     - iter 0-2 stay at the expected duration path (`Cast=43`)
     - iter 3 jumps to the blown-up duration family (`Cast=3162`, `CumSum` in the thousands)
     - iter 4-5 stay in that blown-up family
   - This places the repeated-run transition upstream of `Reshape_1` and inside the same top-level duration path that was already known from the one-shot correctness bug family.

0. **2025-12-20: Config moved to `config_gfx1031.yaml`**
   - `configure_gfx1031.sh` was removed.
   - Configure via `./build_gfx1031.sh configure` (uses `config_gfx1031.yaml`, supports Stage-1/Stage-2).

## Lessons / Fixes

1. **Stage → Dist mirror for Third-Party packages**  
   Ninja starts dependent projects before the dist artefacts exist. Manually mirroring stage to dist keeps `find_package(...)` happy. Useful commands:
   ```
   rsync -a build/<component>/stage/ build/<component>/dist/
   ```
   Needed so far: `FunctionalPlus`, `Eigen3`, `nlohmann-json`, `fmt`, `host-blas`, `SuiteSparse`, `zlib`.

2. **Patch scripts requiring Python**  
   - `third-party/sysdeps/linux/libcap/patch_install.sh`  
   - `third-party/sysdeps/linux/amd-mesa/patch_install.sh`  
   Both now try `Python3_EXECUTABLE` from the environment and fall back to `$(command -v python3)` with a clear error if missing. Ensures rocprofiler/rdc/sysdeps installs do not fail mid-build.

3. **Interrupted builds**  
   With `-j1`, Ninja resumes exactly where it left off; `ccache` shortens re-compiles. No need to restart from scratch after a Ctrl+C, just rerun the same build command.

4. **Live Monitoring**  
   Keep `tail -f build.log` in a second terminal. The file only shows the active command; detailed per-target logs live under `build/logs/therock-*.log`.

5. **CMake version compatibility (fftw3 / c-ares)**  
   The venv `pip install cmake` (4.x) breaks third-party builds that still use `cmake_minimum_required(<3.5)` (fftw3, grpc/cares). Use system CMake 3.28 (`/usr/bin/cmake`) and remove the venv wrappers (`rm ~/.local/bin/cmake ~/.local/bin/cpack ~/.local/bin/ctest`) to avoid patching external sources.

6. **In-tree ROCm env activation (tests)**  
   Use `./test_gfx1031.sh` to run sanity/benchmarks; it auto-activates `.venv` (if present) and sets `ROCM_PATH`/`HIP_PATH`/`LD_LIBRARY_PATH` to `<builddir>/dist/rocm`. If multiple in-tree dist roots exist, it runs the same checks for each (stage2/build/stage1) and writes per-build logs; restrict via `--stage2/--stage1/--build-dir` (or `BUILD_DIR=...`).

7. **Verify gfx1031 HIP kernel/device-lib path (avoid generic fallback)**  
   The critical check is that HIP compiles and links against gfx1031-specific device libs, not generic compatibility bitcode.  
   ```
   ./test_gfx1031.sh --no-bench
   ls $HIP_DEVICE_LIB_PATH/oclc_isa_version_1031.bc
   hipcc -v tests/hipcc_check.cpp -o /tmp/hipcc_check 2>&1 | rg -n "gfx1031|oclc_isa_version_1031|amdgcn/bitcode"
   ```
   Expected: `-mcpu=gfx1031` (or `--offload-arch=gfx1031`) and `oclc_isa_version_1031.bc` in the compile/link line.  
   If you only see `10-3-generic` (or another gfx target), force the arch with `--offload-arch=gfx1031` or set `HIPCC_COMPILE_FLAGS_APPEND="--offload-arch=gfx1031"` and rebuild.
   Verified on 2025-12-18: `hipcc -v` shows `-target-cpu gfx1031` and links `oclc_isa_version_1031.bc`.

8. **Quick Bench Suite (gfx1031, RX 6700 XT)**  
   Run after `./test_gfx1031.sh --no-bench` (or after exporting `ROCM_PATH`/`PATH`/`LD_LIBRARY_PATH` yourself). Results captured on 2025-12-18:
   ```
   rocfft-bench --length 1024 --precision single -t 0 -N 5
   # ~0.0122 ms, ~4.16 GFLOPS

   rocblas-bench -f axpy -r f32_r -n 1048576
   # 77.68 GFLOPS, 466.10 GB/s, 26.99 us

   hipblas-bench -f gemm -r f32_r -m 256 -n 256 -k 256
   # 1161.05 GFLOPS, 27.21 GB/s, 28.9 us

   benchmark_rocrand_generate --size 1048576 --trials 3 --dis uniform-float --engine philox
   # 187.8 GB/s, 46.95 GSample/s, 0.021 ms

   hipsparse-bench -f axpyi -n 1024 -z 256 -i 1
   # 0.03 GFLOPS, 0.22 GB/s, 0.02 ms
   ```

9. **Large GEMM (rocBLAS)**
   ```
   rocblas-bench -f gemm -r f32_r -m 4096 -n 4096 -k 4096
   # 12193.4 GFLOPS, 11271.6 us
   ```

10. **hipBLASLt status**
   `hipblaslt-bench` currently segfaults (Signal 11) even for small sizes on this setup:
   ```
   hipblaslt-bench -f matmul -m 1024 -n 1024 -k 1024
   hipblaslt-bench -f matmul -r f32_r -m 1024 -n 1024 -k 1024 --compute_type f32_r
   ```
   Standalone build from `rocm-libraries/projects/hipblaslt` fails at configure time because gfx1031 is not in the supported GPU list:
   ```
   cmake -S rocm-libraries/projects/hipblaslt -B rocm-libraries/projects/hipblaslt/build-standalone \
     -D CMAKE_C_COMPILER=$ROCM_PATH/lib/llvm/bin/clang \
     -D CMAKE_CXX_COMPILER=$ROCM_PATH/lib/llvm/bin/clang++ \
     -D CMAKE_PREFIX_PATH=$ROCM_PATH \
     -D GPU_TARGETS=gfx1031
   # CMake Error: Unsupported GPU target: gfx1031
   ```

11. **2025-12-18: Partial rebuild (gfx1031) + docs update**
   - Added a repeatable expunge+rebuild workflow (now lives in `./build_gfx1031.sh rebuild ...`).
     (now requires explicit targets; `--include-unsupported` opts into hipBLASLt/hipSPARSELt/rocWMMA).
   - Rotated `build.log` to `build.log.bak-20251218-171506` and continued logging to fresh `build.log`.
   - Cleaned + rebuilt subprojects with memory limits and venv (historical commands):
     ```
     systemd-run --user --scope -p MemoryHigh=28G -p MemoryMax=31G bash -lc 'source .venv/bin/activate && cmake --build build --target hipBLASLt+expunge'
     systemd-run --user --scope -p MemoryHigh=28G -p MemoryMax=31G bash -lc 'source .venv/bin/activate && cmake --build build --target hipSPARSELt+expunge'
     systemd-run --user --scope -p MemoryHigh=28G -p MemoryMax=31G bash -lc 'source .venv/bin/activate && cmake --build build --target rocWMMA+expunge'
     systemd-run --user --scope -p MemoryHigh=28G -p MemoryMax=31G bash -lc 'source .venv/bin/activate && cmake --build build --target hipBLASLt'
     systemd-run --user --scope -p MemoryHigh=28G -p MemoryMax=31G bash -lc 'source .venv/bin/activate && cmake --build build --target hipSPARSELt'
     systemd-run --user --scope -p MemoryHigh=28G -p MemoryMax=31G bash -lc 'source .venv/bin/activate && cmake --build build --target rocWMMA'
     ```
   - Build logs live in `build/logs/{hipBLASLt,hipSPARSELt,rocWMMA}_build.log` (all end with rc=0).
   - Expected warnings: gfx1031 excluded for hipBLASLt/hipSPARSELt/rocWMMA (fallback to defaults),
     plus `rocm_smi_lib` git describe warnings (harmless).

12. **Recommended build profile (LLM / Vision / Audio, gfx1031)**
   Intended for Ollama/Mistral/Qwen, PyTorch, Whisper, and similar workloads
   with best performance on RX 6700 XT. Keep HIP toolchain + core math/ML libs,
   and keep composable kernel, profiler, and tests ON. Disable hipBLASLt and
   hipSPARSELt (unsupported for gfx1031).
   ```
   systemd-run --user --scope -p MemoryHigh=28G -p MemoryMax=31G \
     cmake -B build -GNinja . \
     -DTHEROCK_AMDGPU_TARGETS=gfx1031 \
     -DTHEROCK_ENABLE_ALL=OFF \
     -DTHEROCK_ENABLE_COMPILER=ON \
     -DTHEROCK_ENABLE_CORE_RUNTIME=ON \
     -DTHEROCK_ENABLE_HIP_RUNTIME=ON \
     -DTHEROCK_ENABLE_HIPIFY=ON \
     -DTHEROCK_ENABLE_BLAS=ON \
     -DTHEROCK_ENABLE_PRIM=ON \
     -DTHEROCK_ENABLE_RAND=ON \
     -DTHEROCK_ENABLE_FFT=ON \
     -DTHEROCK_ENABLE_SPARSE=ON \
     -DTHEROCK_ENABLE_SOLVER=ON \
     -DTHEROCK_ENABLE_HIPBLASLT=OFF \
     -DTHEROCK_ENABLE_HIPSPARSELT=OFF \
     -DTHEROCK_ENABLE_MIOPEN=ON \
     -DTHEROCK_ENABLE_HIPDNN=ON \
     -DTHEROCK_ENABLE_COMPOSABLE_KERNEL=ON \
     -DTHEROCK_ENABLE_RCCL=ON \
     -DTHEROCK_ENABLE_ROCWMMA=OFF \
     -DTHEROCK_ENABLE_PROFILER=ON \
     -DTHEROCK_ENABLE_DC_TOOLS=OFF \
     -DBUILD_TESTING=ON
   ```

13. **2025-12-18: Make BLAS Lt components optional for gfx1031**
   - Added `THEROCK_ENABLE_HIPBLASLT` and `THEROCK_ENABLE_HIPSPARSELT` gating in BLAS.
   - rocBLAS now honors `THEROCK_ENABLE_HIPBLASLT` (sets `BUILD_WITH_HIPBLASLT=OFF` when disabled).
   - hipBLASLt/hipSPARSELt artifacts marked optional so packaging won't expect them.
   - README and recommended profile updated to disable unsupported Lt components for gfx1031.

13a. **2026-03-01: Custom ROCm Python feed + PyTorch wheel validation (green)**
   - Built local ROCm Python packages from `build-stage2/artifacts`:
     - `rocm-7.11.0a20260301`
     - `rocm-sdk-core-7.11.0a20260301`
     - `rocm-sdk-devel-7.11.0a20260301`
     - `rocm-sdk-libraries-gfx1031-7.11.0a20260301`
   - Created a local `simple/` index under:
     - `build-stage2/python_packages_gfx1031/dist/simple`
   - Installed ROCm packages from local index into `.venv` and verified:
     - `python -m rocm_sdk version -> 7.11.0a20260301`
     - `python -m rocm_sdk targets -> gfx1031`
   - Rebuilt PyTorch wheel against this custom ROCm package set:
     - `PYTORCH_EXTRA_INSTALL_REQUIREMENTS=rocm[libraries]==7.11.0a20260301`
     - wheel OpenMP dependency validation passes (`libtorch_cpu.so` depends on `libomp`)
     - runtime sanity check passes (`torch.cuda.is_available() == True`)
   - Root cause for prior import failure:
     - `hipsparselt` not present in custom `gfx1031` runtime package (expected for this profile).
   - Mitigation applied:
     - `_rocm_init.py` generation now filters preload shortnames to only libraries found by
       `rocm_sdk.find_libraries(...)`, so missing optional libs do not abort import.

14. **2025-12-18: Clean gfx1031 build helper**
   - Added `build_gfx1031.sh` for a clean full build with memory limits and ccache.
   - Script refuses to run if `build/` is not clean (unless `--clean` or `--no-check-clean` is used).
   - Uses the recommended gfx1031 profile (LLM/Vision/Audio) and logs to `build.log`.

15. **2025-12-18: Clean gfx1031 configure helper**
   - Integrated configure into `./build_gfx1031.sh configure` (with memory limits, venv, ccache).
   - Defaults are in `config_gfx1031.yaml`; CLI/env overrides are supported.

16. **2025-12-18: gfx1031 test helper**
   - Added `test_gfx1031.sh` for sanity checks and lightweight GEMM benchmarks.
   - Summarizes durations and (when available) TFLOPS parsed from bench output.
   - Logs output to `test_gfx1031.log` for quick inspection.

17. **2025-12-18: Switch helpers to clang host compiler**
   - `build_gfx1031.sh configure` sets `CMAKE_C_COMPILER=clang` and `CMAKE_CXX_COMPILER=clang++` (Stage-2 uses the Stage-1 in-tree toolchain).
   - Scripts fail fast if clang/clang++ are missing; ccache launchers remain enabled.

18. **2025-12-18: Switch helper builds to ninja**
   - The helper scripts call `ninja -C <builddir>` directly instead of `cmake --build` to avoid extra process spawning and ordering surprises.
   - Added a `ninja` availability check; keeps expunge + target sequencing explicit.

19. **2025-12-18: Helper QoL (ccache & venv automation)**
   - Bundled ccache 4.11.1 to `.local/bin/ccache`; helpers prepend `.local/bin` if present.
   - `build_gfx1031.sh` auto-evals `build_tools/setup_ccache.py`.
   - `build_gfx1031.sh` auto-creates `.venv` (python3 -m venv + requirements.txt) if missing.
   - README updated with ccache defaults and typical workflows.
   - Note: the YAML-driven configure+bootstrap+build flow still needs end-to-end validation in a fresh Stage-1/Stage-2 build.

20. **2025-12-19: Default BUILD_TESTING=OFF in configure helper**
   - `ENABLE_BUILD_TESTING=false` by default to avoid gcc-related ICEs; clang enforced as host compiler.
   - README notes how to re-enable tests via script toggle or extra CMake arg.

21. **2025-12-19: Composable-kernel toggle propagated to MIOpen**
   - `build_gfx1031.sh configure` sets `THEROCK_MIOPEN_USE_COMPOSABLE_KERNEL` to match the CK flag.
   - README notes: CK unsupported on gfx1031; MIOpen disables CK internally with a warning only.

22. **2025-12-19: Phase 1 clang build (ROCPROFSYS=OFF) stalled on sysdeps cmake configs**
   - Configure succeeds with clang18, BUILD_TESTING=OFF, ROCPROFSYS=OFF, hipBLASLt/hipSPARSELt/ROCWMMA=OFF.
   - Full build failed during early parallel configures of `rocm-half` and `grpc`: missing `ROCmCMakeBuildTools` (from rocm-cmake) and `ZLIBConfig.cmake` (from sysdeps zlib) under `dist/share/...`.
   - Root cause: missing/empty `dist/` configs during early parallel configures (and earlier also a Python3 scoping issue that prevented the stage→dist population rules from running reliably).
   - Fix (current state): add a dedicated bootstrap step (`./build_gfx1031.sh bootstrap`) which builds the minimal `+dist` targets (rocm-cmake + sysdeps + host-blas + common CMake-config deps) before the full build runs.

23. **2025-12-19: Helpers hardened (sysdeps stage→dist + LD_LIBRARY_PATH for host tools)**
   - `./build_gfx1031.sh bootstrap` builds a minimal set of `+dist` targets so dependent `find_package(...)` calls can resolve reliably during later parallel configures.
   - `build_gfx1031.sh` injects `LD_LIBRARY_PATH` with sysdeps dist+stage `rocm_sysdeps/lib` (zstd/zlib/bzip2/liblzma/elfutils/libdrm/numactl) to let host tools like `llvm-min-tblgen` load `librocm_sysdeps_zstd.so.1` during amd-llvm build.
   - Current state: helpers set `LD_LIBRARY_PATH` explicitly (no inheritance by default) to avoid mixing with `/opt/rocm-*`.

24. **2025-12-19: OpenBLAS → SuiteSparse path fixed**
   - SuiteSparse configure failed: `OpenBLASConfig.cmake` not found under `host-blas/dist`.
   - Fix (current state): bootstrap includes `therock-host-blas+dist` so `host-blas/dist` is populated before SuiteSparse configures.
   - Re-run `./build_gfx1031.sh bootstrap` after configure to validate.

25. **2025-12-19: Build helper hygiene (hipcc notice + clang 18 enum fix)**
   - `build_gfx1031.sh configure` prints an explicit notice if `./install/bin/hipcc` is missing (expected during Stage-1 bootstrap): it leaves `CMAKE_HIP_COMPILER` unset and relies on `COMPILER_TOOLCHAIN=amd-hip` for in-tree HIP subprojects (and does not fall back to `/opt/rocm`).
   - Initially tried adding `-Wno-enum-constexpr-conversion` via global `CMAKE_CXX_FLAGS`, but this later broke projects that treat unknown warning options as errors (e.g. `rocminfo` with `-Werror,-Wunknown-warning-option`). The helper no longer sets this globally; if SPIR-V headers cause issues again, the fix must be applied narrowly (to the affected subproject/toolchain only).
   - `build_gfx1031.sh` extends `LD_LIBRARY_PATH` to include the raw `build/.../zlib|zstd/build/b` directories in addition to stage/dist `rocm_sysdeps` to keep host tools (llvm-min-tblgen, etc.) finding `librocm_sysdeps_z*.so` during early compiler build.
   - Next step: rerun a clean Stage-1 (`./build_gfx1031.sh configure --stage1 && ./build_gfx1031.sh bootstrap --stage1 && ./build_gfx1031.sh build --stage1`) to verify amd-llvm now builds cleanly.

26. **2025-12-19: Add explicit bootstrap step for third-party/sysdeps**
   - Added a bootstrap step (`./build_gfx1031.sh bootstrap`) to build the minimum `+dist` targets that tend to be needed early (rocm-cmake, sysdeps zlib/zstd, host-blas, and a few common CMake-config deps) before the full parallel superbuild runs.
   - Goal: avoid intermittent configure failures during the full build due to missing `*Config.cmake` under `dist/` and missing `librocm_sysdeps_*.so` for host tools.
   - Status: script is new; needs validation as part of a full clean run (configure → bootstrap → build).

27. **2025-12-19: Ensure ccache bootstrapping config is active in build helper**
   - TheRock includes `build_tools/setup_ccache.py` which writes a repo-local `./.ccache/ccache.conf` with:
     - `sloppiness = include_file_ctime` (hardlink-friendly)
     - a custom `compiler_check` suitable for compiler bootstrapping
   - `build_gfx1031.sh` now also evals `setup_ccache.py` (not just configure) so the above settings are active during compilation, and exports `CCACHE_SLOPPINESS=include_file_ctime` as an explicit belt-and-suspenders.

28. **2025-12-19: Default to clean configure in helper**
   - `build_gfx1031.sh configure` removes `BUILD_DIR/` by default to ensure the toolchain/config stays coherent (especially when switching compilers or feature flags).
   - Added `--no-clean` for the rare case where an in-place reconfigure is desired.

29. **2025-12-19: Move stage→dist sync into bootstrap + unify logging**
   - Centralized early dependency preparation into `./build_gfx1031.sh bootstrap` so configure stays “pure”.
   - Later refined bootstrap to build `+dist` targets directly (instead of stage→dist symlinks) to avoid symlink edge cases while still providing early `dist/` CMake configs.
   - Bootstrap appends to `build.log` (same log as configure/build).
   - Downgraded the hipcc “missing” message from WARNING to INFO and clarified that hipcc appears only after the compiler/toolchain is built+installed into `./install` (not after the third-party bootstrap step).

30. **2025-12-19: Avoid system hipcc fallback (ensure in-tree HIP toolchain)**
   - `build_gfx1031.sh configure` only sets `CMAKE_HIP_COMPILER` when `./install/bin/hipcc` exists, and does not auto-fallback to `/opt/rocm/bin/hipcc`.
   - Rationale: prevent ABI/version mixing between in-tree ROCm and any system ROCm; TheRock’s HIP subprojects already pin their toolchain via `COMPILER_TOOLCHAIN amd-hip`.

31. **2025-12-19: amd-llvm failed in rocr-runtime configure (missing NUMAConfig)**
   - Failure: `rocr-runtime` (libhsakmt) `find_package(NUMA)` failed because `NUMAConfig.cmake` was expected under `build/third-party/sysdeps/linux/numactl/build/dist/lib/rocm_sysdeps/lib/cmake/NUMA` but sysdeps `therock-numactl` was never built in bootstrap.
   - Fix (current state): bootstrap includes `therock-numactl+dist` and verifies `numa-config.cmake` exists under `dist/`.

32. **2025-12-19: amd-llvm rocr-runtime configure needed LibElfConfig (elfutils)**
   - Failure: `rocr-runtime` (hsa-runtime) `find_package(LibElf)` expected `build/third-party/sysdeps/linux/elfutils/build/dist/lib/rocm_sysdeps/lib/cmake/LibElf` but sysdeps `therock-elfutils` was not in bootstrap.
   - Fix (current state): bootstrap includes `therock-elfutils+dist` and verifies `libelf-config.cmake` exists under `dist/`.

33. **2025-12-19: Dist dirs stayed empty (Python3_EXECUTABLE not exported) → find_package failures**
   - Symptom: subproject configures (notably `amd-comgr-impl`) failed with messages like:
     - `Super-project based find_package(AMDDeviceLibs) config file not found under .../build/compiler/amd-llvm/dist/...`
     - after manually copying one config, it would then fail on the next (`ClangConfig.cmake`, `LLVMConfig.cmake`, …).
   - Root cause: `Python3_EXECUTABLE` was not exported from `therock_setup_python_and_topology()` (function scope), so generated Ninja rules invoked `build_tools/teatime.py` and `build_tools/fileset_tool.py` without an explicit interpreter. This also prevented the stage→dist population step from running reliably, leaving many `dist/` dirs empty.
   - Fix: `cmake/therock_python_setup.cmake` now forces `Python3_EXECUTABLE` into the cache so generated rules consistently use the venv interpreter (`.venv/bin/python3`) for `teatime.py` and `fileset_tool.py`.
   - Recovery: re-run `./build_gfx1031.sh configure --no-clean` to regenerate `build/build.ninja`, then continue the build (optionally detached) via `./build_gfx1031.sh build --detach`.

34. **2025-12-19: fileset_tool copy failed when `dist/` is a symlink**
   - Symptom: stage installs failed with `OSError: Cannot call rmtree on a symbolic link` while running `fileset_tool.py copy .../dist .../stage`.
   - Cause: earlier helper workflows sometimes created `dist/` as a stage→dist symlink for quick bootstrapping. `PatternMatcher.copy_to()` used `shutil.rmtree()` unconditionally when `--remove-dest` is enabled, which raises on symlinks.
   - Fix: `build_tools/_therock_utils/pattern_match.py` now unlinks `destdir` when it is a symlink (instead of calling `rmtree`) and then proceeds with the normal directory copy.

35. **2025-12-19: ROCR-Runtime configure failed (Ninja RPATH relink error)**
   - Failure: `ROCR-Runtime` (hsa-runtime64) configure errored at `runtime/hsa-runtime/CMakeLists.txt:97 (add_library)`:
     - “install … requires changing an RPATH from the build tree … not supported with the Ninja generator … set CMAKE_BUILD_WITH_INSTALL_RPATH”.
   - Fix: added `-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON` to the ROCR-Runtime subproject `CMAKE_ARGS` in `core/CMakeLists.txt`.
   - Recovery: reconfigure (`./build_gfx1031.sh configure --no-clean --no-check-clean`), then `ninja -C build ROCR-Runtime+expunge`, then resume via `./build_gfx1031.sh build --detach`.

36. **2025-12-19: Bootstrap now builds +dist (no stage→dist symlinks)**
   - The bootstrap step uses `+dist` targets instead of creating stage→dist symlinks. This avoids symlink-related edge cases and ensures `dist/` CMake configs are real directories.
   - `build_gfx1031.sh` defaults to *not inheriting* `LD_LIBRARY_PATH` (to avoid accidentally pulling in `/opt/rocm-*`). Set `PRESERVE_LD_LIBRARY_PATH=1` if you explicitly want to append the inherited path.

37. **2025-12-19: rocSPARSE stage install failed when BUILD_CLIENTS_TESTS=OFF**
   - Failure: `rocSPARSE+stage` ran `cmake --install` and failed with:
     - `file INSTALL cannot find ".../rocSPARSE/build/clients/matrices": No such file or directory.`
   - Root cause: `math-libs/BLAS/pre_hook_rocSPARSE.cmake` unconditionally installed `${CMAKE_CURRENT_BINARY_DIR}/clients/matrices`, but that directory is only created when rocSPARSE client tests generate/copy matrices.
   - Fix: mark the install rule as `OPTIONAL` so `cmake --install` doesn’t hard-fail when client tests (and matrices) are disabled.
   - Recovery: `ninja -C build rocSPARSE+expunge rocSPARSE+stage` and then resume the full build.

38. **2025-12-19: hipSPARSE stage install failed for the same reason**
   - Failure: `hipSPARSE+stage` ran `cmake --install` and failed with:
     - `file INSTALL cannot find ".../hipSPARSE/build/clients/matrices": No such file or directory.`
   - Root cause: `math-libs/BLAS/pre_hook_hipSPARSE.cmake` installed `${CMAKE_CURRENT_BINARY_DIR}/clients/matrices` unconditionally, but with `BUILD_CLIENTS_TESTS=OFF` no matrices directory is generated.
   - Fix: mark the install rule as `OPTIONAL`.
   - Recovery: `ninja -C build hipSPARSE+expunge hipSPARSE+stage` and then resume the full build.

39. **2025-12-19: Fix “/opt/rocm contamination” in subproject CMakeCache**
   - Symptom: many subprojects wrote `/opt/rocm` into their `build/**/CMakeCache.txt` (e.g. `ROCM_DIR=/opt/rocm`, `ROCM_ROOT=/opt/rocm`, or `DEFAULT_ROCM_PATH=/opt/rocm`), and some had `ROCM_PATH` unset/empty.
   - Impact: mixed/incorrect search paths (“Looking in /opt/rocm…”, HIP root empty), risking ABI/version mismatches.
   - Fix:
     - `build_gfx1031.sh configure` forces `ROCM_PATH/ROCM_DIR/ROCM_ROOT` and `HIP_PATH/HIP_DIR/HIP_ROOT_DIR` to the in-tree toolchain root at `build/core/clr/dist`.
     - `base/CMakeLists.txt` passes these ROCm root variables explicitly to `amdsmi` to prevent it defaulting to `/opt/rocm`.
     - `base/CMakeLists.txt` also overrides `CPACK_PACKAGING_INSTALL_PREFIX` for `rocm-core` and `rocm_smi_lib` to avoid `/opt/rocm` leaking into caches via packaging defaults.
   - Recovery: requires a clean rebuild (`rm -rf build && ./build_gfx1031.sh configure && ./build_gfx1031.sh bootstrap && ./build_gfx1031.sh build ...`).

40. **2025-12-19: amd-llvm build failed due to -Werror (enum-constexpr-conversion)**
   - Failure: `amd-llvm` (spirv-llvm-translator) compiled with `-Werror` and failed on clang diagnostics:
     - `[-Wenum-constexpr-conversion]` in `llvm/ADT/DenseMapInfo.h`.
   - Fix:
     - Force `-DLLVM_ENABLE_WERROR=OFF` in `compiler/CMakeLists.txt` for the `amd-llvm` subproject.
     - Additionally, avoid editing the `compiler/spirv-llvm-translator` submodule by adding a narrow amd-llvm-only workaround in `compiler/pre_hook_amd-llvm.cmake`:
       - `add_compile_options($<$<CXX_COMPILER_ID:Clang>:-Wno-enum-constexpr-conversion>)`
   - Recovery: `ninja -C build amd-llvm+expunge` and resume the build.

41. **2025-12-19: Stage-1 / Stage-2 bootstrapping to avoid toolchain mixing**
   - Goal: avoid mixing the system toolchain (`/usr/lib/llvm-18`) and the in-tree
     ROCm toolchain across subprojects (can lead to phantom ABI/tool mismatch failures).
   - Approach:
     - Stage-1 in `build-stage1`: build the in-tree toolchain (`amd-llvm` + `hip-clr`)
       using system `clang/clang++`. Werror is disabled intentionally.
     - Stage-2 in `build-stage2`: fresh configure/build where the *top-level* CMake
       compilers/linker are set to the Stage-1 in-tree `clang/clang++/lld`, so even
       subprojects that forget `COMPILER_TOOLCHAIN` don't fall back to system clang.
   - Helpers:
     - `./build_gfx1031.sh configure --stage1` / `./build_gfx1031.sh bootstrap --stage1` / `./build_gfx1031.sh build --stage1`
     - `./build_gfx1031.sh configure --stage2` / `./build_gfx1031.sh bootstrap --stage2` / `./build_gfx1031.sh build --stage2`
   - Notes:
     - Do not switch compilers in-place inside a build directory; always use a new build dir.
     - `build_gfx1031.sh` supports `BUILD_DIR`, `STAGE`, and `STAGE1_BUILD_DIR` via config YAML/env/flags.

42. **2025-12-20: amd-llvm build failed in spirv-llvm-translator with newer clang (enum DenseMapInfo)**
   - Failure (Stage-2, clang 22): `SPIRVToOCL20.cpp` failed with errors in `llvm/ADT/DenseMapInfo.h` about enum sentinel values not being constant expressions (`spv::Op` has no fixed underlying type).
   - Fix: avoid `DenseMap<spv::Op, ...>` by storing the opcode as an integer key (`DenseMap<unsigned, ...>`) and casting on lookup in `compiler/spirv-llvm-translator/lib/SPIRV/SPIRVToOCL20.cpp`.
   - Recovery: `./build_gfx1031.sh expunge --stage2 amd-llvm && ./build_gfx1031.sh configure-sub --stage2 amd-llvm` then resume `./build_gfx1031.sh build --stage2 --detach`.

43. **2025-12-20: Stage-2 consistency check flagged `/opt/rocm` due to rocprofiler-sdk default + CMakeCache comments**
   - Symptom: `./test_gfx1031.sh --stage2 --consistency` failed the “no /opt/rocm in caches” check, even though the build did not actually use `/opt/rocm`:
     - `rocprofiler-sdk` cached `ROCPROFILER_DEFAULT_ROCM_PATH=/opt/rocm` (its upstream default).
     - Some `CMakeCache.txt` help comments mention `/opt/rocm` as the upstream default install prefix (non-functional).
   - Fix:
     - Pass `-DROCPROFILER_DEFAULT_ROCM_PATH=${DEFAULT_ROCM_PATH}` when configuring `rocprofiler-sdk` via `profiler/CMakeLists.txt`.
     - Make the cache scan ignore comment lines (`// ...`) in `test_gfx1031.sh`.
   - Recovery: `./build_gfx1031.sh rebuild --stage2 rocprofiler-sdk`, then rerun `./test_gfx1031.sh --stage2 --consistency`.

44. **2025-12-20: Deep cache scan initially flagged packaging defaults (`CPACK_*` / `CMAKE_INSTALL_PREFIX=/opt/rocm`)**
   - Symptom: `./test_gfx1031.sh --stage2 --consistency --deep` flagged internal packaging defaults like `CPACK_PACKAGING_INSTALL_PREFIX=/opt/rocm` (e.g. in `hipify`) and nested ExternalProject caches (e.g. `rocr-runtime` under `amd-llvm` runtimes).
   - Resolution: keep the deep scan “strict on effective search roots” but exclude known-benign packaging defaults and nested internal caches (still scans everything else, and runs `ldd` checks).

45. **2025-12-20: Build benchmark tools (rocblas-bench/hipblas-bench) without enabling full BUILD_TESTING**
   - Goal: run micro-benchmarks via `./test_gfx1031.sh --bench --stage2` without turning on the full test stack.
   - Change:
     - Added `THEROCK_BUILD_BENCHMARKS` (defaults to `THEROCK_BUILD_TESTING` if unset).
     - `build_gfx1031.sh configure` sets `-DTHEROCK_BUILD_BENCHMARKS=ON` when `config_gfx1031.yaml: build.benchmarks: true` (or `ENABLE_BENCHMARKS=true`).
     - BLAS subprojects (rocBLAS/hipBLAS/rocSPARSE/hipSPARSE etc.) now use `THEROCK_BUILD_BENCHMARKS` for `BUILD_CLIENTS_BENCHMARKS` while keeping `BUILD_CLIENTS_TESTS` tied to `THEROCK_BUILD_TESTING`.
   - Workflow:
     - Set `build.benchmarks: true`, then `./build_gfx1031.sh configure --stage2 --no-clean --no-check-clean && ./build_gfx1031.sh rebuild --stage2 rocBLAS hipBLAS`.

46. **2025-12-20: Bench binaries required host OpenBLAS runtime in `lib/host-math/lib`**
   - Symptom: `rocblas-bench`/`hipblas-bench` existed in `dist/rocm/bin` but failed at runtime with:
     - `error while loading shared libraries: librocm-openblas.so.0: cannot open shared object file`.
   - Root cause: host-blas lives under `dist/rocm/lib/host-math/lib` and some packaging flows did not preserve the SONAME symlink (`librocm-openblas.so.0`).
   - Fix:
     - `third-party/host-blas/CMakeLists.txt` installs compatibility copies (`librocm-openblas.so.0` + `librocm-openblas.so`) alongside the versioned file.
     - `test_gfx1031.sh` adds `.../lib/host-math/lib` to `LD_LIBRARY_PATH` and has a local fallback (temp symlink) so benchmarks run even if the dist is missing the SONAME link.

47. **2025-12-20: Bench enablement required additional build deps + dist refresh**
   - Symptom: enabling benchmark clients for rocBLAS/rocSOLVER/hipSOLVER initially failed to link because the client code expects a reference BLAS/LAPACK implementation (CBLAS/LAPACK symbols).
   - Fix:
     - Add `therock-host-blas` as a `BUILD_DEPS` when `THEROCK_BUILD_BENCHMARKS` is on (rocBLAS, rocSOLVER, hipSOLVER, hipBLAS).
     - Ensure bench binaries are packaged into `dist/rocm/bin` by running `dist-rocm` after rebuilding the affected subprojects.
   - Result: `./test_gfx1031.sh --stage2 --bench` runs `rocblas-bench` + `hipblas-bench` successfully (GFLOPS/TFLOPS parsed from CSV output).

48. **2026-02-08: Build PyTorch from source against in-tree ROCm 7.11**
   - Profile: `validation/config/profiles/pytorch_rocm711_source.yaml` (in-tree backend).
   - Run:
     ```
     python3 validation/validate.py --profile pytorch_rocm711_source --build-dirs build-stage2 --yes --power --log
     ```
   - Build dependency: PyTorch's HIP tooling expects `find_package(hipblaslt REQUIRED)`, so `hipblaslt` must be present under `build-stage2/dist/rocm/lib/cmake/hipblaslt`.
   - Fixes needed for a working install+import:
     - **ABI/mangling mismatch (host clang++ vs HIP)**: PyTorch injects `-fclang-abi-compat=17` into HIP compilation units; host C++ must mirror it or `import torch` can fail with `libtorch_hip.so: undefined symbol: ...const_data_ptr<Half>()`. Implemented in `validation/src/steps/workloads/pytorch/setup.py` for the in-tree source build.
     - **Version validation**: `torch.version.hip` reflects the HIP toolchain (can be 7.2.x) while `torch.version.rocm` reflects the ROCm release (7.11.x). The profile enforces `expected_rocm_substr: "7.11"`.
     - **In-tree runtime enforcement**: `require_rocm_prefix: in-tree` checks the loaded `libamdhip64.so` path via `/proc/self/maps`; fixed the `.so.<ver>` regex so `libamdhip64.so.7...` is detected (`validation/src/steps/workloads/pytorch/functional.py`).
     - **Wheel caching correctness**: always overwrite the cached wheel in `wheels_dir` after a rebuild to avoid accidentally reinstalling a stale/broken wheel (`validation/src/steps/workloads/pytorch/setup.py`).
   - Result (RX 6700 XT / gfx1031, `--power`): both Conv1d (audio) and Conv3d (video) validate with clear GPU activity (high `dW`/`gpu%`).
## TODO / Watchouts

- When new third-party packages are added, verify their `dist/` directories are populated before dependent projects configure.  
- GPU-focused warnings (hipBLASLt/hipSPARSELt/rocWMMA/composable_kernel) are expected on gfx1031 in this branch if those components are enabled; no action required yet.  
- Continue using serial builds unless we add explicit dependencies between stage/dist targets.
- Pending validation: helper automation (auto-venv + ccache 4.11.1 + clang/ninja) has not been executed in a fresh build yet.

48. **2026-03-18: ORT ROCm incremental provider rebuild did not pick up a changed source file**
   - Context: local debugging in the ONNX Runtime ROCm fork under:
     - source: `validation/workspace/cache/git/onnxruntime_rocm711/`
     - build: `validation/workspace/builds/onnxruntime_rocm/build-gfx1031-tlsfix-wheel/Release`
   - Symptom:
     - `onnxruntime/core/providers/rocm/rocm_allocator.cc` was modified and had a newer timestamp than the corresponding object file.
     - A normal incremental rebuild via:
       - `make -C validation/workspace/builds/onnxruntime_rocm/build-gfx1031-tlsfix-wheel/Release libonnxruntime_providers_rocm.so`
       - and even targeted `make` for `.../rocm_allocator.cc.o`
       returned immediately and did **not** refresh the object file timestamp.
     - Result: the produced `libonnxruntime_providers_rocm.so` could silently remain stale even though the source had changed.
   - Verified behavior:
     - source timestamp was newer than:
       - `Release/CMakeFiles/onnxruntime_providers_rocm.dir/.../rocm_allocator.cc.o`
     - but the object remained at the old timestamp until a manual direct compile was executed.
   - Reliable recovery in this state:
     - do not trust the incremental `make` result blindly.
     - force a real rebuild of the affected object (or clean/reconfigure the ORT build tree) and then relink `libonnxruntime_providers_rocm.so`.
     - after relink, verify that the wheel-staged provider in:
       - `validation/workspace/envs/py/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime_providers_rocm.so`
       actually matches the rebuilt `Release/libonnxruntime_providers_rocm.so`.
   - Practical lesson:
     - for ORT ROCm debugging/fixes in this repo, an apparently successful incremental rebuild is not sufficient evidence that the active provider binary contains the source change.
     - before trusting a new runtime result, confirm at least one of:
       - object timestamp changed,
       - provider `.so` timestamp changed,
       - expected new diagnostic string/symbol is present in the rebuilt library,
       - or the build tree was explicitly cleaned/reconfigured.
   - Consequence for future work:
     - treat the ORT ROCm build tree as potentially incrementally inconsistent until we either reproduce and fix the dependency tracking issue or standardize on a more explicit rebuild/verification step in the helper workflow.
   - Follow-up:
     - `validation/scripts/onnxruntime_rocm/verify_onnxruntime_rocm_provider_sync.sh`
       now exists to check source/object freshness plus Release/active-provider
       sync before trusting local ORT validation/debug results.

49. **2026-03-20: TheRock ORT wrapper now defaults to in-tree ROCm and forwards ccache launchers**
   - The integration wrapper `validation/scripts/onnxruntime_rocm/build_onnxruntime_rocm_wheel.sh`
     now defaults `ROCM_PATH` to `build-stage2/dist/rocm` so TheRock-side ORT
     builds no longer silently fall back to `/opt/rocm`.
   - When `ccache` is available and `ORT_USE_CCACHE!=0`, the wrapper now exports:
     - `ORT_CMAKE_C_COMPILER_LAUNCHER=ccache`
     - `ORT_CMAKE_CXX_COMPILER_LAUNCHER=ccache`
   - The ORT fork helper consumes those settings as:
     - `CMAKE_C_COMPILER_LAUNCHER`
     - `CMAKE_CXX_COMPILER_LAUNCHER`
   - Result:
     - the ORT wheel build path now matches the repo-wide policy better:
       custom in-tree ROCm by default, and ccache-backed incremental builds when
       available.
