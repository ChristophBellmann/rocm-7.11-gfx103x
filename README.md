# TheRock gfx1031 fork

This repository is *your* gfx1031-focused fork of [`ROCm/TheRock`](https://github.com/ROCm/TheRock).
The entire `make_my_gpu_useful` tree now lives under `/media/christoph/some_space/` so it can stay isolated from system installs until you're ready to move forward with LM Studio/Ollama. Keep working from that mount until you explicitly set up the system-wide release.
Only the files that support the RX 6700 XT (gfx1031 / RDNA2) workflow are versioned here:

- native `gfx1031` target configuration (see `cmake/therock_amdgpu_targets.cmake`),
- low‑memory build helpers, install scripts, and the IDEA/LLM tooling that consumes `/opt/rocm`,
- the documentation and diagnostics (`docs/`, `BUILD_SUCCESS*.md`, `scripts/`, etc.) that describe what is “working” for you.

## How to use this repo

1. **Read the custom guide** at [`README_CUSTOM.md`](README_CUSTOM.md) – it contains the step-by-step prerequisites, build commands, and install instructions specifically validated for the RX 6700 XT.
2. **Check the cleanup/governance doc** at [`docs/REPO_SCOPE.md`](docs/REPO_SCOPE.md) to understand which files are required and what can be safely considered “upstream noise.”
3. **Launch builds** via `build_low_memory.sh`, `install_systemwide.sh`, or whichever helper matches your workflow; the accompanying scripts in `scripts/` keep things reproducible.

## HIP sanity test

Run `./test_hip_hello.sh` to compile and execute `hello.cpp` against the ROCm build tree; the script exports `ROCM_PATH`/`HIP_PATH`, points `PATH`/`LD_LIBRARY_PATH` at `build/dist/rocm`, and passes `--rocm-device-lib-path` so HIP finds the AMD bitcode libraries. The test prints each GPU thread plus “Hello from host,” proving that HIP offload works end-to-end from this repo.

The repository ships `test_hip_hello.sh` so anyone can rerun the same compilation + execution workflow with a single command after rebuilding or before installing additional software.

If you only need to inspect the HIP stack, run `./run_hipconfig.sh`; it sets the same ROCm environment and executes `hipconfig --full` from `build/dist/rocm` so the tool reports the build tree’s paths instead of relying on `/opt/rocm-*`.

When you want to see how the GPU’s power/clock state reacts to HIP work, `./monitor_hip_power.sh` loops the hello workload in the background for ~5 s while logging `rocm-smi --showtemp --showclocks --showproductname` snapshots every second; it keeps the ROCm env baked in so the measurement is completely self-contained.

If you only care about exercising the GPU without extra stdout (for example, to monitor `rocm-smi` or another tool from another terminal), run `./run_hip_workload.sh <seconds>` (defaults to 20 s); it pumps the same HIP hello kernel to `/dev/null` for the requested duration with the local ROCm env already configured. Ctrl+C stops the loop early.

For a heavier arithmetic test, `./run_hip_gemm.sh [busy_seconds]` compiles and runs a 2048² matrix multiplication kernel (200 iterations) to validate correctness, then increases the matrix dimension over the next `busy_seconds` (default 10 s) in 1‑second segments until `rocm-smi --showmeminfo vram` reports ~50 % VRAM usage while the GPU stays busy (each segment fires the quiet kernel multiple times so you can monitor `rocm-smi` in another terminal).  

The kernel uses `hipDeviceSynchronize()` to block the host until the kernel finishes; the returned `hipError_t` reports whether the previous GPU work succeeded (or if a launch/runtime error occurred), so you can either check it or explicitly cast to `(void)` when you only care about the synchronization side effect.

Everything else (generic TheRock docs, nightlies, OS-specific setup) lives in the original project. If you ever need to compare to upstream, `~/rocm_repo` holds a clean clone so you can diff against it without affecting this repo.
