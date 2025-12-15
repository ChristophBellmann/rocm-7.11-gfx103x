# ROCm & HIP Verification Checklist

Use these quick commands to validate that Christoph’s gfx1031 low-memory build and system install are healthy before you layer on AI tools (LLMs, etc.).

## 1. Shell environment

```bash
source /etc/profile.d/rocm-therock.sh
env | grep -E "ROCM_PATH|HIP_PATH|LD_LIBRARY_PATH"
```

Confirms the `/opt/rocm` stack is loaded so every HIP binary uses the new toolchain.

## 2. Runtime detection

```bash
rocminfo | head
rocm-smi --showproductname
hipconfig --version
```

- `rocminfo` shows the gfx1031 HSA runtime  
- `rocm-smi` should report **AMD Radeon RX 6700 XT**  
- `hipconfig --version` proves the TheRock HIP toolchain (7.2.254xx) is on `PATH`

## 3. HIP compile/run smoke test

```bash
cat <<'EOF' >/tmp/hip_vector_add.cpp
#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void add(float *a, float *b, float *c) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    c[i] = a[i] + b[i];
}
int main() {
    constexpr int N = 1024;
    float *a, *b, *c;
    hipMalloc(&a, N*sizeof(float));
    hipMalloc(&b, N*sizeof(float));
    hipMalloc(&c, N*sizeof(float));
    hipMemset(a, 0, N*sizeof(float));
    hipMemset(b, 0, N*sizeof(float));
    hipLaunchKernelGGL(add, dim3(16), dim3(64), 0, 0, a, b, c);
    hipDeviceSynchronize();
    hipFree(a); hipFree(b); hipFree(c);
    printf("hip vector add ok\n");
}
EOF
hipcc /tmp/hip_vector_add.cpp -o /tmp/hip_vector_add
/tmp/hip_vector_add
```

Expect `hip vector add ok` and no runtime errors. This proves HIP kernels compile and execute.

## 4. Math library sanity checks

- `./build/math-libs/BLAS/rocBLAS/stage/bin/rocblas-bench -f gemm -r f32 -n 256` (requires the math clients build described below); if that still hits `rocblas_status_memory_error`, try `-f axpy -r f32_r -n 4` before increasing the size (our last run still aborted with that error and left `/tmp/rocblas-axpy.log` for reference).
- `./build/math-libs/FFT/rocFFT/stage/bin/rocfft-rider -i 1024 -n 1`  
- `./build/math-libs/BLAS/rocSPARSE/stage/bin/rocSparse-test 0` (if staged, gfx1031 build may skip hipSPARSELt but rocSPARSE should still work)

Pick one or two of these commands to confirm the heavy math stack loads correctly. Each prints device info and exits cleanly when working.

## 5. Optional profiler check

```bash
rocm-smi --showpower
rocprof --stats --event HSA_QUEUE_TLB_FLUSH -- rocminfo
```

Only required if you plan to run profiling; ensures ROCm tooling libraries can interrogate the GPU.

## Notes

- **Math clients:** run `./build_enable_math_clients.sh` (prints the low-memory caps and reconfigures CMake with `BUILD_CLIENTS_*` and OpenMP enabled) before re-running `BUILD_MEM_LIMIT_KB=32505856 ./build_low_memory.sh`. Once the benchmark binaries appear under `build/math-libs/BLAS/rocBLAS/stage/bin`/`build/math-libs/FFT/rocFFT/stage/bin`, rerun the commands above to exercise `rocblas-bench`/`rocfft-rider`. Reduce the workload (for example `-f axpy -r f32_r -n 4`) if you see `rocblas_status_memory_error`—even the smallest axpy run we executed still aborted with that error (see `/tmp/rocblas-axpy.log`). Remember `rocfft-rider` only exists after the math-client rebuild so the command fails until that reconfiguration completes.  
- Always rerun the bannered `build_low_memory.sh` (with `BUILD_MEM_LIMIT_KB=32505856`) if you rebuild ROCm — the helper prints the active limits and pre-creates sparse client paths.  
- After rerunning installers or rebuilding, remember to `source /etc/profile.d/rocm-therock.sh` before rerunning these tests so the environment matches.
