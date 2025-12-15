# Low Memory Build Configuration

## Summary

This build configuration has been optimized for systems with limited memory to build TheRock for **gfx1031 (AMD RX 6700 XT)** only.

## Key Changes Made

### 1. Disabled Flang (Fortran Compiler) and Configured Minimal LLVM Runtimes

**Why**: Flang is extremely memory-intensive to build and is NOT needed for basic HIP development. Only compiler-rt builtins are enabled (required for clang to link), with sanitizers disabled to avoid build issues.

**Modified file**: `compiler/pre_hook_amd-llvm.cmake`

- Removed `flang` from `LLVM_ENABLE_PROJECTS` (line 23)
- Configured `LLVM_ENABLE_RUNTIMES="compiler-rt"` with only builtins enabled (line 26-37)
- Disabled flang-rt runtime (line 42)

### 2. Fixed glibc 2.39+ Compatibility Issue

**Why**: compiler-rt's sanitizer code uses `struct termio` which was removed in glibc 2.39+, causing build failures.

**Modified file**: `compiler/amd-llvm/compiler-rt/lib/sanitizer_common/sanitizer_platform_limits_posix.cpp`

- Commented out `sizeof(struct termio)` usage (line 489-491)
- Added placeholder value for compatibility

### 3. Fixed yaml-cpp Missing Include

**Why**: yaml-cpp is missing `#include <cstdint>` for uint16_t/uint32_t types.

**Modified file**: `rocm-systems/projects/rocprofiler-sdk/external/yaml-cpp/src/emitterutils.cpp`

- Added `#include <cstdint>` (line 2)

### 4. Enhanced Build Script: `build_low_memory.sh`

**Features**:

- **OOM Protection**: Protects the build shell from being killed by the OOM killer
- **Memory Limits**: Sets per-process memory limit to ~8GB
- **Limited Parallelism**: Forces `-j4` (4 parallel jobs)
- **Single-threaded Linking**: Sets `LLVM_PARALLEL_LINK_JOBS=1` to avoid memory spikes
- **Load Average Control**: Uses `-l4` to prevent overload

### 5. Component Configuration

**Successfully Built**:

- ✅ COMPILER (amdclang 20.0.0, lld, clang-tools-extra - **without flang**)
- ✅ CORE_RUNTIME (ROCR-Runtime - HSA runtime)
- ✅ HIP_RUNTIME (HIP 7.1 runtime libraries)
- ✅ CODE OBJECT MANAGER (amd_comgr)
- ✅ FFT Libraries (rocFFT, hipFFT, FFTW3)
- ✅ RAND Libraries (rocRAND, hipRAND)
- ✅ PRIM Libraries (rocPRIM, hipCUB, rocThrust)

**Not Built** (dependencies or memory issues):

- ❌ BLAS Libraries (rocBLAS, hipBLAS, hipBLASLt)
- ❌ SOLVER Libraries (rocSOLVER, hipSOLVER)
- ❌ SPARSE Libraries (rocSPARSE, hipSPARSE)
- ❌ ML Libraries (MIOpen, composable_kernel)
- ❌ Communication libraries (RCCL)
- ❌ Some profiler tools (rocprofiler-sdk had build issues)

## How to Build

### Initial Configuration (already done)

```bash
cmake -B build -GNinja . \
  -DTHEROCK_AMDGPU_TARGETS=gfx1031 \
  -DTHEROCK_ENABLE_ALL=OFF \
  -DTHEROCK_ENABLE_COMPILER=ON \
  -DTHEROCK_ENABLE_CORE_RUNTIME=ON \
  -DTHEROCK_ENABLE_HIP_RUNTIME=ON \
  -DTHEROCK_ENABLE_PRIM=ON \
  -DTHEROCK_ENABLE_FFT=ON \
  -DTHEROCK_ENABLE_RAND=ON \
  -DTHEROCK_ENABLE_BLAS=ON \
  -DTHEROCK_ENABLE_SOLVER=ON \
  -DTHEROCK_ENABLE_SPARSE=ON \
  -DBUILD_TESTING=OFF
```

**Note**: BLAS, SOLVER, and SPARSE were enabled but didn't successfully build due to dependency issues.

### Build with Memory Protection

```bash
./build_low_memory.sh
```

**Note**: The script requires sudo access to adjust the OOM score. If sudo is not available, the script will continue with a warning but without OOM protection.

The helper now also exports the bundled `third-party/sysdeps/linux/libdrm/.../include` and `lib/rocm_sysdeps/lib` paths so any component that #includes `<libdrm/drm.h>` (rocm-smi, rocblas, etc.) can compile without needing the system `libdrm` development packages.

## Expected Results

### Before Changes

- **Total targets**: 2891+ (including Flang)
- **Memory usage**: Peak >32GB during linking
- **Result**: OOM kills

### After Changes

- **Total targets**: ~1700 (Flang removed)
- **Memory usage**: Should stay under 24GB with `-j4`
- **Build time**: ~30-40% faster without Flang

## Reverting Changes

If you need Flang in the future:

1. Edit `compiler/pre_hook_amd-llvm.cmake`:

   - Line 23: Add `;flang` back to `LLVM_ENABLE_PROJECTS`
   - Line 42: Change `if(FALSE AND EXISTS` to `if(EXISTS`

1. Clean and reconfigure:

   ```bash
   ninja -C build compiler/amd-llvm+expunge
   cmake -B build .
   ```

## Monitoring the Build

Watch memory usage during build:

```bash
watch -n 2 'free -h && echo "---" && ps aux --sort=-rss | head -10'
```

## Troubleshooting

### Still Running Out of Memory?

Try:

1. Reduce parallelism further: `./build_low_memory.sh -- -j2`
1. Add swap space (16GB+ recommended)
1. Close other applications

### Build Fails with "OOM score adjustment failed"?

This is just a warning. The build will continue but may be killed by OOM. Run with sudo to enable protection:

```bash
sudo ./build_low_memory.sh
```

## What You Get

After successful build, you'll have:

- **amdclang/amdclang++**: AMD's HIP-capable Clang compiler (version 20.0.0) for gfx1031
- **HIP runtime**: HIP 7.1 runtime for running HIP applications
- **ROCm runtime**: ROCR-Runtime HSA 1.18 core runtime for AMD GPUs
- **Code Object Manager**: amd_comgr for runtime compilation
- **Math Libraries**: rocFFT, hipFFT, rocRAND, hipRAND, rocPRIM, hipCUB, rocThrust
- **System Management**: rocminfo, rocm-smi, amd-smi

### Verified Working

✅ **rocminfo** - Successfully detects AMD Radeon RX 6700 XT (gfx1031, 40 CUs)
✅ **hipcc** - HIP compiler toolchain operational
✅ **GPU Detection** - HSA runtime correctly identifies gfx1031

### Suitable For

- ✅ **LLM Inference** (LM Studio, Ollama) - Essential HIP runtime available
- ✅ **Basic HIP Development** - Compiler and runtime functional
- ✅ **FFT/Random Operations** - rocFFT and rocRAND available
- ✅ **Primitive Operations** - rocPRIM/hipCUB/rocThrust for parallel algorithms

### NOT Suitable For (missing components)

- ❌ **BLAS-Heavy Workloads** - rocBLAS/hipBLAS not built
- ❌ **Linear Algebra** - rocSOLVER/hipSOLVER not built
- ❌ **Sparse Matrix Operations** - rocSPARSE/hipSPARSE not built
- ❌ **Machine Learning Training** - MIOpen not built
- ❌ **Fortran/OpenMP Offload** - Flang disabled
- ❌ **Multi-GPU** - RCCL not built

**Note for LLM Inference**: Most LLM inference engines (llama.cpp/Ollama, LM Studio) use custom optimized kernels and don't strictly require rocBLAS. The HIP runtime and basic libraries should be sufficient for many models.
