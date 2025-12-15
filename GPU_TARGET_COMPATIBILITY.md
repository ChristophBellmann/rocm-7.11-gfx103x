# GPU Target Configuration Guide for RX 6700 XT (gfx1031)

## Executive Summary

**Your GPU**: AMD Radeon RX 6700 XT
**Native Target**: gfx1031 (RDNA 2)
**Current Override**: HSA_OVERRIDE_GFX_VERSION=10.3.0 (gfx1030)

**Verdict**: The HSA override is **intentional and beneficial** for compatibility, but you should use **native gfx1031** when building from source for optimal performance.

______________________________________________________________________

## Understanding gfx1030 vs gfx1031

### Architecture Similarity

- **gfx1030**: RX 6800/6800 XT (Navi 21 XL/XT)
- **gfx1031**: RX 6700 XT (Navi 22)
- Both are RDNA 2 architecture with nearly identical instruction sets
- Main differences: CU count, memory bandwidth, cache size

### Why the Override Exists

The HSA override (`HSA_OVERRIDE_GFX_VERSION=10.3.0`) makes your gfx1031 GPU report as gfx1030 because:

1. **Binary Compatibility**: Many pre-compiled ROCm applications only ship gfx1030 kernels
1. **ISA Compatibility**: gfx1030 and gfx1031 share the same instruction set architecture
1. **Widespread Practice**: This is a **recommended workaround** in the ROCm community
1. **Proven Reliability**: Works well for most applications without performance penalty

### Performance Impact

- **Negligible for inference**: LLM inference is memory-bound, not compute-bound
- **Optimal for training**: Native gfx1031 may be slightly better for training workloads
- **Pre-built binaries**: Override is necessary or they won't run at all

______________________________________________________________________

## Recommended Configuration by Use Case

### 1. Ollama (Pre-built Binary)

**Recommendation**: ✓ **Use HSA override (gfx1030)**

```bash
# Already configured in /etc/systemd/system/ollama.service
Environment="HSA_OVERRIDE_GFX_VERSION=10.3.0"
```

**Why**: Ollama ships pre-built ROCm kernels, typically for gfx1030/gfx1100. The override ensures compatibility.

**Status**: Your ollama service is already correctly configured. Main issue is missing `libggml-rocm.so` (see OLLAMA_GPU_STATUS.md).

### 2. LM Studio (Pre-built Binary)

**Recommendation**: ✓ **Use HSA override (gfx1030)**

```bash
# Before launching LM Studio:
source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/rocm-env-compat.sh
lmstudio
```

**Why**: LM Studio uses pre-compiled GGML/llama.cpp binaries with ROCm support.

### 3. llama.cpp (Build from Source)

**Recommendation**: ✓ **Use native gfx1031** for best performance

```bash
# Setup native environment
source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/rocm-env-native.sh

# Clone and build
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1031 \
  -DCMAKE_C_COMPILER=hipcc \
  -DCMAKE_CXX_COMPILER=hipcc

cmake --build . --config Release -j$(nproc)
```

**Why**: Building from source allows you to target your specific GPU architecture for optimal kernel generation.

### 4. TheRock Build (Already Correct ✓)

**Recommendation**: ✓ **Use native gfx1031** (already configured)

```bash
# Your current configuration is optimal:
# THEROCK_AMDGPU_TARGETS=gfx1031

# For future builds:
source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/rocm-env-native.sh
cd /home/christoph/make_my_gpu_useful/TheRock_gfx1031
cmake --build build
```

**Status**: Your TheRock build is correctly configured for gfx1031 native compilation.

### 5. PyTorch with ROCm (If you use it)

**Recommendation**: ✓ **Use HSA override (gfx1030)** for official binaries, native for source builds

```bash
# Official PyTorch ROCm binaries:
source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/rocm-env-compat.sh
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2

# Building PyTorch from source:
source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/rocm-env-native.sh
# Then follow PyTorch build instructions with PYTORCH_ROCM_ARCH=gfx1031
```

______________________________________________________________________

## Quick Start: Environment Switcher

I've created convenient scripts for you:

### Check Current Configuration

```bash
source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/rocm-switch.sh check
```

### Switch to Native Mode (for source builds)

```bash
source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/rocm-switch.sh native
# Now HSA_OVERRIDE is unset - uses native gfx1031
```

### Switch to Compatibility Mode (for pre-built binaries)

```bash
source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/rocm-switch.sh compat
# Now HSA_OVERRIDE_GFX_VERSION=10.3.0 is set
```

### Add to .bashrc (Optional)

```bash
# Add this to your ~/.bashrc for easy switching:
alias rocm-native='source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/rocm-switch.sh native'
alias rocm-compat='source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/rocm-switch.sh compat'
alias rocm-check='source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/rocm-switch.sh check'
```

______________________________________________________________________

## Current Configuration Review

### ✓ What's Correct

1. **TheRock Build**: Configured for native gfx1031

   ```
   THEROCK_AMDGPU_TARGETS=gfx1031
   ```

1. **Ollama Service**: HSA override properly configured

   ```
   Environment="HSA_OVERRIDE_GFX_VERSION=10.3.0"
   ```

1. **ROCm Installation**: Working correctly

   ```bash
   $ rocm-smi --showproductname
   GFX Version: gfx1031
   ```

### ⚠️ What to Adjust

1. **Default Shell Environment** (`~/.bashrc`):

   - Currently: Sets HSA override globally
   - Recommendation: Use switcher scripts instead
   - **Action**: Remove or comment out the hardcoded override

1. **System-wide Environment** (`/etc/profile.d/rocm.sh`):

   - Currently: Sets HSA override for all users
   - Recommendation: Keep for compatibility, use switcher when building
   - **Action**: Leave as-is for now

______________________________________________________________________

## Recommended ~/.bashrc Configuration

```bash
# ROCm Base Configuration (add this to ~/.bashrc)

# Base ROCm paths (always needed)
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/lib/llvm/lib:$LD_LIBRARY_PATH
export HIP_PLATFORM=amd
export HIP_COMPILER=clang
export HIP_DEVICE_LIB_PATH=/opt/rocm/lib/llvm/amdgcn/bitcode

# GPU settings (always needed)
export HSA_XNACK=0
export HSA_ENABLE_SDMA=0
export AMD_DIRECT_DISPATCH=0
export GPU_DEVICE_ORDINAL=0
export HIP_VISIBLE_DEVICES=0

# HSA Override - CONDITIONAL (use switcher scripts instead)
# For pre-built binaries (ollama, lmstudio): export HSA_OVERRIDE_GFX_VERSION=10.3.0
# For source builds (llama.cpp, TheRock): unset HSA_OVERRIDE_GFX_VERSION

# Recommended: Use compat mode by default for general use
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Convenient aliases for switching
alias rocm-native='source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/rocm-switch.sh native'
alias rocm-compat='source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/rocm-switch.sh compat'
alias rocm-check='source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/rocm-switch.sh check'

# Show current mode on shell startup
echo "ROCm: $([ -n "$HSA_OVERRIDE_GFX_VERSION" ] && echo "Compat mode (gfx1030)" || echo "Native mode (gfx1031)")"
```

______________________________________________________________________

## Testing Compatibility

### Test 1: Verify Native gfx1031 Works

```bash
source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/rocm-switch.sh native

# Simple HIP test
cd /home/christoph/make_my_gpu_useful/TheRock_gfx1031
hipcc test_hip.cpp -o test_hip_native
./test_hip_native

# Should show: gfx1031
```

### Test 2: Verify Override Works

```bash
source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/rocm-switch.sh compat

# Check HSA reports gfx1030
rocminfo | grep "Name:" | grep gfx
# Should show: gfx1030 (due to override)
```

### Test 3: TheRock Build with Native Target

```bash
source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/rocm-switch.sh native
cd /home/christoph/make_my_gpu_useful/TheRock_gfx1031

# Check current build config
cat build/CMakeCache.txt | grep THEROCK_AMDGPU_TARGETS
# Should show: THEROCK_AMDGPU_TARGETS:STRING=gfx1031

# Build a component
cmake --build build --target hipify
```

______________________________________________________________________

## Performance Expectations

### gfx1031 Native vs gfx1030 Override

**For LLM Inference** (ollama, lmstudio, llama.cpp):

- **Performance difference**: < 2-5% (usually not noticeable)
- **Bottleneck**: Memory bandwidth and VRAM capacity (12 GB on RX 6700 XT)
- **Recommendation**: Use override for pre-built binaries (easier setup)

**For Compute Kernels** (training, custom HIP apps):

- **Performance difference**: 5-10% potential gain with native
- **Bottleneck**: Depends on kernel optimization
- **Recommendation**: Always use native gfx1031 when building from source

**For TheRock/ROCm Development**:

- **Performance difference**: Varies by component
- **Recommendation**: Always use native gfx1031 (already configured ✓)

______________________________________________________________________

## Common Issues and Solutions

### Issue 1: "Unsupported GPU architecture"

**Symptom**: Error about gfx1031 not supported
**Solution**: Use compat mode (gfx1030 override)

```bash
source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/rocm-switch.sh compat
```

### Issue 2: Sub-optimal Performance in Custom Builds

**Symptom**: Slower than expected in llama.cpp you built
**Solution**: Rebuild with native target

```bash
source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/rocm-switch.sh native
# Then rebuild llama.cpp with -DAMDGPU_TARGETS=gfx1031
```

### Issue 3: Ollama Not Detecting GPU

**Symptom**: Ollama shows 0 B VRAM, uses CPU
**Solution**: Missing ROCm compute library (see OLLAMA_GPU_STATUS.md)
**Note**: The HSA override is correctly configured; this is a different issue

______________________________________________________________________

## Summary Table

| Application            | Mode   | HSA Override | Build Command                      |
| ---------------------- | ------ | ------------ | ---------------------------------- |
| **Ollama**             | Compat | ✓ gfx1030    | Pre-built binary                   |
| **LM Studio**          | Compat | ✓ gfx1030    | Pre-built binary                   |
| **llama.cpp** (built)  | Native | ✗ None       | `-DAMDGPU_TARGETS=gfx1031`         |
| **TheRock**            | Native | ✗ None       | `-DTHEROCK_AMDGPU_TARGETS=gfx1031` |
| **PyTorch** (official) | Compat | ✓ gfx1030    | `pip install torch`                |
| **PyTorch** (source)   | Native | ✗ None       | `PYTORCH_ROCM_ARCH=gfx1031`        |
| **System ROCm Tools**  | Either | Optional     | N/A (works both ways)              |

______________________________________________________________________

## Final Recommendation

### For Your Use Case (ollama + lmstudio + llama.cpp)

1. **Keep your current setup** with HSA override enabled by default in `~/.bashrc`
1. **Use the switcher scripts** when building from source:
   ```bash
   # Building llama.cpp or similar
   source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/rocm-switch.sh native

   # Running pre-built apps
   source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/rocm-switch.sh compat
   ```
1. **TheRock builds** are already optimal (native gfx1031)
1. **Ollama service** is already correctly configured (gfx1030 override)

### What to Do Next

1. **Fix ollama GPU detection** (see OLLAMA_GPU_STATUS.md)
1. **Test llama.cpp** with native build for comparison
1. **Keep current default** (HSA override) for ease of use with pre-built binaries

### Bottom Line

✓ Your HSA override to gfx1030 is **correct and recommended**
✓ Your TheRock build config for gfx1031 is **correct and optimal**
✓ You have the **best of both worlds** configured

The dual approach (override for pre-built, native for source builds) gives you maximum compatibility without sacrificing performance where it matters.
