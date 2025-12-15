# Ollama GPU Detection Status Report

## Summary

✓ **ROCm Environment**: Properly configured
✓ **Environment Variables**: All correctly set in systemd service
✓ **ROCm Installation**: Working at `/opt/rocm`
✗ **GPU Detection**: **NOT WORKING** - Missing ROCm compute library

## Problem Identified

Your Ollama installation (v0.12.9) at `/usr/local/bin/ollama` **does not have the ROCm compute library** required for AMD GPU acceleration.

### Evidence

1. **Missing Library**:

   ```bash
   ls /usr/local/lib/ollama/
   ```

   Shows:

   - ✓ CUDA libraries: `cuda_v12/libggml-cuda.so`, `cuda_v13/libggml-cuda.so`
   - ✓ CPU libraries: Multiple CPU-specific libraries
   - ✓ ROCm symlink: `rocm -> /opt/rocm`
   - ✗ **MISSING**: `libggml-rocm.so` or similar ROCm compute library

1. **Log Evidence**:

   ```
   time=2025-11-10T21:19:41.734-07:00 level=INFO source=runner.go:76 msg="discovering available GPUs..."
   time=2025-11-10T21:19:41.810-07:00 level=INFO source=types.go:60 msg="inference compute" id=cpu library=cpu compute="" name=cpu...
   time=2025-11-10T21:19:41.810-07:00 level=INFO source=routes.go:1618 msg="entering low vram mode" "total vram"="0 B"
   ```

   Ollama discovers GPUs but only finds CPU. No GPU detected, 0 B VRAM reported.

1. **Environment is Correct**:

   ```bash
   systemctl show ollama -p Environment
   ```

   Shows all ROCm variables properly set:

   - ROCM_PATH=/opt/rocm
   - HIP_PATH=/opt/rocm
   - HSA_OVERRIDE_GFX_VERSION=10.3.0
   - HIP_VISIBLE_DEVICES=0
   - LD_LIBRARY_PATH includes /opt/rocm/lib

## Solution Options

### Option 1: Reinstall Ollama with Official Script (RECOMMENDED)

The official Ollama install script should detect your ROCm installation and download the correct binary with ROCm support:

```bash
# Uninstall current Ollama
sudo systemctl stop ollama
sudo systemctl disable ollama
sudo rm -rf /usr/local/bin/ollama /etc/systemd/system/ollama.service /usr/local/lib/ollama

# Install with official script
curl -fsSL https://ollama.com/install.sh | sh

# Reapply our ROCm configuration
cd /home/christoph/make_my_gpu_useful/TheRock_gfx1031
sudo ./configure_ollama.sh
```

### Option 2: Build Ollama from Source with ROCm

If the official installer doesn't provide ROCm support:

```bash
# Clone Ollama repository
git clone https://github.com/ollama/ollama.git
cd ollama

# Build with ROCm support
go build -tags rocm .

# Install
sudo cp ollama /usr/local/bin/
sudo ./configure_ollama.sh
```

### Option 3: Check for ROCm-enabled Binary

Some distributions may have ROCm-enabled Ollama packages:

```bash
# Check if Fedora has an Ollama package with ROCm
sudo dnf search ollama

# Or check for Ollama with ROCm in other repos
```

## Verification After Fix

Once you have a ROCm-enabled Ollama, verify GPU detection:

```bash
# Check for ROCm library
ls -la /usr/local/lib/ollama/ | grep rocm

# Restart service
sudo systemctl restart ollama

# Check logs for GPU detection
journalctl -u ollama -f

# Test with a model
ollama run mistral:latest "test" --verbose
```

You should see:

- A `libggml-rocm.so` or similar library in `/usr/local/lib/ollama/`
- GPU detected in logs with VRAM amount
- GPU offload happening during model inference

## Current Configuration (Already Applied)

Your systemd service is properly configured with all necessary ROCm environment variables:

**File**: `/etc/systemd/system/ollama.service`

```ini
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3

Environment="ROCM_PATH=/opt/rocm"
Environment="HIP_PATH=/opt/rocm"
Environment="PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"
Environment="LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/lib/llvm/lib"
Environment="HIP_PLATFORM=amd"
Environment="HIP_COMPILER=clang"
Environment="HIP_DEVICE_LIB_PATH=/opt/rocm/lib/llvm/amdgcn/bitcode"
Environment="HIP_VISIBLE_DEVICES=0"
Environment="HSA_OVERRIDE_GFX_VERSION=10.3.0"
Environment="HSA_XNACK=0"
Environment="HSA_ENABLE_SDMA=0"
Environment="AMD_DIRECT_DISPATCH=0"
Environment="GPU_DEVICE_ORDINAL=0"

[Install]
WantedBy=default.target
```

Once you have a ROCm-enabled Ollama binary, this configuration will enable GPU acceleration immediately.

## Next Steps

1. Choose one of the solution options above
1. Reinstall or rebuild Ollama with ROCm support
1. Verify the ROCm compute library is present
1. Test GPU detection
1. Enjoy GPU-accelerated inference on your AMD RX 6700 XT!
