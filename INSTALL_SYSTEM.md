# Installing TheRock ROCm System-Wide

This guide will help you install your TheRock build to `/opt/rocm` and configure Ollama to use it.

## Overview

Your TheRock build includes:

- **Native gfx1031 support** for AMD Radeon RX 6700 XT
- Full ROCm stack (compiler, runtime, libraries)
- 88 compiled components optimized for your GPU

## Installation Steps

### Step 1: Install TheRock to /opt/rocm

Run the installation script:

```bash
cd /home/christoph/make_my_gpu_useful/TheRock_gfx1031
sudo ./install_to_opt_rocm.sh
```

**What this does:**

- Backs up existing `/opt/rocm` to `/opt/rocm.backup-<timestamp>`
- Copies TheRock build from `build/dist/rocm` to `/opt/rocm`
- Configures system library paths (`/etc/ld.so.conf.d/rocm.conf`)
- Creates environment script (`/etc/profile.d/rocm.sh`)

### Step 2: Load the New Environment

Either:

- **Log out and log back in** (recommended), OR
- **Manually load**: `source /etc/profile.d/rocm.sh`

### Step 3: Verify ROCm Installation

Check that ROCm detects your GPU natively as gfx1031:

```bash
rocminfo | grep 'Name:' | head -3
```

Expected output:

```
  Name:                    AMD Ryzen 9 5900X 12-Core Processor
  Name:                    gfx1031
```

Check ROCm paths:

```bash
which rocminfo
# Should show: /opt/rocm/bin/rocminfo

hipcc --version
# Should show LLVM version 20.0.0
```

### Step 4: Update Ollama for ROCm

Run the Ollama update script:

```bash
cd /home/christoph/make_my_gpu_useful/TheRock_gfx1031
./update_ollama_rocm.sh
```

**What this does:**

- Stops any running Ollama processes
- Creates systemd user service with ROCm environment variables
- Configures Ollama to use `/opt/rocm` libraries
- Starts Ollama service automatically

### Step 5: Test Ollama with GPU

Check Ollama service status:

```bash
systemctl --user status ollama
```

Test GPU detection with a model:

```bash
ollama run llama3.2
```

Monitor Ollama logs to verify GPU usage:

```bash
journalctl --user -u ollama -f
```

## Environment Variables

The system-wide ROCm environment (`/etc/profile.d/rocm.sh`) sets:

```bash
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH
unset HSA_OVERRIDE_GFX_VERSION  # Native gfx1031 - no override needed!
```

## Ollama Service Configuration

The Ollama systemd service (`~/.config/systemd/user/ollama.service`) includes:

- **ROCm environment**: Automatically loads ROCm paths
- **Auto-restart**: Service restarts on failure
- **User service**: Runs as your user, not root

### Useful Ollama Commands

```bash
# Service management
systemctl --user status ollama    # Check status
systemctl --user restart ollama   # Restart service
systemctl --user stop ollama      # Stop service
systemctl --user start ollama     # Start service

# View logs
journalctl --user -u ollama       # All logs
journalctl --user -u ollama -f    # Follow logs
journalctl --user -u ollama -n 50 # Last 50 lines

# Ollama commands
ollama list                       # List installed models
ollama pull llama3.2              # Download a model
ollama run llama3.2               # Run a model
ollama ps                         # Show running models
```

## Verification Checklist

After installation, verify everything works:

- [ ] `rocminfo` shows `gfx1031` (not gfx1030)
- [ ] `which rocminfo` shows `/opt/rocm/bin/rocminfo`
- [ ] `hipcc --version` shows LLVM 20.0.0
- [ ] `systemctl --user status ollama` shows service is active
- [ ] `ollama list` works without errors
- [ ] `ollama run llama3.2` detects and uses GPU

## Troubleshooting

### Issue: rocminfo still shows old version

**Fix:**

```bash
# Verify path priority
echo $PATH | tr ':' '\n' | grep rocm
# /opt/rocm/bin should be first

# Reload environment
source /etc/profile.d/rocm.sh

# Or log out and back in
```

### Issue: Ollama service won't start

**Check logs:**

```bash
journalctl --user -u ollama -n 50
```

**Common fixes:**

```bash
# Verify Ollama binary exists
ls -l /usr/local/bin/ollama

# Check if port is in use
sudo netstat -tlnp | grep 11434

# Restart service
systemctl --user restart ollama
```

### Issue: Ollama not using GPU

**Check ROCm detection:**

```bash
# Should show your GPU
rocm-smi

# Check GPU visibility to Ollama
HSA_TOOLS_LIB=/opt/rocm/lib/librocprofiler64.so rocminfo
```

**Verify environment in service:**

```bash
systemctl --user show ollama --property=Environment
```

### Issue: Library not found errors

**Fix library paths:**

```bash
# Update library cache
sudo ldconfig

# Verify library paths
ldconfig -p | grep rocm

# Check library file
ldd /opt/rocm/bin/rocminfo
```

## Rollback Instructions

If you need to revert to the old ROCm installation:

```bash
# Stop Ollama
systemctl --user stop ollama

# Remove TheRock installation
sudo rm -rf /opt/rocm

# Restore backup (find the latest backup)
ls -lad /opt/rocm.backup-*
sudo mv /opt/rocm.backup-YYYYMMDD-HHMMSS /opt/rocm

# Reload environment
source /etc/profile.d/rocm.sh

# Restart Ollama
systemctl --user start ollama
```

## Performance Tips

### Monitor GPU Usage

```bash
# Watch GPU utilization
watch -n 1 rocm-smi

# Check GPU temperature and power
rocm-smi --showtemp --showpower
```

### Optimize for Your RX 6700 XT (gfx1031)

Your GPU specs:

- **40 Compute Units**
- **12GB VRAM**
- **2855 MHz max clock**
- **RDNA 2 architecture**

Recommended model sizes for 12GB VRAM:

- **7B models**: Excellent performance (llama3.2, mistral, etc.)
- **13B models**: Good performance with quantization
- **30B+ models**: Use aggressive quantization (Q4_0, Q4_K_M)

## Additional Resources

- **TheRock Documentation**: `docs/` directory in this repo
- **ROCm Documentation**: https://rocm.docs.amd.com/
- **Ollama Documentation**: https://github.com/ollama/ollama/tree/main/docs

## What You Get

✅ **Native gfx1031 support** - No more GPU override hacks
✅ **System-wide ROCm** - Available to all users
✅ **Optimized Ollama** - Automatic GPU acceleration
✅ **Latest ROCm** - Built from source with all components
✅ **Easy updates** - Rebuild and reinstall anytime

## Keeping TheRock Updated

To rebuild and update your system ROCm:

```bash
cd /home/christoph/make_my_gpu_useful/TheRock_gfx1031

# Pull latest changes
git pull

# Update submodules
python3 ./build_tools/fetch_sources.py

# Rebuild
cmake --build build

# Reinstall
sudo ./install_to_opt_rocm.sh

# Restart Ollama
systemctl --user restart ollama
```

______________________________________________________________________

**Installation Scripts:**

- `install_to_opt_rocm.sh` - Install TheRock to /opt/rocm
- `update_ollama_rocm.sh` - Configure Ollama for ROCm

**Created by:** Claude Code
**Date:** 2025-11-11
**Target:** AMD Radeon RX 6700 XT (gfx1031)
