# System-wide ROCm Installation Instructions

## Quick Start

To install your TheRock build system-wide to `/opt/rocm`, simply run:

```bash
sudo bash /home/christoph/make_my_gpu_useful/TheRock_gfx1031/install_systemwide.sh
```

Enter your password when prompted. The script will:

1. ✓ Backup existing `/opt/rocm` to `/opt/rocm.backup.TIMESTAMP`
1. ✓ Copy TheRock build to `/opt/rocm` (2.4GB)
1. ✓ Set proper permissions (root:root)
1. ✓ Create system environment (`/etc/profile.d/rocm-therock.sh`)
1. ✓ Configure dynamic linker (`/etc/ld.so.conf.d/rocm-therock.conf`)
1. ✓ Verify installation

**Time required**: ~2-3 minutes

______________________________________________________________________

## After Installation

### Step 1: Reload Environment

Log out and log back in, or run:

```bash
source /etc/profile.d/rocm-therock.sh
```

### Step 2: Verify Installation

```bash
rocminfo | grep "Name:" | head -3
```

Expected output:

```
Name: AMD Ryzen 9 5900X 12-Core Processor
Name: gfx1031
```

### Step 3: Verify GPU Detection

```bash
rocm-smi --showproductname
```

Should show:

```
GFX Version: gfx1031
```

### Step 4: Test HIP Compiler

```bash
hipconfig --version
```

______________________________________________________________________

## Using with AI Tools

### Ollama

```bash
# Restart Ollama service to pick up new ROCm
systemctl --user restart ollama

# Test with a model
ollama run llama3.2
```

### LM Studio

```bash
# LM Studio will automatically use /opt/rocm
lms
```

### llama.cpp

```bash
# Rebuild llama.cpp with system ROCm
cd ~/llama.cpp
make clean
make LLAMA_HIPBLAS=1
```

______________________________________________________________________

## Manual Installation (Alternative)

If you prefer to run commands manually:

### 1. Backup Existing ROCm

```bash
sudo mv /opt/rocm /opt/rocm.backup.$(date +%Y%m%d_%H%M%S)
```

### 2. Copy TheRock Build

```bash
sudo cp -a /home/christoph/make_my_gpu_useful/TheRock_gfx1031/build/dist/rocm /opt/rocm
```

### 3. Set Permissions

```bash
sudo chown -R root:root /opt/rocm
sudo chmod -R a+rX /opt/rocm
sudo chmod -R u+w /opt/rocm
```

### 4. Create System Environment

```bash
sudo tee /etc/profile.d/rocm-therock.sh << 'EOF'
# ROCm Environment - TheRock Build
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH}
unset HSA_OVERRIDE_GFX_VERSION
EOF
```

### 5. Configure Dynamic Linker

```bash
sudo tee /etc/ld.so.conf.d/rocm-therock.conf << 'EOF'
/opt/rocm/lib
/opt/rocm/lib64
/opt/rocm/lib/llvm/lib
EOF

sudo ldconfig
```

______________________________________________________________________

## Rollback (If Needed)

To restore the previous installation:

```bash
# Find your backup
ls -ld /opt/rocm.backup.*

# Restore it (replace timestamp with yours)
sudo rm -rf /opt/rocm
sudo mv /opt/rocm.backup.TIMESTAMP /opt/rocm

# Update ldconfig
sudo ldconfig
```

______________________________________________________________________

## What Gets Installed

Your system-wide ROCm installation includes:

### Compiler Toolchain

- LLVM 20.0.0git with AMDGPU backend
- Clang C/C++ compiler
- LLD linker
- HIP compiler (hipcc)

### Runtime Components

- HIP runtime (libamdhip64.so)
- HSA runtime (ROCR-Runtime)
- Device libraries (gfx1031 optimized)

### Math Libraries

- rocBLAS, hipBLAS, hipBLASLt
- rocFFT, hipFFT
- rocRAND, hipRAND
- rocSOLVER, hipSOLVER
- rocSPARSE, hipSPARSE

### ML Libraries

- MIOpen (deep learning primitives)

### Tools & Utilities

- rocminfo (GPU information)
- rocm-smi (GPU monitoring)
- offload-arch (detect GPU targets)

______________________________________________________________________

## Environment Variables Set

After installation, these will be set system-wide:

```bash
ROCM_PATH=/opt/rocm
HIP_PATH=/opt/rocm
PATH=/opt/rocm/bin:$PATH
LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH
```

**Important**: `HSA_OVERRIDE_GFX_VERSION` is explicitly **unset** because you have native gfx1031 support!

______________________________________________________________________

## Troubleshooting

### Issue: Command not found after installation

**Cause**: Environment not loaded

**Fix**:

```bash
source /etc/profile.d/rocm-therock.sh
# Or log out and log back in
```

### Issue: Library not found errors

**Cause**: ldconfig cache not updated

**Fix**:

```bash
sudo ldconfig
```

### Issue: Ollama still using old ROCm

**Cause**: Service needs restart

**Fix**:

```bash
systemctl --user restart ollama
# Check which rocminfo it uses
which rocminfo  # Should be /opt/rocm/bin/rocminfo
```

### Issue: GPU shows as gfx1030 instead of gfx1031

**Cause**: Old override variable still set in user profile

**Fix**:

```bash
# Check your shell config files
grep HSA_OVERRIDE ~/.bashrc ~/.bash_profile ~/.profile

# Remove any lines setting HSA_OVERRIDE_GFX_VERSION
# Then reload environment
source /etc/profile.d/rocm-therock.sh
```

______________________________________________________________________

## Benefits of System-wide Installation

✓ **System-wide access** - All users can use ROCm
✓ **Standard paths** - `/opt/rocm` is the expected location
✓ **Easy integration** - AI tools automatically find ROCm
✓ **Clean environment** - No per-user configuration needed
✓ **Native gfx1031** - No more architecture overrides

______________________________________________________________________

## Installation Size

- Source: `/home/christoph/make_my_gpu_useful/TheRock_gfx1031/build/dist/rocm` (2.4GB)
- Target: `/opt/rocm` (2.4GB)
- Backup: `/opt/rocm.backup.TIMESTAMP` (size varies)

**Total disk space needed**: ~5GB (including backup)

______________________________________________________________________

## Ready to Install?

Run this command:

```bash
sudo bash /home/christoph/make_my_gpu_useful/TheRock_gfx1031/install_systemwide.sh
```

The script is safe and includes:

- Automatic backup of existing installation
- Verification steps
- Clear progress indicators
- Rollback instructions if needed

**Installation time**: 2-3 minutes
**Disk space**: 2.4GB + backup
**Downtime**: None (old backup preserved)

______________________________________________________________________

**Questions?** Check the script: `/home/christoph/make_my_gpu_useful/TheRock_gfx1031/install_systemwide.sh`
