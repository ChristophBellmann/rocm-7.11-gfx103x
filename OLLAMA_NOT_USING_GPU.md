# Ollama NOT Using GPU - Complete Diagnosis

**Status**: ⚠️ Ollama running on CPU despite ROCm installation

## The Problem

Ollama is NOT using your GPU:

- GPU Usage: **0%** during inference
- Reports: **100% CPU**
- Speed: **13 tokens/s** (should be 30-60 on GPU)
- VRAM: 8.5GB used (model loaded but not computing)

## What's Working

✅ System ROCm with native **gfx1031** support
✅ GPU detected correctly (no more gfx1030 override)
✅ Ollama has ROCm libraries built-in
✅ Libraries link to `/opt/rocm` correctly

## Why GPU Isn't Used

Ollama loads model to VRAM but computes on CPU. Common causes:

1. Service environment missing ROCM_PATH
1. Ollama user lacks GPU permissions
1. ROCm runtime initialization failing silently
1. Version mismatch between bundled and system ROCm

## Quick Fixes to Try

### Fix 1: Restart with Proper Environment

```bash
sudo systemctl stop ollama 2>/dev/null
sudo pkill -9 ollama
sudo -E ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm /usr/local/bin/ollama serve &
sleep 5
ollama ps  # Check if shows GPU
```

### Fix 2: Check GPU Permissions

```bash
sudo usermod -aG video,render ollama
sudo systemctl restart ollama
```

### Fix 3: Use LM Studio Instead

```bash
lms  # Should detect GPU automatically
```

## Recommended: Rebuild Ollama

Your best bet is rebuilding Ollama with your system ROCm:

```bash
cd ~ && git clone https://github.com/ollama/ollama.git && cd ollama
export ROCM_PATH=/opt/rocm
export CMAKE_ARGS="-DGGML_HIPBLAS=on -DAMDGPU_TARGETS=gfx1031"
go generate ./... && go build .
sudo cp ollama /usr/local/bin/ollama
```

Full guide: `/home/christoph/make_my_gpu_useful/TheRock_gfx1031/OLLAMA_GPU_STATUS.md`
