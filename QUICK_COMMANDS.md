# 🚀 Quick Commands Reference Guide

## Test Results Summary (2025-11-19)

All systems tested and verified working:

### ✓ ROCm Build & GPU Detection

- **Status**: Working perfectly
- **GPU Detected**: AMD Radeon RX 6700 XT (gfx1030/gfx1031)
- **ROCm Version**: HIP 7.2.25452
- **Compiler**: AMD clang 22.0.0git

### ✓ llama.cpp Server (Port 8080)

- **Status**: Running and healthy
- **Health Check**: `{"status":"ok"}`
- **Model**: DeepSeek-R1 (Qwen 0.5B)
- **Process**: `/home/christoph/llama.cpp/build/bin/llama-server`

### ✓ LM Studio

- **Status**: Server running
- **Port**: 1234
- **Models**: Available but not currently loaded
- **GPU Support**: Ready

### ⚠ Ollama

- **Status**: Conflict with system service on port 11434
- **Solution**: Use `~/ollama-gpu` directly or stop system ollama first
- **GPU Build**: Available at `~/ollama-gpu`

______________________________________________________________________

## 🎯 ROCm & GPU Commands

| Command      | Description                            |
| ------------ | -------------------------------------- |
| `gpu`        | Quick GPU stats (rocm-smi)             |
| `gpu-temp`   | Just GPU temperature                   |
| `gpu-watch`  | Live GPU monitoring (updates every 1s) |
| `gpu-all`    | Complete GPU info (ROCm, HIP, stats)   |
| `rocm-info`  | Quick ROCm system info                 |
| `hip-test`   | Test HIP setup and compiler            |
| `test-rocm`  | Test ROCm agent detection              |
| `test-hip`   | Compile and run simple HIP test        |
| `test-gpu`   | Run both ROCm and HIP tests            |
| `debug-gpu`  | Show detailed GPU debug info           |
| `show-gpu`   | Display formatted GPU info             |
| `show-temps` | Show CPU and GPU temperatures          |

**Examples:**

```bash
# Check GPU status
gpu

# Monitor GPU live
gpu-watch

# Complete GPU diagnostics
gpu-all

# Test HIP compilation
test-hip
```

______________________________________________________________________

## 🤖 llama-server Commands (Port 8080)

| Command         | Description                      |
| --------------- | -------------------------------- |
| `llama-start`   | Start llama-server in background |
| `llama-stop`    | Stop llama-server                |
| `llama-restart` | Restart llama-server             |
| `llama-status`  | Check if server is running       |
| `llama-health`  | Quick health check (JSON)        |
| `llama-logs`    | Follow server logs               |
| `llama-test`    | Test API with model query        |
| `test-llama`    | Test API with chat completion    |

**Examples:**

```bash
# Check server status
llama-status

# Quick health check
llama-health
# Output: {"status":"ok"}

# View live logs
llama-logs

# Test chat API
test-llama
```

______________________________________________________________________

## 📦 Ollama GPU Commands

| Command          | Description                  |
| ---------------- | ---------------------------- |
| `ollama`         | Use GPU-enabled Ollama build |
| `ollama-start`   | Start Ollama systemd service |
| `ollama-stop`    | Stop Ollama service          |
| `ollama-restart` | Restart Ollama service       |
| `ollama-status`  | Check service status         |
| `ollama-logs`    | Follow Ollama logs           |

**Examples:**

```bash
# Run a model (using GPU build)
ollama run llama3.2 "Hello"

# List running models
ollama ps

# Check service status
ollama-status
```

**Note**: If you get "port 11434 already in use", stop the system ollama:

```bash
sudo systemctl stop ollama  # System service
ollama-stop                  # User service
```

______________________________________________________________________

## 🎨 LM Studio Commands

| Command          | Description                   |
| ---------------- | ----------------------------- |
| `lms-status`     | Check LM Studio server status |
| `lms-models`     | Show loaded models            |
| `lms-restart`    | Restart LM Studio with GPU    |
| `lmload <model>` | Load a specific model         |
| `lmps`           | Show loaded models            |
| `lmls`           | List available models         |
| `lmunload`       | Unload a model                |

**Quick model loading:**

```bash
lm-gpt        # Load GPT-OSS 20B
lm-claude     # Load Claude 3.7
lm-deepseek   # Load DeepSeek R1
lm-liquid     # Load Liquid (fast/small)
```

**Examples:**

```bash
# Check what's running
lms-status
# Output: Server: ON (port: 1234)

# See loaded models
lms-models

# Load a model
lmload openai/gpt-oss-20b --yes
```

______________________________________________________________________

## 🏗️ TheRock Build Commands

| Command        | Description                   |
| -------------- | ----------------------------- |
| `rock`         | Navigate to TheRock directory |
| `rock-build`   | Build TheRock                 |
| `rock-rebuild` | Clean build from scratch      |
| `rock-test`    | Run tests                     |
| `rock-clean`   | Remove build directory        |
| `venv-rock`    | Activate TheRock Python venv  |
| `ccache-stats` | Show ccache statistics        |
| `ccache-clear` | Clear ccache                  |

**Examples:**

```bash
# Quick navigation and build
rock
rock-build

# View cache stats
ccache-stats

# Clean rebuild
rock-rebuild
```

______________________________________________________________________

## 🔍 System Monitor Shortcuts

| Command       | Description                                  |
| ------------- | -------------------------------------------- |
| `check-all`   | Check all systems (ROCm, llama, LMS, Ollama) |
| `sys-all`     | System info + GPU stats                      |
| `show-models` | Show loaded LLM models                       |
| `show-temps`  | CPU and GPU temperatures                     |
| `temps`       | Live temperature monitoring                  |
| `gpustat`     | Live GPU statistics                          |
| `sysinfo`     | Detailed system information                  |
| `logs-all`    | View all service logs                        |

**Examples:**

```bash
# Check everything
check-all

# Monitor temperatures
temps

# See what models are loaded
show-models
```

______________________________________________________________________

## 🔧 Environment Switching

| Command           | Description                         |
| ----------------- | ----------------------------------- |
| `use-rocm-native` | Use native gfx1031 (RX 6700 XT)     |
| `use-rocm-compat` | Use gfx1030 compatibility mode      |
| `show-rocm-env`   | Show all ROCm environment variables |

**Examples:**

```bash
# Switch to native mode (faster)
use-rocm-native

# Switch to compatibility mode (more stable)
use-rocm-compat

# Check current settings
show-rocm-env
```

______________________________________________________________________

## 🎓 Open Interpreter

| Command            | Description                            |
| ------------------ | -------------------------------------- |
| `oi`               | Start Open Interpreter (auto-run mode) |
| `open-interpreter` | Start Open Interpreter (ask mode)      |

**Configuration:**

- Uses llama-server on port 8080
- Model: deepseek-r1
- Python warnings suppressed
- Virtual environment auto-activated

**Examples:**

```bash
# Quick start (auto-run code)
oi

# Interactive mode (ask before running)
open-interpreter
```

______________________________________________________________________

## 📊 Complete System Check

Run this to verify everything is working:

```bash
# 1. Check GPU
gpu

# 2. Check all services
check-all

# 3. Test llama-server
llama-health

# 4. Check LM Studio
lms-status

# 5. View loaded models
show-models

# 6. Monitor GPU
gpu-watch
```

______________________________________________________________________

## 🐛 Troubleshooting

### GPU Not Detected

```bash
debug-gpu
test-rocm
```

### llama-server Issues

```bash
llama-status
llama-logs
llama-restart
```

### Ollama Port Conflict

```bash
# Stop conflicting service
sudo systemctl stop ollama
ollama-stop

# Or use alternate port
OLLAMA_HOST=127.0.0.1:11435 ~/ollama-gpu serve
```

### Check All Logs

```bash
logs-all
```

______________________________________________________________________

## 💡 Quick Tips

1. **Banner Refresh**: Type `banner` to see the system banner again

1. **Daily Recipe**: Type `recipe` to see today's cooking recipe

1. **Tab Completion**: Most commands support tab completion in zsh/bash

1. **Live Monitoring**: Commands with `watch` in them update every 1 second (Ctrl+C to exit)

1. **Background Services**:

   - llama-server: Port 8080
   - LM Studio: Port 1234
   - Ollama: Port 11434

1. **GPU Override**: Current setting is `HSA_OVERRIDE_GFX_VERSION=10.3.0` for compatibility

   - Use `use-rocm-native` for native gfx1031 support
   - Use `use-rocm-compat` to restore gfx1030 compatibility

______________________________________________________________________

## 📝 Notes

- All aliases are available in both bash and zsh
- Commands are loaded from `~/.bashrc` and `~/.zshrc`
- ROCm environment is pre-configured at shell startup
- GPU acceleration is enabled for all LLM tools
- TheRock build is installed to `/opt/rocm`

______________________________________________________________________

## 🚀 Most Used Commands

```bash
# Quick status check
gpu && llama-status && lms-status

# Full system check
check-all

# Monitor everything
gpu-watch

# Test LLM
test-llama

# Debug issues
debug-gpu && logs-all
```

______________________________________________________________________

*Last Updated: 2025-11-19*
*System: Fedora 43 with AMD RX 6700 XT (gfx1031)*
*ROCm: TheRock Custom Build (HIP 7.2.25452)*
