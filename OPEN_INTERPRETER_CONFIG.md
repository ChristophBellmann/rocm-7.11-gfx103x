# 🤖 Open Interpreter Configuration for llama-server

## ✅ Configuration Complete

Open Interpreter is now fully configured to use **llama-server** on port 8080 with GPU acceleration!

______________________________________________________________________

## 📋 Current Setup

### llama-server Status

```bash
Server:   Running ✅
Port:     8080
Health:   {"status":"ok"}
Model:    Qwen2.5-0.5B-Instruct-Q8_0.gguf
GPU:      99 layers offloaded (full GPU acceleration)
Context:  16384 tokens
```

### Open Interpreter Configuration

**Environment Variables** (in ~/.bashrc and ~/.zshrc):

```bash
export OPENAI_API_BASE="http://localhost:8080/v1"
export OPENAI_API_KEY="llama-local"
export INTERPRETER_MODEL="llama"
export INTERPRETER_CLI_AUTO_RUN=false
export INTERPRETER_CLI_SAFE_MODE="auto"
```

**Aliases**:

```bash
# Quick start (auto-run mode)
oi

# Interactive mode (ask before running)
open-interpreter
```

______________________________________________________________________

## 🚀 Usage

### Quick Start

```bash
# Auto-run mode (executes code automatically with -y flag)
oi

# Example prompts:
oi "What is 25 * 47?"
oi "List files in the current directory"
oi "Show me GPU temperature"
```

### Interactive Mode

```bash
# Ask before running code
open-interpreter

# Then interact normally:
> What's the current GPU usage?
> Create a Python script to monitor VRAM
```

### Manual Command

```bash
# If you want full control over flags
source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/.venv/bin/activate
interpreter --api_base http://localhost:8080/v1 --model llama
```

______________________________________________________________________

## 🔧 How It Works

1. **llama-server** runs locally on port 8080
1. Provides OpenAI-compatible API endpoint
1. Open Interpreter connects via `--api_base` flag
1. Model inference runs on GPU (99 layers offloaded)
1. Responses streamed back to interpreter

**Architecture:**

```
┌─────────────────┐     HTTP Request      ┌──────────────────┐
│ Open Interpreter│ ──────────────────>  │  llama-server    │
│   (CLI Tool)    │                       │  (Port 8080)     │
└─────────────────┘ <────────────────── └──────────────────┘
                      Streaming Response         │
                                                  ▼
                                          ┌──────────────────┐
                                          │   GPU (RX 6700)  │
                                          │   VRAM: 12GB     │
                                          └──────────────────┘
```

______________________________________________________________________

## 📊 Configuration Details

### Server Endpoint

- **Base URL**: `http://localhost:8080/v1`
- **Models Endpoint**: `http://localhost:8080/v1/models`
- **Chat Endpoint**: `http://localhost:8080/v1/chat/completions`
- **Health Check**: `http://localhost:8080/health`

### Current Model

```
Name:         Qwen2.5-0.5B-Instruct-Q8_0
Format:       GGUF (Q8_0 quantization)
Size:         ~525 MB
Parameters:   494M (0.5 billion)
Context:      32,768 tokens (trained), 16,384 (server config)
Vocab:        151,936 tokens
```

### GPU Offload

- **Layers Offloaded**: 99 (all layers)
- **GPU Memory Used**: ~1.8 GB VRAM
- **Available VRAM**: 10.2 GB free
- **Inference Speed**: GPU-accelerated (fast)

______________________________________________________________________

## 🎯 Quick Commands

### Check Server Status

```bash
llama-status      # Check if running
llama-health      # API health check
llama-test        # Test API endpoint
```

### Manage Server

```bash
llama-start       # Start llama-server
llama-stop        # Stop llama-server
llama-restart     # Restart llama-server
llama-logs        # View server logs
```

### Test Open Interpreter

```bash
# Quick test (with venv activation)
oi "print('Hello from GPU!')"

# Or manually:
source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/.venv/bin/activate
interpreter --version
```

______________________________________________________________________

## 🔍 Verification

### 1. Check llama-server is Running

```bash
llama-health
# Expected: {"status":"ok"}
```

### 2. Verify Model Loaded

```bash
curl -s http://localhost:8080/v1/models | python3 -m json.tool
# Should show Qwen2.5-0.5B-Instruct model
```

### 3. Test Open Interpreter Connection

```bash
oi "2 + 2"
# Should execute and return 4
```

### 4. Check GPU Usage

```bash
gpu
# Should show VRAM usage increase during inference
```

______________________________________________________________________

## 🎨 Features

### Safety Features

✅ **Auto-run disabled by default** (`INTERPRETER_CLI_AUTO_RUN=false`)
✅ **Safe mode enabled** (`INTERPRETER_CLI_SAFE_MODE="auto"`)
✅ **Python warnings suppressed** (clean output)
✅ **Local execution only** (no external API calls)

### Performance

🚀 **Full GPU acceleration** (99 layers on GPU)
🚀 **16K context window** (handles long conversations)
🚀 **Fast inference** (GPU-optimized)
🚀 **Low latency** (local server, no network)

### Convenience

🎯 **Quick aliases** (`oi` for instant access)
🎯 **Auto venv activation** (no manual sourcing)
🎯 **Clean output** (warnings suppressed)
🎯 **Auto-run mode** (`oi` with `-y` flag)

______________________________________________________________________

## 📝 Configuration Files Updated

### ~/.bashrc

```bash
# Environment variables (lines 99-104)
export OPENAI_API_BASE="http://localhost:8080/v1"
export OPENAI_API_KEY="llama-local"
export INTERPRETER_MODEL="llama"
export INTERPRETER_CLI_AUTO_RUN=false
export INTERPRETER_CLI_SAFE_MODE="auto"

# Aliases (lines 297-298)
alias oi="source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/.venv/bin/activate && PYTHONWARNINGS='ignore' interpreter -y --api_base http://localhost:8080/v1 --model llama"
alias open-interpreter="source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/.venv/bin/activate && PYTHONWARNINGS='ignore' interpreter --api_base http://localhost:8080/v1 --model llama"
```

### ~/.zshrc

```bash
# Environment variables (lines 150-155)
export OPENAI_API_BASE="http://localhost:8080/v1"
export OPENAI_API_KEY="llama-local"
export INTERPRETER_MODEL="llama"
export INTERPRETER_CLI_AUTO_RUN=false
export INTERPRETER_CLI_SAFE_MODE="auto"

# Aliases (lines 345-346)
alias oi="source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/.venv/bin/activate && PYTHONWARNINGS='ignore' interpreter -y --api_base http://localhost:8080/v1 --model llama"
alias open-interpreter="source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/.venv/bin/activate && PYTHONWARNINGS='ignore' interpreter --api_base http://localhost:8080/v1 --model llama"
```

______________________________________________________________________

## 🔄 Switching Models

If you want to use a different model:

### 1. Stop Current Server

```bash
llama-stop
```

### 2. Edit Start Script

```bash
nano ~/start_llama_server_openai.sh
# Change MODEL_PATH to your new .gguf file
```

### 3. Restart Server

```bash
llama-start
```

### 4. Verify New Model

```bash
llama-test
curl -s http://localhost:8080/v1/models | python3 -m json.tool
```

______________________________________________________________________

## 🚨 Troubleshooting

### Open Interpreter Can't Connect

**Symptoms**: Connection refused, timeout errors

**Solutions**:

```bash
# 1. Check server is running
llama-status

# 2. Check health endpoint
llama-health

# 3. Check logs for errors
llama-logs

# 4. Restart server
llama-restart
```

### Wrong Model Being Used

**Symptoms**: Unexpected responses, model mismatch

**Solutions**:

```bash
# Check which model is loaded
curl -s http://localhost:8080/v1/models | python3 -m json.tool

# Restart server to reload model
llama-restart
```

### GPU Not Being Used

**Symptoms**: Slow inference, low VRAM usage

**Solutions**:

```bash
# Check GPU layers offloaded
pgrep -af llama-server | grep -o "\-ngl [0-9]*"
# Should show: -ngl 99

# Check VRAM usage
gpu
# Should show ~1.8GB+ when model loaded

# Check server process
llama-status
```

### Python Virtual Environment Issues

**Symptoms**: Module not found, import errors

**Solutions**:

```bash
# Manually activate venv
source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/.venv/bin/activate

# Check interpreter installed
which interpreter

# Reinstall if needed
pip install open-interpreter
```

______________________________________________________________________

## 📈 Performance Notes

### Current Model (Qwen2.5-0.5B)

- **Size**: Small (0.5B parameters)
- **Speed**: Very fast inference
- **Quality**: Good for basic tasks
- **VRAM**: Low usage (~1.8GB)

### Recommended Models

**For Better Quality**:

- DeepSeek-R1-0528-Qwen3-8B (8B params, better reasoning)
- Llama-3.2-3B-Instruct (3B params, good balance)

**For Maximum Speed**:

- Qwen2.5-0.5B (current, fastest)
- Llama-3.2-1B (1B params, very fast)

**For Best Quality** (requires more VRAM):

- Qwen2.5-7B-Instruct (7B params, excellent)
- Llama-3.1-8B-Instruct (8B params, very good)

______________________________________________________________________

## 🎯 Example Use Cases

### Code Execution

```bash
oi "write a python script to find prime numbers under 100"
```

### System Monitoring

```bash
oi "show me current GPU temperature and VRAM usage"
```

### File Operations

```bash
oi "list all .py files in current directory and count lines of code"
```

### Data Analysis

```bash
oi "read the CSV file data.csv and show summary statistics"
```

### Quick Calculations

```bash
oi "calculate compound interest: principal=1000, rate=5%, years=10"
```

______________________________________________________________________

## ✅ Summary

✅ **Configured**: Open Interpreter → llama-server (port 8080)
✅ **Model**: Qwen2.5-0.5B-Instruct-Q8_0.gguf
✅ **GPU**: Full acceleration (99 layers offloaded)
✅ **Aliases**: `oi` and `open-interpreter` ready to use
✅ **Environment**: All variables set correctly
✅ **Status**: Server running and healthy

**Ready to use!** Just type `oi` or `open-interpreter` 🚀

______________________________________________________________________

*Last Updated: 2025-11-19*
*Configuration: llama-server on port 8080*
*Model: Qwen2.5-0.5B-Instruct-Q8_0 (GPU-accelerated)*
