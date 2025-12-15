# 🔧 Open Interpreter + llama-server Setup

## Current Status

⚠️ **Connection Issue**: Open Interpreter's LiteLLM proxy has compatibility issues with llama.cpp server

## ✅ Working Solution: Use LM Studio Instead

LM Studio provides better OpenAI API compatibility for Open Interpreter.

### Quick Setup:

```bash
# 1. Load a model in LM Studio
lm-deepseek   # or lm-gpt, lm-claude, etc.

# 2. Verify LM Studio is running
lms-status
# Should show: Server: ON (port: 1234)

# 3. Use Open Interpreter with LM Studio
OPENAI_API_BASE="http://localhost:1234/v1" \
OPENAI_API_KEY="lm-studio" \
interpreter --model gpt-3.5-turbo
```

### Update Aliases (Recommended):

Edit `~/.bashrc` and `~/.zshrc`:

```bash
# Change port from 8080 to 1234 (LM Studio)
alias oi="source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/.venv/bin/activate && PYTHONWARNINGS='ignore' OPENAI_API_KEY='lm-studio' OPENAI_API_BASE='http://localhost:1234/v1' interpreter -y --model gpt-3.5-turbo"
alias open-interpreter="source /home/christoph/make_my_gpu_useful/TheRock_gfx1031/.venv/bin/activate && PYTHONWARNINGS='ignore' OPENAI_API_KEY='lm-studio' OPENAI_API_BASE='http://localhost:1234/v1' interpreter --model gpt-3.5-turbo"
```

## Alternative: Direct llama.cpp CLI

If you just want to run code with an LLM, use llama-cli directly:

```bash
~/llama.cpp/build/bin/llama-cli \
  --model ~/.lmstudio/models/lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  -ngl 99 \
  -p "Write a Python script to calculate fibonacci numbers"
```

## Troubleshooting llama-server Connection

The issue is that Open Interpreter uses LiteLLM which expects certain OpenAI API behaviors that llama.cpp's server doesn't fully implement.

### Symptoms:

```
litellm.exceptions.InternalServerError: InternalServerError: OpenAIException - Connection error.
```

### Root Cause:

- LiteLLM tries to validate the model with `/v1/models`
- llama.cpp returns full file paths as model IDs
- LiteLLM doesn't recognize the format
- Connection fails before request is sent

### Solutions:

1. **Use LM Studio** (Recommended)

   - Better OpenAI API compatibility
   - Handles model names correctly
   - Works out of the box

1. **Use llama-cli directly**

   - Skip the API layer
   - Direct model inference
   - No HTTP overhead

1. **Wait for compatibility update**

   - Open Interpreter team may fix LiteLLM issues
   - llama.cpp may improve API compatibility

## Current Configuration

**llama-server** (Port 8080):

- Status: Running ✅
- Model: Qwen2.5-0.5B-Instruct-Q8_0
- GPU: Full acceleration (99 layers)
- Use for: Direct API calls, custom scripts

**LM Studio** (Port 1234):

- Status: Server ON, no model loaded
- Use for: Open Interpreter (better compatibility)
- Load model: `lm-deepseek` or `lm-gpt`

## Recommended Workflow

```bash
# 1. For Open Interpreter - Use LM Studio
lm-deepseek                    # Load a model
lms-status                     # Verify running
oi "write a python script..."  # Use with LM Studio

# 2. For Direct Inference - Use llama-server
llama-status                   # Verify running
llama-health                   # Check health
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-3.5-turbo","messages":[...]}'
```

## Files

- `~/.config/interpreter/config.yaml` - Open Interpreter config
- `~/.bashrc` - oi alias
- `~/.zshrc` - oi alias
- `~/start_llama_server_openai.sh` - llama-server startup script

______________________________________________________________________

*Last Updated: 2025-11-19*
*Issue: LiteLLM compatibility with llama.cpp*
*Solution: Use LM Studio for Open Interpreter*
