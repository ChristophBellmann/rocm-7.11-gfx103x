#!/usr/bin/env bash
#
# install-ollama-rocm.sh
#
# Installs Ollama on Linux (x86_64) + ROCm libs and sets up a system-wide systemd service.
#
# Notes:
# - The ROCm tarball may contain ONLY libs (no binary). Therefore we install BOTH:
#   1) https://ollama.com/download/ollama-linux-amd64.tgz
#   2) https://ollama.com/download/ollama-linux-amd64-rocm.tgz
#
# Logs:
#   sudo journalctl -u ollama -f
#

set -euo pipefail

echo "=========================================="
echo "Preparing Ollama + ROCm (manual sudo steps follow)"
echo "=========================================="
echo ""

arch="$(uname -m)"
[[ "$arch" == "x86_64" ]] || { echo "ERROR: amd64 only (found $arch)"; exit 1; }

echo "Stopping any running Ollama..."
systemctl disable --now ollama 2>/dev/null || true
pkill ollama 2>/dev/null || true
pkill -9 ollama 2>/dev/null || true
sleep 1
echo "✓ stopped"
echo ""

# Backup existing binary if present
if [ -f /usr/local/bin/ollama ]; then
  mv /usr/local/bin/ollama \
    /usr/local/bin/ollama.backup-$(date +%Y%m%d-%H%M%S)
  echo "✓ old binary backed up"
fi

download_dir="/media/christoph/some_space/llama-test"
mkdir -p "$download_dir"
cd "$download_dir"

echo "Downloading Ollama base (binary)..."
curl -fsSL https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-base.tgz

echo "Downloading Ollama ROCm libs..."
curl -fsSL https://ollama.com/download/ollama-linux-amd64-rocm.tgz -o ollama-rocm.tgz

echo "Extracting..."
mkdir -p base rocm
tar -xzf ollama-base.tgz -C base
tar -xzf ollama-rocm.tgz -C rocm

# Find binary in base tarball
ollama_path="$(find base -type f -name ollama -perm -u+x 2>/dev/null | head -n 1)"
if [[ -z "${ollama_path}" ]]; then
  echo "ERROR: Could not find an executable 'ollama' in base archive."
  echo "Base archive listing (first 200):"
  tar -tzf ollama-base.tgz | head -n 200
  exit 1
fi

echo "Found ollama binary at: ${ollama_path}"
libdir="$(find rocm -type d -path "*/lib/ollama" 2>/dev/null | head -n 1 || true)"
if [[ -n "${libdir}" ]]; then
  echo "Found ROCm libs at: ${libdir}"
else
  echo "ROCm lib directory not found in the archive."
fi

echo ""
echo "Use install-ollama-rocm-sudo.sh with these values:"
echo "OLLAMA_PATH=${ollama_path}"
echo "LIBDIR=${libdir}"
echo ""
echo "Example:"
echo "sudo bash install-ollama-rocm-sudo.sh \"${ollama_path}\" \"${libdir}\""
