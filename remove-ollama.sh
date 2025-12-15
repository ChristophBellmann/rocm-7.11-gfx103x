#!/usr/bin/env bash
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "This script assumes you will run it with sudo."
  echo "Usage: sudo ./remove-ollama.sh"
  exit 1
fi

echo "Stopping and disabling the Ollama service..."
systemctl stop ollama.service 2>/dev/null || true
systemctl disable ollama.service 2>/dev/null || true

echo "Removing Ollama service unit and related files..."
rm -f /etc/systemd/system/ollama.service
rm -f /etc/systemd/system/multi-user.target.wants/ollama.service
systemctl daemon-reload
systemctl reset-failed

echo "Removing Ollama binaries, libraries, and blobs..."
rm -f /usr/local/bin/ollama
rm -rf /usr/local/lib/ollama
rm -rf /var/lib/ollama

echo "Deleting the dedicated ollama user (if present)..."
userdel -r ollama 2>/dev/null || true

echo "Cleaning up downloaded Ollama assets in /media/christoph/some_space/llama-test..."
rm -rf /media/christoph/some_space/llama-test

echo "Done. Ollama has been removed from the system."
