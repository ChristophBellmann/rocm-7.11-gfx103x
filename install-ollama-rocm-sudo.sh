#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <OLLAMA_PATH> <LIBDIR>"
  echo "Example: $0 /tmp/base/usr/local/bin/ollama /tmp/rocm/usr/local/lib/ollama"
  exit 1
fi

OLLAMA_PATH="$1"
LIBDIR="$2"

sudo mv /usr/local/bin/ollama /usr/local/bin/ollama.backup-$(date +%Y%m%d-%H%M%S) 2>/dev/null || true
sudo install -m 755 "$OLLAMA_PATH" /usr/local/bin/ollama

if [[ -n "$LIBDIR" ]]; then
  sudo mkdir -p /usr/local/lib/ollama
  sudo cp -a "$LIBDIR/." /usr/local/lib/ollama/
fi

sudo useradd -r -s /usr/sbin/nologin -m -d /var/lib/ollama ollama 2>/dev/null || true
sudo usermod -aG render,video ollama || true
sudo mkdir -p /var/lib/ollama
sudo chown -R ollama:ollama /var/lib/ollama

sudo tee /etc/systemd/system/ollama.service >/dev/null <<'EOF'
[Unit]
Description=Ollama (ROCm)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ollama
Group=ollama
SupplementaryGroups=render video
Environment=OLLAMA_HOST=127.0.0.1:11434
Environment=OLLAMA_MODELS=/var/lib/ollama
Environment=LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/usr/local/lib/ollama:/usr/local/lib/ollama/rocm
Environment=ROCM_PATH=/opt/rocm
Environment=HIP_PATH=/opt/rocm
Environment=HIP_PLATFORM=amd
Environment=HIP_COMPILER=clang
Environment=HIP_VISIBLE_DEVICES=0
Environment=HSA_ENABLE_SDMA=0
ExecStart=/usr/local/bin/ollama serve
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now ollama

sudo /usr/local/bin/ollama --version || true
sudo journalctl -u ollama -n 80 --no-pager
