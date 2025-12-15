#!/bin/bash
set -e

echo "=========================================="
echo "Rebuilding Ollama with TheRock ROCm 7.x"
echo "=========================================="
echo ""

# Check if TheRock ROCm is installed
if [ ! -d "/opt/rocm" ]; then
    echo "ERROR: /opt/rocm not found. Run install_to_opt_rocm.sh first."
    exit 1
fi

# Load ROCm environment
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH
unset HSA_OVERRIDE_GFX_VERSION

echo "Using ROCm from: $ROCM_PATH"
echo "HIP version:"
hipcc --version | head -3
echo ""

# Check dependencies
echo "Checking build dependencies..."
for cmd in go git cmake make; do
    if ! command -v $cmd &> /dev/null; then
        echo "ERROR: $cmd not found. Please install it first."
        exit 1
    fi
done
echo "✓ All dependencies found"
echo ""

# Create build directory
BUILD_DIR="/home/christoph/ollama-rocm-build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Clone Ollama if not exists
if [ ! -d "ollama" ]; then
    echo "Cloning Ollama repository..."
    git clone https://github.com/ollama/ollama.git
    echo "✓ Repository cloned"
else
    echo "Updating Ollama repository..."
    cd ollama
    git pull
    cd ..
    echo "✓ Repository updated"
fi

cd ollama

# Stop existing Ollama service
echo ""
echo "Stopping Ollama service..."
systemctl --user stop ollama || true
pkill -9 ollama || true
sleep 2
echo "✓ Ollama stopped"

# Build Ollama with ROCm support
echo ""
echo "Building Ollama with ROCm 7.x support..."
echo "This will take several minutes..."
echo ""

# Set build environment
export CGO_ENABLED=1
export GOFLAGS="-buildvcs=false"

# Build with ROCm
go generate ./...
go build -o ollama .

echo ""
echo "✓ Ollama built successfully"

# Backup old ollama binary
if [ -f "/usr/local/bin/ollama" ]; then
    echo ""
    echo "Backing up old Ollama binary..."
    sudo mv /usr/local/bin/ollama /usr/local/bin/ollama.backup-$(date +%Y%m%d-%H%M%S)
    echo "✓ Backup created"
fi

# Install new binary
echo ""
echo "Installing new Ollama binary..."
sudo cp ollama /usr/local/bin/ollama
sudo chmod +x /usr/local/bin/ollama
echo "✓ Binary installed to /usr/local/bin/ollama"

# Verify version
echo ""
echo "New Ollama version:"
/usr/local/bin/ollama --version

echo ""
echo "=========================================="
echo "Ollama Rebuilt Successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Start Ollama: systemctl --user start ollama"
echo "2. Check GPU detection in logs: journalctl --user -u ollama -f"
echo "3. Test with a model: ollama run llama3.2"
echo ""
