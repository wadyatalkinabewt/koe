#!/bin/bash
# Start Koe server with Parakeet engine in WSL
# Run this from Windows: wsl -d Ubuntu-22.04 -- /opt/koe/start_server.sh

cd /opt/koe
source venv/bin/activate

export WHISPER_ENGINE=parakeet
export WHISPER_MODEL=nvidia/parakeet-ctc-0.6b
export WHISPER_DEVICE=cuda
# HF_TOKEN should be set via .env file or environment
export HF_TOKEN="${HF_TOKEN:-$(grep HF_TOKEN /mnt/c/dev/koe/.env 2>/dev/null | cut -d= -f2)}"

echo "Starting Koe server with Parakeet engine..."
echo "Engine: $WHISPER_ENGINE"
echo "Model: $WHISPER_MODEL"
echo "Device: $WHISPER_DEVICE"
echo ""
echo "Server will be available at http://localhost:9876"
echo "Press Ctrl+C to stop"
echo ""

python -m src.server --host 0.0.0.0 --port 9876
