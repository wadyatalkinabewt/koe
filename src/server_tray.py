"""
Transcription Server (Headless)

Runs the transcription server in the background without a tray icon.
Supports Whisper and Parakeet engines based on WHISPER_ENGINE env var.
Controlled via the /shutdown endpoint or by killing the process.
"""

import os
import sys
from pathlib import Path

# Load .env file for HF_TOKEN and other settings
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Setup CUDA DLLs before any CUDA imports (PATH modification more reliable than os.add_dll_directory)
def _setup_cuda_dlls():
    try:
        import site
        paths_to_add = []
        for sp in [site.getusersitepackages()] + site.getsitepackages():
            cudnn_bin = os.path.join(sp, "nvidia", "cudnn", "bin")
            cublas_bin = os.path.join(sp, "nvidia", "cublas", "bin")
            if os.path.exists(cudnn_bin):
                paths_to_add.append(cudnn_bin)
            if os.path.exists(cublas_bin):
                paths_to_add.append(cublas_bin)
        if paths_to_add:
            os.environ['PATH'] = os.pathsep.join(paths_to_add) + os.pathsep + os.environ.get('PATH', '')
    except Exception:
        pass

_setup_cuda_dlls()

# Load model BEFORE uvicorn to avoid issues
from server import load_model, app
import uvicorn
import yaml

def get_server_port():
    """Get the configured server port."""
    port = os.environ.get("KOE_SERVER_PORT")
    if port:
        return int(port)
        
    try:
        config_path = Path(__file__).parent / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
            misc = config.get("misc", {})
            config_port = misc.get("server_port")
            if config_port:
                return int(config_port)
    except:
        pass
        
    return 9876  # Default fallback


def main():
    """Run the headless server."""
    # Load model
    engine = os.environ.get("WHISPER_ENGINE", "whisper")
    model = os.environ.get("WHISPER_MODEL", "large-v3")
    device = os.environ.get("WHISPER_DEVICE", "cuda")
    compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")

    print(f"[Server] Loading {engine} engine with model {model} on {device}...")
    load_model(model, device, compute_type)
    print("[Server] Model loaded!")

    # Start server
    port = get_server_port()
    print(f"[Server] Starting on http://0.0.0.0:{port}")
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="warning"
    )
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    main()
