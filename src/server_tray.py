"""
Whisper Server (Headless)

Runs the transcription server in the background without a tray icon.
Controlled via the /shutdown endpoint or by killing the process.
"""

import os
import sys
from pathlib import Path

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


def main():
    """Run the headless server."""
    # Load model
    model = os.environ.get("WHISPER_MODEL", "large-v3")
    device = os.environ.get("WHISPER_DEVICE", "cuda")
    compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")

    print(f"[Server] Loading Whisper {model} on {device}...")
    load_model(model, device, compute_type)
    print("[Server] Model loaded!")

    # Start server
    print("[Server] Starting on http://0.0.0.0:9876")
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=9876,
        log_level="warning"
    )
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    main()
