"""
Background server launcher.

Starts the transcription server in the background if not already running.
Supports both Whisper (Windows) and Parakeet (WSL) engines based on config.
Designed to be called from batch files or at startup.
"""

import os
import sys
import subprocess
import time
import requests
import yaml
from pathlib import Path

SERVER_URL = "http://localhost:9876"
SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR.parent / "config.yaml"


def get_engine_config():
    """Get engine configuration from config.yaml.

    Returns:
        tuple: (engine, model, device) from config, with defaults.
    """
    engine = "whisper"
    model = "large-v3"
    device = "auto"

    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                config = yaml.safe_load(f) or {}
            model_options = config.get("model_options", {})
            engine = model_options.get("engine", "whisper")

            # Get engine-specific model and device
            if engine == "whisper":
                whisper_opts = model_options.get("whisper", {})
                model = whisper_opts.get("model", "large-v3")
                device = whisper_opts.get("device", "auto")
            elif engine == "parakeet":
                parakeet_opts = model_options.get("parakeet", {})
                model = parakeet_opts.get("model", "nvidia/parakeet-ctc-0.6b")
                device = parakeet_opts.get("device", "auto")
    except Exception as e:
        print(f"[Launcher] Warning: Could not read config: {e}")

    return engine, model, device


def is_server_running() -> bool:
    """Check if server is already running."""
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=1)
        return response.status_code == 200
    except:
        return False


def start_whisper_server(model: str = "large-v3", device: str = "auto"):
    """Start the Whisper server (Windows native)."""
    print(f"[Launcher] Starting Whisper server (model={model}, device={device})...")

    # Use pythonw for no console window on Windows
    python_exe = sys.executable
    pythonw_exe = python_exe.replace("python.exe", "pythonw.exe")
    if not os.path.exists(pythonw_exe):
        pythonw_exe = python_exe  # Fallback

    # Use server_tray.py which has the tray icon
    server_script = SCRIPT_DIR / "server_tray.py"

    # Set environment variables for the server process
    env = os.environ.copy()
    env["WHISPER_MODEL"] = model
    env["WHISPER_DEVICE"] = device

    # Start server process detached
    if sys.platform == "win32":
        subprocess.Popen(
            [pythonw_exe, str(server_script)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(SCRIPT_DIR.parent),
            env=env
        )
    else:
        subprocess.Popen(
            [python_exe, str(server_script)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            cwd=str(SCRIPT_DIR.parent),
            env=env
        )
    return True


def start_parakeet_server(model: str = "nvidia/parakeet-ctc-0.6b", device: str = "auto"):
    """Start the Parakeet server (WSL with systemd)."""
    print(f"[Launcher] Starting Parakeet server in WSL (model={model}, device={device})...")

    try:
        # Start the systemd service in WSL
        result = subprocess.run(
            ["wsl", "-d", "Ubuntu-22.04", "--", "bash", "-c",
             "systemctl start koe-server 2>&1"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"[Launcher] Warning: systemctl start returned: {result.stderr}")
            # Try starting manually as fallback
            print("[Launcher] Trying manual start...")
            subprocess.Popen(
                ["wsl", "-d", "Ubuntu-22.04", "--", "bash", "-c",
                 f"cd /opt/koe && source venv/bin/activate && "
                 f"HF_TOKEN=$(grep HF_TOKEN /mnt/c/dev/koe/.env | cut -d= -f2) "
                 f"python src/server.py --host 0.0.0.0 --port 9876 --engine parakeet "
                 f"--model {model} --device {device} &"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        return True
    except subprocess.TimeoutExpired:
        print("[Launcher] Warning: WSL command timed out")
        return False
    except FileNotFoundError:
        print("[Launcher] Error: WSL not found. Parakeet requires WSL with Ubuntu-22.04.")
        return False


def start_server_background():
    """Start the server in background if not already running."""
    if is_server_running():
        print("[Launcher] Server already running")
        return True

    # Get engine, model, and device from config
    engine, model, device = get_engine_config()
    print(f"[Launcher] Engine from config: {engine} (model={model}, device={device})")

    if engine == "parakeet":
        start_parakeet_server(model, device)
    else:
        start_whisper_server(model, device)

    # Wait for server to be ready
    print("[Launcher] Waiting for server to initialize...")
    for i in range(90):  # Wait up to 90 seconds (Parakeet loading can take longer)
        time.sleep(1)
        if is_server_running():
            print("[Launcher] Server ready!")
            return True
        if i % 5 == 4:
            print(f"[Launcher] Still loading model... ({i+1}s)")

    print("[Launcher] Warning: Server may not have started properly")
    return False


def stop_server():
    """Stop the server if running."""
    if not is_server_running():
        print("[Launcher] Server not running")
        return True

    try:
        print("[Launcher] Sending shutdown request...")
        response = requests.post(f"{SERVER_URL}/shutdown", timeout=5)
        if response.status_code == 200:
            print("[Launcher] Server shutting down...")
            # Wait for it to actually stop
            for _ in range(10):
                time.sleep(0.5)
                if not is_server_running():
                    print("[Launcher] Server stopped")
                    return True
            print("[Launcher] Warning: Server may still be running")
            return False
    except Exception as e:
        print(f"[Launcher] Error stopping server: {e}")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["start", "stop", "status"], default="start", nargs="?")
    args = parser.parse_args()

    if args.command == "start":
        start_server_background()
    elif args.command == "stop":
        stop_server()
    elif args.command == "status":
        if is_server_running():
            print("Server is running")
        else:
            print("Server is not running")
