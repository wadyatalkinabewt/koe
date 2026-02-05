"""
Background server launcher.

Starts the transcription server in the background if not already running.
Supports both Whisper and Parakeet engines based on config.
Designed to be called from batch files or at startup.
"""

import os
import sys
import subprocess
import time
import requests
import yaml
from pathlib import Path
from datetime import datetime

SERVER_URL = "http://localhost:9876"
SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR.parent / "config.yaml"
LOG_PATH = SCRIPT_DIR.parent / "logs" / "server_launcher.log"

def _log(msg: str):
    """Write log message to file."""
    try:
        LOG_PATH.parent.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {msg}\n")
    except:
        pass


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
    """Start the Parakeet server (Windows native with NeMo)."""
    print(f"[Launcher] Starting Parakeet server (model={model}, device={device})...")

    # Use pythonw for no console window on Windows
    python_exe = sys.executable
    pythonw_exe = python_exe.replace("python.exe", "pythonw.exe")
    if not os.path.exists(pythonw_exe):
        pythonw_exe = python_exe  # Fallback

    # Use server_tray.py which has the tray icon
    server_script = SCRIPT_DIR / "server_tray.py"

    # Set environment variables for the server process
    env = os.environ.copy()
    env["WHISPER_MODEL"] = model  # Reuse same env var
    env["WHISPER_DEVICE"] = device
    env["WHISPER_ENGINE"] = "parakeet"

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


def is_server_ready() -> bool:
    """Check if server is running AND model is loaded."""
    try:
        response = requests.get(f"{SERVER_URL}/status", timeout=2)
        if response.status_code == 200:
            data = response.json()
            return data.get("ready", False)
    except:
        pass
    return False


def is_server_busy() -> bool:
    """Check if server is actively processing requests."""
    try:
        response = requests.get(f"{SERVER_URL}/status", timeout=2)
        if response.status_code == 200:
            data = response.json()
            return data.get("busy", False)
    except:
        pass
    return False


def get_server_status() -> dict:
    """Get full server status including busy state."""
    try:
        response = requests.get(f"{SERVER_URL}/status", timeout=2)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}


def stop_server(force: bool = False, wait_for_idle: bool = True):
    """Stop the server if running and ready.

    Args:
        force: If True, stop even if busy (not recommended)
        wait_for_idle: If True, wait for active requests to complete before stopping
    """
    _log("stop_server() called")

    if not is_server_running():
        _log("Server not running, nothing to stop")
        print("[Launcher] Server not running")
        return True

    # Don't kill the server if it's still loading the model
    if not is_server_ready():
        _log("Server is loading model, not stopping")
        print("[Launcher] Server is loading model, not stopping")
        return True

    # Check if server is busy
    status = get_server_status()
    active_requests = status.get("active_requests", 0)
    _log(f"Server status: {status}")

    if active_requests > 0:
        if force:
            _log(f"Force stopping with {active_requests} active request(s)")
            print(f"[Launcher] WARNING: Force stopping with {active_requests} active request(s)")
        elif wait_for_idle:
            _log(f"Server busy ({active_requests} active), waiting...")
            print(f"[Launcher] Server busy ({active_requests} active request(s)), waiting...")
            for i in range(120):  # Wait up to 2 minutes
                time.sleep(1)
                status = get_server_status()
                active_requests = status.get("active_requests", 0)
                if active_requests == 0:
                    _log("Server idle, proceeding")
                    print("[Launcher] Server idle, proceeding with shutdown")
                    break
                if i % 10 == 9:
                    print(f"[Launcher] Still waiting... ({active_requests} active request(s))")
            else:
                _log("Timeout waiting for idle, aborting")
                print("[Launcher] Timeout waiting for idle, aborting stop")
                return False
        else:
            _log(f"Server busy, not stopping (no wait)")
            print(f"[Launcher] Server busy ({active_requests} active request(s)), not stopping")
            print("[Launcher] Use --force to stop anyway, or --wait to wait for idle")
            return False

    try:
        _log("Sending POST /shutdown...")
        print("[Launcher] Sending shutdown request...")
        response = requests.post(f"{SERVER_URL}/shutdown", timeout=5)
        _log(f"Shutdown response: {response.status_code}")
        if response.status_code == 200:
            print("[Launcher] Server shutting down...")
            # Wait for it to actually stop
            for i in range(10):
                time.sleep(0.5)
                if not is_server_running():
                    _log("Server stopped successfully")
                    print("[Launcher] Server stopped")
                    return True
                _log(f"Still running after {(i+1)*0.5}s...")
            _log("WARNING: Server may still be running after 5s wait")
            print("[Launcher] Warning: Server may still be running")
            return False
        else:
            _log(f"Unexpected response code: {response.status_code}")
            return False
    except Exception as e:
        _log(f"Error stopping server: {e}")
        print(f"[Launcher] Error stopping server: {e}")
        return False


def restart_server():
    """Restart the server (wait for idle, then stop and start)."""
    print("[Launcher] Restarting server...")
    if not stop_server(wait_for_idle=True):
        print("[Launcher] Failed to stop server, aborting restart")
        return False
    time.sleep(1)  # Brief pause before starting
    return start_server_background()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Koe Server Launcher")
    parser.add_argument("command", choices=["start", "stop", "restart", "status"], default="start", nargs="?")
    parser.add_argument("--force", action="store_true", help="Force stop even if busy (not recommended)")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for idle before stopping")
    args = parser.parse_args()

    if args.command == "start":
        start_server_background()
    elif args.command == "stop":
        stop_server(force=args.force, wait_for_idle=not args.no_wait)
    elif args.command == "restart":
        restart_server()
    elif args.command == "status":
        status = get_server_status()
        if status:
            print(f"Server: running")
            print(f"  Model: {status.get('model', 'unknown')}")
            print(f"  Device: {status.get('device', 'unknown')}")
            print(f"  Ready: {status.get('ready', False)}")
            print(f"  Busy: {status.get('busy', False)}")
            print(f"  Active requests: {status.get('active_requests', 0)}")
            print(f"  Diarization: {status.get('diarization_available', False)}")
        elif is_server_running():
            print("Server: starting (not ready yet)")
        else:
            print("Server: not running")
