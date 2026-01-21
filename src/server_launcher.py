"""
Background server launcher.

Starts the Whisper server in the background if not already running.
Designed to be called from batch files or at startup.
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

SERVER_URL = "http://localhost:9876"
SCRIPT_DIR = Path(__file__).parent


def is_server_running() -> bool:
    """Check if server is already running."""
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=1)
        return response.status_code == 200
    except:
        return False


def start_server_background():
    """Start the server in background if not already running."""
    if is_server_running():
        print("[Launcher] Server already running")
        return True

    print("[Launcher] Starting server with tray icon...")

    # Use pythonw for no console window on Windows
    python_exe = sys.executable
    pythonw_exe = python_exe.replace("python.exe", "pythonw.exe")
    if not os.path.exists(pythonw_exe):
        pythonw_exe = python_exe  # Fallback

    # Use server_tray.py which has the tray icon
    server_script = SCRIPT_DIR / "server_tray.py"

    # Start server process detached
    if sys.platform == "win32":
        # Windows: use pythonw (no console) - tray icon will be visible
        subprocess.Popen(
            [pythonw_exe, str(server_script)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(SCRIPT_DIR.parent)
        )
    else:
        # Unix: use nohup-style
        subprocess.Popen(
            [python_exe, str(server_script)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            cwd=str(SCRIPT_DIR.parent)
        )

    # Wait for server to be ready
    print("[Launcher] Waiting for server to initialize...")
    for i in range(60):  # Wait up to 60 seconds (model loading can take a while)
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
        return

    try:
        # Send shutdown request (we'd need to add this endpoint)
        # For now, just report it's running
        print("[Launcher] Server is running - close it manually or restart computer")
    except:
        pass


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
