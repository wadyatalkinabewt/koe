"""
Koe - Local speech-to-text with speaker identification.

Entry point that checks for first-time setup and runs the appropriate mode.
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv


def needs_setup() -> bool:
    """Check if setup needs to run."""
    koe_dir = Path(__file__).parent

    # Check for setup complete marker
    if (koe_dir / ".setup_complete").exists():
        return False

    # Check for existing config with WHISPER_MODEL (new setup) or HF_TOKEN (old setup)
    env_path = koe_dir / ".env"
    config_path = koe_dir / "src" / "config.yaml"

    if env_path.exists() and config_path.exists():
        with open(env_path) as f:
            content = f.read()
            # New setup saves WHISPER_MODEL, old setup required HF_TOKEN
            if "WHISPER_MODEL=" in content or "HF_TOKEN=hf_" in content:
                # Has valid config, mark as complete
                (koe_dir / ".setup_complete").touch()
                return False

    return True


def run_setup():
    """Run the terminal setup."""
    print("First-time setup required...")
    from src.setup_cli import run_setup as cli_setup
    cli_setup()


def run_koe():
    """Run the main Koe application."""
    print('Starting Koe...')
    load_dotenv()
    subprocess.run([sys.executable, os.path.join('src', 'main.py')])


if __name__ == '__main__':
    # Check command line args for forcing setup
    if '--setup' in sys.argv:
        run_setup()
    elif needs_setup():
        run_setup()
    else:
        run_koe()
