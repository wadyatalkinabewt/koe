"""
Koe - Cloud speech-to-text with snippet and meeting modes.

Entry point that checks for first-time setup and runs the appropriate mode.
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import dotenv_values, load_dotenv


# Koe is a small source-run desktop app; keep its working tree free of generated
# bytecode directories during normal launches (including child Scribe windows).
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


ELEVENLABS_KEY_NAMES = ("ELEVENLABS_API_KEY", "ELEVEN_API_KEY", "XI_API_KEY")


def needs_setup(koe_dir: Path | None = None) -> bool:
    """Check if setup needs to run."""
    koe_dir = koe_dir or Path(__file__).parent

    # The marker is advisory only. Always revalidate the actual key so an old
    # installation cannot bypass setup after the supported backend changes.
    marker_path = koe_dir / ".setup_complete"
    env_path = koe_dir / ".env"
    config_path = koe_dir / "src" / "config.yaml"

    if env_path.exists() and config_path.exists():
        values = dotenv_values(env_path)
        if any(str(values.get(name) or "").strip() for name in ELEVENLABS_KEY_NAMES):
            marker_path.touch()
            return False

    marker_path.unlink(missing_ok=True)

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
    subprocess.run([sys.executable, '-B', os.path.join('src', 'main.py')])


if __name__ == '__main__':
    # Check command line args for forcing setup
    if '--setup' in sys.argv:
        run_setup()
    elif needs_setup():
        run_setup()
    else:
        run_koe()
