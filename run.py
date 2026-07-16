"""
Koe - Cloud speech-to-text with snippet and meeting modes.

Entry point that checks for first-time setup and runs the appropriate mode.
"""

import os
import sys
from pathlib import Path
from dotenv import dotenv_values, load_dotenv


# Koe is a small source-run desktop app; keep its working tree free of generated
# bytecode directories during normal launches (including child Scribe windows).
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paths import (  # noqa: E402
    app_data_dir,
    config_path,
    ensure_runtime_dirs,
    env_path,
    setup_marker_path,
)


ELEVENLABS_KEY_NAMES = ("ELEVENLABS_API_KEY", "ELEVEN_API_KEY", "XI_API_KEY")


def needs_setup(data_dir: Path | None = None) -> bool:
    """Check if setup needs to run."""
    marker = (Path(data_dir) / ".setup_complete") if data_dir else setup_marker_path()
    environment = (Path(data_dir) / ".env") if data_dir else env_path()
    config = (Path(data_dir) / "config.yaml") if data_dir else config_path()

    if environment.exists() and config.exists():
        values = dotenv_values(environment)
        if any(str(values.get(name) or "").strip() for name in ELEVENLABS_KEY_NAMES):
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
            return False

    marker.unlink(missing_ok=True)

    return True


def run_setup():
    """Run first-time GUI onboarding."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication
    from ui.setup_window import run_setup_dialog

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication.instance() or QApplication(sys.argv)
    accepted = run_setup_dialog()
    if accepted:
        app.processEvents()
    return accepted


def run_koe(initial_command: str | None = None):
    """Run the main Koe application."""
    load_dotenv(env_path(), override=True)
    from main import main

    return main(initial_command=initial_command)


def run_scribe_window():
    load_dotenv(env_path(), override=True)
    from meeting.app import main

    return main()


if __name__ == '__main__':
    ensure_runtime_dirs()
    if '--scribe-window' in sys.argv:
        run_scribe_window()
        raise SystemExit(0)

    if '--setup' in sys.argv:
        if not run_setup():
            raise SystemExit(1)
    elif needs_setup() and not run_setup():
        raise SystemExit(1)

    requested_command = None
    if '--snippet' in sys.argv:
        requested_command = 'snippet'
    elif '--scribe' in sys.argv:
        requested_command = 'scribe'

    if requested_command:
        from commands import send_command

        if send_command(requested_command):
            raise SystemExit(0)
    run_koe(initial_command=requested_command)
