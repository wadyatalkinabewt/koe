"""Stable resource and data locations for source and packaged Koe."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

APP_NAME = "Koe"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def source_root() -> Path:
    return Path(__file__).resolve().parent.parent


def install_root() -> Path:
    return Path(sys.executable).resolve().parent if is_frozen() else source_root()


def resource_root() -> Path:
    bundle_root = getattr(sys, "_MEIPASS", None)
    return Path(bundle_root).resolve() if bundle_root else source_root()


def resource_path(*parts: str) -> Path:
    return resource_root().joinpath(*parts)


def _known_documents_dir() -> Path:
    """Return Windows' redirected Documents location when available."""
    if sys.platform == "win32":
        buffer = ctypes.create_unicode_buffer(32768)
        try:
            result = ctypes.windll.shell32.SHGetFolderPathW(  # type: ignore[attr-defined]
                None,
                5,  # CSIDL_PERSONAL
                None,
                0,
                buffer,
            )
            if result == 0 and buffer.value:
                return Path(buffer.value)
        except Exception:
            pass
    return Path.home() / "Documents"


def app_data_dir() -> Path:
    override = os.environ.get("KOE_APPDATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if not is_frozen():
        return source_root()
    local_app_data = os.environ.get("LOCALAPPDATA")
    base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    return base / APP_NAME


def documents_dir() -> Path:
    override = os.environ.get("KOE_DOCUMENTS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if not is_frozen():
        return source_root()
    return _known_documents_dir() / APP_NAME


def env_path() -> Path:
    return app_data_dir() / ".env"


def config_path() -> Path:
    return app_data_dir() / "config.yaml"


def setup_marker_path() -> Path:
    return app_data_dir() / ".setup_complete"


def logs_dir() -> Path:
    return app_data_dir() / "logs"


def snippet_recovery_path() -> Path:
    """Single replaceable WAV retained only after a cancelled Snippet."""
    return logs_dir() / "recoverable-snippet.wav"


def scribe_temp_dir() -> Path:
    if not is_frozen() and not os.environ.get("KOE_APPDATA_DIR"):
        return source_root() / ".scribe_temp"
    return app_data_dir() / "scribe-temp"


def default_meetings_dir() -> Path:
    return documents_dir() / "Meetings"


def default_snippets_dir() -> Path:
    return documents_dir() / "Snippets"


def ensure_runtime_dirs() -> None:
    for path in (
        app_data_dir(),
        logs_dir(),
        scribe_temp_dir(),
        default_meetings_dir(),
        default_snippets_dir(),
    ):
        path.mkdir(parents=True, exist_ok=True)
