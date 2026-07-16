"""Shared pytest fixtures for the supported Koe application."""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(autouse=True)
def isolated_runtime_paths(tmp_path, monkeypatch):
    """Keep tests out of the real per-user Koe folders."""
    monkeypatch.setenv("KOE_APPDATA_DIR", str(tmp_path / "appdata"))
    monkeypatch.setenv("KOE_DOCUMENTS_DIR", str(tmp_path / "documents"))
    try:
        from utils import ConfigManager

        ConfigManager._instance = None
    except ImportError:
        pass
    yield
    try:
        from utils import ConfigManager

        ConfigManager._instance = None
    except ImportError:
        pass


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config():
    return {
        "profile": {"user_name": "Test User"},
        "model_options": {
            "common": {"language": None, "initial_prompt": "Koe, ElevenLabs"},
            "elevenlabs": {"keyterms_enabled": True},
        },
        "recording_options": {
            "activation_key": "ctrl+shift+space",
        },
        "meeting_options": {"root_folder": None, "save_audio": False},
        "misc": {
            "hide_status_window": False,
            "noise_on_completion": True,
            "snippets_folder": None,
            "print_to_terminal": False,
        },
    }
