"""
Pytest fixtures for Koe tests.
"""

import os
import sys
import tempfile
from pathlib import Path
import pytest

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_transcript_content():
    """Sample transcript content for testing."""
    return """# Test Meeting

**Date**: 2026-01-20 14:30
**Duration**: 25 minutes
**Participants**: Bryce, Calum, Speaker 1

---

## Agenda
- Test item 1
- Test item 2

## Notes
Some test notes here.

## Action Items
- [ ] Task 1
- [ ] Task 2

---

## Full Transcript

**[00:00] Bryce**: Hello everyone.

**[00:05] Calum**: Hi there.

**[00:10] Speaker 1**: Good to be here.
"""


@pytest.fixture
def sample_recovery_data():
    """Sample recovery file content (JSONL format)."""
    return '''{"_type": "header", "meeting_start": "2026-01-20T14:30:00", "meeting_name": "Test Meeting", "output_dir": "/tmp/meetings"}
{"timestamp": 0.0, "speaker": "Bryce", "text": "Hello everyone.", "source": "mic"}
{"timestamp": 5.0, "speaker": "Calum", "text": "Hi there.", "source": "loopback"}
{"timestamp": 10.0, "speaker": "Speaker 1", "text": "Good to be here.", "source": "loopback"}
'''


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "profile": {
            "user_name": "Test User",
            "my_voice_embedding": None
        },
        "model_options": {
            "engine": "whisper",
            "common": {
                "language": "en",
                "initial_prompt": "Use proper punctuation."
            },
            "local": {
                "model": "base",
                "device": "cpu",
                "vad_filter": True
            }
        },
        "recording_options": {
            "activation_key": "ctrl+shift+space",
            "filter_snippets_to_my_voice": False
        },
        "meeting_options": {
            "root_folder": "/tmp/meetings"
        },
        "misc": {
            "noise_on_completion": True,
            "snippets_folder": None,
            "print_to_terminal": False
        }
    }
