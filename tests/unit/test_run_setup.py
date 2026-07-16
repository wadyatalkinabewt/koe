import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import run


def _prepare(root: Path, env_text: str, marker: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text("profile: {}\n", encoding="utf-8")
    (root / ".env").write_text(env_text, encoding="utf-8")
    if marker:
        (root / ".setup_complete").touch()


def test_valid_elevenlabs_key_completes_setup(tmp_path):
    _prepare(tmp_path, "ELEVENLABS_API_KEY=valid-key\n")

    assert run.needs_setup(tmp_path) is False
    assert (tmp_path / ".setup_complete").exists()


def test_empty_key_cannot_use_stale_marker(tmp_path):
    _prepare(tmp_path, "ELEVENLABS_API_KEY=\n", marker=True)

    assert run.needs_setup(tmp_path) is True
    assert not (tmp_path / ".setup_complete").exists()


def test_unrelated_key_cannot_complete_setup(tmp_path):
    _prepare(tmp_path, "SOME_OTHER_API_KEY=legacy-key\n", marker=True)

    assert run.needs_setup(tmp_path) is True
    assert not (tmp_path / ".setup_complete").exists()
