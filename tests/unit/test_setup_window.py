import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_setup_window_uses_readable_current_api_key_link(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from PyQt5.QtWidgets import QApplication, QLabel
    from ui import theme
    from ui.setup_window import SetupWindow

    app = QApplication.instance() or QApplication([])
    window = SetupWindow()
    labels = [label.text() for label in window.findChildren(QLabel)]
    combined = " ".join(labels)

    assert "Welcome to Koe" in labels
    assert "Add your details once" not in combined
    assert "https://elevenlabs.io/app/api/api-keys" in combined
    assert theme.LINK_COLOR.lower() in combined.lower()

    window.close()
    app.processEvents()


def test_first_run_defaults_have_empty_custom_corrections(monkeypatch):
    from paths import config_path
    from ui.setup_window import write_setup_files

    write_setup_files("Alex", "eleven-key")
    config = yaml.safe_load(config_path().read_text(encoding="utf-8"))

    assert config["profile"]["user_name"] == "Alex"
    assert config["meeting_options"]["save_markdown"] is False
    assert config["meeting_options"]["last_meeting_mode"] == "online_one_on_one"
    assert config["transcription_options"]["provider"] == "elevenlabs"
    assert config["transcription_options"]["corrections"] == {}


def test_first_run_preserves_preloaded_openrouter_key():
    from paths import env_path
    from ui.setup_window import write_setup_files

    env_path().parent.mkdir(parents=True, exist_ok=True)
    env_path().write_text("OPENROUTER_API_KEY=test-summary-key\n", encoding="utf-8")

    write_setup_files("Alex", "eleven-key")
    contents = env_path().read_text(encoding="utf-8")

    assert "ELEVENLABS_API_KEY=eleven-key" in contents
    assert "OPENROUTER_API_KEY=test-summary-key" in contents


def test_elevenlabs_key_validation_uses_batch_scribe_token_endpoint(monkeypatch):
    from ui import setup_window

    captured = {}

    class Response:
        status_code = 200

    def fake_post(url, headers, timeout):
        captured.update(url=url, headers=headers, timeout=timeout)
        return Response()

    monkeypatch.setattr(setup_window.requests, "post", fake_post)

    assert setup_window.validate_elevenlabs_key("test-key") == (True, "")
    assert captured["url"] == (
        "https://api.elevenlabs.io/v1/single-use-token/batch_scribe"
    )
    assert captured["headers"] == {"xi-api-key": "test-key"}


def test_elevenlabs_key_validation_explains_missing_speech_to_text_access(
    monkeypatch,
):
    from ui import setup_window

    class Response:
        status_code = 403

    monkeypatch.setattr(
        setup_window.requests,
        "post",
        lambda url, headers, timeout: Response(),
    )

    valid, message = setup_window.validate_elevenlabs_key("restricted-key")

    assert valid is False
    assert "Speech to Text" in message
