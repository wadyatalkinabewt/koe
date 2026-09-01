import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_setup_window_changes_key_help_with_transcription_provider(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from PyQt5.QtCore import QPoint
    from PyQt5.QtWidgets import QApplication, QLabel
    from ui import theme
    from ui.setup_window import SetupWindow

    app = QApplication.instance() or QApplication([])
    window = SetupWindow()
    labels = [label.text() for label in window.findChildren(QLabel)]
    combined = " ".join(labels)

    assert "Welcome to Koe" in labels
    assert "Add your details once" not in combined
    assert "Welcome to Koe" in labels
    assert "Your name" in labels
    assert "Provider" in labels
    assert "API key" in labels
    assert "OpenRouter API key" in labels
    assert "Finish Setup" not in combined
    assert window.finish_button.text() == "Finish setup"
    assert window.openrouter_input.placeholderText() == (
        "Optional. Adds structured meeting summaries."
    )
    assert window.findChildren(QLabel, "setupFieldLabel")
    assert all(
        field.mapTo(window, QPoint()).x() == window.name_input.mapTo(window, QPoint()).x()
        for field in (
            window.provider_combo,
            window.api_key_input,
            window.openrouter_input,
        )
    )
    assert "https://elevenlabs.io/app/api/api-keys" in combined
    assert theme.LINK_COLOR.lower() in combined.lower()
    assert window.provider_combo.count() == 3
    assert [
        window.provider_combo.itemData(index)
        for index in range(window.provider_combo.count())
    ] == ["elevenlabs", "deepgram", "mistral"]

    window.provider_combo.setCurrentIndex(1)
    assert "Deepgram" in window.api_key_input.placeholderText()
    assert "https://console.deepgram.com/" in window.help_label.text()
    window.provider_combo.setCurrentIndex(2)
    assert "Mistral" in window.api_key_input.placeholderText()
    assert "https://console.mistral.ai/api-keys" in window.help_label.text()

    window.close()
    app.processEvents()


def test_first_run_defaults_have_empty_custom_corrections(monkeypatch):
    from paths import config_path
    from ui.setup_window import write_setup_files

    write_setup_files("Alex", "elevenlabs", "eleven-key")
    config = yaml.safe_load(config_path().read_text(encoding="utf-8"))

    assert config["profile"]["user_name"] == "Alex"
    assert config["meeting_options"]["save_markdown"] is False
    assert config["meeting_options"]["last_meeting_mode"] == "online_one_on_one"
    assert config["transcription_options"]["provider"] == "elevenlabs"
    assert config["transcription_options"]["corrections"] == {}


def test_first_run_writes_selected_provider_key_and_config():
    from paths import config_path, env_path
    from ui.setup_window import write_setup_files

    write_setup_files("Alex", "deepgram", "deepgram-key")

    config = yaml.safe_load(config_path().read_text(encoding="utf-8"))
    contents = env_path().read_text(encoding="utf-8")
    assert config["transcription_options"]["provider"] == "deepgram"
    assert "DEEPGRAM_API_KEY=deepgram-key" in contents


def test_first_run_preserves_preloaded_openrouter_key():
    from paths import env_path
    from ui.setup_window import write_setup_files

    env_path().parent.mkdir(parents=True, exist_ok=True)
    env_path().write_text("OPENROUTER_API_KEY=test-summary-key\n", encoding="utf-8")

    write_setup_files("Alex", "elevenlabs", "eleven-key")
    contents = env_path().read_text(encoding="utf-8")

    assert "ELEVENLABS_API_KEY=eleven-key" in contents
    assert "OPENROUTER_API_KEY=test-summary-key" in contents


def test_setup_validation_dispatches_to_deepgram_speech_to_text(monkeypatch):
    from ui import setup_window

    captured = {}

    def fake_transcribe(stream, api_key, **kwargs):
        captured.update(api_key=api_key, kwargs=kwargs, prefix=stream.read(4))
        return {"text": "", "words": []}, None

    monkeypatch.setattr(setup_window.deepgram, "transcribe_stream", fake_transcribe)

    assert setup_window.validate_provider_key("deepgram", "dg-key") == (True, "")
    assert captured["api_key"] == "dg-key"
    assert captured["kwargs"]["diarize"] is True
    assert captured["prefix"] == b"RIFF"


def test_setup_validation_dispatches_to_mistral_speech_to_text(monkeypatch):
    from ui import setup_window

    captured = {}

    def fake_transcribe(stream, filename, api_key, **kwargs):
        captured.update(
            filename=filename,
            api_key=api_key,
            kwargs=kwargs,
            prefix=stream.read(4),
        )
        return {"text": "", "words": []}, None

    monkeypatch.setattr(setup_window.mistral, "transcribe_stream", fake_transcribe)

    assert setup_window.validate_provider_key("mistral", "mi-key") == (True, "")
    assert captured["filename"] == "koe-setup-check.wav"
    assert captured["api_key"] == "mi-key"
    assert captured["kwargs"]["diarize"] is True
    assert captured["prefix"] == b"RIFF"


@pytest.mark.parametrize(
    ("provider_index", "provider", "env_key"),
    [
        (0, "elevenlabs", "ELEVENLABS_API_KEY"),
        (1, "deepgram", "DEEPGRAM_API_KEY"),
        (2, "mistral", "MISTRAL_API_KEY"),
    ],
)
def test_finish_setup_accepts_and_persists_each_provider(
    monkeypatch, provider_index, provider, env_key
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from PyQt5.QtWidgets import QApplication, QDialog
    from paths import config_path, env_path
    from ui import setup_window

    app = QApplication.instance() or QApplication([])
    validated = []
    monkeypatch.setattr(
        setup_window,
        "validate_provider_key",
        lambda selected, key: validated.append((selected, key)) or (True, ""),
    )

    window = setup_window.SetupWindow()
    window.name_input.setText("Alex")
    window.provider_combo.setCurrentIndex(provider_index)
    window.api_key_input.setText("provider-key")
    window._finish()

    config = yaml.safe_load(config_path().read_text(encoding="utf-8"))
    assert validated == [(provider, "provider-key")]
    assert config["transcription_options"]["provider"] == provider
    assert f"{env_key}=provider-key" in env_path().read_text(encoding="utf-8")
    assert window.result() == QDialog.Accepted
    window.close()
    app.processEvents()


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
