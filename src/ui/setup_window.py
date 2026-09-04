"""First-run GUI onboarding for an installed Koe application."""

from __future__ import annotations

from io import BytesIO
import os
import wave

import requests
import yaml
from dotenv import dotenv_values
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from compat import apply_window_icon, enable_dark_titlebar
from paths import app_data_dir, config_path, env_path, resource_path, setup_marker_path
from providers import deepgram, mistral
from ui import theme

ELEVENLABS_BATCH_SCRIBE_TOKEN_URL = (
    "https://api.elevenlabs.io/v1/single-use-token/batch_scribe"
)

PROVIDERS = {
    "elevenlabs": {
        "name": "ElevenLabs",
        "model": "Scribe v2",
        "env_key": "ELEVENLABS_API_KEY",
        "key_url": "https://elevenlabs.io/app/api/api-keys",
    },
    "deepgram": {
        "name": "Deepgram",
        "model": "Nova-3",
        "env_key": "DEEPGRAM_API_KEY",
        "key_url": "https://console.deepgram.com/",
    },
    "mistral": {
        "name": "Mistral",
        "model": "Voxtral Mini",
        "env_key": "MISTRAL_API_KEY",
        "key_url": "https://console.mistral.ai/api-keys",
    },
}


def validate_elevenlabs_key(api_key: str, timeout: float = 15.0) -> tuple[bool, str]:
    """Validate Speech to Text access without consuming transcription credits."""
    try:
        response = requests.post(
            ELEVENLABS_BATCH_SCRIBE_TOKEN_URL,
            headers={"xi-api-key": api_key.strip()},
            timeout=timeout,
        )
    except requests.Timeout:
        return (
            False,
            "ElevenLabs did not respond in time. Check the connection and try again.",
        )
    except requests.RequestException as exc:
        return False, f"Could not reach ElevenLabs: {exc}"
    if response.status_code == 200:
        return True, ""
    if response.status_code == 401:
        return False, "That ElevenLabs API key was not accepted."
    if response.status_code == 403:
        return (
            False,
            "That key cannot access Speech to Text. Enable Speech to Text access "
            "and check any IP restrictions.",
        )
    return False, f"ElevenLabs returned HTTP {response.status_code}. Try again shortly."


def _validation_audio() -> BytesIO:
    """Return a one-second silent WAV for exact transcription-access checks."""
    audio = BytesIO()
    with wave.open(audio, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 16000)
    audio.seek(0)
    return audio


def _provider_validation_message(provider: str, error: str | None) -> str:
    name = PROVIDERS[provider]["name"]
    detail = str(error or "The transcription check failed.")
    lowered = detail.casefold()
    if "http 401" in lowered:
        return f"That {name} API key was not accepted."
    if "http 402" in lowered:
        return f"{name} requires an active billing plan for transcription."
    if "http 403" in lowered:
        return f"That key cannot access {name} speech-to-text."
    if "timeout" in lowered:
        return f"{name} did not respond in time. Check the connection and try again."
    return detail


def validate_provider_key(
    provider: str, api_key: str, timeout: float = 30.0
) -> tuple[bool, str]:
    """Verify the selected key against its actual speech-to-text capability."""
    provider = provider.strip().casefold()
    if provider == "elevenlabs":
        return validate_elevenlabs_key(api_key, timeout=min(timeout, 15.0))
    audio = _validation_audio()
    if provider == "deepgram":
        result, error = deepgram.transcribe_stream(
            audio,
            api_key.strip(),
            language="en",
            diarize=True,
            timeout=timeout,
        )
    elif provider == "mistral":
        result, error = mistral.transcribe_stream(
            audio,
            "koe-setup-check.wav",
            api_key.strip(),
            language=None,
            diarize=True,
            timeout=timeout,
        )
    else:
        return False, "Choose a supported transcription provider."
    if result is not None:
        return True, ""
    return False, _provider_validation_message(provider, error)


def _existing_openrouter_key() -> str:
    values = dotenv_values(env_path()) if env_path().is_file() else {}
    return str(values.get("OPENROUTER_API_KEY") or "").strip()


def write_setup_files(
    user_name: str,
    provider: str,
    provider_key: str,
    openrouter_key: str = "",
) -> None:
    """Write the first-run secrets and cost-conscious default preferences."""
    provider = provider.strip().casefold()
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported transcription provider: {provider}")
    app_data_dir().mkdir(parents=True, exist_ok=True)
    existing_openrouter = _existing_openrouter_key()
    effective_openrouter = openrouter_key.strip() or existing_openrouter

    existing_values = dotenv_values(env_path()) if env_path().is_file() else {}
    env_values = {
        str(key): str(value)
        for key, value in existing_values.items()
        if key and value is not None
    }
    env_values[str(PROVIDERS[provider]["env_key"])] = provider_key.strip()
    if effective_openrouter:
        env_values["OPENROUTER_API_KEY"] = effective_openrouter
    env_lines = [f"{key}={value}" for key, value in env_values.items()]
    env_temp = env_path().with_suffix(".tmp")
    env_temp.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    os.replace(env_temp, env_path())

    config = {
        "profile": {"user_name": user_name.strip()},
        "meeting_options": {
            "root_folder": None,
            "save_audio": False,
        },
        "model_options": {
            "common": {"language": None},
        },
        "transcription_options": {
            "provider": provider,
            "corrections": {},
        },
        "recording_options": {
            "activation_key": "ctrl+shift+space",
        },
        "misc": {
            "print_to_terminal": False,
            "noise_on_completion": True,
            "snippets_folder": None,
        },
    }
    config_temp = config_path().with_suffix(".tmp")
    config_temp.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.replace(config_temp, config_path())
    setup_marker_path().touch()


class SetupWindow(QDialog):
    """Compact first-run form matching Koe's desktop visual system."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Set up Koe")
        self.setWindowFlags(
            (self.windowFlags() | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
            & ~Qt.WindowContextHelpButtonHint
        )
        self.setFixedWidth(520)
        self.setModal(True)
        self.setStyleSheet(
            theme.application_stylesheet()
            + f"""
                QLabel#setupFieldLabel {{
                    color: {theme.SECONDARY_TEXT};
                    font-size: 9pt;
                    font-weight: 650;
                }}
            """
        )

        icon_path = resource_path("assets", "koe-icon.ico")
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        apply_window_icon(self, icon_path, app_id="Koe.Setup.App")

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 21)
        root.setSpacing(7)

        header = QHBoxLayout()
        header.setSpacing(14)
        mark = QLabel()
        mark.setFixedSize(42, 42)
        mark.setAlignment(Qt.AlignCenter)
        mark_path = resource_path("assets", "koe-icon.png")
        if mark_path.exists():
            mark.setPixmap(
                QPixmap(str(mark_path)).scaled(
                    40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
        header.addWidget(mark, 0, Qt.AlignTop)
        heading = QVBoxLayout()
        heading.setSpacing(2)
        title = QLabel("Welcome to Koe")
        title.setObjectName("windowTitle")
        heading.addWidget(title)
        header.addLayout(heading, 1)
        root.addLayout(header)
        root.addSpacing(8)

        root.addWidget(self._field_label("Your name"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Your first name")
        self.name_input.setMinimumHeight(36)
        root.addWidget(self.name_input)

        root.addSpacing(2)
        root.addWidget(self._field_label("Provider"))
        self.provider_combo = QComboBox()
        self.provider_combo.setMinimumHeight(36)
        self.provider_combo.setAccessibleName("Transcription Provider")
        for provider, details in PROVIDERS.items():
            self.provider_combo.addItem(
                f'{details["name"]} — {details["model"]}', provider
            )
        root.addWidget(self.provider_combo)

        root.addSpacing(2)
        root.addWidget(self._field_label("API key"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setMinimumHeight(36)
        root.addWidget(self.api_key_input)

        self.help_label = QLabel()
        self.help_label.setOpenExternalLinks(True)
        self.help_label.setObjectName("windowSubtitle")
        root.addWidget(self.help_label)

        self.openrouter_input: QLineEdit | None = None
        if not _existing_openrouter_key():
            root.addSpacing(2)
            root.addWidget(self._field_label("OpenRouter API key"))
            self.openrouter_input = QLineEdit()
            self.openrouter_input.setEchoMode(QLineEdit.Password)
            self.openrouter_input.setPlaceholderText(
                "Optional. Adds structured meeting summaries."
            )
            self.openrouter_input.setMinimumHeight(36)
            root.addWidget(self.openrouter_input)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("windowSubtitle")
        root.addWidget(self.status_label)

        self.provider_combo.currentIndexChanged.connect(self._provider_changed)
        self._provider_changed()

        actions = QHBoxLayout()
        actions.addStretch()
        self.finish_button = QPushButton("Finish setup")
        self.finish_button.setObjectName("primaryButton")
        self.finish_button.setMinimumSize(126, 36)
        self.finish_button.clicked.connect(self._finish)
        actions.addWidget(self.finish_button)
        root.addLayout(actions)

    @staticmethod
    def _field_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("setupFieldLabel")
        return label

    def _selected_provider(self) -> str:
        return str(self.provider_combo.currentData())

    def _provider_changed(self, *_args) -> None:
        provider = self._selected_provider()
        details = PROVIDERS[provider]
        name = str(details["name"])
        self.api_key_input.setPlaceholderText(f"Paste your {name} API key")
        self.help_label.setText(
            f'<a href="{details["key_url"]}">'
            f'<span style="color: {theme.LINK_COLOR};">'
            f"Create or copy a {name} API key"
            "</span></a>"
        )
        self.status_label.clear()

    def _finish(self) -> None:
        name = self.name_input.text().strip()
        provider = self._selected_provider()
        provider_name = str(PROVIDERS[provider]["name"])
        provider_key = self.api_key_input.text().strip()
        if not name:
            self.status_label.setText("Enter the name Koe should use for your voice.")
            self.name_input.setFocus()
            return
        if not provider_key:
            self.status_label.setText(
                f"A {provider_name} API key is required for transcription."
            )
            self.api_key_input.setFocus()
            return

        self.finish_button.setEnabled(False)
        self.status_label.setText(f"Checking {provider_name} speech-to-text…")
        QApplication.processEvents()
        valid, message = validate_provider_key(provider, provider_key)
        if not valid:
            self.status_label.setText(message)
            self.finish_button.setEnabled(True)
            return

        try:
            openrouter_key = (
                self.openrouter_input.text() if self.openrouter_input else ""
            )
            write_setup_files(name, provider, provider_key, openrouter_key)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Could not save setup", str(exc))
            self.finish_button.setEnabled(True)
            return
        self.accept()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        enable_dark_titlebar(self)


def run_setup_dialog() -> bool:
    window = SetupWindow()
    return window.exec_() == QDialog.Accepted
