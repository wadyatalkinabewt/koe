"""First-run GUI onboarding for an installed Koe application."""

from __future__ import annotations

import os
from pathlib import Path

import requests
import yaml
from dotenv import dotenv_values
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from compat import apply_window_icon, enable_dark_titlebar
from paths import app_data_dir, config_path, env_path, resource_path, setup_marker_path
from ui import theme


ELEVENLABS_USER_URL = "https://api.elevenlabs.io/v1/user"


def validate_elevenlabs_key(api_key: str, timeout: float = 15.0) -> tuple[bool, str]:
    """Validate a key without consuming transcription credits."""
    try:
        response = requests.get(
            ELEVENLABS_USER_URL,
            headers={"xi-api-key": api_key.strip()},
            timeout=timeout,
        )
    except requests.Timeout:
        return False, "ElevenLabs did not respond in time. Check the connection and try again."
    except requests.RequestException as exc:
        return False, f"Could not reach ElevenLabs: {exc}"
    if response.status_code == 200:
        return True, ""
    if response.status_code in (401, 403):
        return False, "That ElevenLabs API key was not accepted."
    return False, f"ElevenLabs returned HTTP {response.status_code}. Try again shortly."


def _existing_openrouter_key() -> str:
    values = dotenv_values(env_path()) if env_path().is_file() else {}
    return str(values.get("OPENROUTER_API_KEY") or "").strip()


def write_setup_files(user_name: str, elevenlabs_key: str, openrouter_key: str = "") -> None:
    """Write the first-run secrets and cost-conscious default preferences."""
    app_data_dir().mkdir(parents=True, exist_ok=True)
    existing_openrouter = _existing_openrouter_key()
    effective_openrouter = openrouter_key.strip() or existing_openrouter

    env_lines = [f"ELEVENLABS_API_KEY={elevenlabs_key.strip()}"]
    if effective_openrouter:
        env_lines.append(f"OPENROUTER_API_KEY={effective_openrouter}")
    env_temp = env_path().with_suffix(".tmp")
    env_temp.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    os.replace(env_temp, env_path())

    config = {
        "profile": {"user_name": user_name.strip()},
        "meeting_options": {
            "root_folder": None,
            "save_audio": False,
            "save_markdown": False,
            "last_meeting_mode": "online_one_on_one",
        },
        "model_options": {
            "common": {"language": None, "initial_prompt": None},
            "elevenlabs": {"keyterms_enabled": False},
        },
        "recording_options": {
            "activation_key": "ctrl+shift+space",
            "save_audio": False,
        },
        "misc": {
            "print_to_terminal": False,
            "hide_status_window": False,
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
        self.setStyleSheet(theme.application_stylesheet())

        icon_path = resource_path("assets", "koe-icon.ico")
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        apply_window_icon(self, icon_path, app_id="Koe.Setup.App")

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 28)
        root.setSpacing(12)

        title = QLabel("Welcome to Koe")
        title.setObjectName("windowTitle")
        root.addWidget(title)

        subtitle = QLabel("Add your details once, then Koe is ready whenever you are.")
        subtitle.setObjectName("windowSubtitle")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)
        root.addSpacing(10)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignTop)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(14)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Your first name")
        self.name_input.setMinimumHeight(40)
        form.addRow("Your Name", self.name_input)

        self.elevenlabs_input = QLineEdit()
        self.elevenlabs_input.setEchoMode(QLineEdit.Password)
        self.elevenlabs_input.setPlaceholderText("Paste your ElevenLabs API key")
        self.elevenlabs_input.setMinimumHeight(40)
        form.addRow("ElevenLabs Key", self.elevenlabs_input)

        self.openrouter_input: QLineEdit | None = None
        if not _existing_openrouter_key():
            self.openrouter_input = QLineEdit()
            self.openrouter_input.setEchoMode(QLineEdit.Password)
            self.openrouter_input.setPlaceholderText("Optional — Scribe meeting summaries")
            self.openrouter_input.setMinimumHeight(40)
            form.addRow("OpenRouter Key", self.openrouter_input)

        root.addLayout(form)

        help_label = QLabel(
            '<a href="https://elevenlabs.io/app/settings/api-keys">Create or copy an ElevenLabs API key</a>'
        )
        help_label.setOpenExternalLinks(True)
        help_label.setObjectName("windowSubtitle")
        root.addWidget(help_label)
        root.addSpacing(8)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("windowSubtitle")
        root.addWidget(self.status_label)

        actions = QHBoxLayout()
        actions.addStretch()
        self.finish_button = QPushButton("Finish Setup")
        self.finish_button.setObjectName("primaryButton")
        self.finish_button.setMinimumSize(126, 40)
        self.finish_button.clicked.connect(self._finish)
        actions.addWidget(self.finish_button)
        root.addLayout(actions)

    def _finish(self) -> None:
        name = self.name_input.text().strip()
        elevenlabs_key = self.elevenlabs_input.text().strip()
        if not name:
            self.status_label.setText("Enter the name Koe should use for your voice.")
            self.name_input.setFocus()
            return
        if not elevenlabs_key:
            self.status_label.setText("An ElevenLabs API key is required for transcription.")
            self.elevenlabs_input.setFocus()
            return

        self.finish_button.setEnabled(False)
        self.status_label.setText("Checking ElevenLabs…")
        QApplication.processEvents()
        valid, message = validate_elevenlabs_key(elevenlabs_key)
        if not valid:
            self.status_label.setText(message)
            self.finish_button.setEnabled(True)
            return

        try:
            openrouter_key = self.openrouter_input.text() if self.openrouter_input else ""
            write_setup_files(name, elevenlabs_key, openrouter_key)
        except OSError as exc:
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
