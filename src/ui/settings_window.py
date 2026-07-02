"""
Settings window — minimal cloud-transcription config.

Exposes the handful of things that change at runtime:
  - Profile (your name)
  - Output folders (meetings, snippets)
  - Recording (hotkey, beep on completion)
  - Transcription backend
  - AI cleanup (toggle, threshold, model, prompt prefix)
  - STT vocab hint
"""

import sys
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QApplication, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QCheckBox, QWidget, QScrollArea, QFileDialog, QTextEdit, QSpinBox,
    QComboBox,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ui.base_window import BaseWindow
from ui import theme
from utils import ConfigManager


def _label(text: str, color: str = None, size_pt: int = None) -> QLabel:
    lbl = QLabel(text)
    style = []
    if color:
        style.append(f"color: {color};")
    if size_pt:
        style.append(f"font-size: {size_pt}pt;")
    if style:
        lbl.setStyleSheet(" ".join(style))
    return lbl


class SettingsWindow(BaseWindow):
    settings_closed = pyqtSignal()
    settings_saved = pyqtSignal()

    def __init__(self):
        super().__init__("Settings", 540, 720)

        # Set window icon
        icon_path = Path(__file__).parent.parent.parent / "assets" / "koe-icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self._build_ui()
        self._load_values()

    # ---------- styling ----------

    def _stylesheet(self) -> str:
        return f"""
            QWidget {{
                background-color: {theme.BG_COLOR};
                color: {theme.TEXT_COLOR};
                font-family: 'Cascadia Code', Consolas, monospace;
            }}
            QLabel {{ color: {theme.TEXT_COLOR}; }}
            QLineEdit, QTextEdit, QSpinBox, QComboBox {{
                background-color: {theme.INPUT_BG};
                color: {theme.TEXT_COLOR};
                border: 1px solid {theme.INPUT_BORDER};
                border-radius: 6px;
                padding: 6px 8px;
                font-family: 'Cascadia Code', Consolas, monospace;
                font-size: 10pt;
            }}
            QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QComboBox:focus {{
                border: 1px solid {theme.INPUT_FOCUS_BORDER};
            }}
            QComboBox::drop-down {{ border: none; width: 24px; }}
            QPushButton {{
                background-color: {theme.BUTTON_BG};
                color: {theme.TEXT_COLOR};
                border: 1px solid {theme.BUTTON_BORDER};
                border-radius: 6px;
                padding: 6px 14px;
                font-family: 'Cascadia Code', Consolas, monospace;
                font-size: 10pt;
            }}
            QPushButton:hover {{ background-color: {theme.BUTTON_HOVER_BG}; }}
            QCheckBox {{ color: {theme.TEXT_COLOR}; padding: 4px 0; }}
            QCheckBox::indicator {{
                width: 14px; height: 14px;
                border: 1px solid {theme.INPUT_BORDER};
                border-radius: 3px;
                background: {theme.INPUT_BG};
            }}
            QCheckBox::indicator:checked {{ background: {theme.TEXT_COLOR}; }}
            QScrollArea {{ border: none; background: {theme.BG_COLOR}; }}
            QScrollBar:vertical {{
                background: {theme.SCROLLBAR_BG}; width: 8px; border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {theme.SCROLLBAR_HANDLE}; border-radius: 4px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {theme.SCROLLBAR_HANDLE_HOVER};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """

    # ---------- UI scaffolding ----------

    def _build_ui(self):
        # Wrap the whole content in a scroll area so the window stays compact
        self.setStyleSheet(self._stylesheet())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 8, 16, 16)
        layout.setSpacing(14)

        title = _label("settings", theme.TEXT_COLOR, 16)
        f = QFont("Cascadia Code", 16, QFont.Bold)
        title.setFont(f)
        layout.addWidget(title)

        # ----- profile -----
        layout.addWidget(self._section_label("PROFILE"))
        layout.addWidget(_label("Your name", theme.SECONDARY_TEXT, 9))
        self.user_name_input = QLineEdit()
        self.user_name_input.setPlaceholderText("Used to label your audio in Scribe transcripts")
        layout.addWidget(self.user_name_input)

        # ----- output folders -----
        layout.addWidget(self._section_label("OUTPUT FOLDERS"))
        self.meetings_input = self._folder_picker("Meetings folder", layout, default_hint="<koe>/Meetings")
        self.snippets_input = self._folder_picker("Snippets folder", layout, default_hint="<koe>/Snippets")

        # ----- recording -----
        layout.addWidget(self._section_label("RECORDING"))
        layout.addWidget(_label("Activation hotkey", theme.SECONDARY_TEXT, 9))
        self.hotkey_input = QLineEdit()
        self.hotkey_input.setPlaceholderText("ctrl+shift+space")
        layout.addWidget(self.hotkey_input)

        self.beep_checkbox = QCheckBox("Play sound on snippet completion")
        layout.addWidget(self.beep_checkbox)

        # ----- transcription -----
        layout.addWidget(self._section_label("TRANSCRIPTION"))
        layout.addWidget(_label("Backend", theme.SECONDARY_TEXT, 9))
        self.provider_combo = QComboBox()
        self.provider_combo.addItem("ElevenLabs Scribe v2", "elevenlabs")
        self.provider_combo.addItem("Groq Whisper Large v3", "groq")
        layout.addWidget(self.provider_combo)

        self.keyterms_checkbox = QCheckBox("Send vocab hint as ElevenLabs keyterms")
        layout.addWidget(self.keyterms_checkbox)

        # ----- AI cleanup -----
        layout.addWidget(self._section_label("AI CLEANUP (snippets)"))
        self.cleanup_enabled_checkbox = QCheckBox("Enable AI cleanup")
        layout.addWidget(self.cleanup_enabled_checkbox)

        threshold_row = QHBoxLayout()
        threshold_row.addWidget(_label("Minimum duration (seconds):", theme.SECONDARY_TEXT, 9))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 300)
        self.threshold_spin.setSingleStep(5)
        threshold_row.addWidget(self.threshold_spin)
        threshold_row.addStretch()
        layout.addLayout(threshold_row)

        layout.addWidget(_label("Cleanup model (OpenRouter slug)", theme.SECONDARY_TEXT, 9))
        self.cleanup_model_input = QLineEdit()
        self.cleanup_model_input.setPlaceholderText("google/gemini-3.5-flash")
        layout.addWidget(self.cleanup_model_input)

        layout.addWidget(_label("Cleanup prompt prefix", theme.SECONDARY_TEXT, 9))
        self.cleanup_prompt_edit = QTextEdit()
        self.cleanup_prompt_edit.setAcceptRichText(False)
        self.cleanup_prompt_edit.setMinimumHeight(140)
        layout.addWidget(self.cleanup_prompt_edit)

        # ----- STT hint -----
        layout.addWidget(self._section_label("STT VOCAB HINT"))
        layout.addWidget(_label("Comma-separated proper nouns to bias transcription", theme.SECONDARY_TEXT, 9))
        self.initial_prompt_edit = QTextEdit()
        self.initial_prompt_edit.setAcceptRichText(False)
        self.initial_prompt_edit.setMinimumHeight(80)
        layout.addWidget(self.initial_prompt_edit)

        # ----- save row -----
        button_row = QHBoxLayout()
        button_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.close)
        button_row.addWidget(cancel_btn)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save_and_close)
        button_row.addWidget(save_btn)
        layout.addLayout(button_row)

        layout.addStretch()

        scroll.setWidget(content)
        self.main_layout.addWidget(scroll)

    def _section_label(self, text: str) -> QLabel:
        lbl = _label(text, theme.SECONDARY_TEXT, 9)
        f = QFont("Cascadia Code", 9, QFont.Bold)
        lbl.setFont(f)
        return lbl

    def _folder_picker(self, label_text: str, parent_layout: QVBoxLayout, default_hint: str) -> QLineEdit:
        parent_layout.addWidget(_label(label_text, theme.SECONDARY_TEXT, 9))
        row = QHBoxLayout()
        field = QLineEdit()
        field.setPlaceholderText(f"leave empty for default ({default_hint})")
        row.addWidget(field)
        browse = QPushButton("Browse")
        browse.setMinimumWidth(80)
        browse.clicked.connect(lambda: self._browse(field))
        row.addWidget(browse)
        parent_layout.addLayout(row)
        return field

    def _browse(self, field: QLineEdit):
        start = field.text() or str(Path.home() / "Desktop")
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", start)
        if folder:
            field.setText(folder)

    # ---------- load + save ----------

    def _load_values(self):
        self.user_name_input.setText(ConfigManager.get_config_value("profile", "user_name") or "")
        self.meetings_input.setText(ConfigManager.get_config_value("meeting_options", "root_folder") or "")
        self.snippets_input.setText(ConfigManager.get_config_value("misc", "snippets_folder") or "")
        self.hotkey_input.setText(
            ConfigManager.get_config_value("recording_options", "activation_key") or "ctrl+shift+space"
        )
        self.beep_checkbox.setChecked(
            bool(ConfigManager.get_config_value("misc", "noise_on_completion"))
        )
        provider = ConfigManager.get_config_value("model_options", "transcription_provider") or "elevenlabs"
        provider_idx = self.provider_combo.findData(provider)
        self.provider_combo.setCurrentIndex(provider_idx if provider_idx >= 0 else 0)
        self.keyterms_checkbox.setChecked(
            bool(ConfigManager.get_config_value("model_options", "elevenlabs", "keyterms_enabled"))
        )

        self.cleanup_enabled_checkbox.setChecked(
            bool(ConfigManager.get_config_value("post_processing", "ai_cleanup_enabled"))
        )
        self.threshold_spin.setValue(
            int(ConfigManager.get_config_value("post_processing", "ai_cleanup_threshold") or 10)
        )
        self.cleanup_model_input.setText(
            ConfigManager.get_config_value("post_processing", "ai_cleanup_model") or ""
        )
        self.cleanup_prompt_edit.setPlainText(
            ConfigManager.get_config_value("post_processing", "ai_cleanup_prompt") or ""
        )
        self.initial_prompt_edit.setPlainText(
            ConfigManager.get_config_value("model_options", "common", "initial_prompt") or ""
        )

    def _save_and_close(self):
        ConfigManager.set_config_value(self.user_name_input.text().strip() or None,
                                       "profile", "user_name")
        ConfigManager.set_config_value(self.meetings_input.text().strip() or None,
                                       "meeting_options", "root_folder")
        ConfigManager.set_config_value(self.snippets_input.text().strip() or None,
                                       "misc", "snippets_folder")
        ConfigManager.set_config_value(self.hotkey_input.text().strip() or "ctrl+shift+space",
                                       "recording_options", "activation_key")
        ConfigManager.set_config_value(self.beep_checkbox.isChecked(),
                                       "misc", "noise_on_completion")
        ConfigManager.set_config_value(self.provider_combo.currentData() or "elevenlabs",
                                       "model_options", "transcription_provider")
        ConfigManager.set_config_value(self.keyterms_checkbox.isChecked(),
                                       "model_options", "elevenlabs", "keyterms_enabled")

        ConfigManager.set_config_value(self.cleanup_enabled_checkbox.isChecked(),
                                       "post_processing", "ai_cleanup_enabled")
        ConfigManager.set_config_value(self.threshold_spin.value(),
                                       "post_processing", "ai_cleanup_threshold")
        ConfigManager.set_config_value(
            self.cleanup_model_input.text().strip() or "google/gemini-3.5-flash",
            "post_processing", "ai_cleanup_model",
        )
        ConfigManager.set_config_value(self.cleanup_prompt_edit.toPlainText() or None,
                                       "post_processing", "ai_cleanup_prompt")
        ConfigManager.set_config_value(self.initial_prompt_edit.toPlainText() or None,
                                       "model_options", "common", "initial_prompt")

        ConfigManager.save_config()
        self.settings_saved.emit()
        self.close()

    def closeEvent(self, event):
        self.settings_closed.emit()
        event.accept()
