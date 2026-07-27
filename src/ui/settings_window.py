"""Koe settings for the supported ElevenLabs-only desktop app."""

import sys
from pathlib import Path

from PyQt5.QtCore import QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPainter, QPen
from PyQt5.QtWidgets import (
    QAbstractButton,
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ui import theme
from ui.base_window import BaseWindow
from paths import default_meetings_dir, default_snippets_dir, resource_path
from utils import ConfigManager


def _label(text: str, object_name: str | None = None) -> QLabel:
    label = QLabel(text)
    if object_name:
        label.setObjectName(object_name)
    return label


class ToggleSwitch(QAbstractButton):
    """Small native-painted switch that stays crisp at every display scale."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(38, 22)

    def sizeHint(self) -> QSize:
        return QSize(38, 22)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        track = QRectF(1, 2, self.width() - 2, self.height() - 4)
        track_color = theme.ACCENT_COLOR if self.isChecked() else theme.SURFACE_HOVER
        border_color = theme.ACCENT_COLOR if self.isChecked() else theme.BORDER_COLOR
        if not self.isEnabled():
            track_color = theme.INPUT_BG
            border_color = theme.DIVIDER_COLOR
        painter.setPen(QPen(QColor(border_color), 1))
        painter.setBrush(QColor(track_color))
        painter.drawRoundedRect(track, track.height() / 2, track.height() / 2)

        diameter = 14
        knob_x = self.width() - diameter - 4 if self.isChecked() else 4
        knob_color = theme.TEXT_COLOR if self.isEnabled() else theme.DIM_TEXT
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(knob_color))
        painter.drawEllipse(QRectF(knob_x, 4, diameter, diameter))


class ToggleRow(QWidget):
    """A clean label-and-switch row with no shaded container."""

    toggled = pyqtSignal(bool)

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self._text = text
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(12)
        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.setMinimumWidth(0)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.label)
        layout.addStretch()
        self.switch = ToggleSwitch(self)
        self.switch.setAccessibleName(text)
        self.switch.toggled.connect(self.toggled.emit)
        layout.addWidget(self.switch)

    def text(self) -> str:
        return self._text

    def isChecked(self) -> bool:
        return self.switch.isChecked()

    def setChecked(self, checked: bool) -> None:
        self.switch.setChecked(checked)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.rect().contains(event.pos()):
            self.switch.toggle()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class SettingsWindow(BaseWindow):
    settings_closed = pyqtSignal()
    settings_saved = pyqtSignal()

    def __init__(self):
        super().__init__("Koe Settings", 620, 700)
        self._settings_width = 620
        self.setWindowFlags(
            (
                self.windowFlags()
                | Qt.MSWindowsFixedSizeDialogHint
                | Qt.WindowMinimizeButtonHint
                | Qt.WindowCloseButtonHint
            )
            & ~Qt.WindowMaximizeButtonHint
        )
        self._loading_values = True
        self._changed_since_show = False
        self._save_failed = False
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(350)
        self._save_timer.timeout.connect(self._save_values)
        icon_path = resource_path("assets", "koe-icon.ico")
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.setStyleSheet(theme.application_stylesheet() + self._local_stylesheet())
        self._build_ui()
        self._load_values()
        self._connect_autosave()
        self._loading_values = False
        QTimer.singleShot(0, self._refresh_content_size)

    @staticmethod
    def _local_stylesheet() -> str:
        return f"""
            QFrame#settingsHeader {{ background: {theme.BG_COLOR}; }}
            QLabel#autoSaveStatus {{
                color: {theme.SUCCESS_COLOR};
                font-size: 9pt;
                font-weight: 600;
            }}
            QLabel#subsectionTitle {{
                color: {theme.TEXT_COLOR};
                font-size: 9pt;
                font-weight: 600;
            }}
            QTextEdit:disabled {{
                background: #0D121B;
                border-color: {theme.DIVIDER_COLOR};
                color: {theme.DIM_TEXT};
            }}
        """

    def _build_ui(self) -> None:
        self.header = QFrame()
        self.header.setObjectName("settingsHeader")
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(28, 24, 28, 10)
        header_layout.addWidget(_label("Settings", "windowTitle"))
        header_layout.addStretch()
        self.save_status_label = _label("", "autoSaveStatus")
        self.save_status_label.hide()
        header_layout.addWidget(self.save_status_label, 0, Qt.AlignVCenter)
        self.main_layout.addWidget(self.header)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(28, 4, 28, 24)
        self.content_layout.setSpacing(14)
        self.content_layout.setAlignment(Qt.AlignTop)

        profile = self._card("Your Name")
        self.user_name_input = QLineEdit()
        self.user_name_input.setPlaceholderText("Your name")
        profile.layout().addWidget(self.user_name_input)
        self.content_layout.addWidget(profile)

        storage = self._card("Storage", "Choose where transcripts and snippets live.")
        self.snippets_input = self._folder_picker(
            "Snippets Folder",
            storage.layout(),
            f"Leave empty for {default_snippets_dir()}",
        )
        self.meetings_input = self._folder_picker(
            "Meetings Folder",
            storage.layout(),
            f"Leave empty for {default_meetings_dir()}",
        )
        self.save_meeting_audio_checkbox = ToggleRow("Save Scribe meeting audio")
        storage.layout().addWidget(self.save_meeting_audio_checkbox)
        self.content_layout.addWidget(storage)

        scribe = self._card(
            "Scribe",
            "PDF transcripts and summaries are always saved.",
        )
        self.save_markdown_checkbox = ToggleRow("Save Markdown copies")
        scribe.layout().addWidget(self.save_markdown_checkbox)
        self.content_layout.addWidget(scribe)

        snippet = self._card("Snippet")
        snippet.layout().addWidget(_label("Activation Hotkey", "subsectionTitle"))
        self.hotkey_input = QLineEdit()
        self.hotkey_input.setPlaceholderText("ctrl+shift+space")
        snippet.layout().addWidget(self.hotkey_input)
        self.save_snippet_audio_checkbox = ToggleRow("Save snippet audio")
        self.beep_checkbox = ToggleRow("Play a sound when a snippet is ready")
        self.status_checkbox = ToggleRow("Show the snippet status card")
        snippet.layout().addWidget(self.save_snippet_audio_checkbox)
        snippet.layout().addWidget(self.beep_checkbox)
        snippet.layout().addWidget(self.status_checkbox)
        self.content_layout.addWidget(snippet)

        transcription = self._card("Transcription")
        self.transcription_card = transcription
        transcription.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        transcription.layout().setAlignment(Qt.AlignTop)
        transcription.layout().setSpacing(9)
        self.keyterms_checkbox = ToggleRow(
            "Use vocabulary hints for names and technical terms"
        )
        self.keyterms_checkbox.label.setWordWrap(False)
        self.keyterms_checkbox.layout().setContentsMargins(0, 0, 0, 0)
        transcription.layout().addWidget(self.keyterms_checkbox)
        transcription.layout().addWidget(_label("Vocabulary", "subsectionTitle"))
        self.initial_prompt_edit = QTextEdit()
        self.initial_prompt_edit.setAcceptRichText(False)
        self.initial_prompt_edit.setPlaceholderText(
            "Comma-separated names, products, and technical terms"
        )
        self.initial_prompt_edit.setMinimumHeight(96)
        self.initial_prompt_edit.setMaximumHeight(240)
        transcription.layout().addWidget(self.initial_prompt_edit)
        self.content_layout.addWidget(transcription)

        self.scroll.setWidget(self.content)
        self.main_layout.addWidget(self.scroll, 1)

    @staticmethod
    def _card(title: str, description: str | None = None) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(9)
        layout.setAlignment(Qt.AlignTop)
        layout.addWidget(_label(title, "sectionTitle"))
        if description:
            description_label = _label(description, "windowSubtitle")
            description_label.setWordWrap(True)
            layout.addWidget(description_label)
            layout.addSpacing(2)
        return card

    def _folder_picker(
        self,
        label_text: str,
        parent_layout: QVBoxLayout,
        placeholder: str,
    ) -> QLineEdit:
        parent_layout.addWidget(_label(label_text, "subsectionTitle"))
        row = QHBoxLayout()
        row.setSpacing(8)
        field = QLineEdit()
        field.setPlaceholderText(placeholder)
        row.addWidget(field, 1)
        browse = QPushButton("Browse")
        browse.setFixedWidth(78)
        browse.clicked.connect(lambda: self._browse(field))
        row.addWidget(browse)
        parent_layout.addLayout(row)
        return field

    def _browse(self, field: QLineEdit) -> None:
        start = field.text() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Select folder", start)
        if folder:
            field.setText(folder)

    def _load_values(self) -> None:
        self._loading_values = True
        self.user_name_input.setText(
            ConfigManager.get_config_value("profile", "user_name") or ""
        )
        self.meetings_input.setText(
            ConfigManager.get_config_value("meeting_options", "root_folder") or ""
        )
        self.snippets_input.setText(
            ConfigManager.get_config_value("misc", "snippets_folder") or ""
        )
        self.save_meeting_audio_checkbox.setChecked(
            bool(ConfigManager.get_config_value("meeting_options", "save_audio"))
        )
        self.save_markdown_checkbox.setChecked(
            bool(ConfigManager.get_config_value("meeting_options", "save_markdown"))
        )
        self.hotkey_input.setText(
            ConfigManager.get_config_value("recording_options", "activation_key")
            or "ctrl+shift+space"
        )
        self.save_snippet_audio_checkbox.setChecked(
            bool(ConfigManager.get_config_value("recording_options", "save_audio"))
        )
        self.beep_checkbox.setChecked(
            bool(ConfigManager.get_config_value("misc", "noise_on_completion"))
        )
        self.status_checkbox.setChecked(
            not bool(ConfigManager.get_config_value("misc", "hide_status_window"))
        )
        self.keyterms_checkbox.setChecked(
            bool(
                ConfigManager.get_config_value(
                    "model_options", "elevenlabs", "keyterms_enabled"
                )
            )
        )
        self.initial_prompt_edit.setPlainText(
            ConfigManager.get_config_value("model_options", "common", "initial_prompt")
            or ""
        )
        self._sync_vocabulary_editor()
        self._loading_values = False

    def _connect_autosave(self) -> None:
        for field in (
            self.user_name_input,
            self.meetings_input,
            self.snippets_input,
            self.hotkey_input,
        ):
            field.textChanged.connect(self._schedule_save)
        for toggle in (
            self.save_meeting_audio_checkbox,
            self.save_markdown_checkbox,
            self.save_snippet_audio_checkbox,
            self.beep_checkbox,
            self.status_checkbox,
            self.keyterms_checkbox,
        ):
            toggle.toggled.connect(self._schedule_save)
        self.keyterms_checkbox.toggled.connect(self._sync_vocabulary_editor)
        self.initial_prompt_edit.textChanged.connect(self._schedule_save)
        self.initial_prompt_edit.textChanged.connect(self._refresh_content_size)

    def _sync_vocabulary_editor(self, *_args) -> None:
        enabled = self.keyterms_checkbox.isChecked()
        self.initial_prompt_edit.setEnabled(enabled)
        self.initial_prompt_edit.setToolTip(
            "" if enabled else "Turn on vocabulary hints to edit this list."
        )

    def _refresh_content_size(self, *_args) -> None:
        document_height = self.initial_prompt_edit.document().documentLayout().documentSize().height()
        editor_height = max(96, min(240, int(document_height) + 28))
        self.initial_prompt_edit.setFixedHeight(editor_height)
        QTimer.singleShot(0, self._fit_window_to_content)

    def _fit_window_to_content(self) -> None:
        self.content.adjustSize()
        desired_height = self.header.sizeHint().height() + self.content.sizeHint().height() + 8
        screen = QApplication.screenAt(self.frameGeometry().center()) or QApplication.primaryScreen()
        maximum_height = screen.availableGeometry().height() - 48 if screen else 860
        self.setFixedSize(
            self._settings_width,
            max(560, min(desired_height, maximum_height)),
        )

    def _schedule_save(self, *_args) -> None:
        if self._loading_values:
            return
        self._changed_since_show = True
        self.save_status_label.show()
        self.save_status_label.setText("Saving…")
        self._save_timer.start()

    def _save_values(self) -> None:
        ConfigManager.set_config_value(
            self.user_name_input.text().strip() or None, "profile", "user_name"
        )
        ConfigManager.set_config_value(
            self.meetings_input.text().strip() or None,
            "meeting_options",
            "root_folder",
        )
        ConfigManager.set_config_value(
            self.snippets_input.text().strip() or None, "misc", "snippets_folder"
        )
        ConfigManager.set_config_value(
            self.save_meeting_audio_checkbox.isChecked(),
            "meeting_options",
            "save_audio",
        )
        ConfigManager.set_config_value(
            self.save_markdown_checkbox.isChecked(),
            "meeting_options",
            "save_markdown",
        )
        ConfigManager.set_config_value(
            self.hotkey_input.text().strip() or "ctrl+shift+space",
            "recording_options",
            "activation_key",
        )
        ConfigManager.set_config_value(
            self.save_snippet_audio_checkbox.isChecked(),
            "recording_options",
            "save_audio",
        )
        ConfigManager.set_config_value(
            self.beep_checkbox.isChecked(), "misc", "noise_on_completion"
        )
        ConfigManager.set_config_value(
            not self.status_checkbox.isChecked(), "misc", "hide_status_window"
        )
        ConfigManager.set_config_value(
            self.keyterms_checkbox.isChecked(),
            "model_options",
            "elevenlabs",
            "keyterms_enabled",
        )
        ConfigManager.set_config_value(
            self.initial_prompt_edit.toPlainText().strip() or None,
            "model_options",
            "common",
            "initial_prompt",
        )
        try:
            ConfigManager.save_config()
            self._save_failed = False
            self.save_status_label.show()
            self.save_status_label.setStyleSheet("")
            self.save_status_label.setText("Saved")
            self.save_status_label.setToolTip("")
        except Exception as exc:
            self._save_failed = True
            self.save_status_label.show()
            self.save_status_label.setText("Couldn’t Save")
            self.save_status_label.setStyleSheet(f"color: {theme.ERROR_COLOR};")
            self.save_status_label.setToolTip(str(exc))

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()
            event.accept()
            return
        super().keyPressEvent(event)

    def showEvent(self, event) -> None:
        self._load_values()
        self._changed_since_show = False
        self.save_status_label.setStyleSheet("")
        self.save_status_label.clear()
        self.save_status_label.hide()
        self._refresh_content_size()
        super().showEvent(event)

    def closeEvent(self, event) -> None:
        if self._save_timer.isActive():
            self._save_timer.stop()
            self._save_values()
        if self._changed_since_show and not self._save_failed:
            self.settings_saved.emit()
        self.settings_closed.emit()
        event.accept()
