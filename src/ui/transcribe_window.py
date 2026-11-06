"""Desktop window for transcribing an existing audio file.

Thin shell over ``recording_transcript.transcribe_recording``: pick a file,
confirm the names, optionally ask for a summary, then watch the worker. All the
real work stays in the headless module so a pipeline can skip this window.
"""

import os
import socket
import sys
from pathlib import Path

from PyQt5.QtCore import QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compat import apply_window_icon, set_app_user_model_id  # noqa: E402
from paths import resource_path  # noqa: E402
from ui import theme  # noqa: E402
from ui.base_window import BaseWindow  # noqa: E402
from utils import ConfigManager  # noqa: E402

TRANSCRIBE_APP_ID = "Koe.Transcribe.App"
TRANSCRIBE_ICON = resource_path("assets", "koe-icon.ico")
_INSTANCE_PORT = 9879
AUDIO_FILTER = (
    "Audio and video (*.ogg *.opus *.m4a *.mp3 *.wav *.flac *.aac *.amr "
    "*.wma *.mp4 *.webm *.3gp *.caf *.aiff *.mkv);;All files (*)"
)


def acquire_transcribe_lock() -> socket.socket | None:
    """One window at a time, same approach as Scribe."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", _INSTANCE_PORT))
        sock.listen(1)
        return sock
    except OSError:
        return None


class TranscribeWorker(QThread):
    """Run one headless transcription off the UI thread."""

    status_signal = pyqtSignal(str)
    done_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)

    def __init__(self, source: Path, owner: str, other: str, summary: bool):
        super().__init__()
        self.source = source
        self.owner = owner
        self.other = other
        self.summary = summary

    def run(self) -> None:
        from recording_transcript import transcribe_recording

        try:
            result = transcribe_recording(
                self.source,
                user_name=self.owner or None,
                other_name=self.other or None,
                generate_summary=self.summary,
                status_callback=self.status_signal.emit,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the operator
            self.error_signal.emit(str(exc) or exc.__class__.__name__)
            return
        self.done_signal.emit(result)


class TranscribeWindow(BaseWindow):
    """Pick a recording, transcribe it, and open the result."""

    def __init__(self):
        super().__init__("Transcribe recording", 620, 470)
        set_app_user_model_id(TRANSCRIBE_APP_ID)
        apply_window_icon(self, TRANSCRIBE_ICON, app_id=TRANSCRIBE_APP_ID)

        self.source_path: Path | None = None
        self.worker: TranscribeWorker | None = None
        self.result = None

        self.setStyleSheet(theme.application_stylesheet() + self._stylesheet())

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(14)
        self.main_layout.addWidget(content)

        heading = QLabel("Transcribe recording")
        heading.setObjectName("transcribeHeading")
        layout.addWidget(heading)

        subtitle = QLabel(
            "Convert any audio file into a transcript document. When the "
            "recording keeps each party on its own channel, speakers are read "
            "from the channel split rather than guessed."
        )
        subtitle.setObjectName("helpText")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addWidget(self._section("Recording"))
        picker = QHBoxLayout()
        picker.setSpacing(10)
        self.file_label = QLabel("No file selected")
        self.file_label.setObjectName("fileValue")
        self.file_label.setWordWrap(True)
        picker.addWidget(self.file_label, 1)
        self.browse_button = QPushButton("Choose file...")
        self.browse_button.setObjectName("secondaryButton")
        self.browse_button.clicked.connect(self.choose_file)
        picker.addWidget(self.browse_button)
        layout.addLayout(picker)

        self.detail_label = QLabel("")
        self.detail_label.setObjectName("helpText")
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.detail_label)

        layout.addWidget(self._section("Speakers"))
        names = QHBoxLayout()
        names.setSpacing(12)
        names.addLayout(self._field("My name", self._default_owner()))
        self.other_input = QLineEdit()
        self.other_input.setPlaceholderText("Contact name or number")
        names.addLayout(self._field("Other party", self.other_input))
        layout.addLayout(names)

        self.summary_checkbox = QCheckBox("Also generate a summary")
        self.summary_checkbox.setChecked(self._default_summary())
        layout.addWidget(self.summary_checkbox)

        self.status_label = QLabel("")
        self.status_label.setObjectName("helpText")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.warning_label = QLabel("")
        self.warning_label.setObjectName("warningText")
        self.warning_label.setWordWrap(True)
        layout.addWidget(self.warning_label)

        layout.addStretch(1)

        actions = QHBoxLayout()
        actions.setSpacing(10)
        self.folder_button = QPushButton("Open folder")
        self.folder_button.setObjectName("secondaryButton")
        self.folder_button.clicked.connect(self.open_folder)
        self.folder_button.setEnabled(False)
        actions.addWidget(self.folder_button)

        self.transcript_button = QPushButton("Open transcript")
        self.transcript_button.setObjectName("secondaryButton")
        self.transcript_button.clicked.connect(self.open_transcript)
        self.transcript_button.setEnabled(False)
        actions.addWidget(self.transcript_button)

        actions.addStretch(1)
        self.start_button = QPushButton("Transcribe")
        self.start_button.setObjectName("primaryButton")
        self.start_button.setDefault(True)
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start)
        actions.addWidget(self.start_button)
        layout.addLayout(actions)

    # ---------- construction helpers ----------

    def _stylesheet(self) -> str:
        return f"""
        QLabel#transcribeHeading {{
            color: {theme.TEXT_COLOR};
            font-size: 17pt;
            font-weight: 700;
        }}
        QLabel#helpText {{ color: {theme.SECONDARY_TEXT}; font-size: 9.5pt; }}
        QLabel#fileValue {{
            color: {theme.TEXT_COLOR};
            background: {theme.INPUT_BG};
            border: 1px solid {theme.BORDER_COLOR};
            border-radius: 6px;
            padding: 8px 10px;
            font-size: 9.5pt;
        }}
        QLabel#warningText {{ color: {theme.RECORDING_COLOR}; font-size: 9pt; }}
        QLabel#sectionLabel {{
            color: {theme.DIM_TEXT};
            font-size: 8.5pt;
            font-weight: 600;
        }}
        QLineEdit {{
            background: {theme.INPUT_BG};
            border: 1px solid {theme.INPUT_BORDER};
            border-radius: 6px;
            padding: 7px 9px;
            color: {theme.TEXT_COLOR};
        }}
        QLineEdit:focus {{ border-color: {theme.INPUT_FOCUS_BORDER}; }}
        QCheckBox {{ color: {theme.TEXT_COLOR}; font-size: 9.5pt; }}
        QPushButton#primaryButton {{
            background: {theme.ACCENT_COLOR};
            color: #0B0F17;
            border: none;
            border-radius: 6px;
            padding: 9px 22px;
            font-weight: 700;
        }}
        QPushButton#primaryButton:disabled {{
            background: {theme.SURFACE_ELEVATED};
            color: {theme.DIM_TEXT};
        }}
        QPushButton#secondaryButton {{
            background: {theme.BUTTON_BG};
            color: {theme.TEXT_COLOR};
            border: 1px solid {theme.BUTTON_BORDER};
            border-radius: 6px;
            padding: 8px 14px;
        }}
        QPushButton#secondaryButton:hover:enabled {{
            background: {theme.BUTTON_HOVER_BG};
        }}
        QPushButton#secondaryButton:disabled {{ color: {theme.DIM_TEXT}; }}
        """

    def _section(self, text: str) -> QWidget:
        frame = QFrame()
        frame.setObjectName("sectionFrame")
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(0, 6, 0, 0)
        outer.setSpacing(0)
        label = QLabel(text.upper())
        label.setObjectName("sectionLabel")
        outer.addWidget(label)
        return frame

    def _field(self, label_text: str, widget) -> QVBoxLayout:
        box = QVBoxLayout()
        box.setSpacing(4)
        label = QLabel(label_text)
        label.setObjectName("helpText")
        box.addWidget(label)
        if isinstance(widget, str):
            widget = QLineEdit(widget)
            self.owner_input = widget
        box.addWidget(widget)
        return box

    @staticmethod
    def _default_owner() -> str:
        return str(ConfigManager.get_config_value("profile", "user_name") or "")

    @staticmethod
    def _default_summary() -> bool:
        return bool(
            ConfigManager.get_config_value("transcript_options", "default_summary")
        )

    # ---------- actions ----------

    def choose_file(self) -> None:
        start_dir = str(Path.home() / "Downloads")
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Choose a recording", start_dir, AUDIO_FILTER
        )
        if not chosen:
            return
        self.select_file(Path(chosen))

    def select_file(self, path: Path) -> None:
        from recording_transcript import parse_recording_filename, probe_audio

        self.source_path = path
        self.file_label.setText(path.name)
        self.warning_label.setText("")
        self.result = None
        self.folder_button.setEnabled(False)
        self.transcript_button.setEnabled(False)

        try:
            details = probe_audio(path)
        except Exception as exc:  # noqa: BLE001 - shown to the operator
            self.detail_label.setText("")
            self.status_label.setText("")
            self.warning_label.setText(str(exc))
            self.start_button.setEnabled(False)
            return

        minutes, seconds = divmod(int(round(details["duration"])), 60)
        self.detail_label.setText(
            f"{minutes}:{seconds:02d} · {details['codec'] or 'audio'} · "
            f"{details['sample_rate']} Hz · {details['channels']} channel"
            f"{'s' if details['channels'] != 1 else ''}"
        )

        parsed = parse_recording_filename(path)
        if parsed.other_name or parsed.phone_number:
            self.other_input.setText(parsed.other_name or parsed.phone_number)
        if parsed.direction:
            self.detail_label.setText(
                self.detail_label.text() + f" · {parsed.direction}going"
            )

        self.start_button.setEnabled(True)
        self.status_label.setText("Ready.")

    def start(self) -> None:
        if self.source_path is None or self.worker is not None:
            return
        self.start_button.setEnabled(False)
        self.browse_button.setEnabled(False)
        self.warning_label.setText("")
        self.status_label.setText("Starting...")

        self.worker = TranscribeWorker(
            self.source_path,
            self.owner_input.text().strip(),
            self.other_input.text().strip(),
            self.summary_checkbox.isChecked(),
        )
        self.worker.status_signal.connect(self.status_label.setText)
        self.worker.done_signal.connect(self.finished)
        self.worker.error_signal.connect(self.failed)
        self.worker.finished.connect(self._clear_worker)
        self.worker.start()

    def _clear_worker(self) -> None:
        self.worker = None
        self.browse_button.setEnabled(True)

    def finished(self, result) -> None:
        self.result = result
        attribution = {
            "channel": "Speakers read from the channel split.",
            "diarization": "Speakers came from diarization.",
            "existing": "An existing transcript was reused.",
        }.get(result.attribution, "")
        self.status_label.setText(f"Done. {attribution}".strip())
        if result.warnings:
            self.warning_label.setText(" ".join(result.warnings))
        self.start_button.setText("Transcribe again")
        self.start_button.setEnabled(True)
        self.folder_button.setEnabled(True)
        self.transcript_button.setEnabled(bool(result.transcript_pdf or result.transcript_md))

    def failed(self, message: str) -> None:
        self.status_label.setText("")
        self.warning_label.setText(message)
        self.start_button.setText("Transcribe")
        self.start_button.setEnabled(True)

    def open_folder(self) -> None:
        if self.result is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.result.output_dir)))

    def open_transcript(self) -> None:
        if self.result is None:
            return
        target = self.result.transcript_pdf or self.result.transcript_md
        if target is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def closeEvent(self, event) -> None:
        if self.worker is not None:
            self.status_label.setText("Finishing the current transcription first...")
            event.ignore()
            return
        lock = getattr(self, "_instance_lock", None)
        if lock is not None:
            try:
                lock.close()
            except OSError:
                pass
        super().closeEvent(event)


def main() -> int:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    lock = acquire_transcribe_lock()
    if lock is None:
        return 0

    if ConfigManager._instance is None:
        ConfigManager.initialize()

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    window = TranscribeWindow()
    window._instance_lock = lock
    window.show()
    return app.exec_()
