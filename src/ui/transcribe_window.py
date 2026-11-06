"""Desktop window for transcribing an existing audio file.

Thin shell over ``recording_transcript.transcribe_recording``: pick a file,
confirm the names, optionally ask for a summary, then watch the worker. All the
real work stays in the headless module so a pipeline can skip this window.
"""

import os
import socket
import sys
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compat import apply_window_icon, set_app_user_model_id  # noqa: E402
from paths import resource_path  # noqa: E402
from ui import theme  # noqa: E402
from meeting.transcript import format_timestamp  # noqa: E402
from ui.base_window import BaseWindow  # noqa: E402
from ui.settings_window import ToggleRow  # noqa: E402
from utils import ConfigManager  # noqa: E402

TRANSCRIBE_APP_ID = "Koe.Transcribe.App"
TRANSCRIBE_ICON = resource_path("assets", "koe-icon.ico")
_INSTANCE_PORT = 9879
_WINDOW_WIDTH = 640
AUDIO_FILTER = (
    "Audio and video (*.ogg *.opus *.m4a *.mp3 *.wav *.flac *.aac *.amr "
    "*.wma *.mp4 *.webm *.3gp *.caf *.aiff *.mkv);;All files (*)"
)


def _label(text: str, object_name: str | None = None) -> QLabel:
    label = QLabel(text)
    if object_name:
        label.setObjectName(object_name)
    return label


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
        super().__init__("Transcribe recording", _WINDOW_WIDTH, 620)
        set_app_user_model_id(TRANSCRIBE_APP_ID)
        apply_window_icon(self, TRANSCRIBE_ICON, app_id=TRANSCRIBE_APP_ID)

        self.source_path: Path | None = None
        self.worker: TranscribeWorker | None = None
        self.result = None
        self._instance_lock = None

        self.setStyleSheet(theme.application_stylesheet())

        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(32, 26, 32, 14)
        header_layout.setSpacing(2)
        header_layout.addWidget(_label("Transcribe recording", "windowTitle"))
        self.main_layout.addWidget(header)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(32, 0, 32, 24)
        self.content_layout.setSpacing(0)
        self.content_layout.setAlignment(Qt.AlignTop)

        layout = self.content_layout

        recording = self._section("Recording")
        row = QHBoxLayout()
        row.setSpacing(8)
        self.file_input = QLineEdit()
        self.file_input.setReadOnly(True)
        self.file_input.setPlaceholderText("No file selected")
        row.addWidget(self.file_input, 1)
        self.browse_button = QPushButton("Choose file")
        self.browse_button.clicked.connect(self.choose_file)
        row.addWidget(self.browse_button)
        recording.layout().addLayout(row)
        self.detail_label = _label("", "fieldLabel")
        recording.layout().addWidget(self.detail_label)
        layout.addWidget(recording)
        layout.addWidget(self._divider())

        speakers = self._section("Speakers")
        speakers.layout().addWidget(_label("My name", "fieldLabel"))
        self.owner_input = QLineEdit(self._default_owner())
        self.owner_input.setPlaceholderText("Your name")
        speakers.layout().addWidget(self.owner_input)
        speakers.layout().addWidget(_label("Other party", "fieldLabel"))
        self.other_input = QLineEdit()
        self.other_input.setPlaceholderText("Contact name or number")
        speakers.layout().addWidget(self.other_input)
        layout.addWidget(speakers)
        layout.addWidget(self._divider())

        summary = self._section("Summary")
        self.summary_checkbox = ToggleRow("Also generate a summary")
        self.summary_checkbox.setChecked(self._default_summary())
        summary.layout().addWidget(self.summary_checkbox)
        layout.addWidget(summary)
        layout.addWidget(self._divider())

        status = self._section("Result")
        self.status_label = _label("Choose a recording to begin.", "fieldLabel")
        self.status_label.setWordWrap(True)
        status.layout().addWidget(self.status_label)
        self.warning_label = _label("", "warningText")
        self.warning_label.setWordWrap(True)
        self.warning_label.hide()
        status.layout().addWidget(self.warning_label)
        layout.addWidget(status)

        self.scroll.setWidget(self.content)
        self.main_layout.addWidget(self.scroll, 1)

        footer = QFrame()
        footer.setObjectName("transcribeFooter")
        actions = QHBoxLayout(footer)
        actions.setContentsMargins(32, 12, 32, 18)
        actions.setSpacing(8)
        self.folder_button = QPushButton("Open folder")
        self.folder_button.clicked.connect(self.open_folder)
        self.folder_button.setEnabled(False)
        actions.addWidget(self.folder_button)
        self.transcript_button = QPushButton("Open transcript")
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
        self.main_layout.addWidget(footer)

        self.setStyleSheet(
            theme.application_stylesheet()
            + f"""
            QLabel#warningText {{
                color: {theme.RECORDING_COLOR};
                font-size: 9pt;
            }}
            QFrame#transcribeFooter {{
                border-top: 1px solid {theme.DIVIDER_COLOR};
            }}
            """
        )

    # ---------- construction helpers ----------

    @staticmethod
    def _section(title: str) -> QFrame:
        section = QFrame()
        section.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 18, 0, 17)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignTop)
        layout.addWidget(_label(title, "sectionTitle"))
        return section

    @staticmethod
    def _divider() -> QFrame:
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"color: {theme.DIVIDER_COLOR};")
        return divider

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
        if chosen:
            self.select_file(Path(chosen))

    def select_file(self, path: Path) -> None:
        from recording_transcript import parse_recording_filename, probe_audio

        self.source_path = path
        self.file_input.setText(path.name)
        self.file_input.setToolTip(str(path))
        self.warning_label.hide()
        self.warning_label.setText("")
        self.result = None
        self.folder_button.setEnabled(False)
        self.transcript_button.setEnabled(False)

        try:
            details = probe_audio(path)
        except Exception as exc:  # noqa: BLE001 - shown to the operator
            self.detail_label.setText("")
            self.status_label.setText("This file could not be read.")
            self.warning_label.setText(str(exc))
            self.warning_label.show()
            self.start_button.setEnabled(False)
            return

        self.detail_label.setText(format_timestamp(details["duration"]))
        parsed = parse_recording_filename(path)
        if parsed.other_name or parsed.phone_number:
            self.other_input.setText(parsed.other_name or parsed.phone_number)

        self.status_label.setText("Ready to transcribe.")
        self.start_button.setEnabled(True)
        self._fit_window_to_content()

    def start(self) -> None:
        if self.source_path is None or self.worker is not None:
            return
        self.start_button.setEnabled(False)
        self.browse_button.setEnabled(False)
        self.warning_label.hide()
        self.warning_label.setText("")
        self.status_label.setText("Starting…")

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
            "channel": "Speakers taken from the channel split.",
            "diarization": "Speakers estimated, because the channels matched.",
            "existing": "Existing transcript reused.",
        }.get(result.attribution, "Done.")
        self.status_label.setText(attribution)
        if result.warnings:
            self.warning_label.setText(" ".join(result.warnings))
            self.warning_label.show()
        self.start_button.setText("Transcribe again")
        self.start_button.setEnabled(True)
        self.folder_button.setEnabled(True)
        self.transcript_button.setEnabled(
            bool(result.transcript_pdf or result.transcript_md)
        )
        self._fit_window_to_content()

    def failed(self, message: str) -> None:
        self.status_label.setText("Transcription failed.")
        self.warning_label.setText(message)
        self.warning_label.show()
        self.start_button.setText("Transcribe")
        self.start_button.setEnabled(True)
        self._fit_window_to_content()

    def open_folder(self) -> None:
        if self.result is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.result.output_dir)))

    def open_transcript(self) -> None:
        if self.result is None:
            return
        target = self.result.transcript_pdf or self.result.transcript_md
        if target is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._fit_window_to_content()

    def _fit_window_to_content(self) -> None:
        """Size to the content so the status line is never scrolled away."""
        self.content.adjustSize()
        desired = self.content.sizeHint().height() + 150
        screen = (
            QApplication.screenAt(self.frameGeometry().center())
            or QApplication.primaryScreen()
        )
        ceiling = screen.availableGeometry().height() - 48 if screen else 900
        self.setFixedHeight(max(560, min(desired, ceiling)))

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape and self.worker is None:
            self.close()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        if self.worker is not None:
            self.status_label.setText("Finishing the current transcription first…")
            event.ignore()
            return
        lock = self._instance_lock
        if lock is not None:
            try:
                lock.close()
            except OSError:
                pass
            self._instance_lock = None
        super().closeEvent(event)


def main() -> int:
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
