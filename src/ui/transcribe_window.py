"""Desktop window for transcribing an existing audio file.

Thin shell over ``recording_transcript.transcribe_recording``: pick a file,
name the other party, then watch the worker. Summaries are a Settings decision,
so this window does not ask again. All the real work stays in the headless
module so a pipeline can skip this window.
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
from meeting.transcript import format_timestamp  # noqa: E402
from paths import resource_path  # noqa: E402
from ui import theme  # noqa: E402
from ui.base_window import BaseWindow  # noqa: E402
from utils import ConfigManager  # noqa: E402

TRANSCRIBE_APP_ID = "Koe.Transcribe.App"
TRANSCRIBE_ICON = resource_path("assets", "koe-icon.ico")
_INSTANCE_PORT = 9879
_WINDOW_WIDTH = 620
AUDIO_FILTER = (
    "Audio and video (*.ogg *.opus *.m4a *.mp3 *.wav *.flac *.aac *.amr "
    "*.wma *.mp4 *.webm *.3gp *.caf *.aiff *.mkv);;All files (*)"
)


def _short_detail(message: str, limit: int = 220) -> str:
    """Keep operator-facing detail to a readable line or two."""
    text = " ".join(str(message).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


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

    def __init__(self, source: Path, other: str, summary: bool):
        super().__init__()
        self.source = source
        self.other = other
        self.summary = summary

    def run(self) -> None:
        from recording_transcript import transcribe_recording

        try:
            result = transcribe_recording(
                self.source,
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

    IDLE = "idle"
    READY = "ready"
    BUSY = "busy"
    DONE = "done"

    def __init__(self):
        super().__init__("Transcribe recording", _WINDOW_WIDTH, 470)
        set_app_user_model_id(TRANSCRIBE_APP_ID)
        apply_window_icon(self, TRANSCRIBE_ICON, app_id=TRANSCRIBE_APP_ID)
        self.setStyleSheet(theme.application_stylesheet() + self._stylesheet())

        self.source_path: Path | None = None
        self.worker: TranscribeWorker | None = None
        self.result = None
        self._instance_lock = None

        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(32, 26, 32, 12)
        header_layout.setSpacing(0)
        header_layout.addWidget(_label("Transcribe recording", "windowTitle"))
        self.main_layout.addWidget(header)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(32, 0, 32, 18)
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
        self.duration_label = _label("", "fieldLabel")
        self.duration_label.hide()
        recording.layout().addWidget(self.duration_label)
        layout.addWidget(recording)
        layout.addWidget(self._divider())

        speakers = self._section("Other party")
        self.other_input = QLineEdit()
        self.other_input.setPlaceholderText("Contact name or number")
        speakers.layout().addWidget(self.other_input)
        layout.addWidget(speakers)
        layout.addWidget(self._divider())

        self.status_card = QFrame()
        self.status_card.setObjectName("statusCard")
        self.status_card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        card_layout = QVBoxLayout(self.status_card)
        card_layout.setContentsMargins(14, 11, 14, 12)
        card_layout.setSpacing(3)
        self.status_title = _label("", "statusTitle")
        self.status_title.setWordWrap(True)
        card_layout.addWidget(self.status_title)
        self.status_detail = _label("", "statusDetail")
        self.status_detail.setWordWrap(True)
        card_layout.addWidget(self.status_detail)
        self.status_card.hide()
        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 16, 0, 0)
        wrapper_layout.addWidget(self.status_card)
        layout.addWidget(wrapper)

        self.scroll.setWidget(self.content)
        self.main_layout.addWidget(self.scroll, 1)

        footer = QFrame()
        footer.setObjectName("windowFooter")
        actions = QHBoxLayout(footer)
        actions.setContentsMargins(32, 14, 32, 18)
        actions.setSpacing(8)

        self.transcribe_button = QPushButton("Transcribe")
        self.transcribe_button.setObjectName("primaryButton")
        self.transcribe_button.setDefault(True)
        self.transcribe_button.clicked.connect(self.start)
        actions.addWidget(self.transcribe_button)

        actions.addStretch(1)

        self.folder_button = QPushButton("Open folder")
        self.folder_button.clicked.connect(self.open_folder)
        actions.addWidget(self.folder_button)

        self.transcript_button = QPushButton("Open transcript")
        self.transcript_button.setObjectName("primaryButton")
        self.transcript_button.clicked.connect(self.open_transcript)
        actions.addWidget(self.transcript_button)

        self.main_layout.addWidget(footer)
        self._set_state(self.IDLE)

    # ---------- styling ----------

    def _stylesheet(self) -> str:
        return f"""
        QLabel#fieldLabel {{ color: {theme.DIM_TEXT}; font-size: 9pt; }}
        QFrame#windowFooter {{ border-top: 1px solid {theme.DIVIDER_COLOR}; }}
        QFrame#statusCard {{ border-radius: 8px; }}
        QLabel#statusTitle {{ font-size: 9.5pt; font-weight: 650; }}
        QLabel#statusDetail {{ font-size: 9pt; }}
        QPushButton#primaryButton:disabled {{
            background-color: {theme.SURFACE_ELEVATED};
            color: {theme.DIM_TEXT};
            border-color: {theme.DIVIDER_COLOR};
        }}
        QPushButton#primaryButton:disabled:hover {{
            background-color: {theme.SURFACE_ELEVATED};
        }}
        """

    @staticmethod
    def _section(title: str) -> QFrame:
        section = QFrame()
        section.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 18, 0, 17)
        layout.setSpacing(10)
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
    def _summary_enabled() -> bool:
        """Summary is a Settings decision, not a per-run one.

        Read at run time so toggling it in Settings takes effect on the next
        transcription without reopening this window.
        """
        return bool(
            ConfigManager.get_config_value("transcript_options", "default_summary")
        )

    # ---------- state ----------

    def _set_state(self, state: str) -> None:
        """One eager, honest footer: the next useful action, nothing else."""
        self.state = state
        self.transcribe_button.show()
        self.folder_button.hide()
        self.transcript_button.hide()
        self.browse_button.setEnabled(state != self.BUSY)

        if state == self.IDLE:
            self.transcribe_button.setText("Transcribe")
            self.transcribe_button.setEnabled(False)
        elif state == self.READY:
            self.transcribe_button.setText("Transcribe")
            self.transcribe_button.setEnabled(True)
        elif state == self.BUSY:
            self.transcribe_button.setText("Transcribing…")
            self.transcribe_button.setEnabled(False)
        else:  # DONE
            self.transcribe_button.hide()
            self.folder_button.show()
            self.transcript_button.show()

    def _show_status(self, kind: str, title: str, detail: str = "") -> None:
        palette = {
            "working": (theme.SURFACE_ELEVATED, theme.BORDER_COLOR,
                        theme.TEXT_COLOR, theme.SECONDARY_TEXT),
            "clear": (theme.SURFACE_ELEVATED, theme.BORDER_COLOR,
                      theme.TEXT_COLOR, theme.SECONDARY_TEXT),
            "error": (theme.ERROR_SOFT, theme.ERROR_COLOR,
                      theme.ERROR_COLOR, theme.SECONDARY_TEXT),
            "done": (theme.SUCCESS_SOFT, theme.SUCCESS_COLOR,
                     theme.SUCCESS_COLOR, theme.SECONDARY_TEXT),
        }
        background, border, title_colour, detail_colour = palette[kind]
        self.status_card.setStyleSheet(
            f"QFrame#statusCard {{"
            f" background-color: {background};"
            f" border: 1px solid {border};"
            f" border-radius: 8px; }}"
        )
        self.status_title.setStyleSheet(f"color: {title_colour};")
        self.status_detail.setStyleSheet(f"color: {detail_colour};")
        self.status_title.setText(title)
        self.status_detail.setText(detail)
        self.status_detail.setVisible(bool(detail))
        was_visible = self.status_card.isVisible()
        self.status_card.show()
        if not was_visible:
            self._fit_window_to_content()

    def _clear_status(self) -> None:
        was_visible = self.status_card.isVisible()
        self.status_card.hide()
        if was_visible:
            self._fit_window_to_content()

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
        self.result = None
        self._clear_status()
        self.duration_label.hide()

        try:
            details = probe_audio(path)
        except Exception as exc:  # noqa: BLE001 - shown to the operator
            self._set_state(self.IDLE)
            self._show_status(
                "error", "This file can't be read", _short_detail(exc)
            )
            return

        self.duration_label.setText(format_timestamp(details["duration"]))
        self.duration_label.show()
        parsed = parse_recording_filename(path)
        if parsed.other_name or parsed.phone_number:
            self.other_input.setText(parsed.other_name or parsed.phone_number)

        self._set_state(self.READY)
        self._fit_window_to_content()

    def start(self) -> None:
        if self.source_path is None or self.worker is not None:
            return
        self._set_state(self.BUSY)
        self._show_status("working", "Transcribing…", "This can take a moment.")

        self.worker = TranscribeWorker(
            self.source_path,
            self.other_input.text().strip(),
            self._summary_enabled(),
        )
        self.worker.status_signal.connect(self._on_progress)
        self.worker.done_signal.connect(self.finished)
        self.worker.error_signal.connect(self.failed)
        self.worker.finished.connect(self._clear_worker)
        self.worker.start()

    def _on_progress(self, message: str) -> None:
        self._show_status("working", "Transcribing…", message)

    def _clear_worker(self) -> None:
        self.worker = None

    def finished(self, result) -> None:
        self.result = result
        self._set_state(self.DONE)
        summary = {
            "channel": "Speakers taken from the channel split.",
            "diarization": "Speakers estimated, because the channels matched.",
            "existing": "That recording was already transcribed.",
        }.get(result.attribution, "")
        if result.reused:
            title = "Already transcribed"
        else:
            title = "Transcript ready"
        detail = " ".join(filter(None, [summary, *result.warnings]))
        self._show_status("done", title, detail)

    def failed(self, message: str) -> None:
        self._set_state(self.READY if self.source_path else self.IDLE)
        self._show_status("error", "Transcription failed", _short_detail(message))

    def open_folder(self) -> None:
        if self.result is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.result.output_dir)))

    def open_transcript(self) -> None:
        if self.result is None:
            return
        target = self.result.transcript_pdf or self.result.transcript_md
        if target is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    # ---------- window ----------

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._fit_window_to_content()

    def _fit_window_to_content(self) -> None:
        """Size to the content so nothing is ever scrolled out of sight."""
        self.content.adjustSize()
        desired = self.content.sizeHint().height() + 168
        screen = (
            QApplication.screenAt(self.frameGeometry().center())
            or QApplication.primaryScreen()
        )
        ceiling = screen.availableGeometry().height() - 48 if screen else 900
        self.setFixedHeight(max(430, min(desired, ceiling)))

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape and self.worker is None:
            self.close()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        if self.worker is not None:
            self._show_status(
                "working", "Finishing…", "The window closes when this is done."
            )
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
