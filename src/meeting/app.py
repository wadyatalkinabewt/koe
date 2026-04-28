"""
Scribe — minimal meeting transcription UI.

Flow:
  1. Open Scribe → notes textarea + participant field + REC button + timer
  2. REC → captures mic + loopback to temp WAVs (no live transcription)
  3. STOP → window collapses to progress indicator, worker thread kicks off
  4. Worker: transcribe each stream against Groq, interleave segments,
     prompt for participant if missing, run summary, save 3 files
  5. Done indicator stays open until user clicks close
  6. Close = closes indicator only. Does NOT reopen Scribe.
"""

import sys
import socket
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal, QUrl
from PyQt5.QtGui import QIcon, QDesktopServices
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QPushButton, QInputDialog, QMessageBox,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from compat import set_app_user_model_id
from utils import ConfigManager
from ui.theme import (
    BG_COLOR, TEXT_COLOR, SECONDARY_TEXT, DIM_TEXT,
    RECORDING_COLOR, INPUT_BG, INPUT_BORDER, INPUT_FOCUS_BORDER,
    BUTTON_BG, BUTTON_HOVER_BG, BUTTON_BORDER, LINK_COLOR,
    SELECTION_BG, SELECTION_TEXT, SCROLLBAR_BG, SCROLLBAR_HANDLE,
    SCROLLBAR_HANDLE_HOVER,
)
from meeting.capture import AudioCapture, load_wav_as_int16, preprocess_loopback


# ---------- single-instance lock ----------

_INSTANCE_PORT = 9878


def acquire_scribe_lock() -> Optional[socket.socket]:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", _INSTANCE_PORT))
        sock.listen(1)
        return sock
    except OSError:
        return None


# ---------- worker ----------

class MeetingWorker(QThread):
    """Transcribe + summarize in the background. UI updates via signals."""

    status_signal = pyqtSignal(str)
    done_signal = pyqtSignal(str, str)  # (folder_path, summary_path)
    error_signal = pyqtSignal(str)

    def __init__(self, mic_wav: Path, loopback_wav: Path,
                 user_name: str, participant: str,
                 notes_text: str, output_root: Path,
                 started_at: datetime):
        super().__init__()
        self.mic_wav = mic_wav
        self.loopback_wav = loopback_wav
        self.user_name = user_name or "You"
        self.participant = participant or "Speaker"
        self.notes_text = notes_text
        self.output_root = output_root
        self.started_at = started_at

    def run(self):
        try:
            from transcription import transcribe_groq_segments
            from meeting.summarizer import SummarizerClient
            from meeting.transcript import render_transcript

            # ----- transcribe mic (already 16kHz mono) -----
            self.status_signal.emit("Transcribing your audio...")
            mic_audio, mic_sr, mic_ch = load_wav_as_int16(self.mic_wav)
            if mic_sr != 16000 or mic_ch != 1:
                mic_audio = preprocess_loopback(mic_audio, mic_sr, mic_ch, target_rate=16000)
            mic_segments = transcribe_groq_segments(mic_audio, label=self.user_name)

            # ----- transcribe loopback (preprocess to 16kHz mono first) -----
            self.status_signal.emit("Transcribing other audio...")
            lb_audio, lb_sr, lb_ch = load_wav_as_int16(self.loopback_wav)
            lb_audio_16k = preprocess_loopback(lb_audio, lb_sr, lb_ch, target_rate=16000)
            lb_segments = transcribe_groq_segments(lb_audio_16k, label=self.participant)

            all_segments = mic_segments + lb_segments
            if not all_segments:
                self.error_signal.emit("No speech detected in either stream.")
                return

            # ----- transcript -----
            duration = max((s["end"] for s in all_segments), default=0.0)
            transcript_md = render_transcript(
                segments=all_segments,
                meeting_name=self.participant,
                participants=[self.user_name, self.participant],
                started_at=self.started_at,
                duration_seconds=duration,
            )

            # ----- save files -----
            folder_name = f"{self.started_at.strftime('%y_%m_%d')}_{_sanitize(self.participant)}"
            meeting_dir = self.output_root / folder_name
            meeting_dir.mkdir(parents=True, exist_ok=True)
            (meeting_dir / "transcript.md").write_text(transcript_md, encoding="utf-8")

            notes_md = ""
            if self.notes_text.strip():
                notes_md = f"# Notes — {self.participant}\n\n{self.notes_text.strip()}\n"
                (meeting_dir / "notes.md").write_text(notes_md, encoding="utf-8")

            # ----- summary -----
            self.status_signal.emit("Generating summary...")
            summary_path = meeting_dir / "summary.md"
            try:
                doc = "\n\n".join(p for p in (notes_md, transcript_md) if p)
                summary_text = SummarizerClient().summarize(doc)
                summary_path.write_text(summary_text, encoding="utf-8")
            except Exception as e:
                summary_path.write_text(
                    f"# Summary\n\nSummary generation failed: {e}\n", encoding="utf-8"
                )

            # ----- cleanup temp WAVs -----
            try:
                self.mic_wav.unlink(missing_ok=True)
                self.loopback_wav.unlink(missing_ok=True)
                temp_dir = self.mic_wav.parent
                if temp_dir.exists() and not any(temp_dir.iterdir()):
                    temp_dir.rmdir()
            except Exception:
                pass

            self.done_signal.emit(str(meeting_dir), str(summary_path))

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_signal.emit(f"Failed: {e}")


def _sanitize(name: str) -> str:
    """Strip filesystem-unsafe characters from a participant name."""
    name = (name or "Meeting").strip()
    for ch in '<>:"/\\|?*\n\r\t':
        name = name.replace(ch, "_")
    name = name.strip(". ")
    return name or "Meeting"


# ---------- main window ----------

class ScribeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ConfigManager.initialize()

        self.user_name = ConfigManager.get_config_value("profile", "user_name") or "You"
        self.output_root = self._resolve_output_root()

        self.setWindowTitle("Scribe")
        self.resize(720, 600)
        self.setStyleSheet(self._stylesheet())

        self.capture: Optional[AudioCapture] = None
        self.temp_dir: Optional[Path] = None
        self.recording_started_at: Optional[datetime] = None
        self.worker: Optional[MeetingWorker] = None
        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.timeout.connect(self._tick_timer)
        self.tick_seconds = 0

        self._build_ui()

    def _resolve_output_root(self) -> Path:
        configured = ConfigManager.get_config_value("meeting_options", "root_folder")
        if configured:
            return Path(configured).expanduser()
        return PROJECT_ROOT / "Meetings"

    def _stylesheet(self) -> str:
        return f"""
            QMainWindow {{ background-color: {BG_COLOR}; }}
            QLabel {{ color: {TEXT_COLOR}; font-family: 'Cascadia Code', Consolas, monospace; }}
            QLineEdit, QTextEdit {{
                background-color: {INPUT_BG};
                color: {TEXT_COLOR};
                border: 1px solid {INPUT_BORDER};
                border-radius: 6px;
                padding: 8px;
                font-family: 'Cascadia Code', Consolas, monospace;
                font-size: 11pt;
                selection-background-color: {SELECTION_BG};
                selection-color: {SELECTION_TEXT};
            }}
            QLineEdit:focus, QTextEdit:focus {{ border: 1px solid {INPUT_FOCUS_BORDER}; }}
            QLineEdit:disabled, QTextEdit:disabled {{ color: {SECONDARY_TEXT}; }}
            QPushButton {{
                background-color: {BUTTON_BG};
                color: {TEXT_COLOR};
                border: 1px solid {BUTTON_BORDER};
                border-radius: 6px;
                padding: 8px 18px;
                font-family: 'Cascadia Code', Consolas, monospace;
                font-size: 11pt;
            }}
            QPushButton:hover {{ background-color: {BUTTON_HOVER_BG}; }}
            QPushButton:disabled {{ color: {DIM_TEXT}; border-color: {DIM_TEXT}; }}
            QScrollBar:vertical {{
                background: {SCROLLBAR_BG}; width: 10px; border-radius: 5px;
            }}
            QScrollBar::handle:vertical {{ background: {SCROLLBAR_HANDLE}; border-radius: 5px; }}
            QScrollBar::handle:vertical:hover {{ background: {SCROLLBAR_HANDLE_HOVER}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # Header: participant + timer
        header = QHBoxLayout()
        header.setSpacing(12)

        participant_lbl = QLabel("MEETING WITH")
        participant_lbl.setStyleSheet(f"color: {SECONDARY_TEXT}; font-size: 9pt;")
        self.participant_input = QLineEdit()
        self.participant_input.setPlaceholderText("Name (can be added later)")
        self.participant_input.setFixedWidth(280)

        participant_box = QVBoxLayout()
        participant_box.setSpacing(4)
        participant_box.addWidget(participant_lbl)
        participant_box.addWidget(self.participant_input)
        header.addLayout(participant_box)

        header.addStretch()

        self.timer_label = QLabel("00:00")
        self.timer_label.setStyleSheet(f"color: {DIM_TEXT}; font-size: 18pt;")
        self.timer_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        header.addWidget(self.timer_label)

        layout.addLayout(header)

        # Notes
        notes_lbl = QLabel("NOTES")
        notes_lbl.setStyleSheet(f"color: {SECONDARY_TEXT}; font-size: 9pt;")
        layout.addWidget(notes_lbl)

        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText("Type notes during the meeting...")
        layout.addWidget(self.notes_edit, stretch=1)

        # Bottom: REC / STOP button
        button_row = QHBoxLayout()
        button_row.addStretch()
        self.record_button = QPushButton("● REC")
        self.record_button.setStyleSheet(
            f"QPushButton {{ color: {RECORDING_COLOR}; border-color: {RECORDING_COLOR}; "
            f"font-size: 12pt; padding: 10px 24px; }}"
            f"QPushButton:hover {{ background-color: {BUTTON_HOVER_BG}; }}"
        )
        self.record_button.clicked.connect(self._on_record_clicked)
        button_row.addWidget(self.record_button)
        layout.addLayout(button_row)

        # Status (hidden until processing)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {LINK_COLOR}; font-size: 12pt; padding-top: 8px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.hide()
        layout.addWidget(self.status_label)

        # Done view (hidden until done)
        self.done_widget = QWidget()
        done_layout = QVBoxLayout(self.done_widget)
        done_layout.setContentsMargins(0, 8, 0, 0)
        done_layout.setSpacing(8)

        self.done_label = QLabel("")
        self.done_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 12pt;")
        self.done_label.setAlignment(Qt.AlignCenter)
        self.done_label.setOpenExternalLinks(False)
        self.done_label.linkActivated.connect(self._on_link_clicked)
        done_layout.addWidget(self.done_label)

        close_row = QHBoxLayout()
        close_row.addStretch()
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        close_row.addWidget(self.close_button)
        close_row.addStretch()
        done_layout.addLayout(close_row)

        self.done_widget.hide()
        layout.addWidget(self.done_widget)

    # ---------- recording control ----------

    def _on_record_clicked(self):
        if self.capture and self.capture.is_recording():
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        temp_root = PROJECT_ROOT / ".scribe_temp"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp(dir=str(temp_root), prefix="rec_"))

        try:
            self.capture = AudioCapture(self.temp_dir)
        except Exception as e:
            QMessageBox.critical(self, "Audio device error", f"Could not access audio: {e}")
            return

        if not self.capture.start():
            QMessageBox.critical(
                self, "Recording failed",
                "Could not start audio capture. Check that mic and system audio are available."
            )
            self.capture.cleanup()
            self.capture = None
            return

        self.recording_started_at = datetime.now()
        self.tick_seconds = 0
        self.timer_label.setText("00:00")
        self.timer_label.setStyleSheet(f"color: {RECORDING_COLOR}; font-size: 18pt;")
        self.elapsed_timer.start(1000)

        self.record_button.setText("■ STOP")
        self.record_button.setStyleSheet(
            f"QPushButton {{ color: {TEXT_COLOR}; border-color: {TEXT_COLOR}; "
            f"font-size: 12pt; padding: 10px 24px; }}"
            f"QPushButton:hover {{ background-color: {BUTTON_HOVER_BG}; }}"
        )

    def _stop_recording(self):
        if not self.capture:
            return
        self.elapsed_timer.stop()

        self.capture.stop()
        mic_wav = self.capture.mic_path
        loopback_wav = self.capture.loopback_path
        self.capture.cleanup()
        self.capture = None

        # Prompt for participant if empty
        participant = self.participant_input.text().strip()
        if not participant:
            participant, ok = QInputDialog.getText(
                self, "Who was this meeting with?",
                "Participant name (used for the folder + transcript labels):"
            )
            participant = participant.strip() if ok else ""
        participant = participant or "Meeting"
        self.participant_input.setText(participant)

        self._show_processing()

        self.worker = MeetingWorker(
            mic_wav=mic_wav,
            loopback_wav=loopback_wav,
            user_name=self.user_name,
            participant=participant,
            notes_text=self.notes_edit.toPlainText(),
            output_root=self.output_root,
            started_at=self.recording_started_at or datetime.now(),
        )
        self.worker.status_signal.connect(self._on_worker_status)
        self.worker.done_signal.connect(self._on_worker_done)
        self.worker.error_signal.connect(self._on_worker_error)
        self.worker.start()

    def _tick_timer(self):
        self.tick_seconds += 1
        m, s = divmod(self.tick_seconds, 60)
        h, m = divmod(m, 60)
        self.timer_label.setText(f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}")

    # ---------- view transitions ----------

    def _show_processing(self):
        self.participant_input.setEnabled(False)
        self.notes_edit.setEnabled(False)
        self.record_button.hide()
        self.status_label.setText("Processing...")
        self.status_label.show()

    def _on_worker_status(self, msg: str):
        self.status_label.setText(msg)

    def _on_worker_error(self, msg: str):
        self.status_label.hide()
        self.done_label.setText(
            f'<span style="color: {RECORDING_COLOR};">Error:</span><br>{msg}'
        )
        self.done_widget.show()

    def _on_worker_done(self, folder_path: str, summary_path: str):
        self.status_label.hide()
        folder_url = QUrl.fromLocalFile(folder_path).toString()
        summary_url = QUrl.fromLocalFile(summary_path).toString()
        self.done_label.setText(
            f'<div style="color: {TEXT_COLOR};">Done.</div><br>'
            f'<a href="{summary_url}" style="color: {LINK_COLOR};">Open summary</a>'
            f' &nbsp;·&nbsp; '
            f'<a href="{folder_url}" style="color: {LINK_COLOR};">Open folder</a>'
        )
        self.done_widget.show()

    def _on_link_clicked(self, url: str):
        QDesktopServices.openUrl(QUrl(url))

    # ---------- shutdown ----------

    def closeEvent(self, event):
        if self.capture and self.capture.is_recording():
            reply = QMessageBox.question(
                self, "Recording in progress",
                "Discard recording and exit?",
                QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply != QMessageBox.Discard:
                event.ignore()
                return
            try:
                self.capture.stop()
                self.capture.cleanup()
            except Exception:
                pass
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)

        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "Still processing",
                "Transcription / summary is still running. Close anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return

        event.accept()


# ---------- entry point ----------

def main():
    lock = acquire_scribe_lock()
    if lock is None:
        sys.exit(0)

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    set_app_user_model_id("Koe.Scribe.App")
    icon_path = PROJECT_ROOT / "assets" / "koe-icon.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    window = ScribeWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
