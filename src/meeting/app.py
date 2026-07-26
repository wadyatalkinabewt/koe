"""Scribe meeting-mode UI and single-upload transcription workflow.

Microphone and loopback are overlaid into one mono request so an hour-long
meeting is billed as one hour. The original mic track is used locally to map
the host's diarized speaker label and may be retained with loopback on request.
"""

import sys
import os
import re
import socket
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QSize, Qt, QThread, QTimer, pyqtSignal, QUrl
from PyQt5.QtGui import QColor, QDesktopServices, QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QDialog, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QPushButton, QInputDialog, QMessageBox,
    QStackedWidget, QFrame,
)

from compat import apply_window_icon, enable_dark_titlebar, set_app_user_model_id
from paths import default_meetings_dir, resource_path, scribe_temp_dir
from utils import ConfigManager
from ui import theme
from ui.theme import (
    BG_COLOR, TEXT_COLOR, SECONDARY_TEXT, DIM_TEXT,
    RECORDING_COLOR, INPUT_BG, INPUT_BORDER, INPUT_FOCUS_BORDER,
    BUTTON_BG, BUTTON_HOVER_BG, BUTTON_BORDER, LINK_COLOR,
    SELECTION_BG, SELECTION_TEXT, SCROLLBAR_BG, SCROLLBAR_HANDLE,
    SCROLLBAR_HANDLE_HOVER,
)
from meeting.capture import (
    AudioCapture,
    identify_microphone_speaker,
    prepare_mono_meeting_mix,
)


MODE_ONE_ON_ONE = "one_on_one"
MODE_GROUP = "group"
SCRIBE_APP_ID = "Koe.Scribe.App"
SCRIBE_ICON = resource_path("assets", "koe-icon.ico")


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


class MeetingModeDialog(QDialog):
    """Choose the meeting attribution mode before the Scribe window opens."""

    def __init__(self):
        super().__init__()
        self.selected_mode: Optional[str] = None
        self.setWindowTitle("Start Scribe")
        self.setWindowFlags(
            (self.windowFlags() | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
            & ~Qt.WindowContextHelpButtonHint
        )
        self.setFixedSize(440, 176)
        self.setStyleSheet(theme.application_stylesheet() + f"""
            QFrame#modeOptions {{
                background: {theme.SURFACE_COLOR};
                border: 1px solid {theme.BORDER_COLOR};
                border-radius: 10px;
            }}
            QFrame#modeSeparator {{
                background: {theme.BORDER_COLOR};
                border: none;
            }}
            QPushButton#modeOption {{
                text-align: center;
                background: transparent;
                border: none;
                border-radius: 8px;
                padding: 0;
                font-size: 10pt;
                font-weight: 600;
            }}
            QPushButton#modeOption:hover {{
                background: {theme.ACCENT_SOFT};
                color: {theme.ACCENT_HOVER};
            }}
            QLabel#modeTitle {{
                font-size: 16pt;
                font-weight: 700;
            }}
        """)
        apply_window_icon(self, SCRIBE_ICON, app_id=SCRIBE_APP_ID)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 14, 26, 14)
        layout.setSpacing(0)
        layout.addStretch(1)

        title = QLabel("How many other people are\nin this meeting?")
        title.setObjectName("modeTitle")
        title.setWordWrap(False)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addSpacing(16)

        option_row = QHBoxLayout()
        option_row.setSpacing(0)
        option_row.addStretch()

        options = QFrame()
        options.setObjectName("modeOptions")
        options.setFixedSize(225, 40)
        options_layout = QHBoxLayout(options)
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(0)

        one_on_one = QPushButton("One")
        one_on_one.setObjectName("modeOption")
        one_on_one.setCursor(Qt.PointingHandCursor)
        one_on_one.setFixedSize(112, 38)
        one_on_one.clicked.connect(lambda: self._choose(MODE_ONE_ON_ONE))
        options_layout.addWidget(one_on_one)

        separator = QFrame()
        separator.setObjectName("modeSeparator")
        separator.setFixedSize(1, 24)
        options_layout.addWidget(separator, 0, Qt.AlignVCenter)

        group = QPushButton("Multiple")
        group.setObjectName("modeOption")
        group.setCursor(Qt.PointingHandCursor)
        group.setFixedSize(112, 38)
        group.clicked.connect(lambda: self._choose(MODE_GROUP))
        options_layout.addWidget(group)
        option_row.addWidget(options)
        option_row.addStretch()
        layout.addLayout(option_row)
        layout.addStretch(1)

    def _choose(self, mode: str) -> None:
        self.selected_mode = mode
        self.accept()

    def showEvent(self, event):
        super().showEvent(event)
        enable_dark_titlebar(self)
        apply_window_icon(self, SCRIBE_ICON, app_id=SCRIBE_APP_ID)


def _unique_labels(segments: list[dict], preferred_first: str) -> list[str]:
    labels = [preferred_first] if preferred_first else []
    for segment in sorted(segments, key=lambda item: item.get("start", 0)):
        label = str(segment.get("label") or "").strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def _persist_meeting_audio(mic_wav: Path, loopback_wav: Path, meeting_dir: Path) -> None:
    """Stage and verify both source WAVs before promoting either durable file."""
    destinations = {
        mic_wav: meeting_dir / "microphone.wav",
        loopback_wav: meeting_dir / "meeting-audio.wav",
    }
    if any(destination.exists() for destination in destinations.values()):
        raise FileExistsError("Meeting audio already exists in the output folder.")

    stage_dir = Path(tempfile.mkdtemp(prefix=".audio-stage-", dir=str(meeting_dir)))
    promoted: list[Path] = []
    try:
        staged: dict[Path, Path] = {}
        for source, destination in destinations.items():
            staged_path = stage_dir / destination.name
            shutil.copy2(source, staged_path)
            if staged_path.stat().st_size != source.stat().st_size:
                raise OSError(f"Audio copy verification failed for {source.name}")
            staged[source] = staged_path

        for source, destination in destinations.items():
            os.replace(staged[source], destination)
            promoted.append(destination)

        if any(
            destination.stat().st_size != source.stat().st_size
            for source, destination in destinations.items()
        ):
            raise OSError("Final meeting audio verification failed")
    except Exception:
        for destination in promoted:
            destination.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)


def _discard_temp_audio(*sources: Path) -> None:
    """Remove an unusable no-speech attempt without touching a meeting folder."""
    for source in sources:
        source.unlink(missing_ok=True)
    temp_dir = sources[0].parent if sources else None
    if temp_dir and temp_dir.exists() and not any(temp_dir.iterdir()):
        temp_dir.rmdir()


def _relabel_mixed_segments(
    segments: list[dict],
    microphone_label: str | None,
    user_name: str,
    meeting_subject: str,
    meeting_mode: str,
) -> list[dict]:
    """Apply deterministic host attribution after one mixed diarized request."""
    relabelled = [dict(segment) for segment in segments]
    if microphone_label:
        for segment in relabelled:
            if segment.get("label") == microphone_label:
                segment["label"] = user_name

    other_labels = []
    for segment in relabelled:
        label = str(segment.get("label") or "").strip()
        if label and label != user_name and label not in other_labels:
            other_labels.append(label)

    if meeting_mode == MODE_ONE_ON_ONE and len(other_labels) == 1:
        only_other = other_labels[0]
        for segment in relabelled:
            if segment.get("label") == only_other:
                segment["label"] = meeting_subject
        return relabelled

    generic_labels = [
        label for label in other_labels if re.fullmatch(r"Speaker \d+", label, flags=re.IGNORECASE)
    ]
    generic_map = {label: f"Speaker {index}" for index, label in enumerate(generic_labels, 1)}
    for segment in relabelled:
        label = segment.get("label")
        if label in generic_map:
            segment["label"] = generic_map[label]
    return relabelled


# ---------- worker ----------

class MeetingWorker(QThread):
    """Transcribe + summarize in the background. UI updates via signals."""

    status_signal = pyqtSignal(str)
    done_signal = pyqtSignal(str, str)  # (folder_path, summary_path)
    error_signal = pyqtSignal(str)

    def __init__(self, mic_wav: Path, loopback_wav: Path,
                 user_name: str, meeting_subject: str, meeting_mode: str,
                 notes_text: str, output_root: Path,
                 started_at: datetime, save_audio: bool = False,
                 meeting_dir: Optional[Path] = None):
        super().__init__()
        self.mic_wav = mic_wav
        self.loopback_wav = loopback_wav
        self.user_name = user_name or "You"
        self.meeting_subject = meeting_subject or "Meeting"
        self.meeting_mode = meeting_mode
        self.notes_text = notes_text
        self.output_root = output_root
        self.started_at = started_at
        self.save_audio = save_audio
        self.meeting_dir = Path(meeting_dir) if meeting_dir is not None else None

    def run(self):
        try:
            from transcription import transcribe_file_segments
            from meeting.summarizer import SummarizerClient
            from meeting.transcript import render_transcript

            # ----- one mono upload for one-duration billing -----
            self.status_signal.emit("Transcribing your meeting audio...")
            mixed_wav = self.mic_wav.parent / "meeting-mix.wav"
            mic_audio, loopback_audio, mixed_rate = prepare_mono_meeting_mix(
                self.mic_wav,
                self.loopback_wav,
                mixed_wav,
            )
            all_segments = transcribe_file_segments(
                mixed_wav,
                label="Speaker",
                diarize=True,
                use_speaker_library=True,
            )
            microphone_label = identify_microphone_speaker(
                all_segments,
                mic_audio,
                loopback_audio,
                mixed_rate,
            )
            all_segments = _relabel_mixed_segments(
                all_segments,
                microphone_label,
                self.user_name,
                self.meeting_subject,
                self.meeting_mode,
            )
            if not all_segments:
                _discard_temp_audio(self.mic_wav, self.loopback_wav, mixed_wav)
                self.error_signal.emit("No speech detected in either stream.")
                return

            # ----- transcript -----
            duration = max((s["end"] for s in all_segments), default=0.0)
            participants = (
                _unique_labels(all_segments, preferred_first=self.user_name)
                if self.meeting_mode == MODE_GROUP
                else [self.user_name, self.meeting_subject]
            )
            transcript_md = render_transcript(
                segments=all_segments,
                meeting_name=self.meeting_subject,
                participants=participants,
                started_at=self.started_at,
                duration_seconds=duration,
            )

            # ----- save files -----
            meeting_dir = self.meeting_dir or _meeting_directory(
                self.output_root,
                self.meeting_subject,
                self.meeting_mode,
                self.started_at,
            )
            meeting_dir.mkdir(parents=True, exist_ok=True)
            (meeting_dir / "transcript.md").write_text(transcript_md, encoding="utf-8")

            notes_md = ""
            if self.notes_text.strip():
                notes_md = f"# Notes — {self.meeting_subject}\n\n{self.notes_text.strip()}\n"
                (meeting_dir / "notes.md").write_text(notes_md, encoding="utf-8")

            # ----- summary -----
            self.status_signal.emit("Generating summary...")
            pdf_summary = os.getenv("KOE_SUMMARY_FORMAT", "").strip().casefold() == "pdf"
            summary_path = meeting_dir / ("summary.pdf" if pdf_summary else "summary.md")
            try:
                doc = "\n\n".join(p for p in (notes_md, transcript_md) if p)
                summary_text = SummarizerClient().summarize(doc)
            except Exception as e:
                summary_text = f"# Summary\n\nSummary generation failed: {e}\n"
            if pdf_summary:
                from meeting.summary_pdf import render_summary_pdf

                render_summary_pdf(summary_text, summary_path)
            else:
                summary_path.write_text(summary_text, encoding="utf-8")

            # ----- optional durable audio, then temp cleanup -----
            if self.save_audio:
                try:
                    _persist_meeting_audio(self.mic_wav, self.loopback_wav, meeting_dir)
                except Exception as exc:
                    self.error_signal.emit(
                        f"Meeting files were written, but audio could not be saved: {exc}. "
                        f"Temporary sources were preserved in {self.mic_wav.parent}."
                    )
                    return

            mixed_wav.unlink(missing_ok=True)
            self.mic_wav.unlink(missing_ok=True)
            self.loopback_wav.unlink(missing_ok=True)
            temp_dir = self.mic_wav.parent
            if temp_dir.exists() and not any(temp_dir.iterdir()):
                temp_dir.rmdir()

            self.done_signal.emit(str(meeting_dir), str(summary_path))

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_signal.emit(f"Failed: {e}")


def _sanitize(name: str, *, underscores: bool = False) -> str:
    """Strip filesystem-unsafe characters from a participant name."""
    name = (name or "Meeting").strip()
    for ch in '<>:"/\\|?*\n\r\t':
        name = name.replace(ch, "_")
    if underscores:
        name = re.sub(r"\s+", "_", name)
        name = re.sub(r"_+", "_", name)
    name = name.strip(". ")
    return name or "Meeting"


def _meeting_directory(
    output_root: Path,
    meeting_subject: str,
    meeting_mode: str,
    started_at: datetime,
) -> Path:
    folder_component = _sanitize(
        meeting_subject,
        underscores=meeting_mode == MODE_GROUP,
    )
    folder_name = f"{started_at.strftime('%y_%m_%d')}_{folder_component}"
    return output_root / folder_name


# ---------- main window ----------

class ScribeWindow(QMainWindow):
    def __init__(self, meeting_mode: str = MODE_ONE_ON_ONE):
        super().__init__()
        if ConfigManager._instance is None:
            ConfigManager.initialize()

        self.user_name = ConfigManager.get_config_value("profile", "user_name") or "You"
        self.output_root = self._resolve_output_root()
        self.meeting_mode = meeting_mode if meeting_mode in (MODE_ONE_ON_ONE, MODE_GROUP) else MODE_ONE_ON_ONE
        self.save_audio = bool(ConfigManager.get_config_value("meeting_options", "save_audio"))

        self.setWindowTitle("Scribe")
        apply_window_icon(self, SCRIBE_ICON, app_id=SCRIBE_APP_ID)
        self.resize(760, 590)
        self.setMinimumSize(660, 500)
        self.setStyleSheet(self._stylesheet())

        self.capture: Optional[AudioCapture] = None
        self.temp_dir: Optional[Path] = None
        self.recording_started_at: Optional[datetime] = None
        self.meeting_started_at: Optional[datetime] = None
        self.meeting_dir: Optional[Path] = None
        self.worker: Optional[MeetingWorker] = None
        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.timeout.connect(self._tick_timer)
        self.recording_pulse_timer = QTimer(self)
        self.recording_pulse_timer.timeout.connect(self._pulse_recording_indicator)
        self.processing_pulse_timer = QTimer(self)
        self.processing_pulse_timer.timeout.connect(self._pulse_processing_indicator)
        self._pulse_on = True
        self._processing_pulse_on = True
        self._processing_color = theme.ACCENT_COLOR
        self.tick_seconds = 0

        self._build_ui()

    def _resolve_output_root(self) -> Path:
        configured = ConfigManager.get_config_value("meeting_options", "root_folder")
        if configured:
            return Path(configured).expanduser()
        return default_meetings_dir()

    def _meeting_directory_for_session(self, meeting_subject: str) -> Path:
        if self.meeting_started_at is None:
            self.meeting_started_at = self.recording_started_at or datetime.now()
        if self.meeting_dir is None:
            self.meeting_dir = _meeting_directory(
                self.output_root,
                meeting_subject,
                self.meeting_mode,
                self.meeting_started_at,
            )
        return self.meeting_dir

    def _stylesheet(self) -> str:
        return theme.application_stylesheet() + f"""
            QLabel#meetingFieldTitle {{
                color: {theme.ACCENT_COLOR};
                font-size: 11pt;
                font-weight: 600;
            }}
            QLabel#scribeTimer {{
                background: transparent;
                border: none;
                padding: 0;
                color: {theme.SECONDARY_TEXT};
                font-size: 11pt;
                font-weight: 700;
            }}
            QLabel#scribeTimer[recording="true"] {{
                color: {theme.TEXT_COLOR};
            }}
            QPushButton#startButton {{
                min-height: 0;
                padding: 0 10px;
                background: {theme.SURFACE_COLOR};
                border: 1px solid #315D50;
                color: #79E2BD;
            }}
            QPushButton#startButton:hover {{
                background: {theme.SUCCESS_SOFT};
                border-color: {theme.SUCCESS_COLOR};
                color: #A3F0D3;
            }}
            QPushButton#stopButton {{
                min-height: 0;
                padding: 0 10px;
                background: {theme.SURFACE_COLOR};
                border: 1px solid #67343D;
                color: #FF9CA4;
            }}
            QPushButton#stopButton:hover {{
                background: {theme.ERROR_SOFT};
                border-color: {theme.RECORDING_COLOR};
                color: #FFC0C5;
            }}
            QLabel#processingLabel {{
                background: transparent;
                border: none;
                padding: 0;
                color: {theme.SECONDARY_TEXT};
                font-weight: 600;
            }}
            QLabel#readyLabel {{
                color: {theme.SECONDARY_TEXT};
                font-size: 8pt;
                font-weight: 600;
            }}
            QLabel#retryStatus {{
                color: #FF9CA4;
                font-size: 8pt;
                font-weight: 600;
            }}
            QFrame#completionOptions {{
                background: transparent;
                border: none;
            }}
            QPushButton#summaryButton {{
                min-height: 0;
                min-width: 0;
                background: {theme.ACCENT_COLOR};
                border: 1px solid {theme.ACCENT_COLOR};
                color: #FFFFFF;
                padding: 4px 10px;
                border-radius: 8px;
                font-size: 9pt;
                font-weight: 600;
            }}
            QPushButton#summaryButton:hover {{
                background: {theme.ACCENT_HOVER};
                border-color: {theme.ACCENT_HOVER};
            }}
            QPushButton#folderButton {{
                min-height: 0;
                min-width: 0;
                background: {theme.SURFACE_COLOR};
                border: 1px solid {theme.BORDER_COLOR};
                color: {theme.SECONDARY_TEXT};
                padding: 4px 9px;
                border-radius: 8px;
                font-size: 9pt;
                font-weight: 600;
            }}
            QPushButton#folderButton:hover {{
                background: {theme.SURFACE_HOVER};
                color: {theme.TEXT_COLOR};
            }}
        """

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(28, 26, 28, 26)
        layout.setSpacing(14)

        meeting_layout = QHBoxLayout()
        meeting_layout.setContentsMargins(0, 0, 0, 0)
        meeting_layout.setSpacing(16)

        field_title = "Meeting Name" if self.meeting_mode == MODE_GROUP else "Meeting With"
        placeholder = (
            "e.g. Weekly sync or Management meeting"
            if self.meeting_mode == MODE_GROUP
            else "Add a name now or after recording"
        )
        self.meeting_field_label = QLabel(field_title)
        self.meeting_field_label.setObjectName("meetingFieldTitle")
        self.participant_input = QLineEdit()
        self.participant_input.setPlaceholderText(placeholder)
        self.participant_input.setMaximumWidth(360)

        participant_widget = QWidget()
        participant_widget.setMaximumWidth(360)
        participant_box = QVBoxLayout(participant_widget)
        participant_box.setContentsMargins(0, 0, 0, 0)
        participant_box.setSpacing(7)
        participant_box.addWidget(self.meeting_field_label)
        participant_box.addWidget(self.participant_input)
        meeting_layout.addWidget(participant_widget, 1)

        self.action_stack = QStackedWidget()
        self.action_stack.setFixedWidth(328)
        self.action_stack.setFixedHeight(38)

        self.record_controls_widget = QWidget()
        record_controls = QHBoxLayout(self.record_controls_widget)
        record_controls.setContentsMargins(0, 0, 0, 0)
        record_controls.setSpacing(8)
        self.retry_indicator = QLabel("●")
        self.retry_indicator.setFixedWidth(9)
        self.retry_indicator.setStyleSheet(
            f"color: {theme.ERROR_COLOR}; background: transparent; border: none;"
        )
        self.retry_indicator.hide()
        record_controls.addWidget(self.retry_indicator)
        self.retry_label = QLabel("No Speech Detected")
        self.retry_label.setObjectName("retryStatus")
        self.retry_label.setToolTip("No speech detected in either stream.")
        self.retry_label.hide()
        record_controls.addWidget(self.retry_label)
        record_controls.addStretch()

        self.record_button = QPushButton("Start")
        self.record_button.setObjectName("startButton")
        self.record_button.setFixedSize(72, 34)
        self.record_button.setIconSize(QSize(14, 10))
        self.record_button.clicked.connect(self._on_record_clicked)
        record_controls.addWidget(self.record_button)

        self.timer_label = QLabel("00:00")
        self.timer_label.setObjectName("scribeTimer")
        self.timer_label.setProperty("recording", False)
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setFixedSize(62, 34)
        record_controls.addWidget(self.timer_label)
        self.action_stack.addWidget(self.record_controls_widget)

        self.processing_controls_widget = QWidget()
        processing_controls = QHBoxLayout(self.processing_controls_widget)
        processing_controls.setContentsMargins(0, 0, 0, 0)
        processing_controls.setSpacing(8)
        self.processing_indicator = QLabel("●")
        self.processing_indicator.setFixedWidth(9)
        self.processing_indicator.setAlignment(Qt.AlignCenter)
        self.processing_indicator.setStyleSheet(
            f"color: {theme.ACCENT_COLOR}; background: transparent; border: none;"
        )
        processing_controls.addWidget(self.processing_indicator)
        self.processing_label = QLabel("Processing…")
        self.processing_label.setObjectName("processingLabel")
        self.processing_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.processing_label.setFixedHeight(36)
        processing_controls.addWidget(self.processing_label)
        processing_controls.addStretch()
        self.processing_timer_label = QLabel("00:00")
        self.processing_timer_label.setObjectName("scribeTimer")
        self.processing_timer_label.setProperty("recording", False)
        self.processing_timer_label.setAlignment(Qt.AlignCenter)
        self.processing_timer_label.setFixedSize(62, 34)
        processing_controls.addWidget(self.processing_timer_label)
        self.action_stack.addWidget(self.processing_controls_widget)

        self.done_controls_widget = QWidget()
        done_controls = QHBoxLayout(self.done_controls_widget)
        done_controls.setContentsMargins(0, 0, 0, 0)
        done_controls.setSpacing(8)
        self.ready_indicator = QLabel("●")
        self.ready_indicator.setFixedWidth(9)
        self.ready_indicator.setAlignment(Qt.AlignCenter)
        self.ready_indicator.setStyleSheet(
            f"color: {theme.SUCCESS_COLOR}; background: transparent; border: none;"
        )
        done_controls.addWidget(self.ready_indicator)
        self.ready_label = QLabel("Ready")
        self.ready_label.setObjectName("readyLabel")
        done_controls.addWidget(self.ready_label)
        done_controls.addStretch()

        self.completion_options = QFrame()
        self.completion_options.setObjectName("completionOptions")
        completion_layout = QHBoxLayout(self.completion_options)
        completion_layout.setContentsMargins(0, 0, 0, 0)
        completion_layout.setSpacing(8)
        self.open_summary_button = QPushButton("Summary")
        self.open_summary_button.setObjectName("summaryButton")
        self.open_summary_button.setAccessibleName("Open summary")
        self.open_summary_button.setCursor(Qt.PointingHandCursor)
        self.open_summary_button.setToolTip("Open summary")
        self.open_summary_button.setFixedHeight(28)
        self.open_summary_button.clicked.connect(self._open_summary)
        completion_layout.addWidget(self.open_summary_button)
        self.open_folder_button = QPushButton("Folder")
        self.open_folder_button.setObjectName("folderButton")
        self.open_folder_button.setAccessibleName("Open folder")
        self.open_folder_button.setCursor(Qt.PointingHandCursor)
        self.open_folder_button.setToolTip("Open folder")
        self.open_folder_button.setFixedHeight(28)
        self.open_folder_button.clicked.connect(self._open_folder)
        completion_layout.addWidget(self.open_folder_button)
        done_controls.addWidget(self.completion_options)
        self.action_stack.addWidget(self.done_controls_widget)

        meeting_layout.addWidget(self.action_stack, 0, Qt.AlignBottom)
        layout.addLayout(meeting_layout)

        notes_lbl = QLabel("Meeting Notes")
        notes_lbl.setObjectName("sectionTitle")
        layout.addWidget(notes_lbl)

        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText("Add context, decisions, or follow-ups while you talk…")
        layout.addWidget(self.notes_edit, stretch=1)

        self._done_folder_path = ""
        self._done_summary_path = ""

    # ---------- recording control ----------

    def _on_record_clicked(self):
        if self.capture and self.capture.is_recording():
            self._stop_recording()
        else:
            self._start_recording()

    def _set_record_button_state(self, *, recording: bool) -> None:
        self.record_button.setText("Stop" if recording else "Start")
        self.record_button.setIcon(self._recording_dot_icon(active=True) if recording else QIcon())
        self.record_button.setObjectName("stopButton" if recording else "startButton")
        self.record_button.style().unpolish(self.record_button)
        self.record_button.style().polish(self.record_button)
        self.record_button.update()

        self._pulse_on = True
        if recording:
            self.recording_pulse_timer.start(650)
        else:
            self.recording_pulse_timer.stop()

    def _fit_completion_actions(self) -> None:
        for button, horizontal_room in (
            (self.open_summary_button, 26),
            (self.open_folder_button, 24),
        ):
            button.ensurePolished()
            text_width = button.fontMetrics().horizontalAdvance(button.text())
            button.setFixedWidth(text_width + horizontal_room)
        self.completion_options.setFixedSize(
            self.open_summary_button.width() + self.open_folder_button.width() + 8,
            30,
        )

    def _clear_retry_status(self) -> None:
        self.retry_indicator.hide()
        self.retry_label.hide()

    @staticmethod
    def _recording_dot_icon(*, active: bool) -> QIcon:
        pixmap = QPixmap(14, 10)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        color = QColor(theme.RECORDING_COLOR if active else "#74343C")
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(2, 2, 6, 6)
        painter.end()
        return QIcon(pixmap)

    def _pulse_recording_indicator(self) -> None:
        self._pulse_on = not self._pulse_on
        self.record_button.setIcon(self._recording_dot_icon(active=self._pulse_on))

    def _pulse_processing_indicator(self) -> None:
        self._processing_pulse_on = not self._processing_pulse_on
        color = self._processing_color if self._processing_pulse_on else theme.DIM_TEXT
        self.processing_indicator.setStyleSheet(
            f"color: {color}; background: transparent; border: none;"
        )

    def _set_processing_state(self, text: str, color: str, *, pulse: bool) -> None:
        self._processing_color = color
        self._processing_pulse_on = True
        self.processing_indicator.setStyleSheet(
            f"color: {color}; background: transparent; border: none;"
        )
        self.processing_label.setText(text)
        if pulse:
            self.processing_pulse_timer.start(700)
        else:
            self.processing_pulse_timer.stop()

    def _set_timer_recording(self, recording: bool) -> None:
        self.timer_label.setProperty("recording", recording)
        self.timer_label.style().unpolish(self.timer_label)
        self.timer_label.style().polish(self.timer_label)
        self.timer_label.update()

    def _start_recording(self):
        if self.worker and self.worker.isRunning():
            return
        self._clear_retry_status()
        temp_root = scribe_temp_dir()
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
        if self.meeting_started_at is None:
            self.meeting_started_at = self.recording_started_at
        self.tick_seconds = 0
        self.timer_label.setText("00:00")
        self._set_timer_recording(True)
        self.elapsed_timer.start(1000)
        self._set_record_button_state(recording=True)

    def _stop_recording(self):
        if not self.capture:
            return
        self.elapsed_timer.stop()

        self.capture.stop()
        mic_wav = self.capture.mic_path
        loopback_wav = self.capture.loopback_path
        self.capture.cleanup()
        self.capture = None
        self._set_record_button_state(recording=False)
        self._set_timer_recording(False)
        self._show_processing()

        # Prompt for the mode-specific meeting subject if empty.
        meeting_subject = self.participant_input.text().strip()
        if not meeting_subject:
            meeting_subject, ok = self._prompt_for_meeting_subject()
            meeting_subject = meeting_subject.strip() if ok else ""
        meeting_subject = meeting_subject or "Meeting"
        self.participant_input.setText(meeting_subject)
        meeting_dir = self._meeting_directory_for_session(meeting_subject)

        self.worker = MeetingWorker(
            mic_wav=mic_wav,
            loopback_wav=loopback_wav,
            user_name=self.user_name,
            meeting_subject=meeting_subject,
            meeting_mode=self.meeting_mode,
            notes_text=self.notes_edit.toPlainText(),
            output_root=self.output_root,
            started_at=self.meeting_started_at or self.recording_started_at or datetime.now(),
            save_audio=self.save_audio,
            meeting_dir=meeting_dir,
        )
        self.worker.status_signal.connect(self._on_worker_status)
        self.worker.done_signal.connect(self._on_worker_done)
        self.worker.error_signal.connect(self._on_worker_error)
        self.worker.start()

    def _prompt_for_meeting_subject(self) -> tuple[str, bool]:
        group = self.meeting_mode == MODE_GROUP
        dialog = QInputDialog(self)
        dialog.setWindowFlags(
            (dialog.windowFlags() | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
            & ~Qt.WindowContextHelpButtonHint
        )
        dialog.setWindowTitle("Name this meeting" if group else "Who was this meeting with?")
        dialog.setLabelText(
            "Meeting name (used for the folder and documents):"
            if group
            else "Participant name (used for the folder and transcript labels):"
        )
        dialog.setInputMode(QInputDialog.TextInput)
        dialog.setMinimumWidth(410)
        dialog.setStyleSheet(theme.application_stylesheet())
        apply_window_icon(dialog, SCRIBE_ICON, app_id=SCRIBE_APP_ID)
        enable_dark_titlebar(dialog)
        accepted = dialog.exec_() == QDialog.Accepted
        return dialog.textValue(), accepted

    def _tick_timer(self):
        self.tick_seconds += 1
        m, s = divmod(self.tick_seconds, 60)
        h, m = divmod(m, 60)
        self.timer_label.setText(f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}")

    # ---------- view transitions ----------

    def _show_processing(self):
        self._set_record_button_state(recording=False)
        self._set_timer_recording(False)
        self.processing_timer_label.setText(self.timer_label.text())
        self.participant_input.setEnabled(False)
        self.notes_edit.setEnabled(False)
        self.processing_label.setToolTip("")
        self._set_processing_state("Preparing Transcript…", theme.ACCENT_COLOR, pulse=True)
        self.action_stack.setCurrentWidget(self.processing_controls_widget)

    def _on_worker_status(self, msg: str):
        display = {
            "Transcribing your audio...": "Transcribing Audio…",
            "Transcribing other audio...": "Transcribing Audio…",
            "Transcribing your meeting audio...": "Transcribing Audio…",
            "Generating summary...": "Generating Summary…",
        }.get(msg, msg)
        self._set_processing_state(display, theme.ACCENT_COLOR, pulse=True)

    @staticmethod
    def _concise_error(message: str) -> str:
        lowered = message.lower()
        if "no speech detected" in lowered:
            return "No Speech Detected"
        if "elevenlabs http 400" in lowered:
            return "Couldn’t Process Audio"
        first_line = next((line.strip() for line in message.splitlines() if line.strip()), "")
        if not first_line:
            return "Couldn’t Finish"
        return first_line if len(first_line) <= 34 else f"{first_line[:33].rstrip()}…"

    def _on_worker_error(self, msg: str):
        self._set_record_button_state(recording=False)
        if "no speech detected" in msg.lower():
            self.processing_pulse_timer.stop()
            self.participant_input.setEnabled(True)
            self.notes_edit.setEnabled(True)
            self.timer_label.setText("00:00")
            self._set_timer_recording(False)
            self.retry_label.setToolTip(msg)
            self.retry_indicator.show()
            self.retry_label.show()
            self.action_stack.setCurrentWidget(self.record_controls_widget)
            return
        self.processing_label.setToolTip(msg)
        self._set_processing_state(
            self._concise_error(msg), theme.ERROR_COLOR, pulse=False
        )
        self.action_stack.setCurrentWidget(self.processing_controls_widget)

    def _on_worker_done(self, folder_path: str, summary_path: str):
        self._set_record_button_state(recording=False)
        self.processing_pulse_timer.stop()
        self._done_folder_path = folder_path
        self._done_summary_path = summary_path
        self.action_stack.setCurrentWidget(self.done_controls_widget)

    def _open_summary(self):
        if self._done_summary_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._done_summary_path))

    def _open_folder(self):
        if self._done_folder_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._done_folder_path))

    # ---------- shutdown ----------

    def showEvent(self, event):
        super().showEvent(event)
        enable_dark_titlebar(self)
        apply_window_icon(self, SCRIBE_ICON, app_id=SCRIBE_APP_ID)
        self._fit_completion_actions()

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
            self._set_record_button_state(recording=False)
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

    set_app_user_model_id(SCRIBE_APP_ID)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    if SCRIBE_ICON.exists():
        app.setWindowIcon(QIcon(str(SCRIBE_ICON)))

    mode_dialog = MeetingModeDialog()
    if mode_dialog.exec_() != QDialog.Accepted or not mode_dialog.selected_mode:
        return

    window = ScribeWindow(meeting_mode=mode_dialog.selected_mode)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
