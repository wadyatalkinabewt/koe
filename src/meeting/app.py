"""Scribe meeting-mode UI and single-upload transcription workflow.

Microphone and loopback are overlaid into one mono diarized request so an
hour-long meeting is billed as one hour. Online meetings use the synchronized
mic track to map the host when loopback is active. In-person and speakerphone
recordings preserve generic speaker labels rather than falsely treating the
entire shared microphone as the host.
"""

import os
import re
import shutil
import socket
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import QPoint, QSize, Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QColor, QDesktopServices, QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from compat import apply_window_icon, enable_dark_titlebar, set_app_user_model_id
from meeting.capture import (
    AudioCapture,
    identify_microphone_speaker,
    prepare_mono_meeting_mix,
    wav_has_meaningful_audio,
    write_mono_wav,
)
from paths import default_meetings_dir, resource_path, scribe_temp_dir
from ui import theme
from utils import ConfigManager

MODE_ONLINE_ONE_ON_ONE = "online_one_on_one"
MODE_ONLINE_GROUP = "online_group"
MODE_IN_PERSON_ONE_ON_ONE = "in_person_one_on_one"
MODE_IN_PERSON_GROUP = "in_person_group"
MEETING_MODES = {
    MODE_ONLINE_ONE_ON_ONE,
    MODE_ONLINE_GROUP,
    MODE_IN_PERSON_ONE_ON_ONE,
    MODE_IN_PERSON_GROUP,
}
MEETING_MODE_OPTIONS = (
    ("Online — One-on-One", MODE_ONLINE_ONE_ON_ONE),
    ("Online — Group Meeting", MODE_ONLINE_GROUP),
    ("In Person — One-on-One", MODE_IN_PERSON_ONE_ON_ONE),
    ("In Person — Group Meeting", MODE_IN_PERSON_GROUP),
)
SCRIBE_APP_ID = "Koe.Scribe.App"
SCRIBE_ICON = resource_path("assets", "koe-icon.ico")
SUMMARY_READY = "ready"
SUMMARY_NOT_CONFIGURED = "not_configured"
SUMMARY_FAILED = "failed"


def _is_one_on_one(meeting_mode: str) -> bool:
    return meeting_mode in {
        MODE_ONLINE_ONE_ON_ONE,
        MODE_IN_PERSON_ONE_ON_ONE,
    }


def _is_in_person(meeting_mode: str) -> bool:
    return meeting_mode in {
        MODE_IN_PERSON_ONE_ON_ONE,
        MODE_IN_PERSON_GROUP,
    }


# ---------- single-instance lock ----------

_INSTANCE_PORT = 9878


def acquire_scribe_lock() -> socket.socket | None:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", _INSTANCE_PORT))
        sock.listen(1)
        return sock
    except OSError:
        return None


def _unique_labels(segments: list[dict], preferred_first: str) -> list[str]:
    labels = [preferred_first] if preferred_first else []
    for segment in sorted(segments, key=lambda item: item.get("start", 0)):
        label = str(segment.get("label") or "").strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def _persist_meeting_audio(
    mic_wav: Path,
    loopback_wav: Path,
    meeting_dir: Path,
    *,
    include_loopback: bool = True,
) -> None:
    """Stage and verify the selected source WAVs before promoting any file."""
    destinations = {mic_wav: meeting_dir / "microphone.wav"}
    if include_loopback:
        destinations[loopback_wav] = meeting_dir / "meeting-audio.wav"
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


def _relabel_mixed_segments(
    segments: list[dict],
    microphone_label: str | None,
    user_name: str,
) -> list[dict]:
    """Apply deterministic host attribution after one mixed diarized request."""
    relabelled = [dict(segment) for segment in segments]
    if microphone_label:
        for segment in relabelled:
            if segment.get("label") == microphone_label:
                segment["label"] = user_name
    for segment in relabelled:
        label = str(segment.get("label") or "").strip()
        if label.casefold() == user_name.casefold():
            segment["label"] = user_name

    other_labels = []
    for segment in relabelled:
        label = str(segment.get("label") or "").strip()
        if label and label != user_name and label not in other_labels:
            other_labels.append(label)

    generic_labels = [
        label
        for label in other_labels
        if re.fullmatch(r"Speaker \d+", label, flags=re.IGNORECASE)
    ]
    generic_map = {
        label: f"Speaker {index}" for index, label in enumerate(generic_labels, 1)
    }
    for segment in relabelled:
        label = segment.get("label")
        if label in generic_map:
            segment["label"] = generic_map[label]
    return relabelled


def _label_one_on_one(
    segments: list[dict],
    user_name: str,
    participant_name: str,
) -> list[dict]:
    """Use an identified owner to safely name every other one-on-one voice."""
    relabelled = [dict(segment) for segment in segments]
    labels: list[str] = []
    for segment in relabelled:
        label = str(segment.get("label") or "").strip()
        if label and label not in labels:
            labels.append(label)

    owner_labels = [
        label for label in labels if label.casefold() == user_name.casefold()
    ]
    if not owner_labels:
        return relabelled

    owner_label = owner_labels[0]
    for segment in relabelled:
        label = str(segment.get("label") or "").strip()
        if label == owner_label:
            segment["label"] = user_name
        elif label:
            segment["label"] = participant_name
    return relabelled


def _apply_contextual_speaker_mapping(
    segments: list[dict],
    mapping: dict[str, str],
) -> list[dict]:
    """Apply one validated name to every turn in a diarized speaker cluster."""
    if not mapping:
        return [dict(segment) for segment in segments]
    canonical = {source.casefold(): target for source, target in mapping.items()}
    relabelled: list[dict] = []
    for segment in segments:
        updated = dict(segment)
        label = str(updated.get("label") or "").strip()
        if label.casefold() in canonical:
            updated["label"] = canonical[label.casefold()]
        relabelled.append(updated)
    return relabelled


def _participants_for_meeting(
    segments: list[dict],
    meeting_mode: str,
    user_name: str,
    participant_name: str,
) -> list[str]:
    """Build the document participant order from the final speaker labels."""
    if meeting_mode == MODE_ONLINE_GROUP:
        return _unique_labels(segments, preferred_first=user_name)
    if meeting_mode == MODE_ONLINE_ONE_ON_ONE:
        return [user_name, participant_name]
    if meeting_mode == MODE_IN_PERSON_ONE_ON_ONE:
        owner_present = any(
            str(segment.get("label") or "").strip().casefold() == user_name.casefold()
            for segment in segments
        )
        return _unique_labels(
            segments,
            preferred_first=user_name if owner_present else "",
        )
    return _unique_labels(segments, preferred_first="")


# ---------- worker ----------


class MeetingWorker(QThread):
    """Transcribe and optionally summarize in the background."""

    status_signal = pyqtSignal(str)
    done_signal = pyqtSignal(
        str, str, str
    )  # (folder_path, summary_path, summary_status)
    error_signal = pyqtSignal(str)

    def __init__(
        self,
        mic_wav: Path,
        loopback_wav: Path,
        user_name: str,
        meeting_subject: str,
        meeting_mode: str,
        notes_text: str,
        output_root: Path,
        started_at: datetime,
        save_audio: bool = False,
        save_markdown: bool = False,
        meeting_dir: Path | None = None,
        participant_name: str = "",
    ):
        super().__init__()
        self.mic_wav = mic_wav
        self.loopback_wav = loopback_wav
        self.user_name = user_name or "You"
        self.meeting_subject = meeting_subject or "Meeting"
        self.meeting_mode = meeting_mode
        self.participant_name = participant_name or "Participant"
        self.notes_text = notes_text
        self.output_root = output_root
        self.started_at = started_at
        self.save_audio = save_audio
        self.save_markdown = save_markdown
        self.meeting_dir = Path(meeting_dir) if meeting_dir is not None else None

    def run(self):
        try:
            from meeting.summarizer import SummarizerClient
            from meeting.transcript import render_transcript
            from meeting.transcript_pdf import render_transcript_pdf
            from transcription import transcribe_file_segments

            # ----- one mono upload for one-duration billing -----
            self.status_signal.emit("Transcribing your meeting audio...")
            mixed_wav = self.mic_wav.parent / "meeting-mix.wav"
            mic_audio, loopback_audio, mixed_rate = prepare_mono_meeting_mix(
                self.mic_wav,
                self.loopback_wav,
                mixed_wav,
            )
            loopback_meaningful = wav_has_meaningful_audio(self.loopback_wav)
            if _is_in_person(self.meeting_mode) and not loopback_meaningful:
                write_mono_wav(mixed_wav, mic_audio, mixed_rate)
            all_segments = transcribe_file_segments(
                mixed_wav,
                label="Speaker",
                diarize=True,
                use_speaker_library=False,
                num_speakers=2 if _is_one_on_one(self.meeting_mode) else None,
            )
            microphone_label = None
            if not _is_in_person(self.meeting_mode) and loopback_meaningful:
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
            )
            if _is_one_on_one(self.meeting_mode):
                all_segments = _label_one_on_one(
                    all_segments,
                    self.user_name,
                    self.participant_name,
                )
            if not all_segments:
                self.error_signal.emit(
                    "No speech detected in either stream. "
                    f"Temporary audio was preserved in {self.mic_wav.parent}."
                )
                return

            # ----- transcript -----
            duration = max((s["end"] for s in all_segments), default=0.0)
            participants = _participants_for_meeting(
                all_segments,
                self.meeting_mode,
                self.user_name,
                self.participant_name,
            )
            transcript_md = render_transcript(
                segments=all_segments,
                meeting_name=self.meeting_subject,
                participants=participants,
                started_at=self.started_at,
                duration_seconds=duration,
                notes_text=self.notes_text,
            )

            # ----- reserve the output folder -----
            meeting_dir = self.meeting_dir or _meeting_directory(
                self.output_root,
                self.meeting_subject,
                self.meeting_mode,
                self.started_at,
            )
            meeting_dir.mkdir(parents=True, exist_ok=True)
            (meeting_dir / "notes.md").unlink(missing_ok=True)
            transcript_path = meeting_dir / "transcript.pdf"
            # Preserve the deterministic transcript even if the optional
            # OpenRouter post-processing step fails or is interrupted.
            render_transcript_pdf(
                segments=all_segments,
                meeting_name=self.meeting_subject,
                participants=participants,
                started_at=self.started_at,
                duration_seconds=duration,
                recorder_name=self.user_name,
                output_path=transcript_path,
                notes_text=self.notes_text,
            )

            # ----- optional contextual speaker resolution + summary -----
            summary_text = None
            summary_path = meeting_dir / "summary.pdf"
            summary_status = SUMMARY_NOT_CONFIGURED
            if os.getenv("OPENROUTER_API_KEY", "").strip():
                self.status_signal.emit("Generating summary...")
                try:
                    analysis = SummarizerClient().analyze(
                        transcript_md,
                        speaker_labels=_unique_labels(
                            all_segments, preferred_first=""
                        ),
                    )
                    summary_text = analysis.summary
                    if analysis.speaker_mapping:
                        contextual_segments = _apply_contextual_speaker_mapping(
                            all_segments,
                            analysis.speaker_mapping,
                        )
                        contextual_participants = _participants_for_meeting(
                            contextual_segments,
                            self.meeting_mode,
                            self.user_name,
                            self.participant_name,
                        )
                        contextual_transcript_md = render_transcript(
                            segments=contextual_segments,
                            meeting_name=self.meeting_subject,
                            participants=contextual_participants,
                            started_at=self.started_at,
                            duration_seconds=duration,
                            notes_text=self.notes_text,
                        )
                        contextual_path = meeting_dir / ".transcript-contextual.pdf"
                        try:
                            render_transcript_pdf(
                                segments=contextual_segments,
                                meeting_name=self.meeting_subject,
                                participants=contextual_participants,
                                started_at=self.started_at,
                                duration_seconds=duration,
                                recorder_name=self.user_name,
                                output_path=contextual_path,
                                notes_text=self.notes_text,
                            )
                            os.replace(contextual_path, transcript_path)
                        finally:
                            contextual_path.unlink(missing_ok=True)
                        all_segments = contextual_segments
                        participants = contextual_participants
                        transcript_md = contextual_transcript_md
                    summary_status = SUMMARY_READY
                except Exception as exc:
                    summary_text = None
                    summary_status = SUMMARY_FAILED
                    ConfigManager.console_print(
                        f"Scribe summary generation failed: {exc}"
                    )

            if summary_text is not None:
                from meeting.summary_pdf import render_summary_pdf

                render_summary_pdf(
                    summary_text,
                    summary_path,
                    meeting_name=self.meeting_subject,
                    participants=participants,
                    started_at=self.started_at,
                    duration_seconds=duration,
                    recorder_name=self.user_name,
                    meeting_mode=self.meeting_mode,
                    participant_name=self.participant_name,
                )

            transcript_markdown_path = meeting_dir / "transcript.md"
            summary_markdown_path = meeting_dir / "summary.md"
            if self.save_markdown:
                transcript_markdown_path.write_text(transcript_md, encoding="utf-8")
                if summary_text is not None:
                    summary_markdown_path.write_text(summary_text, encoding="utf-8")
                else:
                    summary_markdown_path.unlink(missing_ok=True)
            else:
                transcript_markdown_path.unlink(missing_ok=True)
                summary_markdown_path.unlink(missing_ok=True)

            # ----- optional durable audio, then temp cleanup -----
            if self.save_audio:
                try:
                    _persist_meeting_audio(
                        self.mic_wav,
                        self.loopback_wav,
                        meeting_dir,
                        include_loopback=not (
                            _is_in_person(self.meeting_mode) and not loopback_meaningful
                        ),
                    )
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

            self.done_signal.emit(
                str(meeting_dir),
                str(summary_path) if summary_status == SUMMARY_READY else "",
                summary_status,
            )

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
        underscores=meeting_mode != MODE_ONLINE_ONE_ON_ONE,
    )
    folder_name = f"{started_at.strftime('%y_%m_%d')}_{folder_component}"
    return output_root / folder_name


# ---------- main window ----------


class ScribeWindow(QMainWindow):
    def __init__(self, meeting_mode: str | None = None):
        super().__init__()
        if ConfigManager._instance is None:
            ConfigManager.initialize()

        self.user_name = ConfigManager.get_config_value("profile", "user_name") or "You"
        self.output_root = self._resolve_output_root()
        saved_mode = ConfigManager.get_config_value(
            "meeting_options", "last_meeting_mode"
        )
        self.meeting_mode = (
            meeting_mode
            if meeting_mode in MEETING_MODES
            else (saved_mode if saved_mode in MEETING_MODES else MODE_ONLINE_ONE_ON_ONE)
        )
        self.save_audio = bool(
            ConfigManager.get_config_value("meeting_options", "save_audio")
        )
        self.save_markdown = bool(
            ConfigManager.get_config_value("meeting_options", "save_markdown")
        )

        self.setWindowTitle("Scribe")
        apply_window_icon(self, SCRIBE_ICON, app_id=SCRIBE_APP_ID)
        self.setWindowFlags(
            (
                self.windowFlags()
                | Qt.MSWindowsFixedSizeDialogHint
                | Qt.WindowMinimizeButtonHint
                | Qt.WindowCloseButtonHint
            )
            & ~Qt.WindowMaximizeButtonHint
        )
        self.setFixedSize(760, 590)
        self.setStyleSheet(self._stylesheet())

        self.capture: AudioCapture | None = None
        self.temp_dir: Path | None = None
        self.recording_started_at: datetime | None = None
        self.meeting_started_at: datetime | None = None
        self.meeting_dir: Path | None = None
        self.worker: MeetingWorker | None = None
        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.timeout.connect(self._tick_timer)
        self.recording_pulse_timer = QTimer(self)
        self.recording_pulse_timer.timeout.connect(self._pulse_recording_indicator)
        self._pulse_on = True
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
        return (
            theme.application_stylesheet()
            + f"""
            QLabel#scribeHeading {{
                color: {theme.TEXT_COLOR};
                font-size: 17pt;
                font-weight: 700;
            }}
            QFrame#scribeDivider {{
                background: {theme.DIVIDER_COLOR};
                border: none;
                min-height: 1px;
                max-height: 1px;
            }}
            QLabel#meetingFieldTitle {{
                color: {theme.SECONDARY_TEXT};
                font-size: 9pt;
                font-weight: 650;
            }}
            QComboBox#meetingTypeSelector {{
                min-height: 18px;
                padding: 9px 34px 9px 11px;
            }}
            QComboBox#meetingTypeSelector::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 30px;
                background: transparent;
                border: none;
            }}
            QComboBox#meetingTypeSelector:hover {{
                border-color: {theme.INPUT_FOCUS_BORDER};
            }}
            QComboBox#meetingTypeSelector:disabled {{
                color: {theme.DIM_TEXT};
                border-color: {theme.DIVIDER_COLOR};
            }}
            QComboBox#meetingTypeSelector QAbstractItemView {{
                background: {theme.SURFACE_ELEVATED};
                color: {theme.TEXT_COLOR};
                border: 1px solid {theme.BORDER_COLOR};
                selection-background-color: {theme.ACCENT_SOFT};
                selection-color: {theme.TEXT_COLOR};
                padding: 4px;
                outline: none;
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
            QLabel#workingStatus {{
                min-height: 0;
                padding: 0;
                background: transparent;
                border: none;
                color: #A9B6FF;
                font-size: 9pt;
                font-weight: 650;
            }}
            QLabel#headerStateMessage {{
                background: transparent;
                border: none;
                color: {theme.ERROR_COLOR};
                font-size: 9pt;
                font-weight: 650;
            }}
            QFrame#completionOptions {{
                background: transparent;
                border: none;
            }}
            QPushButton#summaryButton,
            QPushButton#transcriptButton[primaryCompletion="true"] {{
                min-height: 0;
                min-width: 0;
                background: {theme.ACCENT_COLOR};
                border: 1px solid {theme.ACCENT_COLOR};
                color: #FFFFFF;
                padding: 0 12px;
                border-radius: 8px;
                font-size: 9pt;
                font-weight: 600;
            }}
            QPushButton#summaryButton:hover,
            QPushButton#transcriptButton[primaryCompletion="true"]:hover {{
                background: {theme.ACCENT_HOVER};
                border-color: {theme.ACCENT_HOVER};
            }}
            QPushButton#transcriptButton,
            QPushButton#recoveryButton {{
                min-height: 0;
                min-width: 0;
                background: {theme.SURFACE_COLOR};
                border: 1px solid {theme.BORDER_COLOR};
                color: {theme.SECONDARY_TEXT};
                padding: 0 12px;
                border-radius: 8px;
                font-size: 9pt;
                font-weight: 600;
            }}
            QPushButton#transcriptButton:hover,
            QPushButton#recoveryButton:hover {{
                background: {theme.SURFACE_HOVER};
                color: {theme.TEXT_COLOR};
            }}
        """
        )

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(28, 26, 28, 26)
        layout.setSpacing(14)

        meeting_layout = QGridLayout()
        meeting_layout.setContentsMargins(0, 0, 0, 0)
        meeting_layout.setHorizontalSpacing(18)
        meeting_layout.setVerticalSpacing(7)
        meeting_layout.setColumnStretch(0, 1)
        meeting_layout.setColumnStretch(1, 1)

        self.meeting_type_label = QLabel("Meeting type")
        self.meeting_type_label.setObjectName("meetingFieldTitle")
        self.meeting_type_combo = QComboBox()
        self.meeting_type_combo.setObjectName("meetingTypeSelector")
        self.meeting_type_combo.setAccessibleName("Meeting type")
        self.meeting_type_combo.setToolTip(
            "Choose how people are joining. This locks when recording starts."
        )
        self.meeting_type_combo.setMaximumWidth(360)
        for label, mode in MEETING_MODE_OPTIONS:
            self.meeting_type_combo.addItem(label, mode)
        selected_index = self.meeting_type_combo.findData(self.meeting_mode)
        self.meeting_type_combo.setCurrentIndex(max(0, selected_index))
        self.meeting_type_combo.currentIndexChanged.connect(
            self._on_meeting_type_changed
        )

        self.meeting_field_label = QLabel("Meeting name")
        self.meeting_field_label.setObjectName("meetingFieldTitle")
        self.meeting_name_input = QLineEdit()
        self.meeting_name_input.setPlaceholderText(
            "e.g. Invoice workflow or Weekly sync"
        )
        self.meeting_name_input.setMaximumWidth(360)

        self.participant_field_label = QLabel("Participant name")
        self.participant_field_label.setObjectName("meetingFieldTitle")
        self.participant_input = QLineEdit()
        self.participant_input.setPlaceholderText("Full name works best")
        self.participant_input.setMaximumWidth(360)
        meeting_layout.addWidget(self.meeting_type_label, 0, 0)
        meeting_layout.addWidget(self.meeting_type_combo, 1, 0)
        meeting_layout.addWidget(self.meeting_field_label, 0, 1)
        meeting_layout.addWidget(self.meeting_name_input, 1, 1)
        meeting_layout.addWidget(self.participant_field_label, 2, 0)
        meeting_layout.addWidget(self.participant_input, 3, 0)
        self._update_mode_fields()

        self.action_stack = QStackedWidget()
        self.action_stack.setFixedWidth(258)
        self.action_stack.setFixedHeight(38)

        self.record_controls_widget = QWidget()
        record_controls = QHBoxLayout(self.record_controls_widget)
        record_controls.setContentsMargins(0, 0, 0, 0)
        record_controls.setSpacing(8)
        record_controls.addStretch()

        self.record_button = QPushButton("Start")
        self.record_button.setObjectName("startButton")
        self.record_button.setFixedSize(72, 34)
        self.record_button.setIconSize(QSize(14, 10))
        self.record_button.clicked.connect(self._on_record_clicked)
        record_controls.addWidget(self.record_button)
        self.action_stack.addWidget(self.record_controls_widget)

        self.status_controls_widget = QWidget()
        status_controls = QHBoxLayout(self.status_controls_widget)
        status_controls.setContentsMargins(0, 0, 0, 0)
        status_controls.addStretch()
        self.working_status_label = QLabel("Preparing")
        self.working_status_label.setObjectName("workingStatus")
        self.working_status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.working_status_label.setFixedSize(112, 30)
        status_controls.addWidget(self.working_status_label)
        self.action_stack.addWidget(self.status_controls_widget)

        self.recovery_controls_widget = QWidget()
        recovery_controls = QHBoxLayout(self.recovery_controls_widget)
        recovery_controls.setContentsMargins(0, 0, 0, 0)
        recovery_controls.addStretch()
        self.recovery_button = QPushButton("Recovery Folder")
        self.recovery_button.setObjectName("recoveryButton")
        self.recovery_button.setAccessibleName("Open recovery folder")
        self.recovery_button.setToolTip("Open the folder containing temporary audio")
        self.recovery_button.setFixedHeight(34)
        self.recovery_button.clicked.connect(self._open_recovery_folder)
        recovery_controls.addWidget(self.recovery_button)
        self.action_stack.addWidget(self.recovery_controls_widget)

        self.timer_label = QLabel("00:00")
        self.timer_label.setObjectName("scribeTimer")
        self.timer_label.setProperty("recording", False)
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setFixedSize(62, 34)

        self.completion_options = QFrame()
        self.completion_options.setObjectName("completionOptions")
        completion_layout = QHBoxLayout(self.completion_options)
        completion_layout.setContentsMargins(0, 0, 0, 0)
        completion_layout.setSpacing(7)
        self.open_summary_button = QPushButton("Summary")
        self.open_summary_button.setObjectName("summaryButton")
        self.open_summary_button.setAccessibleName("Open summary")
        self.open_summary_button.setCursor(Qt.PointingHandCursor)
        self.open_summary_button.setToolTip("Open summary")
        self.open_summary_button.setFixedHeight(34)
        self.open_summary_button.clicked.connect(self._open_summary)
        completion_layout.addWidget(self.open_summary_button)
        self.open_transcript_button = QPushButton("Transcript")
        self.open_transcript_button.setObjectName("transcriptButton")
        self.open_transcript_button.setAccessibleName("Open transcript")
        self.open_transcript_button.setCursor(Qt.PointingHandCursor)
        self.open_transcript_button.setToolTip("Open transcript")
        self.open_transcript_button.setFixedHeight(34)
        self.open_transcript_button.clicked.connect(self._open_transcript)
        completion_layout.addWidget(self.open_transcript_button)
        self.completion_options.hide()

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(14)
        heading = QLabel("Scribe")
        heading.setObjectName("scribeHeading")
        header_layout.addWidget(heading)
        header_layout.addStretch()
        header_layout.addWidget(self.action_stack, 0, Qt.AlignVCenter)
        header_layout.addWidget(self.timer_label, 0, Qt.AlignVCenter)
        header_layout.addWidget(self.completion_options, 0, Qt.AlignVCenter)

        self.header_state_row = QWidget()
        self.header_state_row.setFixedHeight(28)
        header_state_layout = QHBoxLayout(self.header_state_row)
        header_state_layout.setContentsMargins(0, 8, 0, 1)
        header_state_layout.addStretch()
        self.header_state_label = QLabel("")
        self.header_state_label.setObjectName("headerStateMessage")
        self.header_state_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.header_state_label.setFixedWidth(334)
        header_state_layout.addWidget(self.header_state_label)
        self.header_state_label.hide()

        self.divider = QFrame()
        self.divider.setObjectName("scribeDivider")
        self.divider.setFrameShape(QFrame.HLine)
        self.divider.setFixedHeight(1)

        header_block = QVBoxLayout()
        header_block.setContentsMargins(0, 0, 0, 0)
        header_block.setSpacing(0)
        header_block.addLayout(header_layout)
        header_block.addWidget(self.header_state_row)
        header_block.addWidget(self.divider)
        layout.addLayout(header_block)
        layout.addLayout(meeting_layout)

        notes_lbl = QLabel("Meeting notes")
        notes_lbl.setObjectName("sectionTitle")
        layout.addWidget(notes_lbl)

        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText(
            "Add decisions, follow-ups, or context as you go…"
        )
        layout.addWidget(self.notes_edit, stretch=1)

        self._done_folder_path = ""
        self._done_summary_path = ""
        self._done_transcript_path = ""

    # ---------- recording control ----------

    def _update_mode_fields(self) -> None:
        one_on_one = _is_one_on_one(self.meeting_mode)
        self.participant_field_label.setVisible(one_on_one)
        self.participant_input.setVisible(one_on_one)
        self.centralWidget().updateGeometry()

    def _on_meeting_type_changed(self, index: int) -> None:
        selected_mode = self.meeting_type_combo.itemData(index)
        if selected_mode not in MEETING_MODES or selected_mode == self.meeting_mode:
            return
        self.meeting_mode = selected_mode
        self._update_mode_fields()
        ConfigManager.set_config_value(
            selected_mode, "meeting_options", "last_meeting_mode"
        )
        try:
            ConfigManager.save_config()
        except OSError:
            pass

    def _on_record_clicked(self):
        if self.capture and self.capture.is_recording():
            self._stop_recording()
        else:
            self._start_recording()

    def _set_record_button_state(self, *, recording: bool) -> None:
        self.record_button.setText("Stop" if recording else "Start")
        self.record_button.setIcon(
            self._recording_dot_icon(active=True) if recording else QIcon()
        )
        self.record_button.setObjectName("stopButton" if recording else "startButton")
        self.record_button.style().unpolish(self.record_button)
        self.record_button.style().polish(self.record_button)
        self.record_button.update()

        self._pulse_on = True
        if recording:
            self.recording_pulse_timer.start(650)
        else:
            self.recording_pulse_timer.stop()

    def _fit_header_actions(self) -> None:
        for button, horizontal_room in (
            (self.open_summary_button, 24),
            (self.open_transcript_button, 24),
            (self.recovery_button, 24),
        ):
            button.ensurePolished()
            text_width = button.fontMetrics().horizontalAdvance(button.text())
            button.setFixedWidth(text_width + horizontal_room)
        visible_buttons = [
            button
            for button in (self.open_summary_button, self.open_transcript_button)
            if not button.isHidden()
        ]
        visible_width = sum(button.width() for button in visible_buttons)
        visible_width += max(0, len(visible_buttons) - 1) * 7
        self.completion_options.setFixedSize(visible_width, 38)

    def _clear_retry_status(self) -> None:
        self.header_state_label.clear()
        self.header_state_label.hide()
        self.completion_options.hide()
        self.action_stack.setCurrentWidget(self.record_controls_widget)
        self.action_stack.show()
        self.timer_label.show()

    def _show_header_message(
        self, text: str, action_widget: QWidget, anchor_button: QPushButton
    ) -> None:
        self.completion_options.hide()
        self.action_stack.setCurrentWidget(action_widget)
        self.action_stack.show()
        self.timer_label.show()
        self.header_state_label.setText(text)
        self.header_state_label.show()
        self.action_stack.layout().activate()
        self.header_state_row.layout().activate()
        central = self.centralWidget()
        anchor_offset = anchor_button.mapTo(central, QPoint()).x() - (
            self.header_state_label.mapTo(central, QPoint()).x()
        )
        self.header_state_label.setContentsMargins(max(0, anchor_offset), 0, 0, 0)

    def _show_completion_actions(self, summary_status: str) -> None:
        if summary_status == SUMMARY_FAILED:
            self.header_state_label.setText("Summary unavailable")
            self.header_state_label.setToolTip(
                "The transcript was saved successfully; summary generation failed."
            )
            self.header_state_label.show()
        else:
            self.header_state_label.clear()
            self.header_state_label.setToolTip("")
            self.header_state_label.hide()
        self.action_stack.hide()
        self.timer_label.hide()
        self._fit_header_actions()
        self.completion_options.show()

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

    def _show_working_status(self, text: str) -> None:
        self.header_state_label.clear()
        self.header_state_label.hide()
        self.completion_options.hide()
        self.working_status_label.setText(text)
        self.action_stack.setCurrentWidget(self.status_controls_widget)
        self.action_stack.show()
        self.timer_label.show()

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
            QMessageBox.critical(
                self, "Audio device error", f"Could not access audio: {e}"
            )
            return

        if not self.capture.start():
            QMessageBox.critical(
                self,
                "Recording failed",
                "Could not start audio capture. Check that mic and system audio are available.",
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
        self.meeting_type_combo.setEnabled(False)
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

        meeting_subject = self.meeting_name_input.text().strip()
        if not meeting_subject:
            meeting_subject, ok = self._prompt_for_meeting_name()
            meeting_subject = meeting_subject.strip() if ok else ""
        meeting_subject = meeting_subject or "Meeting"
        self.meeting_name_input.setText(meeting_subject)

        participant_name = self.participant_input.text().strip()
        if _is_one_on_one(self.meeting_mode) and not participant_name:
            participant_name, ok = self._prompt_for_participant_name()
            participant_name = participant_name.strip() if ok else ""
        participant_name = participant_name or "Participant"
        if _is_one_on_one(self.meeting_mode):
            self.participant_input.setText(participant_name)
        meeting_dir = self._meeting_directory_for_session(meeting_subject)

        self.worker = MeetingWorker(
            mic_wav=mic_wav,
            loopback_wav=loopback_wav,
            user_name=self.user_name,
            meeting_subject=meeting_subject,
            meeting_mode=self.meeting_mode,
            notes_text=self.notes_edit.toPlainText(),
            output_root=self.output_root,
            started_at=self.meeting_started_at
            or self.recording_started_at
            or datetime.now(),
            save_audio=self.save_audio,
            save_markdown=self.save_markdown,
            meeting_dir=meeting_dir,
            participant_name=participant_name,
        )
        self.worker.status_signal.connect(self._on_worker_status)
        self.worker.done_signal.connect(self._on_worker_done)
        self.worker.error_signal.connect(self._on_worker_error)
        self.worker.start()

    def _text_prompt(self, title: str, label: str) -> tuple[str, bool]:
        dialog = QInputDialog(self)
        dialog.setWindowFlags(
            (dialog.windowFlags() | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
            & ~Qt.WindowContextHelpButtonHint
        )
        dialog.setWindowTitle(title)
        dialog.setLabelText(label)
        dialog.setInputMode(QInputDialog.TextInput)
        dialog.setMinimumWidth(410)
        dialog.setStyleSheet(theme.application_stylesheet())
        apply_window_icon(dialog, SCRIBE_ICON, app_id=SCRIBE_APP_ID)
        enable_dark_titlebar(dialog)
        accepted = dialog.exec_() == QDialog.Accepted
        return dialog.textValue(), accepted

    def _prompt_for_meeting_name(self) -> tuple[str, bool]:
        return self._text_prompt(
            "Name this meeting",
            "Meeting name (used for the folder and document titles):",
        )

    def _prompt_for_participant_name(self) -> tuple[str, bool]:
        return self._text_prompt(
            "Who was this meeting with?",
            "Participant name (used for transcript labels):",
        )

    def _tick_timer(self):
        self.tick_seconds += 1
        m, s = divmod(self.tick_seconds, 60)
        h, m = divmod(m, 60)
        self.timer_label.setText(
            f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        )

    # ---------- view transitions ----------

    def _show_processing(self):
        self._set_record_button_state(recording=False)
        self._set_timer_recording(False)
        self.meeting_type_combo.setEnabled(False)
        self.meeting_name_input.setEnabled(False)
        self.participant_input.setEnabled(False)
        self.notes_edit.setEnabled(False)
        self.working_status_label.setToolTip("")
        self._show_working_status("Preparing")

    def _on_worker_status(self, msg: str):
        display = {
            "Transcribing your audio...": "Transcribing",
            "Transcribing other audio...": "Transcribing",
            "Transcribing your meeting audio...": "Transcribing",
            "Generating summary...": "Summarising",
        }.get(msg, "Transcribing")
        self._show_working_status(display)

    @staticmethod
    def _concise_error(message: str) -> str:
        lowered = message.lower()
        if "no speech detected" in lowered:
            return "No speech detected"
        if "elevenlabs http 400" in lowered:
            return "Couldn’t process audio"
        first_line = next(
            (line.strip() for line in message.splitlines() if line.strip()), ""
        )
        if not first_line:
            return "Couldn’t Finish"
        return first_line if len(first_line) <= 34 else f"{first_line[:33].rstrip()}…"

    def _on_worker_error(self, msg: str):
        self._set_record_button_state(recording=False)
        if "no speech detected" in msg.lower():
            self.meeting_type_combo.setEnabled(True)
            self.meeting_name_input.setEnabled(True)
            self.participant_input.setEnabled(True)
            self.notes_edit.setEnabled(True)
            self.timer_label.setText("00:00")
            self._set_timer_recording(False)
            self.header_state_label.setToolTip(msg)
            self._show_header_message(
                "No speech detected",
                self.record_controls_widget,
                self.record_button,
            )
            return
        self.header_state_label.setToolTip(msg)
        self._show_header_message(
            "Couldn’t process audio",
            self.recovery_controls_widget,
            self.recovery_button,
        )

    def _on_worker_done(
        self,
        folder_path: str,
        summary_path: str,
        summary_status: str = SUMMARY_READY,
    ):
        self._set_record_button_state(recording=False)
        self._done_folder_path = folder_path
        self._done_summary_path = (
            self._preferred_document_path(folder_path, "summary", summary_path)
            if summary_path
            else ""
        )
        self._done_transcript_path = self._preferred_document_path(
            folder_path, "transcript"
        )
        summary_ready = bool(self._done_summary_path)
        self.open_summary_button.setVisible(summary_ready)
        self.open_transcript_button.setProperty("primaryCompletion", not summary_ready)
        self.open_transcript_button.style().unpolish(self.open_transcript_button)
        self.open_transcript_button.style().polish(self.open_transcript_button)
        self._show_completion_actions(summary_status)

    @staticmethod
    def _preferred_document_path(
        folder_path: str, stem: str, explicit_path: str = ""
    ) -> str:
        candidates = []
        if explicit_path:
            candidates.append(Path(explicit_path))
        folder = Path(folder_path)
        candidates.extend((folder / f"{stem}.pdf", folder / f"{stem}.md"))
        unique_candidates = list(dict.fromkeys(candidates))
        for candidate in unique_candidates:
            if candidate.is_file():
                return str(candidate)
        return str(unique_candidates[0]) if unique_candidates else ""

    def _open_summary(self):
        if self._done_summary_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._done_summary_path))

    def _open_transcript(self):
        if self._done_transcript_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._done_transcript_path))

    def _open_recovery_folder(self):
        recovery_path = None
        if self.worker is not None:
            recovery_path = Path(self.worker.mic_wav).parent
        elif self.temp_dir is not None:
            recovery_path = self.temp_dir
        if recovery_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(recovery_path)))

    # ---------- shutdown ----------

    def showEvent(self, event):
        super().showEvent(event)
        enable_dark_titlebar(self)
        apply_window_icon(self, SCRIBE_ICON, app_id=SCRIBE_APP_ID)
        self._fit_header_actions()

    def closeEvent(self, event):
        if self.capture and self.capture.is_recording():
            reply = QMessageBox.question(
                self,
                "Recording in progress",
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
                self,
                "Still processing",
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

    window = ScribeWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
