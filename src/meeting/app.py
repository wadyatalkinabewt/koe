"""
Meeting Transcriber Application

Standalone application for meeting transcription with speaker separation.
Uses the shared Whisper transcription server.
"""

import os
import sys
import time
import threading
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

# Load environment variables (for HF_TOKEN)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Debug logging to file
_debug_log_path = Path(__file__).parent.parent.parent / "meeting_debug.log"
def _debug_log(msg: str):
    with open(_debug_log_path, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg)  # Also print for terminal users
# Append separator on startup (don't clear - we want to see historical logs)
with open(_debug_log_path, "a", encoding="utf-8") as f:
    f.write(f"\n=== Meeting Session Started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

import pyperclip
import numpy as np
from scipy.signal import resample_poly
from math import gcd
from PyQt5.QtCore import QObject, pyqtSignal, QTimer, Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QMessageBox, QLineEdit, QComboBox, QFileDialog, QInputDialog,
    QMainWindow
)
from PyQt5.QtGui import QIcon, QFont, QColor, QPainter, QBrush, QPainterPath

from .processor import AudioProcessor, AudioChunk
from .transcript import TranscriptWriter
# Note: diarization module imported lazily in _load_diarization_async() to avoid slow startup

# Import the transcription client and config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from transcription_client import TranscriptionClient, is_server_running, DEFAULT_SERVER_URL
from server_launcher import start_server_background
from utils import ConfigManager
from ui import theme

# Summarization imports
from .summary_status import SummaryStatusReader
import subprocess


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename.

    - Removes Windows reserved characters: < > : " / \\ | ? *
    - Handles Windows reserved names: CON, PRN, AUX, NUL, COM1-9, LPT1-9
    - Limits length to 200 characters
    - Returns 'untitled' if the result would be empty
    """
    if not name:
        return "untitled"

    # Remove Windows reserved characters
    name = re.sub(r'[<>:"/\\|?*]', '', name)

    # Remove control characters (0-31)
    name = re.sub(r'[\x00-\x1f]', '', name)

    # Handle Windows reserved names (case-insensitive)
    reserved = {'CON', 'PRN', 'AUX', 'NUL',
                'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
                'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'}
    base_name = name.split('.')[0].upper()  # Check name without extension
    if base_name in reserved:
        name = f"_{name}"

    # Strip trailing dots and spaces (Windows doesn't allow these at end of filename)
    name = name.rstrip('. ')

    # Limit length
    name = name[:200] if name else "untitled"

    return name if name else "untitled"


def post_process_text(text: str) -> str:
    """Clean up transcribed text."""
    if not text:
        return ""

    text = text.strip()

    # Remove initial prompt if it leaked into transcription
    # (Whisper sometimes hallucinates the prompt when audio is unclear)
    try:
        model_options = ConfigManager.get_config_section('model_options')
        initial_prompt = model_options.get('common', {}).get('initial_prompt', '')
        if initial_prompt:
            # Split prompt into lines and sentences, filter each separately
            prompt_parts = []
            for line in initial_prompt.strip().split('\n'):
                line = line.strip()
                if line:
                    prompt_parts.append(line)
                    # Also split on periods for sentence-level matching
                    for sentence in line.split('.'):
                        sentence = sentence.strip()
                        if len(sentence) > 10:  # Only match substantial sentences
                            prompt_parts.append(sentence)

            # Remove any prompt parts that appear in the text
            for part in prompt_parts:
                if part in text:
                    text = text.replace(part, '')
    except Exception:
        pass  # Don't fail transcription if config access fails

    # Filler words to remove
    fillers = [
        r'\bum+\b', r'\buh+\b', r'\bah+\b', r'\beh+\b',
        r'\bhmm+\b', r'\bmm+\b', r'\bhm+\b',
        r'\byou know,?\s*', r'\bI mean,?\s*',
    ]

    for filler in fillers:
        text = re.sub(filler, '', text, flags=re.IGNORECASE)

    # Common Whisper hallucinations (full match - discard entire text)
    hallucination_phrases = [
        r"^\.+$",  # Just periods
        r"^thank you\.?$",
        r"^thanks\.?$",
        r"^thank you for watching\.?$",
        r"^thanks for watching\.?$",
        r"^merci\.?$",
        r"^take care\.?$",
        r"^bye\.?$",
        r"^goodbye\.?$",
        r"^see you\.?$",
        r"^oh\.?$",
        r"^yeah\.?$",
        r"^yes\.?$",
        r"^no\.?$",
        r"^okay\.?$",
        r"^ok\.?$",
        r"^hmm\.?$",
        r"^mhm\.?$",
        r"^uh huh\.?$",
        r"^right\.?$",
        r"^sure\.?$",
        r"^yep\.?$",
        r"^nope\.?$",
        r"^hey\.?$",
        r"^hi\.?$",
        r"^hello\.?$",
        r"^\[.*\]$",  # [Music], [Applause], etc.
        r"^♪.*♪$",  # Music notes
        r"^\.{2,}$",  # Multiple periods
        r"^-+$",  # Just dashes
        r"^Угу\.?.*$",  # Russian "uh huh" hallucination
    ]

    for pattern in hallucination_phrases:
        if re.match(pattern, text, flags=re.IGNORECASE):
            return ""

    # Whisper hallucinations at end (strip from end)
    trailing_hallucinations = [
        r"\s*we'?ll be right back\.?\s*$",
        r"\s*thanks for watching\.?\s*$",
        r"\s*thank you for watching\.?\s*$",
        r"\s*subscribe to (my|the) channel\.?\s*$",
        r"\s*please like and subscribe\.?\s*$",
        r"\s*see you (in the )?next (one|video|time)\.?\s*$",
        r"\s*don'?t forget to subscribe\.?\s*$",
        r"\s*thank you\.?\s*$",
        r"\s*thanks\.?\s*$",
    ]

    for pattern in trailing_hallucinations:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Clean up
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([,.?!])', r'\1', text)
    text = re.sub(r'([,.?!])\s*\1+', r'\1', text)
    text = re.sub(r',\s*\.', '.', text)
    text = re.sub(r'^\s*,\s*', '', text)

    text = text.strip()

    # If too short after cleanup, likely hallucination
    if len(text) < 3:
        return ""

    # Apply name spelling corrections
    try:
        replacements = ConfigManager.get_config_value('post_processing', 'name_replacements') or {}
        for wrong, correct in replacements.items():
            pattern = r'\b' + re.escape(wrong) + r'\b'
            text = re.sub(pattern, correct, text, flags=re.IGNORECASE)
    except Exception:
        pass

    if text and text[-1] not in '.?!':
        text += '.'

    return text


def check_audio_has_speech(audio: np.ndarray, threshold: float = 500) -> bool:
    """Check if audio has actual speech (not just silence/noise)."""
    if audio is None or len(audio) == 0:
        return False
    # Calculate RMS energy
    rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
    return rms > threshold


def preprocess_loopback_audio(
    audio: np.ndarray,
    channels: int,
    source_rate: int,
    target_rate: int = 16000,
    target_rms: float = 3000.0
) -> np.ndarray:
    """
    Preprocess loopback audio for optimal transcription quality.

    Fixes:
    1. Stereo→mono with proper gain compensation (+3dB)
    2. High-quality resampling with anti-aliasing (polyphase filter)
    3. Normalization to target RMS level for Whisper

    Args:
        audio: Raw audio as int16
        channels: Number of audio channels
        source_rate: Source sample rate (e.g., 48000)
        target_rate: Target sample rate (16000 for Whisper)
        target_rms: Target RMS level (3000 = ~-20dB, good for Whisper)

    Returns:
        Preprocessed audio as int16 at target_rate
    """
    # Convert to float32 for processing
    audio_float = audio.astype(np.float32)

    # Step 1: Stereo to mono with gain compensation
    if channels > 1:
        try:
            audio_float = audio_float.reshape(-1, channels)
            # Sum channels (not mean) to preserve energy, then normalize
            # This is equivalent to averaging but compensating for the energy loss
            audio_float = audio_float.sum(axis=1) / np.sqrt(channels)
        except Exception as e:
            _debug_log(f"[Preprocess] Stereo conversion failed: {e}")

    # Step 2: High-quality resampling with anti-aliasing
    if source_rate != target_rate:
        # Find GCD for efficient polyphase resampling
        g = gcd(target_rate, source_rate)
        up = target_rate // g
        down = source_rate // g

        # resample_poly applies anti-aliasing filter automatically
        audio_float = resample_poly(audio_float, up, down)
        _debug_log(f"[Preprocess] Resampled {source_rate}→{target_rate} ({up}/{down})")

    # Step 3: Normalize to target RMS level
    current_rms = np.sqrt(np.mean(audio_float ** 2))
    if current_rms > 10:  # Avoid division by zero / boosting silence
        gain = target_rms / current_rms
        # Limit gain to avoid excessive amplification of noise
        gain = min(gain, 20.0)
        audio_float = audio_float * gain
        _debug_log(f"[Preprocess] Normalized RMS: {current_rms:.1f} → {target_rms:.1f} (gain: {gain:.2f}x)")

    # Clip to int16 range and convert back
    audio_float = np.clip(audio_float, -32768, 32767)
    return audio_float.astype(np.int16)


class SpeakerEnrollmentDialog(QMainWindow):
    """Dialog for enrolling unknown speakers from a meeting."""

    enrolledSignal = pyqtSignal(str, str)  # session_label, name
    closedSignal = pyqtSignal()  # Emitted when dialog closes

    BG_COLOR = QColor(10, 10, 15, 245)
    BORDER_COLOR = QColor(0, 255, 136)
    TEXT_COLOR = theme.TEXT_COLOR

    def __init__(self, unknown_speakers: dict, speaker_samples: dict, diarizer, transcript_path: Path):
        """
        Args:
            unknown_speakers: Dict mapping "Speaker 1" -> embedding
            speaker_samples: Dict mapping speaker name -> list of sample transcriptions
            diarizer: DiarizationManager instance for enrollment
            transcript_path: Path to transcript file for rewriting after enrollment
        """
        super().__init__()
        self.unknown_speakers = unknown_speakers
        self.speaker_samples = speaker_samples
        self.diarizer = diarizer
        self.transcript_path = transcript_path
        self._drag_pos = None
        self._name_inputs = {}  # label -> QLineEdit (for unknown speakers)
        self._enroll_buttons = {}  # label -> QPushButton (for disabling after enrollment)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Enroll Speakers')
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        # Calculate height based on number of unknown speakers
        num_unknown = len(self.unknown_speakers)
        base_height = 100  # Header + footer
        per_speaker_height = 180  # Each speaker row (more space for samples)
        total_height = base_height + (num_unknown * per_speaker_height)
        # Larger dialog for better readability (taller to accommodate expanded transcripts)
        self.setFixedSize(700, min(total_height, 800))

        main_widget = QWidget(self)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(18, 14, 18, 14)
        main_layout.setSpacing(10)

        # Header
        header = QLabel("> Enroll speakers")
        header.setFont(QFont('Cascadia Code', 12, QFont.Bold))
        header.setStyleSheet(f"color: {self.TEXT_COLOR};")
        main_layout.addWidget(header)

        # Scrollable area for speakers
        from PyQt5.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QWidget { background: transparent; }
        """)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(8)

        # Unknown speakers (with name input + Enroll button)
        for label in sorted(self.unknown_speakers.keys()):
            row = self._create_unknown_speaker_row(label)
            scroll_layout.addWidget(row)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll, 1)

        # Close button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("[ESC] Close")
        close_btn.setFont(QFont('Cascadia Code', 9))
        close_btn.setStyleSheet("""
            QPushButton { color: #3a4a4a; background: transparent; border: none; padding: 4px 8px; }
            QPushButton:hover { color: #ff6666; }
        """)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        main_layout.addLayout(btn_layout)

        self.setCentralWidget(main_widget)
        self._center_window()

    def _create_unknown_speaker_row(self, label: str) -> QWidget:
        """Create a row for an unknown speaker (with name input + Enroll button)."""
        row = QWidget()
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 12)
        row_layout.setSpacing(6)

        # Speaker label with entry count
        samples = self.speaker_samples.get(label, [])
        speaker_label = QLabel(f"{label} ({len(samples)} entries):")
        speaker_label.setFont(QFont('Cascadia Code', 10, QFont.Bold))
        speaker_label.setStyleSheet(f"color: {self.TEXT_COLOR};")
        row_layout.addWidget(speaker_label)

        # Sample text - compact preview
        sample_text = self._format_samples(samples)
        sample_label = QLabel(sample_text)
        sample_label.setFont(QFont('Cascadia Code', 9))
        sample_label.setStyleSheet("color: #888; padding-left: 8px;")
        sample_label.setWordWrap(True)
        sample_label.setTextFormat(Qt.PlainText)
        sample_label.setMinimumHeight(40)
        row_layout.addWidget(sample_label)

        # "Show all" toggle + full transcript area (only if more than 5 entries)
        if len(samples) > 5:
            from PyQt5.QtWidgets import QTextEdit
            full_text_area = QTextEdit()
            full_text_area.setReadOnly(True)
            full_text_area.setFont(QFont('Cascadia Code', 9))
            full_text_area.setPlainText(self._format_samples(samples, full=True))
            full_text_area.setStyleSheet("""
                QTextEdit {
                    color: #888;
                    background: rgba(0, 0, 0, 0.2);
                    border: 1px solid #2a3a3a;
                    border-radius: 3px;
                    padding: 6px 8px;
                }
            """)
            full_text_area.setMaximumHeight(200)
            full_text_area.setVisible(False)

            toggle_btn = QPushButton(f"▸ Show all {len(samples)} entries")
            toggle_btn.setFont(QFont('Cascadia Code', 8))
            toggle_btn.setStyleSheet("""
                QPushButton { color: #00ff88; background: transparent; border: none; padding: 2px 8px; text-align: left; }
                QPushButton:hover { color: #33ffaa; }
            """)
            toggle_btn.setCursor(Qt.PointingHandCursor)

            def _toggle(checked, btn=toggle_btn, preview=sample_label, full_area=full_text_area, count=len(samples)):
                if full_area.isVisible():
                    full_area.setVisible(False)
                    preview.setVisible(True)
                    btn.setText(f"▸ Show all {count} entries")
                else:
                    full_area.setVisible(True)
                    preview.setVisible(False)
                    btn.setText(f"▾ Show preview")

            toggle_btn.clicked.connect(_toggle)
            row_layout.addWidget(toggle_btn)
            row_layout.addWidget(full_text_area)

        # Name input row
        input_row = QHBoxLayout()
        input_row.setSpacing(8)

        name_input = QLineEdit()
        name_input.setPlaceholderText("Enter name to enroll...")
        name_input.setFont(QFont('Cascadia Code', 10))
        name_input.setStyleSheet("""
            QLineEdit {
                background: rgba(0, 0, 0, 0.3);
                border: 1px solid #3a4a4a;
                border-radius: 3px;
                padding: 6px 10px;
                color: #00ff88;
            }
            QLineEdit:focus { border-color: #00ff88; }
        """)
        self._name_inputs[label] = name_input
        input_row.addWidget(name_input, 1)

        enroll_btn = QPushButton("Enroll")
        enroll_btn.setFont(QFont('Cascadia Code', 9))
        enroll_btn.setStyleSheet("""
            QPushButton {
                color: #00ff88;
                background: rgba(0, 255, 136, 0.1);
                border: 1px solid #00ff88;
                border-radius: 3px;
                padding: 6px 12px;
            }
            QPushButton:hover { background: rgba(0, 255, 136, 0.2); }
            QPushButton:disabled { color: #555; border-color: #555; background: transparent; }
        """)
        enroll_btn.setCursor(Qt.PointingHandCursor)
        # Capture label in closure for this button
        captured_label = label  # Explicit capture
        _debug_log(f"[Enroll] Creating button for label='{captured_label}'")
        enroll_btn.clicked.connect(lambda checked, lbl=captured_label: self._enroll_speaker(lbl))
        self._enroll_buttons[label] = enroll_btn
        input_row.addWidget(enroll_btn)

        row_layout.addLayout(input_row)
        return row

    def _format_samples(self, samples: list, full: bool = False) -> str:
        """Format sample transcriptions for display.

        Args:
            samples: List of sample transcription strings.
            full: If True, show all samples without truncation.
        """
        if not samples:
            return "(no text samples)"

        if full:
            # Show everything, no truncation
            lines = [f"• {sample}" for sample in samples]
            return "\n".join(lines)

        # Preview mode: show up to 5 samples, truncated
        display_samples = samples[:5]
        lines = []
        for sample in display_samples:
            if len(sample) > 120:
                sample = sample[:120] + "..."
            lines.append(f"• {sample}")

        result = "\n".join(lines)

        if len(samples) > 5:
            result += f"\n... and {len(samples) - 5} more entries"

        return result

    def _center_window(self):
        screen_geo = QApplication.desktop().availableGeometry()
        x = (screen_geo.width() - self.width()) // 2
        y = (screen_geo.height() - self.height()) // 2
        self.move(x, y)

    def _find_similar_speakers(self, target_label: str) -> list:
        """Find other unknown speakers with similar embeddings to target.

        Returns list of labels that should be merged with target.
        """
        if target_label not in self.unknown_speakers:
            return []

        target_embedding = self.unknown_speakers[target_label]
        similar = []

        SIMILARITY_THRESHOLD = 0.40  # Same as consolidation threshold

        for label, embedding in self.unknown_speakers.items():
            if label == target_label:
                continue

            # Cosine similarity (with zero-norm protection)
            norm_product = np.linalg.norm(target_embedding) * np.linalg.norm(embedding)
            if norm_product == 0:
                continue
            similarity = np.dot(target_embedding, embedding) / norm_product

            if similarity >= SIMILARITY_THRESHOLD:
                _debug_log(f"[Enroll] Auto-merge: '{label}' similar to '{target_label}' ({similarity:.3f})")
                similar.append(label)

        return similar

    def _rewrite_transcript(self, old_labels: list, new_name: str):
        """Rewrite transcript file, replacing old speaker labels with new name.

        Args:
            old_labels: List of labels to replace (e.g., ["Speaker 1", "Speaker 3"])
            new_name: New name to use (e.g., "Sritam")
        """
        if not self.transcript_path or not self.transcript_path.exists():
            _debug_log(f"[Enroll] Transcript not found: {self.transcript_path}")
            return

        try:
            content = self.transcript_path.read_text(encoding='utf-8')

            for old_label in old_labels:
                # Replace transcript entries: **[HH:MM] Speaker 1**: → **[HH:MM] Sritam**:
                # Pattern handles both [MM:SS] and [HH:MM:SS] formats
                pattern = rf'\*\*\[(\d{{2}}:\d{{2}}(?::\d{{2}})?)\] {re.escape(old_label)}\*\*:'
                replacement = rf'**[\1] {new_name}**:'
                content = re.sub(pattern, replacement, content)

            # Update Participants header line
            # Match: **Participants**: Name1, Name2, Speaker 1, Name3
            participants_pattern = r'(\*\*Participants\*\*: )(.+)'
            match = re.search(participants_pattern, content)
            if match:
                prefix = match.group(1)
                participants_str = match.group(2)
                participants = [p.strip() for p in participants_str.split(',')]

                # Replace old labels with new name, remove duplicates
                new_participants = []
                seen = set()
                for p in participants:
                    if p in old_labels:
                        if new_name not in seen:
                            new_participants.append(new_name)
                            seen.add(new_name)
                    else:
                        if p not in seen:
                            new_participants.append(p)
                            seen.add(p)

                new_participants_str = ', '.join(new_participants)
                content = re.sub(participants_pattern, prefix + new_participants_str, content)

            # Save updated content
            self.transcript_path.write_text(content, encoding='utf-8')
            _debug_log(f"[Enroll] Rewrote transcript: {old_labels} -> {new_name}")

        except Exception as e:
            _debug_log(f"[Enroll] Error rewriting transcript: {e}")

    def _enroll_speaker(self, label: str):
        """Enroll an unknown speaker with the given name."""
        _debug_log(f"[Enroll] _enroll_speaker called with label='{label}'")
        _debug_log(f"[Enroll] Available inputs: {list(self._name_inputs.keys())}")

        name_input = self._name_inputs.get(label)
        if not name_input:
            _debug_log(f"[Enroll] ERROR: No input found for label '{label}'")
            return

        # Log all input values for debugging
        for lbl, inp in self._name_inputs.items():
            _debug_log(f"[Enroll] Input '{lbl}' has text: '{inp.text()}'")

        name = name_input.text().strip()
        _debug_log(f"[Enroll] Captured name='{name}' from label='{label}'")

        if not name:
            _debug_log(f"[Enroll] ERROR: Empty name for label '{label}'")
            return

        # Check if name is already enrolled (prevent accidental overwrites)
        if self.diarizer:
            existing_speakers = self.diarizer.list_enrolled_speakers()
            if name in existing_speakers:
                _debug_log(f"[Enroll] WARNING: '{name}' is already enrolled - showing warning")
                # Show warning and abort
                name_input.setStyleSheet("""
                    QLineEdit {
                        background: rgba(255, 100, 100, 0.2);
                        border: 1px solid #ff6666;
                        border-radius: 3px;
                        padding: 6px 10px;
                        color: #ff6666;
                    }
                """)
                name_input.setText(f"'{name}' already enrolled!")
                # Re-enable after 2 seconds
                QTimer.singleShot(2000, lambda: self._reset_input_after_warning(label, name_input))
                return

        # Check if this name was just used in this session (another speaker enrolled with same name)
        # This is a MANUAL MERGE - user is saying "this speaker is also <name>"
        for other_label, other_input in self._name_inputs.items():
            if other_label != label and not other_input.isEnabled():
                enrolled_text = other_input.text()
                if f"Enrolled as {name}" == enrolled_text or f"Merged as {name}" == enrolled_text:
                    _debug_log(f"[Enroll] Manual merge: '{label}' is also '{name}' (user override)")
                    # User recognizes this is the same person from transcript context
                    # Just rewrite the transcript, don't save new embedding (first one is usually better)
                    self._rewrite_transcript([label], name)

                    # Update UI to show merged
                    name_input.setEnabled(False)
                    name_input.setText(f"Merged as {name}")
                    name_input.setStyleSheet("""
                        QLineEdit {
                            background: rgba(0, 255, 136, 0.1);
                            border: 1px solid #00ff88;
                            border-radius: 3px;
                            padding: 6px 10px;
                            color: #00ff88;
                        }
                    """)
                    enroll_btn = self._enroll_buttons.get(label)
                    if enroll_btn:
                        enroll_btn.setEnabled(False)
                    self.unknown_speakers.pop(label, None)
                    _debug_log(f"[Enroll] Manual merge complete: '{label}' -> '{name}'")
                    return

        # Get the embedding from our stored copy (safer than relying on diarizer's session)
        embedding = self.unknown_speakers.get(label)
        if embedding is None:
            _debug_log(f"[Enroll] ERROR: No embedding found for label '{label}'")
            return

        _debug_log(f"[Enroll] Using embedding from unknown_speakers dict (shape={embedding.shape})")

        # Enroll using the embedding we already have (safer than enroll_speaker_from_session)
        if self.diarizer and self.diarizer.enroll_speaker_with_embedding(name, embedding):
            _debug_log(f"[Enroll] Successfully enrolled '{label}' as '{name}'")

            # Find similar speakers to auto-merge
            similar_labels = self._find_similar_speakers(label)
            all_labels = [label] + similar_labels

            # Rewrite transcript with all matching labels
            self._rewrite_transcript(all_labels, name)

            # Update UI for all similar speakers too
            _debug_log(f"[Enroll] Found {len(similar_labels)} similar speakers: {similar_labels}")
            for similar_label in similar_labels:
                _debug_log(f"[Enroll] Processing similar speaker '{similar_label}'")
                similar_input = self._name_inputs.get(similar_label)
                if similar_input:
                    merge_text = f"Merged as {name}"
                    _debug_log(f"[Enroll] Setting '{similar_label}' text to: '{merge_text}'")
                    similar_input.setEnabled(False)
                    similar_input.setText(merge_text)
                    similar_input.setStyleSheet("""
                        QLineEdit {
                            background: rgba(0, 255, 136, 0.1);
                            border: 1px solid #00ff88;
                            border-radius: 3px;
                            padding: 6px 10px;
                            color: #00ff88;
                        }
                    """)
                # Disable the enroll button for similar speaker
                similar_btn = self._enroll_buttons.get(similar_label)
                if similar_btn:
                    similar_btn.setEnabled(False)
                # Remove from unknown_speakers to prevent re-processing
                self.unknown_speakers.pop(similar_label, None)

            # Update UI for enrolled speaker
            enroll_text = f"Enrolled as {name}"
            _debug_log(f"[Enroll] Setting '{label}' text to: '{enroll_text}'")
            name_input.setEnabled(False)
            name_input.setText(enroll_text)
            name_input.setStyleSheet("""
                QLineEdit {
                    background: rgba(0, 255, 136, 0.1);
                    border: 1px solid #00ff88;
                    border-radius: 3px;
                    padding: 6px 10px;
                    color: #00ff88;
                }
            """)
            # Disable the enroll button for this speaker
            enroll_btn = self._enroll_buttons.get(label)
            if enroll_btn:
                enroll_btn.setEnabled(False)
            # Remove from unknown_speakers to prevent re-processing
            self.unknown_speakers.pop(label, None)

            self.enrolledSignal.emit(label, name)

    def _reset_input_after_warning(self, label: str, name_input):
        """Reset input field after showing duplicate name warning."""
        if name_input.isEnabled():
            return  # Already processed, don't reset
        name_input.setEnabled(True)
        name_input.setText("")
        name_input.setPlaceholderText("Enter name to enroll...")
        name_input.setStyleSheet("""
            QLineEdit {
                background: rgba(0, 0, 0, 0.3);
                border: 1px solid #3a4a4a;
                border-radius: 3px;
                padding: 6px 10px;
                color: #00ff88;
            }
            QLineEdit:focus { border-color: #00ff88; }
        """)
        _debug_log(f"[Enroll] Reset input for '{label}' after warning")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), 8, 8)
        painter.fillPath(path, QBrush(self.BG_COLOR))
        painter.setPen(self.BORDER_COLOR)
        painter.drawPath(path)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self._drag_pos:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, event):
        """Emit signal when dialog closes."""
        self.closedSignal.emit()
        super().closeEvent(event)


class SummaryStatusWindow(QMainWindow):
    """Small popup window showing summary generation progress."""

    closeSignal = pyqtSignal()
    newMeetingSignal = pyqtSignal()
    linkClickedSignal = pyqtSignal(str)
    startSummarizationSignal = pyqtSignal(object)  # Emits transcript_path when ready to summarize

    # Terminal colors (from centralized theme)
    BG_COLOR = QColor(10, 10, 15, 245)
    BORDER_COLOR = QColor(0, 255, 136)
    TEXT_COLOR = theme.TEXT_COLOR
    SECONDARY_TEXT = theme.SECONDARY_TEXT

    def __init__(self, meeting_name: str, duration_seconds: int):
        super().__init__()
        self._drag_pos = None
        self._unknown_speakers = {}  # Will be set after meeting ends
        self._speaker_samples = {}
        self._diarizer = None
        self._enrollment_dialog = None
        self._transcript_path = None  # For deferred summarization
        self._summarization_started = False  # Prevent double-triggering
        self.initUI()

    def initUI(self):
        """Initialize the compact UI."""
        self.setWindowTitle('Summary')
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFixedSize(450, 85)  # Slightly wider and taller for better spacing

        # Main widget
        self.main_widget = QWidget(self)
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(18, 14, 18, 14)
        self.main_layout.setSpacing(12)

        # Top row: status message (centered)
        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(0)

        status_row.addStretch(1)

        self.status_label = QLabel("> Starting...")
        self.status_label.setFont(QFont('Cascadia Code', 13, QFont.Bold))
        self.status_label.setStyleSheet(f"color: {self.TEXT_COLOR}; padding: 2px 0px;")
        self.status_label.setTextFormat(Qt.RichText)
        self.status_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.status_label.setMinimumHeight(24)  # Prevent jumping when text changes
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.linkActivated.connect(self.linkClickedSignal.emit)
        status_row.addWidget(self.status_label)

        status_row.addStretch(1)

        self.main_layout.addLayout(status_row)

        # Bottom row: buttons
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(8)

        button_layout.addStretch()

        # Enroll speakers button (hidden by default, shown when speakers available)
        self.enroll_btn = QPushButton('Enroll Speakers')
        self.enroll_btn.setFont(QFont('Cascadia Code', 9))
        self.enroll_btn.setStyleSheet("""
            QPushButton {
                color: #ffaa00;
                background: transparent;
                border: 1px solid #ffaa00;
                border-radius: 3px;
                padding: 4px 12px;
            }
            QPushButton:hover {
                background: rgba(255, 170, 0, 0.1);
            }
        """)
        self.enroll_btn.setCursor(Qt.PointingHandCursor)
        self.enroll_btn.clicked.connect(self._open_enrollment)
        self.enroll_btn.hide()  # Hidden until speakers are set
        button_layout.addWidget(self.enroll_btn)

        # New Meeting button
        new_meeting_btn = QPushButton('New Meeting')
        new_meeting_btn.setFont(QFont('Cascadia Code', 9))
        new_meeting_btn.setStyleSheet("""
            QPushButton {
                color: #00ff88;
                background: transparent;
                border: 1px solid #00ff88;
                border-radius: 3px;
                padding: 4px 12px;
            }
            QPushButton:hover {
                background: rgba(0, 255, 136, 0.1);
            }
        """)
        new_meeting_btn.setCursor(Qt.PointingHandCursor)
        new_meeting_btn.clicked.connect(self.newMeetingSignal.emit)
        button_layout.addWidget(new_meeting_btn)

        # ESC button
        esc_btn = QPushButton('[ESC]')
        esc_btn.setFont(QFont('Cascadia Code', 9))
        esc_btn.setStyleSheet("""
            QPushButton {
                color: #3a4a4a;
                background: transparent;
                border: none;
                padding: 4px 8px;
            }
            QPushButton:hover {
                color: #ff6666;
            }
        """)
        esc_btn.setCursor(Qt.PointingHandCursor)
        esc_btn.clicked.connect(self.closeSignal.emit)
        button_layout.addWidget(esc_btn)

        self.main_layout.addLayout(button_layout)

        self.setCentralWidget(self.main_widget)
        self.center_window()

    def center_window(self):
        """Center window on screen."""
        screen_geo = QApplication.desktop().availableGeometry()
        x = (screen_geo.width() - self.width()) // 2
        y = (screen_geo.height() - self.height()) // 2
        self.move(x, y)

    def update_status(self, text: str):
        """Update the status text."""
        self.status_label.setText(f"> {text}")

    def show_summary_link(self, summary_path: Path):
        """Show clickable summary link."""
        url_path = str(summary_path).replace("\\", "/")
        clickable = f'<a href="file:///{url_path}" style="color: #00ffaa; text-decoration: underline;">{summary_path.name}</a>'
        self.status_label.setText(f"✓ {clickable}")

    def set_enrollment_data(self, unknown_speakers: dict, speaker_samples: dict, diarizer, transcript_path: Path):
        """Set data for speaker enrollment.

        Args:
            unknown_speakers: Dict mapping "Speaker 1" -> embedding
            speaker_samples: Dict mapping speaker name -> list of sample transcriptions
            diarizer: DiarizationManager instance for enrollment
            transcript_path: Path to transcript file for rewriting
        """
        self._unknown_speakers = unknown_speakers
        self._speaker_samples = speaker_samples
        self._diarizer = diarizer
        self._transcript_path = transcript_path

        # Only show enrollment button if there are UNKNOWN speakers
        if unknown_speakers:
            self.enroll_btn.show()
            # Make window a bit wider to fit the button
            self.setFixedSize(520, 85)
            self.center_window()

    def _trigger_summarization_if_needed(self):
        """Start summarization if it hasn't been started yet."""
        if not self._summarization_started and self._transcript_path:
            self._summarization_started = True
            self.startSummarizationSignal.emit(self._transcript_path)

    def _on_enrollment_closed(self):
        """Called when enrollment dialog is closed - trigger summarization."""
        self._trigger_summarization_if_needed()

    def _open_enrollment(self):
        """Open the speaker enrollment dialog."""
        if not self._unknown_speakers:
            return

        self._enrollment_dialog = SpeakerEnrollmentDialog(
            self._unknown_speakers,
            self._speaker_samples,
            self._diarizer,
            self._transcript_path
        )

        # Connect closed signal to trigger summarization
        self._enrollment_dialog.closedSignal.connect(self._on_enrollment_closed)

        self._enrollment_dialog.show()

    def paintEvent(self, event):
        """Custom paint for rounded border."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background with rounded corners
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), 8, 8)
        painter.fillPath(path, QBrush(self.BG_COLOR))

        # Border
        painter.setPen(self.BORDER_COLOR)
        painter.drawPath(path)

    def mousePressEvent(self, event):
        """Handle mouse press for dragging."""
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        """Handle mouse move for dragging."""
        if event.buttons() == Qt.LeftButton and self._drag_pos:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()

    def keyPressEvent(self, event):
        """Handle ESC key."""
        if event.key() == Qt.Key_Escape:
            self.closeSignal.emit()

    def closeEvent(self, event):
        """Handle window close - ensure summarization starts."""
        self._trigger_summarization_if_needed()
        super().closeEvent(event)


class MeetingTranscriberApp(QObject):
    """Main application for meeting transcription."""

    # Signals for thread-safe UI updates
    transcription_ready = pyqtSignal(str, str, float)  # speaker, text, timestamp
    status_changed = pyqtSignal(str)
    recording_time_updated = pyqtSignal(int)  # seconds
    summary_status_changed = pyqtSignal(str)  # For summary window updates
    start_summary_polling = pyqtSignal(object)  # Trigger polling setup on main thread (passes Path object)
    enrollment_data_ready = pyqtSignal(dict, dict, object)  # unknown_speakers, speaker_samples, transcript_path

    def __init__(self, user_name: str = "Bryce", server_url: str = DEFAULT_SERVER_URL, use_diarization: bool = True):
        super().__init__()

        # Single-instance check: use a reliable named Mutex on Windows
        import ctypes
        from ctypes.wintypes import HANDLE, BOOL, DWORD
        
        ERROR_ALREADY_EXISTS = 183
        self._mutex_name = "KoeMeetingAppMutex_v1"
        try:
            self._mutex = ctypes.windll.kernel32.CreateMutexW(None, False, self._mutex_name)  # type: ignore
            last_error = ctypes.windll.kernel32.GetLastError()  # type: ignore
        except AttributeError:
            self._mutex = None
            last_error = 0
            
        if last_error == ERROR_ALREADY_EXISTS:
            # Another instance is already running
            print("[Scribe] Another instance is already running. Exiting.")
            sys.exit(0)

        self.user_name = user_name
        self.server_url = server_url
        self.use_diarization = use_diarization

        self.app = QApplication.instance() or QApplication(sys.argv)

        # Set Windows AppUserModelID for proper taskbar icon
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('Koe.Transcription.App')
        except Exception:
            pass

        icon_path = str(Path(__file__).parent.parent.parent / "assets" / "koe-icon.ico")
        self.app.setWindowIcon(QIcon(icon_path))
        self.app.setQuitOnLastWindowClosed(True)  # Close app when window closes

        # Transcription client (connects to server)
        self.client = TranscriptionClient(server_url)

        # Speaker diarization (will be loaded in background)
        self._diarizer = None
        self._diarization_available = False
        self._diarization_loading = False

        # Components
        self.processor: Optional[AudioProcessor] = None
        self.transcript = TranscriptWriter()

        # Speaker tracking (for loopback audio)
        self._speaker_counter = 0
        self._speaker_map: dict = {}  # Maps pyannote labels to friendly names

        # State
        self._recording = False
        self._meeting_start_time: float = 0
        self._timer: Optional[QTimer] = None
        # ===== REMOVED: self._debug_timer (no longer used for buffer status updates) =====
        self._server_checked = False
        self._chunks_processed = 0
        self._server_diarization_available = False  # For remote mode fallback

        # Concurrency control: only allow 1 chunk to be transcribed at a time
        # Prevents request storms that overwhelm the GPU and freeze the PC
        self._chunk_semaphore = threading.Semaphore(1)
        self._consecutive_failures = 0  # Circuit breaker counter

        # Pre-meeting file state (for opening existing agenda files)
        self._opened_filepath: Optional[Path] = None
        self._existing_transcript_entries: list = []

        # Summarization tracking
        self._status_readers: dict = {}  # Maps transcript path to SummaryStatusReader
        self._summary_window: Optional[SummaryStatusWindow] = None
        self._last_meeting_name: str = ""
        self._last_meeting_duration: int = 0

        # Setup UI first (show window immediately)
        self._setup_live_window()

        # Connect signals
        self.transcription_ready.connect(self._on_transcription)
        self.status_changed.connect(self._on_status_change)
        self.recording_time_updated.connect(self._on_time_update)
        self.start_summary_polling.connect(self._start_polling_summary_status)
        self.enrollment_data_ready.connect(self._set_enrollment_data)

        # Check for crash recovery data AFTER window is set up
        self._check_crash_recovery()

        # Load diarization model in background (after window is visible)
        if use_diarization:
            self._load_diarization_async()

    def _setup_live_window(self):
        """Setup the live transcription preview window."""
        self.live_window = QWidget()
        self.live_window.setWindowTitle("Scribe")
        self.live_window.setWindowIcon(QIcon(str(Path(__file__).parent.parent.parent / "assets" / "koe-icon.png")))
        self.live_window.setMinimumSize(700, 550)  # Smaller minimum for high-DPI laptops
        self.live_window.resize(900, 700)  # Reasonable default that works on 14" laptops

        # Terminal-style dark theme with green accents
        self.live_window.setStyleSheet("""
            QWidget {
                background-color: #0a0a0f;
                color: #00ff88;
                font-family: 'Cascadia Code', 'Consolas', 'Courier New', monospace;
            }
            QLabel {
                color: #00ff88;
            }
            QLabel#sectionHeader {
                color: #00ffaa;
                font-size: 14px;
                font-weight: bold;
                padding: 4px 0px;
                border-bottom: 1px solid #1a3a2a;
            }
            QLineEdit {
                background-color: #0d1117;
                border: 1px solid #1a3a2a;
                border-radius: 4px;
                padding: 10px 12px;
                font-size: 14px;
                color: #00ff88;
                font-family: 'Cascadia Code', 'Consolas', monospace;
            }
            QLineEdit:focus {
                border: 1px solid #00ff88;
                background-color: #0f1419;
            }
            QLineEdit::placeholder {
                color: #3a5a4a;
            }
            QTextEdit {
                background-color: #0d1117;
                border: 1px solid #1a3a2a;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Cascadia Code', 'Consolas', monospace;
                font-size: 13px;
                color: #c0c0c0;
                selection-background-color: #00ff88;
                selection-color: #0a0a0f;
            }
            QTextEdit:focus {
                border: 1px solid #00ff88;
            }
            QComboBox {
                background-color: #0d1117;
                border: 1px solid #1a3a2a;
                border-radius: 4px;
                padding: 10px 12px;
                font-size: 14px;
                color: #00ff88;
                font-family: 'Cascadia Code', 'Consolas', monospace;
            }
            QComboBox:focus {
                border: 1px solid #00ff88;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 8px;
            }
            QComboBox::down-arrow {
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #00ff88;
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                background-color: #0d1117;
                border: 1px solid #1a3a2a;
                color: #00ff88;
                selection-background-color: #1a3a2a;
            }
            QPushButton {
                background-color: #1a2a2a;
                border: 1px solid #2a4a3a;
                border-radius: 3px;
                padding: 6px 12px;
                font-size: 12px;
                font-weight: 500;
                color: #00ff88;
                font-family: 'Cascadia Code', 'Consolas', monospace;
            }
            QPushButton:hover {
                background-color: #2a3a3a;
                border-color: #00ff88;
            }
            QPushButton:pressed {
                background-color: #0a1a1a;
            }
            QPushButton#recordButton {
                background-color: rgba(80, 20, 20, 0.6);
                border: 2px solid #aa3333;
                color: #ff6666;
                font-weight: 600;
                font-size: 13px;
                border-radius: 3px;
            }
            QPushButton#recordButton:hover {
                background-color: rgba(100, 30, 30, 0.8);
                border-color: #cc4444;
                color: #ff8888;
            }
            QPushButton#recordButton:pressed {
                background-color: rgba(60, 15, 15, 0.9);
                border-color: #aa3333;
            }
            QPushButton#addButton {
                background-color: #1a2a2a;
                border: 1px solid #2a4a3a;
                padding: 10px 12px;
                font-size: 16px;
                font-weight: bold;
                min-width: 36px;
                max-width: 36px;
            }
            QPushButton#addButton:hover {
                background-color: #2a3a3a;
                border-color: #00ff88;
            }
            QPushButton#subtleButton {
                background-color: transparent;
                border: 1px solid #2a4a3a;
                border-radius: 3px;
                color: #4a7a6a;
                font-size: 12px;
                padding: 4px 12px;
            }
            QPushButton#subtleButton:hover {
                color: #00ff88;
                border-color: #00ff88;
            }
            QMenu {
                background-color: #0d1117;
                border: 1px solid #1a3a2a;
                color: #00ff88;
                padding: 4px;
            }
            QMenu::item {
                background-color: transparent;
                padding: 6px 20px;
                border-radius: 2px;
            }
            QMenu::item:selected {
                background-color: #1a3a2a;
                color: #00ffaa;
            }
            QMenu::separator {
                height: 1px;
                background: #1a3a2a;
                margin: 4px 8px;
            }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        # Header with status and time
        header_layout = QHBoxLayout()

        self.status_label = QLabel("> Ready")
        self.status_label.setStyleSheet("font-size: 22px; font-weight: 600; color: #00ff88;")
        # Enable rich text for clickable links (summary links)
        self.status_label.setTextFormat(Qt.RichText)
        self.status_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.status_label.linkActivated.connect(self._open_file)
        header_layout.addWidget(self.status_label)

        header_layout.addStretch()

        self.time_label = QLabel("00:00")
        self.time_label.setStyleSheet("""
            font-size: 28px;
            font-weight: 700;
            color: #00ffaa;
            font-family: 'Cascadia Code', 'Consolas', monospace;
        """)
        header_layout.addWidget(self.time_label)

        # Recording indicator (hidden by default) - just a pulsating red circle
        self.rec_dot = QLabel("●")
        self.rec_dot.setStyleSheet("""
            font-size: 32px;
            font-weight: 700;
            color: #ff4444;
            background: transparent;
            border: none;
        """)
        self.recording_indicator = self.rec_dot
        self.recording_indicator.hide()
        self._rec_opacity = 1.0
        self._rec_fade_direction = -1
        header_layout.addWidget(self.recording_indicator)

        layout.addLayout(header_layout)

        # Server status row with discreet Open/Save buttons
        server_row = QHBoxLayout()
        server_row.setSpacing(8)

        self.server_label = QLabel("// checking server...")
        self.server_label.setStyleSheet("color: #3a5a4a; font-size: 13px;")
        server_row.addWidget(self.server_label)

        server_row.addStretch()

        # Discreet file buttons (for pre-meeting agenda workflow)
        self.open_button = QPushButton("Open")
        self.open_button.setObjectName("subtleButton")
        self.open_button.setFixedHeight(26)
        self.open_button.clicked.connect(self._open_existing_meeting)
        server_row.addWidget(self.open_button)

        self.save_button = QPushButton("Save")
        self.save_button.setObjectName("subtleButton")
        self.save_button.setFixedHeight(26)
        self.save_button.clicked.connect(self._save_notes_only)
        server_row.addWidget(self.save_button)

        layout.addLayout(server_row)

        # Form inputs - organized in 2 rows for compact layout
        LABEL_WIDTH = 100

        # Row 1: NAME
        row1_layout = QHBoxLayout()
        row1_layout.setSpacing(16)

        # NAME section
        self.name_layout = QHBoxLayout()
        self.name_layout.setSpacing(8)
        self.name_label = QLabel("NAME")
        self.name_label.setObjectName("sectionHeader")
        self.name_label.setFixedWidth(LABEL_WIDTH)
        self.name_layout.addWidget(self.name_label)

        self.meeting_name_input = QLineEdit()
        self.meeting_name_input.setMaxLength(100)
        self.name_layout.addWidget(self.meeting_name_input)
        row1_layout.addLayout(self.name_layout, 1)  # stretch factor 1

        layout.addLayout(row1_layout)

        # Row 2: CATEGORY + SUB (hierarchically related)
        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(16)

        # CATEGORY section
        self.category_layout = QHBoxLayout()
        self.category_layout.setSpacing(8)
        self.category_label = QLabel("CATEGORY")
        self.category_label.setObjectName("sectionHeader")
        self.category_label.setFixedWidth(LABEL_WIDTH)
        self.category_layout.addWidget(self.category_label)

        self.category_combo = QComboBox()
        self.category_combo.setFocusPolicy(Qt.StrongFocus)
        self.category_combo.wheelEvent = lambda e: e.ignore()
        self.category_combo.currentIndexChanged.connect(self._on_category_changed)
        self.category_layout.addWidget(self.category_combo)

        # Add category button
        self.add_category_button = QPushButton("+")
        self.add_category_button.setObjectName("addButton")
        self.add_category_button.clicked.connect(self._add_new_category)
        self.add_category_button.setToolTip("Create new category")
        self.category_layout.addWidget(self.add_category_button)
        row2_layout.addLayout(self.category_layout, 1)  # stretch factor 1

        # SUB section
        self.subcategory_layout = QHBoxLayout()
        self.subcategory_layout.setSpacing(8)
        self.subcategory_label = QLabel("FOLDER")
        self.subcategory_label.setObjectName("sectionHeader")
        self.subcategory_label.setFixedWidth(LABEL_WIDTH)
        self.subcategory_layout.addWidget(self.subcategory_label)

        self.subcategory_combo = QComboBox()
        self.subcategory_combo.setFocusPolicy(Qt.StrongFocus)
        self.subcategory_combo.wheelEvent = lambda e: e.ignore()
        self.subcategory_layout.addWidget(self.subcategory_combo)

        # Add subcategory button
        self.add_subcategory_button = QPushButton("+")
        self.add_subcategory_button.setObjectName("addButton")
        self.add_subcategory_button.clicked.connect(self._add_new_subcategory)
        self.add_subcategory_button.setToolTip("Create new subcategory")
        self.subcategory_layout.addWidget(self.add_subcategory_button)
        row2_layout.addLayout(self.subcategory_layout, 1)  # stretch factor 1

        layout.addLayout(row2_layout)

        # Load categories from folder
        self._load_categories()

        # Separator line
        separator = QLabel("")
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: #1a3a2a;")
        layout.addWidget(separator)

        # Notes section with fixed headers and editable content areas
        # Agenda section
        agenda_header = QLabel("AGENDA")
        agenda_header.setObjectName("sectionHeader")
        layout.addWidget(agenda_header)

        self.agenda_edit = QTextEdit()
        self.agenda_edit.setMinimumHeight(60)
        self.agenda_edit.setMaximumHeight(120)
        layout.addWidget(self.agenda_edit)

        # Notes section
        notes_header = QLabel("NOTES")
        notes_header.setObjectName("sectionHeader")
        layout.addWidget(notes_header)

        self.notes_edit = QTextEdit()
        layout.addWidget(self.notes_edit, 1)  # stretch factor - takes remaining space

        # Action Items section
        actions_header = QLabel("ACTION ITEMS")
        actions_header.setObjectName("sectionHeader")
        layout.addWidget(actions_header)

        self.actions_edit = QTextEdit()
        self.actions_edit.setMinimumHeight(60)
        self.actions_edit.setMaximumHeight(120)
        layout.addWidget(self.actions_edit)

        # Buttons - compact, right-aligned
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        button_layout.addStretch()  # Push buttons to the right

        self.record_button = QPushButton("● REC")
        self.record_button.setObjectName("recordButton")
        self.record_button.setFixedSize(100, 36)
        self.record_button.clicked.connect(self.toggle_recording)
        button_layout.addWidget(self.record_button)

        exit_button = QPushButton("Exit")
        exit_button.setFixedSize(70, 36)
        exit_button.clicked.connect(self.exit_app)
        button_layout.addWidget(exit_button)

        layout.addLayout(button_layout)

        self.live_window.setLayout(layout)
        self.live_window.show()

        # Check server status
        self._check_server()

    def _check_server(self):
        """Check if transcription server is available."""
        def prettify_model(name: str) -> str:
            """Make model name more readable."""
            # Map common model names to prettier versions
            name_map = {
                "large-v3": "Whisper Large v3",
                "large-v2": "Whisper Large v2",
                "large-v1": "Whisper Large v1",
                "large": "Whisper Large",
                "medium": "Whisper Medium",
                "small": "Whisper Small",
                "base": "Whisper Base",
                "tiny": "Whisper Tiny",
            }
            return name_map.get(name.lower(), name)

        def check():
            available = self.client.is_server_available(force_check=True)

            # Auto-start server if not running (only for localhost)
            if not available and self.server_url.startswith("http://localhost"):
                self.status_changed.emit("Starting server...")
                self.server_label.setText("// Starting Whisper server (this may take a minute)...")
                self.server_label.setStyleSheet("color: #ffaa00; font-size: 13px;")
                _debug_log("[Meeting] Server not running, auto-starting...")

                # This blocks until server is ready (up to 60s)
                if start_server_background():
                    available = True
                    _debug_log("[Meeting] Server auto-started successfully")
                else:
                    _debug_log("[Meeting] Failed to auto-start server")

            if available:
                # Trigger server-side diarization loading by calling /speakers
                # (diarization is lazy-loaded, this ensures it loads before we check status)
                self.client.get_speakers()

                status = self.client.get_status()
                raw_model = status.get("model", "")
                raw_device = status.get("device", "")
                self._server_diarization_available = status.get("diarization_available", False)

                # Handle case where server is up but model not yet loaded
                if not raw_model or raw_model == "unknown":
                    model = "Loading model"
                    device = "..."
                else:
                    model = prettify_model(raw_model)
                    device = raw_device.upper()  # CUDA looks better than cuda

                # Show diarization status
                if self._diarization_loading:
                    diar_status = "Loading..."
                elif self._diarization_available:
                    diar_status = "Diarization: Local"
                elif self._server_diarization_available:
                    diar_status = "Diarization: Server"
                else:
                    diar_status = "Diarization: Off"

                self.status_changed.emit("Ready")
                self.server_label.setText(f"// {model} ({device}) • {diar_status}")
                self.server_label.setStyleSheet("color: #00aa66; font-size: 13px;")
            else:
                self.status_changed.emit("Server offline")
                # Add helpful hint for remote connections
                if not self.server_url.startswith("http://localhost"):
                    self.server_label.setText("// ERROR: server not running (Check desktop is running and Tailscale is connected)")
                else:
                    self.server_label.setText("// ERROR: server not running")
                self.server_label.setStyleSheet("color: #ff4444; font-size: 13px;")
            self._server_checked = True

        threading.Thread(target=check, daemon=True).start()

    def _load_diarization_async(self):
        """Load diarization model in background thread."""
        self._diarization_loading = True
        _debug_log("[Meeting] Starting async diarization load...")

        def load():
            try:
                # First check if server has diarization - if so, skip heavy local loading
                # This saves ~10-30 seconds and ~1.5GB VRAM
                server_status = self.client.get_status()
                if server_status.get("diarization_available", False):
                    _debug_log("[Meeting] Server has diarization - skipping local model load")
                    self._server_diarization_available = True
                    # Still get diarizer for enrollment capability (saving embeddings)
                    # but don't load the heavy models
                    from .diarization import get_diarizer
                    self._diarizer = get_diarizer()
                    return

                # Server doesn't have diarization - load locally
                _debug_log("[Meeting] Server lacks diarization - loading locally...")

                # Lazy import to avoid slow startup (pyannote/torch imports are heavy)
                from .diarization import get_diarizer, PYANNOTE_AVAILABLE

                # Always get diarizer for enrollment capability (saving embeddings)
                self._diarizer = get_diarizer()

                if not PYANNOTE_AVAILABLE:
                    _debug_log("[Meeting] Pyannote not installed. Local diarization disabled, enrollment still available.")
                    return

                _debug_log(f"[Meeting] Diarizer is_available: {self._diarizer.is_available()}")
                if self._diarizer.is_available():
                    _debug_log("[Meeting] Loading diarization model...")
                    self._diarization_available = self._diarizer.load()
                    _debug_log(f"[Meeting] Diarization load result: {self._diarization_available}")
                else:
                    _debug_log("[Meeting] Diarization not configured. Set HF_TOKEN for speaker identification.")
            except Exception as e:
                _debug_log(f"[Meeting] Diarization load error: {e}")
            finally:
                self._diarization_loading = False
                # Update server label to reflect new diarization status
                self._check_server()

        threading.Thread(target=load, daemon=True).start()

    def _check_crash_recovery(self):
        """Check if there's recovery data from a crashed meeting and offer to save it."""
        if not TranscriptWriter.has_recovery_data():
            return

        recovery_info = TranscriptWriter.get_recovery_info()
        if not recovery_info:
            return

        meeting_name = recovery_info.get("meeting_name", "Unknown Meeting")
        entry_count = recovery_info.get("entry_count", 0)
        meeting_start = recovery_info.get("meeting_start", "")

        # Format the start time for display
        try:
            from datetime import datetime
            start_dt = datetime.fromisoformat(meeting_start)
            start_str = start_dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            start_str = meeting_start

        # Show recovery dialog
        msg = QMessageBox(self.live_window)
        msg.setWindowTitle("Recover Meeting Data")
        msg.setText(f"Found unsaved meeting data from a previous session:")
        msg.setInformativeText(
            f"Meeting: {meeting_name}\n"
            f"Started: {start_str}\n"
            f"Entries: {entry_count} transcript entries\n\n"
            f"Would you like to save this data?"
        )
        msg.setIcon(QMessageBox.Question)
        msg.setStandardButtons(QMessageBox.Save | QMessageBox.Discard)
        msg.setDefaultButton(QMessageBox.Save)

        # Apply dark theme
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #0a0a0f;
                color: #00ff88;
            }
            QMessageBox QLabel {
                color: #00ff88;
            }
            QPushButton {
                background-color: #1a1a2e;
                color: #00ff88;
                border: 1px solid #00ff88;
                padding: 8px 16px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #2a2a4e;
            }
        """)

        result = msg.exec_()

        if result == QMessageBox.Save:
            # Load the recovered transcript and save it
            recovered = TranscriptWriter.load_from_recovery()
            if recovered and recovered.entries:
                # Generate recovery filename
                if recovered.meeting_start:
                    date_prefix = recovered.meeting_start.strftime("%y_%m_%d")
                else:
                    date_prefix = datetime.now().strftime("%y_%m_%d")

                safe_name = "".join(c for c in (recovered.meeting_name or "Recovered") if c.isalnum() or c in " _-")
                filename = f"{date_prefix}_{safe_name}_recovered.md"

                # Get output directory from recovery info
                output_dir = recovery_info.get("output_dir")
                if output_dir:
                    recovered.output_dir = Path(output_dir)

                # Save the recovered transcript
                filepath = recovered.save(filename=filename)
                _debug_log(f"[Meeting] Recovered transcript saved to: {filepath}")

                # Show success message
                QMessageBox.information(
                    self.live_window,
                    "Recovery Complete",
                    f"Recovered transcript saved to:\n{filepath}"
                )
            else:
                _debug_log("[Meeting] Recovery file was empty or corrupted")
                TranscriptWriter.delete_recovery_file()
        else:
            # User chose to discard
            _debug_log("[Meeting] User discarded recovery data")
            TranscriptWriter.delete_recovery_file()

    def _get_transcripts_dir(self) -> Path:
        """Get the transcripts directory (root_folder/Transcripts or default)."""
        root_folder = ConfigManager.get_config_value('meeting_options', 'root_folder')
        if root_folder:
            transcripts_dir = Path(root_folder) / "Transcripts"
        else:
            # Default to Meetings/Transcripts in source folder
            transcripts_dir = Path(__file__).parent.parent.parent / "Meetings" / "Transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        return transcripts_dir

    def _load_categories(self):
        """Load category folders from transcripts directory."""
        self.category_combo.clear()

        transcripts_dir = self._get_transcripts_dir()

        # Scan for subfolders (categories)
        categories = []
        try:
            for item in sorted(transcripts_dir.iterdir()):
                if item.is_dir():
                    categories.append(item.name)
        except Exception:
            pass

        if categories:
            # Add categories to dropdown
            for cat in categories:
                self.category_combo.addItem(cat, cat)
        else:
            # No categories yet - allow saving to root
            self.category_combo.addItem("(No categories)", "")

        # Load subcategories for first category
        self._load_subcategories()

    def _on_category_changed(self, index: int):
        """Handle category selection change - update subcategories."""
        self._load_subcategories()

    def _load_subcategories(self):
        """Load subcategory folders for the selected category."""
        self.subcategory_combo.clear()

        selected_category = self.category_combo.currentData()
        if not selected_category:
            self.subcategory_combo.addItem("(None)", "")
            self.subcategory_combo.setEnabled(False)
            self.add_subcategory_button.setEnabled(False)
            return

        self.subcategory_combo.setEnabled(True)
        self.add_subcategory_button.setEnabled(True)

        transcripts_dir = self._get_transcripts_dir()
        category_dir = transcripts_dir / selected_category

        # Scan for subfolders (subcategories)
        subcategories = []
        try:
            for item in sorted(category_dir.iterdir()):
                if item.is_dir():
                    subcategories.append(item.name)
        except Exception:
            pass

        # Always have "(None)" option to save directly in category
        self.subcategory_combo.addItem("(None)", "")

        for subcat in subcategories:
            self.subcategory_combo.addItem(subcat, subcat)

    def _add_new_category(self):
        """Create a new category folder."""
        name, ok = QInputDialog.getText(
            self.live_window,
            "New Category",
            "Category name:",
            QLineEdit.Normal,
            ""
        )

        if not ok or not name.strip():
            return

        name = name.strip()
        # Sanitize folder name
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', name)

        transcripts_dir = self._get_transcripts_dir()
        new_category_dir = transcripts_dir / safe_name

        try:
            new_category_dir.mkdir(parents=True, exist_ok=True)
            self._load_categories()
            # Select the new category
            for i in range(self.category_combo.count()):
                if self.category_combo.itemData(i) == safe_name:
                    self.category_combo.setCurrentIndex(i)
                    break
            self.status_changed.emit(f"Created category: {safe_name}")
        except Exception as e:
            QMessageBox.warning(
                self.live_window,
                "Error",
                f"Could not create category: {e}"
            )

    def _add_new_subcategory(self):
        """Create a new subcategory folder within the selected category."""
        selected_category = self.category_combo.currentData()
        if not selected_category:
            QMessageBox.warning(
                self.live_window,
                "No Category",
                "Please select or create a category first."
            )
            return

        name, ok = QInputDialog.getText(
            self.live_window,
            "New Subcategory",
            f"Subcategory name (in {selected_category}):",
            QLineEdit.Normal,
            ""
        )

        if not ok or not name.strip():
            return

        name = name.strip()
        # Sanitize folder name
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', name)

        transcripts_dir = self._get_transcripts_dir()
        new_subcat_dir = transcripts_dir / selected_category / safe_name

        try:
            new_subcat_dir.mkdir(parents=True, exist_ok=True)
            self._load_subcategories()
            # Select the new subcategory
            for i in range(self.subcategory_combo.count()):
                if self.subcategory_combo.itemData(i) == safe_name:
                    self.subcategory_combo.setCurrentIndex(i)
                    break
            self.status_changed.emit(f"Created subcategory: {selected_category}/{safe_name}")
        except Exception as e:
            QMessageBox.warning(
                self.live_window,
                "Error",
                f"Could not create subcategory: {e}"
            )

    def _get_notes_markdown(self) -> str:
        """Get combined notes markdown from all three editors."""
        sections = []

        agenda = self.agenda_edit.toPlainText().strip()
        if agenda:
            sections.append(f"## Agenda\n\n{agenda}")
        else:
            sections.append("## Agenda\n")

        notes = self.notes_edit.toPlainText().strip()
        if notes:
            sections.append(f"## Notes\n\n{notes}")
        else:
            sections.append("## Notes\n")

        actions = self.actions_edit.toPlainText().strip()
        if actions:
            sections.append(f"## Action Items\n\n{actions}")
        else:
            sections.append("## Action Items\n")

        return "\n\n".join(sections)

    def _set_notes_content(self, agenda: str = "", notes: str = "", actions: str = ""):
        """Set content in the three note editors."""
        self.agenda_edit.setPlainText(agenda)
        self.notes_edit.setPlainText(notes)
        self.actions_edit.setPlainText(actions)

    def _clear_notes(self):
        """Clear all note editors."""
        self.agenda_edit.clear()
        self.notes_edit.clear()
        self.actions_edit.clear()

    def _new_meeting(self):
        """Reset UI for a new meeting."""
        # Clear all fields
        self.meeting_name_input.clear()
        self._clear_notes()

        # Reset dropdowns to defaults
        if self.category_combo.count() > 0:
            self.category_combo.setCurrentIndex(0)
        if self.subcategory_combo.count() > 0:
            self.subcategory_combo.setCurrentIndex(0)

        # Reset max speakers to default (5)

        # Clear any opened file state
        self._opened_filepath = None
        self._existing_transcript_entries = []

        # Reset recording state
        self._recording = False
        if self._timer:
            self._timer.stop()
            self._timer = None
        if hasattr(self, '_blink_timer') and self._blink_timer:
            self._blink_timer.stop()
            self._blink_timer = None

        # Reset timer display
        self.time_label.setText("00:00")

        # Reset and show record button
        self.record_button.setText("● REC")
        self.record_button.setEnabled(True)
        self.record_button.setStyleSheet("""
            background-color: #2a1a1a;
            border: 1px solid #ff4444;
            border-radius: 4px;
            padding: 12px 24px;
            font-size: 13px;
            font-weight: 600;
            color: #ff6666;
            font-family: 'Cascadia Code', 'Consolas', monospace;
        """)
        self.record_button.show()

        # Hide recording indicator
        self.recording_indicator.hide()

        # Enable all inputs
        self._enable_inputs()

        # Show form fields again
        self._show_form_fields()

        self.add_category_button.show()
        self.add_subcategory_button.show()
        self.open_button.show()
        self.save_button.show()

        # Enable text editors
        self.agenda_edit.setEnabled(True)
        self.notes_edit.setEnabled(True)
        self.actions_edit.setEnabled(True)

        # Update status
        self.status_label.setText("> Ready")
        self.status_label.setStyleSheet("font-size: 22px; font-weight: 600; color: #00ff88;")

        # Reset transcript for new meeting
        self.transcript = TranscriptWriter()

        # Focus on meeting name for quick entry
        self.meeting_name_input.setFocus()

    def _open_existing_meeting(self):
        """Open an existing agenda file (notes only, not transcripts)."""
        transcripts_dir = self._get_transcripts_dir()

        filepath, _ = QFileDialog.getOpenFileName(
            self.live_window,
            "Open Agenda",
            str(transcripts_dir),
            "Markdown files (*.md)"
        )

        if not filepath:
            return

        filepath = Path(filepath)

        # Check if this is a transcript file (has date prefix YY_MM_DD_)
        # Agenda files don't have date prefix
        date_prefix_pattern = re.compile(r'^\d{2}_\d{2}_\d{2}_')
        if date_prefix_pattern.match(filepath.name):
            QMessageBox.warning(
                self.live_window,
                "Cannot Open Transcript",
                "This appears to be a completed transcript (has date prefix).\n\n"
                "You can only open agenda/notes files that were saved before recording."
            )
            return

        # Parse the file
        try:
            content = filepath.read_text(encoding='utf-8')
        except Exception as e:
            QMessageBox.warning(
                self.live_window,
                "Error",
                f"Could not read file: {e}"
            )
            return

        parsed = self._parse_meeting_file(content)

        # Set meeting name from file
        if parsed['meeting_name']:
            self.meeting_name_input.setText(parsed['meeting_name'])

        # Determine category and subcategory from file path
        # Path could be: Transcripts/Category/file.md or Transcripts/Category/Subcategory/file.md
        transcripts_dir = self._get_transcripts_dir()
        rel_path = filepath.relative_to(transcripts_dir) if filepath.is_relative_to(transcripts_dir) else None

        if rel_path and len(rel_path.parts) >= 2:
            # At least in a category folder
            category = rel_path.parts[0]
            for i in range(self.category_combo.count()):
                if self.category_combo.itemData(i) == category:
                    self.category_combo.setCurrentIndex(i)
                    break

            # Check for subcategory (3 parts: category/subcategory/file.md)
            if len(rel_path.parts) >= 3:
                subcategory = rel_path.parts[1]
                # Subcategories are loaded via _on_category_changed, give it a moment
                self._load_subcategories()
                for i in range(self.subcategory_combo.count()):
                    if self.subcategory_combo.itemData(i) == subcategory:
                        self.subcategory_combo.setCurrentIndex(i)
                        break

        # Set notes in editors
        self._set_notes_content(
            agenda=parsed.get('agenda', ''),
            notes=parsed.get('notes', ''),
            actions=parsed.get('actions', '')
        )

        # Store existing transcript entries for appending (shouldn't have any for agenda files)
        self._existing_transcript_entries = parsed['transcript_entries']
        self._opened_filepath = filepath

        self.status_changed.emit(f"Opened: {filepath.name}")

    def _parse_meeting_file(self, content: str) -> dict:
        """Parse an existing meeting markdown file."""
        result = {
            'meeting_name': '',
            'agenda': '',
            'notes': '',
            'actions': '',
            'transcript_entries': []
        }

        lines = content.split('\n')

        # Extract meeting name from title (# Title)
        for line in lines:
            if line.startswith('# '):
                result['meeting_name'] = line[2:].strip()
                break

        # Find section boundaries
        agenda_start = -1
        notes_start = -1
        actions_start = -1
        transcript_start = -1

        for i, line in enumerate(lines):
            line_lower = line.strip().lower()
            if line_lower == '## agenda':
                agenda_start = i + 1
            elif line_lower == '## notes':
                notes_start = i + 1
            elif line_lower in ('## action items', '## actions'):
                actions_start = i + 1
            elif line_lower == '## full transcript':
                transcript_start = i
                break

        # Helper to extract content between sections
        def extract_section(start: int, end_markers: list) -> str:
            if start < 0:
                return ''
            end = len(lines)
            for marker in end_markers:
                if marker >= 0 and marker > start:
                    end = min(end, marker - 1)  # -1 to exclude the header line
                    break
            section_lines = lines[start:end]
            # Strip empty lines and separators
            section_lines = [l for l in section_lines if l.strip() and l.strip() != '---']
            return '\n'.join(section_lines).strip()

        # Extract each section
        result['agenda'] = extract_section(agenda_start, [notes_start, actions_start, transcript_start])
        result['notes'] = extract_section(notes_start, [actions_start, transcript_start])
        result['actions'] = extract_section(actions_start, [transcript_start])

        # Parse transcript entries (for appending)
        if transcript_start > 0:
            # Format: **[MM:SS] Speaker**: Text
            entry_pattern = r'\*\*\[(\d+:\d+(?::\d+)?)\] ([^*]+)\*\*: (.+)'
            for line in lines[transcript_start:]:
                match = re.match(entry_pattern, line)
                if match:
                    result['transcript_entries'].append({
                        'timestamp': match.group(1),
                        'speaker': match.group(2),
                        'text': match.group(3)
                    })

        return result

    def _save_notes_only(self):
        """Save notes without transcript (for pre-meeting agenda prep)."""
        meeting_name = self.meeting_name_input.text().strip()
        if not meeting_name:
            QMessageBox.warning(
                self.live_window,
                "Meeting Name Required",
                "Please enter a meeting name before saving."
            )
            self.meeting_name_input.setFocus()
            return

        # Get output directory (category/subcategory)
        transcripts_dir = self._get_transcripts_dir()
        selected_category = self.category_combo.currentData()
        selected_subcategory = self.subcategory_combo.currentData() if self.subcategory_combo.isEnabled() else ""

        if selected_category:
            output_dir = transcripts_dir / selected_category
            if selected_subcategory:
                output_dir = output_dir / selected_subcategory
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = transcripts_dir

        # Generate filename WITHOUT date prefix (for agenda files)
        safe_name = sanitize_filename(meeting_name)
        filename = f"{safe_name}.md"
        filepath = output_dir / filename

        # Get notes from editors
        notes_md = self._get_notes_markdown()

        # Generate content (notes only, no transcript)
        content_lines = [
            f"# {meeting_name}",
            "",
            f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Duration**: 0 minutes",
            f"**Participants**: ",
            "",
            "---",
            "",
            notes_md,
            ""
        ]

        filepath.write_text('\n'.join(content_lines), encoding='utf-8')

        # Build save message with full path
        if selected_category:
            if selected_subcategory:
                save_msg = f"Saved to {selected_category}/{selected_subcategory}/{filename}"
            else:
                save_msg = f"Saved to {selected_category}/{filename}"
        else:
            save_msg = f"Saved: {filename}"
        self.status_changed.emit(save_msg)

    def toggle_recording(self):
        """Toggle recording on/off."""
        if self._recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        """Start recording a meeting (with async heavy work)."""
        if self._recording:
            return

        # Validate inputs (instant - keep on main thread)
        meeting_name = self.meeting_name_input.text().strip()
        if not meeting_name:
            QMessageBox.warning(
                self.live_window,
                "Meeting Name Required",
                "Please enter a meeting name before starting."
            )
            self.meeting_name_input.setFocus()
            return

        selected_category = self.category_combo.currentData()
        has_categories = self.category_combo.count() > 0 and self.category_combo.itemData(0) != ""
        if has_categories and not selected_category:
            QMessageBox.warning(
                self.live_window,
                "Category Required",
                "Please select a category folder."
            )
            return

        # Change UI immediately (no processEvents needed - truly instant now)
        self.record_button.setText("...")
        self.record_button.setEnabled(False)
        self.status_changed.emit("Checking server...")

        # Disable inputs immediately
        self.meeting_name_input.setEnabled(False)
        self.category_combo.setEnabled(False)
        self.subcategory_combo.setEnabled(False)

        # Do heavy work in background thread
        threading.Thread(target=self._start_recording_async, daemon=True).start()

    def _start_recording_async(self):
        """Background thread: do heavy startup work."""
        try:
            # Server check (this is the slow HTTP call)
            if not self.client.is_server_available(force_check=True):
                # Failed - update UI on main thread
                self.status_changed.emit("Server offline")
                QTimer.singleShot(0, self._show_server_error)
                return

            self.status_changed.emit("Initializing...")

            # Reset speaker tracking
            if self._diarizer:
                self._diarizer.reset_session()
            if self._server_diarization_available:
                self.client.reset_diarization()
            self._speaker_counter = 0
            self._speaker_map.clear()
            self._consecutive_failures = 0  # Reset circuit breaker

            # Setup transcript
            self.transcript.start_meeting()
            meeting_name = self.meeting_name_input.text().strip()
            if meeting_name:
                self.transcript.meeting_name = meeting_name
            self._meeting_start_time = time.time()

            # Preload existing transcript entries if any
            if self._existing_transcript_entries:
                for entry in self._existing_transcript_entries:
                    ts_parts = entry['timestamp'].split(':')
                    if len(ts_parts) == 2:
                        secs = int(ts_parts[0]) * 60 + int(ts_parts[1])
                    else:
                        secs = int(ts_parts[0]) * 3600 + int(ts_parts[1]) * 60 + int(ts_parts[2])
                    self.transcript.add_entry(secs, entry['speaker'], entry['text'])
                self._existing_transcript_entries = []

            # Start audio capture on main thread
            QTimer.singleShot(0, self._finalize_start_recording)

        except Exception as e:
            _debug_log(f"[Meeting] Error in _start_recording_async: {e}")
            self.status_changed.emit("Error starting")
            QTimer.singleShot(0, self._reset_start_button)

    def _finalize_start_recording(self):
        """Finalize recording start on main thread (for Qt operations)."""
        self.status_changed.emit("Starting audio capture...")

        # Setup audio processor
        self.processor = AudioProcessor(
            sample_rate=16000,
            min_chunk_duration=15.0,
            max_chunk_duration=30.0,
            silence_duration_ms=800,
            on_chunk_ready=self._on_chunk_ready
        )

        if not self.processor.start():
            self.record_button.setText("● REC")
            self.record_button.setEnabled(True)
            self._enable_inputs()
            self.status_changed.emit("Audio error")
            QMessageBox.warning(
                self.live_window,
                "Error",
                "Failed to start audio capture. Check microphone permissions."
            )
            return

        # Success - update UI to recording state
        self._recording = True
        self.record_button.setText("■ STOP")
        self.record_button.setEnabled(True)
        self.record_button.setStyleSheet("""
            background-color: #1a2a1a;
            border: 1px solid #00ff88;
            border-radius: 4px;
            padding: 12px 24px;
            font-size: 13px;
            font-weight: 600;
            color: #00ff88;
            font-family: 'Cascadia Code', 'Consolas', monospace;
        """)

        # Hide buttons
        self.add_category_button.hide()
        self.add_subcategory_button.hide()
        self.open_button.hide()
        self.save_button.hide()

        # Hide form fields (not editable during recording)
        self._hide_form_fields()

        # Start timer
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_time)
        self._timer.start(1000)

        # Show recording indicator with smooth pulsating animation
        self._rec_opacity = 1.0
        self._rec_fade_direction = -1
        self.recording_indicator.show()
        self._blink_timer = QTimer()
        self._blink_timer.timeout.connect(self._blink_recording_indicator)
        self._blink_timer.start(50)  # Update every 50ms for smooth pulsating

        self._chunks_processed = 0
        self.status_changed.emit("Recording...")

    def _show_server_error(self):
        """Show server error dialog on main thread."""
        self.record_button.setText("● REC")
        self.record_button.setEnabled(True)
        self._enable_inputs()
        self.status_changed.emit("Server offline")
        QMessageBox.warning(
            self.live_window,
            "Server Not Running",
            "The transcription server is not running.\n\n"
            "Start it with:\n"
            "python -m src.server\n\n"
            "Or run 'Start Whisper Server.bat'"
        )

    def _reset_start_button(self):
        """Reset button after error on main thread."""
        self.record_button.setText("● REC")
        self.record_button.setEnabled(True)
        self._enable_inputs()
        self.status_changed.emit("Error")

    def _enable_inputs(self):
        """Helper to re-enable form inputs."""
        self.meeting_name_input.setEnabled(True)
        self.category_combo.setEnabled(True)
        self.subcategory_combo.setEnabled(True)

    def _hide_form_fields(self):
        """Hide form fields when recording (since they're not editable during recording)."""
        # Hide all widgets in each layout
        for layout in [self.name_layout, self.category_layout, self.subcategory_layout]:
            for i in range(layout.count()):
                widget = layout.itemAt(i).widget()
                if widget:
                    widget.hide()

    def _show_form_fields(self):
        """Show form fields again when not recording."""
        # Show all widgets in each layout
        for layout in [self.name_layout, self.category_layout, self.subcategory_layout]:
            for i in range(layout.count()):
                widget = layout.itemAt(i).widget()
                if widget:
                    widget.show()

    def stop_recording(self):
        """Stop recording (with async heavy work)."""
        if not self._recording:
            return

        self._recording = False

        # Stop timers immediately
        if self._timer:
            self._timer.stop()
            self._timer = None
        if hasattr(self, '_blink_timer') and self._blink_timer:
            self._blink_timer.stop()
            self._blink_timer = None

        # Hide recording indicator
        self.recording_indicator.hide()

        # Store meeting info for summary window
        self._last_meeting_name = self.meeting_name_input.text().strip()
        self._last_meeting_duration = int(time.time() - self._meeting_start_time)

        # Hide main window and show summary window IMMEDIATELY
        self.live_window.hide()

        # Show small summary status window
        self._summary_window = SummaryStatusWindow(
            self._last_meeting_name,
            self._last_meeting_duration
        )
        self._summary_window.closeSignal.connect(self._close_summary_window)
        self._summary_window.newMeetingSignal.connect(self._new_meeting_from_summary)
        self._summary_window.linkClickedSignal.connect(self._open_file)
        self._summary_window.startSummarizationSignal.connect(self._start_summarization_after_enrollment)

        # Connect status signal for thread-safe updates
        self.summary_status_changed.connect(self._summary_window.update_status)

        self._summary_window.show()

        # Initial status in summary window
        self.summary_status_changed.emit("Stopping...")

        # Do heavy work in background
        threading.Thread(target=self._stop_recording_async, daemon=True).start()

    def _stop_recording_async(self):
        """Background thread: do heavy stop work."""
        try:
            _debug_log("[Meeting] Background stop started")

            # Process final audio chunk (slow - transcription + diarization)
            self.summary_status_changed.emit("Processing final audio...")
            _debug_log("[Meeting] Processing final audio chunk")

            if self.processor:
                final_chunk = self.processor.stop()
                if final_chunk:
                    self._process_chunk(final_chunk)
                self.processor = None

            _debug_log("[Meeting] Final audio processed")

            # Retry any failed audio chunks from this session (server may have recovered)
            self._retry_failed_chunks()

            # Save transcript (disk I/O)
            self.summary_status_changed.emit("Saving transcript...")
            _debug_log("[Meeting] Saving transcript")

            # Determine output directory
            transcripts_dir = self._get_transcripts_dir()
            selected_category = self.category_combo.currentData()
            selected_subcategory = self.subcategory_combo.currentData() if self.subcategory_combo.isEnabled() else ""

            if selected_category:
                output_dir = transcripts_dir / selected_category
                if selected_subcategory:
                    output_dir = output_dir / selected_subcategory
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir = transcripts_dir

            self.transcript.output_dir = output_dir

            # Get notes and save
            notes_md = self._get_notes_markdown()
            meeting_name = self.meeting_name_input.text().strip()
            date_prefix = self.transcript.meeting_start.strftime("%y_%m_%d") if self.transcript.meeting_start else datetime.now().strftime("%y_%m_%d")

            safe_name = sanitize_filename(meeting_name)
            filename = f"{date_prefix}_{safe_name}.md"
            filepath = self.transcript.save(filename=filename, notes_markdown=notes_md)

            _debug_log(f"[Meeting] Transcript saved to: {filepath}")

            # Capture session speakers for potential enrollment (before any reset)
            # This must happen BEFORE diarizer.reset_session() which is called on next meeting start
            unknown_speakers = {}
            speaker_samples = {}
            speaker_merges = {}  # Track consolidated speakers for transcript rewriting

            if self._diarizer and self._diarization_available:
                # Local diarization mode - consolidate similar speakers first
                _debug_log("[Meeting] Consolidating session speakers...")
                speaker_merges = self._diarizer.consolidate_session_speakers(similarity_threshold=0.40)

                # Rewrite transcript entries with merged speaker names
                if speaker_merges:
                    merged_count = 0
                    for entry in self.transcript.entries:
                        if entry.speaker in speaker_merges:
                            old_speaker = entry.speaker
                            entry.speaker = speaker_merges[old_speaker]
                            merged_count += 1
                    if merged_count > 0:
                        _debug_log(f"[Meeting] Rewrote {merged_count} transcript entries after speaker consolidation")
                        # Re-save transcript with consolidated speakers
                        self.transcript.save(filename=filename, notes_markdown=notes_md)
                        _debug_log(f"[Meeting] Transcript re-saved with consolidated speakers")

                # Now get remaining unenrolled speakers
                unknown_speakers = self._diarizer.get_unenrolled_session_speakers()
            elif self._server_diarization_available:
                # Server diarization mode - fetch from server
                _debug_log("[Meeting] Fetching unenrolled speakers from server...")
                unknown_speakers = self.client.get_unenrolled_speakers()

            if unknown_speakers:
                _debug_log(f"[Meeting] Found {len(unknown_speakers)} unknown speakers from diarization")
                # Get ALL transcriptions for each unknown speaker (for better identification)
                for entry in self.transcript.entries:
                    if entry.speaker in unknown_speakers:
                        if entry.speaker not in speaker_samples:
                            speaker_samples[entry.speaker] = []
                        # Keep ALL samples (dialog will display them in scrollable area)
                        # Include timestamp for context
                        timestamp = f"[{entry.timestamp}]" if hasattr(entry, 'timestamp') else ""
                        sample = f"{timestamp} {entry.text}".strip()
                        speaker_samples[entry.speaker].append(sample)

                # Filter out speakers with no transcript entries (likely hallucinations)
                # A speaker can exist in diarization but have no transcribed text if pyannote
                # detected something (noise, echo) but Whisper produced no text for it
                speakers_with_entries = {k: v for k, v in unknown_speakers.items() if k in speaker_samples}
                if len(speakers_with_entries) < len(unknown_speakers):
                    filtered_out = set(unknown_speakers.keys()) - set(speakers_with_entries.keys())
                    _debug_log(f"[Meeting] Filtered out {len(filtered_out)} phantom speakers with no text: {filtered_out}")
                unknown_speakers = speakers_with_entries

                # Pass enrollment data to summary window (must do on main thread via signal)
                if unknown_speakers:
                    _debug_log(f"[Meeting] {len(unknown_speakers)} unknown speakers for enrollment")
                    self.enrollment_data_ready.emit(unknown_speakers, speaker_samples, filepath)

            # Delete original agenda file if any
            if self._opened_filepath and self._opened_filepath.exists():
                try:
                    if self._opened_filepath != filepath:
                        self._opened_filepath.unlink()
                        _debug_log(f"[Meeting] Deleted original agenda file: {self._opened_filepath.name}")
                except Exception as e:
                    _debug_log(f"[Meeting] Warning: Could not delete original file: {e}")
                self._opened_filepath = None

            # Check if we have unknown speakers - if so, delay summarization
            if unknown_speakers:
                # Delay summarization until after enrollment dialog closes
                # The dialog's closedSignal will trigger summarization
                _debug_log(f"[Meeting] {len(unknown_speakers)} unknown speakers - delaying summary until after enrollment")
                self.summary_status_changed.emit("Waiting for speaker enrollment...")
                # Summarization will be triggered by SummaryStatusWindow.startSummarizationSignal
            else:
                # No unknown speakers - start summarization immediately
                _debug_log("[Meeting] All speakers known - starting summary immediately")
                self.summary_status_changed.emit("Generating summary...")
                self._spawn_summarization_subprocess(filepath)
                # Emit signal to start polling on main thread (MUST use signal from background thread)
                _debug_log("[Meeting] Emitting signal to start polling on main thread")
                self.start_summary_polling.emit(filepath)

            _debug_log("[Meeting] Background stop complete")

        except Exception as e:
            _debug_log(f"[Meeting] Error in _stop_recording_async: {e}")
            import traceback
            _debug_log(f"[Meeting] Traceback: {traceback.format_exc()}")
            self.summary_status_changed.emit("Error saving")
            QTimer.singleShot(0, self._reset_stop_button)

    def _reset_stop_button(self):
        """Reset button after error - show main window again."""
        # Close summary window if it exists
        if self._summary_window:
            self._summary_window.close()
            self._summary_window = None

        # Show main window
        self.live_window.show()

        # Reset UI
        self.record_button.setText("● REC")
        self.record_button.setEnabled(True)
        self.record_button.setStyleSheet("""
            background-color: #2a1a1a;
            border: 1px solid #ff4444;
            border-radius: 4px;
            padding: 12px 24px;
            font-size: 13px;
            font-weight: 600;
            color: #ff6666;
            font-family: 'Cascadia Code', 'Consolas', monospace;
        """)
        self._enable_inputs()
        self._show_form_fields()
        self.add_category_button.show()
        self.add_subcategory_button.show()
        self.open_button.show()
        self.save_button.show()
        self.status_changed.emit("Error")

    def _on_chunk_ready(self, chunk: AudioChunk):
        """Handle a new audio chunk (runs in audio thread)."""
        thread = threading.Thread(
            target=self._process_chunk_serialized,
            args=(chunk,),
            daemon=True
        )
        thread.start()

    def _process_chunk_serialized(self, chunk: AudioChunk):
        """Serialize chunk processing to prevent request storms.

        Only one chunk can be transcribed at a time. If the server is overwhelmed
        (5+ consecutive failures), new chunks are saved directly as failed audio
        instead of hammering the server.
        """
        # Circuit breaker: if server has failed 5+ times in a row, save audio and skip
        if self._consecutive_failures >= 5:
            _debug_log(f"[Meeting] Circuit breaker OPEN ({self._consecutive_failures} consecutive failures) - saving chunk as failed audio")
            self._save_chunk_as_failed(chunk)
            return

        # Wait for previous chunk to finish (serialize transcription)
        acquired = self._chunk_semaphore.acquire(timeout=120)
        if not acquired:
            _debug_log("[Meeting] Chunk timed out waiting for semaphore (120s) - saving as failed audio")
            self._save_chunk_as_failed(chunk)
            return

        try:
            self._process_chunk(chunk)
        finally:
            self._chunk_semaphore.release()

    def _save_chunk_as_failed(self, chunk: AudioChunk):
        """Save a chunk's audio to failed_audio files for later retry."""
        if chunk.mic_audio is not None and len(chunk.mic_audio) > 0:
            self.client._save_failed_audio(chunk.mic_audio, 16000, "circuit_breaker")
        if chunk.loopback_audio is not None and len(chunk.loopback_audio) > 0:
            self.client._save_failed_audio(chunk.loopback_audio, chunk.loopback_sample_rate, "circuit_breaker")

    def _retry_failed_chunks(self):
        """Retry transcription of any failed audio chunks from this session.

        Called when stopping a meeting - the server may have recovered from earlier failures.
        Bails out early if the first retry fails (server is still down).
        """
        # Get failed audio files from the last 60 minutes
        failed_files = self.client.get_failed_audio_files(max_age_minutes=60)

        if not failed_files:
            _debug_log("[Meeting] No failed audio chunks to retry")
            return

        _debug_log(f"[Meeting] Found {len(failed_files)} failed audio chunks to retry")

        # Get transcription settings
        model_options = ConfigManager.get_config_section('model_options')
        language = model_options['common'].get('language') or 'en'
        initial_prompt = model_options['common'].get('initial_prompt')

        retried = 0
        succeeded = 0
        consecutive_failures = 0

        for audio_path in failed_files:
            retried += 1
            self.summary_status_changed.emit(f"Retrying failed audio ({retried}/{len(failed_files)})...")

            result, success = self.client.retry_failed_audio(
                audio_path, language=language, initial_prompt=initial_prompt
            )

            if success and result.strip():
                succeeded += 1
                consecutive_failures = 0
                # Add to transcript as "Recovered" speaker (we don't know who it was)
                # Use current time as timestamp since we don't know original timestamp
                current_time = time.time() - self._meeting_start_time if self._meeting_start_time else 0
                text = post_process_text(result)
                if text:
                    self.transcription_ready.emit("(Recovered)", text, current_time)
                    _debug_log(f"[Meeting] Recovered chunk: {text[:50]}...")
            else:
                consecutive_failures += 1
                # Bail out early if server is still unresponsive (2 consecutive failures)
                if consecutive_failures >= 2:
                    remaining = len(failed_files) - retried
                    _debug_log(f"[Meeting] Aborting retry - server still unresponsive after {consecutive_failures} consecutive failures ({remaining} files skipped)")
                    self.summary_status_changed.emit("Server unresponsive, skipping retries")
                    break

        if succeeded > 0:
            _debug_log(f"[Meeting] Recovered {succeeded}/{retried} failed chunks")
        else:
            _debug_log(f"[Meeting] No chunks recovered from {retried} retry attempts")

    def _process_chunk(self, chunk: AudioChunk):
        """Process an audio chunk through transcription."""
        timestamp = chunk.timestamp - self._meeting_start_time
        chunk_had_success = False

        try:
            # Get language from config (force English to prevent hallucinations)
            model_options = ConfigManager.get_config_section('model_options')
            language = model_options['common'].get('language') or 'en'  # Default to English
            initial_prompt = model_options['common'].get('initial_prompt')

            # Transcribe mic audio (user) - only if it has actual speech
            if chunk.mic_audio is not None and len(chunk.mic_audio) > 0:
                if check_audio_has_speech(chunk.mic_audio, threshold=300):
                    self._chunks_processed += 1
                    # Use diarization for mic audio when available (timing consistency + embedding extraction)
                    if self._diarization_available and self._diarizer:
                        # Local diarization: run through same pipeline as loopback for timing consistency
                        # Also extracts + updates user's embedding for adaptive learning
                        _debug_log("[Mic] Using LOCAL diarization for timing consistency")
                        self._process_mic_with_diarization(chunk.mic_audio, 16000, timestamp)
                        chunk_had_success = True
                    else:
                        # No diarization: fall back to direct transcription
                        _debug_log("[Mic] Using direct transcription (no diarization)")
                        text, success = self.client.transcribe(chunk.mic_audio, sample_rate=16000, language=language, initial_prompt=initial_prompt)
                        if success:
                            chunk_had_success = True
                            text = post_process_text(text)
                            if text:
                                self.transcription_ready.emit(self.user_name, text, timestamp)
                        else:
                            self.status_changed.emit(f"Mic error: {text[:30]}")

            # Transcribe loopback audio (others) - only if it has actual speech
            if chunk.loopback_audio is not None and len(chunk.loopback_audio) > 0:
                target_rate = 16000

                # High-quality preprocessing: stereo→mono, resample, normalize
                loopback = preprocess_loopback_audio(
                    audio=chunk.loopback_audio,
                    channels=chunk.loopback_channels,
                    source_rate=chunk.loopback_sample_rate,
                    target_rate=target_rate,
                    target_rms=3000.0  # ~-20dB, optimal for Whisper
                )

                # Only transcribe if there's actual speech in loopback
                if check_audio_has_speech(loopback, threshold=300):
                    # Use diarization to identify speakers if available
                    _debug_log(f"[Loopback] _diarization_available={self._diarization_available}, _server_diarization_available={self._server_diarization_available}")
                    if self._diarization_available and self._diarizer:
                        # Use local diarization (desktop mode)
                        _debug_log("[Loopback] Using LOCAL diarization path")
                        self._process_loopback_with_diarization(loopback, target_rate, timestamp)
                        chunk_had_success = True
                    elif self._server_diarization_available:
                        # Use server-side diarization (remote mode)
                        _debug_log("[Loopback] Using SERVER diarization path")
                        self._process_loopback_with_server_diarization(loopback, target_rate, timestamp)
                        chunk_had_success = True
                    else:
                        # Fallback: just label as "Other"
                        _debug_log("[Loopback] Using fallback 'Other' path - NO DIARIZATION AVAILABLE")
                        text, success = self.client.transcribe(loopback, sample_rate=target_rate, language=language, initial_prompt=initial_prompt)
                        if success:
                            chunk_had_success = True
                            text = post_process_text(text)
                            if text:
                                self.transcription_ready.emit("Other", text, timestamp)
        except Exception as e:
            _debug_log(f"[Meeting] Exception in _process_chunk: {e}")
            chunk_had_success = False
            # Save audio so it's not lost
            self._save_chunk_as_failed(chunk)

        # Update circuit breaker based on chunk outcome
        if chunk_had_success:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 5:
                _debug_log(f"[Meeting] Circuit breaker OPENED after {self._consecutive_failures} consecutive failures - will save audio instead of sending to server")

    def _process_loopback_with_diarization(self, audio: np.ndarray, sample_rate: int, base_timestamp: float):
        """Process loopback audio with speaker diarization."""
        # Hardcoded max_speakers - auto-merge handles speaker consolidation
        max_speakers = 4

        # Get language from config (force English to prevent hallucinations)
        model_options = ConfigManager.get_config_section('model_options')
        language = model_options['common'].get('language') or 'en'  # Default to English
        initial_prompt = model_options['common'].get('initial_prompt')

        # Run diarization to get speaker segments
        _debug_log(f"[Diarize] Running on {len(audio)} samples ({len(audio)/sample_rate:.1f}s), max_speakers={max_speakers}, language={language}")
        segments = self._diarizer.diarize(audio, sample_rate=sample_rate, max_speakers=max_speakers)
        _debug_log(f"[Diarize] Returned {len(segments)} segments")

        if not segments:
            # Fallback if diarization fails
            _debug_log("[Diarize] No segments found, falling back to 'Other'")
            text, success = self.client.transcribe(audio, sample_rate=sample_rate, language=language, initial_prompt=initial_prompt)
            if success:
                text = post_process_text(text)
                if text:
                    self.transcription_ready.emit("Other", text, base_timestamp)
            return

        # Process each speaker segment
        for i, segment in enumerate(segments):
            # Extract audio for this segment
            start_sample = int(segment.start * sample_rate)
            end_sample = int(segment.end * sample_rate)
            segment_audio = audio[start_sample:end_sample]

            # Speaker label is already consistent from diarizer (embedding-matched)
            speaker_name = segment.speaker
            _debug_log(f"[Segment {i}] {speaker_name} ({segment.start:.1f}s-{segment.end:.1f}s)")

            if len(segment_audio) < sample_rate * 0.5:  # Skip segments < 0.5s
                _debug_log(f"[Segment {i}] Skipping - too short")
                continue

            # Transcribe segment
            text, success = self.client.transcribe(segment_audio, sample_rate=sample_rate, language=language, initial_prompt=initial_prompt)
            _debug_log(f"[Transcribe] success={success}, raw_text='{text[:100] if text else '(empty)'}...'")
            if not success:
                _debug_log(f"[Transcribe] FAILED: {text}")  # text contains error message on failure
            if success:
                original_text = text
                text = post_process_text(text)
                if text:
                    segment_timestamp = base_timestamp + segment.start
                    _debug_log(f"[Result] {speaker_name}: {text[:50]}...")
                    self.transcription_ready.emit(speaker_name, text, segment_timestamp)
                else:
                    _debug_log(f"[Transcribe] Text filtered out by post_process. Original: '{original_text[:100]}'")

    def _process_loopback_with_server_diarization(self, audio: np.ndarray, sample_rate: int, base_timestamp: float):
        """Process loopback audio using server-side diarization (for remote mode)."""
        # Hardcoded max_speakers - auto-merge handles speaker consolidation
        max_speakers = 4

        # Get language from config (force English to prevent hallucinations)
        model_options = ConfigManager.get_config_section('model_options')
        language = model_options['common'].get('language') or 'en'  # Default to English
        initial_prompt = model_options['common'].get('initial_prompt')

        _debug_log(f"[ServerDiarize] Running on {len(audio)} samples ({len(audio)/sample_rate:.1f}s), max_speakers={max_speakers}, language={language}")

        # Call server's combined diarization + transcription endpoint
        segments, success = self.client.transcribe_meeting(
            audio,
            sample_rate=sample_rate,
            language=language,
            initial_prompt=initial_prompt,
            max_speakers=max_speakers,
            user_name=None  # Don't set user_name for loopback (handled separately)
        )

        if not success or not segments:
            # Fallback to simple transcription
            _debug_log("[ServerDiarize] Failed or no segments, falling back to 'Other'")
            text, success = self.client.transcribe(audio, sample_rate=sample_rate, language=language, initial_prompt=initial_prompt)
            if success:
                text = post_process_text(text)
                if text:
                    self.transcription_ready.emit("Other", text, base_timestamp)
            return

        _debug_log(f"[ServerDiarize] Received {len(segments)} segments")

        # Process each segment from the server
        for seg in segments:
            text = post_process_text(seg.text)
            if text:
                segment_timestamp = base_timestamp + seg.start
                _debug_log(f"[ServerDiarize] {seg.speaker}: {text[:50]}...")
                self.transcription_ready.emit(seg.speaker, text, segment_timestamp)

    def _process_mic_with_diarization(self, audio: np.ndarray, sample_rate: int, base_timestamp: float):
        """
        Process mic audio with diarization for timing consistency and embedding extraction.

        Unlike loopback, mic audio is always from the user. We run diarization to:
        1. Ensure timing consistency with loopback (both through same pipeline)
        2. Extract user's embedding for adaptive learning
        3. Get timing segments for proper transcription
        """
        # Get language from config
        model_options = ConfigManager.get_config_section('model_options')
        language = model_options['common'].get('language') or 'en'
        initial_prompt = model_options['common'].get('initial_prompt')

        _debug_log(f"[UserMic] Processing {len(audio)} samples ({len(audio)/sample_rate:.1f}s) for {self.user_name}")

        # Run diarization for embedding extraction + timing
        segments = self._diarizer.extract_user_embedding(audio, self.user_name, sample_rate)

        if not segments:
            # Fallback: transcribe entire audio
            _debug_log("[UserMic] No segments, transcribing full audio")
            text, success = self.client.transcribe(audio, sample_rate=sample_rate, language=language, initial_prompt=initial_prompt)
            if success:
                text = post_process_text(text)
                if text:
                    self.transcription_ready.emit(self.user_name, text, base_timestamp)
            return

        # Transcribe each segment
        for segment in segments:
            start_sample = int(segment.start * sample_rate)
            end_sample = int(segment.end * sample_rate)
            segment_audio = audio[start_sample:end_sample]

            if len(segment_audio) < sample_rate * 0.5:  # Skip segments < 0.5s
                _debug_log(f"[UserMic] Skipping short segment ({len(segment_audio)/sample_rate:.2f}s)")
                continue

            text, success = self.client.transcribe(segment_audio, sample_rate=sample_rate, language=language, initial_prompt=initial_prompt)
            _debug_log(f"[UserMic Transcribe] success={success}, raw_text='{text[:100] if text else '(empty)'}...'")
            if not success:
                _debug_log(f"[UserMic Transcribe] FAILED: {text}")  # text contains error message on failure
            if success:
                original_text = text
                text = post_process_text(text)
                if text:
                    segment_timestamp = base_timestamp + segment.start
                    _debug_log(f"[UserMic] {self.user_name}: {text[:50]}...")
                    self.transcription_ready.emit(self.user_name, text, segment_timestamp)
                else:
                    _debug_log(f"[UserMic] Text filtered out by post_process. Original: '{original_text[:100]}'")

    def _get_speaker_name(self, pyannote_label: str) -> str:
        """Convert pyannote speaker label to friendly name."""
        if pyannote_label not in self._speaker_map:
            # Assign a new speaker number
            self._speaker_counter += 1
            # Use names like "Speaker 1", "Speaker 2", etc.
            self._speaker_map[pyannote_label] = f"Speaker {self._speaker_counter}"
        return self._speaker_map[pyannote_label]

    def _on_transcription(self, speaker: str, text: str, timestamp: float):
        """Handle transcription result (main thread)."""
        self.transcript.add_entry(timestamp, speaker, text)

    def _on_status_change(self, status: str):
        """Handle status change (main thread)."""
        self.status_label.setText(f"> {status}")
        # Reset to normal style (not the bright green saved style)
        self.status_label.setStyleSheet("font-size: 22px; font-weight: 600; color: #00ff88;")

    def _on_time_update(self, seconds: int):
        """Handle time update (main thread)."""
        mins = seconds // 60
        secs = seconds % 60
        self.time_label.setText(f"{mins:02d}:{secs:02d}")

    def _update_time(self):
        """Update recording time display."""
        if self._recording:
            elapsed = int(time.time() - self._meeting_start_time)
            self.recording_time_updated.emit(elapsed)

    def _blink_recording_indicator(self):
        """Smooth pulsating effect for recording dot."""
        if not self._recording:
            return

        # Smooth fade in/out
        self._rec_opacity += 0.05 * self._rec_fade_direction

        # Reverse direction at boundaries
        if self._rec_opacity <= 0.3:
            self._rec_opacity = 0.3
            self._rec_fade_direction = 1
        elif self._rec_opacity >= 1.0:
            self._rec_opacity = 1.0
            self._rec_fade_direction = -1

        # Calculate color with opacity
        red_val = int(255 * self._rec_opacity)
        color = f"#{red_val:02x}4444"
        self.rec_dot.setStyleSheet(f"""
            font-size: 32px;
            font-weight: 700;
            color: {color};
            background: transparent;
            border: none;
        """)

    # ===== REMOVED: _check_processor() function =====
    # Users don't need to see buffer status updates - it was distracting
    # Status now shows static "> Recording..._" message instead

    def copy_recent(self, seconds: int = 30):
        """Copy recent transcript to clipboard."""
        text = self.transcript.get_recent_text(seconds)
        if text:
            pyperclip.copy(text)
            self.status_changed.emit(f"Copied last {seconds}s to clipboard")

    def copy_all(self):
        """Copy full transcript to clipboard."""
        text = self.transcript.get_full_text()
        if text:
            pyperclip.copy(text)
            self.status_changed.emit("Copied to clipboard")

    def toggle_live_window(self):
        """Toggle visibility of live window."""
        if self.live_window.isVisible():
            self.live_window.hide()
        else:
            self.live_window.show()

    def exit_app(self):
        """Exit the application."""
        if self._recording:
            self.stop_recording()
        QApplication.quit()

    def _spawn_summarization_subprocess(self, transcript_path: Path):
        """Spawn a detached subprocess to generate summary."""
        try:
            # Build command: python -m src.meeting.summarize_detached "C:\path\to\transcript.md"
            cmd = [
                sys.executable,  # Current Python interpreter
                "-m",
                "src.meeting.summarize_detached",
                str(transcript_path)
            ]

            # Spawn detached process (Windows-specific flags)
            DETACHED_PROCESS = 0x00000008
            CREATE_NO_WINDOW = 0x08000000

            subprocess.Popen(
                cmd,
                creationflags=DETACHED_PROCESS | CREATE_NO_WINDOW,
                cwd=str(Path(__file__).parent.parent.parent),  # koe root directory
                close_fds=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            _debug_log(f"[Summarization] Spawned subprocess for {transcript_path.name}")

        except Exception as e:
            _debug_log(f"[Summarization] Failed to spawn subprocess: {e}")
            # Non-critical - transcript is already saved

    def _start_polling_summary_status(self, transcript_path: Path):
        """Start polling the summary status file (MUST run on main Qt thread)."""
        _debug_log(f"[Summarization] _start_polling_summary_status() called on main thread")

        transcript_path_str = str(transcript_path)
        reader = SummaryStatusReader(transcript_path)
        self._status_readers[transcript_path_str] = {
            'reader': reader,
            'started_at': time.time()
        }

        _debug_log(f"[Summarization] Started polling for: {transcript_path_str}")
        _debug_log(f"[Summarization] Status file: {reader.status_file}")

        # Start polling after 2 seconds (give subprocess time to start)
        # This QTimer.singleShot is NOW safe because we're on the main thread
        _debug_log(f"[Summarization] Scheduling first poll in 2 seconds...")
        QTimer.singleShot(2000, lambda: self._poll_summary_status(transcript_path_str))
        _debug_log(f"[Summarization] Poll scheduled successfully")

    def _poll_summary_status(self, transcript_path_str: str):
        """Poll summary status file and update UI."""
        _debug_log(f"[Summarization] _poll_summary_status() CALLED for: {transcript_path_str}")

        if transcript_path_str not in self._status_readers:
            _debug_log(f"[Summarization] Polling stopped - path no longer tracked")
            return  # Stopped polling

        reader_info = self._status_readers[transcript_path_str]
        reader = reader_info['reader']
        started_at = reader_info['started_at']

        # Safety timeout: if polling for >5 minutes, assume failure
        elapsed = time.time() - started_at
        _debug_log(f"[Summarization] Poll attempt - elapsed: {elapsed:.1f}s")

        if elapsed > 300:  # 5 minutes
            _debug_log(f"[Summarization] Polling timeout after {elapsed:.0f}s - assuming failure")
            if self._summary_window:
                self._summary_window.update_status("Summary timed out")
            reader.cleanup()
            del self._status_readers[transcript_path_str]
            QTimer.singleShot(3000, self._close_summary_window)
            return

        _debug_log(f"[Summarization] Checking status file: {reader.status_file}")
        _debug_log(f"[Summarization] Status file exists: {reader.status_file.exists()}")

        status = reader.read()

        if status:
            _debug_log(f"[Summarization] Status read: {status.status}, Stage: {status.stage}, Path: {status.summary_path}")
            if status.status == 'in_progress':
                # Update summary window with progress
                _debug_log(f"[Summarization] In progress - updating UI")
                if self._summary_window:
                    self._summary_window.update_status(status.stage)
                # Poll again in 2 seconds
                QTimer.singleShot(2000, lambda: self._poll_summary_status(transcript_path_str))

            elif status.status == 'complete':
                # Show clickable link
                _debug_log(f"[Summarization] COMPLETE! Summary path: {status.summary_path}")
                if status.summary_path:
                    summary_path = Path(status.summary_path)
                    _debug_log(f"[Summarization] Summary file exists: {summary_path.exists()}")
                    self._show_summary_link(summary_path)
                else:
                    _debug_log(f"[Summarization] ERROR: No summary path in status!")
                # Clean up status file (parent's responsibility)
                reader.cleanup()
                # Stop polling
                del self._status_readers[transcript_path_str]
                _debug_log(f"[Summarization] Polling complete - stopped")

            elif status.status == 'failed':
                # Show error in summary window
                error_msg = status.error or "Unknown error"
                _debug_log(f"[Summarization] FAILED: {error_msg}")
                if self._summary_window:
                    self._summary_window.update_status(f"Summary failed: {error_msg}")
                # Clean up status file
                reader.cleanup()
                # Stop polling
                del self._status_readers[transcript_path_str]
                # Close summary window after 3 seconds
                QTimer.singleShot(3000, self._close_summary_window)
        else:
            # Status file doesn't exist yet or read error - retry
            if reader.status_file.exists():
                _debug_log(f"[Summarization] Status file exists but couldn't read it - will retry")
            else:
                _debug_log(f"[Summarization] Status file doesn't exist yet - will retry")
            QTimer.singleShot(2000, lambda: self._poll_summary_status(transcript_path_str))

    def _restore_ui_after_summary(self):
        """Restore UI after summary completes or fails."""
        # Clear notes and re-enable inputs
        self._clear_notes()
        self._enable_inputs()

        # Re-enable text editors
        self.agenda_edit.setEnabled(True)
        self.notes_edit.setEnabled(True)
        self.actions_edit.setEnabled(True)

        # Show buttons for next meeting
        self.record_button.show()
        self.record_button.setText("● REC")
        self.record_button.setEnabled(True)
        self.record_button.setStyleSheet("""
            background-color: #2a1a1a;
            border: 1px solid #ff4444;
            border-radius: 4px;
            padding: 12px 24px;
            font-size: 13px;
            font-weight: 600;
            color: #ff6666;
            font-family: 'Cascadia Code', 'Consolas', monospace;
        """)

        self.add_category_button.show()
        self.add_subcategory_button.show()
        self.open_button.show()
        self.save_button.show()

    def _restore_ui_after_summary_failure(self):
        """Restore UI after summary failure."""
        self._restore_ui_after_summary()
        self.status_changed.emit("Ready")

    def _show_summary_link(self, summary_path: Path):
        """Show clickable link in summary window."""
        if self._summary_window:
            self._summary_window.show_summary_link(summary_path)
        # Show "New Meeting" button now that recording is complete

    def _set_enrollment_data(self, unknown_speakers: dict, speaker_samples: dict, transcript_path):
        """Set enrollment data on the summary window (must be called from main thread)."""
        _debug_log(f"[Meeting] _set_enrollment_data called with {len(unknown_speakers)} unknown speakers")
        if self._summary_window:
            self._summary_window.set_enrollment_data(
                unknown_speakers,
                speaker_samples,
                self._diarizer,
                transcript_path
            )
            _debug_log("[Meeting] Enrollment data set on summary window")
        else:
            _debug_log("[Meeting] WARNING: _summary_window is None, cannot set enrollment data")

    def _start_summarization_after_enrollment(self, transcript_path: Path):
        """Start summarization after enrollment dialog is closed."""
        _debug_log(f"[Meeting] Enrollment closed, starting summarization for {transcript_path}")
        self.summary_status_changed.emit("Generating summary...")
        self._spawn_summarization_subprocess(transcript_path)
        self.start_summary_polling.emit(transcript_path)

    def _close_summary_window(self):
        """Close the summary window."""
        if self._summary_window:
            self._summary_window.close()
            self._summary_window = None

    def _new_meeting_from_summary(self):
        """Open a new meeting (reset and show main window)."""
        # Reset the main window for a new meeting
        self._new_meeting()
        # Show the main window
        self.live_window.show()
        # Keep the summary window open (user can close it manually with ESC)

    def _open_file(self, url: str):
        """Open folder containing file with the file selected in Explorer."""
        from urllib.parse import unquote, urlparse

        # Convert file:/// URL back to Windows path
        # Handle URL encoding (spaces become %20, etc.)
        parsed = urlparse(url)
        filepath = unquote(parsed.path)

        # Remove leading slash for Windows paths (e.g., /C:/path -> C:/path)
        if filepath.startswith('/') and len(filepath) > 2 and filepath[2] == ':':
            filepath = filepath[1:]

        # Convert to Windows path separators
        filepath = filepath.replace("/", "\\")

        # Open Explorer with the file selected
        try:
            subprocess.Popen(['explorer', '/select,', filepath], shell=False)
            _debug_log(f"[Summarization] Opened folder with {filepath} selected")
        except Exception as e:
            _debug_log(f"[Summarization] Failed to open folder: {e}")

    def cleanup(self):
        """Clean up resources before exit."""
        # Release the single-instance lock mutex if it exists
        if hasattr(self, '_mutex') and self._mutex:
            try:
                import ctypes
                ctypes.windll.kernel32.ReleaseMutex(self._mutex)  # type: ignore
                ctypes.windll.kernel32.CloseHandle(self._mutex)  # type: ignore
            except:
                pass

    def run(self):
        """Run the application."""
        try:
            return self.app.exec_()
        finally:
            self.cleanup()


def main():
    """Entry point for meeting transcription mode."""
    print("=" * 50)
    print("Meeting Transcriber")
    print("=" * 50)
    print()
    print("This mode captures meeting audio and transcribes with speaker separation.")
    print("Make sure the Whisper server is running first:")
    print("  python -m src.server")
    print()

    # Enable high-DPI scaling BEFORE creating QApplication
    from PyQt5.QtWidgets import QApplication
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # Initialize config manager
    ConfigManager.initialize()

    # Get user name from config, default to "Me" if not set
    user_name = ConfigManager.get_config_value('profile', 'user_name') or "Me"
    app = MeetingTranscriberApp(user_name=user_name)
    sys.exit(app.run())


if __name__ == "__main__":
    main()
