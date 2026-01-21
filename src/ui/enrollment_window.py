"""
Speaker enrollment window with terminal styling.
Records audio from microphone or system audio and creates a voice fingerprint.
"""

import sys
import os
import time
import numpy as np
from enum import Enum, auto
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer, QRectF
from PyQt5.QtGui import QFont, QPainter, QBrush, QColor, QPainterPath, QPen, QKeyEvent, QMouseEvent
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QMessageBox, QApplication
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ui import theme


class EnrollmentState(Enum):
    NAME_INPUT = auto()
    COUNTDOWN = auto()
    RECORDING = auto()
    PROCESSING = auto()
    SUCCESS = auto()
    FAILURE = auto()


class RecordingThread(QThread):
    """Background thread for audio recording."""
    finished = pyqtSignal(object)  # numpy array or None
    error = pyqtSignal(str)

    def __init__(self, mode: str, sample_rate: int = 16000):
        super().__init__()
        self.mode = mode
        self.sample_rate = sample_rate
        self._stop_flag = False
        self._frames = []

    def stop(self):
        self._stop_flag = True

    def run(self):
        try:
            if self.mode == "mic":
                audio = self._record_mic()
            else:
                audio = self._record_loopback()
            self.finished.emit(audio)
        except Exception as e:
            self.error.emit(str(e))

    def _record_mic(self) -> np.ndarray:
        """Record from default microphone."""
        import sounddevice as sd

        self._frames = []
        self._stop_flag = False

        def callback(indata, frame_count, time_info, status):
            if not self._stop_flag:
                self._frames.append(indata.copy())

        with sd.InputStream(samplerate=self.sample_rate, channels=1,
                           dtype='int16', callback=callback):
            while not self._stop_flag:
                time.sleep(0.05)

        if self._frames:
            return np.concatenate(self._frames).flatten()
        return np.array([], dtype=np.int16)

    def _record_loopback(self) -> np.ndarray:
        """Record from system audio loopback."""
        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            raise RuntimeError("pyaudiowpatch not installed")

        # Find loopback device
        p = pyaudio.PyAudio()
        device = None

        try:
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_output = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

            for i in range(p.get_device_count()):
                dev = p.get_device_info_by_index(i)
                if dev.get("isLoopbackDevice", False):
                    if default_output["name"] in dev["name"]:
                        device = dev
                        break

            if not device:
                for i in range(p.get_device_count()):
                    dev = p.get_device_info_by_index(i)
                    if dev.get("isLoopbackDevice", False):
                        device = dev
                        break
        finally:
            if not device:
                p.terminate()
                raise RuntimeError("No loopback device found")

        # Record
        native_rate = int(device['defaultSampleRate'])
        channels = device['maxInputChannels']
        chunk_size = 1024

        self._frames = []
        self._stop_flag = False

        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=native_rate,
            input=True,
            input_device_index=device['index'],
            frames_per_buffer=chunk_size
        )

        try:
            while not self._stop_flag:
                data = stream.read(chunk_size, exception_on_overflow=False)
                self._frames.append(np.frombuffer(data, dtype=np.int16))
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

        if not self._frames:
            return np.array([], dtype=np.int16)

        audio = np.concatenate(self._frames)

        # Convert stereo to mono
        if channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)

        # Resample to 16kHz if needed
        if native_rate != self.sample_rate:
            original_length = len(audio)
            target_length = int(original_length * self.sample_rate / native_rate)
            indices = np.linspace(0, original_length - 1, target_length)
            audio = np.interp(indices, np.arange(original_length),
                            audio.astype(np.float32)).astype(np.int16)

        return audio


class ProcessingThread(QThread):
    """Background thread for enrollment processing."""
    finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, name: str, audio: np.ndarray, sample_rate: int = 16000):
        super().__init__()
        self.name = name
        self.audio = audio
        self.sample_rate = sample_rate

    def run(self):
        try:
            from meeting.diarization import get_diarizer

            diarizer = get_diarizer()

            # Load model if needed (first enrollment takes ~3s)
            if not diarizer.load():
                self.finished.emit(False, "Failed to load diarization model")
                return

            # Enroll speaker
            if diarizer.enroll_speaker(self.name, self.audio, self.sample_rate):
                self.finished.emit(True, f"'{self.name}' enrolled!")
            else:
                self.finished.emit(False, "Embedding extraction failed")

        except Exception as e:
            self.finished.emit(False, str(e))


class EnrollmentWindow(QMainWindow):
    """Terminal-styled speaker enrollment window."""

    enrollmentComplete = pyqtSignal(bool, str)  # success, name

    # Terminal color scheme (from centralized theme)
    BG_COLOR = QColor(10, 10, 15, 245)
    BORDER_COLOR = QColor(0, 255, 136)
    TEXT_COLOR = theme.TEXT_COLOR
    SECONDARY_TEXT = theme.SECONDARY_TEXT
    RECORDING_COLOR = theme.RECORDING_COLOR
    ERROR_COLOR = theme.ERROR_COLOR
    SUCCESS_COLOR = theme.SUCCESS_COLOR

    def __init__(self, mode: str = "mic", parent=None):
        """
        Initialize enrollment window.

        Args:
            mode: "mic" for microphone, "loopback" for system audio
        """
        super().__init__(parent)
        self.mode = mode
        self.state = EnrollmentState.NAME_INPUT
        self.speaker_name = ""
        self.recording_start_time = None
        self.countdown_value = 3 if mode == "mic" else 5

        self._drag_pos = None
        self._recording_thread = None
        self._processing_thread = None

        # Timers
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_timer)
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self._toggle_blink)
        self.blink_state = True
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self._countdown_tick)

        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle('Enroll Speaker')
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFixedSize(500, 185)

        self.main_widget = QWidget(self)
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(16, 12, 16, 12)
        self.main_layout.setSpacing(8)

        # Header row with indicator and title
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)

        self.indicator = QLabel("●")
        self.indicator.setFont(QFont('Cascadia Code', 12))
        self.indicator.setStyleSheet(f"color: {self.TEXT_COLOR};")
        self.indicator.setFixedWidth(18)
        header_layout.addWidget(self.indicator)

        mode_text = "Microphone" if self.mode == "mic" else "System Audio"
        self.title_label = QLabel(f"Enroll from {mode_text}")
        self.title_label.setFont(QFont('Cascadia Code', 11, QFont.Bold))
        self.title_label.setStyleSheet(f"color: {self.TEXT_COLOR};")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()

        # Close button
        self.close_btn = QPushButton("×")
        self.close_btn.setFont(QFont('Cascadia Code', 14))
        self.close_btn.setFixedSize(24, 24)
        self.close_btn.setStyleSheet(f"""
            QPushButton {{
                color: {self.SECONDARY_TEXT};
                background: transparent;
                border: none;
            }}
            QPushButton:hover {{
                color: {self.ERROR_COLOR};
            }}
        """)
        self.close_btn.clicked.connect(self._cancel)
        header_layout.addWidget(self.close_btn)

        self.main_layout.addLayout(header_layout)

        # Status/prompt label
        self.status_label = QLabel("> Enter speaker name:")
        self.status_label.setFont(QFont('Cascadia Code', 10))
        self.status_label.setStyleSheet(f"color: {self.TEXT_COLOR};")
        self.main_layout.addWidget(self.status_label)

        # Name input field
        self.name_input = QLineEdit()
        self.name_input.setFont(QFont('Cascadia Code', 12))
        self.name_input.setMinimumHeight(40)
        self.name_input.setStyleSheet(f"""
            QLineEdit {{
                background: rgba(0, 0, 0, 0.3);
                border: 1px solid {self.SECONDARY_TEXT};
                border-radius: 4px;
                color: {self.TEXT_COLOR};
                padding: 10px 14px;
                font-size: 12pt;
            }}
            QLineEdit:focus {{
                border-color: {self.TEXT_COLOR};
            }}
        """)
        self.name_input.returnPressed.connect(self._start_enrollment)
        self.main_layout.addWidget(self.name_input)

        # Bottom row: timer/status and buttons
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(8)

        self.timer_label = QLabel("")
        self.timer_label.setFont(QFont('Cascadia Code', 10))
        self.timer_label.setStyleSheet(f"color: {self.SECONDARY_TEXT};")
        bottom_layout.addWidget(self.timer_label)
        bottom_layout.addStretch()

        self.action_btn = QPushButton("Start")
        self.action_btn.setFont(QFont('Cascadia Code', 10))
        self.action_btn.setStyleSheet(f"""
            QPushButton {{
                color: {self.TEXT_COLOR};
                background: rgba(0, 255, 136, 0.1);
                border: 1px solid {self.TEXT_COLOR};
                border-radius: 4px;
                padding: 4px 12px;
            }}
            QPushButton:hover {{
                background: rgba(0, 255, 136, 0.2);
            }}
            QPushButton:disabled {{
                color: {self.SECONDARY_TEXT};
                border-color: {self.SECONDARY_TEXT};
                background: transparent;
            }}
        """)
        self.action_btn.clicked.connect(self._on_action_clicked)
        bottom_layout.addWidget(self.action_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setFont(QFont('Cascadia Code', 10))
        self.cancel_btn.setStyleSheet(f"""
            QPushButton {{
                color: {self.SECONDARY_TEXT};
                background: transparent;
                border: 1px solid {self.SECONDARY_TEXT};
                border-radius: 4px;
                padding: 4px 12px;
            }}
            QPushButton:hover {{
                color: {self.ERROR_COLOR};
                border-color: {self.ERROR_COLOR};
            }}
        """)
        self.cancel_btn.clicked.connect(self._cancel)
        bottom_layout.addWidget(self.cancel_btn)

        self.main_layout.addLayout(bottom_layout)
        self.setCentralWidget(self.main_widget)

    def paintEvent(self, event):
        """Draw rounded rectangle background with border."""
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 8, 8)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.setBrush(QBrush(self.BG_COLOR))
        painter.setPen(Qt.NoPen)
        painter.drawPath(path)

        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(self.BORDER_COLOR, 1))
        painter.drawPath(path)

    def keyPressEvent(self, event: QKeyEvent):
        """Handle key presses."""
        if event.key() == Qt.Key_Escape:
            self._cancel()
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.state == EnrollmentState.RECORDING:
                self._stop_recording()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        """Start dragging."""
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle dragging."""
        if event.buttons() == Qt.LeftButton and self._drag_pos is not None:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Stop dragging."""
        if event.button() == Qt.LeftButton:
            self._drag_pos = None
            event.accept()

    def show(self):
        """Center window on screen and show."""
        screen = QApplication.primaryScreen()
        screen_geo = screen.geometry()
        x = (screen_geo.width() - self.width()) // 2
        y = (screen_geo.height() - self.height()) // 2
        self.move(x, y)
        super().show()
        self.name_input.setFocus()
        self.activateWindow()

    def _on_action_clicked(self):
        """Handle action button click based on current state."""
        if self.state == EnrollmentState.NAME_INPUT:
            self._start_enrollment()
        elif self.state == EnrollmentState.RECORDING:
            self._stop_recording()

    def _start_enrollment(self):
        """Validate name and start countdown."""
        name = self.name_input.text().strip()

        if not name:
            self.status_label.setText("> Name cannot be empty!")
            self.status_label.setStyleSheet(f"color: {self.ERROR_COLOR};")
            return

        # Sanitize filename
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, '')

        if not name:
            self.status_label.setText("> Invalid name!")
            self.status_label.setStyleSheet(f"color: {self.ERROR_COLOR};")
            return

        # Check if speaker already exists
        embedding_path = Path(__file__).parent.parent.parent / "speaker_embeddings" / f"{name}.npy"
        if embedding_path.exists():
            reply = QMessageBox.question(
                self, "Speaker Exists",
                f"'{name}' is already enrolled. Replace?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        self.speaker_name = name
        self._start_countdown()

    def _start_countdown(self):
        """Start countdown before recording."""
        self.state = EnrollmentState.COUNTDOWN
        self.name_input.setEnabled(False)
        self.action_btn.setEnabled(False)

        self.countdown_value = 3 if self.mode == "mic" else 5
        self._update_countdown_display()
        self.countdown_timer.start(1000)

    def _countdown_tick(self):
        """Handle countdown timer tick."""
        self.countdown_value -= 1
        if self.countdown_value <= 0:
            self.countdown_timer.stop()
            self._start_recording()
        else:
            self._update_countdown_display()

    def _update_countdown_display(self):
        """Update UI during countdown."""
        if self.mode == "loopback":
            self.status_label.setText(f"> Play audio now! Starting in {self.countdown_value}...")
        else:
            self.status_label.setText(f"> Get ready... {self.countdown_value}")
        self.status_label.setStyleSheet(f"color: {self.TEXT_COLOR};")
        self.timer_label.setText("")  # Clear timer label during countdown

    def _start_recording(self):
        """Start audio recording."""
        self.state = EnrollmentState.RECORDING
        self.recording_start_time = time.time()

        # Update UI
        self.indicator.setStyleSheet(f"color: {self.RECORDING_COLOR};")
        if self.mode == "mic":
            self.status_label.setText("> Speak clearly into microphone_")
        else:
            self.status_label.setText("> Recording system audio_")
        self.status_label.setStyleSheet(f"color: {self.TEXT_COLOR};")
        self.timer_label.setText("00:00")
        self.action_btn.setText("Stop")
        self.action_btn.setEnabled(True)

        # Start timers
        self.timer.start(1000)
        self.blink_timer.start(500)

        # Start recording thread
        self._recording_thread = RecordingThread(self.mode)
        self._recording_thread.finished.connect(self._on_recording_finished)
        self._recording_thread.error.connect(self._on_recording_error)
        self._recording_thread.start()

    def _stop_recording(self):
        """Stop the recording."""
        if self._recording_thread and self._recording_thread.isRunning():
            self._recording_thread.stop()

    def _toggle_blink(self):
        """Toggle recording indicator."""
        self.blink_state = not self.blink_state
        if self.blink_state:
            self.indicator.setStyleSheet(f"color: {self.RECORDING_COLOR};")
        else:
            self.indicator.setStyleSheet("color: transparent;")

    def _update_timer(self):
        """Update timer display."""
        if self.recording_start_time:
            elapsed = time.time() - self.recording_start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.timer_label.setText(f"{minutes:02d}:{seconds:02d}")

    def _on_recording_finished(self, audio: np.ndarray):
        """Handle recording completion."""
        self.timer.stop()
        self.blink_timer.stop()

        if audio is None or len(audio) == 0:
            self._show_error("No audio recorded")
            return

        duration = len(audio) / 16000
        if duration < 1.0:
            self._show_error("Recording too short (< 1s)")
            return

        # Start processing
        self._start_processing(audio)

    def _on_recording_error(self, error: str):
        """Handle recording error."""
        self.timer.stop()
        self.blink_timer.stop()
        self._show_error(error)

    def _start_processing(self, audio: np.ndarray):
        """Process recorded audio and create embedding."""
        self.state = EnrollmentState.PROCESSING

        self.indicator.setText("◉")
        self.indicator.setStyleSheet(f"color: {self.TEXT_COLOR};")
        self.status_label.setText("> Extracting voice profile...")
        self.action_btn.setEnabled(False)

        self._processing_thread = ProcessingThread(self.speaker_name, audio)
        self._processing_thread.finished.connect(self._on_processing_finished)
        self._processing_thread.start()

    def _on_processing_finished(self, success: bool, message: str):
        """Handle processing completion."""
        if success:
            self._show_success(message)
        else:
            self._show_error(message)

    def _show_success(self, message: str):
        """Show success state and auto-close."""
        self.state = EnrollmentState.SUCCESS
        self.indicator.setText("✓")
        self.indicator.setStyleSheet(f"color: {self.SUCCESS_COLOR};")
        self.status_label.setText(f"> {message}")
        self.status_label.setStyleSheet(f"color: {self.SUCCESS_COLOR};")
        self.timer_label.setText("")
        self.action_btn.hide()
        self.cancel_btn.setText("Close")

        self.enrollmentComplete.emit(True, self.speaker_name)

        # Auto-close after 2 seconds
        QTimer.singleShot(2000, self.close)

    def _show_error(self, message: str):
        """Show error state."""
        self.state = EnrollmentState.FAILURE
        self.indicator.setText("✗")
        self.indicator.setStyleSheet(f"color: {self.ERROR_COLOR};")
        self.status_label.setText(f"> {message}")
        self.status_label.setStyleSheet(f"color: {self.ERROR_COLOR};")
        self.action_btn.hide()
        self.cancel_btn.setText("Close")

        self.enrollmentComplete.emit(False, self.speaker_name)

    def _cancel(self):
        """Cancel enrollment and close window."""
        # Stop any running threads
        if self._recording_thread and self._recording_thread.isRunning():
            self._recording_thread.stop()
            self._recording_thread.wait(1000)

        self.timer.stop()
        self.blink_timer.stop()
        self.countdown_timer.stop()

        self.close()

    def closeEvent(self, event):
        """Clean up on close."""
        self._cancel()
        super().closeEvent(event)


if __name__ == '__main__':
    # Test the window
    app = QApplication(sys.argv)
    window = EnrollmentWindow(mode="mic")
    window.show()
    sys.exit(app.exec_())
