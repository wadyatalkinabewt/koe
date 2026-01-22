"""
Koe Setup Wizard - First-time setup experience.

Guides new users through:
1. System requirements check
2. API key configuration
3. Model download
4. User profile setup
5. Voice enrollment
6. Folder configuration
7. Launch Koe
"""

import sys
import os
import subprocess
import time
import threading
from pathlib import Path
from enum import Enum, auto

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer, QRectF, QSize
from PyQt5.QtGui import QFont, QPainter, QBrush, QColor, QPainterPath, QPen, QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QStackedWidget, QProgressBar,
    QFileDialog, QMessageBox, QFrame, QTextEdit, QCheckBox
)

from ui import theme


class SetupPage(Enum):
    WELCOME = 0
    SYSTEM_CHECK = 1
    API_KEYS = 2
    MODEL_DOWNLOAD = 3
    USER_PROFILE = 4
    VOICE_ENROLLMENT = 5
    FOLDERS = 6
    COMPLETE = 7


class SystemCheckThread(QThread):
    """Thread to check system requirements."""
    progress = pyqtSignal(str, bool)  # message, success
    finished = pyqtSignal(dict)  # results dict

    def run(self):
        results = {
            'python': False,
            'gpu': False,
            'cuda': False,
            'packages': False,
            'errors': []
        }

        # Check Python version
        self.progress.emit("Checking Python version...", True)
        time.sleep(0.3)
        py_version = sys.version_info
        if py_version >= (3, 10):
            results['python'] = True
            self.progress.emit(f"✓ Python {py_version.major}.{py_version.minor} detected", True)
        else:
            results['errors'].append(f"Python 3.10+ required, found {py_version.major}.{py_version.minor}")
            self.progress.emit(f"✗ Python {py_version.major}.{py_version.minor} (need 3.10+)", False)

        time.sleep(0.3)

        # Check for NVIDIA GPU
        self.progress.emit("Checking for NVIDIA GPU...", True)
        time.sleep(0.3)
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                gpu_info = result.stdout.strip().split(',')
                gpu_name = gpu_info[0].strip()
                gpu_memory = gpu_info[1].strip() if len(gpu_info) > 1 else "Unknown"
                results['gpu'] = True
                results['gpu_name'] = gpu_name
                results['gpu_memory'] = gpu_memory
                self.progress.emit(f"✓ {gpu_name} ({gpu_memory})", True)
            else:
                results['errors'].append("No NVIDIA GPU detected")
                self.progress.emit("✗ No NVIDIA GPU detected", False)
        except FileNotFoundError:
            results['errors'].append("nvidia-smi not found - NVIDIA drivers not installed")
            self.progress.emit("✗ NVIDIA drivers not installed", False)
        except Exception as e:
            results['errors'].append(f"GPU check failed: {e}")
            self.progress.emit(f"✗ GPU check failed", False)

        time.sleep(0.3)

        # Check CUDA availability via PyTorch
        self.progress.emit("Checking CUDA support...", True)
        time.sleep(0.3)
        try:
            import torch
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                results['cuda'] = True
                results['cuda_version'] = cuda_version
                self.progress.emit(f"✓ CUDA {cuda_version} available", True)
            else:
                results['errors'].append("CUDA not available in PyTorch")
                self.progress.emit("✗ CUDA not available", False)
        except ImportError:
            results['errors'].append("PyTorch not installed")
            self.progress.emit("✗ PyTorch not installed", False)

        time.sleep(0.3)

        # Check key packages
        self.progress.emit("Checking required packages...", True)
        time.sleep(0.3)
        required_packages = [
            ('faster_whisper', 'faster-whisper'),
            ('pyannote.audio', 'pyannote-audio'),
            ('sounddevice', 'sounddevice'),
            ('PyQt5', 'PyQt5')
        ]

        missing = []
        for module, package in required_packages:
            try:
                __import__(module.split('.')[0])
            except ImportError:
                missing.append(package)

        if not missing:
            results['packages'] = True
            self.progress.emit("✓ All required packages installed", True)
        else:
            results['errors'].append(f"Missing packages: {', '.join(missing)}")
            self.progress.emit(f"✗ Missing: {', '.join(missing)}", False)

        time.sleep(0.3)
        self.finished.emit(results)


class ModelDownloadThread(QThread):
    """Thread to download/verify models."""
    progress = pyqtSignal(str, int)  # message, percent
    finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, hf_token: str):
        super().__init__()
        self.hf_token = hf_token

    def run(self):
        try:
            # Set HF token in environment
            os.environ['HF_TOKEN'] = self.hf_token

            self.progress.emit("Loading Whisper model (this may take a few minutes)...", 10)

            # Load Whisper model
            from faster_whisper import WhisperModel

            self.progress.emit("Downloading Whisper large-v3 (~3GB)...", 20)

            # This will download if not cached
            model = WhisperModel("large-v3", device="cuda", compute_type="float16")
            del model

            self.progress.emit("✓ Whisper model ready", 50)
            time.sleep(0.5)

            # Load pyannote models
            self.progress.emit("Loading speaker diarization models...", 60)

            from pyannote.audio import Pipeline

            self.progress.emit("Downloading pyannote models...", 70)

            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            )
            del pipeline

            self.progress.emit("✓ Diarization models ready", 90)
            time.sleep(0.5)

            self.progress.emit("✓ All models downloaded successfully!", 100)
            self.finished.emit(True, "Models ready")

        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                self.finished.emit(False, "Invalid HuggingFace token. Please check your token.")
            elif "403" in error_msg or "access" in error_msg.lower():
                self.finished.emit(False, "Access denied. Please accept the model licenses on HuggingFace.")
            else:
                self.finished.emit(False, f"Download failed: {error_msg[:100]}")


class VoiceRecordThread(QThread):
    """Thread for voice recording during enrollment."""
    finished = pyqtSignal(object)  # numpy array
    error = pyqtSignal(str)

    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self.sample_rate = sample_rate
        self._stop_flag = False
        self._frames = []

    def stop(self):
        self._stop_flag = True

    def run(self):
        try:
            import sounddevice as sd
            import numpy as np

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
                audio = np.concatenate(self._frames).flatten()
                self.finished.emit(audio)
            else:
                self.finished.emit(np.array([], dtype=np.int16))

        except Exception as e:
            self.error.emit(str(e))


class EnrollmentThread(QThread):
    """Thread for processing voice enrollment."""
    finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, name: str, audio, sample_rate: int = 16000):
        super().__init__()
        self.name = name
        self.audio = audio
        self.sample_rate = sample_rate

    def run(self):
        try:
            from meeting.diarization import get_diarizer

            diarizer = get_diarizer()

            if not diarizer.load():
                self.finished.emit(False, "Failed to load diarization model")
                return

            if diarizer.enroll_speaker(self.name, self.audio, self.sample_rate):
                self.finished.emit(True, f"Voice enrolled as '{self.name}'")
            else:
                self.finished.emit(False, "Failed to extract voice profile")

        except Exception as e:
            self.finished.emit(False, str(e))


class SetupWizard(QMainWindow):
    """Terminal-styled setup wizard for Koe."""

    def __init__(self):
        super().__init__()

        # State
        self.current_page = SetupPage.WELCOME
        self.system_results = {}
        self.hf_token = ""
        self.anthropic_key = ""
        self.user_name = ""
        self.voice_audio = None
        self.meetings_folder = ""
        self.snippets_folder = ""

        # Threads
        self._system_thread = None
        self._model_thread = None
        self._voice_thread = None
        self._enroll_thread = None

        # Timers
        self.voice_timer = QTimer()
        self.voice_timer.timeout.connect(self._update_voice_timer)
        self.voice_start_time = None

        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Koe Setup")
        self.setFixedSize(700, 550)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Set window icon
        icon_path = Path(__file__).parent.parent / "assets" / "koe-icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # Main container
        self.container = QWidget(self)
        self.setCentralWidget(self.container)

        main_layout = QVBoxLayout(self.container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Content frame with rounded corners
        self.content_frame = QFrame()
        self.content_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {theme.BG_COLOR};
                border: 2px solid {theme.BORDER_COLOR};
                border-radius: 12px;
            }}
        """)

        content_layout = QVBoxLayout(self.content_frame)
        content_layout.setContentsMargins(24, 20, 24, 20)
        content_layout.setSpacing(16)

        # Header
        header_layout = QHBoxLayout()

        self.title_label = QLabel("Koe Setup")
        self.title_label.setFont(QFont('Cascadia Code', 18, QFont.Bold))
        self.title_label.setStyleSheet(f"color: {theme.TEXT_COLOR}; background: transparent; border: none;")
        header_layout.addWidget(self.title_label)

        header_layout.addStretch()

        # Close button
        close_btn = QPushButton("×")
        close_btn.setFont(QFont('Cascadia Code', 16))
        close_btn.setFixedSize(30, 30)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                color: {theme.SECONDARY_TEXT};
                background: transparent;
                border: none;
            }}
            QPushButton:hover {{
                color: {theme.ERROR_COLOR};
            }}
        """)
        close_btn.clicked.connect(self._confirm_close)
        header_layout.addWidget(close_btn)

        content_layout.addLayout(header_layout)

        # Progress indicator
        self.progress_label = QLabel("Step 1 of 7")
        self.progress_label.setFont(QFont('Cascadia Code', 9))
        self.progress_label.setStyleSheet(f"color: {theme.SECONDARY_TEXT}; background: transparent; border: none;")
        content_layout.addWidget(self.progress_label)

        # Stacked widget for pages
        self.pages = QStackedWidget()
        self.pages.setStyleSheet("background: transparent; border: none;")

        # Create all pages
        self._create_welcome_page()
        self._create_system_check_page()
        self._create_api_keys_page()
        self._create_model_download_page()
        self._create_user_profile_page()
        self._create_voice_enrollment_page()
        self._create_folders_page()
        self._create_complete_page()

        content_layout.addWidget(self.pages, 1)

        # Navigation buttons
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(12)

        self.back_btn = QPushButton("← Back")
        self.back_btn.setFont(QFont('Cascadia Code', 10))
        self.back_btn.setStyleSheet(self._button_style(secondary=True))
        self.back_btn.clicked.connect(self._go_back)
        self.back_btn.hide()
        nav_layout.addWidget(self.back_btn)

        nav_layout.addStretch()

        self.skip_btn = QPushButton("Skip")
        self.skip_btn.setFont(QFont('Cascadia Code', 10))
        self.skip_btn.setStyleSheet(self._button_style(secondary=True))
        self.skip_btn.clicked.connect(self._skip_step)
        self.skip_btn.hide()
        nav_layout.addWidget(self.skip_btn)

        self.next_btn = QPushButton("Get Started →")
        self.next_btn.setFont(QFont('Cascadia Code', 10))
        self.next_btn.setStyleSheet(self._button_style())
        self.next_btn.clicked.connect(self._go_next)
        nav_layout.addWidget(self.next_btn)

        content_layout.addLayout(nav_layout)

        main_layout.addWidget(self.content_frame)

        # Enable dragging
        self._drag_pos = None

    def _button_style(self, secondary=False):
        """Get button stylesheet."""
        if secondary:
            return f"""
                QPushButton {{
                    color: {theme.SECONDARY_TEXT};
                    background: transparent;
                    border: 1px solid {theme.SECONDARY_TEXT};
                    border-radius: 6px;
                    padding: 8px 20px;
                    min-width: 80px;
                }}
                QPushButton:hover {{
                    color: {theme.TEXT_COLOR};
                    border-color: {theme.TEXT_COLOR};
                }}
                QPushButton:disabled {{
                    color: {theme.DIM_TEXT};
                    border-color: {theme.DIM_TEXT};
                }}
            """
        return f"""
            QPushButton {{
                color: {theme.BG_COLOR};
                background: {theme.TEXT_COLOR};
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                min-width: 100px;
            }}
            QPushButton:hover {{
                background: {theme.LINK_COLOR};
            }}
            QPushButton:disabled {{
                background: {theme.DIM_TEXT};
            }}
        """

    def _input_style(self):
        """Get input field stylesheet."""
        return f"""
            QLineEdit {{
                background: {theme.INPUT_BG};
                border: 1px solid {theme.INPUT_BORDER};
                border-radius: 6px;
                color: {theme.TEXT_COLOR};
                padding: 12px;
                font-family: 'Cascadia Code';
                font-size: 11pt;
            }}
            QLineEdit:focus {{
                border-color: {theme.INPUT_FOCUS_BORDER};
            }}
            QLineEdit:disabled {{
                color: {theme.DIM_TEXT};
                background: {theme.BG_COLOR};
            }}
        """

    def _create_welcome_page(self):
        """Create welcome page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # Welcome text
        welcome = QLabel("Welcome to Koe")
        welcome.setFont(QFont('Cascadia Code', 24, QFont.Bold))
        welcome.setStyleSheet(f"color: {theme.TEXT_COLOR};")
        welcome.setAlignment(Qt.AlignCenter)
        layout.addWidget(welcome)

        subtitle = QLabel("Local speech-to-text with speaker identification")
        subtitle.setFont(QFont('Cascadia Code', 11))
        subtitle.setStyleSheet(f"color: {theme.SECONDARY_TEXT};")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacing(20)

        # Feature list
        features = [
            "✓ Hotkey transcription to clipboard",
            "✓ Meeting transcription with speaker IDs",
            "✓ AI-powered meeting summaries",
            "✓ Voice enrollment for recognition",
            "✓ Works offline (except AI summaries)"
        ]

        features_text = QLabel("\n".join(features))
        features_text.setFont(QFont('Cascadia Code', 10))
        features_text.setStyleSheet(f"color: {theme.TEXT_COLOR};")
        features_text.setAlignment(Qt.AlignCenter)
        layout.addWidget(features_text)

        layout.addSpacing(20)

        info = QLabel("This wizard will help you set up Koe in about 5 minutes.")
        info.setFont(QFont('Cascadia Code', 9))
        info.setStyleSheet(f"color: {theme.SECONDARY_TEXT};")
        info.setAlignment(Qt.AlignCenter)
        layout.addWidget(info)

        layout.addStretch()

        self.pages.addWidget(page)

    def _create_system_check_page(self):
        """Create system requirements check page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        title = QLabel("System Requirements")
        title.setFont(QFont('Cascadia Code', 14, QFont.Bold))
        title.setStyleSheet(f"color: {theme.TEXT_COLOR};")
        layout.addWidget(title)

        desc = QLabel("Checking your system compatibility...")
        desc.setFont(QFont('Cascadia Code', 10))
        desc.setStyleSheet(f"color: {theme.SECONDARY_TEXT};")
        layout.addWidget(desc)

        layout.addSpacing(10)

        # Status output
        self.system_output = QTextEdit()
        self.system_output.setReadOnly(True)
        self.system_output.setFont(QFont('Cascadia Code', 10))
        self.system_output.setStyleSheet(f"""
            QTextEdit {{
                background: {theme.INPUT_BG};
                border: 1px solid {theme.INPUT_BORDER};
                border-radius: 6px;
                color: {theme.TEXT_COLOR};
                padding: 12px;
            }}
        """)
        self.system_output.setMaximumHeight(200)
        layout.addWidget(self.system_output)

        # Status label
        self.system_status = QLabel("")
        self.system_status.setFont(QFont('Cascadia Code', 10))
        self.system_status.setStyleSheet(f"color: {theme.SECONDARY_TEXT};")
        self.system_status.setWordWrap(True)
        layout.addWidget(self.system_status)

        layout.addStretch()

        self.pages.addWidget(page)

    def _create_api_keys_page(self):
        """Create API keys input page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        title = QLabel("API Configuration")
        title.setFont(QFont('Cascadia Code', 14, QFont.Bold))
        title.setStyleSheet(f"color: {theme.TEXT_COLOR};")
        layout.addWidget(title)

        # HuggingFace token
        hf_label = QLabel("HuggingFace Token (required for speaker diarization)")
        hf_label.setFont(QFont('Cascadia Code', 10))
        hf_label.setStyleSheet(f"color: {theme.TEXT_COLOR};")
        layout.addWidget(hf_label)

        self.hf_input = QLineEdit()
        self.hf_input.setPlaceholderText("hf_...")
        self.hf_input.setEchoMode(QLineEdit.Password)
        self.hf_input.setStyleSheet(self._input_style())
        layout.addWidget(self.hf_input)

        hf_link = QLabel('<a href="https://huggingface.co/settings/tokens" style="color: ' + theme.LINK_COLOR + ';">Get token from huggingface.co/settings/tokens</a>')
        hf_link.setFont(QFont('Cascadia Code', 9))
        hf_link.setOpenExternalLinks(True)
        layout.addWidget(hf_link)

        layout.addSpacing(16)

        # Anthropic key (optional)
        anthropic_label = QLabel("Anthropic API Key (optional - for AI meeting summaries)")
        anthropic_label.setFont(QFont('Cascadia Code', 10))
        anthropic_label.setStyleSheet(f"color: {theme.TEXT_COLOR};")
        layout.addWidget(anthropic_label)

        self.anthropic_input = QLineEdit()
        self.anthropic_input.setPlaceholderText("sk-ant-... (optional)")
        self.anthropic_input.setEchoMode(QLineEdit.Password)
        self.anthropic_input.setStyleSheet(self._input_style())
        layout.addWidget(self.anthropic_input)

        anthropic_link = QLabel('<a href="https://console.anthropic.com" style="color: ' + theme.LINK_COLOR + ';">Get key from console.anthropic.com</a>')
        anthropic_link.setFont(QFont('Cascadia Code', 9))
        anthropic_link.setOpenExternalLinks(True)
        layout.addWidget(anthropic_link)

        layout.addSpacing(8)

        cost_note = QLabel("AI summaries cost ~$0.04 per 60-minute meeting")
        cost_note.setFont(QFont('Cascadia Code', 9))
        cost_note.setStyleSheet(f"color: {theme.SECONDARY_TEXT};")
        layout.addWidget(cost_note)

        layout.addStretch()

        self.pages.addWidget(page)

    def _create_model_download_page(self):
        """Create model download page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        title = QLabel("Downloading Models")
        title.setFont(QFont('Cascadia Code', 14, QFont.Bold))
        title.setStyleSheet(f"color: {theme.TEXT_COLOR};")
        layout.addWidget(title)

        desc = QLabel("This will download ~3-4GB of AI models. Please wait...")
        desc.setFont(QFont('Cascadia Code', 10))
        desc.setStyleSheet(f"color: {theme.SECONDARY_TEXT};")
        layout.addWidget(desc)

        layout.addSpacing(20)

        # Progress bar
        self.model_progress = QProgressBar()
        self.model_progress.setStyleSheet(f"""
            QProgressBar {{
                background: {theme.INPUT_BG};
                border: 1px solid {theme.INPUT_BORDER};
                border-radius: 6px;
                height: 24px;
                text-align: center;
                color: {theme.TEXT_COLOR};
                font-family: 'Cascadia Code';
            }}
            QProgressBar::chunk {{
                background: {theme.TEXT_COLOR};
                border-radius: 5px;
            }}
        """)
        self.model_progress.setValue(0)
        layout.addWidget(self.model_progress)

        # Status label
        self.model_status = QLabel("Waiting to start...")
        self.model_status.setFont(QFont('Cascadia Code', 10))
        self.model_status.setStyleSheet(f"color: {theme.TEXT_COLOR};")
        self.model_status.setWordWrap(True)
        layout.addWidget(self.model_status)

        layout.addSpacing(10)

        note = QLabel("Note: First-time download may take 5-15 minutes depending on your internet speed.")
        note.setFont(QFont('Cascadia Code', 9))
        note.setStyleSheet(f"color: {theme.SECONDARY_TEXT};")
        note.setWordWrap(True)
        layout.addWidget(note)

        layout.addStretch()

        self.pages.addWidget(page)

    def _create_user_profile_page(self):
        """Create user profile page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        title = QLabel("Your Profile")
        title.setFont(QFont('Cascadia Code', 14, QFont.Bold))
        title.setStyleSheet(f"color: {theme.TEXT_COLOR};")
        layout.addWidget(title)

        desc = QLabel("Enter your name. This will be used to label your voice in meeting transcripts.")
        desc.setFont(QFont('Cascadia Code', 10))
        desc.setStyleSheet(f"color: {theme.SECONDARY_TEXT};")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        layout.addSpacing(16)

        name_label = QLabel("Your Name")
        name_label.setFont(QFont('Cascadia Code', 10))
        name_label.setStyleSheet(f"color: {theme.TEXT_COLOR};")
        layout.addWidget(name_label)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter your first name...")
        self.name_input.setStyleSheet(self._input_style())
        self.name_input.setMaxLength(50)
        layout.addWidget(self.name_input)

        layout.addStretch()

        self.pages.addWidget(page)

    def _create_voice_enrollment_page(self):
        """Create voice enrollment page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Title with optional label
        title_layout = QHBoxLayout()
        title = QLabel("Voice Enrollment")
        title.setFont(QFont('Cascadia Code', 14, QFont.Bold))
        title.setStyleSheet(f"color: {theme.TEXT_COLOR};")
        title_layout.addWidget(title)

        optional_label = QLabel("(Optional)")
        optional_label.setFont(QFont('Cascadia Code', 10))
        optional_label.setStyleSheet(f"color: {theme.SECONDARY_TEXT};")
        title_layout.addWidget(optional_label)
        title_layout.addStretch()
        layout.addLayout(title_layout)

        desc = QLabel("Record your voice so Koe can identify you in meetings.\nYou can skip this and enroll later via the tray menu.")
        desc.setFont(QFont('Cascadia Code', 10))
        desc.setStyleSheet(f"color: {theme.SECONDARY_TEXT};")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        layout.addSpacing(8)

        # Sample text to read
        sample_frame = QFrame()
        sample_frame.setStyleSheet(f"""
            QFrame {{
                background: {theme.INPUT_BG};
                border: 1px solid {theme.INPUT_BORDER};
                border-radius: 6px;
                padding: 8px;
            }}
        """)
        sample_layout = QVBoxLayout(sample_frame)
        sample_layout.setContentsMargins(12, 8, 12, 8)
        sample_layout.setSpacing(4)

        sample_header = QLabel("Read this aloud:")
        sample_header.setFont(QFont('Cascadia Code', 9))
        sample_header.setStyleSheet(f"color: {theme.SECONDARY_TEXT};")
        sample_layout.addWidget(sample_header)

        sample_text = QLabel(
            '"The quick brown fox jumps over the lazy dog. '
            'I\'m recording my voice so Koe can recognize me in meetings. '
            'This sample should take about ten to fifteen seconds to read at a normal pace."'
        )
        sample_text.setFont(QFont('Cascadia Code', 10))
        sample_text.setStyleSheet(f"color: {theme.TEXT_COLOR};")
        sample_text.setWordWrap(True)
        sample_layout.addWidget(sample_text)

        layout.addWidget(sample_frame)

        layout.addSpacing(8)

        # Recording indicator and status
        status_layout = QHBoxLayout()

        self.voice_indicator = QLabel("●")
        self.voice_indicator.setFont(QFont('Cascadia Code', 24))
        self.voice_indicator.setStyleSheet(f"color: {theme.DIM_TEXT};")
        status_layout.addWidget(self.voice_indicator)

        self.voice_status = QLabel("Press 'Start Recording' and read the text above")
        self.voice_status.setFont(QFont('Cascadia Code', 11))
        self.voice_status.setStyleSheet(f"color: {theme.TEXT_COLOR};")
        status_layout.addWidget(self.voice_status)

        status_layout.addStretch()

        self.voice_timer_label = QLabel("")
        self.voice_timer_label.setFont(QFont('Cascadia Code', 14))
        self.voice_timer_label.setStyleSheet(f"color: {theme.TEXT_COLOR};")
        status_layout.addWidget(self.voice_timer_label)

        layout.addLayout(status_layout)

        layout.addSpacing(8)

        # Record button
        btn_layout = QHBoxLayout()

        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setFont(QFont('Cascadia Code', 11))
        self.record_btn.setStyleSheet(self._button_style())
        self.record_btn.setFixedWidth(180)
        self.record_btn.clicked.connect(self._toggle_recording)
        btn_layout.addWidget(self.record_btn)

        btn_layout.addStretch()

        layout.addLayout(btn_layout)

        # Tips
        tips = QLabel("Tip: Speak in your natural voice. Background noise is okay but try to minimize it.")
        tips.setFont(QFont('Cascadia Code', 9))
        tips.setStyleSheet(f"color: {theme.SECONDARY_TEXT};")
        layout.addWidget(tips)

        layout.addStretch()

        self.pages.addWidget(page)

    def _create_folders_page(self):
        """Create folder configuration page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        title = QLabel("Output Folders")
        title.setFont(QFont('Cascadia Code', 14, QFont.Bold))
        title.setStyleSheet(f"color: {theme.TEXT_COLOR};")
        layout.addWidget(title)

        desc = QLabel("Choose where to save your transcriptions. You can change these later in Settings.")
        desc.setFont(QFont('Cascadia Code', 10))
        desc.setStyleSheet(f"color: {theme.SECONDARY_TEXT};")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        layout.addSpacing(16)

        # Meetings folder
        meetings_label = QLabel("Meetings Folder")
        meetings_label.setFont(QFont('Cascadia Code', 10))
        meetings_label.setStyleSheet(f"color: {theme.TEXT_COLOR};")
        layout.addWidget(meetings_label)

        meetings_row = QHBoxLayout()
        self.meetings_input = QLineEdit()
        default_meetings = str(Path(__file__).parent.parent / "Meetings")
        self.meetings_input.setText(default_meetings)
        self.meetings_input.setStyleSheet(self._input_style())
        meetings_row.addWidget(self.meetings_input)

        meetings_browse = QPushButton("Browse")
        meetings_browse.setStyleSheet(self._button_style(secondary=True))
        meetings_browse.clicked.connect(lambda: self._browse_folder(self.meetings_input))
        meetings_row.addWidget(meetings_browse)

        layout.addLayout(meetings_row)

        layout.addSpacing(12)

        # Snippets folder
        snippets_label = QLabel("Snippets Folder (hotkey transcriptions)")
        snippets_label.setFont(QFont('Cascadia Code', 10))
        snippets_label.setStyleSheet(f"color: {theme.TEXT_COLOR};")
        layout.addWidget(snippets_label)

        snippets_row = QHBoxLayout()
        self.snippets_input = QLineEdit()
        default_snippets = str(Path(__file__).parent.parent / "Snippets")
        self.snippets_input.setText(default_snippets)
        self.snippets_input.setStyleSheet(self._input_style())
        snippets_row.addWidget(self.snippets_input)

        snippets_browse = QPushButton("Browse")
        snippets_browse.setStyleSheet(self._button_style(secondary=True))
        snippets_browse.clicked.connect(lambda: self._browse_folder(self.snippets_input))
        snippets_row.addWidget(snippets_browse)

        layout.addLayout(snippets_row)

        layout.addStretch()

        self.pages.addWidget(page)

    def _create_complete_page(self):
        """Create completion page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        layout.addStretch()

        # Success icon
        icon = QLabel("✓")
        icon.setFont(QFont('Cascadia Code', 48))
        icon.setStyleSheet(f"color: {theme.SUCCESS_COLOR};")
        icon.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon)

        title = QLabel("Setup Complete!")
        title.setFont(QFont('Cascadia Code', 18, QFont.Bold))
        title.setStyleSheet(f"color: {theme.TEXT_COLOR};")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        desc = QLabel("Koe is ready to use. Here's how to get started:")
        desc.setFont(QFont('Cascadia Code', 10))
        desc.setStyleSheet(f"color: {theme.SECONDARY_TEXT};")
        desc.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc)

        layout.addSpacing(16)

        instructions = QLabel(
            "• Press Ctrl+Shift+Space to transcribe speech\n"
            "• Right-click tray icon → Start Scribe for meetings\n"
            "• Right-click tray icon → Settings to customize"
        )
        instructions.setFont(QFont('Cascadia Code', 10))
        instructions.setStyleSheet(f"color: {theme.TEXT_COLOR};")
        instructions.setAlignment(Qt.AlignCenter)
        layout.addWidget(instructions)

        layout.addStretch()

        self.pages.addWidget(page)

    def _browse_folder(self, input_field: QLineEdit):
        """Open folder browser dialog."""
        current = input_field.text() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", current)
        if folder:
            input_field.setText(folder)

    def _update_page(self):
        """Update UI for current page."""
        page_num = self.current_page.value
        self.pages.setCurrentIndex(page_num)
        self.progress_label.setText(f"Step {page_num + 1} of 7")

        # Update navigation buttons
        self.back_btn.setVisible(page_num > 0 and page_num < SetupPage.COMPLETE.value)

        # Update next button text
        if page_num == SetupPage.WELCOME.value:
            self.next_btn.setText("Get Started →")
            self.skip_btn.hide()
        elif page_num == SetupPage.COMPLETE.value:
            self.next_btn.setText("Launch Koe")
            self.skip_btn.hide()
            self.back_btn.hide()
        else:
            self.next_btn.setText("Next →")

        # Show skip button for optional steps
        if page_num == SetupPage.VOICE_ENROLLMENT.value:
            self.skip_btn.show()
            self.skip_btn.setText("Skip for now")
        elif page_num == SetupPage.API_KEYS.value:
            self.skip_btn.hide()  # HF token is required
        else:
            self.skip_btn.hide()

        # Run page-specific initialization
        if page_num == SetupPage.SYSTEM_CHECK.value:
            self._run_system_check()
        elif page_num == SetupPage.MODEL_DOWNLOAD.value:
            self._run_model_download()

    def _run_system_check(self):
        """Run system requirements check."""
        self.system_output.clear()
        self.system_status.setText("")
        self.next_btn.setEnabled(False)

        self._system_thread = SystemCheckThread()
        self._system_thread.progress.connect(self._on_system_progress)
        self._system_thread.finished.connect(self._on_system_finished)
        self._system_thread.start()

    def _on_system_progress(self, message: str, success: bool):
        """Handle system check progress."""
        color = theme.TEXT_COLOR if success else theme.ERROR_COLOR
        self.system_output.append(f'<span style="color: {color}">{message}</span>')

    def _on_system_finished(self, results: dict):
        """Handle system check completion."""
        self.system_results = results

        if results['python'] and results['gpu'] and results['cuda'] and results['packages']:
            self.system_status.setText("✓ All requirements met!")
            self.system_status.setStyleSheet(f"color: {theme.SUCCESS_COLOR};")
            self.next_btn.setEnabled(True)
        else:
            errors = results.get('errors', [])
            self.system_status.setText("Some requirements not met:\n" + "\n".join(errors))
            self.system_status.setStyleSheet(f"color: {theme.ERROR_COLOR};")
            # Still allow continuing if GPU is present (might work)
            if results['gpu']:
                self.next_btn.setEnabled(True)

    def _run_model_download(self):
        """Run model download."""
        self.model_progress.setValue(0)
        self.model_status.setText("Starting download...")
        self.next_btn.setEnabled(False)
        self.back_btn.setEnabled(False)

        self._model_thread = ModelDownloadThread(self.hf_token)
        self._model_thread.progress.connect(self._on_model_progress)
        self._model_thread.finished.connect(self._on_model_finished)
        self._model_thread.start()

    def _on_model_progress(self, message: str, percent: int):
        """Handle model download progress."""
        self.model_progress.setValue(percent)
        self.model_status.setText(message)

    def _on_model_finished(self, success: bool, message: str):
        """Handle model download completion."""
        self.back_btn.setEnabled(True)

        if success:
            self.model_status.setText(f"✓ {message}")
            self.model_status.setStyleSheet(f"color: {theme.SUCCESS_COLOR};")
            self.next_btn.setEnabled(True)
        else:
            self.model_status.setText(f"✗ {message}")
            self.model_status.setStyleSheet(f"color: {theme.ERROR_COLOR};")
            # Allow retry by going back
            self.model_progress.setValue(0)

    def _toggle_recording(self):
        """Toggle voice recording."""
        if self._voice_thread and self._voice_thread.isRunning():
            # Stop recording
            self._voice_thread.stop()
            self.record_btn.setEnabled(False)
            self.record_btn.setText("Processing...")
        else:
            # Start recording
            self.voice_audio = None
            self.voice_start_time = time.time()
            self.voice_timer.start(100)

            self.voice_indicator.setStyleSheet(f"color: {theme.RECORDING_COLOR};")
            self.voice_status.setText("Recording... Read the sample text aloud")
            self.record_btn.setText("Stop Recording")

            self._voice_thread = VoiceRecordThread()
            self._voice_thread.finished.connect(self._on_voice_finished)
            self._voice_thread.error.connect(self._on_voice_error)
            self._voice_thread.start()

    def _update_voice_timer(self):
        """Update voice recording timer."""
        if self.voice_start_time:
            elapsed = time.time() - self.voice_start_time
            self.voice_timer_label.setText(f"{elapsed:.1f}s")

            # Blink indicator
            if int(elapsed * 2) % 2:
                self.voice_indicator.setStyleSheet(f"color: {theme.RECORDING_COLOR};")
            else:
                self.voice_indicator.setStyleSheet(f"color: {theme.DIM_TEXT};")

    def _on_voice_finished(self, audio):
        """Handle voice recording completion."""
        self.voice_timer.stop()
        self.voice_indicator.setStyleSheet(f"color: {theme.DIM_TEXT};")

        import numpy as np

        if audio is None or len(audio) < 16000:  # Less than 1 second
            self.voice_status.setText("Recording too short. Try again.")
            self.voice_status.setStyleSheet(f"color: {theme.ERROR_COLOR};")
            self.record_btn.setText("Start Recording")
            self.record_btn.setEnabled(True)
            return

        self.voice_audio = audio
        duration = len(audio) / 16000

        self.voice_status.setText(f"Recorded {duration:.1f}s. Processing...")
        self.voice_status.setStyleSheet(f"color: {theme.TEXT_COLOR};")

        # Start enrollment
        self._enroll_thread = EnrollmentThread(self.user_name, audio)
        self._enroll_thread.finished.connect(self._on_enrollment_finished)
        self._enroll_thread.start()

    def _on_voice_error(self, error: str):
        """Handle voice recording error."""
        self.voice_timer.stop()
        self.voice_indicator.setStyleSheet(f"color: {theme.DIM_TEXT};")
        self.voice_status.setText(f"Error: {error}")
        self.voice_status.setStyleSheet(f"color: {theme.ERROR_COLOR};")
        self.record_btn.setText("Start Recording")
        self.record_btn.setEnabled(True)

    def _on_enrollment_finished(self, success: bool, message: str):
        """Handle enrollment completion."""
        self.record_btn.setText("Start Recording")
        self.record_btn.setEnabled(True)

        if success:
            self.voice_indicator.setStyleSheet(f"color: {theme.SUCCESS_COLOR};")
            self.voice_status.setText(f"✓ {message}")
            self.voice_status.setStyleSheet(f"color: {theme.SUCCESS_COLOR};")
            self.next_btn.setEnabled(True)
        else:
            self.voice_status.setText(f"✗ {message}")
            self.voice_status.setStyleSheet(f"color: {theme.ERROR_COLOR};")

    def _go_next(self):
        """Go to next page."""
        page = self.current_page

        # Validate current page
        if page == SetupPage.API_KEYS:
            self.hf_token = self.hf_input.text().strip()
            self.anthropic_key = self.anthropic_input.text().strip()

            if not self.hf_token or not self.hf_token.startswith("hf_"):
                QMessageBox.warning(self, "Invalid Token",
                    "Please enter a valid HuggingFace token starting with 'hf_'")
                return

        elif page == SetupPage.USER_PROFILE:
            self.user_name = self.name_input.text().strip()
            if not self.user_name:
                QMessageBox.warning(self, "Name Required",
                    "Please enter your name to continue.")
                return

        elif page == SetupPage.FOLDERS:
            self.meetings_folder = self.meetings_input.text().strip()
            self.snippets_folder = self.snippets_input.text().strip()

            # Create folders if they don't exist
            for folder in [self.meetings_folder, self.snippets_folder]:
                if folder:
                    Path(folder).mkdir(parents=True, exist_ok=True)

            # Save configuration
            self._save_configuration()

        elif page == SetupPage.COMPLETE:
            # Launch Koe
            self._launch_koe()
            return

        # Go to next page
        next_page = SetupPage(page.value + 1)
        self.current_page = next_page
        self._update_page()

    def _go_back(self):
        """Go to previous page."""
        if self.current_page.value > 0:
            self.current_page = SetupPage(self.current_page.value - 1)
            self._update_page()

    def _skip_step(self):
        """Skip optional step."""
        if self.current_page == SetupPage.VOICE_ENROLLMENT:
            self.current_page = SetupPage.FOLDERS
            self._update_page()

    def _save_configuration(self):
        """Save all configuration to files."""
        import yaml

        koe_dir = Path(__file__).parent.parent

        # Save .env file
        env_path = koe_dir / ".env"
        env_content = f"HF_TOKEN={self.hf_token}\n"
        if self.anthropic_key:
            env_content += f"ANTHROPIC_API_KEY={self.anthropic_key}\n"
        env_content += "WHISPER_SERVER_URL=http://localhost:9876\n"

        with open(env_path, 'w') as f:
            f.write(env_content)

        # Save config.yaml
        config = {
            'profile': {
                'user_name': self.user_name,
                'my_voice_embedding': self.user_name if self.voice_audio is not None else None
            },
            'meeting_options': {
                'root_folder': self.meetings_folder if self.meetings_folder else None
            },
            'recording_options': {
                'activation_key': 'ctrl+shift+space',
                'recording_mode': 'press_to_toggle',
                'sample_rate': 16000,
                'silence_duration': 900,
                'filter_snippets_to_my_voice': False
            },
            'misc': {
                'noise_on_completion': True,
                'snippets_folder': self.snippets_folder if self.snippets_folder else None,
                'print_to_terminal': True
            }
        }

        config_path = koe_dir / "src" / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Create folders
        if self.meetings_folder:
            Path(self.meetings_folder).mkdir(parents=True, exist_ok=True)
            (Path(self.meetings_folder) / "Transcripts").mkdir(exist_ok=True)
            (Path(self.meetings_folder) / "Summaries").mkdir(exist_ok=True)

        if self.snippets_folder:
            Path(self.snippets_folder).mkdir(parents=True, exist_ok=True)

        # Create setup complete marker
        marker_path = koe_dir / ".setup_complete"
        marker_path.touch()

    def _launch_koe(self):
        """Launch Koe application."""
        koe_dir = Path(__file__).parent.parent
        run_path = koe_dir / "run.py"

        # Start Koe in background
        if sys.platform == 'win32':
            subprocess.Popen([sys.executable, str(run_path)],
                           creationflags=subprocess.CREATE_NO_WINDOW,
                           cwd=str(koe_dir))
        else:
            subprocess.Popen([sys.executable, str(run_path)],
                           cwd=str(koe_dir))

        # Close wizard
        self.close()

    def _confirm_close(self):
        """Confirm before closing wizard."""
        if self.current_page.value < SetupPage.COMPLETE.value:
            reply = QMessageBox.question(
                self, "Cancel Setup",
                "Are you sure you want to cancel setup?\nYou can run it again later.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        self.close()

    def show(self):
        """Show wizard centered on screen."""
        super().show()
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

    # Dragging support
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self._drag_pos:
            self.move(event.globalPos() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None


def needs_setup() -> bool:
    """Check if setup wizard needs to run."""
    koe_dir = Path(__file__).parent.parent

    # Check for setup complete marker
    if (koe_dir / ".setup_complete").exists():
        return False

    # Check for existing config
    config_path = koe_dir / "src" / "config.yaml"
    env_path = koe_dir / ".env"

    if config_path.exists() and env_path.exists():
        # Check if env has HF_TOKEN
        with open(env_path) as f:
            if "HF_TOKEN=" in f.read():
                return False

    return True


def run_wizard():
    """Run the setup wizard."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    wizard = SetupWizard()
    wizard.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    run_wizard()
