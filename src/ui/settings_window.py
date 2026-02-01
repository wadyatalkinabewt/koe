import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from PyQt5.QtWidgets import (
    QApplication, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QCheckBox, QMessageBox, QWidget, QFrame, QScrollArea,
    QSizePolicy, QFileDialog, QGroupBox, QTextEdit
)
from PyQt5.QtCore import Qt, pyqtSignal, QRectF
from PyQt5.QtGui import QFont, QPainter, QBrush, QColor, QPainterPath, QPen, QIcon

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ui.base_window import BaseWindow
from ui import theme
from utils import ConfigManager

# Import engine factory for checking available engines
try:
    from engines import get_available_engines, is_engine_available, create_engine
    ENGINES_AVAILABLE = True
except ImportError:
    ENGINES_AVAILABLE = False
    def get_available_engines():
        return ["whisper"]
    def is_engine_available(engine_id):
        return engine_id == "whisper"
    def create_engine(engine_id):
        return None

load_dotenv()


class SettingsWindow(BaseWindow):
    settings_closed = pyqtSignal()
    settings_saved = pyqtSignal()

    # Terminal color scheme (from centralized theme)
    BG_COLOR = theme.BG_COLOR
    BORDER_COLOR = theme.BORDER_COLOR
    TEXT_COLOR = theme.TEXT_COLOR
    SECONDARY_TEXT = theme.SECONDARY_TEXT
    DIM_TEXT = theme.DIM_TEXT
    INPUT_BG = theme.INPUT_BG
    INPUT_BORDER = theme.INPUT_BORDER

    def __init__(self):
        """Initialize the settings window."""
        super().__init__('Settings', 500, 700)
        self.schema = ConfigManager.get_schema()

        # Set application icon for taskbar
        icon_path = str(Path(__file__).parent.parent.parent / "assets" / "koe-icon.ico")
        app = QApplication.instance()
        if app:
            app.setWindowIcon(QIcon(icon_path))
        # Also set window icon
        self.setWindowIcon(QIcon(icon_path))

        # Style close button for terminal theme
        for btn in self.findChildren(QPushButton):
            if btn.text() == '×':
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: transparent;
                        border: none;
                        color: {self.TEXT_COLOR};
                        font-size: 18px;
                    }}
                    QPushButton:hover {{
                        color: #ff6666;
                    }}
                """)
                break

        self.init_settings_ui()

        # Capture original values to detect changes
        self._original_values = self._get_current_values()

    def _get_current_values(self) -> dict:
        """Get current values of all settings fields."""
        values = {
            'user_name': self.user_name_input[1].text().strip() or None,
            'my_voice': self.my_voice_dropdown.currentData(),
            'filter_snippets': self.filter_snippets_checkbox.isChecked(),
            'meetings_folder': self.meetings_folder_input[1].text().strip() or None,
            'snippets_folder': self.snippets_folder_input[1].text().strip() or None,
            'activation_key': self.activation_key_input[1].text().strip() or "ctrl+shift+space",
            'sound_on_completion': self.sound_checkbox.isChecked(),
            'initial_prompt': self.initial_prompt_input.toPlainText().strip() or None,
        }
        # Add engine settings if available
        if hasattr(self, 'engine_dropdown'):
            values['engine'] = self.engine_dropdown.currentData()
        if hasattr(self, 'model_dropdown'):
            values['model'] = self.model_dropdown.currentData()
        if hasattr(self, 'device_dropdown'):
            values['device'] = self.device_dropdown.currentData()
        return values

    def init_settings_ui(self):
        """Initialize the terminal-style settings UI."""
        # Apply terminal dark theme with green accents
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {self.BG_COLOR};
                color: {self.TEXT_COLOR};
                font-family: 'Cascadia Code', 'Consolas', 'Courier New', monospace;
            }}
            QGroupBox {{
                font-size: 13px;
                font-weight: 600;
                border: 1px solid {self.INPUT_BORDER};
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                color: {self.TEXT_COLOR};
            }}
            QLineEdit, QComboBox {{
                background-color: {self.INPUT_BG};
                border: 1px solid {self.INPUT_BORDER};
                border-radius: 4px;
                padding: 10px 12px;
                font-size: 13px;
                color: {self.TEXT_COLOR};
                font-family: 'Cascadia Code', 'Consolas', monospace;
                min-height: 20px;
            }}
            QLineEdit:focus, QComboBox:focus {{
                border-color: {self.BORDER_COLOR};
            }}
            QLineEdit::placeholder {{
                color: {self.DIM_TEXT};
            }}
            QTextEdit {{
                background-color: {self.INPUT_BG};
                border: 1px solid {self.INPUT_BORDER};
                border-radius: 4px;
                padding: 8px;
                font-size: 13px;
                color: {self.TEXT_COLOR};
                font-family: 'Cascadia Code', 'Consolas', monospace;
            }}
            QTextEdit:focus {{
                border-color: {self.BORDER_COLOR};
            }}
            QComboBox::drop-down {{
                border: none;
                padding-right: 8px;
            }}
            QComboBox::down-arrow {{
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid {self.TEXT_COLOR};
                margin-right: 8px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {self.INPUT_BG};
                border: 1px solid {self.INPUT_BORDER};
                color: {self.TEXT_COLOR};
                selection-background-color: {self.INPUT_BORDER};
            }}
            QCheckBox {{
                spacing: 8px;
                font-size: 13px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 3px;
                border: 1px solid {self.INPUT_BORDER};
                background-color: {self.INPUT_BG};
            }}
            QCheckBox::indicator:checked {{
                background-color: {self.BORDER_COLOR};
                border-color: {self.BORDER_COLOR};
            }}
            QPushButton {{
                background-color: #1a2a2a;
                border: 1px solid #2a4a3a;
                border-radius: 4px;
                padding: 12px 24px;
                font-size: 13px;
                font-weight: 500;
                color: {self.TEXT_COLOR};
                font-family: 'Cascadia Code', 'Consolas', monospace;
            }}
            QPushButton:hover {{
                background-color: #2a3a3a;
                border-color: {self.BORDER_COLOR};
            }}
            QPushButton:pressed {{
                background-color: #0a1a1a;
            }}
            QPushButton#saveButton {{
                background-color: #1a3a2a;
                border: 1px solid {self.BORDER_COLOR};
                font-weight: 600;
            }}
            QPushButton#saveButton:hover {{
                background-color: #2a4a3a;
            }}
            QLabel {{
                font-size: 13px;
            }}
            QLabel#headerLabel {{
                font-size: 20px;
                font-weight: 600;
                color: {self.TEXT_COLOR};
            }}
            QLabel#helpText {{
                color: {self.DIM_TEXT};
                font-size: 11px;
            }}
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            QScrollBar:vertical {{
                background-color: {self.INPUT_BG};
                width: 12px;
                border-radius: 6px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {self.INPUT_BORDER};
                border-radius: 6px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {self.BORDER_COLOR};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
            }}
        """)

        # Create scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 8, 24, 24)
        content_layout.setSpacing(16)

        # Header
        header = QLabel("> Settings_")
        header.setObjectName("headerLabel")
        content_layout.addWidget(header)

        # ===== PROFILE SECTION =====
        profile_group = self._create_section("Profile")
        profile_layout = QVBoxLayout()
        profile_layout.setSpacing(12)

        # User name
        self.user_name_input = self._create_text_field(
            "Your Name",
            "Used to label your voice in Scribe",
            ConfigManager.get_config_value('profile', 'user_name') or ""
        )
        profile_layout.addLayout(self.user_name_input[0])

        profile_group.setLayout(profile_layout)
        content_layout.addWidget(profile_group)

        # ===== OUTPUT FOLDERS SECTION =====
        output_group = self._create_section("Output Folders")
        output_layout = QVBoxLayout()
        output_layout.setSpacing(12)

        # Meetings root folder picker
        self.meetings_folder_input = self._create_folder_picker(
            "Meetings Root Folder",
            ConfigManager.get_config_value('meeting_options', 'root_folder') or ""
        )
        output_layout.addLayout(self.meetings_folder_input[0])

        # Help text for meetings folder
        meeting_help = QLabel("// category subfolders (Standups, Investors) go inside Transcripts/")
        meeting_help.setObjectName("helpText")
        meeting_help.setWordWrap(True)
        output_layout.addWidget(meeting_help)

        # Snippets folder picker
        self.snippets_folder_input = self._create_folder_picker(
            "Snippets Folder",
            ConfigManager.get_config_value('misc', 'snippets_folder') or ""
        )
        output_layout.addLayout(self.snippets_folder_input[0])

        # Help text for snippets
        snippets_help = QLabel("// rolling transcriptions from hotkey mode (default: Koe/Snippets)")
        snippets_help.setObjectName("helpText")
        snippets_help.setWordWrap(True)
        output_layout.addWidget(snippets_help)

        output_group.setLayout(output_layout)
        content_layout.addWidget(output_group)

        # ===== TRANSCRIPTION ENGINE SECTION =====
        engine_group = self._create_section("Transcription Engine")
        engine_layout = QVBoxLayout()
        engine_layout.setSpacing(12)

        engine_help = QLabel("// engine used by the transcription server")
        engine_help.setObjectName("helpText")
        engine_layout.addWidget(engine_help)

        # Engine dropdown
        engine_label = QLabel("Engine")
        engine_layout.addWidget(engine_label)

        self.engine_dropdown = QComboBox()
        self.engine_dropdown.setFocusPolicy(Qt.StrongFocus)
        self.engine_dropdown.wheelEvent = lambda e: e.ignore()
        self._populate_engine_dropdown()
        self.engine_dropdown.currentIndexChanged.connect(self._on_engine_changed)
        engine_layout.addWidget(self.engine_dropdown)

        # Model dropdown (changes based on engine)
        model_label = QLabel("Model")
        engine_layout.addWidget(model_label)

        self.model_dropdown = QComboBox()
        self.model_dropdown.setFocusPolicy(Qt.StrongFocus)
        self.model_dropdown.wheelEvent = lambda e: e.ignore()
        self._populate_model_dropdown()
        engine_layout.addWidget(self.model_dropdown)

        # Model info label
        self.model_info_label = QLabel("")
        self.model_info_label.setObjectName("helpText")
        self.model_info_label.setWordWrap(True)
        engine_layout.addWidget(self.model_info_label)
        self._update_model_info()

        # Device dropdown
        device_label = QLabel("Device")
        engine_layout.addWidget(device_label)

        self.device_dropdown = QComboBox()
        self.device_dropdown.setFocusPolicy(Qt.StrongFocus)
        self.device_dropdown.wheelEvent = lambda e: e.ignore()
        self._populate_device_dropdown()
        engine_layout.addWidget(self.device_dropdown)

        device_help = QLabel("// auto detects GPU, use CPU if no NVIDIA GPU")
        device_help.setObjectName("helpText")
        engine_layout.addWidget(device_help)

        # Restart warning
        restart_label = QLabel("// changing engine/device requires server restart")
        restart_label.setObjectName("helpText")
        engine_layout.addWidget(restart_label)

        engine_group.setLayout(engine_layout)
        content_layout.addWidget(engine_group)

        # ===== ENROLLED SPEAKERS SECTION =====
        speakers_group = self._create_section("Enrolled Speakers")
        speakers_layout = QVBoxLayout()
        speakers_layout.setSpacing(8)

        speakers_help = QLabel("// voice fingerprints for speaker identification in Scribe")
        speakers_help.setObjectName("helpText")
        speakers_layout.addWidget(speakers_help)

        # Container for speaker list (will be populated/refreshed)
        self.speakers_container = QVBoxLayout()
        self.speakers_container.setSpacing(4)
        self._refresh_speakers_list()
        speakers_layout.addLayout(self.speakers_container)

        # My Voice dropdown
        speakers_layout.addSpacing(8)
        my_voice_label = QLabel("My Voice")
        speakers_layout.addWidget(my_voice_label)

        self.my_voice_dropdown = QComboBox()
        self.my_voice_dropdown.setFocusPolicy(Qt.StrongFocus)
        self.my_voice_dropdown.wheelEvent = lambda e: e.ignore()
        self._populate_my_voice_dropdown()
        self.my_voice_dropdown.currentIndexChanged.connect(self._on_my_voice_changed)
        speakers_layout.addWidget(self.my_voice_dropdown)

        my_voice_help = QLabel("// required for snippet voice filtering")
        my_voice_help.setObjectName("helpText")
        speakers_layout.addWidget(my_voice_help)

        # Filter snippets checkbox
        speakers_layout.addSpacing(8)
        self.filter_snippets_checkbox = QCheckBox("Filter snippets to my voice")
        self.filter_snippets_checkbox.setChecked(
            ConfigManager.get_config_value('recording_options', 'filter_snippets_to_my_voice') or False
        )
        speakers_layout.addWidget(self.filter_snippets_checkbox)

        filter_help = QLabel("// adds ~0.5-1s latency; use in noisy environments")
        filter_help.setObjectName("helpText")
        speakers_layout.addWidget(filter_help)

        # Update checkbox enabled state based on dropdown
        self._update_filter_checkbox_state()

        speakers_group.setLayout(speakers_layout)
        content_layout.addWidget(speakers_group)

        # ===== RECORDING SECTION =====
        recording_group = self._create_section("Recording")
        recording_layout = QVBoxLayout()
        recording_layout.setSpacing(12)

        # Mode description
        mode_label = QLabel("// press hotkey to start, press again to stop")
        mode_label.setObjectName("helpText")
        recording_layout.addWidget(mode_label)

        # Activation key
        self.activation_key_input = self._create_text_field(
            "Toggle Hotkey",
            "e.g., ctrl+shift+space",
            ConfigManager.get_config_value('recording_options', 'activation_key') or "ctrl+shift+space"
        )
        recording_layout.addLayout(self.activation_key_input[0])

        # Sound on completion
        self.sound_checkbox = self._create_checkbox(
            "Play sound on completion",
            ConfigManager.get_config_value('misc', 'noise_on_completion') or False
        )
        recording_layout.addWidget(self.sound_checkbox)

        recording_group.setLayout(recording_layout)
        content_layout.addWidget(recording_group)

        # ===== TRANSCRIPTION PROMPT SECTION =====
        prompt_group = self._create_section("Transcription Prompt")
        prompt_layout = QVBoxLayout()
        prompt_layout.setSpacing(8)

        prompt_help = QLabel("// hint for Whisper (names, punctuation style)")
        prompt_help.setObjectName("helpText")
        prompt_layout.addWidget(prompt_help)

        self.initial_prompt_input = self._create_multiline_field(
            ConfigManager.get_config_value('model_options', 'common', 'initial_prompt') or ""
        )
        prompt_layout.addWidget(self.initial_prompt_input)

        prompt_group.setLayout(prompt_layout)
        content_layout.addWidget(prompt_group)

        # Spacer
        content_layout.addStretch()

        scroll.setWidget(content)
        self.main_layout.addWidget(scroll)

        # ===== BUTTONS =====
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.close)
        button_layout.addWidget(cancel_button)

        button_layout.addStretch()

        save_button = QPushButton("Save")
        save_button.setObjectName("saveButton")
        save_button.clicked.connect(self.save_settings)
        button_layout.addWidget(save_button)

        self.main_layout.addLayout(button_layout)

    def _create_section(self, title: str) -> QGroupBox:
        """Create a styled section group."""
        group = QGroupBox(title)
        return group

    def _create_text_field(self, label: str, hint: str, value: str, password: bool = False):
        """Create a labeled text input field."""
        layout = QVBoxLayout()
        layout.setSpacing(4)

        lbl = QLabel(label)
        layout.addWidget(lbl)

        field = QLineEdit(value or "")
        field.setPlaceholderText(hint)
        if password:
            field.setEchoMode(QLineEdit.Password)
        layout.addWidget(field)

        return (layout, field)

    def _create_multiline_field(self, value: str) -> QTextEdit:
        """Create a multi-line text input field."""
        field = QTextEdit()
        field.setAcceptRichText(False)  # Plain text only - prevents formatting on paste
        field.setPlainText(value or "")
        field.setMinimumHeight(60)
        field.setMaximumHeight(80)
        field.setPlaceholderText("e.g., Use proper punctuation including periods, commas, and question marks.")
        return field

    def _create_dropdown(self, label: str, options: list, current: str, display_map: dict = None):
        """Create a labeled dropdown."""
        layout = QVBoxLayout()
        layout.setSpacing(4)

        lbl = QLabel(label)
        layout.addWidget(lbl)

        combo = QComboBox()
        combo.setFocusPolicy(Qt.StrongFocus)
        combo.wheelEvent = lambda e: e.ignore()
        for opt in options:
            display_text = display_map.get(opt, opt) if display_map else opt
            combo.addItem(display_text, opt)

        # Set current value
        for i in range(combo.count()):
            if combo.itemData(i) == current:
                combo.setCurrentIndex(i)
                break

        layout.addWidget(combo)

        return (layout, combo)

    def _create_checkbox(self, label: str, checked: bool) -> QCheckBox:
        """Create a styled checkbox."""
        cb = QCheckBox(label)
        cb.setChecked(checked)
        return cb

    def _create_folder_picker(self, label: str, value: str):
        """Create a folder picker with browse button."""
        layout = QVBoxLayout()
        layout.setSpacing(4)

        lbl = QLabel(label)
        layout.addWidget(lbl)

        h_layout = QHBoxLayout()
        h_layout.setSpacing(8)

        field = QLineEdit(value or "")
        field.setPlaceholderText("leave empty for default")
        h_layout.addWidget(field)

        browse_btn = QPushButton("Browse")
        browse_btn.setMinimumWidth(80)
        browse_btn.clicked.connect(lambda: self._browse_folder(field))
        h_layout.addWidget(browse_btn)

        layout.addLayout(h_layout)
        return (layout, field)

    def _browse_folder(self, field: QLineEdit):
        """Open folder picker dialog."""
        start_dir = field.text() or str(Path.home() / "Desktop")
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder",
            start_dir
        )
        if folder:
            field.setText(folder)

    def save_settings(self):
        """Save all settings."""
        # Check if anything actually changed
        current_values = self._get_current_values()
        if current_values == self._original_values:
            # Nothing changed, just close
            self.close()
            return

        # Profile
        user_name = current_values['user_name']
        ConfigManager.set_config_value(user_name, 'profile', 'user_name')

        my_voice = current_values['my_voice']
        ConfigManager.set_config_value(my_voice, 'profile', 'my_voice_embedding')

        # Voice filtering
        ConfigManager.set_config_value(
            current_values['filter_snippets'],
            'recording_options', 'filter_snippets_to_my_voice'
        )

        # Output Folders
        ConfigManager.set_config_value(current_values['meetings_folder'], 'meeting_options', 'root_folder')
        ConfigManager.set_config_value(current_values['snippets_folder'], 'misc', 'snippets_folder')

        # Recording
        ConfigManager.set_config_value(
            current_values['activation_key'],
            'recording_options', 'activation_key'
        )
        ConfigManager.set_config_value(
            current_values['sound_on_completion'],
            'misc', 'noise_on_completion'
        )

        # Advanced - Prompt
        ConfigManager.set_config_value(current_values['initial_prompt'], 'model_options', 'common', 'initial_prompt')

        # Engine settings
        if 'engine' in current_values:
            ConfigManager.set_config_value(current_values['engine'], 'model_options', 'engine')
        if current_values.get('engine'):
            engine = current_values['engine']
            if engine == 'whisper':
                if 'model' in current_values:
                    ConfigManager.set_config_value(current_values['model'], 'model_options', 'whisper', 'model')
                if 'device' in current_values:
                    ConfigManager.set_config_value(current_values['device'], 'model_options', 'whisper', 'device')
            elif engine == 'parakeet':
                if 'model' in current_values:
                    ConfigManager.set_config_value(current_values['model'], 'model_options', 'parakeet', 'model')
                if 'device' in current_values:
                    ConfigManager.set_config_value(current_values['device'], 'model_options', 'parakeet', 'device')

        ConfigManager.save_config()

        QMessageBox.information(
            self,
            'Settings Saved',
            'Settings have been saved. The application will now restart.'
        )
        self.settings_saved.emit()
        self.close()

    def closeEvent(self, event):
        """Handle window close."""
        self.settings_closed.emit()
        event.accept()

    def _get_speakers_dir(self) -> Path:
        """Get the speaker embeddings directory."""
        return Path(__file__).parent.parent.parent / "speaker_embeddings"

    def _get_enrolled_speakers(self) -> list:
        """Get list of enrolled speaker names."""
        speakers_dir = self._get_speakers_dir()
        if not speakers_dir.exists():
            return []
        return sorted([f.stem for f in speakers_dir.glob("*.npy")])

    def _refresh_speakers_list(self):
        """Refresh the enrolled speakers list in the UI."""
        # Clear existing items
        while self.speakers_container.count():
            item = self.speakers_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        speakers = self._get_enrolled_speakers()

        if not speakers:
            no_speakers = QLabel("No speakers enrolled")
            no_speakers.setStyleSheet(f"color: {self.DIM_TEXT}; font-style: italic;")
            self.speakers_container.addWidget(no_speakers)
            return

        for speaker in speakers:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 2, 0, 2)
            row_layout.setSpacing(8)

            # Speaker name label
            name_label = QLabel(speaker)
            name_label.setStyleSheet(f"color: {self.TEXT_COLOR};")
            row_layout.addWidget(name_label)

            row_layout.addStretch()

            # Delete button
            delete_btn = QPushButton("×")
            delete_btn.setFixedSize(24, 24)
            delete_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    border: 1px solid {self.INPUT_BORDER};
                    border-radius: 4px;
                    color: {self.DIM_TEXT};
                    font-size: 14px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: #3a1a1a;
                    border-color: #ff6666;
                    color: #ff6666;
                }}
            """)
            delete_btn.setToolTip(f"Remove {speaker}")
            delete_btn.clicked.connect(lambda checked, s=speaker: self._delete_speaker(s))
            row_layout.addWidget(delete_btn)

            self.speakers_container.addWidget(row)

    def _delete_speaker(self, speaker_name: str):
        """Delete a speaker's voice fingerprint."""
        reply = QMessageBox.question(
            self,
            "Remove Speaker",
            f"Remove voice fingerprint for '{speaker_name}'?\n\nThey will appear as 'Speaker N' in future meetings.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            speaker_file = self._get_speakers_dir() / f"{speaker_name}.npy"
            try:
                if speaker_file.exists():
                    speaker_file.unlink()
                self._refresh_speakers_list()
                # Also refresh the my voice dropdown
                self._populate_my_voice_dropdown()
                self._update_filter_checkbox_state()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to remove speaker: {e}")

    def _populate_my_voice_dropdown(self):
        """Populate the My Voice dropdown with enrolled speakers."""
        self.my_voice_dropdown.clear()
        self.my_voice_dropdown.addItem("(None)", None)

        speakers = self._get_enrolled_speakers()
        current_voice = ConfigManager.get_config_value('profile', 'my_voice_embedding')

        selected_index = 0
        for i, speaker in enumerate(speakers):
            self.my_voice_dropdown.addItem(speaker, speaker)
            if speaker == current_voice:
                selected_index = i + 1  # +1 because of "(None)" at index 0

        self.my_voice_dropdown.setCurrentIndex(selected_index)

    def _on_my_voice_changed(self):
        """Handle My Voice dropdown selection change."""
        self._update_filter_checkbox_state()

    def _update_filter_checkbox_state(self):
        """Enable/disable filter checkbox based on whether a voice is selected."""
        has_voice = self.my_voice_dropdown.currentData() is not None
        self.filter_snippets_checkbox.setEnabled(has_voice)
        if not has_voice:
            self.filter_snippets_checkbox.setChecked(False)

    def _populate_engine_dropdown(self):
        """Populate the engine dropdown with available engines."""
        self.engine_dropdown.clear()

        engines = [
            ("whisper", "Whisper (faster-whisper)", "Well-tested, multilingual"),
            ("parakeet", "Parakeet (NVIDIA NeMo)", "~50x faster, English only"),
        ]

        current_engine = ConfigManager.get_config_value('model_options', 'engine') or 'whisper'
        selected_index = 0

        for i, (engine_id, name, description) in enumerate(engines):
            available = is_engine_available(engine_id)
            if available:
                self.engine_dropdown.addItem(name, engine_id)
            else:
                # Show unavailable engines as disabled with hint
                display_name = f"{name} (not installed)"
                self.engine_dropdown.addItem(display_name, engine_id)
                # Disable the item
                model = self.engine_dropdown.model()
                item = model.item(self.engine_dropdown.count() - 1)
                item.setEnabled(False)

            if engine_id == current_engine and available:
                selected_index = i

        self.engine_dropdown.setCurrentIndex(selected_index)

    def _populate_model_dropdown(self):
        """Populate the model dropdown based on selected engine."""
        self.model_dropdown.clear()

        engine_id = self.engine_dropdown.currentData() if hasattr(self, 'engine_dropdown') else 'whisper'

        if engine_id == 'whisper':
            models = [
                ("tiny", "Tiny (~75MB)", 500),
                ("tiny.en", "Tiny English (~75MB)", 500),
                ("base", "Base (~150MB)", 600),
                ("base.en", "Base English (~150MB)", 600),
                ("small", "Small (~500MB)", 1000),
                ("small.en", "Small English (~500MB)", 1000),
                ("medium", "Medium (~1.5GB)", 2000),
                ("medium.en", "Medium English (~1.5GB)", 2000),
                ("large-v3", "Large v3 (~3GB) - Best accuracy", 4000),
                ("large-v2", "Large v2 (~3GB)", 4000),
            ]
            current_model = ConfigManager.get_config_value('model_options', 'whisper', 'model') or 'large-v3'
        elif engine_id == 'parakeet':
            # Note: TDT models disabled due to CUDA 12.8 incompatibility
            models = [
                ("nvidia/parakeet-ctc-0.6b", "Parakeet CTC 0.6B - Recommended", 2000),
                ("nvidia/parakeet-ctc-1.1b", "Parakeet CTC 1.1B - Higher accuracy", 3000),
            ]
            current_model = ConfigManager.get_config_value('model_options', 'parakeet', 'model') or 'nvidia/parakeet-ctc-0.6b'
        else:
            models = []
            current_model = None

        selected_index = 0
        for i, (model_id, name, vram) in enumerate(models):
            self.model_dropdown.addItem(name, model_id)
            if model_id == current_model:
                selected_index = i

        if models:
            self.model_dropdown.setCurrentIndex(selected_index)

    def _on_engine_changed(self):
        """Handle engine dropdown selection change."""
        self._populate_model_dropdown()
        self._populate_device_dropdown()
        self._update_model_info()

    def _update_model_info(self):
        """Update the model info label based on current selection."""
        if not hasattr(self, 'model_info_label'):
            return

        engine_id = self.engine_dropdown.currentData() if hasattr(self, 'engine_dropdown') else 'whisper'
        model_id = self.model_dropdown.currentData() if hasattr(self, 'model_dropdown') else None

        if engine_id == 'whisper':
            vram_map = {
                'tiny': 500, 'tiny.en': 500,
                'base': 600, 'base.en': 600,
                'small': 1000, 'small.en': 1000,
                'medium': 2000, 'medium.en': 2000,
                'large-v1': 4000, 'large-v2': 4000, 'large-v3': 4000, 'large': 4000,
            }
            vram = vram_map.get(model_id, 2000)
            self.model_info_label.setText(f"// ~{vram}MB VRAM required")
        elif engine_id == 'parakeet':
            if not is_engine_available('parakeet'):
                import sys
                if sys.platform == 'win32':
                    self.model_info_label.setText("// Requires WSL2 on Windows. Run server in WSL.")
                else:
                    self.model_info_label.setText("// pip install nemo_toolkit[asr]")
            else:
                self.model_info_label.setText("// ~2GB VRAM, English only, ~50x faster")
        else:
            self.model_info_label.setText("")

    def _populate_device_dropdown(self):
        """Populate the device dropdown with available options."""
        self.device_dropdown.clear()

        devices = [
            ("auto", "Auto (detect GPU)"),
            ("cuda", "CUDA (NVIDIA GPU)"),
            ("cpu", "CPU (no GPU)"),
        ]

        engine_id = self.engine_dropdown.currentData() if hasattr(self, 'engine_dropdown') else 'whisper'

        # Get current device from config
        if engine_id == 'whisper':
            current_device = ConfigManager.get_config_value('model_options', 'whisper', 'device') or 'auto'
        elif engine_id == 'parakeet':
            current_device = ConfigManager.get_config_value('model_options', 'parakeet', 'device') or 'auto'
        else:
            current_device = 'auto'

        selected_index = 0
        for i, (device_id, name) in enumerate(devices):
            self.device_dropdown.addItem(name, device_id)
            if device_id == current_device:
                selected_index = i

        self.device_dropdown.setCurrentIndex(selected_index)

    def paintEvent(self, event):
        """Override to use terminal dark background with green border."""
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 6, 6)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Fill background
        painter.setBrush(QBrush(QColor(10, 10, 15)))  # #0a0a0f
        painter.setPen(Qt.NoPen)
        painter.drawPath(path)

        # Draw border
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(0, 255, 136), 1))  # #00ff88
        painter.drawPath(path)
