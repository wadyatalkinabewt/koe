"""
Koe — hotkey transcription app.

Entry point. Boots a QApplication, registers the global hotkey, and runs
in the system tray. Audio capture happens in ResultThread; transcription
is delegated to transcription.transcribe() which calls Groq + cleanup.
"""

import os
import sys
import time
import threading
from pathlib import Path
from datetime import datetime

import pyperclip
from PyQt5.QtCore import QObject, QProcess, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QAction, QMessageBox

from compat import (
    acquire_single_instance_lock,
    release_single_instance_lock,
    set_app_user_model_id,
    clipboard_copy_fallback,
    play_sound_file,
)
from key_listener import KeyListener
from result_thread import ResultThread
from ui.main_window import MainWindow
from ui.settings_window import SettingsWindow
from ui.status_window import StatusWindow
from ui.initialization_window import InitializationWindow
from utils import ConfigManager

_DEBUG_LOG = Path(__file__).parent.parent / "logs" / "debug.log"


def _debug(msg: str):
    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass


class KoeApp(QObject):
    # Minimum recording time before hotkey can stop. Prevents accidental double-press from discarding recordings.
    MIN_RECORDING_SECONDS = 1.0

    def __init__(self, qapp=None, init_window=None):
        super().__init__()
        self.continuous_stopped = False
        self._instance_lock = acquire_single_instance_lock()
        self.recording_start_time = None
        self.processing_result = False
        self._thread_lock = threading.Lock()
        self.app = qapp if qapp else QApplication(sys.argv)
        self.init_window = init_window

        if not qapp:
            set_app_user_model_id()
            icon_path = str(Path(__file__).parent.parent / "assets" / "koe-icon.ico")
            self.app.setWindowIcon(QIcon(icon_path))
            self.app.setQuitOnLastWindowClosed(False)
            if not init_window:
                self.init_window = InitializationWindow()
                self.init_window.show()
                self.app.processEvents()

        if ConfigManager._instance is None:
            ConfigManager.initialize()

        self.settings_window = SettingsWindow()
        self.settings_window.settings_closed.connect(self.on_settings_closed)
        self.settings_window.settings_saved.connect(self.restart_app)

        if ConfigManager.config_file_exists():
            self.initialize_components()
        else:
            print("No valid configuration file found. Opening settings window...")
            if self.init_window:
                self.init_window.close()
            self.settings_window.show()

    def initialize_components(self):
        self.key_listener = KeyListener()
        self.key_listener.add_callback("on_activate", self.on_activation)
        self.key_listener.add_callback("on_deactivate", self.on_deactivation)

        self.result_thread = None

        self.main_window = MainWindow()
        self.main_window.openSettings.connect(self.settings_window.show)
        self.main_window.startListening.connect(self.key_listener.start)
        self.main_window.closeApp.connect(self.exit_app)

        if not ConfigManager.get_config_value("misc", "hide_status_window"):
            self.status_window = StatusWindow()

        self.create_tray_icon()
        self.key_listener.start()

        if self.init_window:
            self.init_window.close()

    def create_tray_icon(self):
        from ui.theme import (BG_COLOR, TEXT_COLOR, SECONDARY_TEXT,
                              BUTTON_HOVER_BG, INPUT_BORDER)

        icon_path = str(Path(__file__).parent.parent / "assets" / "koe-icon.ico")
        self.tray_icon = QSystemTrayIcon(QIcon(icon_path), self.app)

        menu_style = f"""
            QMenu {{
                background-color: {BG_COLOR};
                color: {TEXT_COLOR};
                border: 1px solid {INPUT_BORDER};
                border-radius: 8px;
                padding: 8px 4px;
                font-family: 'Segoe UI', 'Cascadia Code', Consolas, monospace;
                font-size: 11pt;
            }}
            QMenu::item {{
                padding: 10px 32px 10px 20px;
                border-radius: 4px;
                margin: 2px 4px;
            }}
            QMenu::item:selected {{ background-color: {BUTTON_HOVER_BG}; color: {TEXT_COLOR}; }}
            QMenu::item:disabled {{ color: {SECONDARY_TEXT}; }}
            QMenu::separator {{ height: 1px; background-color: {INPUT_BORDER}; margin: 6px 12px; }}
        """

        self.tray_menu = QMenu()
        self.tray_menu.setStyleSheet(menu_style)

        self.meeting_action = QAction("Start Scribe", self.app)
        self.meeting_action.triggered.connect(self.start_meeting_mode)
        self.tray_menu.addAction(self.meeting_action)

        self.settings_action = QAction("Settings", self.app)
        self.settings_action.triggered.connect(self.settings_window.show)
        self.tray_menu.addAction(self.settings_action)

        self.tray_menu.addSeparator()

        self.exit_action = QAction("Exit", self.app)
        self.exit_action.triggered.connect(self.exit_app)
        self.tray_menu.addAction(self.exit_action)

        self.tray_icon.setContextMenu(self.tray_menu)
        self.tray_icon.show()

    def start_meeting_mode(self):
        """Launch Scribe as a separate process."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        QProcess.startDetached(sys.executable, ["-m", "src.meeting.app"], project_root)

    def cleanup(self):
        if self.key_listener:
            self.key_listener.stop()
        if self._instance_lock:
            release_single_instance_lock(self._instance_lock)

    def exit_app(self):
        self.cleanup()
        QApplication.quit()

    def restart_app(self):
        self.cleanup()
        QProcess.startDetached(sys.executable, sys.argv)
        QApplication.quit()

    def on_settings_closed(self):
        if not os.path.exists(os.path.join("src", "config.yaml")):
            QMessageBox.information(
                self.settings_window,
                "Using Default Values",
                "Settings closed without saving. Default values are being used."
            )
            self.initialize_components()

    def on_activation(self):
        if self.result_thread and self.result_thread.isRunning():
            if self.recording_start_time is not None:
                elapsed = time.time() - self.recording_start_time
                if elapsed < self.MIN_RECORDING_SECONDS:
                    ConfigManager.console_print(f'Ignoring stop - only {elapsed:.1f}s recorded (min: {self.MIN_RECORDING_SECONDS}s)')
                    return

            recording_mode = ConfigManager.get_config_value("recording_options", "recording_mode")
            if recording_mode == "continuous":
                self.continuous_stopped = True

            self.result_thread.stop_recording()
            return

        self.continuous_stopped = False
        self.start_result_thread()

    def on_deactivation(self):
        if ConfigManager.get_config_value("recording_options", "recording_mode") == "hold_to_record":
            if self.result_thread and self.result_thread.isRunning():
                if self.recording_start_time is not None:
                    elapsed = time.time() - self.recording_start_time
                    if elapsed < self.MIN_RECORDING_SECONDS:
                        ConfigManager.console_print(f'Hold too short ({elapsed:.1f}s), waiting for min duration')
                        return
                self.result_thread.stop_recording()

    def start_result_thread(self):
        with self._thread_lock:
            # Guard against rapid double-press
            if self.recording_start_time is not None:
                if time.time() - self.recording_start_time < 0.5:
                    ConfigManager.console_print('Thread starting, ignoring duplicate press')
                    return

            if self.result_thread and self.result_thread.isRunning():
                return

            if self.processing_result:
                ConfigManager.console_print('Still processing previous transcription...')
                return

            self.recording_start_time = time.time()
            self.result_thread = ResultThread()
            if not ConfigManager.get_config_value("misc", "hide_status_window"):
                self.result_thread.statusSignal.connect(self.status_window.updateStatus)
                self.result_thread.errorSignal.connect(self.status_window.showError)
            self.result_thread.errorSignal.connect(self.on_transcription_error)
            self.result_thread.resultSignal.connect(self.on_transcription_complete)
            self.result_thread.start()

    def stop_result_thread(self):
        if self.result_thread and self.result_thread.isRunning():
            self.recording_start_time = None
            self.processing_result = False
            self.result_thread.stop()

    def _copy_to_clipboard(self, text, retries=3):
        for _ in range(retries):
            try:
                pyperclip.copy(text)
                if pyperclip.paste() == text:
                    return True
            except Exception:
                pass
            time.sleep(0.1)
        return clipboard_copy_fallback(text)

    def on_transcription_error(self, error_msg):
        if hasattr(self, 'tray_icon') and self.tray_icon:
            self.tray_icon.showMessage(
                "Koe - Transcription Failed",
                error_msg,
                QSystemTrayIcon.Warning,
                5000
            )
        ConfigManager.console_print(f'Transcription error: {error_msg}')

    def on_transcription_complete(self, result):
        self.recording_start_time = None
        self.processing_result = True

        try:
            if result and result.strip():
                success = self._copy_to_clipboard(result)
                if success:
                    ConfigManager.console_print(f'Copied to clipboard: {result[:50]}...')
                else:
                    ConfigManager.console_print('WARNING: clipboard copy failed')

            if ConfigManager.get_config_value("misc", "noise_on_completion"):
                try:
                    play_sound_file(Path(__file__).parent.parent / "assets" / "beep.wav")
                except Exception as e:
                    ConfigManager.console_print(f'Beep failed: {e}')

            if not ConfigManager.get_config_value("misc", "hide_status_window"):
                if self.status_window.isVisible():
                    self.status_window.updateStatus('complete')

            if (ConfigManager.get_config_value("recording_options", "recording_mode") == "continuous"
                    and not self.continuous_stopped):
                self.start_result_thread()
            else:
                self.key_listener.start()

        except Exception as e:
            _debug(f"on_transcription_complete EXCEPTION: {e}")
            import traceback
            _debug(f"Traceback: {traceback.format_exc()}")
            traceback.print_exc()
        finally:
            self.processing_result = False

    def run(self):
        sys.exit(self.app.exec_())


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    qapp = QApplication(sys.argv)
    set_app_user_model_id()

    icon_path = str(Path(__file__).parent.parent / "assets" / "koe-icon.ico")
    qapp.setWindowIcon(QIcon(icon_path))
    qapp.setQuitOnLastWindowClosed(False)

    init_window = InitializationWindow()
    init_window.show()
    qapp.processEvents()

    app = KoeApp(qapp=qapp, init_window=init_window)
    app.run()
