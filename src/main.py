"""
Koe — hotkey transcription app.

Entry point. Boots a QApplication, registers the global hotkey, and runs
in the system tray. Audio capture happens in ResultThread; transcription
 is delegated to transcription.transcribe(), which calls ElevenLabs Scribe v2.
"""

import os
import sys
import time
import threading
from pathlib import Path
from datetime import datetime

import pyperclip
from PyQt5.QtCore import QObject, QPoint, QProcess, QRect, QSize, Qt
from PyQt5.QtGui import QCursor, QIcon
from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QAction, QMessageBox

from compat import (
    acquire_single_instance_lock,
    release_single_instance_lock,
    ensure_windows_shortcut,
    set_app_user_model_id,
    clipboard_copy_fallback,
    play_sound_file,
)
from key_listener import KeyListener
from result_thread import ResultThread
from ui.settings_window import SettingsWindow
from ui.status_window import StatusWindow
from ui.initialization_window import InitializationWindow
from ui import theme
from utils import ConfigManager

_DEBUG_LOG = Path(__file__).parent.parent / "logs" / "debug.log"


def _debug(msg: str):
    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass


def tray_menu_position(cursor: QPoint, menu_size: QSize, available: QRect) -> QPoint:
    """Place a tray menu above the taskbar and keep it on the active screen."""
    x = cursor.x() - menu_size.width() + 18
    x = max(available.left(), min(x, available.right() - menu_size.width() + 1))
    y = min(
        cursor.y() - menu_size.height() - 8,
        available.bottom() - menu_size.height() + 1,
    )
    y = max(available.top(), y)
    return QPoint(x, y)


class KoeApp(QObject):
    # Minimum recording time before hotkey can stop. Prevents accidental double-press from discarding recordings.
    MIN_RECORDING_SECONDS = 1.0

    def __init__(self, qapp=None, init_window=None):
        super().__init__()
        self._instance_lock = acquire_single_instance_lock()
        self.recording_start_time = None
        self.processing_result = False
        self.suppress_current_result = False
        self._thread_lock = threading.Lock()
        self._components_initialized = False
        self.result_thread = None
        self.status_window = None
        self._status_thread = None
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
        self.settings_window.settings_saved.connect(self.apply_settings)

        if ConfigManager.config_file_exists():
            self.initialize_components()
        else:
            print("No valid configuration file found. Opening settings window...")
            if self.init_window:
                self.init_window.close()
            self.settings_window.show()

    def initialize_components(self):
        if self._components_initialized:
            return
        self.key_listener = KeyListener()
        self.key_listener.add_callback("on_activate", self.on_activation)
        self.key_listener.add_callback("on_deactivate", self.on_deactivation)

        self._sync_status_window()

        self.create_tray_icon()
        self.key_listener.start()
        self._components_initialized = True

        if self.init_window:
            self.init_window.close()

    def create_tray_icon(self):
        icon_path = str(Path(__file__).parent.parent / "assets" / "koe-icon.ico")
        self.tray_icon = QSystemTrayIcon(QIcon(icon_path), self.app)

        self.tray_menu = QMenu()
        self.tray_menu.setStyleSheet(theme.tray_menu_stylesheet())

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

        self.tray_icon.activated.connect(self._on_tray_activated)
        self.tray_icon.show()

    def _on_tray_activated(self, reason) -> None:
        if reason != QSystemTrayIcon.Context:
            return
        self.tray_menu.ensurePolished()
        cursor = QCursor.pos()
        screen = QApplication.screenAt(cursor) or QApplication.primaryScreen()
        available = screen.availableGeometry()
        position = tray_menu_position(cursor, self.tray_menu.sizeHint(), available)
        self.tray_menu.popup(position)

    def start_meeting_mode(self):
        """Launch Scribe as a separate process."""
        project_root = Path(__file__).resolve().parent.parent
        python_path = Path(sys.executable).resolve()
        pythonw_path = python_path.with_name("pythonw.exe")
        if not pythonw_path.exists():
            pythonw_path = python_path

        if sys.platform == "win32":
            try:
                start_menu = (
                    Path(os.environ["APPDATA"])
                    / "Microsoft"
                    / "Windows"
                    / "Start Menu"
                    / "Programs"
                )
                shortcut = ensure_windows_shortcut(
                    start_menu / "Koe Scribe.lnk",
                    app_id="Koe.Scribe.App",
                    target_path=pythonw_path,
                    arguments="-m src.meeting.app",
                    working_directory=project_root,
                    icon_path=project_root / "assets" / "koe-icon.ico",
                )
                if shortcut:
                    os.startfile(str(shortcut))
                    return
            except Exception as exc:
                _debug(f"Scribe shortcut launch failed; using direct fallback: {exc}")

        QProcess.startDetached(
            str(pythonw_path),
            ["-m", "src.meeting.app"],
            str(project_root),
        )

    def cleanup(self):
        key_listener = getattr(self, "key_listener", None)
        if key_listener:
            key_listener.stop()
        if self._instance_lock:
            release_single_instance_lock(self._instance_lock)

    def exit_app(self):
        self.cleanup()
        QApplication.quit()

    def _ensure_status_window(self) -> StatusWindow:
        if self.status_window is None:
            self.status_window = StatusWindow()
            self.status_window.cancelSignal.connect(self.cancel_active_snippet)
            self.status_window.dismissSignal.connect(self.dismiss_active_transcription)
        return self.status_window

    def _disconnect_status_thread(self) -> None:
        thread = self._status_thread
        status_window = self.status_window
        if thread is not None and status_window is not None:
            for signal, slot in (
                (thread.statusSignal, status_window.updateStatus),
                (thread.errorSignal, status_window.showError),
            ):
                try:
                    signal.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
        self._status_thread = None

    def _connect_status_thread(self, thread: ResultThread) -> None:
        status_window = self._ensure_status_window()
        if self._status_thread is thread:
            return
        self._disconnect_status_thread()
        thread.statusSignal.connect(status_window.updateStatus)
        thread.errorSignal.connect(status_window.showError)
        self._status_thread = thread

    def _sync_status_window(self) -> None:
        enabled = not bool(ConfigManager.get_config_value("misc", "hide_status_window"))
        thread = getattr(self, "result_thread", None)
        active = thread is not None and thread.isRunning()
        if enabled:
            status_window = self._ensure_status_window()
            if active:
                self._connect_status_thread(thread)
                if thread.is_recording and not status_window.isVisible():
                    status_window.updateStatus("recording")
            return

        if active:
            # A visibility preference applies to the next snippet. Never make
            # the current capture disappear or detach its completion state.
            self._connect_status_thread(thread)
            _debug("Status-card visibility change deferred until active snippet finishes")
            return

        self._disconnect_status_thread()
        if self.status_window is not None:
            self.status_window.close()

    def _on_result_thread_finished(self) -> None:
        if bool(ConfigManager.get_config_value("misc", "hide_status_window")):
            self._disconnect_status_thread()
            if self.status_window is not None:
                self.status_window.close()

    def apply_settings(self) -> None:
        """Apply autosaved settings without restarting or touching active audio."""
        if not self._components_initialized:
            return
        key_listener = getattr(self, "key_listener", None)
        if key_listener is not None:
            key_listener.load_activation_keys()
        self._sync_status_window()
        _debug("Settings applied live; process and active recording preserved")

    def on_settings_closed(self):
        if self._components_initialized:
            return
        if not os.path.exists(os.path.join("src", "config.yaml")):
            QMessageBox.information(
                self.settings_window,
                "Using Default Values",
                "Settings closed without saving. Default values are being used."
            )
        self.initialize_components()

    def on_activation(self):
        _debug(
            "Hotkey activation received "
            f"(thread_running={bool(self.result_thread and self.result_thread.isRunning())}, "
            f"recording={bool(self.result_thread and self.result_thread.is_recording)})"
        )
        if self.result_thread and self.result_thread.isRunning():
            if self.recording_start_time is not None:
                elapsed = time.time() - self.recording_start_time
                if elapsed < self.MIN_RECORDING_SECONDS:
                    ConfigManager.console_print(f'Ignoring stop - only {elapsed:.1f}s recorded (min: {self.MIN_RECORDING_SECONDS}s)')
                    return

            self.result_thread.stop_recording(reason="hotkey toggle")
            return

        self.start_result_thread()

    def on_deactivation(self):
        """Hotkey release is intentionally inert in press-to-toggle mode."""

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
            self.suppress_current_result = False
            self.result_thread = ResultThread()
            if not ConfigManager.get_config_value("misc", "hide_status_window"):
                self._connect_status_thread(self.result_thread)
            self.result_thread.errorSignal.connect(self.on_transcription_error)
            self.result_thread.resultSignal.connect(self.on_transcription_complete)
            self.result_thread.cancelledSignal.connect(self.on_snippet_cancelled)
            self.result_thread.finished.connect(self._on_result_thread_finished)
            self.result_thread.start()

    def cancel_active_snippet(self):
        """Cancel only an active capture; never interrupt transcription."""
        thread = self.result_thread
        if not thread or not thread.isRunning() or not thread.is_recording:
            return
        _debug("Listening card close requested; cancelling active capture")
        thread.cancel_recording()

    def dismiss_active_transcription(self):
        """Hide an in-flight transcription and suppress clipboard/completion feedback."""
        thread = self.result_thread
        if not thread or not thread.isRunning() or thread.is_recording:
            return
        self.suppress_current_result = True
        _debug("Transcribing card close requested; clipboard and beep suppressed")

    def on_snippet_cancelled(self):
        self.recording_start_time = None
        self.processing_result = False
        if hasattr(self, "status_window"):
            self.status_window.updateStatus("cancelled")
        ConfigManager.console_print("Snippet cancelled.")

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
        suppressed = self.suppress_current_result

        try:
            if suppressed:
                _debug("Transcription completed after dismissal; result kept out of clipboard")
            elif result and result.strip():
                success = self._copy_to_clipboard(result)
                if success:
                    ConfigManager.console_print(f'Copied to clipboard: {result[:50]}...')
                else:
                    ConfigManager.console_print('WARNING: clipboard copy failed')

            if not suppressed and ConfigManager.get_config_value("misc", "noise_on_completion"):
                try:
                    play_sound_file(Path(__file__).parent.parent / "assets" / "beep.wav")
                except Exception as e:
                    ConfigManager.console_print(f'Beep failed: {e}')

            if not ConfigManager.get_config_value("misc", "hide_status_window"):
                if self.status_window.isVisible():
                    self.status_window.updateStatus('complete')

            self.key_listener.start()

        except Exception as e:
            _debug(f"on_transcription_complete EXCEPTION: {e}")
            import traceback
            _debug(f"Traceback: {traceback.format_exc()}")
            traceback.print_exc()
        finally:
            self.processing_result = False
            self.suppress_current_result = False

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
