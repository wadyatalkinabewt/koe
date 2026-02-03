import os
import sys

# Add NVIDIA cuDNN/cuBLAS DLL paths for GPU acceleration (must be before any CUDA imports)
def _setup_cuda_dlls():
    """Add cuDNN and cuBLAS DLL directories to PATH (more reliable than os.add_dll_directory)."""
    try:
        import site
        paths_to_add = []
        for sp in [site.getusersitepackages()] + site.getsitepackages():
            cudnn_bin = os.path.join(sp, "nvidia", "cudnn", "bin")
            cublas_bin = os.path.join(sp, "nvidia", "cublas", "bin")
            if os.path.exists(cudnn_bin):
                paths_to_add.append(cudnn_bin)
            if os.path.exists(cublas_bin):
                paths_to_add.append(cublas_bin)
        if paths_to_add:
            os.environ['PATH'] = os.pathsep.join(paths_to_add) + os.pathsep + os.environ.get('PATH', '')
    except Exception:
        pass  # Silently fail if not on Windows or DLLs not installed

_setup_cuda_dlls()

def preload_model():
    """Load the Whisper model.

    If the transcription server is running, we skip loading the local model
    since we'll use the shared model from the server.
    """
    from utils import ConfigManager
    from transcription import create_local_model, check_server_available

    ConfigManager.initialize()
    if ConfigManager.config_file_exists():
        # Check if server is running - if so, use that instead of loading local
        if check_server_available():
            print("[Koe] Using shared transcription server")
            return None

        # No server, load local model
        print("[Koe] Loading local model...")
        return create_local_model()
    return None


# Now safe to import PyQt5
import time
import socket
import pyperclip
from audioplayer import AudioPlayer
from PyQt5.QtCore import QObject, QProcess
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QAction, QMessageBox

from key_listener import KeyListener
from result_thread import ResultThread
from ui.main_window import MainWindow
from ui.settings_window import SettingsWindow
from ui.status_window import StatusWindow
from ui.initialization_window import InitializationWindow
from transcription import create_local_model
from utils import ConfigManager
from server_launcher import is_server_running, start_server_background


class KoeApp(QObject):
    # Minimum recording time (seconds) before hotkey can stop recording
    # This prevents accidental double-press from discarding recordings
    MIN_RECORDING_SECONDS = 1.0

    def __init__(self, qapp=None, init_window=None, preloaded_model=None):
        # Flag to track if user explicitly stopped continuous mode (prevents auto-restart)
        self.continuous_stopped = False
        super().__init__()

        # Single-instance check: bind to a specific port as a lock
        self._lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self._lock_socket.bind(('127.0.0.1', 19877))
        except OSError:
            # Another instance is already running
            print("[Koe] Another instance is already running. Exiting.")
            sys.exit(0)

        self.preloaded_model = preloaded_model
        self.recording_start_time = None  # Track when recording started
        self.processing_result = False  # Prevent new recordings during result processing
        self.app = qapp if qapp else QApplication(sys.argv)
        self.init_window = init_window

        # If no QApplication was passed, set it up now
        if not qapp:
            # Set Windows AppUserModelID for proper taskbar icon
            try:
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('Koe.Transcription.App')
            except:
                pass

            # Set app icon with absolute path
            from pathlib import Path
            icon_path = str(Path(__file__).parent.parent / "assets" / "koe-icon.ico")
            self.app.setWindowIcon(QIcon(icon_path))
            self.app.setQuitOnLastWindowClosed(False)  # Keep app alive in tray

            # Show initialization window if not already shown
            if not init_window:
                self.init_window = InitializationWindow()
                self.init_window.show()
                self.app.processEvents()  # Force UI update

        if ConfigManager._instance is None:
            ConfigManager.initialize()

        self.settings_window = SettingsWindow()
        self.settings_window.settings_closed.connect(self.on_settings_closed)
        self.settings_window.settings_saved.connect(self.restart_app)

        if ConfigManager.config_file_exists():
            self.initialize_components()
        else:
            print("No valid configuration file found. Opening settings window...")
            # Close initialization window if opening settings
            if hasattr(self, 'init_window') and self.init_window:
                self.init_window.close()
            self.settings_window.show()

    def initialize_components(self):
        from transcription import check_server_available

        self.key_listener = KeyListener()
        self.key_listener.add_callback("on_activate", self.on_activation)
        self.key_listener.add_callback("on_deactivate", self.on_deactivation)

        if self.preloaded_model is not None:
            self.local_model = self.preloaded_model
        elif check_server_available():
            # Using server, no local model needed
            self.local_model = None
        else:
            self.local_model = create_local_model()

        self.result_thread = None

        self.main_window = MainWindow()
        self.main_window.openSettings.connect(self.settings_window.show)
        self.main_window.startListening.connect(self.key_listener.start)
        self.main_window.closeApp.connect(self.exit_app)

        if not ConfigManager.get_config_value("misc", "hide_status_window"):
            self.status_window = StatusWindow()
            # Set cancel callback to stop the thread
            self.status_window.set_cancel_callback(self.on_cancel_recording)
            # Note: We intentionally do NOT connect closeSignal to stop_result_thread
            # Closing the status window should not cancel ongoing transcription

        self.create_tray_icon()
        # Auto-start listening (skip main window, go straight to tray)
        self.key_listener.start()

        # Close initialization window now that we're ready
        if hasattr(self, 'init_window') and self.init_window:
            self.init_window.close()

    def create_tray_icon(self):
        from pathlib import Path
        from ui.theme import (BG_COLOR, TEXT_COLOR, SECONDARY_TEXT,
                              BUTTON_HOVER_BG, INPUT_BORDER, BORDER_COLOR)

        icon_path = str(Path(__file__).parent.parent / "assets" / "koe-icon.ico")
        self.tray_icon = QSystemTrayIcon(QIcon(icon_path), self.app)

        # Dark terminal-themed menu stylesheet
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
            QMenu::item:selected {{
                background-color: {BUTTON_HOVER_BG};
                color: {TEXT_COLOR};
            }}
            QMenu::item:disabled {{
                color: {SECONDARY_TEXT};
            }}
            QMenu::separator {{
                height: 1px;
                background-color: {INPUT_BORDER};
                margin: 6px 12px;
            }}
            QMenu::indicator {{
                width: 16px;
                height: 16px;
                margin-left: 6px;
            }}
            QMenu::right-arrow {{
                width: 12px;
                height: 12px;
                margin-right: 8px;
            }}
        """

        # Store menu and actions as instance variables to prevent garbage collection
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
        # Launch Scribe from the project root directory
        # Scribe will handle server availability itself
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        QProcess.startDetached(sys.executable, ["-m", "src.meeting.app"], project_root)

    def cleanup(self):
        if self.key_listener:
            self.key_listener.stop()

        # Release the single-instance lock
        if hasattr(self, '_lock_socket'):
            try:
                self._lock_socket.close()
            except:
                pass

    def exit_app(self):
        self.cleanup()
        # Stop the server when Koe exits
        from server_launcher import stop_server
        stop_server()
        QApplication.quit()

    def restart_app(self):
        self.cleanup()
        QApplication.quit()
        QProcess.startDetached(sys.executable, sys.argv)

    def on_settings_closed(self):
        if not os.path.exists(os.path.join("src", "config.yaml")):
            QMessageBox.information(
                self.settings_window,
                "Using Default Values",
                "Settings closed without saving. Default values are being used."
            )
            self.initialize_components()

    def on_cancel_recording(self):
        """Handle cancel action from the status window."""
        from pathlib import Path
        from datetime import datetime
        debug_log = Path(__file__).parent.parent / "logs" / "debug.log"
        def _debug(msg):
            try:
                with open(debug_log, "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
            except:
                pass

        _debug("on_cancel_recording() called")
        # User clicked [ESC] or pressed Escape - stop the recording thread
        _debug("  Calling stop_result_thread()")
        self.stop_result_thread()
        _debug("on_cancel_recording() done")

    def on_activation(self):
        if self.result_thread and self.result_thread.isRunning():
            # Protection against accidental double-press:
            # Only allow stopping if we've been recording for MIN_RECORDING_SECONDS
            if self.recording_start_time is not None:
                elapsed = time.time() - self.recording_start_time
                if elapsed < self.MIN_RECORDING_SECONDS:
                    ConfigManager.console_print(f'Ignoring stop - only {elapsed:.1f}s recorded (min: {self.MIN_RECORDING_SECONDS}s)')
                    return

            recording_mode = ConfigManager.get_config_value("recording_options", "recording_mode")
            if recording_mode == "continuous":
                # Mark that user explicitly stopped - prevents auto-restart after transcription
                self.continuous_stopped = True

            # Stop recording and trigger transcription (for all modes)
            self.result_thread.stop_recording()
            return

        # Starting a new recording
        self.continuous_stopped = False  # Reset flag when starting fresh
        self.start_result_thread()

    def on_deactivation(self):
        if ConfigManager.get_config_value("recording_options", "recording_mode") == "hold_to_record":
            if self.result_thread and self.result_thread.isRunning():
                # For hold_to_record, also check minimum recording time
                if self.recording_start_time is not None:
                    elapsed = time.time() - self.recording_start_time
                    if elapsed < self.MIN_RECORDING_SECONDS:
                        ConfigManager.console_print(f'Hold too short ({elapsed:.1f}s), waiting for min duration')
                        # Don't stop yet - let the user keep holding or it will auto-stop
                        return
                self.result_thread.stop_recording()

    def start_result_thread(self):
        # Guard against rapid double-press: if we JUST set recording_start_time, don't create new thread
        # (isRunning() may not be True yet if thread is still spinning up)
        if self.recording_start_time is not None:
            elapsed = time.time() - self.recording_start_time
            if elapsed < 0.5:  # Thread was just started, don't create another
                ConfigManager.console_print(f'Thread starting, ignoring duplicate press')
                return

        if self.result_thread and self.result_thread.isRunning():
            return

        # Don't start a new recording if still processing previous result
        if self.processing_result:
            ConfigManager.console_print('Still processing previous transcription...')
            return

        self.recording_start_time = time.time()  # Track when recording started (SET BEFORE creating thread)
        self.result_thread = ResultThread(self.local_model)
        if not ConfigManager.get_config_value("misc", "hide_status_window"):
            self.result_thread.statusSignal.connect(self.status_window.updateStatus)
            self.result_thread.errorSignal.connect(self.status_window.showError)
        self.result_thread.errorSignal.connect(self.on_transcription_error)
        self.result_thread.resultSignal.connect(self.on_transcription_complete)
        self.result_thread.start()

    def stop_result_thread(self):
        _debug("stop_result_thread() called")
        if self.result_thread and self.result_thread.isRunning():
            _debug("  Thread is running, calling stop()")
            self.recording_start_time = None  # Reset since we're cancelling
            self.processing_result = False  # Clear flag since we're cancelling
            self.result_thread.stop()
        else:
            _debug("  Thread not running, nothing to stop")

    def _copy_to_clipboard(self, text, retries=3):
        """Copy text to clipboard with retries and fallback."""
        import subprocess

        for attempt in range(retries):
            try:
                pyperclip.copy(text)
                if pyperclip.paste() == text:
                    return True
            except Exception:
                pass
            time.sleep(0.1)

        # Fallback: use Windows clip.exe directly
        try:
            process = subprocess.Popen(['clip'], stdin=subprocess.PIPE)
            process.communicate(text.encode('utf-16le'))
            return True
        except Exception:
            pass

        return False

    def on_transcription_error(self, error_msg):
        """Handle transcription errors with tray notification."""
        # Show tray notification so user knows even if they weren't looking
        if hasattr(self, 'tray_icon') and self.tray_icon:
            self.tray_icon.showMessage(
                "Koe - Transcription Failed",
                error_msg,
                QSystemTrayIcon.Warning,
                5000  # Show for 5 seconds
            )
        ConfigManager.console_print(f'Transcription error: {error_msg}')

    def on_transcription_complete(self, result):
        from pathlib import Path
        from datetime import datetime

        # Debug logging to file
        debug_log = Path(__file__).parent.parent / "logs" / "debug.log"
        def _debug(msg):
            try:
                with open(debug_log, "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
            except:
                pass

        _debug("on_transcription_complete() STARTED")
        self.recording_start_time = None  # Reset recording start time
        self.processing_result = True  # Block new recordings during processing

        try:
            _debug(f"  result length: {len(result)}")
            ConfigManager.console_print(f'=== on_transcription_complete called with result length: {len(result)} ===')

            # Copy to clipboard only (no typing into text fields)
            if result and result.strip():
                _debug("  Copying to clipboard...")
                ConfigManager.console_print('Copying to clipboard...')
                success = self._copy_to_clipboard(result)
                _debug(f"  Clipboard copy success: {success}")
                if success:
                    ConfigManager.console_print(f'Copied to clipboard: {result[:50]}...')
                else:
                    ConfigManager.console_print('WARNING: Failed to copy to clipboard')
            else:
                _debug("  Skipping clipboard (empty)")
                ConfigManager.console_print('Skipping clipboard (empty result)')

            # Play beep sound
            _debug("  Checking beep setting...")
            if ConfigManager.get_config_value("misc", "noise_on_completion"):
                try:
                    _debug("  Playing beep...")
                    ConfigManager.console_print('Playing beep...')
                    beep_path = Path(__file__).parent.parent / "assets" / "beep.wav"
                    # Use winsound on Windows for more reliable playback
                    import sys
                    if sys.platform == 'win32':
                        import winsound
                        winsound.PlaySound(str(beep_path), winsound.SND_FILENAME)
                    else:
                        AudioPlayer(str(beep_path)).play(block=True)
                    _debug("  Beep played")
                    ConfigManager.console_print('Beep played')
                except Exception as e:
                    _debug(f"  Beep failed: {e}")
                    ConfigManager.console_print(f'Failed to play beep: {e}')

            # Signal completion so status window shows "Complete!" for 2 seconds
            _debug("  Checking status window...")
            if not ConfigManager.get_config_value("misc", "hide_status_window"):
                # Only update if window is still visible (might have been closed by cancel)
                if self.status_window.isVisible():
                    _debug("  Updating status window to 'complete'...")
                    ConfigManager.console_print('Signaling completion to status window...')
                    # Call updateStatus directly - statusSignal is for window-to-app communication
                    self.status_window.updateStatus('complete')
                    _debug("  Status window updated")
                    ConfigManager.console_print('Status window will close after showing completion')
                else:
                    _debug("  Status window already closed, skipping update")
                    ConfigManager.console_print('Status window already closed')

            _debug("  Restarting key listener...")
            if ConfigManager.get_config_value("recording_options", "recording_mode") == "continuous" and not self.continuous_stopped:
                # Auto-restart recording in continuous mode (unless user explicitly stopped)
                self.start_result_thread()
            else:
                # Just restart the key listener for next activation
                self.key_listener.start()
            _debug("  Key listener restarted")

        except Exception as e:
            _debug(f"  EXCEPTION: {e}")
            ConfigManager.console_print(f'EXCEPTION in on_transcription_complete: {e}')
            import traceback
            _debug(f"  Traceback: {traceback.format_exc()}")
            traceback.print_exc()
        finally:
            # Always clear the flag, even if something fails
            self.processing_result = False
            _debug("on_transcription_complete() FINISHED")
            ConfigManager.console_print('=== on_transcription_complete finished ===\n')

    def run(self):
        sys.exit(self.app.exec_())


if __name__ == "__main__":
    # Enable high-DPI scaling BEFORE creating QApplication
    from PyQt5.QtCore import Qt
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # Create QApplication first so we can show initialization window
    qapp = QApplication(sys.argv)

    # Set Windows AppUserModelID for proper taskbar icon
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('Koe.Transcription.App')
    except:
        pass

    # Set app icon with absolute path
    from pathlib import Path
    icon_path = str(Path(__file__).parent.parent / "assets" / "koe-icon.ico")
    qapp.setWindowIcon(QIcon(icon_path))
    qapp.setQuitOnLastWindowClosed(False)  # Keep app alive in tray

    # Show initialization window immediately
    init_window = InitializationWindow()
    init_window.show()
    qapp.processEvents()  # Force UI update to show window

    # Now load the model (window stays visible during loading)
    preloaded_model = preload_model()

    # Create KoeApp with the existing QApplication and initialization window
    app = KoeApp(qapp=qapp, init_window=init_window, preloaded_model=preloaded_model)
    app.run()
