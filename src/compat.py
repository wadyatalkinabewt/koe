"""
Platform helpers (Windows-focused after Mac/Linux/server stack rip).

Kept as a thin abstraction so calling code stays clean — single-instance
locks, sound playback, clipboard fallback, AppUserModelID for taskbar
grouping. Mac/Linux fallbacks are best-effort and untested post-rip.
"""

import sys
import os
from pathlib import Path

IS_WINDOWS = sys.platform == 'win32'


def acquire_single_instance_lock(lock_name="KoeTranscriptionApp"):
    """Acquire a single-instance lock. Exits if another instance is running.

    Windows: named mutex via kernel32.
    Other: file-based fcntl lock (best-effort).
    """
    if IS_WINDOWS:
        import ctypes
        ERROR_ALREADY_EXISTS = 183
        mutex = ctypes.windll.kernel32.CreateMutexW(None, False, f"{lock_name}Mutex_v1")  # type: ignore
        if ctypes.windll.kernel32.GetLastError() == ERROR_ALREADY_EXISTS:  # type: ignore
            print("[Koe] Another instance is already running. Exiting.")
            sys.exit(0)
        return mutex
    try:
        import fcntl
        lock_path = Path.home() / f".{lock_name}.lock"
        lock_file = open(lock_path, 'w')
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_file.write(str(os.getpid()))
        lock_file.flush()
        return lock_file
    except (IOError, OSError):
        print("[Koe] Another instance is already running. Exiting.")
        sys.exit(0)


def release_single_instance_lock(lock_handle):
    """Release the single-instance lock."""
    if lock_handle is None:
        return
    if IS_WINDOWS:
        try:
            import ctypes
            ctypes.windll.kernel32.ReleaseMutex(lock_handle)  # type: ignore
            ctypes.windll.kernel32.CloseHandle(lock_handle)  # type: ignore
        except Exception:
            pass
    else:
        try:
            import fcntl
            fcntl.flock(lock_handle, fcntl.LOCK_UN)
            lock_handle.close()
        except Exception:
            pass


def set_app_user_model_id(app_id="Koe.Transcription.App"):
    """Set Windows AppUserModelID for proper taskbar grouping. No-op elsewhere."""
    if not IS_WINDOWS:
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)  # type: ignore
    except Exception:
        pass


def play_sound_file(file_path):
    """Play a sound file. Windows uses winsound; other platforms best-effort."""
    file_path = str(file_path)
    if IS_WINDOWS:
        try:
            import winsound
            winsound.PlaySound(file_path, winsound.SND_FILENAME)
        except Exception:
            pass
        return
    # Generic POSIX fallback (best-effort, untested post-rip)
    try:
        import subprocess
        subprocess.run(['afplay' if sys.platform == 'darwin' else 'aplay', file_path],
                       check=True, capture_output=True)
    except Exception:
        pass


def clipboard_copy_fallback(text):
    """Platform-specific clipboard fallback when pyperclip fails."""
    import subprocess
    if IS_WINDOWS:
        try:
            process = subprocess.Popen(['clip'], stdin=subprocess.PIPE)
            process.communicate(text.encode('utf-16le'))
            return True
        except Exception:
            return False
    return False
