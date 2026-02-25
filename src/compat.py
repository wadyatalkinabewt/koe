"""
Platform-specific utilities for cross-platform compatibility.

Provides abstractions for Windows/macOS differences:
- Single-instance locking (Windows mutex vs file lock)
- Sound playback (winsound vs afplay)
- Clipboard fallback (clip.exe vs pbcopy)
- CUDA DLL setup (Windows-only)
- ffmpeg discovery
- Default compute device selection
"""

import sys
import os
from pathlib import Path

IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform == 'darwin'
IS_LINUX = sys.platform == 'linux'


def setup_cuda_dlls():
    """Add cuDNN and cuBLAS DLL directories to PATH. Windows-only, no-op elsewhere."""
    if not IS_WINDOWS:
        return
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
        pass


def acquire_single_instance_lock(lock_name="KoeTranscriptionApp"):
    """
    Acquire a single-instance lock. Exits if another instance is running.

    Windows: Named Mutex via kernel32.
    macOS/Linux: File-based lock via fcntl.

    Returns a lock handle that must be kept alive (prevents GC release).
    """
    if IS_WINDOWS:
        import ctypes
        ERROR_ALREADY_EXISTS = 183
        mutex = ctypes.windll.kernel32.CreateMutexW(None, False, f"{lock_name}Mutex_v1")  # type: ignore
        last_error = ctypes.windll.kernel32.GetLastError()  # type: ignore
        if last_error == ERROR_ALREADY_EXISTS:
            print("[Koe] Another instance is already running. Exiting.")
            sys.exit(0)
        return mutex
    else:
        import fcntl
        lock_path = Path.home() / f".{lock_name}.lock"
        lock_file = open(lock_path, 'w')
        try:
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
    """Set Windows AppUserModelID for proper taskbar grouping. No-op on other platforms."""
    if not IS_WINDOWS:
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)  # type: ignore
    except Exception:
        pass


def play_sound_file(file_path):
    """Play a sound file using the platform's native method."""
    file_path = str(file_path)
    if IS_WINDOWS:
        try:
            import winsound
            winsound.PlaySound(file_path, winsound.SND_FILENAME)
            return
        except Exception:
            pass
    if IS_MACOS:
        try:
            import subprocess
            subprocess.run(['afplay', file_path], check=True, capture_output=True)
            return
        except Exception:
            pass
    # Fallback: audioplayer library
    try:
        from audioplayer import AudioPlayer
        AudioPlayer(file_path).play(block=True)
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
    elif IS_MACOS:
        try:
            process = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
            process.communicate(text.encode('utf-8'))
            return True
        except Exception:
            return False
    return False


def get_default_device():
    """Get the default compute device for this platform.

    macOS: Always 'cpu' (MPS has bugs with pyannote timestamps).
    Windows/Linux: 'cuda' if available, else 'cpu'.
    """
    if IS_MACOS:
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def get_default_engine():
    """Get the default transcription engine for this platform.

    macOS: 'mlx' if mlx-whisper is available, else 'whisper' (CPU fallback).
    Windows/Linux: 'whisper' (faster-whisper with CUDA).
    """
    if IS_MACOS:
        try:
            import mlx_whisper
            return "mlx"
        except ImportError:
            pass
    return "whisper"


def find_ffmpeg():
    """Find ffmpeg executable on the system."""
    import subprocess
    # Check PATH first
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return "ffmpeg"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    # Windows: check winget install location
    if IS_WINDOWS:
        local_app_data = os.environ.get('LOCALAPPDATA', '')
        if local_app_data:
            winget_glob = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
            if winget_glob.exists():
                for ffmpeg_bin in winget_glob.rglob("ffmpeg.exe"):
                    return str(ffmpeg_bin)
        # Hardcoded fallback
        winget_path = r"C:\Users\Galbraith\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
        if os.path.exists(winget_path):
            return winget_path
    # macOS: check Homebrew
    if IS_MACOS:
        for brew_path in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
            if os.path.exists(brew_path):
                return brew_path
    raise RuntimeError(
        "ffmpeg not found. Install with: "
        + ("brew install ffmpeg" if IS_MACOS else "winget install ffmpeg")
    )


def maybe_clear_gpu_cache(audio_duration: float, cumulative_seconds: float, threshold: float = 180.0):
    """Clear GPU cache if needed. Only does anything on CUDA systems.

    Returns updated cumulative_seconds.
    """
    if IS_MACOS:
        return cumulative_seconds  # No CUDA cache to clear

    cumulative_seconds += audio_duration
    force_clear = audio_duration > 40.0

    if force_clear or cumulative_seconds >= threshold:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                reason = f"long chunk ({audio_duration:.0f}s)" if force_clear else f"{cumulative_seconds:.0f}s cumulative"
                print(f"[Server] Cleared CUDA cache: {reason}")
                return 0.0
        except Exception as e:
            print(f"[Server] Warning: failed to clear CUDA cache: {e}")

    return cumulative_seconds
