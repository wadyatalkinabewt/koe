# macOS Port - Complete

## Overview
macOS Apple Silicon (M-series) support added to Koe/Scribe while keeping all Windows functionality untouched. All changes use `sys.platform` branching so Windows code paths are unchanged.

## Project Location
`C:\dev\koe\` - Python app with PyQt5 UI for speech-to-text (hotkey snippets) and meeting transcription (Scribe) with speaker diarization.

## Completed Changes

### 1. `src/compat.py` (NEW)
Platform abstraction module with:
- `setup_cuda_dlls()` - Windows CUDA DLL setup, no-op on Mac
- `acquire_single_instance_lock()` / `release_single_instance_lock()` - Windows mutex vs macOS fcntl file lock
- `set_app_user_model_id()` - Windows taskbar ID, no-op on Mac
- `play_sound_file()` - winsound (Win) / afplay (Mac) / audioplayer fallback
- `clipboard_copy_fallback()` - clip.exe (Win) / pbcopy (Mac)
- `get_default_device()` - "cuda" on Windows, "cpu" on Mac
- `get_default_engine()` - "whisper" on Windows, "mlx" on Mac (if mlx-whisper installed)
- `find_ffmpeg()` - Checks PATH, then winget paths (Win) / Homebrew paths (Mac)
- `maybe_clear_gpu_cache()` - CUDA cache clearing, no-op on Mac
- `IS_WINDOWS`, `IS_MACOS`, `IS_LINUX` constants

### 2. `src/engines/mlx_engine.py` (NEW)
MLX Whisper engine for Apple Silicon:
- `ENGINE_ID = "mlx"`, registered via `@register_engine`
- `is_available()` checks `sys.platform == 'darwin'` AND `import mlx_whisper`
- `load()` resolves short names (e.g. "large-v3" -> "mlx-community/whisper-large-v3-mlx")
- `transcribe()` calls `mlx_whisper.transcribe()` and wraps result in `TranscriptionResult`
- Model aliases: large-v3-turbo, large-v3, medium, small, base
- Recommended model: `mlx-community/whisper-large-v3-turbo` (~15-20x realtime on M2)

### 3. `src/engines/factory.py` (UPDATED)
Added `from . import mlx_engine` in `_register_engines()` (with try/except ImportError)

### 4. `src/main.py` (UPDATED)
All Win32 APIs replaced with `compat` imports:
- `_setup_cuda_dlls()` → `from compat import setup_cuda_dlls`
- `ctypes.windll.kernel32.CreateMutexW` → `from compat import acquire_single_instance_lock`
- `ctypes.windll.kernel32.ReleaseMutex` → `from compat import release_single_instance_lock`
- `ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID` → `from compat import set_app_user_model_id`
- `winsound.PlaySound` / `AudioPlayer` → `from compat import play_sound_file`
- `subprocess.Popen(['clip'])` → `from compat import clipboard_copy_fallback`

### 5. `src/server.py` (UPDATED)
- Inline `_setup_cuda_dlls()` → `from compat import setup_cuda_dlls`
- `_find_ffmpeg()` → delegates to `compat.find_ffmpeg()`
- `_maybe_clear_cuda_cache()` → uses `compat.maybe_clear_gpu_cache()`
- Lifespan defaults: `get_default_engine()` and `get_default_device()` instead of hardcoded "whisper"/"cuda"
- Argparse defaults: same treatment

### 6. `src/meeting/capture.py` (REWRITTEN)
- Conditional imports: `pyaudiowpatch` on Windows, `sounddevice` on Mac
- Separate callbacks per platform (sounddevice receives numpy arrays, PyAudio receives bytes)
- BlackHole device detection via `sd.query_devices()` with manual index tracking
- Platform-specific stop methods
- Public API unchanged

### 7. `src/meeting/diarization.py` (UPDATED)
- Added `import sys`
- Device forced to CPU on macOS in `load()` method
- Both diarization pipeline AND embedding model get CPU on Mac

### 8. `src/config_schema.yaml` (UPDATED)
- `mlx` added to engine options list
- MLX config section with 5 model options (turbo, large-v3, medium, small, base)

### 9. `requirements-mac.txt` (NEW)
- macOS-specific dependencies, mlx-whisper instead of faster-whisper
- No PyAudioWPatch, BlackHole install instructions in comments

### 10. `src/setup_wizard.py` (UPDATED)
- SystemCheckThread: Apple Silicon detection, MLX check, mac-specific package list
- ModelDownloadThread: downloads MLX turbo model on Mac
- _save_configuration: sets `engine: 'mlx'` on macOS

### 11. `src/server_launcher.py` (UPDATED)
- Added MLX case to `get_engine_config()` (reads MLX model, sets device="cpu")
- Sets `os.environ["WHISPER_ENGINE"] = engine` before launching for whisper/mlx

## Architecture Notes

- **Engine system:** `src/engines/` uses a factory pattern with `@register_engine` decorator. Each engine implements `TranscriptionEngine` base class with `load()`, `transcribe()`, `is_available()`. The factory auto-discovers engines via imports in `_register_engines()`.
- **Config:** `src/config.yaml` is the user's config, `src/config_schema.yaml` defines defaults/schema. `ConfigManager` reads both.
- **Audio capture for Scribe:** `src/meeting/capture.py` runs two streams (mic + loopback). On Mac, mic uses `sounddevice`, loopback uses BlackHole via `sounddevice`.
- **Diarization:** pyannote 3.1 works on Mac CPU. MPS (Metal) produces wrong timestamps - this is a known upstream bug. Forced to CPU.
- **No changes needed to:** `src/transcription.py` (already uses engine factory), `src/result_thread.py`, `src/key_listener.py` (pynput is cross-platform), `src/ui/` files (PyQt5 is cross-platform).

## Design Principles

- Every change guarded with `sys.platform` or `compat.IS_WINDOWS`/`IS_MACOS`
- `config.yaml` (user's personal config) never modified, only `config_schema.yaml`
- `requirements.txt` unchanged (Windows deps). `requirements-mac.txt` is separate.
- Zero Windows behavior changes - all macOS code in platform branches
