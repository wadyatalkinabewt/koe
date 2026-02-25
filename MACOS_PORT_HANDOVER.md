# macOS Port - Handover Document

## What We're Doing
Adding macOS (Apple Silicon M2 Pro 64GB) support to Koe/Scribe while keeping all Windows functionality untouched. All changes use `sys.platform` branching so Windows code paths are unchanged.

## Project Location
`C:\dev\koe\` - Python app with PyQt5 UI for speech-to-text (hotkey snippets) and meeting transcription (Scribe) with speaker diarization.

## What's DONE

### 1. `src/compat.py` (NEW - COMPLETE)
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

### 2. `src/engines/mlx_engine.py` (NEW - COMPLETE)
MLX Whisper engine for Apple Silicon:
- `ENGINE_ID = "mlx"`, registered via `@register_engine`
- `is_available()` checks `sys.platform == 'darwin'` AND `import mlx_whisper`
- `load()` resolves short names (e.g. "large-v3" -> "mlx-community/whisper-large-v3-mlx")
- `transcribe()` calls `mlx_whisper.transcribe()` and wraps result in `TranscriptionResult`
- Model aliases: large-v3-turbo, large-v3, medium, small, base
- Recommended model: `mlx-community/whisper-large-v3-turbo` (~15-20x realtime on M2)

### 3. `src/engines/factory.py` (UPDATED - COMPLETE)
Added `from . import mlx_engine` in `_register_engines()` (with try/except ImportError)

### 4. `src/main.py` (UPDATED - COMPLETE)
All Win32 APIs replaced with `compat` imports:
- `_setup_cuda_dlls()` inline function → `from compat import setup_cuda_dlls`
- `ctypes.windll.kernel32.CreateMutexW` → `from compat import acquire_single_instance_lock`
- `ctypes.windll.kernel32.ReleaseMutex` → `from compat import release_single_instance_lock`
- `ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID` → `from compat import set_app_user_model_id`
- `winsound.PlaySound` / `AudioPlayer` → `from compat import play_sound_file`
- `subprocess.Popen(['clip'])` → `from compat import clipboard_copy_fallback`
- Both the class constructor AND `__main__` block AppUserModelID calls updated

### 5. `src/server.py` (UPDATED - COMPLETE)
- Inline `_setup_cuda_dlls()` → `from compat import setup_cuda_dlls`
- `_find_ffmpeg()` → delegates to `compat.find_ffmpeg()`
- `_maybe_clear_cuda_cache()` → uses `compat.maybe_clear_gpu_cache()`
- Lifespan defaults: `get_default_engine()` and `get_default_device()` instead of hardcoded "whisper"/"cuda"
- Argparse defaults: same treatment

---

## What's LEFT TO DO

### 6. `src/meeting/capture.py` - macOS Audio Capture (BlackHole)
**Current state:** Uses `pyaudiowpatch` (Windows WASAPI loopback only)
**What to do:**
- Add `sys.platform` check at top of file
- On Windows: keep existing `pyaudiowpatch` code exactly as-is
- On macOS: use `sounddevice` (already a dependency) to find BlackHole virtual audio device
- The `_find_loopback_device()` method needs a macOS branch that searches for a device named "BlackHole" via `sounddevice.query_devices()`
- The `__init__` should use `sounddevice` instead of `pyaudio.PyAudio()` on Mac
- `start()` / `stop()` / callbacks should use `sounddevice.InputStream` on Mac
- Key: `import pyaudiowpatch as pyaudio` should be conditional (only on Windows)
- Hardcoded debug log path `r"C:\dev\koe\capture_debug.log"` should use `Path(__file__).parent.parent.parent / "capture_debug.log"`

### 7. `src/meeting/diarization.py` - Force CPU on macOS
**Current state:** Line 57: `def __init__(self, device: str = "cuda")`
**What to do:**
- In `load()` method (line 116): change device selection to:
  ```python
  if sys.platform == 'darwin':
      device = torch.device("cpu")  # MPS has timestamp bugs with pyannote
  else:
      device = torch.device("cuda" if self._device == "cuda" and torch.cuda.is_available() else "cpu")
  ```
- Same for embedding model on line 126: force CPU on macOS
- The CUDA empty_cache calls in `server.py` transcribe_meeting endpoint (line 749-750) should be guarded with `if torch.cuda.is_available()` (already is, so OK)

### 8. `src/config_schema.yaml` - Add MLX Engine Option
**Current state:** Engine options are `whisper` and `parakeet` (line 27-28)
**What to do:**
- Add `mlx` to the engine options list (line 28):
  ```yaml
  options:
    - whisper
    - parakeet
    - mlx
  ```
- Add MLX engine config section after parakeet (after line 145):
  ```yaml
  # MLX Whisper engine (Apple Silicon only)
  mlx:
    model:
      value: mlx-community/whisper-large-v3-turbo
      type: str
      description: "MLX Whisper model. Apple Silicon only, uses unified memory GPU."
      options:
        - mlx-community/whisper-large-v3-turbo
        - mlx-community/whisper-large-v3-mlx
        - mlx-community/whisper-medium-mlx
        - mlx-community/whisper-small-mlx
        - mlx-community/whisper-base-mlx
  ```

### 9. `requirements-mac.txt` (NEW)
Create at `C:\dev\koe\requirements-mac.txt`:
```
# macOS (Apple Silicon) dependencies for Koe
# Install: pip install -r requirements-mac.txt

# Core (same as Windows)
aiohttp==3.8.4
annotated-types==0.6.0
anyio==4.3.0
audioplayer==0.6
certifi==2023.5.7
colorama==0.4.6
filelock==3.12.0
fsspec==2023.12.2
h11==0.14.0
httpcore==1.0.5
httpx==0.27.0
huggingface-hub==0.20.1
idna==3.4
Jinja2==3.1.2
MarkupSafe==2.1.2
networkx==3.1
numpy==1.24.3
packaging==23.2
Pillow==9.5.0
pydantic==2.7.1
pydantic_core==2.18.2
pynput==1.7.6
pyperclip==1.8.2
python-dotenv==1.0.0
PyYAML==6.0.1
regex==2023.5.5
requests==2.31.0
sounddevice==0.4.6
soundfile==0.12.1
tqdm==4.65.0
typing_extensions==4.11.0
urllib3==2.0.2
webrtcvad-wheels==2.0.11.post1

# Server
fastapi==0.128.0
uvicorn==0.40.0

# AI summarization
anthropic==0.76.0

# Audio processing
scipy==1.11.4

# macOS-specific: MLX Whisper (Apple Silicon GPU acceleration)
mlx-whisper

# macOS-specific: PyQt5 via pip (or use brew install pyqt@5)
PyQt5==5.15.10

# GPU/ML packages for macOS:
# pip install torch torchvision torchaudio (CPU/MPS, no CUDA)
# pip install pyannote.audio

# NOT needed on macOS (Windows-only):
# PyAudioWPatch (WASAPI loopback)
# nvidia-cudnn-cu12
# nvidia-cublas-cu12
# ctranslate2 (used by faster-whisper)
# faster-whisper (use mlx-whisper instead)

# System audio capture on macOS:
# Install BlackHole: brew install blackhole-2ch
# Then create Multi-Output Device in Audio MIDI Setup
```

### 10. `src/setup_wizard.py` - macOS System Checks
**Current state:** `SystemCheckThread.run()` (line 52) checks for nvidia-smi, CUDA, faster_whisper
**What to do:**
- In `SystemCheckThread.run()`, add platform branches:
  - **GPU check:** On macOS, skip nvidia-smi. Instead check for Apple Silicon: `platform.machine() == 'arm64'`
  - **CUDA check:** On macOS, skip CUDA. Instead check for MLX: `import mlx_whisper`
  - **Package check:** On macOS, check for `mlx_whisper` instead of `faster_whisper`. Check for `sounddevice` (not PyAudioWPatch)
- In `ModelDownloadThread.run()` (line 156):
  - On macOS, download MLX model instead of faster-whisper model:
    ```python
    if sys.platform == 'darwin':
        import mlx_whisper
        mlx_whisper.transcribe(np.zeros(16000, dtype=np.float32),
                              path_or_hf_repo="mlx-community/whisper-large-v3-turbo")
    else:
        from faster_whisper import WhisperModel
        model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    ```
- In `_save_configuration()` (line 1189):
  - Set engine to "mlx" on macOS in the generated config.yaml
- In `_launch_koe()` (line 1245):
  - Already has platform branching for `CREATE_NO_WINDOW` - this is fine

---

## Key Architecture Notes

- **Engine system:** `src/engines/` uses a factory pattern with `@register_engine` decorator. Each engine implements `TranscriptionEngine` base class with `load()`, `transcribe()`, `is_available()`. The factory auto-discovers engines via imports in `_register_engines()`.
- **Config:** `src/config.yaml` is the user's config, `src/config_schema.yaml` defines defaults/schema. `ConfigManager` reads both.
- **Audio capture for Scribe:** `src/meeting/capture.py` runs two streams (mic + loopback). On Mac, mic uses `sounddevice` (works already), loopback needs BlackHole device via `sounddevice`.
- **Diarization:** pyannote 3.1 works on Mac CPU. MPS (Metal) produces wrong timestamps - this is a known upstream bug. Force CPU.
- **No changes needed to:** `src/transcription.py` (already uses engine factory), `src/result_thread.py`, `src/key_listener.py` (pynput is cross-platform), `src/ui/` files (PyQt5 is cross-platform).

## Important: Don't Break Windows
- Every change must be guarded with `sys.platform` or use `compat.IS_WINDOWS`/`IS_MACOS`
- The existing `config.yaml` at `src/config.yaml` is the user's personal config - don't modify it
- Only modify `config_schema.yaml` (adds new options, doesn't change existing ones)
- `requirements.txt` stays as-is (Windows deps). `requirements-mac.txt` is new/separate.
