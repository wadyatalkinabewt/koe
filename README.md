# Koe

**Koe** (声 *"koh-eh"* - Japanese for "voice") is a local, privacy-focused speech-to-text application for Windows and macOS with GPU-accelerated transcription and intelligent speaker identification.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [License](#license)

---

## Overview

Koe provides two primary modes of operation:

| Mode | Purpose | Use Case |
|------|---------|----------|
| **Snippet** | Hotkey-triggered transcription | Quick voice notes, dictation, clipboard transcription |
| **Scribe** | Continuous meeting transcription | Meeting notes with speaker identification and AI summaries |

### Key Capabilities

- **Local Processing**: All speech recognition runs locally on your GPU - no cloud services required for transcription
- **Multi-Engine Support**: Choose between Whisper (99 languages), Parakeet (~50x faster, English-only), or MLX Whisper (Apple Silicon)
- **Cross-Platform**: Windows (NVIDIA CUDA) and macOS (Apple Silicon MLX) with platform-specific GPU acceleration
- **Speaker Diarization**: Identifies who said what using voice fingerprinting
- **AI Summarization**: Auto-generates meeting summaries using Claude API (~$0.04/meeting)
- **Remote Support**: Use from a laptop by connecting to your desktop's GPU over Tailscale
- **Privacy-First**: Audio never leaves your network (except optional AI summarization)

### Transcription Engines

| Feature | Whisper | Parakeet | MLX Whisper |
|---------|---------|----------|-------------|
| Speed | 1x (baseline) | ~50x faster | ~15-20x realtime |
| Languages | 99+ languages | English only | 99+ languages |
| Memory | ~3GB VRAM (large-v3) | ~2GB VRAM | ~3GB unified (large-v3-turbo) |
| Platform | Windows (CUDA) | Windows (CUDA) | macOS (Apple Silicon) |
| Setup | Automatic | One-time NeMo install | Automatic |

### Technical Foundation

Built on [WhisperWriter](https://github.com/savbell/whisper-writer) by savbell, extensively modified with:
- Shared server architecture (single GPU model serves multiple clients)
- Multi-engine support (Whisper, Parakeet, and MLX Whisper)
- macOS Apple Silicon support via MLX framework
- Speaker diarization with pyannote-audio
- Meeting transcription with category organization
- Remote transcription over Tailscale
- AI-powered meeting summaries

---

## Features

### Hotkey Transcription (Snippet)

- **Global Hotkey**: `Ctrl+Shift+Space` triggers recording from any application
- **Press-to-Toggle**: Press hotkey to start, press again to stop
- **Clipboard Integration**: Transcribed text copied directly to clipboard
- **Rolling Snippets**: Last 5 transcriptions saved to files (configurable)
- **Voice Filtering**: Optionally transcribe only your voice in noisy environments
- **Audio Feedback**: Configurable beep sound on completion

### Meeting Transcription (Scribe)

- **Dual Audio Capture**: Records both microphone and system audio simultaneously
- **Speaker Identification**: Names speakers using enrolled voice fingerprints
- **Category Organization**: Organize meetings into folders (Standups, One-on-ones, Investors, etc.)
- **Pre-Meeting Agendas**: Save agenda templates before meetings, auto-merged with transcript
- **Note-Taking Interface**: AGENDA, NOTES, ACTION ITEMS sections with live editing
- **AI Summarization**: Automatic meeting summaries with key points and action items

### AI Summarization

- **Claude Sonnet 4.5**: High-quality summaries using Anthropic's API
- **Background Processing**: Window can close - summary continues in detached process
- **Live Progress**: Status updates with clickable link to open completed summary
- **Cost-Effective**: ~$0.04 per 60-minute meeting
- **Anti-Hallucination**: Strict prompting ensures factual accuracy

### Speaker Enrollment

- **Post-Meeting Enrollment (Recommended)**: Enroll unknown speakers directly from meeting recordings - highest quality embeddings
- **Auto-Transcript Rewriting**: When you enroll "Speaker 1" as "Alice", the transcript is automatically updated
- **Similar Speaker Auto-Merge**: If "Speaker 3" matches "Speaker 1", both are replaced with the enrolled name
- **Deferred Summarization**: Summary generation waits until enrollment is complete (uses correct names)
- **Adaptive Learning**: Voice fingerprints improve over time with high-confidence matches
- **CLI Tools**: Available for advanced users/debugging

### Remote Transcription

- **Tailscale Integration**: Secure network connection between devices
- **Lightweight Client**: Laptop requires only ~50MB of dependencies (no GPU packages)
- **Full Feature Parity**: All features work remotely, processing happens on desktop GPU

---

## System Requirements

### Windows Desktop (Server)

| Component | Requirement |
|-----------|-------------|
| **Operating System** | Windows 10/11 (64-bit) |
| **GPU** | NVIDIA GPU with 6GB+ VRAM (8GB+ recommended) |
| **GPU Driver** | NVIDIA Driver 525.60+ with CUDA support |
| **CPU** | Any modern multi-core processor |
| **RAM** | 16GB recommended |
| **Storage** | ~5GB for models and dependencies |
| **Python** | Python 3.10 or higher |

### macOS (Apple Silicon)

| Component | Requirement |
|-----------|-------------|
| **Operating System** | macOS 13+ (Ventura or later) |
| **Chip** | Apple M1/M2/M3/M4 (Apple Silicon required) |
| **RAM** | 16GB+ unified memory (64GB recommended for large models + diarization) |
| **Storage** | ~5GB for models and dependencies |
| **Python** | Python 3.10 or higher |
| **System Audio** | [BlackHole](https://github.com/ExistentialAudio/BlackHole) for Scribe mode |

### Laptop (Remote Client)

| Component | Requirement |
|-----------|-------------|
| **Operating System** | Windows 10/11 or macOS |
| **Network** | Tailscale installed and connected |
| **Storage** | ~100MB for dependencies |
| **Python** | Python 3.10 or higher |

### GPU Memory Usage

| Component | Windows (VRAM) | macOS (Unified Memory) |
|-----------|---------------|----------------------|
| Whisper large-v3 / MLX large-v3-turbo | ~3GB | ~3GB |
| Parakeet CTC 0.6B | ~2GB | N/A (Windows only) |
| Pyannote diarization | ~0.5-1GB (GPU) | ~0.5-1GB (CPU) |
| **Total during meeting** | ~3.5-4GB | ~3.5-4GB |

---

## Architecture

### System Overview

```
    WINDOWS DESKTOP (NVIDIA GPU)              LAPTOP (Remote Client)
    ════════════════════════════              ════════════════════════

         SYSTEM TRAY                              SYSTEM TRAY
    ┌─────────────────┐                      ┌─────────────────┐
    │  Koe            │                      │  Koe            │
    │  Scribe         │◄──── Tailscale ─────►│    (remote)     │
    │  Settings       │     (100.x.x.x)      │  Scribe         │
    └────────┬────────┘                      └─────────────────┘
             │
             ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Koe Server (Port 9876)                       │
    │                                                                 │
    │  ENGINE: Whisper OR Parakeet OR MLX Whisper                    │
    │  ─────────────────────────────────────────                     │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │  Whisper large-v3 (Windows default)      (~3GB VRAM)    │   │
    │  │  - faster-whisper (CTranslate2), 99 languages           │   │
    │  │  OR                                                     │   │
    │  │  Parakeet CTC 0.6B (Windows)             (~2GB VRAM)    │   │
    │  │  - NVIDIA NeMo toolkit, ~50x faster, English only       │   │
    │  │  OR                                                     │   │
    │  │  MLX Whisper large-v3-turbo (macOS)     (~3GB unified)  │   │
    │  │  - Apple MLX framework, ~15-20x realtime on M2 Pro      │   │
    │  └─────────────────────────────────────────────────────────┘   │
    │                                                                 │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │  Pyannote Diarization          (~0.5-1GB GPU or CPU)    │   │
    │  │  - Speaker segmentation (who speaks when)               │   │
    │  │  - Voice embedding extraction (wespeaker)               │   │
    │  │  - GPU on Windows, CPU on macOS (MPS has timestamp bugs)│   │
    │  └─────────────────────────────────────────────────────────┘   │
    │                                                                 │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │  Speaker Embeddings                                     │   │
    │  │  - Enrolled voice fingerprints (speaker_embeddings/)    │   │
    │  │  - Similarity matching (threshold: 0.35)                │   │
    │  └─────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────┘
```

### Component Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              Koe Application                             │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐      │
│  │   System Tray   │    │  Key Listener   │    │  Result Thread  │      │
│  │   (main.py)     │───►│ (key_listener)  │───►│ (result_thread) │      │
│  │                 │    │                 │    │                 │      │
│  │  - Koe menu     │    │  - Global       │    │  - Audio        │      │
│  │  - Scribe menu  │    │    hotkey       │    │    recording    │      │
│  │  - Settings     │    │  - Key capture  │    │  - VAD          │      │
│  │  - Enrollment   │    │                 │    │  - Buffering    │      │
│  └─────────────────┘    └─────────────────┘    └────────┬────────┘      │
│                                                          │               │
│                                                          ▼               │
│                                              ┌─────────────────┐         │
│                                              │  Transcription  │         │
│                                              │  (transcription)│         │
│                                              │                 │         │
│                                              │  - Local model  │         │
│                                              │  - Server API   │         │
│                                              └────────┬────────┘         │
│                                                       │                  │
│                                                       ▼                  │
│                                              ┌─────────────────┐         │
│                                              │    Clipboard    │         │
│                                              │   + Snippets    │         │
│                                              └─────────────────┘         │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                            Scribe Application                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐      │
│  │   Meeting UI    │    │  Audio Capture  │    │    Processor    │      │
│  │   (app.py)      │───►│   (capture.py)  │───►│  (processor.py) │      │
│  │                 │    │                 │    │                 │      │
│  │  - Name/Cat     │    │  - Microphone   │    │  - VAD chunking │      │
│  │  - Notes editor │    │  - Loopback     │    │  - 30-60s       │      │
│  │  - Recording    │    │  - Dual stream  │    │    segments     │      │
│  └─────────────────┘    └─────────────────┘    └────────┬────────┘      │
│                                                          │               │
│                                                          ▼               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐      │
│  │   Summarizer    │◄───│   Transcript    │◄───│   Diarization   │      │
│  │ (summarizer.py) │    │ (transcript.py) │    │(diarization.py) │      │
│  │                 │    │                 │    │                 │      │
│  │  - Claude API   │    │  - Markdown     │    │  - Pyannote     │      │
│  │  - Background   │    │    formatting   │    │  - Speaker ID   │      │
│  │  - Progress     │    │  - Notes merge  │    │  - Embeddings   │      │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘      │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

#### Hotkey Transcription Flow

```
User presses Ctrl+Shift+Space
         │
         ▼
┌─────────────────┐
│  Status Window  │ "Recording..."
│   (draggable)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Audio Capture  │ WebRTC VAD detects speech
│   (16kHz mono)  │
└────────┬────────┘
         │
         ▼ (silence detected OR hotkey pressed again)
┌─────────────────┐
│  Transcription  │ Server or local model
│   Processing    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Post-Process   │ Remove filler words, fix punctuation
└────────┬────────┘
         │
         ├──────────────────────────────┐
         ▼                              ▼
┌─────────────────┐           ┌─────────────────┐
│   Clipboard     │           │  Rolling        │
│   (pyperclip)   │           │  Snippets       │
└─────────────────┘           │  (1-5 files)    │
                              └─────────────────┘
```

#### Meeting Transcription Flow

```
User starts Scribe
         │
         ▼
┌─────────────────┐
│  Meeting Setup  │ Name (required), Category, Subcategory
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ "Start Recording│
└────────┬────────┘
         │
         ├───────────────────────────────────────┐
         ▼                                       ▼
┌─────────────────┐                   ┌─────────────────┐
│   Microphone    │                   │  System Audio   │
│   (16kHz)       │                   │  (WASAPI/       │
│   → Your voice  │                   │   BlackHole)    │
└────────┬────────┘                   └────────┬────────┘
         │                                     │
         └──────────────┬──────────────────────┘
                        ▼
              ┌─────────────────┐
              │  Audio Chunking │ 30-60 second segments
              │  (VAD-based)    │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Server API     │ /transcribe_meeting
              │  + Diarization  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Speaker Matching│ Compare to enrolled voices
              │ Cross-chunk     │ Track speakers across chunks
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Transcript     │ Timestamped, speaker-labeled
              │  Accumulation   │
              └────────┬────────┘
                       │
         ▼ (User clicks "Stop Recording")
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
┌─────────────────┐       ┌─────────────────┐
│  Save Markdown  │       │  AI Summary     │
│  Transcript     │       │  (detached)     │
│                 │       │                 │
│  Notes +        │       │  Claude API     │
│  Full Transcript│       │  → Summaries/   │
└─────────────────┘       └─────────────────┘
```

### File Organization

```
C:\dev\koe\
│
├── run.py                          # Application entry point
├── config.yaml                     # User configuration
├── config_schema.yaml              # Configuration schema with defaults
├── .env                            # Environment variables (API keys, server URL)
│
├── requirements.txt                # Windows desktop dependencies (~3GB)
├── requirements-mac.txt            # macOS Apple Silicon dependencies
├── requirements-remote.txt         # Laptop dependencies (~50MB)
│
├── assets/
│   ├── koe-icon.ico               # Application icon (multi-size)
│   ├── koe-icon.png               # Application icon (256x256)
│   └── beep.wav                   # Completion sound
│
├── scripts/
│   ├── Start Koe Desktop.bat      # Desktop launcher
│   ├── Start Koe Desktop.vbs      # Hidden console wrapper
│   ├── Start Koe Remote.bat       # Laptop launcher
│   ├── Start Scribe Desktop.bat   # Scribe desktop launcher
│   ├── Start Scribe Remote.bat    # Scribe laptop launcher
│   ├── Stop Koe.bat               # Kill all processes
│   ├── create_shortcuts.ps1       # Generate .lnk shortcuts
│   └── generate_icon.py           # Generate icon files
│
├── .setup_complete                    # Marker file (created after setup wizard)
├── .session_state.npz                 # Diarization session (survives server restarts, 1hr TTL)
├── .transcript_recovery.jsonl         # Crash recovery (auto-deleted on save)
│
├── src/
│   ├── main.py                    # Koe application
│   ├── setup_wizard.py            # First-time setup wizard
│   ├── transcription.py           # Transcription logic
│   ├── transcription_client.py    # HTTP client for server
│   ├── result_thread.py           # Recording thread with VAD
│   ├── key_listener.py            # Global hotkey detection
│   ├── utils.py                   # ConfigManager singleton
│   ├── compat.py                  # Platform abstraction (Windows/macOS)
│   ├── logger.py                  # Centralized error logging
│   │
│   ├── server.py                  # FastAPI transcription server
│   ├── server_tray.py             # Server with system tray
│   ├── server_launcher.py         # Background server starter
│   │
│   ├── meeting/
│   │   ├── app.py                 # Scribe application
│   │   ├── capture.py             # Dual audio capture
│   │   ├── processor.py           # Audio chunking with VAD
│   │   ├── transcript.py          # Markdown transcript writer
│   │   ├── diarization.py         # Speaker identification
│   │   ├── enroll_speaker.py      # Speaker enrollment (mic)
│   │   ├── record_loopback.py     # Speaker enrollment (system audio)
│   │   ├── summarizer.py          # Claude API client
│   │   ├── summarize_detached.py  # Background summarization
│   │   └── summary_status.py      # Status file management
│   │
│   ├── engines/
│   │   ├── base.py                # Engine abstract base class
│   │   ├── factory.py             # Engine registry and creation
│   │   ├── whisper_engine.py      # Whisper (faster-whisper, Windows)
│   │   ├── parakeet_engine.py     # Parakeet (NVIDIA NeMo, Windows)
│   │   └── mlx_engine.py          # MLX Whisper (Apple Silicon, macOS)
│   │
│   └── ui/
│       ├── base_window.py         # Base window class
│       ├── main_window.py         # Main Koe window
│       ├── settings_window.py     # Settings UI
│       ├── status_window.py       # Recording status popup
│       ├── initialization_window.py  # Startup splash
│       └── theme.py               # Color theme constants
│
├── speaker_embeddings/            # Voice fingerprints
│   ├── Bryce.npy                  # Example enrolled speaker
│   └── Calum.npy                  # Example enrolled speaker
│
├── Snippets/                      # Rolling hotkey snippets (configurable)
│   ├── snippet_1.md               # Most recent
│   ├── snippet_2.md
│   └── ...                        # Up to 5 files
│
├── Meetings/                      # Meeting output (configurable)
│   ├── Transcripts/
│   │   ├── Standups/              # Category folder
│   │   │   └── 26_01_20_Daily.md
│   │   └── One-on-ones/
│   │       └── Calum/             # Subcategory folder
│   │           └── 26_01_15_Weekly.md
│   └── Summaries/                 # AI summaries (mirrors Transcripts)
│       └── ...
│
├── logs/
│   ├── koe_errors.log             # Application error log
│   ├── debug.log                  # Koe hotkey debug log (rotates at 1MB)
│   ├── meeting_debug.log          # Scribe meeting debug log
│   └── failed_audio_*.wav         # Failed transcriptions (retried on meeting stop)
│
└── .summary_status/               # Temporary summarization status (auto-cleaned)
```

---

## Installation

### Windows Desktop Setup

#### Prerequisites

1. **NVIDIA GPU with CUDA support**
   ```powershell
   # Verify GPU is detected
   nvidia-smi
   ```

2. **Python 3.10+**
   - Download from [python.org](https://www.python.org/downloads/)
   - **Important**: Check "Add Python to PATH" during installation

3. **Git** (optional, for updates)
   - Download from [git-scm.com](https://git-scm.com/downloads)

#### Quick Install with Setup Wizard (Recommended)

The setup wizard guides you through the entire setup process automatically:

```powershell
# 1. Clone or download the repository
git clone https://github.com/wadyatalkinabewt/koe.git C:\dev\koe

# 2. Install dependencies
cd C:\dev\koe
pip install -r requirements.txt

# 3. Run Koe - setup wizard launches automatically on first run
python run.py
```

The wizard will:
- Check system requirements (GPU, CUDA, Python)
- Guide you through API key setup
- Download required AI models (~3-4GB)
- Configure your name and voice
- Set up output folders
- Launch Koe when complete

**To re-run the setup wizard later:**
```powershell
python run.py --setup
```

#### Manual Installation (Alternative)

If you prefer manual setup or the wizard doesn't work:

1. **Clone or download the repository**
   ```powershell
   git clone https://github.com/wadyatalkinabewt/koe.git C:\dev\koe
   # Or download and extract ZIP to C:\dev\koe
   ```

2. **Install dependencies**
   ```powershell
   cd C:\dev\koe
   pip install -r requirements.txt
   ```

3. **Install GPU packages** (for speaker diarization)
   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install pyannote.audio
   ```
   This installs ~3GB of PyTorch and speaker diarization models.

4. **Configure environment variables**

   Create or edit `.env` file:
   ```ini
   # Required for pyannote speaker diarization models
   HF_TOKEN=hf_your_huggingface_token

   # Optional: For AI meeting summaries
   ANTHROPIC_API_KEY=sk-ant-your_anthropic_key

   # Server URL (localhost for desktop)
   WHISPER_SERVER_URL=http://localhost:9876
   ```

   **HuggingFace Token**: Get from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

   **Anthropic API Key**: Get from [console.anthropic.com](https://console.anthropic.com)

5. **Accept model licenses**

   Visit these pages and accept the license agreements:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

6. **Configure your name**

   Edit `config.yaml`:
   ```yaml
   profile:
     user_name: YourName
   ```

7. **Test the installation**
   ```powershell
   # Start Koe
   python run.py
   ```
   First startup takes 30-60 seconds as models load into GPU memory.

8. **Mark setup complete** (skips wizard on future launches)
   ```powershell
   # Create marker file
   echo. > .setup_complete
   ```

### macOS Setup (Apple Silicon)

#### Prerequisites

1. **Apple Silicon Mac** (M1/M2/M3/M4)
2. **Python 3.10+** (via Homebrew recommended: `brew install python`)
3. **BlackHole** (for Scribe system audio capture):
   ```bash
   brew install blackhole-2ch
   ```
4. **ffmpeg** (for audio file processing):
   ```bash
   brew install ffmpeg
   ```

#### Quick Install

```bash
# 1. Clone the repository
git clone https://github.com/wadyatalkinabewt/koe.git ~/dev/koe

# 2. Install macOS dependencies
cd ~/dev/koe
pip install -r requirements-mac.txt

# 3. Install PyTorch (CPU/MPS) and pyannote
pip install torch torchvision torchaudio
pip install pyannote.audio

# 4. Run Koe - setup wizard launches automatically
python run.py
```

The setup wizard detects Apple Silicon and configures MLX Whisper automatically.

#### BlackHole Setup (for Scribe)

To capture system audio on macOS, you need a virtual audio device:

1. Install BlackHole: `brew install blackhole-2ch`
2. Open **Audio MIDI Setup** (Spotlight → "Audio MIDI Setup")
3. Click `+` → **Create Multi-Output Device**
4. Check both your speakers/headphones AND **BlackHole 2ch**
5. Set the Multi-Output Device as your system output
6. Koe will automatically detect BlackHole as the loopback input

#### macOS Notes

- **Transcription engine**: MLX Whisper uses Apple's MLX framework for GPU-accelerated transcription via unified memory (~15-20x realtime on M2 Pro)
- **Speaker diarization**: Runs on CPU (pyannote MPS has known timestamp bugs). Works correctly but is slower than GPU (~5-15 min per hour of audio)
- **Hotkeys**: macOS requires granting Accessibility permissions (System Settings → Privacy & Security → Accessibility) for `pynput` to capture global hotkeys
- **PyQt5**: Install via `pip install PyQt5` or `brew install pyqt@5`

### Laptop Setup (Remote Client)

#### Prerequisites

1. **Tailscale** installed on both desktop and laptop
   - Download from [tailscale.com/download](https://tailscale.com/download)
   - Sign in with the same account on both machines

2. **Desktop running with Koe started**

#### Installation Steps

1. **Copy the koe folder to laptop**
   ```
   C:\dev\koe
   ```

2. **Install lightweight dependencies**
   ```powershell
   cd C:\dev\koe
   pip install -r requirements-remote.txt
   ```
   This installs only ~50MB (no GPU packages).

3. **Configure server URL**

   Edit `.env`:
   ```ini
   # Your desktop's Tailscale IP
   WHISPER_SERVER_URL=http://100.x.x.x:9876
   ```

   Find your desktop's Tailscale IP:
   ```powershell
   tailscale ip
   ```

4. **Test the connection**
   ```powershell
   curl http://100.x.x.x:9876/status
   ```

### Parakeet Engine Setup (Optional, ~50x Faster)

Parakeet provides ~50x faster transcription but only supports English.

#### Prerequisites

1. **Visual C++ Build Tools** (required for NeMo dependencies)
   ```powershell
   winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
   ```

2. **Install NeMo toolkit**
   ```bash
   pip install nemo_toolkit[asr]
   ```

#### Setup Steps

1. **Select Parakeet in Settings**
   - Open Settings → Transcription Engine → Select "Parakeet"
   - Select model (CTC 0.6B recommended)
   - Click Save

2. **Restart Koe**
   - Exit via tray icon
   - Launch again - Parakeet server starts (~30 seconds to load)

3. **Verify**
   ```powershell
   curl http://localhost:9876/status
   # Should show: {"model":"nvidia/parakeet-ctc-0.6b",...}
   ```

---

## Configuration

### Configuration Files

| File | Purpose |
|------|---------|
| `config.yaml` | User settings (hotkey, recording options, etc.) |
| `config_schema.yaml` | Schema with defaults and validation |
| `.env` | Environment variables (API keys, server URL) |

### Settings (configurable in UI)

| Setting | Description |
|---------|-------------|
| **Your Name** | Labels your mic audio in Scribe transcripts |
| **My Voice** | Select your enrolled voice for snippet filtering |
| **Filter snippets to my voice** | Only transcribe your voice in Snippet mode (adds latency) |
| **Meetings Root Folder** | Where meeting transcripts are saved |
| **Snippets Folder** | Where rolling hotkey snippets are saved |
| **Engine** | Transcription engine: Whisper (Windows default), Parakeet (~50x faster), or MLX (macOS default) |
| **Model** | Model to use (e.g., large-v3 for Whisper, parakeet-ctc-0.6b for Parakeet) |
| **Device** | Auto (detect GPU), CUDA (NVIDIA GPU), or CPU (no GPU required) |
| **Toggle Hotkey** | Global hotkey (default: `Ctrl+Shift+Space`) |
| **Play sound on completion** | Beep when transcription finishes |
| **Transcription Prompt** | Hint for transcription (names, punctuation style) |

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Desktop | HuggingFace token for pyannote models |
| `WHISPER_SERVER_URL` | Remote | Desktop server URL (e.g., `http://100.x.x.x:9876`) |
| `ANTHROPIC_API_KEY` | Optional | For AI meeting summaries |

---

## Usage Guide

### Snippet (Hotkey Transcription)

1. **Start Koe**
   - Double-click `Start Koe Desktop` shortcut
   - Wait for tray icon to appear (first startup: 30-60 seconds)

2. **Record and transcribe**
   - Press `Ctrl+Shift+Space` to start recording
   - Speak clearly
   - Press `Ctrl+Shift+Space` again to stop
   - Text is copied to clipboard

3. **Cancel recording**
   - Press the hotkey again to stop and discard

4. **Access settings**
   - Right-click tray icon → Settings

5. **Exit**
   - Right-click tray icon → Exit

### Scribe (Meeting Transcription)

1. **Start Scribe**
   - Double-click `Start Scribe Desktop` shortcut
   - Or right-click Koe tray icon → Start Scribe

2. **Set up meeting**
   - Enter meeting name (required)
   - Select category/subcategory (optional)

3. **Pre-meeting agenda (optional)**
   - Write agenda in AGENDA section
   - Click "Save Notes" to save template
   - Click "Open" later to load saved agenda

4. **Record meeting**
   - Click `[ START RECORDING ]`
   - Take notes in NOTES and ACTION ITEMS sections
   - Transcription happens in background

5. **End meeting**
   - Click `[ STOP RECORDING ]`
   - Wait for "Saving transcript..." to complete
   - AI summary generates automatically (if API key configured)

6. **View outputs**
   - Transcript: `Meetings/Transcripts/[Category]/YY_MM_DD_Name.md`
   - Summary: `Meetings/Summaries/[Category]/YY_MM_DD_Name.md`

### Speaker Enrollment

#### Via Post-Meeting Dialog (Recommended)

The easiest way to enroll speakers is directly from a meeting recording:

1. **Record a meeting** in Scribe with the person speaking
2. **Stop recording** - an "Enroll Speakers" button appears in the summary window
3. **Click "Enroll Speakers"** - dialog shows:
   - **Unknown speakers** (green): "Speaker 1", "Speaker 2" with sample transcriptions
   - **Enrolled speakers** (blue): Speakers who were on the call with "Update" option
4. **For unknown speakers**: Enter their name and click "Enroll"
5. **For enrolled speakers**: Click "Update embedding" to improve quality with meeting audio

**Benefits:**
- No separate enrollment sessions needed
- Uses running average of entire meeting (higher quality than short recordings)
- See sample transcriptions to identify who is who

#### Via Command Line (For Debugging)

```powershell
# Enroll from microphone
python -m src.meeting.enroll_speaker "YourName"

# Enroll from system audio (e.g., YouTube video)
python -m src.meeting.record_loopback "TheirName" --duration 30 --enroll

# List enrolled speakers
python -m src.meeting.enroll_speaker --list dummy

# Remove speaker
python -m src.meeting.enroll_speaker "Name" --remove
```

### Remote Usage (Laptop)

1. **Ensure desktop is running** with Koe started
2. **Start Koe Remote** or **Start Scribe Remote**
3. Use exactly as on desktop - processing happens on your desktop GPU

---

## API Reference

The Koe server exposes a REST API on port 9876.

### Endpoints

#### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

#### GET /status

Server status and capabilities.

**Response:**
```json
{
  "status": "ok",
  "model": "large-v3",
  "device": "cuda",
  "ready": true,
  "diarization_available": true,
  "supports_long_audio": true,
  "busy": false,
  "active_requests": 0
}
```

| Field | Description |
|-------|-------------|
| `busy` | `true` if any transcription requests are in progress |
| `active_requests` | Number of in-flight transcription requests |
| `supports_long_audio` | `true` if server can handle >60 second audio |

#### POST /transcribe

Transcribe audio without speaker diarization.

**Request:**
```json
{
  "audio_base64": "<base64-encoded int16 PCM audio>",
  "sample_rate": 16000,
  "language": null,
  "initial_prompt": null,
  "vad_filter": true
}
```

**Response:**
```json
{
  "text": "Transcribed text here.",
  "duration_seconds": 5.2
}
```

#### POST /transcribe_meeting

Transcribe audio with speaker diarization (for Scribe).

**Request:**
```json
{
  "audio_base64": "<base64-encoded int16 PCM audio>",
  "sample_rate": 16000,
  "language": null,
  "initial_prompt": null,
  "max_speakers": 8
}
```

**Response:**
```json
{
  "segments": [
    {
      "speaker": "Bryce",
      "text": "Hello everyone.",
      "start": 0.0,
      "end": 1.5
    },
    {
      "speaker": "Speaker 1",
      "text": "Thanks for joining.",
      "start": 1.5,
      "end": 3.2
    }
  ],
  "duration_seconds": 30.5
}
```

#### POST /diarization/reset

Reset diarization session (call at start of new meeting).

**Response:**
```json
{
  "status": "ok"
}
```

#### GET /speakers

List enrolled speakers available for matching.

**Response:**
```json
{
  "speakers": ["Bryce", "Calum", "Sash"]
}
```

---

## Dependencies

### Windows Desktop Dependencies (requirements.txt)

#### Core ML/Audio Stack

| Package | Version | Purpose |
|---------|---------|---------|
| `faster-whisper` | 1.0.2 | Optimized Whisper implementation |
| `ctranslate2` | 4.2.1 | Efficient transformer inference |
| `torch` | - | PyTorch deep learning framework |
| `onnxruntime` | 1.16.3 | Neural network runtime |
| `pyannote-audio` | - | Speaker diarization |
| `pyannote-core` | - | Diarization core library |

#### Audio Processing

| Package | Version | Purpose |
|---------|---------|---------|
| `sounddevice` | 0.4.6 | Microphone input capture |
| `pyaudiowpatch` | - | WASAPI loopback (system audio) |
| `soundfile` | 0.12.1 | Audio file I/O |
| `webrtcvad-wheels` | 2.0.11.post1 | Voice activity detection |
| `numpy` | 1.26.4 | Numerical array processing |
| `scipy` | 1.11.4 | High-quality resampling with anti-aliasing |

#### UI Framework

| Package | Version | Purpose |
|---------|---------|---------|
| `PyQt5` | 5.15.10 | Desktop GUI framework |
| `PyQt5-Qt5` | 5.15.2 | Qt5 bindings |

#### Networking & API

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | - | HTTP API framework (server) |
| `uvicorn` | - | ASGI server |
| `requests` | 2.31.0 | HTTP client |
| `anthropic` | 0.76.0 | Claude API client |

#### System Integration

| Package | Version | Purpose |
|---------|---------|---------|
| `pynput` | 1.7.6 | Global hotkey detection |
| `pyperclip` | 1.8.2 | Clipboard access |
| `python-dotenv` | 1.0.0 | Environment variable loading |
| `PyYAML` | 6.0.1 | YAML configuration files |
| `pydantic` | 2.7.1 | Data validation |

#### Other

| Package | Version | Purpose |
|---------|---------|---------|
| `huggingface-hub` | 0.20.1 | Model downloads |
| `audioplayer` | 0.6 | Beep sound playback |
| `Pillow` | 9.5.0 | Image processing (icons) |
| `tqdm` | 4.65.0 | Progress bars |
| `colorama` | 0.4.6 | Colored console output |

### macOS Dependencies (requirements-mac.txt)

| Package | Purpose |
|---------|---------|
| `mlx-whisper` | MLX Whisper for Apple Silicon GPU transcription |
| `sounddevice` | Audio capture (mic + BlackHole loopback) |
| `webrtcvad-wheels` | Voice activity detection |
| `numpy` | Audio processing |
| `soundfile` | Audio file I/O |
| `scipy` | High-quality audio resampling |
| `PyQt5` | GUI framework |
| `pynput` | Hotkey detection |
| `pyperclip` | Clipboard access |
| `fastapi` / `uvicorn` | Server |
| `anthropic` | AI summarization |
| `requests` | HTTP client |
| `pyyaml` | Configuration |
| `python-dotenv` | Environment variables |

**Not needed on macOS:** `PyAudioWPatch`, `nvidia-cudnn-cu12`, `nvidia-cublas-cu12`, `ctranslate2`, `faster-whisper`

**Also install separately:** `torch`, `pyannote.audio` (for speaker diarization)

### Remote Dependencies (requirements-remote.txt)

Lightweight ~50MB package set for laptop clients:

| Package | Purpose |
|---------|---------|
| `sounddevice` | Microphone capture |
| `pyaudiowpatch` | System audio capture |
| `webrtcvad-wheels` | Voice activity detection |
| `numpy` | Audio processing |
| `soundfile` | Audio file I/O |
| `PyQt5` | GUI framework |
| `pynput` | Hotkey detection |
| `pyperclip` | Clipboard access |
| `requests` | HTTP client (to server) |
| `pyyaml` | Configuration |
| `python-dotenv` | Environment variables |
| `audioplayer` | Notification sound |

---

## Troubleshooting

### Desktop Issues

#### CUDA/GPU Not Detected

```powershell
# Verify GPU is visible
nvidia-smi

# Check CUDA version
nvcc --version
```

**Solutions:**
- Update NVIDIA drivers: [nvidia.com/drivers](https://www.nvidia.com/drivers)
- Ensure 6GB+ VRAM is available
- Close other GPU-intensive applications

#### Model Download Fails

**Causes:**
- Invalid HuggingFace token
- Model license not accepted
- Network issues

**Solutions:**
1. Verify token in `.env`: `HF_TOKEN=hf_xxx`
2. Accept licenses at huggingface.co
3. Check internet connection

#### First Startup Very Slow

**Normal behavior**: First startup loads ~3GB of models into GPU memory. Subsequent startups are faster as models are cached.

#### Hotkey Not Working

**Solutions:**
1. Check tray icon exists
2. Some applications block global hotkeys (games, admin windows)
3. Try different hotkey in Settings
4. Restart Koe

#### Multiple Tray Icons

Koe has single-instance protection. If multiple icons appear, exit via tray icon → Exit (server stops automatically) and restart.

### macOS Issues

#### MLX Whisper Not Detected

```bash
# Verify MLX is installed
python -c "import mlx_whisper; print('OK')"

# Verify Apple Silicon
python -c "import platform; print(platform.machine())"  # Should print "arm64"
```

**Solutions:**
- Install mlx-whisper: `pip install mlx-whisper`
- Ensure you're on Apple Silicon (Intel Macs are not supported)

#### No System Audio in Scribe (macOS)

BlackHole must be installed and configured:
1. Install: `brew install blackhole-2ch`
2. Open Audio MIDI Setup
3. Create Multi-Output Device (speakers + BlackHole)
4. Set as system output
5. Restart Koe

#### Hotkey Not Working (macOS)

macOS requires Accessibility permissions for global hotkeys:
1. System Settings → Privacy & Security → Accessibility
2. Add your terminal app or Python to the list
3. Restart Koe

#### Diarization Slow on macOS

This is expected - pyannote runs on CPU on macOS (MPS/Metal has timestamp bugs). Diarization of a 1-hour meeting takes ~5-15 minutes on CPU. The 64GB unified memory ensures no memory constraints.

### Parakeet Engine Issues

#### Parakeet Shows "Not Installed"

**Cause:** NeMo toolkit not installed.

**Solutions:**
1. Install Visual C++ Build Tools: `winget install Microsoft.VisualStudio.2022.BuildTools`
2. Install NeMo: `pip install nemo_toolkit[asr]`
3. Verify: `python -c "import nemo.collections.asr"`

#### Parakeet Server Not Starting

```bash
# Check if NeMo is installed
pip show nemo_toolkit

# Check server status
curl http://localhost:9876/status

# Check for import errors
python -c "import nemo.collections.asr"
```

#### Parakeet Slower Than Expected

1. First transcription is slow (model loading) - subsequent ones are fast
2. Verify GPU is being used: `curl http://localhost:9876/status` should show `"device":"cuda"`
3. Check CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

#### Switching Between Engines

1. Change engine in Settings → Transcription Engine
2. Exit Koe completely (tray icon → Exit) - server stops automatically
3. Restart Koe - new engine will start

### Remote/Laptop Issues

#### "Server Not Available" Error

1. **Verify Tailscale connection**
   ```powershell
   tailscale status
   ```

2. **Verify desktop is running Koe**
   - Check for tray icon on desktop

3. **Test server connectivity**
   ```powershell
   curl http://100.x.x.x:9876/status
   ```

4. **Check firewall**
   - Windows Firewall may block port 9876
   - Add exception or temporarily disable to test

#### Wrong Server URL

Edit `.env` on laptop:
```ini
WHISPER_SERVER_URL=http://YOUR_DESKTOP_TAILSCALE_IP:9876
```

### Scribe Issues

#### Too Many Speakers Detected

In a 2-person meeting, if you see "Speaker 1", "Speaker 2", "Speaker 3"...:

1. **Use enrollment dialog**: After meeting ends, enroll one of the unknown speakers
2. **Auto-merge kicks in**: Similar speakers are automatically merged by embedding similarity
3. **Transcript rewrites**: All instances of merged speakers get renamed automatically
4. Re-enroll existing speakers with longer, clearer audio samples if matching isn't working

#### Speaker Not Recognized

**Solutions:**
1. Re-enroll with 10-30 seconds of clear audio
2. Ensure quiet environment during enrollment
3. Try enrolling from different audio sources

#### AI Summary Not Generating

1. Verify `ANTHROPIC_API_KEY` in `.env`
2. Check for errors in `logs/koe_errors.log`
3. Ensure API key has sufficient credits

#### Scribe Crashed Mid-Meeting

If Scribe crashes while recording:
1. **Recovery on startup**: Next time you open Scribe, a dialog will appear asking to save recovered transcript
2. **Click "Save"**: Recovered transcript saved with "_recovered" suffix
3. **Manual recovery**: Check if `.transcript_recovery.jsonl` exists in koe folder

**Note**: Recovery saves transcript text only - speaker embeddings may be incomplete.

#### Server Crashed/Restarted Mid-Meeting

If the server crashes during a Scribe meeting:
1. **Session state persists**: Speaker embeddings saved to `.session_state.npz` (1 hour TTL)
2. **Failed chunks retried**: When meeting stops, Scribe retries failed audio files
3. **Safe restart**: Use `python src/server_launcher.py restart` which waits for active transcriptions

### General Issues

#### Clipboard Copy Fails

Koe uses multiple clipboard methods with automatic fallback:
1. pyperclip (primary)
2. Windows clip.exe (fallback)
3. Retries on failure

If issues persist, restart Koe.

#### Error Logs

Check `logs/koe_errors.log` for detailed error information:
```powershell
Get-Content logs\koe_errors.log -Tail 50
```

---

## Development

### Project Structure

```
src/
├── main.py              # Application entry, tray menu, settings
├── compat.py            # Platform abstraction (Windows/macOS)
├── transcription.py     # Core transcription logic
├── key_listener.py      # Global hotkey capture
├── result_thread.py     # Recording thread with VAD
├── utils.py             # ConfigManager, utilities
├── logger.py            # Centralized logging
│
├── server.py            # FastAPI transcription server
├── server_launcher.py   # Server process management
│
├── engines/             # Transcription engine backends
│   ├── base.py          # Abstract base class
│   ├── factory.py       # Engine registry
│   ├── whisper_engine.py    # Whisper (Windows, CUDA)
│   ├── parakeet_engine.py   # Parakeet (Windows, CUDA)
│   └── mlx_engine.py       # MLX Whisper (macOS, Apple Silicon)
│
├── meeting/             # Scribe module
│   ├── app.py           # Main Scribe application
│   ├── capture.py       # Audio capture (mic + loopback, cross-platform)
│   ├── processor.py     # Audio chunking
│   ├── transcript.py    # Markdown output
│   ├── diarization.py   # Speaker identification (GPU on Windows, CPU on macOS)
│   └── summarizer.py    # AI summaries
│
└── ui/                  # PyQt5 UI components
    ├── theme.py         # Color constants
    ├── settings_window.py
    └── status_window.py
```

### Key Design Patterns

- **Singleton ConfigManager**: Single source of truth for configuration
- **Signal/Slot (PyQt5)**: Thread-safe UI updates
- **Background Threads**: Heavy operations don't block UI
- **Detached Subprocess**: AI summarization continues after window closes
- **Platform Abstraction**: `src/compat.py` provides cross-platform helpers (locks, sound, clipboard, GPU). All platform-specific code uses `sys.platform` guards
- **Engine Factory**: `@register_engine` decorator pattern for pluggable transcription backends (Whisper, Parakeet, MLX)

### Running from Source

```powershell
cd C:\dev\koe
python run.py
```

### Running Tests

```powershell
# Test server connectivity
curl http://localhost:9876/status

# Test transcription
python -c "from src.transcription import transcribe; print('OK')"
```

### Code Style

- Python 3.10+ type hints
- Pydantic for data validation
- PyQt5 signals for cross-thread communication
- Centralized error logging

---

## License

Based on [WhisperWriter](https://github.com/savbell/whisper-writer) by savbell.

---

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Optimized Whisper implementation (Windows)
- [mlx-whisper](https://github.com/ml-explore/mlx-examples) - Apple MLX Whisper implementation (macOS)
- [pyannote-audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization
- [Anthropic Claude](https://anthropic.com) - AI summarization
- [BlackHole](https://github.com/ExistentialAudio/BlackHole) - macOS virtual audio driver
