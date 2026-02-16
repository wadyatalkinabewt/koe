# Koe - Claude Context

## Overview

Local speech-to-text tool with two modes:
1. **Koe** - Hotkey-triggered transcription → clipboard
2. **Scribe** - Continuous meeting transcription with speaker separation

Both modes share a transcription server, avoiding GPU memory conflicts. Two engines available:
- **Whisper** (default) - Supports all languages, ~3GB VRAM for large-v3
- **Parakeet** (~50x faster) - English only, ~2GB VRAM, requires one-time NeMo install

Select engine in Settings → Transcription Engine. Server restarts automatically when engine changes.

**User:** Bryce (New Zealand accent)
**Location:** C:\dev\koe
**GitHub:** https://github.com/wadyatalkinabewt/koe
**GPU:** NVIDIA RTX 3070 (8GB VRAM)

---

## Quick Start

### Koe (Hotkey Transcription)
```
Double-click: Start Koe Desktop shortcut (in koe folder)
Hotkey: Ctrl+Shift+Space (press to start, press again to stop)
Status: Shows "Recording..." → "Transcribing..." → closes on beep
Cancel: Press hotkey again to stop and discard (popup is draggable)
Output: Copied to clipboard + saved to Snippets/ folder
Beep: Plays when transcription completes (configurable in Settings)
Note: Once transcribing starts, result always delivered (even if Escape pressed)
```

### Scribe (Meeting Transcription)
```
Double-click: Start Scribe Desktop shortcut (in koe folder)
Enter meeting name (required), select category/folder
Take notes in the editor (Agenda/Notes/Action Items template)
Click "● REC" when meeting begins
Output: Markdown transcript in Meetings/Transcripts/[Category]/[Folder]/
        AI summary in Meetings/Summaries/[Category]/[Folder]/
```

### AI Summarization (Auto-Generated)
```
Stop Recording → Status shows "Generating summary..." (~30-60s)
Window can be closed - summary continues in background
When complete: Clickable green link appears (if window still open)
Click link → Opens summary in VS Code
Summary saved to Meetings/Summaries/ (mirrors Transcripts folder)
Cost: ~$0.04 per 60-minute meeting

Setup Required:
1. Add ANTHROPIC_API_KEY to .env file
2. Get key at: https://console.anthropic.com/
```

### Pre-Meeting Agenda Prep
```
1. Open Scribe
2. Enter meeting name, select category/folder
3. Write agenda under AGENDA section
4. Click "Save" (top right, subtle button) → saves as MeetingName.md (no date)
5. Later: Click "Open" → select your agenda file
6. Click "● REC" → notes + transcript saved as YY_MM_DD_MeetingName.md
```

### Speaker Enrollment (Post-Meeting)
```
After a meeting ends with unknown speakers (Speaker 1, Speaker 2, etc.):
1. Summary window shows orange "Enroll Speakers" button
2. Click to open enrollment dialog showing unknown speakers with sample transcriptions
3. Type name and click "Enroll" to save the speaker
4. Transcript auto-rewrites: all "Speaker 1" entries become the enrolled name
5. Similar speakers auto-merge: if "Speaker 3" matches "Speaker 1", both become the enrolled name
6. Summary generation waits until enrollment complete (uses correct names)

Note: Meeting embeddings are running averages from the entire call -
higher quality than quick recordings. This is the recommended way to enroll speakers.
```

### File Transcription (CLI)
```
Transcribe any audio file via the Koe server:
python transcribe_file.py <audio_file>

Examples:
  python transcribe_file.py voice_note.ogg
  python transcribe_file.py recording.mp3

Supports: .ogg, .oga, .mp3, .wav, .m4a, and most audio formats
Requires: Koe server running, ffmpeg installed

For remote server, set WHISPER_SERVER_URL environment variable.

Use case: AI agents transcribing Telegram voice notes, batch processing, etc.
```

### Ronin/Moltbot Integration
```
Ronin (AI agent running in Docker) uses Koe for voice message transcription.

Script: transcribe_audio.sh (in koe folder)
  - Called by moltbot's media understanding pipeline
  - Converts audio to PCM, base64 encodes, sends to Koe server
  - Server accessible from container via host.docker.internal:9876

Config: C:\dev\ronin\config\moltbot.json → tools.media.audio
Mount: C:\dev mounted at /mnt/dev:ro in container

To verify:
  docker logs moltbot-moltbot-gateway-1 --tail 50 2>&1 | grep "Media understanding"

See: C:\dev\ronin\CLAUDE.md for full Ronin setup
```

### Laptop (Remote Transcription)
```
Prerequisites: Desktop must be running with server started
1. Copy folder to laptop, run: pip install -r requirements-remote.txt
2. Edit .env:
   - Set WHISPER_SERVER_URL=http://100.78.59.64:9876
   - Optional: Set KOE_API_TOKEN=<same-token-as-desktop> (for authentication)
3. Double-click: Start Koe Remote shortcut
4. Hotkey: Ctrl+Shift+Space
5. For meetings: Start Scribe Remote shortcut
```

### First-Time Setup (New Installations)
```
Run: python run.py
```
Terminal-based setup walks through:
1. **Model Selection** - Choose based on hardware:
   | Model     | Size    | Best For                              |
   |-----------|---------|---------------------------------------|
   | tiny      | ~75MB   | Testing, very fast                    |
   | base      | ~150MB  | CPU-only systems (recommended)        |
   | small     | ~500MB  | Better accuracy, still CPU-friendly   |
   | medium    | ~1.5GB  | High accuracy, needs ~2GB VRAM        |
   | large-v3  | ~3GB    | Best accuracy, needs ~4GB VRAM (GPU)  |

2. **HuggingFace Token** (optional) - Enables speaker diarization
   - Skip if you only need basic transcription
   - Can add later via `.env` file: `HF_TOKEN=hf_...`

3. **Anthropic API Key** (optional) - Enables AI meeting summaries
   - Skip if you don't need auto-generated summaries
   - Can add later via `.env` file: `ANTHROPIC_API_KEY=sk-ant-...`

4. **Your Name** - Labels your voice in transcripts

5. **Output Folders** - Where to save meetings/snippets

Re-run setup anytime: `python run.py --setup`

### Parakeet Engine Setup (Optional, ~50x Faster)
Parakeet runs natively on Windows using NVIDIA NeMo. One-time setup:

```bash
# 1. Install Visual C++ Build Tools (required for NeMo dependencies)
winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

# 2. Install NeMo toolkit
pip install nemo_toolkit[asr]

# 3. Select "Parakeet" in Settings → Transcription Engine
```

**Notes:**
- First load takes ~30-60 seconds (model download + initialization)
- Subsequent starts are faster (~30 seconds)
- Uses same server architecture as Whisper (localhost:9876)

---

## Architecture

**IMPORTANT: Desktop = HOST (runs server), Laptop = CLIENT (connects to server)**

- File systems are synced via Syncthing, but Python environments are separate
- Server changes require restart on DESKTOP (host)
- Client-only changes (UI, hotkeys) require restart on the machine using them
- `pip install` must be run separately on each machine

```
    DESKTOP / HOST (RTX 3070)                   LAPTOP / CLIENT (no GPU)
    ═════════════════════════                   ════════════════════════

         SYSTEM TRAY                              SYSTEM TRAY
    ┌─────────────────┐                      ┌─────────────────┐
    │ Koe             │                      │ Koe             │
    │ Koe Server ◄────┼────Tailscale────────►│   (remote)      │
    │ Scribe          │    100.78.59.64      │ Scribe          │
    └─────────────────┘                      └─────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Koe Server (:9876)                          │
│                                                                 │
│  ENGINE: Whisper OR Parakeet (both native Windows)             │
│  ─────────────────────────────────────────────                 │
│  Whisper:   All languages, large-v3 (~3GB VRAM)                │
│  Parakeet:  English only, ~50x faster (~2GB VRAM)              │
│                                                                 │
│  + Pyannote diarization (~0.5-1GB VRAM, loaded async)          │
│  + Speaker embeddings (Bryce, Calum, Sash, etc.)               │
│  + Serves both desktop (localhost) and laptop (Tailscale)      │
└─────────────────────────────────────────────────────────────────┘
```

### Engine Comparison

| Feature | Whisper | Parakeet |
|---------|---------|----------|
| Speed | 1-2x realtime | 50-80x realtime (with local attention) |
| Languages | 99+ languages | English only |
| VRAM (model) | ~3GB (large-v3) | ~2GB |
| Diarization | Yes | Yes |
| Setup | Automatic | One-time NeMo install |
| Long audio | Handles well | Fully supported (local attention enabled by default) |

---

## Icon Design

Koe uses a custom "sound bars" icon in terminal green (#00ff88):
- Three vertical bars of varying heights (60%, 100%, 80%)
- Rounded corners for modern look
- Fills entire canvas edge-to-edge
- Multi-size ICO file (16, 24, 32, 48, 256px) for proper display at all scales
- Generated programmatically via `scripts/generate_icon.py`

**Design rationale:**
- Visually distinct from other apps (no generic Python/microphone icons)
- Matches terminal green theme used throughout UI
- Simple, recognizable shape works at small tray icon size

**Implementation:**
- Windows AppUserModelID set to 'Koe.Transcription.App' for consistent taskbar grouping
- Absolute paths used for icon loading (prevents issues with working directory)
- VBS launchers hide console windows for clean startup

---

## Folder Structure

### Default Output (in source folder)
By default, all output stays in the Koe source folder. Both folders are configurable in Settings → Output Folders.
```
C:\dev\koe\
├── Snippets/                   # Rolling hotkey snippets (configurable in Settings)
│   ├── snippet_1.md            # Newest
│   ├── snippet_2.md
│   ├── snippet_3.md
│   ├── snippet_4.md
│   └── snippet_5.md            # Oldest (auto-deleted when 6th created)
│
└── Meetings/                   # Root folder (configurable in Settings)
    ├── Transcripts/            # Meeting transcripts organized by category
    │   ├── Standups/           # Category folders (user-created via + button)
    │   │   ├── 26_01_20_Daily.md
    │   │   └── Daily_Standup.md        # Pre-meeting agenda (no date)
    │   ├── One-on-ones/
    │   │   ├── Calum/                  # Subcategory (nested folder)
    │   │   │   └── 26_01_15_Weekly.md
    │   │   └── Sash/
    │   │       └── 26_01_18_Sync.md
    │   └── Investors/
    │       ├── Sequoia/                # Subcategory for specific investor
    │       │   └── 26_01_10_Pitch.md
    │       └── a16z/
    │           └── 26_01_12_Intro.md
    └── Summaries/              # AI summaries (mirrors Transcripts structure)
        ├── Standups/
        │   └── 26_01_20_Daily.md
        ├── One-on-ones/
        │   ├── Calum/
        │   │   └── 26_01_15_Weekly.md
        │   └── Sash/
        │       └── 26_01_18_Sync.md
        └── Investors/
            ├── Sequoia/
            │   └── 26_01_10_Pitch.md
            └── a16z/
                └── 26_01_12_Intro.md
```

### Source Code
```
C:\dev\koe\
├── run.py                      # Entry point (loads .env, runs main)
├── config.yaml                 # User configuration
├── .env                        # HF_TOKEN + WHISPER_SERVER_URL + KOE_API_TOKEN + ANTHROPIC_API_KEY
├── requirements.txt            # Full dependencies (desktop)
├── requirements-remote.txt     # Light dependencies (laptop)
├── .summary_status/            # Temporary status files for AI summarization (auto-cleaned)
│   └── MeetingName_a1b2c3d4.json  # Progress tracker (deleted when complete)
├── .session_state.npz          # Diarization session state (survives server restarts, 1hr TTL)
├── .transcript_recovery.jsonl  # Incremental transcript backup (crash recovery, auto-deleted on save)
├── speaker_embeddings/         # Voice fingerprints
│   ├── Bryce.npy              # Your voice embedding
│   └── Calum.npy              # Enrolled speaker
├── logs/
│   ├── debug.log              # Koe hotkey debug log (rotates at 1MB)
│   ├── meeting_debug.log      # Scribe meeting debug log
│   ├── koe_errors.log         # Centralized error log
│   └── failed_audio_*.wav     # Failed transcription chunks (retried on meeting stop)
├── assets/
│   ├── koe-icon.ico           # App icon (multi-size, 16/24/32/48/256)
│   ├── koe-icon.png           # App icon PNG (256x256)
│   └── beep.wav               # Completion sound
│
├── Start Koe Desktop.lnk      # Shortcut: starts server + Koe
├── Start Koe Remote.lnk       # Shortcut: connects to remote server
│
├── scripts/                   # Launch scripts + utilities
│   ├── Start Koe Desktop.bat
│   ├── Start Koe Desktop.vbs  # VBS wrapper (hides console window)
│   ├── Start Koe Remote.bat
│   ├── Start Koe Remote.vbs
│   ├── Start Scribe Desktop.bat
│   ├── Start Scribe Desktop.vbs
│   ├── Start Scribe Remote.bat
│   ├── Start Scribe Remote.vbs
│   ├── Stop Koe.bat           # Nuclear option: kills all Python
│   ├── Stop Koe.vbs
│   ├── create_shortcuts.ps1   # Regenerate .lnk shortcuts
│   └── generate_icon.py       # Regenerate icon files
│
└── src/
    ├── main.py                 # Koe app (PyQt5)
    ├── transcription.py        # Transcription logic (local/server/API)
    ├── result_thread.py        # Recording thread with VAD
    ├── key_listener.py         # Hotkey detection
    ├── utils.py                # ConfigManager singleton
    ├── logger.py               # Centralized error logging (logs/koe_errors.log)
    │
    │   # Transcription Engines
    ├── engines/
    │   ├── __init__.py         # Engine exports
    │   ├── base.py             # Abstract base class for engines
    │   ├── factory.py          # Engine registry and creation
    │   ├── whisper_engine.py   # Whisper (faster-whisper) implementation
    │   └── parakeet_engine.py  # Parakeet (NVIDIA NeMo) implementation
    │
    │   # Server
    ├── server.py               # FastAPI transcription server (supports multiple engines)
    ├── server_tray.py          # Server with system tray icon
    ├── server_launcher.py      # Background server starter
    ├── transcription_client.py # HTTP client for server
    │
    │   # Scribe
    ├── meeting/
    │   ├── __init__.py
    │   ├── app.py              # Meeting transcription UI
    │   ├── capture.py          # Dual audio capture (mic + loopback)
    │   ├── processor.py        # Audio chunking with VAD
    │   ├── transcript.py       # Markdown transcript writer
    │   ├── diarization.py      # Speaker diarization + fingerprinting
    │   ├── enroll_speaker.py   # Speaker enrollment CLI (mic)
    │   ├── record_loopback.py  # Speaker enrollment (system audio)
    │   ├── summary_status.py   # AI summarization status tracking
    │   ├── summarizer.py       # Claude Sonnet 4.5 API client
    │   └── summarize_detached.py  # Detached summarization subprocess
    │
    └── ui/
        ├── base_window.py           # Base window class
        ├── main_window.py           # Start/Settings buttons
        ├── settings_window.py       # Configuration UI
        ├── theme.py                 # Centralized color theme (terminal green #00ff88)
        ├── initialization_window.py # Initialization splash (brief startup indicator)
        └── status_window.py         # Recording status (terminal style, draggable)
```

---

## Configuration

### Key Settings (config.yaml)
```yaml
profile:
  user_name: Bryce                   # Your name - labels mic audio in Scribe
  my_voice_embedding: Bryce          # Your enrolled voice fingerprint (required for snippet filtering)

meeting_options:
  root_folder: C:\Users\Galbraith\Desktop\Koe\Meetings  # Where meetings are saved

recording_options:
  activation_key: ctrl+shift+space   # Toggle hotkey: press to start, press again to stop
  recording_mode: press_to_toggle    # Always press-to-toggle (not configurable in UI)
  filter_snippets_to_my_voice: false # Filter hotkey snippets to only your voice (adds ~0.5-1s latency)

model_options:
  engine: whisper                # Transcription engine: whisper (default) or parakeet (~50x faster, English only)
  whisper:
    model: large-v3              # Whisper model (see Settings UI for options)
    device: auto                 # auto, cuda, or cpu
  parakeet:
    model: nvidia/parakeet-ctc-0.6b  # Parakeet model (CTC recommended, TDT has CUDA 12.8 issues)
    device: auto                 # auto, cuda, or cpu
  common:
    initial_prompt: "Bryce, Calum, Sash. Use proper punctuation."  # Names help ASR accuracy

post_processing:
  name_replacements:             # Fix common ASR misspellings
    Callum: Calum                # Wrong spelling -> Correct spelling
    Shritam: Sritam
  remove_fillers: true           # Remove "um", "uh", etc.
  ai_cleanup_enabled: true       # Use Claude to polish longer snippets
  ai_cleanup_threshold: 30       # Minimum seconds before AI cleanup applies

misc:
  noise_on_completion: true          # Play beep when transcription completes
  snippets_folder: null              # Where to save rolling snippets (default: Koe/Snippets)
```

### Settings UI
The Settings window (terminal-themed, green #00ff88) exposes these options:

| Section | Setting | Description |
|---------|---------|-------------|
| **Profile** | Your Name | Labels your mic audio in Scribe transcripts |
| **Output Folders** | Meetings Root Folder | Where meeting transcripts are saved |
| | Snippets Folder | Where rolling hotkey snippets are saved |
| **Transcription Engine** | Engine | Whisper (default, all languages) or Parakeet (~50x faster, English only) |
| | Model | Model to use (e.g., large-v3 for Whisper, parakeet-ctc-0.6b for Parakeet) |
| | Device | Auto (detect GPU), CUDA (NVIDIA GPU), or CPU (no GPU required) |
| **Enrolled Speakers** | Speaker list | Shows voice fingerprints, × button to remove |
| | My Voice | Dropdown to select which enrolled speaker is you |
| | Filter snippets to my voice | Only transcribe your voice in hotkey mode (adds ~0.5-1s latency) |
| **Recording** | Toggle Hotkey | Keyboard shortcut (default: ctrl+shift+space) |
| | Play sound on completion | Beep when transcription finishes |
| **AI Cleanup (Snippets)** | Enable AI cleanup | Use Claude Sonnet 4.5 to polish grammar/punctuation for longer snippets |
| | Minimum duration | Threshold in seconds before AI cleanup applies (default: 30) |
| **Transcription Prompt** | Initial Prompt | Hint for transcription (names, punctuation style) |

### Post-Processing (Config Only)
These settings are in `src/config.yaml` but not exposed in the Settings UI:

```yaml
post_processing:
  name_replacements:      # Fix ASR misspellings: {wrong: correct}
    Callum: Calum
    Strretum: Sritam      # Parakeet struggles with "Sritam"
  remove_fillers: true    # Strip "um", "uh", "ah" etc.
```

**Tip:** Add names to `initial_prompt` (in Settings) as hints, then use `name_replacements` as a safety net for when ASR still gets them wrong.

### AI Cleanup (Snippets)
For longer snippets, Claude Sonnet 4.5 can polish the transcription:
- Fixes grammar and punctuation
- Removes filler words missed by rule-based filtering
- Preserves meaning and speaking style

**Settings** (in Settings UI → AI Cleanup):
- `ai_cleanup_enabled`: Enable/disable the feature (default: true)
- `ai_cleanup_threshold`: Minimum snippet duration in seconds (default: 30)

**Requirements:**
- `ANTHROPIC_API_KEY` in `.env` file (same key used for meeting summaries)

### Model Auto-Download
Models are downloaded automatically on first use - no manual download required:
- **Whisper models**: Downloaded to `~/.cache/huggingface/hub/` (managed by faster-whisper)
- **Parakeet models**: Downloaded to `~/.cache/torch/NeMo/` (managed by NeMo toolkit)

First startup with a new model will show "Still loading model..." while downloading. Subsequent starts are fast (model loaded from cache).

### Manual Configuration (Without Settings UI)
All settings can be edited directly in `config.yaml`:
```bash
# Open in any text editor
notepad C:\dev\koe\config.yaml    # Windows
code C:\dev\koe\config.yaml       # VS Code
```

**Example: Switch to CPU mode (no GPU):**
```yaml
model_options:
  engine: whisper
  whisper:
    model: base          # Smaller model recommended for CPU
    device: cpu          # Force CPU mode
```

**Example: Use a different Whisper model:**
```yaml
model_options:
  engine: whisper
  whisper:
    model: medium.en     # English-only medium model
    device: auto         # Let it detect GPU
```

After editing, restart Koe (Exit via tray icon, then launch again).

---

## Server Details

### Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (returns `{"status": "ok"}`) |
| `/status` | GET | Server status (model, device, ready, busy, active_requests, diarization_available, supports_long_audio) |
| `/transcribe` | POST | Transcribe audio (base64-encoded int16 PCM) |
| `/transcribe_meeting` | POST | Transcribe with speaker diarization (for Scribe) |
| `/diarization/reset` | POST | Reset diarization session (call at start of new meeting) |
| `/speakers` | GET | List enrolled speakers available for matching |
| `/shutdown` | POST | Gracefully shutdown the server |

### Busy Tracking (Multi-Instance Prevention)
The server tracks active transcription requests to prevent data loss during shutdown:
- `busy: true` - Server is actively processing transcription(s)
- `active_requests: N` - Number of in-flight transcription requests

**Launcher behavior:**
- `stop` - Waits up to 2 minutes for active requests to complete before stopping
- `stop --force` - Force stop even if busy (not recommended, may lose data)
- `stop --no-wait` - Don't wait for idle, abort if busy
- `restart` - Waits for idle, stops, then starts a new server

### API Token Authentication (Optional)
For secure remote access over Tailscale, you can enable API token authentication:

1. Generate a random token (e.g., `openssl rand -hex 32`)
2. Add to `.env` on **both desktop and laptop**: `KOE_API_TOKEN=<your-token>`
3. Restart Koe on both machines

**Behavior:**
- If `KOE_API_TOKEN` is not set, no authentication required (backward compatible)
- If set, all requests must include `X-API-Token` header matching the token
- Public endpoints (`/health`, `/status`, `/docs`) don't require authentication
- Invalid/missing token returns HTTP 401 with message "Invalid or missing API token"

### Security Limits
- **Audio size limit:** 50MB max (~25 minutes at 16kHz mono int16). Returns HTTP 413 if exceeded.
- **Error messages:** Internal errors return generic "Internal transcription error" (details logged server-side only)
- **File writes:** Config and transcript saves use atomic write pattern (write to .tmp, then rename)
- **Meeting names:** Sanitized to remove Windows reserved characters and names before use in filenames

### CUDA Memory Management
The server automatically clears CUDA cache to prevent VRAM fragmentation:
- **Threshold:** After 10 minutes of cumulative audio transcription
- **Action:** Calls `torch.cuda.empty_cache()` to release unused GPU memory
- **Impact:** ~100-200ms overhead on next transcription (negligible)
- **Why:** Prevents server freezes after many long transcriptions

### Long Audio Protection
For recordings >60 seconds, the client pre-saves audio before sending to server:
- **Pre-save location:** `logs/pending_audio_<timestamp>.wav`
- **On success:** Pre-save file is deleted
- **On failure/crash:** File remains for manual recovery via `python transcribe_file.py`
- **Why:** If process is killed mid-request, audio isn't lost

### Request Format (POST /transcribe)
```json
{
  "audio_base64": "<base64-encoded int16 PCM>",
  "sample_rate": 16000,
  "language": null,
  "initial_prompt": null,
  "vad_filter": true
}
```

### Response Format
```json
{
  "text": "Transcribed text here.",
  "duration_seconds": 5.2
}
```

### Remote Access (Tailscale)
The server binds to `0.0.0.0:9876`, so it's accessible from:
- Local: `http://localhost:9876`
- Remote: `http://100.78.59.64:9876` (Bryce's Tailscale IP)

---

## Remote/Laptop Setup

### Overview
The same codebase works for both desktop (with GPU) and laptop (no GPU). The laptop connects to the desktop's Koe server over Tailscale.

### Setup (Laptop)
```bash
# 1. Copy the koe folder to laptop
# 2. Install light dependencies (no torch/pyannote needed)
pip install -r requirements-remote.txt

# 3. Edit .env - change WHISPER_SERVER_URL to desktop's Tailscale IP
WHISPER_SERVER_URL=http://100.78.59.64:9876

# 4. Optional: Add API token for authentication (must match desktop's .env)
KOE_API_TOKEN=your-secret-token-here

# 5. Run
Start Koe Remote shortcut          # For hotkey transcription
Start Scribe Remote shortcut      # For Scribe
```

### How It Works
1. Laptop records audio locally (mic + system audio for meetings)
2. Audio sent to desktop server over Tailscale
3. Desktop runs Whisper transcription + pyannote diarization on GPU
4. Results sent back to laptop
5. Text copied to clipboard / displayed in UI

### Dependencies
**Desktop (requirements.txt):** ~3GB - includes torch, faster-whisper, pyannote
**Laptop (requirements-remote.txt):** ~50MB - audio capture + UI only

---

## Scribe Details

### Audio Capture
- **Microphone**: Default input device → labeled with your name from Settings
- **System Audio**: WASAPI loopback → diarized and matched to enrolled speakers
- **Sample Rate**: 16000 Hz (mic), native rate resampled (loopback)
- **Chunk Duration**: 30-60 seconds (VAD-based, chunks on natural pauses)

### Audio Preprocessing (Loopback)
System audio undergoes high-quality preprocessing before transcription:

1. **Stereo→Mono Conversion**: Energy-preserving sum (divide by √channels, not mean)
2. **Resampling**: scipy.signal.resample_poly with anti-aliasing filter (48kHz→16kHz)
3. **Normalization**: Boost to target RMS ~3000 (~-20dB) for optimal Whisper input

**Why this matters:**
- WASAPI loopback captures at device native rate (usually 48kHz stereo)
- Naive linear interpolation causes aliasing artifacts (metallic/robotic sound)
- System audio is often quiet (~6-9% RMS) - Whisper performs poorly on quiet audio
- Proper preprocessing dramatically improves transcription accuracy

### Speaker Diarization
Uses pyannote-audio models running on GPU (desktop) or server-side (laptop):
- **pyannote/speaker-diarization-3.1** - Identifies who's speaking when
- **pyannote/wespeaker-voxceleb-resnet34-LM** - Voice embeddings for fingerprinting

Flow:
1. Mic audio → diarization (for timing consistency + embedding extraction) → force-labeled as your name
2. Loopback audio → diarization → matched to enrolled speakers or "Speaker 1", "Speaker 2"

**Why diarize mic audio?**
- **Timing consistency**: Both streams go through same pipeline, preventing ordering issues
- **Embedding extraction**: Your voice embedding is extracted and updated (adaptive learning)
- **Self-enrollment**: You appear in post-meeting dialog and can update your embedding
- **Force-labeling**: Unlike loopback, mic is always labeled as `user_name` from config (not identified)

**Matching logic (loopback only):**
1. Extract wespeaker embedding for each detected speaker in chunk
2. Compare to enrolled speakers (Bryce, Calum, Sash) - threshold 0.35
3. If no match, compare to session speakers seen in earlier chunks
4. If still no match AND under max limit (8), create new "Speaker N"
5. If at limit, force-merge with closest existing session speaker
6. Session embeddings update with running average for better cross-chunk matching

**Speaker limit enforcement:**
- Hardcoded max of 8 speakers (sufficient for most meetings)
- Post-processing enforces the limit by merging similar speakers
- When limit reached, new detections merge with closest session speaker by embedding similarity
- Prevents speaker fragmentation (e.g., one person split into Speaker 1, 2, 3)

**Auto-merge on enrollment:**
- When enrolling a speaker, similar unknown speakers are auto-merged
- Uses same 0.35 similarity threshold as matching
- Transcript is automatically rewritten with correct names

**Adaptive learning:**
- **Loopback speakers**: When similarity >0.6, embedding is updated (95% old + 5% new)
- **Mic audio (user)**: Always updates user's embedding (90% old + 10% new) since we're certain it's them
- Updated embeddings saved to disk at start of next meeting
- Over time, fingerprints adapt to voice variations (different mics, time of day, etc.)

**Session persistence (crash recovery):**
- Session state (unknown speaker embeddings) saved to `.session_state.npz` when new speakers are detected
- On server restart, session state is auto-restored if file is <1 hour old
- Ensures enrollment dialog still works even if server crashes mid-meeting
- Session file deleted when new meeting starts (via `reset_session()`)

### Meeting Name & Category
Meeting name is **required**. Category selection is required if categories exist.

**File naming:**
- Transcripts: `YY_MM_DD_MeetingName.md` (e.g., `26_01_20_Daily.md`)
- Pre-meeting agendas: `MeetingName.md` (no date prefix)

**Category/Folder structure:**
1. Set root folder in Settings → Output Folders
2. Create categories via "+" button next to Category dropdown
3. Create subfolders via "+" button next to Folder dropdown
4. Example: `Investors` → `Sequoia` → transcript saved to `Transcripts/Investors/Sequoia/`

**Pre-meeting workflow:**
1. Open Scribe, enter name, select category/folder
2. Write agenda in the AGENDA section
3. Click "Save" (top right) → saves as `MeetingName.md` (no date prefix)
4. Later: Click "Open" → select agenda file (only shows files without date prefix)
5. Click "● REC" → take notes during meeting
6. Stop → saves as `YY_MM_DD_MeetingName.md`, deletes original agenda file

**Note-taking during meetings:**
- Three separate text areas with fixed headers: AGENDA, NOTES, ACTION ITEMS
- Headers are immovable labels (not editable markdown)
- Transcription happens in background - no live transcript display
- User focuses on taking notes, transcription saved automatically
- Notes saved at top of transcript file with proper `## Heading` markdown, transcript below

**Recording indicator:**
- Smooth pulsating red circle (●) in top-right corner during recording
- Fades between 30% and 100% opacity every 50ms (smooth, not distracting)
- No text, no box - just a clean visual indicator
- Impossible to forget you're recording

**UI Changes During Recording:**
- Form fields (NAME, CATEGORY, FOLDER) completely hidden (not just disabled)
- Open/Save buttons hidden during recording
- Cleaner, less cluttered interface while recording
- Focus on note-taking in AGENDA, NOTES, ACTION ITEMS sections
- Form fields reappear after recording stops

**Stop/Summarization UI:**
- Click "■ STOP" → window collapses to minimal UI (status, time, Exit only)
- Text editors disabled (data already sent to LLM)
- All buttons hidden except Exit
- Status shows: "Stopping..." → "Processing final audio..." → "Saving transcript..." → "Generating summary..."
- After ~30-60s: Clickable green summary link appears
- Click link → opens summary in VS Code
- UI automatically restores to full view ready for next meeting

**UI Theme:**
- Terminal-style dark theme with green accents (`#00ff88`)
- Monospace font (Cascadia Code/Consolas)
- Status messages: `> Ready`, `> Recording...`, `> Starting...`, `> Analyzing transcript...`, `> Generating summary...` (22px, prominent)
  - Clean and static - no distracting "Transcribing chunk #X" spam
  - User-friendly messages only (no technical API jargon)
- Server status row: `// Whisper Large v3 (CUDA) • Diarization: Local` (13px, subtle) with Open/Save buttons on right
  - Shows model name, device (CUDA/CPU), and diarization mode (Local/Server/Off/Loading)
  - Open/Save are subtle bordered buttons for pre-meeting agenda workflow
- Initialization window: Small popup showing "> Initializing_" with blinking cursor, minimum 1.5s display time
- Koe status popup: Blinking red indicator during recording, draggable
- Window size: Opens at 900x700, minimum 700x550 (works well on 14" high-DPI laptops)
- Form layout: Compact 2-row design (NAME on row 1, CATEGORY + FOLDER on row 2)
- Buttons: Compact, right-aligned
  - "● REC" (red outline, semi-transparent fill) → "■ STOP" when recording
  - "Exit" button
- Context menus: Dark terminal theme with green text (#00ff88), not Windows default white
- Summary window: 450x85 with centered status text, minimum height prevents text jumping
- File opening: Clicks on summary links open in VS Code (tries multiple locations, falls back to default handler)

**Tray Menu Structure:**
Dark terminal-themed menu (matches app aesthetic, not Windows default white).
```
Right-click Koe tray icon:
├── Start Scribe          → Launch meeting transcription
├── Settings              → Open settings window
├── ─────────────────
└── Exit                  → Close Koe
```

### Speaker Enrollment
Enroll speakers so they appear by name instead of "Speaker 1":

**Via Post-Meeting Dialog (Recommended):**
1. After meeting ends, click orange "Enroll Speakers" button in summary window
2. Dialog shows unknown speakers with sample transcriptions for identification
3. Type name and click "Enroll" to save the speaker
4. Transcript auto-rewrites with correct name (all "Speaker 1" entries updated)
5. Similar speakers auto-merge (if "Speaker 3" ≈ "Speaker 1", both become enrolled name)
6. Summary generation waits until enrollment complete (uses correct names)
7. Embeddings from meetings are running averages - higher quality than quick recordings

**Via CLI (For debugging):**
```bash
# Record from mic (10 seconds)
python -m src.meeting.enroll_speaker "Calum"

# Record from system audio (YouTube video of someone)
python -m src.meeting.record_loopback "Sash" --duration 30 --enroll

# List enrolled speakers
python -m src.meeting.enroll_speaker --list dummy

# Remove a speaker (or use Settings → Enrolled Speakers → ×)
python -m src.meeting.enroll_speaker "Calum" --remove
```

Embeddings saved to `speaker_embeddings/Name.npy`

**Currently enrolled:** Bryce, Calum, Sash

### AI Summarization Status Files

**Location:** `C:\dev\koe\.summary_status\` (centralized, hidden folder)

These are **temporary progress tracking files** for AI summarization. They communicate between the detached subprocess and the UI.

**Naming:** `MeetingName_<hash>.json` (hash ensures uniqueness)

**Lifecycle:**
1. **Created:** When you click "Stop Recording"
2. **Updated:** Every ~2 seconds by the detached subprocess
3. **Polled:** By the summary status window to show progress
4. **Cleaned up:** Automatically deleted when summarization completes/fails

**Contents (JSON):**
```json
{
  "status": "in_progress",  // or "complete", "failed"
  "stage": "Analyzing transcript...",
  "progress_percent": 45,
  "summary_path": null,  // Set when complete
  "transcript_path": "C:\\...\\26_01_21_Meeting.md",
  "pid": 12345,
  "timestamp": 1737447890.123
}
```

**Why centralized?**
- Keeps meeting folders clean (no temp files mixed with transcripts)
- Easy to find/delete stale files if process crashes
- Automatic cleanup after completion

**Safe to delete** if you see old files (>10 minutes) - they're stale from crashed processes.

### Transcript Format
```markdown
# Weekly Sync

**Date**: 2024-01-15 14:30
**Duration**: 25 minutes
**Participants**: Bryce, Calum, Speaker 1

---

## Agenda
- Review Q1 roadmap
- Discuss hiring timeline

## Notes
Team agreed to prioritize mobile over web for Q1.
Engineering capacity limited due to onboarding.

## Action Items
- [ ] Bryce: Send updated timeline by Friday
- [ ] Calum: Schedule interviews for next week

---

## Full Transcript

**[00:00] Bryce**: Hello everyone, let's get started.

**[00:05] Calum**: Sounds good, I have the updates ready.

**[00:32] Speaker 1**: I'll share my screen.
```

### Summary Format
AI-generated summaries use this markdown hierarchy for clear visual structure:
```markdown
# Meeting Name - 22 Jan 2026
Duration: 69 minutes | Participants: Bryce, Calum, Sash

---

## Summary
2-4 paragraphs capturing the meeting's purpose, key outcomes, and overall context.

---

## Key Decisions
- Decision one
- Decision two

---

## Topics Discussed

##### Topic Name
Brief description of what was discussed.

##### Another Topic
Description here.

---

## Action Items

##### Bryce
- Task one
- Task two

##### Sash
- Task three

##### Bryce & Sash
- Shared task

---

## Open Questions
- Unresolved question one
- Question requiring follow-up
```

**Hierarchy:**
- `#` H1 - Meeting title with date
- `##` H2 - Section headings (Summary, Key Decisions, Topics Discussed, Action Items, Open Questions)
- `#####` H5 - Subtopics and action item owners (provides strong visual contrast from H2)
- `---` - Horizontal rules before each H2 section for clear separation

---

## Transcription Flow

### Koe (Hotkey Mode)
```
1. User presses Ctrl+Shift+Space
2. Status popup appears (draggable)
3. Recording starts (WebRTC VAD detects speech)
4. User speaks
5. Silence detected (900ms) → recording stops automatically
   OR user presses hotkey again → manual stop
6. Status shows "> Transcribing_"
7. Audio sent to server (or local model if no server)
8. OPTIONAL: If "Filter snippets to my voice" enabled:
   - Server runs diarization on audio
   - Filters to segments matching your enrolled voice
   - Only transcribes your voice (adds ~0.5-1s latency)
9. Transcription returned and post-processed (prompt leak removal, filler removal, punctuation, hallucination filtering)
10. Saved to rolling snippet files
11. Result copied to clipboard
12. Beep sound plays (if enabled)
13. Status window closes automatically

Note: Steps 10-12 ALWAYS complete after transcription finishes, even if you pressed
Escape or the hotkey during transcription. Once transcribing starts, the result will
be delivered to your clipboard.
```

**Voice Filtering (Optional):**
Enable in Settings → Enrolled Speakers → "Filter snippets to my voice"
- Requires: Your voice enrolled + "My Voice" dropdown set
- Use case: Noisy environments, other people talking nearby
- Tradeoff: Adds ~0.5-1 second latency to transcription
- Result: Only your voice transcribed, background voices ignored

### Scribe
```
1. Window appears immediately (900x700, terminal theme)
2. Three note sections with fixed headers: AGENDA, NOTES, ACTION ITEMS
3. User enters meeting name (required)
4. User selects category + folder (optional: click "+" to create new)
5. Optional: Click "Open" (top right) to load pre-saved agenda file
6. User clicks "● REC"
7. Button changes to "■ STOP"
8. Open/Save/+ buttons hidden during recording
9. Status shows "> Recording..." (clean, no chunk spam)
10. Mic + loopback streams start
11. User takes notes in editors while transcription happens in background
12. Every 30-60 seconds:
    - Chunk sent to server
    - Transcription returned (with speaker labels)
    - Entry added to transcript (not shown in UI)
    - Status stays clean (no "Transcribing chunk #X" spam)
13. User clicks "■ STOP"
14. Status shows "> Stopping..." → "> Processing final audio..."
15. Final chunk processed
16. Failed chunks retried (any failed_audio_*.wav from last hour)
17. Status shows "> Saving transcript..."
18. Markdown transcript saved: notes at top, transcript below
19. If opened from agenda file: original file deleted
20. Status shows "> Generating summary..."
21. Summary generated (~30-60s), clickable link appears
22. Clicking link opens summary in VS Code
```

---

## Dependencies

### Core
- `faster-whisper` - Optimized Whisper implementation
- `PyQt5` - Desktop UI framework
- `sounddevice` - Audio capture (Koe)
- `pyaudiowpatch` - WASAPI loopback (Scribe)
- `webrtcvad` - Voice activity detection
- `pynput` - Keyboard hotkey detection

### Server
- `fastapi` - HTTP API framework
- `uvicorn` - ASGI server
- `requests` - HTTP client

### AI Summarization
- `anthropic` - Claude API client (~$0.04 per meeting)

### Audio Processing
- `scipy` - High-quality resampling with anti-aliasing filters
- `numpy` - Audio array operations

### Parakeet Engine
- `nemo_toolkit[asr]` - NVIDIA NeMo for Parakeet ASR
- Requires Visual C++ Build Tools for compilation

### Other
- `pyperclip` - Clipboard access
- `pyyaml` - Configuration files

---

## Bugs Fixed

| Bug | Cause | Fix |
|-----|-------|-----|
| CUDA/Qt segfault | PyQt5 conflicts with CUDA initialization | Load model BEFORE QApplication |
| cuDNN DLL not found | Windows DLL search path | `os.add_dll_directory()` for nvidia packages |
| Ctrl+Shift+R triggers recording | Unknown keys defaulted to SPACE | Return `None` for unknown keys |
| Previous transcription bleeding | `condition_on_previous_text: true` | Set to `false` in config |
| Clipboard failures | pyperclip race conditions | Retries + Windows clip.exe fallback |
| GPU memory conflict | Two apps loading separate models | Shared server architecture |
| Hotkey mode unresponsive | Second hotkey press ignored in VAD mode | Handle all modes in `on_activation()` |
| Recording stop delay | `Event.wait()` blocked indefinitely | Added 100ms timeout to check flags |
| Speaker matching failed | Pipeline embeddings != wespeaker embeddings | Extract audio per speaker, use wespeaker model |
| Too many speakers detected | Cross-chunk tracking not working | Use same embedding model for enrollment + matching |
| Recording lost on double-press | Hotkey pressed twice quickly → recording too short → discarded | 1-second minimum recording time before stop allowed |
| Recording lost on window close | Status window X button cancelled transcription | Removed closeSignal → stop_result_thread connection |
| Slow Scribe startup | Pyannote/torch imports blocking at module level | Lazy import diarization in background thread |
| Mixed speaker labels (SPEAKER_00 vs Speaker 1) | Short segments failed embedding extraction, leaked raw pyannote labels | Fallback assigns consistent "Speaker N" label for unmapped speakers |
| Too many false speakers in 2-person meetings | Pyannote detected up to 6 speakers, cross-chunk matching failed | Lowered threshold to 0.35, running average for session embeddings, auto-merge on enrollment |
| Multiple Koe instances running | No single-instance check | Socket lock on localhost:19877 (Koe) and localhost:9878 (Scribe) |
| Console window appears on launch | Batch files run in visible window | VBScript wrappers run batch files with WindowStyle = 0 (hidden) |
| Taskbar icon shows Python logo | Windows uses python.exe icon instead of app icon | Set Windows AppUserModelID + absolute icon paths |
| Settings "Browse" button text cut off | Button too narrow for text | Changed "Browse..." to "Browse", set min width 80px |
| Ugly scrollbar in Settings | Default system scrollbar | Custom terminal-themed scrollbar with green accent |
| Excessive spacing above Settings header | Top margin too large | Reduced from 24px to 8px |
| Enrollment window text cut off | Window too narrow, input font too small | Increased window from 320x140 to 500x185, input font from 11pt to 12pt, min height 40px |
| Action items placeholder too specific | "task assigned to person" too prescriptive | Changed to "- [ ] todo item..." for generic todo list |
| Double countdown in enrollment window | Countdown shown in both status label and timer label | Clear timer label during countdown, only show in status |
| Enrolled speaker not showing in Settings | Settings didn't refresh after enrollment | Auto-refresh speakers list, dropdown, and filter checkbox on enrollment |
| [ESC] button doesn't stop transcription | Status window closed but thread continued in background | Connect statusSignal to stop thread on cancel action |
| Status window closes before showing "Transcribing..." | Window closed immediately after recording stopped | Added 300ms delay before closing on 'complete' status |
| Scribe feels slow and unresponsive | All operations blocking UI thread, no feedback during heavy work | Phase 1: Instant button state changes, progress messages, disable inputs during recording |
| Start recording lag (1-3s button freeze) | Server check + diarization reset blocked UI before button changed | Button now changes to "⏳ STARTING..." immediately with progress updates |
| Stop recording lag (5-15s button freeze) | Transcription + file I/O blocked UI before button changed | Button now changes to "⏳ STOPPING..." immediately with progress updates |
| Distracting buffer status updates | Status label updated every 500ms with buffer info | Removed `_check_processor()` timer, now shows static "> Recording..._" |
| Form inputs editable during recording | No input locking during active recording | All inputs (name, category, folder) now disabled during recording |
| Post-recording state confusing | Notes retained, unclear if ready for new meeting | Notes auto-cleared on save, shows "✓ Saved: filename" + "[Ready for new meeting]" |
| UI still freezes during start/stop (Phase 1 limitation) | Heavy operations (server check, transcription, file I/O) blocked main thread despite instant button feedback | Phase 2: True async - moved blocking operations to background threads with QTimer.singleShot() callbacks for Qt-safe UI updates |
| Entire "● REC" indicator flashes | Text and dot both blinked, visually distracting | Split into separate dot + text labels, only dot blinks (less annoying) |
| Can edit notes after stopping | Text editors remained enabled while data sent to LLM | Editors disabled immediately when stop clicked |
| Cluttered UI during stop/summarization | All buttons and inputs visible during processing | Window collapses to minimal UI (status, time, Exit only) during stop/summarization |
| "[X entries]" counter shown | Entry count displayed during/after recording | Removed counter entirely - not needed, user takes notes instead |
| Summary generation stuck/incomplete | Subprocess cleaned up status file before parent could read it, parent polled forever | Subprocess leaves status file, parent reads final status then cleans up - no more infinite loops |
| Status window appears broken (closes immediately, no beep) | GPU transcription too fast (<1s), 300ms delay insufficient to see completion | Increased delay to 1.2s, added "Complete!" message with checkmark for visual confirmation |
| Distracting "Transcribing chunk #X" status spam | Status updated every time a chunk was transcribed | Keep status as static "Recording..." - no chunk numbers |
| Technical API status messages | Status showed "Connecting to API..." and "Calling API..." during summarization | Replaced with user-friendly messages: "Analyzing transcript...", "Generating summary..." |
| Summary link opens in Notepad++ instead of VS Code | Windows default file association used | Try multiple VS Code locations first, fall back to os.startfile |
| White context menu on right-click | No QMenu styling defined | Added dark terminal-themed QMenu styles (tray menu + submenus) |
| Summary window text jumps/shifts | Status label height changed when text updated | Added minimum height to status label for consistent layout |
| Summary window text misaligned | Status text left-aligned, not centered properly | Wrapped status label in horizontal layout with stretch factors, centered alignment |
| Status window text misaligned | Text left-aligned instead of centered between dot and [ESC] | Centered status/timer labels with stretch factor, removed nested timer row |
| Clipboard copy interrupted by hotkey | Pressing hotkey during transcription started new recording, interrupting clipboard copy and "Complete!" display | Added `processing_result` flag to block new recordings until clipboard copy, beep, and status display finish |
| Transcription saved but not copied to clipboard | Pressing Escape or hotkey during transcription stopped thread early, before result signal emitted | Removed early-exit check after transcription - result always emitted once transcription completes |
| "Complete!" message unnecessary and distracting | Extra UI feedback when beep already indicates completion | Removed "Complete!" status message - window closes immediately after beep (sufficient feedback) |
| Config schema mismatch | config.yaml had fields not in config_schema.yaml | Added missing fields to schema with proper defaults |
| Wrong schema default for condition_on_previous_text | Set to true causing transcription bleeding | Changed default to false in schema |
| Relative path bugs | beep.wav and speaker_embeddings used relative paths | Fixed to use Path(__file__).parent pattern for absolute paths |
| No config validation | Invalid config values caused silent crashes | Added validation in ConfigManager.load_user_config() |
| No error logging | Errors only printed to console, lost in pythonw.exe | Created centralized logger.py writing to logs/koe_errors.log |
| Duplicate color constants | 8 files defined own color values | Created centralized theme.py with all colors |
| Scribe window too small | Minimum size 800x700 inadequate for note-taking | Increased to 1000x800 minimum |
| Poor loopback audio quality | Linear interpolation resampling (np.interp) caused aliasing artifacts | Replaced with scipy.signal.resample_poly (polyphase filter with anti-aliasing) |
| Loopback audio too quiet | RMS at 6-9% of nominal, Whisper expects louder audio | Added normalization to target RMS ~3000 (~-20dB) with gain limiting |
| Stereo-to-mono energy loss | Simple channel averaging lost 3dB energy | Changed to sum/√channels for energy-preserving conversion |
| max_speakers not enforced | Pyannote treats max_speakers as hint, not hard limit | Added post-processing to force-merge speakers when limit exceeded |
| Speaker 4 missing (gaps in numbering) | Session counter incremented even when speakers should merge | Force-merge with closest session speaker instead of creating new when at limit |
| Koe crashes after transcription | Unicode characters (checkmarks) in console_print couldn't be encoded by Windows console codepage | Replaced Unicode symbols with ASCII alternatives in log messages |
| max_speakers doesn't count enrolled speakers | Setting max to 2 allowed 4 speakers (Bryce, Calum + Speaker 1, Speaker 2) | Track enrolled speakers seen in session, count total when enforcing limit |
| Foreign language hallucinations in Scribe | Whisper auto-detected language, outputting Russian/Persian/Italian for unclear audio | Set language=en by default, updated initial_prompt to reinforce English-only |
| Phantom speakers in 2-person meetings | Diarization detected echoes/noise as separate speakers | Combined with max_speakers fix - now properly limits total speakers |
| Initial prompt leaking into transcription | Whisper hallucinated the initial_prompt text when audio was unclear/silent | Post-processing filters out prompt text; if segment becomes empty, entry is dropped |
| Status window stuck after cancelled recording | Early exit path (when stop() called during recording) returned without emitting resultSignal, leaving window stranded | Emit empty resultSignal on early exit so window closes properly |
| Scribe crashes when clicking Record | `self.speakers_layout` referenced in `_hide_form_fields()` but never defined, causing AttributeError | Removed undefined `speakers_layout` from layout list in `_hide_form_fields()` and `_show_form_fields()` |
| Enrollment button not showing after meeting | `QTimer.singleShot(0, lambda)` from background thread doesn't reliably schedule on main thread | Use PyQt signal (`enrollment_data_ready`) instead of QTimer for cross-thread UI updates |
| Status window disappears before showing "Transcribing" | `on_transcription_complete()` directly called `status_window.close()` instead of updating status | Call `status_window.updateStatus('complete')` so window shows "Complete!" for 2 seconds before closing |
| Transcript entries out of chronological order | Entries appended in order received, not sorted by timestamp; mic/loopback processed in parallel | Sort entries by timestamp in `generate_markdown()`, `get_recent_text()`, and `get_full_text()` |
| Enrolled speaker not recognized in next meeting | `enroll_speaker_from_session()` looked up embedding from `_session_speakers` which may have been cleared | Added `enroll_speaker_with_embedding()` that takes embedding directly; dialog passes stored embedding instead of relying on diarizer session state |
| Phantom speakers in enrollment dialog | Diarization detected "speakers" with no actual transcript text (noise/echoes) | Filter out speakers with no transcript entries before showing enrollment dialog |
| App crashes after transcription completes | `statusSignal.emit('complete')` sent to wrong handler (on_status_window_action instead of updateStatus) | Call `status_window.updateStatus('complete')` directly instead of emitting on statusSignal |
| Random crash during recording | Audio callback thread and main loop both access `audio_buffer` without synchronization | Added `Lock` to protect `audio_buffer` access between threads |
| Beep sometimes not playing | `audioplayer` library unreliable on Windows | Use `winsound` module on Windows for more reliable audio playback |
| Crash on fast hotkey double-press | `isRunning()` returns False while thread is still spinning up, causing new thread to overwrite the old one | Added guard checking `recording_start_time` to prevent duplicate thread creation within 0.5s |
| Empty Scribe transcript with no errors | Server down but transcription failures silently ignored (no logging) | Added error logging for failed transcriptions in meeting debug log |
| Freeze during server transcription | No debug logging in transcription_client.py made it impossible to diagnose where HTTP requests hung | Added granular debug logging to transcription_client.py with `[client]` prefix |
| Slow failure on server unreachable | Single 60s timeout meant waiting full minute if server down | Added separate connect timeout (5s) vs read timeout (60s) |
| Slow Scribe startup (10-30s) | Scribe loaded local diarization even when server already had it | Check server diarization first, skip local load if available |
| Too many speakers detected (5 in 3-person meeting) | Fallback path created new speakers when embedding extraction failed, without trying to match existing session speakers | Fallback now reuses last active or first session speaker instead of creating new ones |
| Same person split into multiple speakers | Session similarity threshold (0.35) too strict - same person's embeddings varied 0.1-0.9 across chunks | Lowered threshold to 0.25 for better intra-session matching |
| Speaker fragmentation not consolidated after meeting | No mechanism to merge obviously-similar speakers before enrollment dialog | Added `consolidate_session_speakers()` that merges speakers with >0.40 similarity after meeting ends |
| Enrollment dialog shows too little context | Only showed 150 chars of samples, making speaker identification difficult | Now shows up to 5 full transcript entries per speaker with timestamps |
| Could accidentally overwrite enrolled speaker | No validation when typing name that's already enrolled | Added validation - shows warning and prevents overwrite if name exists |
| Long recordings crash/freeze (112+ seconds) | Fixed 60s HTTP timeout too short for long audio transcription | Dynamic timeout: 30s base + 3x audio duration (capped at 900s) |
| Parakeet extremely slow on long audio | Audio >60s caused O(n²) attention slowdown (180s took 160s = 1.1x realtime) | Enabled local attention + auto-chunking. 180s now takes 2.6s = 69x realtime |
| Lost audio on transcription failure | Timeout/error discarded audio forever, no recovery possible | Failed audio saved to `logs/failed_audio_<reason>_<timestamp>.wav` |
| Manual merge not possible in enrollment | No way to assign multiple unknown speakers to same person | Typing same name for different speakers merges them (rewrites transcript) |
| Crash when cancelling recording | `wait()` in stop() blocked main thread while worker tried to emit signals needing main thread | Removed `wait()` - thread finishes naturally without blocking |
| Crash when clicking ESC during snippet | `on_transcription_complete()` called `updateStatus('complete')` on already-closed status window | Added `isVisible()` checks before updating status window |
| Parakeet crashes with CUDA 12.8 | TDT model's CUDA graph decoder expects 6 return values from `cu_call()`, but CUDA 12.8 returns 5 | Switched default to CTC model which doesn't use CUDA graphs |
| Crash when clicking ESC during recording | PyQt signal emitted to closed window during signal delivery caused crash | Replaced signal with direct callback; ESC now only works during recording (not transcribing) |
| Server not stopping when Koe exits | `stop_server()` didn't actually stop the server, just printed a message | Implemented proper shutdown via POST to `/shutdown` endpoint; Koe's `exit_app()` now calls `stop_server()` |
| Silent fallback to broken local mode | Parakeet configured but server not ready → fell back to local (which can't work on Windows) → empty transcription | `transcribe()` now raises RuntimeError if engine=parakeet and server unavailable; shows clear error message |
| Diarization shows "Off" in Scribe | Server didn't load .env file, so HF_TOKEN unavailable for pyannote authentication | Added `load_dotenv()` to server_tray.py and server.py |
| Diarization not loading at startup | Diarizer only lazy-loaded when first meeting transcription requested, so `/status` reported `diarization_available: false` | Added background thread in server lifespan to proactively load diarizer |
| Long audio stuck/hung on old server | Server started before local attention code was added would hang on >60s audio | Added `supports_long_audio` to server status; client checks before transcribing >60s audio and refuses with clear error + saves audio |
| Session state lost on server restart | Server restart during meeting wiped diarizer session, losing unknown speaker embeddings | Session state persisted to `.session_state.npz`, auto-restored on server restart (1 hour TTL) |
| No enrollment dialog after server crash | Server restart cleared session speakers, `get_unenrolled_speakers()` returned empty | Session persistence ensures speaker embeddings survive server restarts |
| Failed chunks never recovered | Transcription timeouts saved audio but never retried | `_retry_failed_chunks()` called on meeting stop, retries failed audio files (max 60 min age) |
| Scribe crash loses all transcript entries | Entries stored only in memory until meeting stop, crash = total loss | Incremental save to `.transcript_recovery.jsonl`, recovery dialog on startup |
| Empty transcription loses audio | Local engine silent failure (empty result) didn't backup audio | Audio saved to `failed_audio_empty_result_*.wav` if transcription >2s returns empty |
| Server killed during active transcription | No busy tracking, `taskkill` during transcription lost data | Server tracks `busy`/`active_requests`, launcher waits for idle before stopping |
| Windows console crash on config validation | Unicode `⚠` emoji in print statement caused UnicodeEncodeError | Changed to ASCII `[!]` in utils.py:119 |
| Memory exhaustion via large audio payload | No size limit on `/transcribe` and `/transcribe_meeting` endpoints | Added 50MB limit with HTTP 413 response |
| Race condition in server check | `_server_client` and `_server_mode` globals accessed without lock | Added `_server_lock` in transcription.py |
| Config/transcript corruption on crash | Direct file writes could corrupt if interrupted | Implemented atomic write pattern (write to .tmp, then rename) |
| Meeting name could create invalid filename | Windows reserved names (CON, NUL) and special chars not filtered | Added `sanitize_filename()` helper with comprehensive validation |
| Internal error details leaked to clients | HTTPException included full exception message | Log traceback server-side, return generic "Internal transcription error" |
| Remote transcription fails on network glitch | Single request with no retry, lost audio on transient failures | Added retry logic with exponential backoff (1s, 2s), saves audio only after all retries exhausted |
| Malformed WHISPER_SERVER_URL causes cryptic error | No URL validation, requests library threw confusing errors | Added `_validate_server_url()` that checks scheme and hostname at startup |
| Remote connection failure has unhelpful message | "Server not running" shown for all failures, no Tailscale hint | Added remote-specific hint: "Check desktop is running and Tailscale is connected" |
| Scribe crashes on laptop without scipy | scipy used for audio resampling but not in requirements-remote.txt | Added scipy to requirements-remote.txt |
| Unenrolled speakers timeout on slow networks | 5s timeout too short for high-latency Tailscale connections | Increased `/diarization/unenrolled` timeout to 15s |
| Remote server accessible without authentication | No authentication on API endpoints, anyone on Tailscale could access | Added optional `KOE_API_TOKEN` middleware with `X-API-Token` header verification |
| Long audio lost when process killed mid-request | Audio only saved on exception catch; killing process bypassed all error handling | Pre-save long audio (>60s) to `logs/pending_audio_*.wav` before sending to server; delete on success, rename to `failed_audio_*` on failure |
| Server freezes after many long transcriptions | VRAM fragmentation/accumulation after multiple transcriptions exhausted GPU memory | Auto-clear CUDA cache after 10 minutes cumulative audio (`torch.cuda.empty_cache()`) |
| Engine change in Settings doesn't take effect | `restart_app()` didn't stop server, so old engine kept running after Settings save | Added `stop_server()` call to `restart_app()` so new engine starts on restart |
| Scribe request storm freezes entire PC | Chunks dispatched in unbounded parallel threads; timeouts triggered retries, compounding into 10+ concurrent requests that saturated GPU | Serialize chunk transcription with semaphore (1 at a time) + circuit breaker after 5 consecutive failures saves audio instead of hammering server |
| `supports_long_audio` false positive on busy server | `/status` check had 2s timeout; when server was busy, timeout caused `supports_long_audio()` to return False, rejecting valid long audio | Cache last known value, increased timeout to 5s, return cached value on timeout instead of False |
| Server launcher always starts Whisper regardless of config | `CONFIG_PATH` in `server_launcher.py` was `SCRIPT_DIR.parent / "config.yaml"` (project root) instead of `SCRIPT_DIR / "config.yaml"` (src/), so config file was never found and engine defaulted to Whisper | Fixed path to `SCRIPT_DIR / "config.yaml"` |
| Diarization exceptions crash chunk processor silently | Diarization method calls in `_process_chunk()` had no try-except; exceptions killed the daemon thread without updating circuit breaker or saving audio | Wrapped entire `_process_chunk()` body in try-except; exceptions now log error, save audio as failed, and feed into circuit breaker |

---

## Development Phases

### Phase 1: Core Functionality ✅
- [x] Koe hotkey transcription
- [x] Scribe with dual audio capture
- [x] Shared Whisper server architecture
- [x] Live transcription display

### Phase 2: Speaker Diarization ✅
- [x] pyannote-audio integration (GPU)
- [x] Voice fingerprinting with wespeaker embeddings
- [x] Speaker enrollment from mic or system audio
- [x] Cross-chunk speaker tracking (same person across chunks)
- [x] Named speaker identification (Bryce, Calum enrolled)

### Phase 3: Meeting Enhancements (In Progress)
See `SCRIBE_PERFORMANCE_PLAN.md` for detailed performance implementation plan.
- [x] Custom meeting naming (required before recording)
- [x] Category folder organization (Standups, One-on-ones, Investors, etc.)
- [x] Configurable output folders (Settings → Output Folders for meetings and snippets)
- [x] Async diarization loading (instant window startup)
- [x] Auto-merge similar speakers on enrollment (reduces false speaker splits)
- [x] Improved cross-chunk speaker tracking (running average embeddings)
- [x] Adaptive voice fingerprinting (embeddings improve over time with high-confidence matches)
- [x] Terminal-themed Settings UI (matches Koe popup and Scribe window)
- [x] Post-meeting speaker enrollment (auto transcript rewriting, deferred summarization)
- [x] Custom app icon (sound bars, terminal green #00ff88)
- [x] VBS launchers for hidden console windows
- [x] Single-instance protection (prevents duplicate processes)
- [x] **Performance Phase 1: Instant Feedback (completed 2026-01-21)**
  - [x] Start/stop buttons change state immediately with progress messages
  - [x] Form inputs disabled during recording
  - [x] Notes auto-cleared after save for next meeting
  - [x] Removed distracting buffer status updates
- [x] **Performance Phase 2: True Async Operations (completed 2026-01-21)**
  - [x] Refactored start_recording() to spawn background thread
  - [x] Created _start_recording_async() for heavy work (server check, diarization reset)
  - [x] Created _finalize_start_recording() for Qt operations on main thread
  - [x] Refactored stop_recording() to spawn background thread
  - [x] Created _stop_recording_async() for heavy work (transcription, file I/O)
  - [x] Created _finalize_stop_recording() for Qt operations on main thread
  - [x] Added helper methods (_enable_inputs, _show_server_error, _reset_start_button, _reset_stop_button)
  - [x] UI now truly non-blocking - can interact with window during start/stop operations
- [x] **Performance Phase 3: Visual Polish (completed 2026-01-21)**
  - [x] Blinking "● REC" indicator in header (bright/dim red, 500ms toggle) - impossible to forget you're recording
  - [x] Compact button design: "● REC" / "■ STOP" with dark red outline styling
  - [x] Open/Save moved to subtle buttons in server status row
  - [ ] Keyboard shortcuts (declined by user)
- [x] **AI Summarization (completed 2026-01-21)**
  - [x] Claude Sonnet 4.5 integration (~$0.04/meeting)
  - [x] Detached subprocess (window can close, summary continues)
  - [x] Live progress updates with clickable VS Code link
  - [x] Mirrored Summaries/ folder structure
  - [x] Anti-hallucination prompt with strict guidelines
  - [x] Retry logic (3 attempts with exponential backoff)
  - [x] Error logging to summarization_errors.log
  - [x] See SUMMARIZATION_IMPLEMENTATION.md for full details
- [x] **Audio Quality Improvements (completed 2026-01-21)**
  - [x] High-quality resampling with scipy.signal.resample_poly (anti-aliasing filter)
  - [x] Audio normalization to target RMS ~3000 for optimal Whisper input
  - [x] Energy-preserving stereo-to-mono conversion (sum/√channels)
  - [x] Strict max_speakers enforcement via post-processing (force-merge when exceeded)
- [x] **Post-Processing Improvements (completed 2026-01-22)**
  - [x] Initial prompt leak filtering (removes hallucinated prompt text from transcriptions)
  - [x] Applied to both Koe hotkey mode and Scribe meetings
  - [x] Filters each line/sentence of the prompt separately
  - [x] Empty segments after filtering are dropped entirely
- [x] **Post-Meeting Speaker Enrollment (completed 2026-01-22)**
  - [x] "Enroll Speakers" button in summary window (only if unknown speakers exist)
  - [x] Shows unknown speakers (Speaker 1, etc.) with sample transcriptions + "Enroll" button
  - [x] Transcript auto-rewrites when speaker enrolled (all "Speaker 1" → enrolled name)
  - [x] Auto-merges similar speakers by embedding similarity
  - [x] Summary generation waits until enrollment dialog closes (uses correct names)
  - [x] Uses high-quality running-average embeddings from entire meeting
  - [x] DiarizationManager methods: get_unenrolled_session_speakers(), enroll_speaker_from_session()
- [x] **Removed manual enrollment from tray menu (2026-01-22)** - post-meeting enrollment preferred
- [x] **Error Recovery & Debug Improvements (completed 2026-01-23)**
  - [x] Debug log rotation (auto-rotates at 1MB, keeps 1 backup)
  - [x] Error message shown in status window for 3 seconds before closing
  - [x] Tray notification on transcription failure
  - [x] Failed audio saved to `logs/failed_audio_<timestamp>.wav`
- [x] **Crash Recovery & Resilience (completed 2026-02-05)**
  - [x] Session state persistence (`.session_state.npz`) - survives server restarts
  - [x] Session state auto-restored on server startup (1 hour TTL)
  - [x] Failed chunk retry on meeting stop (retries `failed_audio_*.wav` files)
  - [x] Recovered chunks added to transcript as "(Recovered)" speaker
  - [x] Enrollment dialog works even after mid-meeting server restart
  - [x] **Transcript crash recovery** (`.transcript_recovery.jsonl`) - entries saved incrementally
  - [x] Recovery dialog on Scribe startup if crash data found
  - [x] Empty transcription audio backup (silent engine failures save audio)
  - [x] Server busy tracking prevents shutdown during active transcription
- [ ] Notion integration (auto-create pages in database)

### Phase 4: Remote Transcription ✅
- [x] FastAPI server on desktop
- [x] Server accessible over Tailscale
- [x] Unified codebase (desktop + laptop in same folder)
- [x] Server-side diarization for remote Scribe

### Phase 5: Parakeet Engine (~50x Faster) ✅
- [x] **Engine Abstraction (completed 2026-02-01)**
  - [x] Abstract base class for transcription engines (`src/engines/base.py`)
  - [x] Factory pattern for engine registration and creation (`src/engines/factory.py`)
  - [x] Whisper engine wrapper (`src/engines/whisper_engine.py`)
  - [x] Parakeet engine wrapper (`src/engines/parakeet_engine.py`)
  - [x] Server updated to use engine factory
- [x] **Native Windows Support (completed 2026-02-03)**
  - [x] Parakeet runs natively on Windows (no WSL required)
  - [x] NeMo toolkit installed via pip with Visual C++ Build Tools
  - [x] Same server architecture as Whisper (server_tray.py)
  - [x] ~30 second startup, ~50x faster transcription than Whisper
- [x] **Config Integration (completed 2026-02-01)**
  - [x] Engine selection in config.yaml (`model_options.engine`)
  - [x] Settings UI dropdown for engine selection
  - [x] Server launcher auto-detects engine from config
  - [x] Single shortcut starts correct engine (Whisper or Parakeet)

---

## Troubleshooting

### Server won't start
```bash
# Check if port is in use
netstat -ano | findstr :9876

# Check server status
cd C:\dev\koe
python src/server_launcher.py status
```

### Server didn't stop when exiting Koe
If the server keeps running after exiting Koe via tray:
1. Check `logs/server_launcher.log` for shutdown errors
2. Find and kill the process: `netstat -ano | findstr :9876` → note PID → `taskkill /F /PID <pid>`
3. Note: Python 3.13 uses `pythonw3.13.exe` not `pythonw.exe` - check for that too

**Common causes:**
- Settings save triggered restart but didn't stop server (fixed in recent update)
- Shutdown request timed out (logged in server_launcher.log)

### Koe not responding to hotkey
1. Check tray icon exists
2. Try restarting: Exit via tray icon, then run `Start Koe Desktop` shortcut
3. Check `config.yaml` for correct `activation_key`

### Koe crashes or freezes during snippet transcription
Debug logging is enabled in `logs/debug.log`. If Koe crashes or freezes:
1. Check `logs/debug.log` for the execution trace leading up to the crash
2. Look for `EXCEPTION:` entries or incomplete sequences (e.g., `STARTED` without matching `FINISHED`)
3. The log traces: recording → transcription → snippet save → clipboard → beep → window close
4. Common issues: multiple instances (duplicate `ResultThread.run() STARTED` at same timestamp), file permission errors

**Log prefixes:**
- `[HH:MM:SS]` - General messages from result_thread.py
- `[HH:MM:SS] [transcription]` - Messages from transcription.py
- `[HH:MM:SS] [client]` - Messages from transcription_client.py (HTTP requests)

**Diagnosing freezes:** If the app freezes during transcription, check the last `[client]` log entry:
- Last line `POST .../transcribe` → Server not responding (hung or crashed)
- Last line `Converting to base64` → Memory issue encoding audio
- Last line `Response received` → Hang is after HTTP, in response parsing

**Log rotation:** Debug log auto-rotates at 1MB (keeps 1 backup as `debug.log.1`).

**Error recovery:** If transcription fails:
- Status window shows error message for 3 seconds (red X icon)
- Tray notification appears with error details
- Audio is saved to `logs/failed_audio_<timestamp>.wav` so recording isn't lost

### Scribe shows "Server not running"
1. Server should auto-start, but if not:
2. Close Scribe
3. Run `Start Koe Desktop shortcut` first (starts server)
4. Then run `Start Scribe Desktop shortcut`

### "Server not ready" error with Parakeet
Parakeet takes 30-60s to load. If you try a snippet before it's ready, you'll see this error.
- Wait for the server to finish loading (check `curl http://localhost:9876/status`)
- Audio is saved to `logs/failed_audio_*.wav` so your recording isn't lost
- The error prevents silent fallback to broken local mode

### "Server does not support long audio" error
**Parakeet fully supports long audio** (tested up to 6+ minutes) when local attention is enabled, which is the default. This rare error only appears if running a stale server instance that predates the local attention code.
- **Threshold**: Client checks `supports_long_audio` before transcribing >60 seconds
- **Cause**: Server was started before local attention code existed (stale process)
- **Check**: `curl http://localhost:9876/status` should show `supports_long_audio: true`
- **Fix**: Restart the server: Exit Koe via tray icon, then restart
- **Recovery**: Audio is saved to `logs/failed_audio_no_long_audio_support_*.wav`

### Out of VRAM
Close Koe (right-click tray icon → Exit) to release ~3GB VRAM.

### No audio in Scribe
1. Check default microphone in Windows Sound settings
2. Check default output device (loopback captures from this)
3. Verify pyaudiowpatch is installed: `pip install pyaudiowpatch`

### Laptop: "Server Not Available" error
1. Make sure desktop is running and server is started
2. Check Tailscale is connected on both machines
3. Verify `.env` has correct IP: `WHISPER_SERVER_URL=http://100.78.59.64:9876`
4. Test connection: `curl http://100.78.59.64:9876/status`

### Laptop: "Authentication failed" error
If you get "Authentication failed: invalid or missing API token":
1. Check `KOE_API_TOKEN` is set in `.env` on both desktop and laptop
2. Verify the tokens match exactly (no extra whitespace)
3. Restart Koe on both machines after changing the token
4. If you don't want authentication, remove `KOE_API_TOKEN` from both `.env` files

### Laptop: "Invalid WHISPER_SERVER_URL" error
If you see "Invalid WHISPER_SERVER_URL: must start with http:// or https://":
1. Check `.env` has the full URL including protocol: `WHISPER_SERVER_URL=http://100.78.59.64:9876`
2. Don't use just `100.78.59.64:9876` - must include `http://`

### Laptop: Import errors
Make sure you installed the remote requirements:
```bash
pip install -r requirements-remote.txt
```

### Too many speakers detected in Scribe
If a 2-person meeting shows "Speaker 1", "Speaker 2", "Speaker 3"... etc:
1. **Use enrollment dialog**: After meeting ends, enroll one of the unknown speakers
2. **Auto-merge kicks in**: Similar speakers (by embedding) are automatically merged
3. **Transcript rewrites**: All instances of merged speakers get renamed automatically
4. If enrolled speakers aren't matching, re-enroll them with longer/clearer audio samples

### Server crashed/restarted mid-meeting
If the server crashes or is restarted during a Scribe meeting:
1. **Session state persists**: Speaker embeddings are saved to `.session_state.npz` and auto-restored on server restart (valid for 1 hour)
2. **Failed chunks retried**: When you stop the meeting, Scribe retries any `failed_audio_*.wav` files from the last hour
3. **Recovered text**: Successfully retried chunks appear in transcript as "(Recovered)" speaker
4. **If enrollment dialog missing**: The session state should preserve unknown speakers - if not, check `meeting_debug.log` for `[Session]` entries

### Many transcription timeouts during meeting
If `meeting_debug.log` shows many timeouts (server overload, memory issues):
1. **Audio is saved**: Each timeout saves audio to `logs/failed_audio_timeout_*.wav`
2. **Auto-retry on stop**: When meeting stops, Scribe retries all failed chunks
3. **Check server health**: `curl http://localhost:9876/status` - if unresponsive, server may need restart
4. **Memory hog**: Check Task Manager for Python processes using excessive RAM (>3GB for Parakeet is abnormal)

### Scribe crashed mid-meeting (transcript recovery)
If Scribe crashes while recording a meeting:
1. **Recovery on startup**: Next time you open Scribe, a dialog will appear asking if you want to save the recovered transcript
2. **Incremental backup**: Transcript entries are saved to `.transcript_recovery.jsonl` as they arrive
3. **Click "Save"**: Recovered transcript is saved with "_recovered" suffix in the original output folder
4. **Click "Discard"**: Recovery file is deleted, data is lost
5. **Manual recovery**: If dialog doesn't appear, check if `.transcript_recovery.jsonl` exists in the koe folder

**Note:** Recovery saves transcript text only - speaker embeddings and meeting metadata may be incomplete.

### Empty transcription saved (no text)
If a snippet or meeting transcription returns empty but you spoke clearly:
1. **Audio is backed up**: For snippets >2 seconds that return empty, audio is saved to `logs/failed_audio_empty_result_*.wav`
2. **Check the audio**: Listen to the saved WAV file - if you can hear yourself, transcription engine may have issues
3. **Server status**: Check `python src/server_launcher.py status` - engine should show Ready: True
4. **Re-transcribe**: Use `python transcribe_file.py logs/failed_audio_empty_result_*.wav` to retry

### Long snippet killed/crashed mid-transcription
If you killed the process or it crashed while transcribing a long (>60s) snippet:
1. **Check for pending audio**: `ls logs/pending_audio_*.wav`
2. **Transcribe manually**: `python transcribe_file.py logs/pending_audio_*.wav`
3. **Why it exists**: Recordings >60s are pre-saved before sending to server as a safety net
4. **Clean up**: Delete the pending_audio file after successful transcription

### Server freezes after many long transcriptions
If the server becomes unresponsive after multiple long recordings:
1. **Root cause**: VRAM fragmentation after many transcriptions
2. **Auto-fix**: Server now clears CUDA cache every 10 minutes of cumulative audio
3. **If still frozen**: Kill and restart: `taskkill /F /IM pythonw.exe` then `python src/server_launcher.py start`
4. **Prevention**: Server restart clears all fragmentation - do this proactively if doing many hours of transcription

### Taskbar icon shows Python logo instead of Koe icon
Windows aggressively caches taskbar icons. To fix:
1. Exit Koe (right-click tray icon → Exit)
2. Clear icon cache:
   ```bash
   taskkill /F /IM python.exe /F /IM pythonw.exe
   del /A /Q "%localappdata%\IconCache.db"
   del /A /F /Q "%localappdata%\Microsoft\Windows\Explorer\iconcache*.db"
   ie4uinit.exe -show
   ```
3. Restart Windows Explorer or reboot PC
4. Launch Koe again - icon should now show correctly

**Note:** The AppUserModelID fix ensures icons work correctly after cache clear/reboot.

### Multiple Koe tray icons appearing
This shouldn't happen with single-instance protection. If you see multiple icons:
1. Check Task Manager for multiple python.exe/pythonw.exe processes
2. Kill all: `taskkill /F /IM python.exe /F /IM pythonw.exe`
3. Launch Koe once via shortcut

If it still happens, the socket lock may not be releasing properly - try rebooting.

### Parakeet engine not starting
If you selected Parakeet but the server doesn't start:
1. Check NeMo is installed: `pip show nemo_toolkit`
2. Check server status: `curl http://localhost:9876/status`
3. Check for import errors: `python -c "import nemo.collections.asr"`

**Common issues:**
- NeMo not installed: Run `pip install nemo_toolkit[asr]`
- Missing Visual C++ Build Tools: Install via `winget install Microsoft.VisualStudio.2022.BuildTools`
- CUDA not available: Check `nvidia-smi` works

### Parakeet slower than expected
Parakeet should be 50-80x realtime. If it's slow:
1. First transcription is slow (model loading) - subsequent ones are fast
2. Check GPU is being used: Server status should show `device: cuda`
3. Check local attention is enabled: Server logs should show "Local attention enabled"

### Switching between Whisper and Parakeet
1. Change engine in Settings → Transcription Engine
2. Exit Koe completely (tray icon → Exit) - server stops automatically
3. Restart Koe - new engine will start

---

## Claude Development Notes

### Restarting Koe
Koe has single-instance protection (socket lock on localhost:19877), so you can safely double-click the shortcut - it will exit if already running. To restart:

```
Right-click tray icon → Exit (server stops automatically)
Double-click Start Koe Desktop shortcut
```

**Note:** Scribe also has single-instance protection (socket lock on localhost:9878).

### Starting/Restarting the Server
**ALWAYS check if a server is already running before starting a new one!**

Multiple Parakeet servers will each try to load the model (~5GB), causing massive slowdown.

```bash
# Check server status (shows busy state)
cd /c/dev/koe && python src/server_launcher.py status

# If not running, start a new one
cd /c/dev/koe && python src/server_launcher.py start
```

**Safe restart (waits for active transcriptions to complete):**
```bash
cd /c/dev/koe && python src/server_launcher.py restart
```

**IMPORTANT: If a meeting or long transcription is in progress:**
- `restart` automatically waits up to 2 minutes for the request to complete
- NEVER use `--force` or `taskkill` during active transcription (will lose data)
- Check `busy: true` or `active_requests > 0` in status output

**Only if server is completely stuck (not responding to /status):**
```bash
# Nuclear option - only if server won't respond
taskkill //F //IM pythonw.exe 2>/dev/null
sleep 2
cd /c/dev/koe && python src/server_launcher.py start
```

### Null Device Redirection (Windows vs Bash)
**Do NOT use `> nul` in bash commands.** This creates an actual file named `nul` which:
- Is a Windows reserved name and very hard to delete
- Breaks Syncthing synchronization

**Correct usage:**
- **Bash/git-bash:** `> /dev/null 2>&1`
- **Windows cmd.exe:** `> NUL 2>&1`

If you need to suppress output, use `/dev/null` since Claude Code runs in a bash-like shell.

---
