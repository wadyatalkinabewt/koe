# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]
### Added
- **Crash Recovery & Data Loss Prevention (2026-02-05)**:
  - **Transcript crash recovery**: Scribe now saves entries incrementally to `.transcript_recovery.jsonl` as they arrive. If app crashes mid-meeting, recovery dialog on next startup offers to save the recovered transcript.
  - **Server busy tracking**: Server tracks `busy` and `active_requests` in `/status` endpoint. Launcher waits up to 2 minutes for active transcriptions to complete before stopping.
  - **Empty transcription audio backup**: If transcription >2 seconds returns empty (silent engine failure), audio is saved to `logs/failed_audio_empty_result_*.wav`.
  - **Safe server restart**: `python src/server_launcher.py restart` waits for idle before stopping.
  - **Session state persistence**: Speaker embeddings saved to `.session_state.npz`, survives server restarts (1 hour TTL).
  - **Failed chunk retry**: Scribe automatically retries `failed_audio_*.wav` files when meeting stops.
- **Parakeet Engine (2026-02-01)**: NVIDIA NeMo-based transcription ~50x faster than Whisper
  - Engine abstraction with factory pattern (`src/engines/`)
  - WSL integration with systemd service for Parakeet
  - Settings UI for engine/model/device selection
  - CTC models recommended (TDT models have CUDA 12.8 incompatibility)
  - Auto-downloads models on first use
- **Scribe**: Continuous transcription with speaker diarization
- **Server Architecture**: Shared Whisper server (saves GPU memory, enables remote transcription)
- **Speaker Identification**: Pyannote-based diarization with voice fingerprinting
- **Speaker Enrollment**: Enroll speakers via post-meeting dialog with terminal-styled UI
- **Voice Filtering for Snippets**: Optional diarization to filter hotkey transcriptions to only your voice (Settings → Enrolled Speakers)
- **Adaptive Voice Fingerprinting**: Speaker embeddings improve over time with high-confidence matches
- **Cross-Chunk Speaker Tracking**: Running average embeddings for better speaker consistency across meeting chunks
- **Auto-merge Similar Speakers**: Automatically consolidates duplicate speaker detections via embedding similarity
- **Meeting Organization**: Category/subcategory folders (Standups, One-on-ones/Calum, Investors/Sequoia, etc.)
- **Configurable Output Folders**: Set custom paths for meetings and snippets in Settings
- **Pre-Meeting Agendas**: Save agenda files without date prefix, load and continue during meeting
- **Remote Transcription**: Use laptop over Tailscale without GPU
- **Custom Icon**: Sound bars design in terminal green (#00ff88) for app, tray, and shortcuts
- **Terminal-Themed UI**: Settings window with custom scrollbar matching Koe aesthetic
- **Single-Instance Protection**: Prevents duplicate Koe/Scribe processes via socket locks
- **Initialization Window**: Small popup shows "Initializing" during startup, disappears when tray icon is ready
- **VBS Launchers**: Hidden console windows for clean startup experience
- **Draggable Status Window**: Move recording popup anywhere on screen, click [ESC] or press Escape to cancel
- **AI Summarization (2026-01-21)**: Auto-generates meeting summaries using Claude Sonnet 4.5
  - Detached subprocess - window can close, summary continues in background
  - Live progress updates with clickable VS Code link
  - Centralized status files stored in `.summary_status/` folder (keeps meeting folders clean)
  - Anti-hallucination prompt with strict accuracy guidelines
  - Mirrored Summaries/ folder structure matching Transcripts/
  - Retry logic (3 attempts, exponential backoff)
  - ~$0.04 per 60-minute meeting
- New continuous recording mode ([Issue #40](https://github.com/savbell/whisper-writer/issues/40))
- New option to play a sound when transcription finishes ([Issue #40](https://github.com/savbell/whisper-writer/issues/40))
- **Setup Wizard (2026-01-21)**: Guided first-time setup experience
  - Automatic system requirements check (GPU, CUDA, Python, packages)
  - API key configuration (HuggingFace required, Anthropic optional)
  - Model download with progress (~3-4GB)
  - User profile setup (name input)
  - Voice enrollment with recording UI
  - Output folder configuration
  - Auto-launches Koe when complete
  - Re-run anytime with `python run.py --setup`
- **Comprehensive Documentation (2026-01-21)**: Complete documentation overhaul
  - README.md expanded to 1100+ lines with full architecture diagrams
  - Table of contents, system requirements, API reference
  - Categorized dependencies with version numbers and purposes
  - Step-by-step installation guides for desktop and laptop
  - Complete configuration reference with all options
  - Troubleshooting guide for common issues
  - SETUP.md condensed to quick-start guide referencing README
- **Centralized Theme System (2026-01-21)**: Created `src/ui/theme.py` with all color constants (terminal green #00ff88)
  - Eliminates duplicate color definitions across 8 files
  - Single source of truth for UI theming
- **Centralized Error Logging (2026-01-21)**: Created `src/logger.py` for silent error logging
  - Logs to `logs/koe_errors.log` (no console output for pythonw.exe processes)
  - Convenience functions: `log_error()` and `log_exception()`
- **Config Validation (2026-01-21)**: Added validation to ConfigManager
  - Type checking against schema
  - Invalid values reset to defaults with warnings
  - Prevents silent crashes from malformed config
- **Post-Meeting Speaker Enrollment (2026-01-22)**: Enroll speakers directly from meeting recordings
  - "Enroll Speakers" button appears in summary window if there are unknown speakers
  - Shows unknown speakers with sample transcriptions for identification
  - Auto-rewrites transcript when speaker is enrolled (replaces "Speaker 1" with enrolled name)
  - Auto-merges similar speakers by embedding similarity (if "Speaker 3" matches "Speaker 1", both replaced)
  - Summary generation waits until enrollment dialog is closed (uses correct names)
  - Uses session running-average embeddings (higher quality than short recordings)
- **Initial Prompt Leak Filtering (2026-01-22)**: Prevents Whisper initial prompt from appearing in transcriptions
  - Filters prompt text and sentence fragments from transcription output
  - Applied to both Scribe meetings and Koe hotkey snippets
- **Mic Audio Diarization (2026-01-22)**: Mic audio now processed through diarization pipeline
  - Ensures timing consistency with loopback audio (both through same pipeline)
  - Extracts user's voice embedding for adaptive learning
  - Still force-labeled as user_name from config (not identified by diarization)

### Changed
- Renamed project from WhisperWriter to Koe
- Scribe window opens instantly (diarization models load async in background)
- **Window Sizes (2026-01-21)**:
  - Scribe minimum size increased from 800x700 to 1000x800 (more comfortable for note-taking)
  - Enrollment window increased from 480x170 to 500x185 (better text visibility)
- **Scribe Performance Phase 1 (2026-01-21)**: Instant UI feedback - start/stop buttons change state immediately with progress messages, form inputs locked during recording, notes auto-cleared after save, removed distracting buffer status updates
- **Scribe Performance Phase 2 (2026-01-21)**: True async operations - moved blocking operations (server check, transcription, file I/O) to background threads, UI now truly non-blocking and responsive during start/stop operations
- **Scribe Performance Phase 3 - Partial (2026-01-21)**: Visual polish - added blinking "● REC" indicator in header (impossible to forget you're recording), "New Meeting" button for quick reset between meetings
- **Scribe UI Improvements (2026-01-21)**:
  - Window opens larger (1100x900) for comfortable note-taking
  - Form layout reorganized: NAME on row 1, CATEGORY + SUB on row 2 (simpler, cleaner)
  - "New Meeting" button now hidden until recording completes (no pointless clearing of empty fields)
  - Clean status messages: removed "Transcribing chunk #X" spam, user-friendly summarization messages
  - Dark terminal-themed context menus (no more white Windows menus)
  - Summary window improved spacing (450x85, centered status text, minimum height prevents text jumping)
  - Summary links open directly in VS Code (tries multiple locations, falls back gracefully)
  - Recording indicator changed to smooth pulsating red circle (30%-100% opacity) - no box, no text
  - Form fields (NAME, CATEGORY, SUB, SPEAKERS) completely hidden during recording (cleaner UI)
- **AI Summarization Status Files (2026-01-21)**: Moved from transcript folders to centralized `.summary_status/` directory
  - Keeps meeting folders clean (no `.json` files mixed with transcripts)
  - Hash-based naming ensures uniqueness: `MeetingName_a1b2c3d4.json`
  - Auto-cleanup after completion
  - Easy to find/delete stale files if process crashes
- Snippets folder configurable in Settings (default: Koe/Snippets)
- Status window stays visible during transcription, closes immediately after beep (no "Complete!" message - beep is sufficient feedback)
- Utility scripts moved to scripts/ folder (create_shortcuts.ps1, generate_icon.py)
- Batch launchers use pythonw instead of python (no console window flash)
- Expanded hallucination filter with more YouTube outro patterns and trailing phrases
- Settings window auto-refreshes speaker list after enrollment
- Migrated status window from using `tkinter` to `PyQt5`
- Migrated from using JSON to using YAML to store configuration settings
- Upgraded to latest versions of `openai` and `faster-whisper`, including support for local API ([Issue #32](https://github.com/savbell/whisper-writer/issues/32))

### Removed
- No longer using `keyboard` package to listen for key presses
- Removed obsolete enrollment batch files (now done via post-meeting dialog)
- Removed debug files (debug_init.py, DEBUG_STATUS.md, meeting_debug.log)
- Removed legacy asset files (old tray icons, microphone.png, pencil.png)
- **Manual Enrollment from Tray Menu (2026-01-22)**: "Enroll Speaker" submenu removed
  - Use post-meeting enrollment instead (better quality embeddings, auto transcript cleanup)
  - CLI tools still available for debugging: `python -m src.meeting.enroll_speaker`
- **Summary .summary suffix (2026-01-22)**: Summaries now use same filename as transcripts (in separate Summaries/ folder)
  - Old: `26_01_22_Standup.summary.md`
  - New: `26_01_22_Standup.md` (in Summaries/ folder instead of Transcripts/)
- **ATTENDEES Dropdown (2026-01-22)**: Removed per-meeting max_speakers selection from Scribe UI
  - Auto-merge handles speaker consolidation automatically
  - Hardcoded default of 8 max speakers is sufficient for most meetings
  - Simpler UI with less per-meeting configuration needed

### Fixed
- [ESC] button now properly stops transcription instead of leaving it running in background
- Status window timing: shows "Transcribing..." for full duration, closes after beep (2s display time)
- Status window alignment: centered text and timer between dot and [ESC] button
- Summary window alignment: centered status text for consistent layout
- Clipboard copy interrupted: pressing hotkey during transcription no longer interrupts clipboard copy or "Complete!" display
- QMenu constructor error: removed invalid QApplication parent argument
- Hallucinations: added `hallucination_silence_threshold=0.5` to all Whisper calls
- Speaker matching: improved threshold (0.35) and session tracking for fewer false speakers
- Windows taskbar icon: set AppUserModelID for consistent Koe icon display
- Enrollment window text cut off: increased window size and font size
- Settings scrollbar styling: custom terminal-themed scrollbar with green accent
- Summarization status messages: replaced technical "Connecting to API..." with user-friendly "Analyzing transcript..."
- **Config schema mismatch (2026-01-21)**: Added missing fields to config_schema.yaml (max_speakers, post_processing section with 5 fields)
- **Wrong schema default (2026-01-21)**: Changed condition_on_previous_text default from true to false (prevents transcription bleeding)
- **Relative path bugs (2026-01-21)**: Fixed beep.wav and speaker_embeddings paths to use absolute paths (Path(__file__).parent pattern)
- **No config validation (2026-01-21)**: Added type checking and options validation in ConfigManager (invalid values reset to defaults)
- **No error logging (2026-01-21)**: Errors no longer lost when using pythonw.exe - now logged to logs/koe_errors.log
- **Duplicate color constants (2026-01-21)**: Eliminated duplicate color definitions across 8 files with centralized theme.py
- **Initial prompt leaking (2026-01-22)**: Whisper initial prompt (e.g., "Use proper punctuation...") no longer appears in transcription output
- **Mic/loopback timing inconsistency (2026-01-22)**: Mic audio (fast) could arrive before loopback (slow diarization), causing transcript ordering issues. Both now go through diarization pipeline.
- **User self-enrollment gap (2026-01-22)**: User couldn't enroll/update themselves from post-meeting dialog because mic wasn't diarized. Now mic embedding is extracted and tracked in session.
- **Post-meeting enrollment not appearing in remote mode (2026-01-23)**: When using server-side diarization (laptop over Tailscale), unknown speakers like "Speaker 1" appeared in transcripts but the enrollment prompt never showed. Fixed by adding `/diarization/unenrolled` API endpoint to return session speakers with embeddings, and fetching from server when in remote mode.
- **Parakeet TDT CUDA 12.8 crash (2026-02-01)**: TDT model's CUDA graph decoder expected 6 return values from `cu_call()` but CUDA 12.8 returns 5. Switched default to CTC model which doesn't use CUDA graphs.
- **Parakeet "not installed" in Settings (2026-02-01)**: WSL availability check failed due to UTF-16 LE encoding of `wsl --list` output on Windows. Fixed with proper decoding.
- **Crash when clicking ESC during recording (2026-02-03)**: PyQt signal emitted to closed window during signal delivery caused crash. Replaced signal with direct callback; ESC cancel now only works during recording state (ignored during transcription).
- **Parakeet extremely slow on long audio (2026-02-03)**: Audio >60s caused O(n²) attention slowdown (180s audio took 160s to transcribe = 1.1x realtime). Enabled local attention (`change_attention_model('rel_pos_local_attn', [128, 128])`) and auto-chunking. Now 180s transcribes in 2.6s = 69x realtime.
- **Transcription timeout too short (2026-02-03)**: 300s cap was insufficient for long recordings. Increased to 900s cap with 3x audio duration formula.
- **Lost audio on transcription failure (2026-02-03)**: When server timed out or errored, audio was lost forever. Now saves failed audio to `logs/failed_audio_<reason>_<timestamp>.wav` for recovery.
- **Unicode emoji crash (2026-02-01)**: Config validation warnings used ⚠ emoji which Windows console couldn't encode. Changed to ASCII `[!]`.

## [1.0.1] - 2024-01-28
### Added
- New message to identify whether Whisper was being called using the API or running locally.
- Additional hold-to-talk ([PR #28](https://github.com/savbell/whisper-writer/pull/28)) and press-to-toggle recording methods ([Issue #21](https://github.com/savbell/whisper-writer/issues/21)).
- New configuration options to:
  - Choose recording method (defaulting to voice activity detection).
  - Choose which sound device and sample rate to use.
  - Hide the status window ([PR #28](https://github.com/savbell/whisper-writer/pull/28)).

### Changed
- Migrated from `whisper` to `faster-whisper` ([Issue #11](https://github.com/savbell/whisper-writer/issues/11)).
- Migrated from `pyautogui` to `pynput` ([PR #10](https://github.com/savbell/whisper-writer/pull/10)).
- Migrated from `webrtcvad` to `webrtcvad-wheels` ([PR #17](https://github.com/savbell/whisper-writer/pull/17)).
- Changed default activation key combo from `ctrl+alt+space` to `ctrl+shift+space`.
- Changed to using a local model rather than the API by default.
- Revamped README.md, including new Roadmap, Contributing, and Credits sections.

### Fixed
- Local model is now only loaded once at start-up, rather than every time the activation key combo was pressed.
- Default configuration now auto-chooses compute type for the local model to avoid warnings.
- Graceful degradation to CPU if CUDA isn't available ([PR #30](https://github.com/savbell/whisper-writer/pull/30)).
- Removed long prefix of spaces in transcription ([PR #19](https://github.com/savbell/whisper-writer/pull/19)).

## [1.0.0] - 2023-05-29
### Added
- Initial release of WhisperWriter (original project by savbell).
- Added CHANGELOG.md.
- Added Versioning and Known Issues to README.md.

### Changed
- Updated Whisper Python package; the local model is now compatible with Python 3.11.

[Unreleased]: https://github.com/savbell/whisper-writer/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/savbell/whisper-writer/releases/tag/v1.0.0...v1.0.1
[1.0.0]: https://github.com/savbell/whisper-writer/releases/tag/v1.0.0
