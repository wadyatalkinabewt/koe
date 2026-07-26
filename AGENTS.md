# Koe repository contract

Koe is a Windows-only tray application with two workflows:

1. Snippets: global toggle hotkey -> ElevenLabs Scribe v2 -> clipboard and
   five rotating Markdown files.
2. Scribe: microphone plus Windows loopback -> one aligned mono upload ->
   diarized meeting transcript -> optional OpenRouter summary.

## Invariants

- ElevenLabs Scribe v2 is the only speech-to-text backend.
- Every transcription request uses `no_verbatim=true`.
- Snippet formatting may only normalize whitespace, punctuation, and the
  paste-friendly trailing space.
- OpenRouter is confined to `src/meeting/summarizer.py` and Scribe summaries.
- The snippet hotkey is press-to-toggle. ElevenLabs remains the only
  transcription path.
- Settings autosave and must not restart Koe or interrupt an active snippet.
- The snippet status card keeps fixed geometry across Listening/Transcribing.
- Scribe sends one mono meeting file with `use_multi_channel=false`. Never
  reintroduce separate billable mic and loopback transcription requests or
  multichannel billing.
- The original mic track is local attribution evidence only. Its detected
  diarized label maps to the current Settings name.
- Group Scribe always enables diarization and speaker-library matching.
- Operator install defaults keep vocabulary hints disabled and vocabulary empty.
- Operator's PDF-summary executable enables `KOE_SUMMARY_FORMAT=pdf` at build time;
  source/dev Koe keeps Markdown summaries.

## Data boundaries

Source/dev runs keep Alex's complete working state under the repository:

- `C:\Projects\koe`: `.env`, `config.yaml`, `.setup_complete`, and local shortcuts.
- `C:\Projects\koe\logs` and `C:\Projects\koe\.scribe_temp`: diagnostics and temporary audio.
- `C:\Projects\koe\Snippets` and `C:\Projects\koe\Meetings`: durable output.

Packaged installs retain normal per-user Windows storage:

- `%LOCALAPPDATA%\Koe`: secrets, settings, logs, and Scribe temp audio.
- `%USERPROFILE%\Documents\Koe`: durable snippets and meetings.

Never delete or rewrite either runtime layout without explicit user approval.
Never commit `.env`, `config.yaml`, private build secrets, recordings,
transcripts, vocabulary, or local paths. Installer upgrades and uninstall must
preserve packaged runtime state.

## Source map

- `run.py`: GUI setup gate and command-aware app bootstrap.
- `src/main.py`: tray, hotkey lifecycle, clipboard, and Scribe launch.
- `src/commands.py`: localhost single-instance shortcut command channel.
- `src/paths.py`: source-local and packaged per-user runtime locations.
- `src/result_thread.py`: snippet microphone capture and lifecycle.
- `src/transcription.py`: ElevenLabs requests and rotating snippet storage.
- `src/meeting/capture.py`: mic/loopback capture, mono mixing, and host mapping.
- `src/meeting/app.py`: Scribe UI and single-upload worker.
- `src/meeting/transcript.py`: Markdown transcript rendering.
- `src/meeting/summarizer.py`: optional OpenRouter summary client.
- `src/meeting/summary_pdf.py`: clean PDF rendering for Scribe summaries.
- `src/ui/setup_window.py`: first-run GUI onboarding.
- `src/ui/theme.py`: shared desktop visual system.
- `src/config_schema.yaml`: authoritative preference schema.
- `packaging/`: PyInstaller and Inno Setup release definitions.

## Working rules

- Read this file and live source before changing behavior.
- Preserve unrelated user changes and private output data.
- Prefer deleting retired paths over keeping compatibility switches.
- Do not add a VBS/Python installed launcher, second backend, cleanup stage, or
  second configuration source.
- Keep the dark-slate/indigo/coral visual system consistent across all surfaces.
- Use `apply_patch` for source edits. Verify resolved paths before bulk deletion.
- `packaging/private-Operator.env` is ignored and must contain only the dedicated,
  spending-capped OpenRouter key. Operator's ElevenLabs key comes from onboarding.
- Release one `Koe-Operator-Setup.exe`; do not produce a parallel portable zip.
- A user-requested Operator in-place hotfix may be published as one private,
  versioned `Koe.exe` release asset. Never commit generated executables, and
  scan the binary for key-shaped values before uploading it.

## Verification

Run focused tests first, then:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
$files = @('run.py') + (Get-ChildItem src -Recurse -Filter *.py | ForEach-Object FullName)
.\.venv\Scripts\python.exe -m py_compile $files
git diff --check
```

Before restarting Koe, confirm no live snippet or Scribe recording would be
interrupted. For a release, compile the installer, inspect its shortcuts and
preservation flags, and test first-run onboarding on a clean Windows profile.
