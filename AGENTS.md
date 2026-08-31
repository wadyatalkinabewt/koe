# Koe repository contract

Koe is a Windows-only tray application with two workflows:

1. Snippets: global toggle hotkey -> ElevenLabs Scribe v2 -> clipboard and
   five rotating Markdown files.
2. Scribe: microphone plus Windows loopback -> one aligned mono upload ->
   diarized meeting transcript -> optional OpenRouter summary.

## Invariants

- ElevenLabs Scribe v2 is the only speech-to-text backend.
- Every transcription request uses `no_verbatim=true`.
- Snippet formatting may normalize whitespace, punctuation, the paste-friendly
  trailing space, and the small exact-token correction map in
  `src/transcription.py`. Add corrections only for stable observed substitutions.
- Snippet audio is transient and must never be written to disk. Only rotating
  snippet Markdown and text diagnostics may persist. Scribe meeting audio
  remains independently optional.
- OpenRouter is confined to `src/meeting/summarizer.py` and Scribe
  post-processing. It may propose contextual names only for generic labels;
  exact transcript evidence must pass local validation before both documents
  are relabelled.
- The snippet hotkey is press-to-toggle. ElevenLabs remains the only
  transcription path.
- Settings autosave and must not restart Koe or interrupt an active snippet.
- The snippet status card keeps fixed geometry across Listening/Transcribing.
- Scribe sends one mono meeting file with `use_multi_channel=false`. Never
  reintroduce separate billable mic and loopback transcription requests or
  multichannel billing.
- Every Scribe mode enables diarization with speaker-library matching disabled.
  One-on-one requests cap the expected result at two speakers.
- Online group mic audio remains local attribution evidence only. Its detected
  diarized label maps to the current Settings name.
- Online one-on-one uses that same mic evidence when loopback is active, then
  maps every other diarized voice to the entered participant.
- In-person Scribe treats the microphone as a shared-room source, never forces
  one microphone speaker to the Settings name, and still captures loopback for
  remote callers. Effectively empty loopback is neither uploaded nor retained.
- In-person or speakerphone recordings preserve honest generic speaker labels
  unless contextual analysis validates a name from exact transcript evidence.
- Every successfully decoded ElevenLabs transcription response is immediately
  deleted from ElevenLabs using its returned `transcription_id`.
- OpenRouter meeting analysis must enforce per-request Zero Data Retention.
- Failed Scribe transcription attempts preserve local temporary audio for
  recovery. Successful runs remove temporary audio after document generation
  and any requested durable audio copies are verified.
- Every successful Scribe run writes `transcript.pdf` and `summary.pdf`.
  Markdown copies are written only when **Save Markdown copies** is enabled.
  The transcript PDF puts known names before cleanly renumbered anonymous
  speakers and reserves Koe green for the participant matching the current
  Settings name.
- Meeting notes are appended under a clearly labelled Notes section in the
  transcript PDF and optional transcript Markdown. Never write `notes.md`.
- Both one-on-one Scribe modes capture a separate meeting name and participant
  name; the meeting name drives the folder and document title.
- Scribe selects its four meeting modes inside the main window, remembers the
  last selection, and locks it once recording begins.
- Scribe summaries require Summary, Key Decisions, Topics Discussed, Action
  Items, and Open Questions; incomplete model responses are retried.

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
- `src/transcription.py`: ElevenLabs requests and rotating snippet text storage.
- `src/meeting/capture.py`: mic/loopback capture, mono mixing, and host mapping.
- `src/meeting/app.py`: Scribe UI and single-upload worker.
- `src/meeting/transcript.py`: Markdown transcript rendering.
- `src/meeting/pdf_theme.py`: shared meeting PDF typography, colour, and header.
- `src/meeting/transcript_pdf.py`: coloured-card PDF transcript rendering.
- `src/meeting/summarizer.py`: optional OpenRouter summary client.
- `src/meeting/summary_pdf.py`: clean PDF rendering for Scribe summaries.
- `src/ui/setup_window.py`: first-run GUI onboarding.
- `src/ui/theme.py`: shared desktop visual system.
- `src/config_schema.yaml`: authoritative preference schema.

## Working rules

- Read this file and live source before changing behavior.
- Preserve unrelated user changes and private output data.
- Prefer deleting retired paths over keeping compatibility switches.
- Do not add a VBS/Python installed launcher, second backend, cleanup stage, or
  second configuration source.
- Keep the dark-slate/indigo/coral visual system consistent across all surfaces.
- Use `apply_patch` for source edits. Verify resolved paths before bulk deletion.
- Private packaging, handoff artifacts, and generated executables are ignored
  and must never be committed.

## Verification

Run focused tests first, then:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
$files = @('run.py') + (Get-ChildItem src -Recurse -Filter *.py | ForEach-Object FullName)
.\.venv\Scripts\python.exe -m py_compile $files
git diff --check
```

Before restarting Koe, confirm no live snippet or Scribe recording would be
interrupted.
