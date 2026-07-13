# Koe repository contract

Koe is a Windows-only tray application with two workflows:

1. Snippets: global hotkey capture -> ElevenLabs Scribe v2 -> clipboard and
   five rotating Markdown files.
2. Scribe: microphone plus Windows loopback -> ElevenLabs Scribe v2 ->
   stream-labelled meeting transcript -> optional OpenRouter summary.

## Invariants

- ElevenLabs Scribe v2 is the only speech-to-text backend.
- Every transcription request uses `no_verbatim=true`.
- Snippets receive no AI cleanup pass; local formatting may only normalize
  whitespace, punctuation, and the paste-friendly trailing space.
- OpenRouter is confined to `src/meeting/summarizer.py` and Scribe summaries.
- The snippet hotkey is press-to-toggle. Do not restore continuous, hold, VAD,
  local-model, GPU, Whisper, Groq, or backend-selection paths.
- Settings autosave and must not restart Koe or interrupt an active snippet.
- The snippet status card keeps fixed geometry across Listening/Transcribing.
- A Scribe source shorter than ElevenLabs' 100 ms minimum is skipped. Report No
  Speech Detected only when neither stream yields speech.
- One-on-one Scribe uses deterministic mic/loopback labels. Group Scribe keeps
  the mic owner deterministic and diarizes loopback with speaker-library
  matching.

## Data boundaries

Never delete or rewrite user data under `Meetings/`, `Snippets/`, or `logs/`.
Never commit `.env`, `src/config.yaml`, `.scribe_temp/`, `.setup_complete`, or
the private runtime-data directories. Do not expose API keys, recordings,
transcripts, vocabulary, or local paths in review packets.

## Source map

- `run.py`: setup gate and app bootstrap.
- `src/main.py`: tray, hotkey lifecycle, clipboard, and Scribe launch.
- `src/result_thread.py`: snippet microphone capture and lifecycle.
- `src/transcription.py`: ElevenLabs requests and rolling snippet storage.
- `src/meeting/capture.py`: microphone and WASAPI loopback capture.
- `src/meeting/app.py`: Scribe UI and worker.
- `src/meeting/transcript.py`: Markdown transcript rendering.
- `src/meeting/summarizer.py`: optional OpenRouter summary client.
- `src/ui/theme.py`: shared desktop visual system.
- `src/config_schema.yaml`: authoritative preference schema.

## Working rules

- Read this file and the live source before changing behavior.
- Preserve unrelated user changes and private output data.
- Prefer deleting retired paths over keeping compatibility switches for them.
- Do not add a second launcher, backend, cleanup stage, or configuration source.
- Keep the dark-slate/indigo/coral visual system consistent across all surfaces.
- Use `apply_patch` for source edits. Verify resolved paths before bulk deletion.

## Verification

Run focused tests first, then:

```powershell
python -m pytest -q
$files = @('run.py') + (Get-ChildItem src -Recurse -Filter *.py | ForEach-Object FullName)
python -m py_compile $files
git diff --check
```

For UI changes, render representative Qt states offscreen. Before restarting
Koe, confirm no live snippet or Scribe recording would be interrupted; then
confirm the `run.py` parent and `src/main.py` child remain running.
