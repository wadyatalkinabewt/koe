# Koe

Koe is a Windows tray app for two ElevenLabs Scribe v2 workflows:

- **Snippet:** press `Ctrl+Shift+Space`, speak, and press it again to copy the
  transcript to the clipboard.
- **Scribe:** record a meeting, keep notes, produce a labelled transcript, and
  optionally generate a structured OpenRouter summary.

Every transcription request uses `no_verbatim=true`. ElevenLabs is the single
speech-to-text path.

## Scribe billing and speaker labels

Koe opens both capture devices before either starts, records microphone and
Windows loopback separately, overlays them into one aligned mono WAV, and sends
that file once with `use_multi_channel=false`. A one-hour meeting therefore
produces one one-hour transcription upload rather than two one-hour channel
uploads.

The original microphone track is never transcribed separately. Koe uses its
timing and energy locally to identify which diarized label belongs to the name
in Settings. Other speakers keep ElevenLabs speaker-library names when known,
or readable `Speaker 1`, `Speaker 2`, and so on when unknown.

Koe writes Markdown summaries by default. Set `KOE_SUMMARY_FORMAT=pdf` to write
a clean PDF summary instead. Transcripts and optional notes remain Markdown. If
**Save Scribe meeting audio** is enabled, the original source tracks are also
kept with the meeting:

```text
transcript.md
summary.pdf         # when KOE_SUMMARY_FORMAT=pdf
summary.md          # default
notes.md           # only when notes were entered
microphone.wav     # only when audio retention is enabled
meeting-audio.wav  # only when audio retention is enabled
```

## Development

Koe is developed and tested on Windows 11 with Python 3.13.

Source runs keep the complete local development instance inside the checkout:

```text
C:\Projects\koe\.env
C:\Projects\koe\config.yaml
C:\Projects\koe\logs\
C:\Projects\koe\.scribe_temp\
C:\Projects\koe\Snippets\
C:\Projects\koe\Meetings\
C:\Projects\koe\.venv\
```

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe run.py
```

Tests can override the runtime layout with `KOE_APPDATA_DIR` and
`KOE_DOCUMENTS_DIR`.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
$files = @('run.py') + (Get-ChildItem src -Recurse -Filter *.py | ForEach-Object FullName)
.\.venv\Scripts\python.exe -m py_compile $files
git diff --check
```
