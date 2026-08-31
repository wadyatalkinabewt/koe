# Koe

Koe is a Windows tray app for fast voice snippets and meeting transcripts. It
uses ElevenLabs Scribe v2 for speech-to-text and can use OpenRouter to produce
structured meeting summaries.

## What it does

- **Snippet:** press a global toggle hotkey, speak, and press it again to copy
  the transcript to the clipboard.
- **Scribe:** capture microphone and Windows loopback audio, submit one aligned
  mono recording for diarized transcription, and generate transcript and
  summary PDFs.
- **Private corrections:** supply local exact-token corrections without putting
  names, organisations, or domain terminology in the repository.

Snippet audio is kept in memory and is never written to disk. Successfully
decoded ElevenLabs transcripts are deleted from ElevenLabs by transcription ID.
Meeting source audio is retained only when the user enables that option.

See [docs/OPERATOR_GUIDE.md](docs/OPERATOR_GUIDE.md) for the meeting modes,
storage behaviour, and privacy boundaries.

## Requirements

- Windows 11
- Python 3.13
- an ElevenLabs API key
- an OpenRouter API key only if meeting summaries are enabled

## Run from source

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe run.py
```

The first-run window creates local settings. Secrets, settings, logs,
transcripts, recordings, and custom corrections are ignored by Git.

## Custom corrections

For stable speech-to-text substitutions, add a private `corrections` mapping to
Koe's existing `config.yaml` in the runtime data directory:

- source run: `<checkout>\config.yaml`
- packaged run: `%LOCALAPPDATA%\Koe\config.yaml`

This file-based setting intentionally has no Settings-window editor.

```yaml
transcription_options:
  corrections:
    ack me: Acme
```

Corrections match complete words or phrases, ignore case, and preserve the
matched text's lower/upper/title-style casing. They are applied locally after
transcription and are never sent to a provider. The real config file is private
and must not be committed.

## Development

Architecture, runtime paths, and verification commands are documented in
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

Koe currently runs directly from source.
