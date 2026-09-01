<img src="assets/koe-icon.png" alt="Koe app icon" width="80" align="right">

# Koe

**Catch a thought. Capture the room. Keep the useful part.**

`Windows 11` &nbsp;·&nbsp; `Python 3.13` &nbsp;·&nbsp;
`ElevenLabs Scribe v2` &nbsp;·&nbsp; `MIT`

[![Windows tests](https://github.com/wadyatalkinabewt/koe/actions/workflows/tests.yml/badge.svg)](https://github.com/wadyatalkinabewt/koe/actions/workflows/tests.yml)

Koe is a Windows tray app built for the gap between *I should write that down*
and *what did we actually decide?* Tap a global hotkey for a quick voice
snippet, or open Scribe to turn microphone and system audio into a diarized
meeting transcript, clean PDFs, and a structured summary.

It stays out of the way, treats retention as a design constraint, and keeps
private names and terminology in local configuration rather than in the code.

## Two ways to use it

| | Snippet | Scribe |
|---|---|---|
| **Best for** | Fleeting thoughts, prompts, and dictated text | Calls, interviews, and in-room meetings |
| **Capture** | Press the hotkey, speak, press again | Record microphone and Windows loopback together |
| **Result** | Formatted text on the clipboard | Diarized transcript and summary PDFs |
| **Audio retention** | Never written to disk | Kept only when requested; recovery audio survives failures |

<p align="center">
  <img src="assets/readme/workflows.png" alt="Koe Snippet and Scribe workflows" width="100%">
</p>

Scribe aligns microphone and loopback audio into **one mono timeline** before
uploading it. That avoids two separately transcribed recordings drifting out of
sync, duplicating speakers, or doubling the transcription work.

<p align="center">
  <img src="assets/readme/documents.png" alt="Synthetic Koe transcript and summary PDF examples" width="100%">
</p>

### Built in layers

- **Capture, transcription, correction, speaker resolution, summarisation, and
  document rendering are separate modules.** ElevenLabs Scribe v2 is the current
  transcription adapter; the capture, correction, and document stages do not
  depend on its interface.
- **The summary model is easy to swap within OpenRouter.** It is selected in one
  place and sits behind a structured JSON contract. A replacement model still
  needs to honour that contract and pass the summary tests.
- **Post-transcription corrections are local and provider-independent.** They
  can fix recurring names or jargon without sending a private dictionary to a
  model provider.

The source map and the invariants between those modules are documented in
[the development guide](docs/DEVELOPMENT.md).

## Privacy is part of the pipeline

- Snippet audio stays in memory and is never written to disk.
- Scribe uploads one aligned recording rather than separate microphone and
  loopback tracks.
- Successfully decoded ElevenLabs transcripts are deleted from ElevenLabs by
  transcription ID; failed deletions are retried and recorded locally.
- Meeting source audio is retained only when enabled. Failed jobs preserve
  recovery audio instead of silently destroying it.
- OpenRouter is confined to meeting post-processing and must use a Zero Data
  Retention endpoint.
- Custom corrections, settings, secrets, logs, recordings, and generated
  documents are excluded from Git.

For the exact meeting modes, storage paths, and failure behaviour, see the
[operator guide](docs/OPERATOR_GUIDE.md).

## Run from source

Koe currently targets **Windows 11** and **Python 3.13**. Create an
[ElevenLabs API key](https://elevenlabs.io/app/api/api-keys) with only
**Speech to Text → Access** enabled. Koe does not require User, History, or any
administrative permission. An OpenRouter key is needed only for model-assisted
meeting summaries and contextual speaker resolution.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

The first-run window creates local settings. Source runs keep private runtime
state in the checkout; packaged runs use the current Windows profile. The
[development guide](docs/DEVELOPMENT.md) covers contributor setup, runtime
boundaries, architecture, and verification.

## Teach Koe your vocabulary

Add exact-token corrections to the private `config.yaml` used by the current
runtime:

- source run: `<checkout>\config.yaml`
- packaged run: `%LOCALAPPDATA%\Koe\config.yaml`

```yaml
transcription_options:
  corrections:
    ack me: Acme
```

Matching ignores case while preserving lower-, upper-, or title-style casing.
Corrections run locally after transcription. The real configuration is private
and must never be committed.

## Follow the build

This repository preserves Koe's development history rather than presenting a
single polished snapshot:

[Commit activity](https://github.com/wadyatalkinabewt/koe/graphs/commit-activity)
&nbsp;·&nbsp;
[Contributors graph](https://github.com/wadyatalkinabewt/koe/graphs/contributors)
&nbsp;·&nbsp;
[Commit log](https://github.com/wadyatalkinabewt/koe/commits/main)
&nbsp;·&nbsp;
[Repository insights](https://github.com/wadyatalkinabewt/koe/pulse)

## License

Koe's original source code is released under the [MIT License](LICENSE).
Third-party dependencies retain their own licences. PyQt5 is dual-licensed
under GPLv3 or a commercial Riverbank licence; review
[Riverbank's licensing terms](https://www.riverbankcomputing.com/software/pyqt)
before redistributing a bundled application.
