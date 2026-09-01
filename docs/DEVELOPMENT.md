# Development

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Runtime boundary

Koe is a Windows-only tray application. Source runs resolve runtime data from
the checkout; frozen builds resolve private data from the current Windows user
profile. Tests override both locations with `KOE_APPDATA_DIR` and
`KOE_DOCUMENTS_DIR` so they never touch real settings or recordings.

Source-run private state:

```text
<checkout>\.env
<checkout>\config.yaml
<checkout>\logs\
<checkout>\.scribe_temp\
<checkout>\Snippets\
<checkout>\Meetings\
```

Do not commit or rewrite those paths. Before restarting a running source
instance, confirm that no snippet or Scribe recording is active.

## Source map

- `run.py`: setup gate and application bootstrap.
- `src/commands.py`: single-instance shortcut command channel.
- `src/paths.py`: source and packaged runtime locations.
- `src/main.py`: tray, hotkey, clipboard, and Scribe launch lifecycle.
- `src/result_thread.py`: snippet capture and transcription lifecycle.
- `src/transcription.py`: provider dispatch, custom corrections, and rotating
  snippet storage.
- `src/providers/`: Deepgram and Mistral request/response adapters.
- `src/meeting/capture.py`: microphone/loopback capture and mono mixing.
- `src/meeting/app.py`: Scribe UI and single-upload worker.
- `src/meeting/transcript.py`: Markdown transcript rendering.
- `src/meeting/transcript_pdf.py`: PDF transcript rendering.
- `src/meeting/summarizer.py`: optional OpenRouter summary client.
- `src/meeting/summary_pdf.py`: PDF summary rendering.
- `src/config_schema.yaml`: preference schema.

## Invariants

- ElevenLabs Scribe v2 is the default transcription backend. Deepgram Nova-3
  and Mistral Voxtral adapters declare the same Snippet and Scribe contracts.
- First-run setup persists the selected provider and matching environment key
  only after a real request to that provider's speech-to-text endpoint succeeds.
- Provider credentials remain distinct: `ELEVENLABS_API_KEY`,
  `DEEPGRAM_API_KEY`, and `MISTRAL_API_KEY`. `OPENROUTER_API_KEY` is optional
  post-processing configuration and is never used as a transcription key.
- Every ElevenLabs request uses `no_verbatim=true`.
- Snippet audio never persists to disk.
- Scribe sends one aligned mono upload. ElevenLabs explicitly sets
  `use_multi_channel=false`; other adapters receive that same mono file.
- OpenRouter is confined to Scribe post-processing and requires Zero Data
  Retention. A missing key skips that optional stage; a provider failure leaves
  the transcript as the sole deliverable and never produces a diagnostic
  `summary.pdf` or `summary.md`.
- Successfully decoded ElevenLabs responses are deleted by transcription ID.
- Deepgram and Mistral return no equivalent deletable transcript ID; their
  provider retention policies apply and are documented in the README matrix.
- Settings autosave must not restart Koe or interrupt active audio.

## Verification

Run focused tests for changed behaviour, then:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
$files = @('run.py') + (Get-ChildItem src -Recurse -Filter *.py | ForEach-Object FullName)
.\.venv\Scripts\python.exe -m py_compile $files
git diff --check
```
