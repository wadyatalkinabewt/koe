# Development

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Runtime boundary

Koe is a Windows-only tray application. Source runs resolve private runtime data
from the checkout; frozen builds resolve it from the current Windows user
profile. Tests override both locations with `KOE_APPDATA_DIR` and
`KOE_DOCUMENTS_DIR` so they never touch real settings or recordings.

Source-run private state:

```text
<checkout>\.env
<checkout>\config.yaml
<checkout>\logs\
<checkout>\.scribe_temp\
<checkout>\Snippets\
```

Durable documents resolve to the user's Documents folder in source and packaged
runs alike, so they survive a checkout being moved or rebuilt:

```text
%USERPROFILE%\Documents\Koe\Scribe\
%USERPROFILE%\Documents\Koe\Transcripts\
```

Do not commit or rewrite those paths. Before restarting a running source
instance, confirm that no snippet or Scribe recording is active.

## Source map

- `run.py`: setup gate and application bootstrap.
- `src/commands.py`: single-instance shortcut command channel.
- `src/paths.py`: source and packaged runtime locations.
- `src/main.py`: tray, hotkey, clipboard, and Scribe launch lifecycle.
- `src/result_thread.py`: snippet capture and transcription lifecycle.
- `src/microphone_mute.py`: Windows mute control for the selected snippet input.
- `src/transcription.py`: provider dispatch, the shared ElevenLabs Snippet
  retry/slow-request recovery coordinator, custom corrections, and rotating
  snippet storage.
- `src/providers/`: Deepgram and Mistral request/response adapters.
- `src/meeting/capture.py`: microphone/loopback capture and mono mixing.
- `src/meeting/app.py`: Scribe UI and single-upload worker.
- `src/meeting/transcript.py`: Markdown transcript rendering.
- `src/meeting/transcript_pdf.py`: PDF transcript rendering.
- `src/meeting/summarizer.py`: optional OpenRouter summary client.
- `src/meeting/summary_pdf.py`: PDF summary rendering.
- `src/recording_transcript.py`: headless transcription of existing audio files
  (ffmpeg decode, filename convention, channel attribution, documents).
- `src/ui/transcribe_window.py`: desktop window over that headless module.
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
- Normal Snippet audio never persists to disk. Cancellation or a failed
  transcription may retain one verified `logs/recoverable-snippet.wav`, which
  the next Snippet start deletes. Saving recovery audio makes no provider
  request; failure notifications report the save outcome and deletion warning.
- ElevenLabs Snippets share three attempts per ten-minute audio chunk across
  the initial upload, one optional 15-second pending-request hedge, one allowed
  pre-upload connection-timeout retry, and transient-429 retries. Busy retries
  wait 5/10 seconds before attempts 2/3, respecting a longer `Retry-After`.
  Only `system_busy` and `too_many_concurrent_requests` are eligible 429 codes;
  no new retry starts while another request for that chunk is pending. Retried
  requests do not get their own hedge. Earlier completed chunks are not resent.
- Snippet `transcribe(..., status_callback=...)` passes ElevenLabs recovery
  state through the ResultThread signal to the card. `retrying` is sticky until
  completion/failure and dismissible without stopping work. All processing
  states preserve fixed card/control geometry and the recording duration.
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
- Existing recordings are decoded with ffmpeg to one 16 kHz mono upload. When
  the stereo channels carry genuinely different signals, speakers are attributed
  from channel energy and the provider is not asked to diarize; correlated
  channels fall back to diarization. A segment whose left/right margin is too
  small keeps the previous speaker instead of flipping on cross-talk.
- Recording transcription is idempotent: an existing `transcript.md` is reused
  unless the caller forces a new run.
- `ffmpeg` and `ffprobe` must be on `PATH` for recording transcription; the
  module raises a clear error rather than decoding silently.
- `transcription_options.corrections` is re-read from the private config on
  each transcription, keyed by the file's modification stamp and size. Editing
  the dictionary therefore takes effect on the next Snippet or Scribe run
  without a restart; an absent, unreadable, or malformed file falls back to the
  in-memory config instead of failing transcription.

## Verification

Run focused tests for changed behaviour, then:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
$files = @('run.py') + (Get-ChildItem src -Recurse -Filter *.py | ForEach-Object FullName)
.\.venv\Scripts\python.exe -m py_compile $files
git diff --check
```

`pytest.ini` pins collection to `tests`, so a bare `pytest -q` measures only
Koe's suite regardless of what sits in ignored scratch folders. Use pytest's
reported test count for the current checkout; dated private receipts
may describe older counts. Add `testpaths` coverage rather than relying on
root-directory collection.
