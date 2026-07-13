# Koe

Koe is a Windows tray app for two speech-to-text workflows:

- **Snippets:** press `Ctrl+Shift+Space`, speak, press it again, and receive the
  transcript on the clipboard.
- **Scribe:** record microphone and Windows system audio into a timestamped
  meeting transcript with an optional summary.

ElevenLabs Scribe v2 is the only transcription backend. Every request uses
`no_verbatim=true`. OpenRouter is used only for optional Scribe summaries.

## Install

Koe is developed and tested on Windows 11 with Python 3.13.

```powershell
cd C:\Projects\koe
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python run.py --setup
```

`ELEVENLABS_API_KEY` is required. `OPENROUTER_API_KEY` is optional and only
affects meeting summaries. Both belong in `.env`, which is excluded from Git.

Start Koe with:

```powershell
pythonw run.py
```

The local `Start Koe.lnk` launches the same entry point without a console.

## Snippets

The activation hotkey toggles one recording:

```text
Listening -> Transcribing -> clipboard
```

While listening, the status-card `x` discards the recording. During
transcription, it dismisses the card and suppresses clipboard and sound output
without cancelling the archive/transcription work.

The newest five transcripts rotate through `Snippets/snippet_1.md` to
`snippet_5.md`. Every valid raw recording is also retained without rotation in
`Snippets/Eleven Labs voice clone/` for voice-clone training.

## Scribe

Choose **Start Scribe** from the tray menu, then choose the meeting shape:

- **One:** microphone audio is labelled with your name; Windows system audio is
  labelled with the other participant's name.
- **Multiple:** microphone audio keeps your name; system audio is diarized by
  ElevenLabs with speaker-library matching enabled.

Scribe records microphone and Windows loopback as separate sources, transcribes
valid streams, and interleaves their word timestamps. An empty source is
skipped; **No Speech Detected** appears only when neither source yields speech.

Completed meetings are written under `Meetings/YY_MM_DD_Subject/`:

```text
transcript.md
summary.md
notes.md           # only when notes were entered
microphone.wav     # only when Save Scribe meeting audio is enabled
meeting-audio.wav  # only when Save Scribe meeting audio is enabled
```

The two WAV files are deliberately not merged.

## Settings

Settings autosave and apply without restarting Koe or interrupting an active
snippet. They control:

- your Scribe name;
- snippet and meeting folders;
- Scribe audio retention;
- the activation hotkey;
- completion sound and snippet-card visibility;
- ElevenLabs vocabulary hints.

Runtime preferences live in ignored `src/config.yaml`. The supported schema is
`src/config_schema.yaml`, and `src/config.yaml.example` is the safe template.

## Repository

```text
assets/       icons and completion sound
scripts/      console-free Windows launcher
src/          application source
tests/        regression tests for the finished behavior
run.py        setup/key validation and tray-process bootstrap
```

The following are runtime data, not source control content:

- `Meetings/`, `Snippets/`, and `logs/` contain private user data;
- `.scribe_temp/` is recreated for in-progress Scribe audio and cleaned after a
  successful attempt;
- `.setup_complete` is a local setup marker;
- Python and test cache directories are disposable.

## Verification

```powershell
python -m pytest -q
$files = @('run.py') + (Get-ChildItem src -Recurse -Filter *.py | ForEach-Object FullName)
python -m py_compile $files
git diff --check
```
