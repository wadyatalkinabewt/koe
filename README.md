# Koe

Koe is a Windows tray app for two ElevenLabs Scribe v2 workflows:

- **Snippet:** press `Ctrl+Shift+Space`, speak, and press it again to copy the
  transcript to the clipboard.
- **Scribe:** record a meeting, keep notes, produce a labelled transcript, and
  optionally generate a structured OpenRouter summary.

Every transcription request uses `no_verbatim=true`. ElevenLabs is the single
speech-to-text path.

## Install

Run `Koe-Operator-Setup.exe`. It installs per-user, so administrator access is not
required, and creates two desktop and Start Menu shortcuts:

Because this private build is not commercially code-signed, Windows may show
SmartScreen on first launch. Choose **More info → Run anyway** once.

- **Koe Snippet** starts Koe if needed, then starts or stops a snippet.
- **Koe Scribe** starts Koe if needed, then opens the meeting chooser.

The shortcuts target `Koe.exe` directly. Python and VBS are not required on the
installed computer.

First run asks for the user's name and ElevenLabs API key. The key is validated
through ElevenLabs' user endpoint without consuming transcription credits.
Vocabulary hints start disabled with an empty vocabulary because ElevenLabs
charges extra when keyterm prompting is enabled.

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

If **Save Scribe meeting audio** is enabled, the original source tracks are
kept with the meeting:

```text
transcript.md
summary.md
notes.md           # only when notes were entered
microphone.wav     # only when audio retention is enabled
meeting-audio.wav  # only when audio retention is enabled
```

## Data locations

Code and private data are deliberately separate:

```text
%LOCALAPPDATA%\Programs\Koe\     installed application
%LOCALAPPDATA%\Koe\.env          API keys
%LOCALAPPDATA%\Koe\config.yaml   settings
%LOCALAPPDATA%\Koe\logs\         diagnostic logs
%LOCALAPPDATA%\Koe\scribe-temp\  in-progress Scribe audio
<Windows Documents>\Koe\Snippets\
<Windows Documents>\Koe\Meetings\
```

Koe asks Windows for the current user's Documents location, so workplace or
OneDrive folder redirection is respected.

Uninstalling or upgrading Koe does not remove the API keys, settings,
transcripts, snippets, or meeting audio.

## Development

Koe is developed and tested on Windows 11 with Python 3.13.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe run.py
```

Source runs and packaged runs use the same per-user data locations. Override
them only in tests with `KOE_APPDATA_DIR` and `KOE_DOCUMENTS_DIR`.

## Build the private installer

1. Copy `packaging\private-Operator.env.example` to the ignored
   `packaging\private-Operator.env`.
2. Add the dedicated, spending-capped `OPENROUTER_API_KEY`. Never add an
   ElevenLabs key; first-run setup collects that from the user.
3. Build:

```powershell
.\packaging\build.ps1
```

The only handoff artifact is `dist\Koe-Operator-Setup.exe`. The dedicated
OpenRouter key is embedded in that private installer and copied to `.env` only
when the destination does not already have one, so keep the installer private.

### Managed Windows devices

This private build is not publicly code-signed. A managed device that enables
Defender ASR rule `01443614-cd74-433a-b99e-2ecdc07bfc25` may block both the
installer and `Koe.exe` because a brand-new private executable has no Microsoft
cloud reputation. That is a policy decision, not a malware finding. The safe
resolution is an IT-managed per-rule exclusion for Koe's fully qualified path
or a publicly trusted distribution/signing route; do not disable Defender.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
$files = @('run.py') + (Get-ChildItem src -Recurse -Filter *.py | ForEach-Object FullName)
.\.venv\Scripts\python.exe -m py_compile $files
git diff --check
```
