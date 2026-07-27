# Koe

Koe is a Windows tray app for two ElevenLabs Scribe v2 workflows:

- **Snippet:** press `Ctrl+Shift+Space`, speak, and press it again to copy the
  transcript to the clipboard. Enable **Save snippet audio** in Settings to
  keep successful recordings longer than 15 seconds as timestamped WAV files
  under `Snippets\Audio Files`.
- **Scribe:** record a meeting, keep notes, produce a labelled transcript, and
  optionally generate a structured OpenRouter summary.

Every transcription request uses `no_verbatim=true`. ElevenLabs is the single
speech-to-text path.

Scribe keeps the meeting type in the recording window and remembers the last
selection:

- **Online / One-on-One:** enter a meeting name and the other participant's
  name. Koe diarizes the recording, maps the microphone-aligned voice to the
  current user when loopback is active, and maps the other voice to the named
  participant.
- **Online / Group Meeting:** the microphone is the current user and loopback
  may contain multiple remote participants.
- **In Person / One-on-One:** the microphone is shared by both local speakers.
  Koe uses the speaker library to identify the Settings owner, then safely maps
  every other voice to the entered participant. If the owner is not recognised,
  Koe keeps generic labels instead of guessing.
- **In Person / Group Meeting:** the microphone is shared by any number of
  local speakers. Loopback is still captured for anyone joining by call.

For either in-person mode, an effectively empty loopback track is omitted from
transcription and retention.

## Scribe billing and speaker labels

Koe opens both capture devices before either starts, records microphone and
Windows loopback separately, overlays them into one aligned mono WAV, and sends
that file once with `use_multi_channel=false`. A one-hour meeting therefore
produces one one-hour transcription upload rather than two one-hour channel
uploads.

The original microphone track is never transcribed separately. Every mode uses
one diarized, speaker-library-enabled upload. Online meetings use synchronized
mic/loopback timing locally to map the microphone-aligned voice to the name in
Settings. When a loudspeaker or in-person setup puts every voice on the shared
microphone, Koe relies on the speaker library instead of forcing the whole
recording to the owner. Other speakers keep recognised library names or
readable `Speaker 1`, `Speaker 2`, and so on when unknown.

Every successful Scribe meeting writes polished transcript and summary PDFs.
The transcript PDF lists recognised names first, renumbers remaining anonymous
voices from `Speaker 1`, and consistently assigns Koe green to the participant
matching **Your Name** in Settings.

Enable **Save Markdown copies** in Settings to keep `transcript.md` and
`summary.md` alongside those PDFs. Meeting notes are folded into a clearly
labelled Notes section at the bottom of the transcript PDF and optional
Markdown transcript; Koe does not create a separate notes file.

One-on-one PDF summaries use the meeting name, participant, date, and duration
as a compact header; group summaries use a duration-and-participants panel.
Summaries keep the overview brief, group actions by owner, and visually
distinguish decisions and open questions. If **Save Scribe meeting audio** is
enabled, the original source tracks are also kept with the meeting:

```text
transcript.pdf
summary.pdf
transcript.md       # only when Save Markdown copies is enabled
summary.md          # only when Save Markdown copies is enabled
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
