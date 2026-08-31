# Koe

Koe is a Windows tray app for two ElevenLabs Scribe v2 workflows:

- **Snippet:** press `Ctrl+Shift+Space`, speak, and press it again to copy the
  transcript to the clipboard.
- **Scribe:** record a meeting, keep notes, produce a labelled transcript, and
  optionally generate a structured OpenRouter summary.

Every transcription request uses `no_verbatim=true`. ElevenLabs is the single
speech-to-text path.

After Koe safely decodes a successful ElevenLabs response, it immediately
deletes the server-side transcript by its returned `transcription_id`. This
applies independently to every snippet chunk and every Scribe meeting upload.
Deletion failures are retried and recorded in Koe's local diagnostic log
without discarding the transcript already received.

After transcription, Koe applies a deliberately small exact-token correction
map for stable Scribe substitutions that vocabulary hints do not prevent. The
same corrections apply to snippets and Scribe meeting transcripts.

Snippet recordings stay in memory only and are never saved as audio. The
**Save Scribe meeting audio** preference applies only to Scribe meetings.
Successful Scribe runs remove their temporary microphone, loopback, and mixed
WAV files after the documents and any requested durable audio copies are
verified. Failed transcription attempts preserve those temporary WAVs for
local recovery.

Scribe keeps the meeting type in the recording window and remembers the last
selection:

- **Online / One-on-One:** enter a meeting name and the other participant's
  name. Koe diarizes the recording, maps the microphone-aligned voice to the
  current user when loopback is active, and maps the other voice to the named
  participant.
- **Online / Group Meeting:** the microphone is the current user and loopback
  may contain multiple remote participants.
- **In Person / One-on-One:** the microphone is shared by both local speakers.
  Koe keeps diarized generic labels unless the later contextual transcript pass
  has exact evidence for a name. It does not use ElevenLabs speaker-library
  matching or guess the Settings owner from a shared microphone.
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
one diarized upload with speaker-library matching disabled. Online meetings use
synchronized mic/loopback timing locally to map the microphone-aligned voice to
the name in Settings. When a loudspeaker or in-person setup puts every voice on
the shared microphone, Koe preserves readable `Speaker 1`, `Speaker 2`, and so
on instead of forcing the whole recording to the owner.

The existing OpenRouter analysis then asks `google/gemini-3.7-flash` for both
the structured summary and conservative contextual identity proposals for any
remaining generic labels. Every request enforces OpenRouter Zero Data Retention
and fails rather than routing meeting text to a retaining provider endpoint.
Koe accepts only high-confidence proposals backed by
exact transcript excerpts, applies the validated mapping to both documents,
and preserves distinct numbering when several unknown people share one role or
organisation. Ambiguous speakers remain `Speaker N`. This contextual pass does
not inspect voices or replace the separate proposed local speaker-embedding
library in ``.

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
