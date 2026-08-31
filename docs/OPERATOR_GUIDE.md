# Operator guide

## Snippets

The snippet hotkey is press-to-toggle. A second press stops capture, sends the
audio to ElevenLabs Scribe v2, copies the formatted result to the clipboard,
and rotates five local Markdown snippets. Snippet audio is transient and is not
written to disk.

## Scribe meeting modes

- **Online / One-on-One:** microphone timing identifies the local participant;
  the other diarized voice is mapped to the entered participant.
- **Online / Group:** microphone timing identifies the local participant while
  loopback may contain several remote participants.
- **In Person / One-on-One:** the microphone is a shared-room source. Generic
  labels remain unless exact transcript evidence supports a name.
- **In Person / Group:** the microphone is shared by any number of local
  speakers, while loopback can still capture remote callers.

Koe opens microphone and loopback together, aligns them, mixes one mono meeting
file, and sends a single transcription request with diarization enabled and
speaker-library matching disabled. An effectively empty loopback track is not
uploaded or retained.

## Documents and optional audio

Every successful meeting produces `transcript.pdf` and `summary.pdf`. Optional
settings can also retain Markdown copies and the original meeting audio:

```text
transcript.pdf
summary.pdf
transcript.md       # optional
summary.md          # optional
microphone.wav      # optional
meeting-audio.wav   # optional
```

Notes are appended to the transcript rather than written to a separate file.
Failed transcription attempts preserve temporary audio for recovery. Successful
runs remove temporary audio after document generation and any requested durable
copies have been verified.

## Speaker labels and summaries

Known names are listed before cleanly renumbered anonymous speakers. Online
microphone attribution remains authoritative for the name configured in
Settings. Shared-room recordings keep honest generic labels unless the optional
OpenRouter analysis proposes a name with exact transcript evidence that passes
local validation.

OpenRouter is used only for Scribe post-processing. Requests require a Zero Data
Retention endpoint and must produce Summary, Key Decisions, Topics Discussed,
Action Items, and Open Questions.

## Storage and privacy

Source runs store settings and output in the checkout. Packaged builds use:

- `%LOCALAPPDATA%\Koe` for secrets, settings, logs, custom corrections, and
  temporary Scribe audio;
- `%USERPROFILE%\Documents\Koe` for snippets and meetings.

Every successful ElevenLabs response is deleted from ElevenLabs by its returned
transcription ID. Deletion failures are retried and recorded locally without
discarding the decoded transcript.
