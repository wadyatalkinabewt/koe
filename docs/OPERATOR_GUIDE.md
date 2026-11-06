# Operator guide

## Snippets

The snippet hotkey is press-to-toggle. A second press stops capture, sends the
audio to the transcription provider selected during setup, copies the formatted
result to the clipboard, and rotates five local Markdown snippets. Normal
Snippet audio is transient. Cancelling from the Listening card atomically saves
one `recoverable-snippet.wav` in Koe's logs folder without transcribing it. The
next Snippet start deletes that recovery file before recording, so recover or
copy it before starting another Snippet.

Starting a Snippet clears Windows mute on the microphone Koe actually opens,
before audio capture starts. It leaves that microphone unmuted afterwards,
including for other apps using it; input volume is unchanged. If Koe cannot
identify or unmute that endpoint, it reports an error instead of recording.
A headset's independent physical mute button can still block audio even when
Windows reports the microphone as unmuted. Scribe does not change mute settings.

During transcription, the card keeps the recorded duration visible without
counting transcription time. After 15 seconds its label changes to **Waiting**.
This indicates a slow request, not confirmed provider progress. Dismissing this
card hides it and suppresses clipboard copying and the completion sound; a
successful response still saves the text to the rotating Snippets files. The
request continues in the background.

ElevenLabs snippets recover from a request that remains pending for 15 seconds
by starting at most one additional request to ElevenLabs. The first nonempty
successful result is used once; the late response is also deleted from the
provider. If the recovery fails, Koe can still use the original response. This
can incur a second transcription charge for a slow snippet, and both requests
can still be slow. Another recovery cannot start until the previous pair has
finished. Scribe meetings keep their single-upload behaviour.

Connection establishment has a 10-second timeout and one retry when it times
out before upload. An eventual request failure is reported as a failure rather
than a successful empty snippet. Timing diagnostics separate connection,
upload, and waiting for response headers; the last stage still combines network
delay with provider processing. Normal process exit lets pending requests and
remote cleanup finish.

## Transcription provider setup

First-run setup offers ElevenLabs Scribe v2, Deepgram Nova-3, and Mistral
Voxtral Mini Transcribe 2. The API-key link and validation request change with
the selected provider, and Koe saves the selection only after that provider's
speech-to-text endpoint accepts the key.

- [ElevenLabs API keys](https://elevenlabs.io/app/api/api-keys): enable only
  **Speech to Text -> Access**. User, History, and administrative access are not
  required.
- [Deepgram API keys](https://developers.deepgram.com/docs/create-additional-api-keys)
- [Mistral audio transcription](https://docs.mistral.ai/studio/audio/speech_to_text/offline_transcription)

The same selected adapter handles both Snippet and Scribe. An OpenRouter key is
separate and optional; it enables meeting summaries and conservative contextual
speaker resolution, not transcription.

## Scribe meeting modes

Starting Scribe first opens a mandatory two-step chooser with no remembered
default. First select Online or In person, then select 2 participants or 3+
participants. The second step includes Back. The chosen mode appears as quiet
read-only context beneath the Scribe title; close Scribe and reopen it to choose
a different mode.

- **Online / One-on-One:** microphone timing identifies the local participant;
  the other diarized voice is mapped to the entered participant.
- **Online / Group:** microphone timing identifies the local participant while
  loopback may contain several remote participants.
- **In Person / One-on-One:** the microphone is a shared-room source. Generic
  labels remain unless exact transcript evidence supports a name.
- **In Person / Group:** the microphone is shared by any number of local
  speakers, while loopback can still capture remote callers.

Koe opens microphone and loopback together, aligns them, mixes one mono meeting
file, and sends one transcription request through the selected adapter. Each
adapter requests its native diarized output; ElevenLabs additionally disables
speaker-library matching. An effectively empty loopback track is not uploaded
or retained.

## Documents and optional audio

Every successful transcription produces `transcript.pdf` and `transcript.md`.
A successful OpenRouter post-processing request also produces `summary.pdf` and
`summary.md`. The original meeting audio remains optional:

```text
transcript.pdf
summary.pdf          # only after successful OpenRouter post-processing
transcript.md       # optional
summary.md          # optional; only when summary.pdf exists
microphone.wav      # optional
meeting-audio.wav   # optional
```

Notes are appended to the transcript rather than written to a separate file.
Failed transcription attempts preserve temporary audio for recovery. Successful
runs remove temporary audio after document generation and any requested durable
copies have been verified.

If `OPENROUTER_API_KEY` is absent, Scribe deliberately completes in
transcript-only mode: it does not call OpenRouter or create summary files. If
OpenRouter returns an error, Scribe records the diagnostic locally, keeps the
completed transcript, and creates no `summary.pdf` or `summary.md` containing
the error. This is a completed transcription, so normal temporary-audio cleanup
still applies.

On completion, **Summary** appears only when `summary.pdf` was created and
**Transcript** opens `transcript.pdf`. For transcript-only completion,
**Transcript** is the sole primary action. A provider failure is shown as
**Summary unavailable** without putting the raw provider diagnostic in a
meeting document. If an available PDF is missing and its optional Markdown copy
exists, the corresponding action opens the Markdown document instead.

## Speaker labels and summaries

Known names are listed before cleanly renumbered anonymous speakers. Online
microphone attribution remains authoritative for the name configured in
Settings. Shared-room recordings keep honest generic labels unless the optional
OpenRouter analysis proposes a name with exact transcript evidence that passes
local validation.

OpenRouter is used only for Scribe post-processing. Requests require a Zero Data
Retention endpoint and must produce Summary, Key Decisions, Topics Discussed,
Action Items, and Open Questions. Missing configuration and provider failures
never turn diagnostic text into a summary deliverable.

## Transcribing an existing recording

Choose **Transcribe recording…** in the tray menu. Pick a file, name the other
party, and optionally ask for a summary. Your own name comes from Settings, so
it is never asked for twice. Any container ffmpeg can decode works; the
extension is never trusted for format detection.

When the recording keeps each party on its own stereo channel, speakers are
attributed from channel energy rather than guessed by diarization. That is
deterministic, costs no diarization work, and holds for the whole file instead
of being decided per segment. If the channels turn out to carry the same
signal, Koe says so and falls back to diarization.

Two parties sharing one channel cannot be separated this way; use Scribe's
diarized modes for those recordings.

Output is written to `Documents\Koe\Transcripts\<folder>`, named the same way
Scribe names meetings: the recording date and the other party, for example
`26_11_06_Amy`, or `26_11_06_0210385860` when the caller was unresolved (NZ
`+64` numbers are shown with a leading `0`). A
second call with the same person on the same day gets `_2` appended.

Each folder records which audio file produced it, so transcribing the same file
again reuses that folder instead of calling the provider again.

The same work is available headlessly, without Qt:

```python
from recording_transcript import transcribe_recording
result = transcribe_recording("C:/path/recording.ogg", generate_summary=False)
print(result.output_dir)
```

The transcript header records the call direction (`Outgoing call` or
`Incoming call`) when the filename carries it. Direction is deliberately not
part of the folder name.

`ffmpeg` and `ffprobe` must be on `PATH`.

## Storage and privacy

Settings, logs, custom corrections, and temporary audio follow the runtime: a
source run keeps them in the checkout, and a frozen build keeps them in
`%LOCALAPPDATA%\Koe`.

Durable documents always land in your Documents folder:

- `%USERPROFILE%\Documents\Koe\Scribe` for Scribe meetings;
- `%USERPROFILE%\Documents\Koe\Transcripts` for transcribed recordings;
- snippets stay beside a source checkout, or in
  `%USERPROFILE%\Documents\Koe\Snippets` for a frozen build.

Every successful ElevenLabs response is deleted from ElevenLabs by its returned
transcription ID. Deletion failures are retried and recorded locally without
discarding the decoded transcript. Deepgram and Mistral return synchronous
responses without an equivalent deletable transcript ID, so their account
retention terms apply.
