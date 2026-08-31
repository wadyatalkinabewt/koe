# Koe contributor contract

Koe is a Windows-only tray application with two workflows:

1. Snippets: global toggle hotkey → ElevenLabs Scribe v2 → clipboard and
   five rotating Markdown files.
2. Scribe: microphone plus Windows loopback → one aligned mono upload →
   diarized meeting transcript → optional OpenRouter summary.

## Invariants

- ElevenLabs Scribe v2 is the only speech-to-text backend.
- Every transcription request uses `no_verbatim=true`.
- Snippet formatting may normalize whitespace, punctuation, the paste-friendly
  trailing space, and exact-token corrections from the private `config.yaml`.
  Add corrections only for stable observed substitutions.
- Snippet audio is transient and must never be written to disk. Only rotating
  snippet Markdown and text diagnostics may persist. Scribe meeting audio
  remains independently optional.
- OpenRouter is confined to `src/meeting/summarizer.py` and Scribe
  post-processing. It may propose contextual names only for generic labels;
  exact transcript evidence must pass local validation before both documents
  are relabelled.
- The snippet hotkey is press-to-toggle. ElevenLabs remains the only
  transcription path.
- Settings changes autosave and must not restart Koe or interrupt active audio.
- The snippet status card keeps fixed geometry across Listening/Transcribing.
- Scribe sends one mono meeting file with `use_multi_channel=false`. Never
  reintroduce separate billable mic and loopback transcription requests or
  multichannel billing.
- Every Scribe mode enables diarization with speaker-library matching disabled.
  One-on-one requests cap the expected result at two speakers.
- Online group mic audio remains local attribution evidence only. Its detected
  diarized label maps to the current Settings name.
- Online one-on-one uses that same mic evidence when loopback is active, then
  maps every other diarized voice to the entered participant.
- In-person Scribe treats the microphone as a shared-room source, never forces
  one microphone speaker to the Settings name, and still captures loopback for
  remote callers. Effectively empty loopback is neither uploaded nor retained.
- In-person or speakerphone recordings preserve honest generic speaker labels
  unless contextual analysis validates a name from exact transcript evidence.
- Every successfully decoded ElevenLabs transcription response is immediately
  deleted from ElevenLabs using its returned `transcription_id`.
- OpenRouter meeting analysis must enforce per-request Zero Data Retention.
- Failed Scribe transcription attempts preserve local temporary audio for
  recovery. Successful runs remove temporary audio after document generation
  and any requested durable audio copies are verified.
- Every successful Scribe run writes `transcript.pdf` and `summary.pdf`.
  Markdown copies are written only when **Save Markdown copies** is enabled.
  The transcript PDF puts known names before cleanly renumbered anonymous
  speakers and reserves Koe green for the participant matching the current
  Settings name.
- Meeting notes are appended under a clearly labelled Notes section in the
  transcript PDF and optional transcript Markdown. Never write `notes.md`.
- Both one-on-one Scribe modes capture a separate meeting name and participant
  name; the meeting name drives the folder and document title.
- Scribe selects its four meeting modes inside the main window, remembers the
  last selection, and locks it once recording begins.
- Scribe summaries require Summary, Key Decisions, Topics Discussed, Action
  Items, and Open Questions; incomplete model responses are retried.

## Data boundaries

Source/dev runs keep the operator's complete working state under the repository:

- `<checkout>`: `.env`, `config.yaml`, `.setup_complete`, and local shortcuts.
- `<checkout>\logs` and `<checkout>\.scribe_temp`: diagnostics and temporary audio.
- `<checkout>\Snippets` and `<checkout>\Meetings`: durable output.

Packaged installs retain normal per-user Windows storage:

- `%LOCALAPPDATA%\Koe`: secrets, settings, logs, and Scribe temp audio.
- `%USERPROFILE%\Documents\Koe`: durable snippets and meetings.

Never delete or rewrite either runtime layout without explicit user approval.
Never commit `.env`, `config.yaml`, secrets, recordings, transcripts, or local
paths. Frozen-app upgrades and uninstall must preserve packaged runtime state.

See `docs/DEVELOPMENT.md` for the source map and verification commands.

## Working rules

- Read this file and live source before changing behavior.
- Preserve unrelated user changes and private output data.
- Prefer deleting retired paths over keeping compatibility switches.
- Do not add a VBS/Python installed launcher, second backend, cleanup stage, or
  second configuration source.
- Keep the dark-slate/indigo/coral visual system consistent across all surfaces.
- Use `apply_patch` for source edits. Verify resolved paths before bulk deletion.

## Running-process safety

- Treat a running Koe instance as user-owned live recording equipment. Never
  stop, restart, kill, signal, replace, or otherwise interfere with its process
  while a snippet or Scribe recording may be active.
- A request to stop or restart Koe is always conditional on Koe being idle. It
  never authorizes interrupting an active recording.
- Before any process-affecting action, inspect the live recording state. If an
  active recording cannot be ruled out, leave the process untouched and ask the
  user to confirm that Koe is idle.
- Git, documentation, and repository work must proceed without touching the
  running process. Do not restart Koe merely to validate unrelated changes.
- Snippet audio exists only in memory until recording stops. Terminating Koe
  mid-snippet causes irreversible data loss and is never acceptable.
