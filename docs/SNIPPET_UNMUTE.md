# Snippet Windows mute safeguard

Implemented and verified on 2026-10-24 (Pacific/Auckland).

Starting a snippet unmutes the actual selected capture endpoint in Windows
before starting the audio stream. Input gain is unchanged. The endpoint stays
unmuted afterwards. Independent headset mute buttons remain effective.

Fallback microphones use their own mute control. A legacy capture API must
match exactly one WASAPI input by name; ambiguous or unavailable controls fail
with an error instead of starting capture. Scribe behaviour is unchanged.

Verification:

- Focused mute and recording tests: 29 passed, with synthetic COM mute changes.
- Full working-tree suite: 214 passed, 10 skipped.
- Isolated staged snapshot, excluding unrelated local work: 156 passed.
- Source compilation and `git diff --check`: passed.
- Read-only Windows check reached the selected Windows microphone endpoint and
  read its mute state successfully. No live mute change or recording was made.

Activation completed after explicit restart approval and confirmation that the
last recording, transcription, and provider cleanup had finished. The source
process was restarted and the local command server was verified listening.
No test recording was triggered. Future snippets log `Windows microphone mute`
in `logs/debug.log` when they check or clear the Windows mute state.
