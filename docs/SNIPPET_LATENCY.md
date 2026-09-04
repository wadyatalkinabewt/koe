# Snippet latency investigation - 2026-09-05

## Finding and change

ElevenLabs word timestamp generation is a major source of the observed delay
for longer snippets. Snippets consume the response's flat `text` field but were
requesting `timestamps_granularity=word` through the shared Scribe request
builder. They now explicitly request `none`. Meeting requests retain `word` for
speaker attribution. Audio format, sample rate, Scribe v2, `no_verbatim=true`,
corrections, transient snippet audio, and background deletion remain intact.

The API documents `none` as a supported timestamp granularity:
<https://elevenlabs.io/docs/api-reference/speech-to-text/convert>.

## Evidence

After background deletion was introduced, a 211.74-second real snippet still
took 19.07 seconds in the transcription request. Deletion separately took 0.79
seconds. This ruled out cleanup as the main remaining delay for that request.

Bounded live API tests used the same 180-second synthetic English recording,
generated locally with Windows SAPI. No user recording was replayed. Each
successfully received test transcript was deleted from ElevenLabs.

| Request | Total request seconds | Socket send seconds |
| --- | ---: | ---: |
| 32 kHz WAV, word timestamps | 20.973 | 1.319 |
| 16 kHz raw PCM, word timestamps | 21.660 | 0.338 |
| 16 kHz WAV, word timestamps | 21.551 | 1.174 |
| 16 kHz raw PCM, no timestamps | 4.101 | 0.260 |
| 32 kHz WAV, no timestamps, confirmation | 7.602 | 0.892 |
| 32 kHz WAV, word timestamps, confirmation control | 24.044 | 1.657 |

The final patched `transcribe_elevenlabs` function returned the same synthetic
recording's text in 7.199 seconds, excluding background cleanup. All five
selected content anchor words were present in both confirmation results and
the implementation result. This is a basic content check, not a comprehensive
accuracy benchmark. Timings vary with provider and network conditions; these
results do not establish why the provider's timestamp path became slower
recently or promise a fixed latency for every recording.

Production request timing logs now include the requested timestamp mode, so
`timestamps=none` confirms that a restarted process is using this fix.

## Verification and rollout

- 139 tests passed, including snippet no-timestamp and meeting word-timestamp
  request contracts, and nonblocking background deletion with failure retries.
- Python compilation and `git diff --check` passed.
- Patched code passed the live synthetic test above; cleanup success was logged.
- After the operator confirmed idle and authorized restart, Koe was restarted
  at 11:59 NZST. The new process owns the shortcut listener and startup stderr
  is empty. A real longer snippet can provide the next field measurement;
  production timing logs identify the new request with `timestamps=none`.
