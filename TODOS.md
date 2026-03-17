# Koe TODOs

## Server-side request timeout / watchdog

**What:** Add a timeout mechanism to `/transcribe_meeting` (and `/transcribe`) that kills hung requests after a configurable duration (e.g., 5 minutes).

**Why:** On 2026-03-17, the server hung for 7 minutes processing a single diarization+transcription request. This held the client-side semaphore, causing 20 subsequent audio chunks to be saved as failed (circuit breaker). The entire meeting's loopback audio was effectively lost. Without a server-side timeout, a single hung request can cascade into total meeting data loss.

**Pros:** Prevents one stuck request from starving all subsequent chunks. Failed chunk retry can recover individual losses, but only if the server becomes responsive again.

**Cons:** Killing a request mid-inference could leave CUDA in a bad state. Need to understand what actually hangs before choosing a kill mechanism (thread interrupt vs process restart vs CUDA context reset).

**Context:** Server logging is being added first (logs/server.log + logs/server_stderr.log) so we can diagnose what the actual failure mode is. The hang could be: pyannote diarization deadlock, CUDA OOM, driver-level GPU hang, or something else entirely. The timeout design depends on which of these it is.

**Depends on / blocked by:** Server file logging must be deployed and capture at least one crash before designing this fix.

---

## Embedding match quality monitoring

**What:** Log a warning when enrolled speakers consistently score below threshold across multiple chunks in the same meeting.

**Why:** On 2026-03-17, Calum's enrolled embedding scored 0.06-0.10 against his own voice over loopback (threshold 0.25) for every chunk. The embedding was enrolled from mic audio, but loopback audio has very different acoustic characteristics after WASAPI capture + resampling + normalization. There was no indication during the meeting that the embedding was useless — it silently fell through to "Speaker 1" every time.

**Pros:** Early detection of stale/incompatible embeddings. Could prompt mid-meeting notification like "Calum's voice hasn't matched in 5 chunks — embedding may be stale." Saves the user from discovering the problem only at the end-of-meeting enrollment dialog.

**Cons:** Low urgency — the enrollment dialog already handles the failure case, just with a delayed feedback loop. Adding mid-meeting warnings could be distracting if the threshold is poorly tuned.

**Context:** The root issue is that mic-enrolled embeddings don't cross-match with loopback audio well. The recommended enrollment path is the post-meeting dialog (which uses loopback-sourced embeddings). This monitoring would catch cases where a previously-good embedding goes stale or was enrolled from the wrong audio source.

**Depends on / blocked by:** Nothing — can be implemented independently. But lower priority than server logging.
