"""Bounded transient-429 recovery shares the existing stalled-request budget."""

import io
import json
import threading
import wave
from email.utils import formatdate

import numpy as np
import pytest
import requests


def response(status=200, code="system_busy", retry_after=None, text="recovered"):
    reply = requests.Response()
    reply.status_code = status
    payload = ({"text": text, "transcription_id": "test-id"} if status == 200 else
               {"detail": {"code": code, "status": code, "type": "rate_limit_error"}})
    reply._content = json.dumps(payload).encode()
    if retry_after is not None:
        reply.headers["Retry-After"] = retry_after
    return reply


@pytest.fixture
def transport(monkeypatch):
    import transcription as tr

    calls, delays, deleted = [], [], []
    monkeypatch.setattr(tr.time, "sleep", delays.append)
    monkeypatch.setattr(tr, "_debug", lambda _message: None)
    monkeypatch.setattr(tr, "_delete_elevenlabs_transcript_in_background",
                        lambda result, _key: deleted.append(result["transcription_id"]))
    monkeypatch.setattr(tr, "_SNIPPET_HEDGE_SLOT", threading.BoundedSemaphore(1))

    def install(outcomes):
        outcomes = iter(outcomes)

        def post(_url, **kwargs):
            calls.append((kwargs["files"]["file"][1].read(), kwargs["data"]))
            outcome = next(outcomes)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(tr.snippet_http, "post", post)

    return tr, install, calls, delays, deleted


@pytest.mark.parametrize("final_success", [False, True])
def test_busy_gets_three_total_identical_uploads(transport, final_success):
    tr, install, calls, delays, deleted = transport
    install([response(429), response(429), response(200 if final_success else 429)])
    states = []
    result, error = tr._elevenlabs_post(
        io.BytesIO(b"whole recording"), [("no_verbatim", "true")], "test", 180,
        status_callback=states.append,
    )
    assert calls == [(b"whole recording", [("no_verbatim", "true")])] * 3
    assert delays == [5.0, 10.0]
    assert states == ["retrying", "retrying"]
    assert bool(error) is not final_success
    assert result == ({"text": "recovered", "transcription_id": "test-id"}
                      if final_success else None)
    assert deleted == (["test-id"] if final_success else [])


@pytest.mark.parametrize("code", ["system_busy", "too_many_concurrent_requests"])
def test_retry_after_is_respected(transport, code):
    tr, install, calls, delays, _deleted = transport
    install([response(429, code, "8"), response(429, code, "2"), response()])
    assert tr._elevenlabs_post(io.BytesIO(b"audio"), [], "test", 180)[1] is None
    assert len(calls) == 3
    assert delays == [8.0, 10.0]  # Never shorter than either server or local delay.


@pytest.mark.parametrize("status, code", [
    (400, "bad_request"), (401, "payment_issue"), (402, "payment_required"),
    (403, "forbidden"), (500, "system_busy"), (429, "quota_exceeded"),
    (429, "unknown"),
])
def test_permanent_or_unknown_errors_do_not_retry(transport, status, code):
    tr, install, calls, delays, _deleted = transport
    install([response(status, code)])
    states = []
    result, error = tr._elevenlabs_post(
        io.BytesIO(b"audio"), [], "test", 180, status_callback=states.append
    )
    assert result is None
    assert error.startswith(f"ElevenLabs HTTP {status}:")
    assert len(calls) == 1
    assert delays == states == []


@pytest.mark.parametrize("error", [requests.ReadTimeout(), requests.ConnectionError()])
def test_ambiguous_transport_failure_is_not_retried(transport, error):
    tr, install, calls, delays, _deleted = transport
    install([error])
    result, error = tr._elevenlabs_post(io.BytesIO(b"audio"), [], "test", 180)
    assert result is None and error
    assert len(calls) == 1
    assert delays == []


def test_malformed_429_is_not_retried(transport):
    tr, install, calls, delays, _deleted = transport
    malformed = response(429)
    malformed._content = b"not JSON"
    install([malformed])
    assert tr._elevenlabs_post(io.BytesIO(b"audio"), [], "test", 180) == (
        None, "ElevenLabs HTTP 429: not JSON"
    )
    assert len(calls) == 1
    assert delays == []


@pytest.mark.parametrize("connect_first, expected_delays", [(True, [10.0]), (False, [5.0])])
def test_connect_and_busy_retries_share_budget(transport, connect_first, expected_delays):
    tr, install, calls, delays, _deleted = transport
    failures = [requests.ConnectTimeout(), response(429)]
    if not connect_first:
        failures.reverse()
    install([*failures, response()])
    assert tr._elevenlabs_post(io.BytesIO(b"audio"), [], "test", 180)[1] is None
    assert len(calls) == 3
    assert delays == expected_delays


@pytest.mark.parametrize("primary_success", [False, True])
def test_hedge_429_waits_for_pending_primary_and_shares_budget(monkeypatch, primary_success):
    import transcription as tr

    primary_release = threading.Event()
    hedge_returned = threading.Event()
    third_started = threading.Event()
    slot = threading.BoundedSemaphore(1)
    calls, results, delays, states = [], [], [], []
    monkeypatch.setattr(tr, "_SNIPPET_HEDGE_SLOT", slot)
    monkeypatch.setattr(tr, "ELEVENLABS_SNIPPET_HEDGE_DELAY", 0.01)
    monkeypatch.setattr(tr, "_debug", lambda _message: None)
    monkeypatch.setattr(tr.time, "sleep", delays.append)

    def post(buffer, _data, _key, _timeout, *, attempt):
        calls.append((attempt, buffer.read()))
        if attempt == 1:
            assert primary_release.wait(3)
            return (tr._SnippetReply(result={"text": "primary"}) if primary_success else
                    tr._SnippetReply(error="busy primary", retry_kind="busy"))
        if attempt == 2:
            hedge_returned.set()
            return tr._SnippetReply(error="busy hedge", retry_kind="busy")
        third_started.set()
        assert primary_release.is_set()
        return tr._SnippetReply(result={"text": "third"})

    monkeypatch.setattr(tr, "_elevenlabs_post_once", post)
    worker = threading.Thread(target=lambda: results.append(tr._elevenlabs_post(
        io.BytesIO(b"same audio"), [], "test", 180, status_callback=states.append
    )))
    worker.start()
    try:
        assert hedge_returned.wait(3)
        assert not third_started.wait(0.05), "429 launched a third overlapping request"
        assert worker.is_alive(), "failed before the primary could return"
    finally:
        primary_release.set()
        worker.join(3)
    assert not worker.is_alive()
    assert results == [({"text": "primary" if primary_success else "third"}, None)]
    assert sorted(calls) == [(n, b"same audio") for n in range(1, 3 if primary_success else 4)]
    assert delays == ([] if primary_success else [10.0])
    assert all(state == "retrying" for state in states)
    assert slot.acquire(timeout=1), "recovery semaphore leaked"
    slot.release()


def test_slow_retry_does_not_get_its_own_hedge(monkeypatch):
    import transcription as tr

    release = threading.Event()
    retry_started = threading.Event()
    calls, results = [], []
    monkeypatch.setattr(tr, "ELEVENLABS_SNIPPET_HEDGE_DELAY", 0.01)
    monkeypatch.setattr(tr.time, "sleep", lambda _seconds: None)

    def post(_buffer, _data, _key, _timeout, *, attempt):
        calls.append(attempt)
        if attempt == 1:
            return tr._SnippetReply(error="busy", retry_kind="busy")
        retry_started.set()
        assert release.wait(3)
        return tr._SnippetReply(result={"text": "done"})

    monkeypatch.setattr(tr, "_elevenlabs_post_once", post)
    worker = threading.Thread(target=lambda: results.append(
        tr._elevenlabs_post(io.BytesIO(b"audio"), [], "test", 180)))
    worker.start()
    try:
        assert retry_started.wait(3)
        # Hold the retry beyond the initial hedge interval without monkeypatching sleep.
        assert not release.wait(0.05)
        assert calls == [1, 2]
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive()
    assert results == [({"text": "done"}, None)]


@pytest.mark.parametrize("header, expected", [
    (None, 0), ("", 0), ("5", 5), ("2.5", 2.5), ("-2", 0),
    ("nan", 0), ("inf", 0), ("nonsense", 0),
    (formatdate(130, usegmt=True), 30), (formatdate(90, usegmt=True), 0),
])
def test_retry_after_parser(monkeypatch, header, expected):
    import transcription as tr

    monkeypatch.setattr(tr.time, "time", lambda: 100)
    assert tr._retry_after_seconds(header) == expected


@pytest.mark.parametrize("final_success", [False, True])
def test_result_thread_retries_only_rejected_chunk_and_recovers_whole_audio(
    transport, tmp_path, monkeypatch, final_success
):
    import result_thread

    tr, install, calls, delays, deleted = transport
    install([response(text="first chunk"), response(429), response(429),
             response(200 if final_success else 429, text="second chunk")])
    monkeypatch.setattr(tr, "_api_key_from_env", lambda *_names: "test-key")
    monkeypatch.setattr(tr, "transcription_provider", lambda: "elevenlabs")
    monkeypatch.setattr(tr, "_chunk_max_samples", lambda _rate: 1600)
    monkeypatch.setattr(tr, "_normalize_quiet_audio", lambda audio: audio)
    stored, diagnostics = [], []
    monkeypatch.setattr(tr, "save_rolling_transcription", stored.append)
    monkeypatch.setattr(tr, "save_transcription_debug", lambda *args: diagnostics.append(args))
    thread = result_thread.ResultThread()
    thread.sample_rate = 16000
    audio = np.arange(-1600, 1600, dtype=np.int16)
    recovery = tmp_path / "recoverable-snippet.wav"
    monkeypatch.setattr(thread, "_record_audio", lambda: audio)
    monkeypatch.setattr(result_thread, "snippet_recovery_path", lambda: recovery)
    results, states, errors = [], [], []
    thread.resultSignal.connect(results.append)
    thread.statusSignal.connect(states.append)
    thread.errorSignal.connect(errors.append)

    thread.run()

    assert len(calls) == 4  # Chunk 1 is not re-billed when chunk 2 is rejected.
    assert calls[0][0] != calls[1][0]
    assert calls[1:] == [calls[1]] * 3
    assert all(("no_verbatim", "true") in data for _audio, data in calls)
    assert delays == [5.0, 10.0]
    assert states[:4] == ["recording", "transcribing", "retrying", "retrying"]
    if final_success:
        assert results == stored == ["first chunk second chunk. "]
        assert len(diagnostics) == 1
        assert len(deleted) == 2
        assert not errors and not recovery.exists()
    else:
        assert results == [""]
        assert stored == diagnostics == []
        assert len(deleted) == 1
        assert states[-1] == "error"
        assert errors[0].startswith(f"Audio saved to {recovery}")
        with wave.open(str(recovery), "rb") as recovered:
            assert recovered.getnframes() == len(audio)
            assert recovered.readframes(len(audio)) == audio.astype("<i2").tobytes()
    assert not list(tmp_path.glob("*.tmp.wav"))
