"""Exercise actual request selection and cleanup with controlled stalled I/O."""

import io
import threading

import numpy as np
import pytest
import requests


@pytest.mark.parametrize("primary_error", [False, True])
def test_stalled_primary_delivers_secondary_without_waiting_and_cleans_both(
    monkeypatch, primary_error
):
    import transcription as tr

    release_primary = threading.Event()
    primary_finished = threading.Event()
    calls = []
    deleted = []
    slot = threading.BoundedSemaphore(1)
    monkeypatch.setattr(tr, "_SNIPPET_HEDGE_SLOT", slot)
    monkeypatch.setattr(tr, "ELEVENLABS_SNIPPET_HEDGE_DELAY", 0.02)
    monkeypatch.setattr(tr, "_debug", lambda _message: None)
    monkeypatch.setattr(tr, "_delete_elevenlabs_transcript_in_background",
                        lambda result, _key: deleted.append(result["transcription_id"]))

    class Response:
        status_code = 200

        def __init__(self, name):
            self.name = name

        def json(self):
            return {"text": self.name, "transcription_id": self.name}

    def post(_url, **kwargs):
        name = threading.current_thread().name
        calls.append((name, kwargs["files"]["file"][1].read(), kwargs["data"],
                      kwargs["timeout"], threading.current_thread().daemon))
        if name == "koe-snippet-request":
            try:
                assert release_primary.wait(2), "test failed to release primary"
                if primary_error:
                    raise requests.ReadTimeout("stalled request")
                return Response("original")
            finally:
                primary_finished.set()
        return Response("recovered")

    monkeypatch.setattr(tr.snippet_http, "post", post)
    try:
        result, error = tr._elevenlabs_post(
            io.BytesIO(b"same audio"), [("no_verbatim", "true")], "test", 180
        )
        assert result["text"] == "recovered"
        assert error is None
        assert not primary_finished.is_set(), "delivery waited for stalled request"
        assert deleted == ["recovered"]
        assert len(calls) == 2
        assert all(call[1] == b"same audio" for call in calls)
        assert all(call[2] == [("no_verbatim", "true")] for call in calls)
        assert all(call[3] == (10.0, 180.0) for call in calls)
        assert all(call[4] is False for call in calls)
        assert not slot.acquire(blocking=False), "pending loser lost its slot"
    finally:
        release_primary.set()
        assert primary_finished.wait(2)
        assert slot.acquire(timeout=2), "recovery slot leaked"
        slot.release()
    assert sorted(deleted) == (["recovered"] if primary_error else ["original", "recovered"])


@pytest.mark.parametrize("secondary_outcome", ["error", "empty"])
def test_failed_or_empty_recovery_does_not_discard_late_primary(monkeypatch, secondary_outcome):
    import transcription as tr

    release_primary = threading.Event()
    monkeypatch.setattr(tr, "_SNIPPET_HEDGE_SLOT", threading.BoundedSemaphore(1))
    monkeypatch.setattr(tr, "ELEVENLABS_SNIPPET_HEDGE_DELAY", 0.02)
    monkeypatch.setattr(tr, "_debug", lambda _message: None)

    def post(_buffer, _data, _key, _timeout, *, attempt):
        if attempt == 1:
            assert release_primary.wait(2)
            return {"text": "original speech"}, None
        release_primary.set()
        return (None, "ElevenLabs timeout") if secondary_outcome == "error" else ({"text": ""}, None)

    monkeypatch.setattr(tr, "_elevenlabs_post_once", post)
    result, error = tr._elevenlabs_post(io.BytesIO(b"audio"), [], "test", 180)
    assert result == {"text": "original speech"}
    assert error is None


def test_normal_snippet_uses_one_request_and_no_recovery_slot(monkeypatch):
    import transcription as tr

    calls = []
    slot = threading.BoundedSemaphore(1)
    monkeypatch.setattr(tr, "_SNIPPET_HEDGE_SLOT", slot)

    def post(*args, **kwargs):
        calls.append(kwargs["attempt"])
        return {"text": "fast"}, None

    monkeypatch.setattr(tr, "_elevenlabs_post_once", post)
    assert tr._elevenlabs_post(io.BytesIO(b"audio"), [], "test", 180) == ({"text": "fast"}, None)
    assert calls == [1]
    assert slot.acquire(blocking=False)
    slot.release()


def test_pending_recovery_limits_background_requests(monkeypatch):
    import transcription as tr

    slot = threading.BoundedSemaphore(1)
    assert slot.acquire(blocking=False)
    release = threading.Event()
    expired = threading.Event()
    monkeypatch.setattr(tr, "_SNIPPET_HEDGE_SLOT", slot)
    monkeypatch.setattr(tr, "ELEVENLABS_SNIPPET_HEDGE_DELAY", 0.02)
    monkeypatch.setattr(tr, "_debug", lambda message: expired.set() if "earlier recovery" in message else None)
    calls = []
    results = []

    def post(*args, **kwargs):
        calls.append(kwargs["attempt"])
        assert release.wait(2)
        return {"text": "finished"}, None

    monkeypatch.setattr(tr, "_elevenlabs_post_once", post)
    worker = threading.Thread(target=lambda: results.append(
        tr._elevenlabs_post(io.BytesIO(b"audio"), [], "test", 180)))
    worker.start()
    try:
        assert expired.wait(2)
        assert calls == [1]
    finally:
        release.set()
        worker.join(2)
        slot.release()
    assert not worker.is_alive()
    assert results == [({"text": "finished"}, None)]


def test_provider_timeout_is_reported_instead_of_silent_empty_success(monkeypatch):
    import transcription as tr

    monkeypatch.setattr(tr, "_elevenlabs_post", lambda *args, **kwargs: (None, "ElevenLabs timeout"))
    with pytest.raises(RuntimeError, match="ElevenLabs timeout"):
        tr._transcribe_elevenlabs_audio(np.ones(1600, dtype=np.int16), [], "test", 16000, 180)


def test_both_stalled_attempts_fail_without_a_third_request(monkeypatch):
    import transcription as tr

    release = threading.Event()
    calls = []
    slot = threading.BoundedSemaphore(1)
    monkeypatch.setattr(tr, "_SNIPPET_HEDGE_SLOT", slot)
    monkeypatch.setattr(tr, "ELEVENLABS_SNIPPET_HEDGE_DELAY", 0.02)
    monkeypatch.setattr(tr, "_debug", lambda _message: None)

    def post(*args, **kwargs):
        calls.append(kwargs["attempt"])
        if kwargs["attempt"] == 1:
            assert release.wait(2)
        else:
            release.set()
        return None, "ElevenLabs timeout"

    monkeypatch.setattr(tr, "_elevenlabs_post_once", post)
    assert tr._elevenlabs_post(io.BytesIO(b"audio"), [], "test", 180) == (None, "ElevenLabs timeout")
    assert sorted(calls) == [1, 2]
    assert slot.acquire(timeout=2)
    slot.release()


@pytest.mark.parametrize("second_error", [None, "ElevenLabs connection timeout"])
def test_connection_timeout_retries_once_without_discarding_audio(monkeypatch, second_error):
    import transcription as tr

    calls = []
    monkeypatch.setattr(tr, "_debug", lambda _message: None)

    def post(buffer, _data, _key, _timeout, *, attempt):
        calls.append((attempt, buffer.read()))
        if attempt == 1:
            return None, "ElevenLabs connection timeout"
        return (None, second_error) if second_error else ({"text": "recovered"}, None)

    monkeypatch.setattr(tr, "_elevenlabs_post_once", post)
    result, error = tr._elevenlabs_post(io.BytesIO(b"all audio"), [], "test", 180)
    assert calls == [(1, b"all audio"), (2, b"all audio")]
    assert error == second_error
    assert result == (None if second_error else {"text": "recovered"})
