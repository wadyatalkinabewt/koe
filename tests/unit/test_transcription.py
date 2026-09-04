import io
import sys
import threading
import wave
from pathlib import Path

import numpy as np
import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def _config_section(*keys):
    if keys == ("model_options",):
        return {"common": {"language": None}}
    return {}


def test_request_is_fixed_to_scribe_v2_no_verbatim(monkeypatch):
    import transcription

    monkeypatch.setattr(
        transcription.ConfigManager, "get_config_section", _config_section
    )
    data = transcription._elevenlabs_request_data()

    assert ("model_id", "scribe_v2") in data
    assert ("no_verbatim", "true") in data
    assert ("tag_audio_events", "false") in data
    assert ("use_multi_channel", "false") in data
    assert ("timestamps_granularity", "word") in data
    assert not any(key in ("diarize", "use_speaker_library") for key, _value in data)
    assert not any(key == "language_code" for key, _value in data)


def test_configured_language_is_sent(monkeypatch):
    import transcription

    monkeypatch.setattr(
        transcription.ConfigManager,
        "get_config_section",
        lambda _section: {"common": {"language": "en"}},
    )

    assert ("language_code", "en") in transcription._elevenlabs_request_data()


def test_group_request_enables_diarization_and_speaker_library(monkeypatch):
    import transcription

    monkeypatch.setattr(
        transcription.ConfigManager, "get_config_section", _config_section
    )
    data = transcription._elevenlabs_request_data(
        diarize=True,
        use_speaker_library=True,
        num_speakers=2,
    )

    assert ("diarize", "true") in data
    assert ("use_speaker_library", "true") in data
    assert ("num_speakers", "2") in data
    assert ("no_verbatim", "true") in data
    assert ("use_multi_channel", "false") in data


def test_invalid_speaker_count_is_rejected(monkeypatch):
    import transcription

    monkeypatch.setattr(
        transcription.ConfigManager, "get_config_section", _config_section
    )
    with pytest.raises(ValueError, match="between 1 and 32"):
        transcription._elevenlabs_request_data(diarize=True, num_speakers=33)


def test_speaker_labels_split_on_changes_and_preserve_library_ids():
    import transcription

    result = {
        "words": [
            {
                "type": "word",
                "text": "Hello",
                "start": 0.0,
                "end": 0.2,
                "speaker_id": "speaker_0",
            },
            {
                "type": "word",
                "text": "there",
                "start": 0.2,
                "end": 0.4,
                "speaker_id": "speaker_0",
            },
            {
                "type": "word",
                "text": "Hi",
                "start": 0.4,
                "end": 0.6,
                "speaker_id": "Omar",
            },
            {
                "type": "word",
                "text": "team.",
                "start": 0.6,
                "end": 0.8,
                "speaker_id": "Omar",
            },
            {
                "type": "word",
                "text": "Morning.",
                "start": 0.8,
                "end": 1.0,
                "speaker_id": "speaker_1",
            },
        ]
    }

    assert transcription._segments_from_elevenlabs_words(
        result,
        label="Speaker",
        use_speaker_labels=True,
    ) == [
        {"start": 0.0, "end": 0.4, "text": "Hello there", "label": "Speaker 1"},
        {"start": 0.4, "end": 0.8, "text": "Hi team.", "label": "Omar"},
        {"start": 0.8, "end": 1.0, "text": "Morning.", "label": "Speaker 2"},
    ]


def test_non_diarized_words_split_when_local_source_label_changes():
    import transcription

    result = {
        "words": [
            {"type": "word", "text": "Hello", "start": 0.0, "end": 0.2},
            {"type": "word", "text": "Jordan.", "start": 0.2, "end": 0.4},
            {"type": "word", "text": "Hi", "start": 0.5, "end": 0.7},
            {"type": "word", "text": "Alex.", "start": 0.7, "end": 0.9},
        ]
    }

    def resolve(start, _end):
        return "Alex" if start < 0.5 else "Jordan"

    assert transcription._segments_from_elevenlabs_words(
        result,
        label="Alex",
        label_resolver=resolve,
    ) == [
        {"start": 0.0, "end": 0.4, "text": "Hello Jordan.", "label": "Alex"},
        {"start": 0.5, "end": 0.9, "text": "Hi Alex.", "label": "Jordan"},
    ]


def test_configured_transcript_substitutions_are_corrected_as_whole_tokens():
    import transcription

    corrections = {
        "ack me": "Acme",
        "north wnd": "Northwind",
    }
    assert transcription.apply_transcript_corrections(
        "Ack me, ack me, and ACK ME met North Wnd, north wnd, and NORTH WND.",
        corrections,
    ) == ("Acme, acme, and ACME met Northwind, northwind, and NORTHWIND.")
    assert (
        transcription.apply_transcript_corrections(
            "Ack myself and North Wnds are different tokens.",
            corrections,
        )
        == "Ack myself and North Wnds are different tokens."
    )


def test_custom_corrections_apply_to_scribe_segments(monkeypatch):
    import transcription

    monkeypatch.setattr(
        transcription.ConfigManager,
        "get_config_section",
        lambda *keys: (
            {"ack me": "Acme", "north wnd": "Northwind"}
            if keys == ("transcription_options", "corrections")
            else {}
        ),
    )
    result = {
        "words": [
            {"type": "word", "text": "Ask", "start": 0.0, "end": 0.2},
            {"type": "word", "text": "Ack", "start": 0.2, "end": 0.3},
            {"type": "word", "text": "me", "start": 0.3, "end": 0.4},
            {"type": "word", "text": "about", "start": 0.4, "end": 0.6},
            {"type": "word", "text": "North", "start": 0.6, "end": 0.7},
            {"type": "word", "text": "Wnd.", "start": 0.7, "end": 0.8},
        ]
    }

    assert transcription._segments_from_elevenlabs_words(result, label="Speaker") == [
        {
            "start": 0.0,
            "end": 0.8,
            "text": "Ask Acme about Northwind.",
            "label": "Speaker",
        }
    ]


def test_invalid_custom_corrections_config_is_ignored(monkeypatch):
    import transcription

    monkeypatch.setattr(
        transcription.ConfigManager,
        "get_config_section",
        lambda *_keys: ["not", "a", "mapping"],
    )

    assert transcription.load_transcript_corrections() == {}


def test_group_file_path_streams_one_request_with_speaker_options(
    tmp_path, monkeypatch
):
    import transcription

    wav_path = tmp_path / "loopback.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(np.zeros(1600, dtype=np.int16).tobytes())

    captured = []
    monkeypatch.setattr(
        transcription.ConfigManager, "get_config_section", _config_section
    )
    monkeypatch.setattr(transcription, "_api_key_from_env", lambda *_names: "test-key")

    def fake_post(file_path, data, api_key, timeout):
        captured.append((file_path, data, api_key, timeout))
        return {
            "words": [
                {
                    "type": "word",
                    "text": "Hello.",
                    "start": 0.0,
                    "end": 0.5,
                    "speaker_id": "speaker_0",
                }
            ]
        }, None

    monkeypatch.setattr(transcription, "_elevenlabs_post_file", fake_post)
    segments = transcription.transcribe_file_segments(
        wav_path,
        diarize=True,
        use_speaker_library=True,
    )

    assert len(captured) == 1
    assert ("diarize", "true") in captured[0][1]
    assert ("timestamps_granularity", "word") in captured[0][1]
    assert ("use_speaker_library", "true") in captured[0][1]
    assert segments == [
        {"start": 0.0, "end": 0.5, "text": "Hello.", "label": "Speaker 1"}
    ]


@pytest.mark.parametrize(
    ("provider", "key_name", "adapter_name"),
    [
        ("deepgram", "DEEPGRAM_API_KEY", "deepgram"),
        ("mistral", "MISTRAL_API_KEY", "mistral"),
    ],
)
def test_alternative_provider_scribe_dispatch_preserves_segment_contract(
    tmp_path, monkeypatch, provider, key_name, adapter_name
):
    import transcription

    wav_path = tmp_path / "meeting.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(np.zeros(1600, dtype=np.int16).tobytes())

    monkeypatch.setattr(
        transcription.ConfigManager,
        "get_config_value",
        lambda *keys: (
            provider
            if keys == ("transcription_options", "provider")
            else None
        ),
    )
    monkeypatch.setattr(
        transcription,
        "_api_key_from_env",
        lambda *names: "provider-key" if names == (key_name,) else "",
    )
    adapter = getattr(transcription, adapter_name)
    captured = {}

    def fake_transcribe_file(file_path, api_key, **kwargs):
        captured.update(file_path=file_path, api_key=api_key, **kwargs)
        return {
            "text": "Hello.",
            "words": [
                {
                    "type": "word",
                    "text": "Hello.",
                    "start": 0.0,
                    "end": 0.5,
                    "speaker_id": "speaker_0",
                }
            ],
        }, None

    monkeypatch.setattr(adapter, "transcribe_file", fake_transcribe_file)

    segments = transcription.transcribe_file_segments(wav_path, diarize=True)

    assert captured["file_path"] == wav_path
    assert captured["api_key"] == "provider-key"
    assert captured["diarize"] is True
    assert segments == [
        {"start": 0.0, "end": 0.5, "text": "Hello.", "label": "Speaker 1"}
    ]


@pytest.mark.parametrize(
    ("provider", "function_name"),
    [("deepgram", "transcribe_deepgram"), ("mistral", "transcribe_mistral")],
)
def test_snippet_dispatch_uses_configured_provider(monkeypatch, provider, function_name):
    import transcription

    monkeypatch.setattr(
        transcription,
        "transcription_provider",
        lambda: provider,
    )
    monkeypatch.setattr(
        transcription,
        function_name,
        lambda audio, sample_rate: f"{provider} result",
    )
    monkeypatch.setattr(transcription, "save_rolling_transcription", lambda _text: None)
    monkeypatch.setattr(transcription, "save_transcription_debug", lambda *_args: None)

    result = transcription.transcribe(np.ones(1600, dtype=np.int16), 16000)

    assert result == f"{provider} result. "


def test_group_upload_retries_wrapped_write_timeout_from_byte_zero(
    tmp_path, monkeypatch
):
    import transcription

    wav_path = tmp_path / "meeting-mix.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(np.full(1600, 900, dtype=np.int16).tobytes())

    positions = []
    delays = []

    class FakeResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"words": []}

    def fake_post(_url, *, headers, files, data, timeout):
        _name, audio_file, _mime = files["file"]
        positions.append(audio_file.tell())
        audio_file.read(16)
        if len(positions) < 3:
            raise requests.ConnectionError(
                "('Connection aborted.', TimeoutError('The write operation timed out'))"
            )
        return FakeResponse()

    monkeypatch.setattr(transcription.requests, "post", fake_post)
    monkeypatch.setattr(transcription.time, "sleep", delays.append)

    result, error = transcription._elevenlabs_post_file(
        wav_path,
        [("model_id", "scribe_v2")],
        "test-key",
        timeout=900,
    )

    assert error is None
    assert result == {"words": []}
    assert positions == [0, 0, 0]
    assert delays == [2.0, 5.0]


@pytest.mark.parametrize("delete_status", [200, 503])
def test_snippet_returns_while_background_deletion_is_blocked(monkeypatch, delete_status):
    import transcription

    events = []
    deleted = {}
    delete_started = threading.Event()
    release_delete = threading.Event()
    delete_finished = threading.Event()
    messages = []
    monkeypatch.setattr(transcription, "_debug", messages.append)
    monkeypatch.setattr(transcription.time, "sleep", lambda _delay: None)

    class PostResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            events.append("decoded")
            return {
                "transcription_id": "snippet/transcript id",
                "text": "Received safely.",
            }

    class DeleteResponse:
        status_code = delete_status

    monkeypatch.setattr(
        transcription.requests,
        "post",
        lambda *_args, **_kwargs: PostResponse(),
    )

    def fake_delete(url, *, headers, timeout):
        deleted["worker"] = threading.current_thread()
        delete_started.set()
        release_delete.wait(timeout=5)
        events.append("deleted")
        deleted.update(url=url, headers=headers, timeout=timeout)
        delete_finished.set()
        return DeleteResponse()

    monkeypatch.setattr(transcription.requests, "delete", fake_delete)

    try:
        result, error = transcription._elevenlabs_post(
            io.BytesIO(b"RIFF"),
            [("model_id", "scribe_v2")],
            "test-key",
            timeout=180,
        )

        assert error is None
        assert result["text"] == "Received safely."
        assert delete_started.wait(timeout=2)
        assert not delete_finished.is_set(), "text delivery waited for deletion"
        assert events == ["decoded"]
        assert not deleted["worker"].daemon
    finally:
        release_delete.set()
        if "worker" in deleted:
            deleted["worker"].join(timeout=2)
            assert not deleted["worker"].is_alive()

    attempts = 1 if delete_status == 200 else transcription.ELEVENLABS_DELETE_MAX_ATTEMPTS
    assert events == ["decoded"] + ["deleted"] * attempts
    assert deleted["url"].endswith("/snippet%2Ftranscript%20id")
    assert deleted["headers"] == {"xi-api-key": "test-key"}
    assert deleted["timeout"] == transcription.ELEVENLABS_DELETE_TIMEOUT
    assert any("snippet request completed in" in message for message in messages)
    assert any("background deletion finished in" in message for message in messages)


def test_meeting_response_is_deleted_after_successful_receipt(tmp_path, monkeypatch):
    import transcription

    wav_path = tmp_path / "meeting-mix.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(np.zeros(1600, dtype=np.int16).tobytes())

    events = []

    class PostResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            events.append("decoded")
            return {"transcription_id": "meeting-id", "words": []}

    class DeleteResponse:
        status_code = 204

    monkeypatch.setattr(
        transcription.requests,
        "post",
        lambda *_args, **_kwargs: PostResponse(),
    )
    monkeypatch.setattr(
        transcription.requests,
        "delete",
        lambda *_args, **_kwargs: events.append("deleted") or DeleteResponse(),
    )

    result, error = transcription._elevenlabs_post_file(
        wav_path,
        [("model_id", "scribe_v2")],
        "test-key",
        timeout=900,
    )

    assert error is None
    assert result == {"transcription_id": "meeting-id", "words": []}
    assert events == ["decoded", "deleted"]


def test_invalid_transcription_response_is_not_deleted(monkeypatch):
    import transcription

    class PostResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            raise ValueError("invalid JSON")

    monkeypatch.setattr(
        transcription.requests,
        "post",
        lambda *_args, **_kwargs: PostResponse(),
    )
    monkeypatch.setattr(
        transcription.requests,
        "delete",
        lambda *_args, **_kwargs: pytest.fail(
            "deletion requires a successfully decoded transcript ID"
        ),
    )

    result, error = transcription._elevenlabs_post(
        io.BytesIO(b"RIFF"),
        [("model_id", "scribe_v2")],
        "test-key",
        timeout=180,
    )

    assert result is None
    assert error == "ElevenLabs returned an invalid JSON response"


def test_transcript_deletion_retries_a_transient_server_failure(monkeypatch):
    import transcription

    statuses = iter((503, 200))
    delays = []

    class DeleteResponse:
        def __init__(self, status_code):
            self.status_code = status_code

    monkeypatch.setattr(
        transcription.requests,
        "delete",
        lambda *_args, **_kwargs: DeleteResponse(next(statuses)),
    )
    monkeypatch.setattr(transcription.time, "sleep", delays.append)

    assert (
        transcription._delete_elevenlabs_transcript(
            {"transcription_id": "retry-id"},
            "test-key",
        )
        is True
    )
    assert delays == [1.0]


def test_group_upload_timeout_scales_for_very_slow_connections(tmp_path, monkeypatch):
    import transcription

    wav_path = tmp_path / "meeting-mix.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(np.zeros(1600, dtype=np.int16).tobytes())

    captured = {}

    class FakeResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"words": []}

    def fake_post(_url, *, headers, files, data, timeout):
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(transcription.requests, "post", fake_post)
    monkeypatch.setattr(transcription, "MIN_MEETING_UPLOAD_BYTES_PER_SECOND", 1)

    result, error = transcription._elevenlabs_post_file(
        wav_path,
        [("model_id", "scribe_v2")],
        "test-key",
        timeout=900,
    )

    expected_upload_timeout = (
        wav_path.stat().st_size + transcription.MEETING_UPLOAD_TIMEOUT_MARGIN
    )
    assert error is None
    assert result == {"words": []}
    assert captured["timeout"] == (expected_upload_timeout, 900.0)


def test_group_upload_does_not_retry_ambiguous_read_timeout(tmp_path, monkeypatch):
    import transcription

    wav_path = tmp_path / "meeting-mix.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(np.zeros(1600, dtype=np.int16).tobytes())

    calls = []

    def fake_post(*_args, **_kwargs):
        calls.append(True)
        raise requests.ReadTimeout("response timed out")

    monkeypatch.setattr(transcription.requests, "post", fake_post)
    monkeypatch.setattr(
        transcription.time,
        "sleep",
        lambda _delay: pytest.fail("read timeouts must not be retried"),
    )

    result, error = transcription._elevenlabs_post_file(
        wav_path,
        [("model_id", "scribe_v2")],
        "test-key",
        timeout=900,
    )

    assert result is None
    assert error == "ElevenLabs request error: response timed out"
    assert calls == [True]


def test_group_empty_stream_is_skipped_before_upload(tmp_path, monkeypatch):
    import transcription

    wav_path = tmp_path / "loopback.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(48000)

    called = []
    monkeypatch.setattr(
        transcription,
        "_elevenlabs_post_file",
        lambda *_args, **_kwargs: called.append(True),
    )

    assert (
        transcription.transcribe_file_segments(
            wav_path,
            diarize=True,
            use_speaker_library=True,
        )
        == []
    )
    assert called == []


def test_group_file_limit_failure_happens_before_upload(tmp_path, monkeypatch):
    import transcription

    wav_path = tmp_path / "loopback.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 160)

    monkeypatch.setattr(
        transcription,
        "_wav_duration_seconds",
        lambda _path: transcription.MAX_ELEVENLABS_DURATION_SECONDS + 1,
    )
    called = []
    monkeypatch.setattr(
        transcription,
        "_elevenlabs_post_file",
        lambda *_args, **_kwargs: called.append(True),
    )

    with pytest.raises(ValueError, match="10-hour"):
        transcription.transcribe_file_segments(wav_path, diarize=True)
    assert called == []


def test_snippet_call_path_sends_no_verbatim(monkeypatch):
    import transcription

    captured = {}
    monkeypatch.setattr(
        transcription.ConfigManager, "get_config_section", _config_section
    )
    monkeypatch.setattr(transcription, "_api_key_from_env", lambda *_names: "test-key")
    monkeypatch.setattr(transcription, "_normalize_quiet_audio", lambda audio: audio)
    monkeypatch.setattr(transcription, "save_rolling_transcription", lambda _text: None)
    monkeypatch.setattr(transcription, "save_transcription_debug", lambda *_args: None)

    def fake_post(_buffer, data, _api_key, timeout):
        captured["data"] = data
        captured["timeout"] = timeout
        return {"text": "Clear result"}, None

    monkeypatch.setattr(transcription, "_elevenlabs_post", fake_post)
    result = transcription.transcribe(np.ones(1600, dtype=np.int16), sample_rate=16000)

    assert result == "Clear result. "
    assert ("no_verbatim", "true") in captured["data"]
    assert ("timestamps_granularity", "none") in captured["data"]
    assert ("file_format", "other") in captured["data"]


def test_snippet_is_transcribed_without_creating_audio_storage(tmp_path, monkeypatch):
    import transcription

    snippets_dir = tmp_path / "Snippets"
    transcribed = []
    monkeypatch.setattr(
        transcription.ConfigManager,
        "get_config_value",
        lambda *keys: (
            str(snippets_dir) if tuple(keys) == ("misc", "snippets_folder") else None
        ),
    )
    monkeypatch.setattr(
        transcription,
        "transcribe_elevenlabs",
        lambda audio, sample_rate: (
            transcribed.append((len(audio), sample_rate)) or "Snippet input works."
        ),
    )
    monkeypatch.setattr(transcription, "save_rolling_transcription", lambda _text: None)
    monkeypatch.setattr(transcription, "save_transcription_debug", lambda *_args: None)

    audio = np.ones(6 * 48000, dtype=np.int16)
    result = transcription.transcribe(audio, sample_rate=48000)

    assert result.strip() == "Snippet input works."
    assert transcribed == [(len(audio), 48000)]
    assert not (snippets_dir / "Audio Files").exists()


def test_local_formatting_preserves_spoken_words():
    from utils import TextProcessor

    spoken = "Um I I think, you know, this is fine"
    assert TextProcessor.process(spoken, add_trailing_space=True) == (
        "Um I I think, you know, this is fine. "
    )


def test_snippet_post_processing_applies_custom_corrections(monkeypatch):
    import transcription

    monkeypatch.setattr(
        transcription.ConfigManager,
        "get_config_section",
        lambda *keys: (
            {"ack me": "Acme", "north wnd": "Northwind"}
            if keys == ("transcription_options", "corrections")
            else {}
        ),
    )
    assert (
        transcription.post_process_transcription(
            "Ask Ack me whether North Wnd can help"
        )
        == "Ask Acme whether Northwind can help. "
    )


def test_quiet_audio_normalization_is_bounded():
    import transcription

    audio = np.full(1600, 100, dtype=np.int16)
    normalized = transcription._normalize_quiet_audio(audio)

    assert normalized.dtype == np.int16
    assert np.max(np.abs(normalized)) <= 32767
    assert np.mean(np.abs(normalized)) > np.mean(np.abs(audio))
