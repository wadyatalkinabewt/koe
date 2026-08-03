import io
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def _config_section(_section):
    return {
        "common": {"language": None, "initial_prompt": "Koe, ElevenLabs, Koe"},
        "elevenlabs": {"keyterms_enabled": True},
    }


def test_request_is_fixed_to_scribe_v2_no_verbatim(monkeypatch):
    import transcription

    monkeypatch.setattr(transcription.ConfigManager, "get_config_section", _config_section)
    data = transcription._elevenlabs_request_data()

    assert ("model_id", "scribe_v2") in data
    assert ("no_verbatim", "true") in data
    assert ("tag_audio_events", "false") in data
    assert ("use_multi_channel", "false") in data
    assert not any(key in ("diarize", "use_speaker_library") for key, _value in data)
    assert not any(key == "language_code" for key, _value in data)
    assert [value for key, value in data if key == "keyterms"] == ["Koe", "ElevenLabs"]


def test_keyterms_are_serialized_as_repeated_multipart_fields(monkeypatch):
    import transcription

    monkeypatch.setattr(transcription.ConfigManager, "get_config_section", _config_section)
    data = transcription._elevenlabs_request_data()
    request = requests.Request(
        "POST",
        "https://example.invalid",
        files={"file": ("audio.wav", io.BytesIO(b"RIFF"), "audio/wav")},
        data=data,
    ).prepare()

    assert isinstance(request.body, bytes)
    assert request.body.count(b'name="keyterms"') == 2
    assert b'name="keyterms"\r\n\r\nKoe\r\n' in request.body
    assert b'name="keyterms"\r\n\r\nElevenLabs\r\n' in request.body


def test_configured_language_is_sent(monkeypatch):
    import transcription

    monkeypatch.setattr(
        transcription.ConfigManager,
        "get_config_section",
        lambda _section: {
            "common": {"language": "en", "initial_prompt": None},
            "elevenlabs": {"keyterms_enabled": False},
        },
    )

    assert ("language_code", "en") in transcription._elevenlabs_request_data()


def test_group_request_enables_diarization_and_speaker_library(monkeypatch):
    import transcription

    monkeypatch.setattr(transcription.ConfigManager, "get_config_section", _config_section)
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

    monkeypatch.setattr(transcription.ConfigManager, "get_config_section", _config_section)
    with pytest.raises(ValueError, match="between 1 and 32"):
        transcription._elevenlabs_request_data(diarize=True, num_speakers=33)


def test_speaker_labels_split_on_changes_and_preserve_library_ids():
    import transcription

    result = {
        "words": [
            {"type": "word", "text": "Hello", "start": 0.0, "end": 0.2, "speaker_id": "speaker_0"},
            {"type": "word", "text": "there", "start": 0.2, "end": 0.4, "speaker_id": "speaker_0"},
            {"type": "word", "text": "Hi", "start": 0.4, "end": 0.6, "speaker_id": "Omar"},
            {"type": "word", "text": "team.", "start": 0.6, "end": 0.8, "speaker_id": "Omar"},
            {"type": "word", "text": "Morning.", "start": 0.8, "end": 1.0, "speaker_id": "speaker_1"},
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
            {"type": "word", "text": "Casey.", "start": 0.2, "end": 0.4},
            {"type": "word", "text": "Hi", "start": 0.5, "end": 0.7},
            {"type": "word", "text": "Alex.", "start": 0.7, "end": 0.9},
        ]
    }

    def resolve(start, _end):
        return "Alex" if start < 0.5 else "Casey"

    assert transcription._segments_from_elevenlabs_words(
        result,
        label="Alex",
        label_resolver=resolve,
    ) == [
        {"start": 0.0, "end": 0.4, "text": "Hello Casey.", "label": "Alex"},
        {"start": 0.5, "end": 0.9, "text": "Hi Alex.", "label": "Casey"},
    ]


def test_known_transcript_substitutions_are_corrected_as_whole_tokens():
    import transcription

    assert transcription.apply_transcript_corrections(
        "Groq, groq, and GROQ heard Taylor, Taylor, and Taylor at "
        "Ack Me, Ack Me, Ack Me, Ack Me, Ack Me, Ack Me, "
        "Ack Me, Ack Me, and Ack Me."
    ) == (
        "Grok, grok, and GROK heard Taylor, Taylor, and Taylor "
        "at Acme, Acme, Acme, Acme, Acme, Acme, "
        "Acme, Acme, and Acme."
    )
    assert transcription.apply_transcript_corrections(
        "GroqCloud, Taylor, AckMeson, AckMeson, and AckMes are different tokens."
    ) == "GroqCloud, Taylor, AckMeson, AckMeson, and AckMes are different tokens."


def test_known_transcript_substitutions_apply_to_scribe_segments():
    import transcription

    result = {
        "words": [
            {"type": "word", "text": "Ask", "start": 0.0, "end": 0.2},
            {"type": "word", "text": "Taylor", "start": 0.2, "end": 0.4},
            {"type": "word", "text": "about", "start": 0.4, "end": 0.6},
            {"type": "word", "text": "Groq.", "start": 0.6, "end": 0.8},
        ]
    }

    assert transcription._segments_from_elevenlabs_words(result, label="Speaker") == [
        {"start": 0.0, "end": 0.8, "text": "Ask Taylor about Grok.", "label": "Speaker"}
    ]


def test_group_file_path_streams_one_request_with_speaker_options(tmp_path, monkeypatch):
    import transcription

    wav_path = tmp_path / "loopback.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(np.zeros(1600, dtype=np.int16).tobytes())

    captured = []
    monkeypatch.setattr(transcription.ConfigManager, "get_config_section", _config_section)
    monkeypatch.setattr(transcription, "_api_key_from_env", lambda *_names: "test-key")

    def fake_post(file_path, data, api_key, timeout):
        captured.append((file_path, data, api_key, timeout))
        return {
            "words": [
                {"type": "word", "text": "Hello.", "start": 0.0, "end": 0.5, "speaker_id": "speaker_0"}
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
    assert ("use_speaker_library", "true") in captured[0][1]
    assert segments == [{"start": 0.0, "end": 0.5, "text": "Hello.", "label": "Speaker 1"}]


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

    assert transcription.transcribe_file_segments(
        wav_path,
        diarize=True,
        use_speaker_library=True,
    ) == []
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
    monkeypatch.setattr(transcription, "_elevenlabs_post_file", lambda *_args, **_kwargs: called.append(True))

    with pytest.raises(ValueError, match="10-hour"):
        transcription.transcribe_file_segments(wav_path, diarize=True)
    assert called == []


def test_snippet_call_path_sends_no_verbatim(monkeypatch):
    import transcription

    captured = {}
    monkeypatch.setattr(transcription.ConfigManager, "get_config_section", _config_section)
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
        lambda audio, sample_rate: transcribed.append((len(audio), sample_rate))
        or "Snippet input works.",
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


def test_snippet_post_processing_applies_known_substitutions():
    import transcription

    assert transcription.post_process_transcription(
        "Ask Taylor whether Groq can help"
    ) == "Ask Taylor whether Grok can help. "


def test_quiet_audio_normalization_is_bounded():
    import transcription

    audio = np.full(1600, 100, dtype=np.int16)
    normalized = transcription._normalize_quiet_audio(audio)

    assert normalized.dtype == np.int16
    assert np.max(np.abs(normalized)) <= 32767
    assert np.mean(np.abs(normalized)) > np.mean(np.abs(audio))
