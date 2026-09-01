import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class _Response:
    status_code = 200
    text = ""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_deepgram_nova3_contract_returns_normalized_diarized_words(monkeypatch):
    from providers import deepgram

    captured = {}
    payload = {
        "metadata": {"diarize_info": {"arch": "v2", "model_uuid": "test"}},
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "Hello there.",
                            "words": [
                                {
                                    "word": "hello",
                                    "punctuated_word": "Hello",
                                    "start": 0.0,
                                    "end": 0.3,
                                    "speaker": 0,
                                },
                                {
                                    "word": "there",
                                    "punctuated_word": "there.",
                                    "start": 0.3,
                                    "end": 0.6,
                                    "speaker": 0,
                                },
                            ],
                        }
                    ]
                }
            ]
        },
    }

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return _Response(payload)

    monkeypatch.setattr(deepgram.requests, "post", fake_post)
    result, error = deepgram.transcribe_stream(
        io.BytesIO(b"RIFF"),
        "deepgram-key",
        language="en",
        diarize=True,
        timeout=900,
    )

    assert error is None
    assert captured["url"] == "https://api.deepgram.com/v1/listen"
    assert captured["headers"]["Authorization"] == "Token deepgram-key"
    assert captured["params"] == {
        "model": "nova-3",
        "smart_format": "true",
        "punctuate": "true",
        "filler_words": "false",
        "language": "en",
        "diarize_model": "latest",
    }
    assert result == {
        "text": "Hello there.",
        "words": [
            {
                "type": "word",
                "text": "Hello",
                "start": 0.0,
                "end": 0.3,
                "speaker_id": "speaker_0",
            },
            {
                "type": "word",
                "text": "there.",
                "start": 0.3,
                "end": 0.6,
                "speaker_id": "speaker_0",
            },
        ],
    }


def test_deepgram_rejects_a_meeting_response_without_requested_diarizer():
    from providers import deepgram

    payload = {
        "metadata": {},
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "Hello.",
                            "words": [
                                {"word": "hello", "start": 0.0, "end": 0.2}
                            ],
                        }
                    ]
                }
            ]
        },
    }

    result, error = deepgram._normalize(payload, diarize=True)

    assert result is None
    assert error == "Deepgram did not run the requested diarizer"


def test_mistral_voxtral_contract_returns_normalized_diarized_segments(monkeypatch):
    from providers import mistral

    captured = {}
    payload = {
        "model": "voxtral-mini-2602",
        "text": "Hello there.",
        "segments": [
            {
                "text": "Hello there.",
                "start": 0.0,
                "end": 0.7,
                "speaker_id": "speaker_0",
            }
        ],
    }

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return _Response(payload)

    monkeypatch.setattr(mistral.requests, "post", fake_post)
    result, error = mistral.transcribe_stream(
        io.BytesIO(b"RIFF"),
        "meeting.wav",
        "mistral-key",
        language="en",
        diarize=True,
        timeout=900,
    )

    assert error is None
    assert captured["url"] == "https://api.mistral.ai/v1/audio/transcriptions"
    assert captured["headers"] == {"Authorization": "Bearer mistral-key"}
    assert captured["data"] == [
        ("model", "voxtral-mini-latest"),
        ("diarize", "true"),
        ("timestamp_granularities", "segment"),
    ]
    assert result == {
        "text": "Hello there.",
        "words": [
            {
                "type": "word",
                "text": "Hello there.",
                "start": 0.0,
                "end": 0.7,
                "speaker_id": "speaker_0",
            }
        ],
    }


def test_mistral_snippet_keeps_configured_language_without_timestamps(monkeypatch):
    from providers import mistral

    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return _Response({"text": "Hello.", "segments": []})

    monkeypatch.setattr(mistral.requests, "post", fake_post)
    result, error = mistral.transcribe_stream(
        io.BytesIO(b"RIFF"),
        "snippet.wav",
        "mistral-key",
        language="en",
        diarize=False,
        timeout=30,
    )

    assert error is None
    assert result == {"text": "Hello.", "words": []}
    assert captured["data"] == [
        ("model", "voxtral-mini-latest"),
        ("language", "en"),
    ]


def test_mistral_rejects_unlabelled_diarized_segments():
    from providers import mistral

    result, error = mistral._normalize(
        {
            "text": "Hello.",
            "segments": [{"text": "Hello.", "start": 0.0, "end": 0.2}],
        },
        diarize=True,
    )

    assert result is None
    assert error == "Mistral diarization returned an unlabelled segment"
