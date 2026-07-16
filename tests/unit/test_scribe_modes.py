import shutil
import sys
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def _write_wav(path: Path, frames: int = 1600, value: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(np.full(frames, value, dtype=np.int16).tobytes())


def test_group_folder_component_uses_underscores():
    from meeting.app import _sanitize

    assert _sanitize("Management Meeting", underscores=True) == "Management_Meeting"
    assert _sanitize("Casey", underscores=False) == "Casey"


def test_unique_labels_puts_Alex_first_and_preserves_library_names():
    from meeting.app import _unique_labels

    segments = [
        {"start": 0.2, "label": "Speaker 1"},
        {"start": 0.4, "label": "Omar"},
        {"start": 0.6, "label": "Speaker 1"},
    ]
    assert _unique_labels(segments, "Alex") == ["Alex", "Speaker 1", "Omar"]


def test_audio_archive_is_all_or_nothing(tmp_path, monkeypatch):
    from meeting.app import _persist_meeting_audio

    source_dir = tmp_path / "source"
    meeting_dir = tmp_path / "meeting"
    mic = source_dir / "mic.wav"
    loopback = source_dir / "loopback.wav"
    _write_wav(mic)
    _write_wav(loopback)
    meeting_dir.mkdir()

    real_copy2 = shutil.copy2
    copy_count = 0

    def fail_second_copy(source, destination):
        nonlocal copy_count
        copy_count += 1
        if copy_count == 2:
            raise OSError("simulated disk failure")
        return real_copy2(source, destination)

    monkeypatch.setattr("meeting.app.shutil.copy2", fail_second_copy)
    with pytest.raises(OSError, match="simulated disk failure"):
        _persist_meeting_audio(mic, loopback, meeting_dir)

    assert mic.exists() and loopback.exists()
    assert not (meeting_dir / "microphone.wav").exists()
    assert not (meeting_dir / "meeting-audio.wav").exists()


def test_group_worker_sends_one_mixed_request_and_saves_original_sources(tmp_path, monkeypatch):
    import transcription
    from meeting import summarizer
    from meeting.app import MODE_GROUP, MeetingWorker

    source_dir = tmp_path / "temp"
    mic = source_dir / "mic.wav"
    loopback = source_dir / "loopback.wav"
    _write_wav(mic)
    _write_wav(loopback)
    captured = {}

    def fake_file_transcription(file_path, **kwargs):
        captured["file_path"] = Path(file_path)
        captured.update(kwargs)
        return [
            {"start": 0.0, "end": 0.5, "text": "Welcome.", "label": "Speaker 1"},
            {"start": 1.1, "end": 1.5, "text": "Morning.", "label": "Omar"},
        ]

    monkeypatch.setattr(transcription, "transcribe_file_segments", fake_file_transcription)
    monkeypatch.setattr("meeting.app.identify_microphone_speaker", lambda *_args: "Speaker 1")
    monkeypatch.setattr(summarizer.SummarizerClient, "summarize", lambda _self, _doc: "# Summary\n\nDone.\n")

    worker = MeetingWorker(
        mic_wav=mic,
        loopback_wav=loopback,
        user_name="Alex",
        meeting_subject="Management Meeting",
        meeting_mode=MODE_GROUP,
        notes_text="Decision made.",
        output_root=tmp_path / "Meetings",
        started_at=datetime(2026, 7, 13, 9, 0),
        save_audio=True,
    )
    worker.run()

    meeting_dir = tmp_path / "Meetings" / "26_07_13_Management_Meeting"
    assert captured == {
        "file_path": source_dir / "meeting-mix.wav",
        "label": "Speaker",
        "diarize": True,
        "use_speaker_library": True,
    }
    assert (meeting_dir / "transcript.md").exists()
    assert (meeting_dir / "summary.md").exists()
    assert (meeting_dir / "microphone.wav").exists()
    assert (meeting_dir / "meeting-audio.wav").exists()
    transcript = (meeting_dir / "transcript.md").read_text(encoding="utf-8")
    assert "Alex" in transcript and "Omar" in transcript
    assert not mic.exists() and not loopback.exists()


def test_group_worker_http_boundary_is_one_mono_non_multichannel_upload(tmp_path, monkeypatch):
    import transcription
    from meeting import summarizer
    from meeting.app import MODE_GROUP, MeetingWorker

    source_dir = tmp_path / "temp"
    mic = source_dir / "mic.wav"
    loopback = source_dir / "loopback.wav"
    _write_wav(mic, frames=3200, value=1800)
    _write_wav(loopback, frames=4800)
    requests_seen = []

    class FakeResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "words": [
                    {
                        "type": "word",
                        "text": "Hello.",
                        "start": 0.0,
                        "end": 0.2,
                        "speaker_id": "speaker_0",
                    }
                ]
            }

    def fake_post(url, *, headers, files, data, timeout):
        _name, audio_file, mime = files["file"]
        original_position = audio_file.tell()
        with wave.open(audio_file, "rb") as wav_file:
            metadata = {
                "channels": wav_file.getnchannels(),
                "rate": wav_file.getframerate(),
                "frames": wav_file.getnframes(),
            }
        audio_file.seek(original_position)
        requests_seen.append(
            {
                "url": url,
                "has_key": bool(headers.get("xi-api-key")),
                "mime": mime,
                "data": list(data),
                **metadata,
            }
        )
        return FakeResponse()

    monkeypatch.setattr(transcription, "_api_key_from_env", lambda *_names: "test-key")
    monkeypatch.setattr(transcription.requests, "post", fake_post)
    monkeypatch.setattr(
        summarizer.SummarizerClient,
        "summarize",
        lambda _self, _doc: "# Summary\n\nDone.\n",
    )
    worker = MeetingWorker(
        mic_wav=mic,
        loopback_wav=loopback,
        user_name="Operator",
        meeting_subject="Management Meeting",
        meeting_mode=MODE_GROUP,
        notes_text="",
        output_root=tmp_path / "Meetings",
        started_at=datetime(2026, 7, 15, 9, 0),
    )

    worker.run()

    assert len(requests_seen) == 1
    request = requests_seen[0]
    assert request["has_key"] is True
    assert request["mime"] == "audio/wav"
    assert request["channels"] == 1
    assert request["rate"] == 16000
    assert request["frames"] == 4800
    assert ("use_multi_channel", "false") in request["data"]
    assert ("diarize", "true") in request["data"]
    assert ("use_speaker_library", "true") in request["data"]
    assert "Operator" in (
        tmp_path / "Meetings" / "26_07_15_Management_Meeting" / "transcript.md"
    ).read_text(encoding="utf-8")


def test_group_worker_continues_when_loopback_stream_is_empty(tmp_path, monkeypatch):
    import transcription
    from meeting import summarizer
    from meeting.app import MODE_GROUP, MeetingWorker

    source_dir = tmp_path / "temp"
    mic = source_dir / "mic.wav"
    loopback = source_dir / "loopback.wav"
    _write_wav(mic)
    _write_wav(loopback, frames=0)
    errors = []
    completed = []

    monkeypatch.setattr(
        transcription,
        "transcribe_file_segments",
        lambda *_args, **_kwargs: [
            {"start": 0.0, "end": 0.5, "text": "Solo update.", "label": "Speaker 1"}
        ],
    )
    monkeypatch.setattr("meeting.app.identify_microphone_speaker", lambda *_args: "Speaker 1")
    monkeypatch.setattr(
        summarizer.SummarizerClient,
        "summarize",
        lambda _self, _doc: "# Summary\n\nDone.\n",
    )

    worker = MeetingWorker(
        mic_wav=mic,
        loopback_wav=loopback,
        user_name="Alex",
        meeting_subject="Management Meeting",
        meeting_mode=MODE_GROUP,
        notes_text="",
        output_root=tmp_path / "Meetings",
        started_at=datetime(2026, 7, 14, 9, 0),
    )
    worker.error_signal.connect(errors.append)
    worker.done_signal.connect(lambda folder, summary: completed.append((folder, summary)))

    worker.run()

    meeting_dir = tmp_path / "Meetings" / "26_07_14_Management_Meeting"
    assert errors == []
    assert len(completed) == 1
    assert (meeting_dir / "transcript.md").exists()
    assert "Solo update." in (meeting_dir / "transcript.md").read_text(encoding="utf-8")
    assert not source_dir.exists()


def test_one_on_one_worker_maps_mixed_speakers_and_does_not_save_audio(tmp_path, monkeypatch):
    import transcription
    from meeting import summarizer
    from meeting.app import MODE_ONE_ON_ONE, MeetingWorker

    source_dir = tmp_path / "temp"
    mic = source_dir / "mic.wav"
    loopback = source_dir / "loopback.wav"
    _write_wav(mic)
    _write_wav(loopback)
    calls = []

    def fake_file_segments(file_path, **kwargs):
        calls.append((Path(file_path), kwargs))
        return [
            {"start": 0.0, "end": 0.5, "text": "Hello.", "label": "Speaker 1"},
            {"start": 0.6, "end": 1.0, "text": "Hi.", "label": "Speaker 2"},
        ]

    monkeypatch.setattr(transcription, "transcribe_file_segments", fake_file_segments)
    monkeypatch.setattr("meeting.app.identify_microphone_speaker", lambda *_args: "Speaker 1")
    monkeypatch.setattr(summarizer.SummarizerClient, "summarize", lambda _self, _doc: "# Summary\n")

    worker = MeetingWorker(
        mic_wav=mic,
        loopback_wav=loopback,
        user_name="Alex",
        meeting_subject="Casey",
        meeting_mode=MODE_ONE_ON_ONE,
        notes_text="",
        output_root=tmp_path / "Meetings",
        started_at=datetime(2026, 7, 13, 9, 0),
        save_audio=False,
    )
    worker.run()

    meeting_dir = tmp_path / "Meetings" / "26_07_13_Casey"
    assert calls == [(
        source_dir / "meeting-mix.wav",
        {"label": "Speaker", "diarize": True, "use_speaker_library": True},
    )]
    assert (meeting_dir / "transcript.md").exists()
    transcript = (meeting_dir / "transcript.md").read_text(encoding="utf-8")
    assert "Alex" in transcript and "Casey" in transcript
    assert not (meeting_dir / "microphone.wav").exists()
    assert not (meeting_dir / "meeting-audio.wav").exists()


def test_no_speech_retry_preserves_reserved_meeting_folder_and_cleans_attempt(tmp_path, monkeypatch):
    import transcription
    from meeting.app import MODE_ONE_ON_ONE, MeetingWorker

    source_dir = tmp_path / "attempt"
    mic = source_dir / "mic.wav"
    loopback = source_dir / "loopback.wav"
    _write_wav(mic)
    _write_wav(loopback)
    reserved_dir = tmp_path / "Meetings" / "26_07_14_Casey"
    reserved_dir.mkdir(parents=True)
    marker = reserved_dir / "existing.txt"
    marker.write_text("keep", encoding="utf-8")
    errors = []

    monkeypatch.setattr(transcription, "transcribe_file_segments", lambda *_args, **_kwargs: [])
    worker = MeetingWorker(
        mic_wav=mic,
        loopback_wav=loopback,
        user_name="Alex",
        meeting_subject="Casey",
        meeting_mode=MODE_ONE_ON_ONE,
        notes_text="",
        output_root=tmp_path / "Meetings",
        started_at=datetime(2026, 7, 14, 9, 0),
        meeting_dir=reserved_dir,
    )
    worker.error_signal.connect(errors.append)

    worker.run()

    assert errors == ["No speech detected in either stream."]
    assert marker.read_text(encoding="utf-8") == "keep"
    assert reserved_dir.exists()
    assert not source_dir.exists()
