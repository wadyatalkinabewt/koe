import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def _write(path: Path, audio: np.ndarray, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(np.asarray(audio, dtype=np.int16).tobytes())


def _write_multichannel(
    path: Path,
    audio: np.ndarray,
    *,
    rate: int,
    channels: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(np.asarray(audio, dtype=np.int16).reshape(-1).tobytes())


def test_capture_diagnostics_cannot_break_recording(monkeypatch):
    from meeting import capture as capture_module

    def broken_print(*_args, **_kwargs):
        raise UnicodeEncodeError("charmap", "→", 0, 1, "unsupported")

    monkeypatch.setattr("builtins.print", broken_print)
    capture_module._log("Recording started -> test")


def test_capture_opens_both_devices_before_starting_either(tmp_path, monkeypatch):
    from meeting import capture as capture_module

    started = []

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def start_stream(self):
            started.append(self.name)

        def stop_stream(self):
            pass

        def close(self):
            pass

    class FakePyAudio:
        def __init__(self):
            self.opens = []

        def get_default_input_device_info(self):
            return {"name": "Mic", "index": 1}

        def get_default_wasapi_loopback(self):
            return {
                "name": "Loopback",
                "index": 2,
                "defaultSampleRate": 48000,
                "maxInputChannels": 2,
            }

        def open(self, **kwargs):
            self.opens.append(kwargs)
            return FakeStream("mic" if kwargs["input_device_index"] == 1 else "loopback")

        def terminate(self):
            pass

    fake = FakePyAudio()
    monkeypatch.setattr(capture_module.pyaudio, "PyAudio", lambda: fake)
    capture = capture_module.AudioCapture(tmp_path)

    assert capture.start() is True
    assert len(fake.opens) == 2
    assert all(call["start"] is False for call in fake.opens)
    assert started == ["loopback", "mic"]

    capture.cleanup()


def test_mono_mix_overlays_sources_without_doubling_duration(tmp_path):
    from meeting.capture import prepare_mono_meeting_mix

    mic = np.zeros(16000, dtype=np.int16)
    loopback = np.zeros(16000, dtype=np.int16)
    mic[:8000] = 1200
    loopback[8000:] = 1800
    mic_path = tmp_path / "mic.wav"
    loopback_path = tmp_path / "loopback.wav"
    mixed_path = tmp_path / "mixed.wav"
    _write(mic_path, mic)
    _write(loopback_path, loopback)

    mic_aligned, loop_aligned, rate = prepare_mono_meeting_mix(
        mic_path,
        loopback_path,
        mixed_path,
    )

    with wave.open(str(mixed_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 16000
        assert wav_file.getnframes() == 16000
    assert rate == 16000
    assert mic_aligned.size == loop_aligned.size == 16000


def test_realistic_48khz_stereo_loopback_still_yields_one_timeline(tmp_path):
    from meeting.capture import prepare_mono_meeting_mix

    mic_path = tmp_path / "mic.wav"
    loopback_path = tmp_path / "loopback.wav"
    mixed_path = tmp_path / "mixed.wav"
    _write(mic_path, np.full(8000, 1200, dtype=np.int16))
    stereo_loopback = np.full((48000, 2), 900, dtype=np.int16)
    _write_multichannel(loopback_path, stereo_loopback, rate=48000, channels=2)

    mic_aligned, loop_aligned, rate = prepare_mono_meeting_mix(
        mic_path,
        loopback_path,
        mixed_path,
    )

    with wave.open(str(mixed_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 16000
        assert wav_file.getnframes() == 16000
    assert rate == 16000
    assert mic_aligned.size == loop_aligned.size == 16000


def test_mixed_wav_is_the_aligned_sum_not_concatenated_sources(tmp_path):
    from meeting.capture import load_wav_as_int16, prepare_mono_meeting_mix

    mic = np.zeros(32000, dtype=np.int16)
    mic[4000:8000] = 1200
    loopback = np.zeros((48000 * 3, 2), dtype=np.int16)
    loopback[48000:72000, :] = 1800
    mic_path = tmp_path / "mic.wav"
    loopback_path = tmp_path / "loopback.wav"
    mixed_path = tmp_path / "mixed.wav"
    _write(mic_path, mic)
    _write_multichannel(loopback_path, loopback, rate=48000, channels=2)

    mic_aligned, loop_aligned, rate = prepare_mono_meeting_mix(
        mic_path,
        loopback_path,
        mixed_path,
    )
    mixed, mixed_rate, channels = load_wav_as_int16(mixed_path)

    assert rate == mixed_rate == 16000
    assert channels == 1
    assert mixed.size == mic_aligned.size == loop_aligned.size == 48000
    expected = mic_aligned.astype(np.int32) + loop_aligned.astype(np.int32)
    assert np.max(np.abs(expected)) < 32767
    np.testing.assert_array_equal(mixed, expected.astype(np.int16))
    assert np.max(np.abs(mixed[:12000])) > 0
    assert np.max(np.abs(mixed[16000:24000])) > 0


def test_microphone_speaker_is_identified_from_original_source_timing():
    from meeting.capture import identify_microphone_speaker

    mic = np.zeros(16000, dtype=np.int16)
    loopback = np.zeros(16000, dtype=np.int16)
    mic[:8000] = 2400
    loopback[8000:] = 2400
    segments = [
        {"start": 0.0, "end": 0.45, "label": "Speaker 1", "text": "Local"},
        {"start": 0.55, "end": 1.0, "label": "Speaker 2", "text": "Remote"},
    ]

    assert identify_microphone_speaker(segments, mic, loopback, 16000) == "Speaker 1"


def test_one_on_one_relabels_host_and_single_remote_participant():
    from meeting.app import MODE_ONE_ON_ONE, _relabel_mixed_segments

    segments = [
        {"start": 0.0, "end": 0.4, "label": "Speaker 1", "text": "Hello"},
        {"start": 0.5, "end": 1.0, "label": "Speaker 2", "text": "Hi"},
    ]
    result = _relabel_mixed_segments(
        segments,
        "Speaker 1",
        "Alex",
        "Casey",
        MODE_ONE_ON_ONE,
    )

    assert [segment["label"] for segment in result] == ["Alex", "Casey"]
