import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_record_audio_flushes_callback_frames_after_stop(monkeypatch):
    import result_thread
    from result_thread import ResultThread

    thread = ResultThread()
    thread.is_recording = True
    thread.is_running = True

    frame_size = 480
    frames = [
        np.full((frame_size, 1), value, dtype=np.int16)
        for value in range(1, 6)
    ]

    class FakeInputStream:
        def __init__(self, *args, callback, **kwargs):
            self.callback = callback

        def __enter__(self):
            for frame in frames:
                self.callback(frame, frame_size, None, None)
            thread.stop_recording()
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        result_thread.ConfigManager,
        "get_config_section",
        lambda section: {"activation_key": "ctrl+shift+space"},
    )
    monkeypatch.setattr(result_thread.sd, "InputStream", FakeInputStream)
    monkeypatch.setattr(result_thread.sd, "query_hostapis", lambda: [])
    monkeypatch.setattr(result_thread.sd, "query_devices", lambda device=None: [])

    audio = thread._record_audio()

    assert audio is not None
    assert len(audio) == frame_size * len(frames)
    assert np.array_equal(audio[:frame_size], frames[0][:, 0])
    assert np.array_equal(audio[-frame_size:], frames[-1][:, 0])


def test_record_audio_falls_back_when_default_input_fails(monkeypatch):
    import result_thread
    from result_thread import ResultThread

    thread = ResultThread()
    thread.is_recording = True
    thread.is_running = True

    frame_size = 480
    frame = np.full((frame_size, 1), 7, dtype=np.int16)
    opened_devices = []
    devices = [
        {
            "name": "Broken MME Mic",
            "hostapi": 0,
            "max_input_channels": 1,
            "max_output_channels": 0,
            "default_samplerate": 44100.0,
        },
        {
            "name": "Working WASAPI Mic",
            "hostapi": 1,
            "max_input_channels": 1,
            "max_output_channels": 0,
            "default_samplerate": 16000.0,
        },
    ]

    class FakeInputStream:
        def __init__(self, *args, callback, device=None, **kwargs):
            opened_devices.append(device)
            if device is None:
                raise RuntimeError("default input failed")
            self.callback = callback

        def __enter__(self):
            for _ in range(4):
                self.callback(frame, frame_size, None, None)
            thread.stop_recording()
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_query_devices(device=None):
        if device is None:
            return devices
        return devices[device]

    monkeypatch.setattr(
        result_thread.ConfigManager,
        "get_config_section",
        lambda section: {"activation_key": "ctrl+shift+space"},
    )
    monkeypatch.setattr(result_thread.sd, "InputStream", FakeInputStream)
    monkeypatch.setattr(
        result_thread.sd,
        "query_hostapis",
        lambda: [{"name": "MME"}, {"name": "Windows WASAPI"}],
    )
    monkeypatch.setattr(result_thread.sd, "query_devices", fake_query_devices)

    audio = thread._record_audio()

    assert opened_devices == [None, 1]
    assert audio is not None
    assert np.array_equal(audio, np.tile(frame[:, 0], 4))


def test_cancel_discards_audio_before_transcription(monkeypatch):
    import result_thread
    from result_thread import ResultThread

    thread = ResultThread()
    cancelled = []
    results = []
    thread.cancelledSignal.connect(lambda: cancelled.append(True))
    thread.resultSignal.connect(results.append)

    def record_then_cancel():
        thread.sample_rate = 16000
        thread.cancel_recording()
        return np.ones(1600, dtype=np.int16)

    monkeypatch.setattr(thread, "_record_audio", record_then_cancel)
    monkeypatch.setattr(
        result_thread,
        "transcribe",
        lambda *_args, **_kwargs: pytest.fail("cancelled audio must not be transcribed"),
    )

    thread.run()

    assert cancelled == [True]
    assert results == []


def test_stop_recording_logs_the_explicit_caller_reason(monkeypatch):
    import result_thread
    from result_thread import ResultThread

    messages = []
    monkeypatch.setattr(result_thread, "_debug", messages.append)
    thread = ResultThread()
    thread.is_recording = True

    thread.stop_recording(reason="hotkey toggle")

    assert thread.is_recording is False
    assert any("hotkey toggle" in message and "was_recording=True" in message for message in messages)
