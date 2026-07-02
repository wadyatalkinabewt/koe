import sys
from pathlib import Path

import numpy as np

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
        lambda section: {
            "sample_rate": 16000,
            "silence_duration": 900,
            "recording_mode": "press_to_toggle",
            "sound_device": None,
            "min_duration": 100,
        },
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
        lambda section: {
            "sample_rate": 16000,
            "silence_duration": 900,
            "recording_mode": "press_to_toggle",
            "sound_device": None,
            "min_duration": 1,
        },
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
    assert np.array_equal(audio, frame[:, 0])
