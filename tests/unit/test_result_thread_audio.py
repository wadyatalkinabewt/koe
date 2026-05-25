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

    audio = thread._record_audio()

    assert audio is not None
    assert len(audio) == frame_size * len(frames)
    assert np.array_equal(audio[:frame_size], frames[0][:, 0])
    assert np.array_equal(audio[-frame_size:], frames[-1][:, 0])
