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

    frame_size = 1440
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
    monkeypatch.setattr(result_thread, "_refresh_input_devices", lambda: True)
    monkeypatch.setattr(result_thread.sd, "InputStream", FakeInputStream)
    devices = [
        {
            "name": "WASAPI Mic",
            "hostapi": 0,
            "max_input_channels": 1,
            "default_samplerate": 48000.0,
        }
    ]

    def fake_query_devices(device=None):
        return devices if device is None else devices[device]

    monkeypatch.setattr(
        result_thread.sd,
        "query_hostapis",
        lambda index=None: (
            [{"name": "Windows WASAPI", "default_input_device": 0}]
            if index is None
            else {"name": "Windows WASAPI", "default_input_device": 0}
        ),
    )
    monkeypatch.setattr(result_thread.sd, "query_devices", fake_query_devices)

    audio = thread._record_audio()

    assert audio is not None
    assert len(audio) == frame_size * len(frames)
    assert thread.sample_rate == 48000
    assert np.array_equal(audio[:frame_size], frames[0][:, 0])
    assert np.array_equal(audio[-frame_size:], frames[-1][:, 0])


def test_record_audio_prefers_wasapi_default_at_native_rate(monkeypatch):
    import result_thread
    from result_thread import ResultThread

    thread = ResultThread()
    thread.is_recording = True
    thread.is_running = True

    frame_size = 1440
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
            "default_samplerate": 48000.0,
        },
    ]

    class FakeInputStream:
        def __init__(self, *args, callback, device=None, **kwargs):
            opened_devices.append(device)
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
    monkeypatch.setattr(result_thread, "_refresh_input_devices", lambda: True)
    monkeypatch.setattr(result_thread.sd, "InputStream", FakeInputStream)
    monkeypatch.setattr(
        result_thread.sd,
        "query_hostapis",
        lambda index=None: (
            [
                {"name": "MME", "default_input_device": 0},
                {"name": "Windows WASAPI", "default_input_device": 1},
            ]
            if index is None
            else [
                {"name": "MME", "default_input_device": 0},
                {"name": "Windows WASAPI", "default_input_device": 1},
            ][index]
        ),
    )
    monkeypatch.setattr(result_thread.sd, "query_devices", fake_query_devices)

    audio = thread._record_audio()

    assert opened_devices == [1]
    assert audio is not None
    assert thread.sample_rate == 48000
    assert np.array_equal(audio, np.tile(frame[:, 0], 4))


@pytest.mark.parametrize(
    "new_default",
    [1, 0],
    ids=["switch-to-webcam", "switch-to-pro-x"],
)
def test_record_audio_refreshes_default_device_before_each_snippet(
    monkeypatch,
    new_default,
):
    import result_thread
    from result_thread import ResultThread

    thread = ResultThread()
    thread.is_recording = True
    thread.is_running = True
    stale_default = 0 if new_default == 1 else 1
    state = {"default": stale_default}
    refresh_calls = []
    opened = []
    devices = [
        {
            "name": "Microphone (5- Logitech USB Headset Wireless Gaming Headset)",
            "hostapi": 0,
            "max_input_channels": 1,
            "max_output_channels": 0,
            "default_samplerate": 48000.0,
        },
        {
            "name": "Microphone (HD Pro Webcam C920)",
            "hostapi": 0,
            "max_input_channels": 2,
            "max_output_channels": 0,
            "default_samplerate": 48000.0,
        },
    ]

    class FakeInputStream:
        def __init__(self, *args, callback, device=None, samplerate=None, blocksize=None, **kwargs):
            opened.append((device, samplerate, blocksize))
            self.callback = callback
            self.blocksize = blocksize

        def __enter__(self):
            frame = np.full((self.blocksize, 1), 7, dtype=np.int16)
            for _ in range(4):
                self.callback(frame, self.blocksize, None, None)
            thread.stop_recording()
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_terminate():
        refresh_calls.append("terminate")

    def fake_initialize():
        refresh_calls.append("initialize")
        state["default"] = new_default

    def fake_query_hostapis(index=None):
        hostapi = {
            "name": "Windows WASAPI",
            "default_input_device": state["default"],
        }
        return [hostapi] if index is None else hostapi

    def fake_query_devices(device=None, kind=None):
        if device is None and kind == "input":
            return devices[state["default"]]
        return devices if device is None else devices[device]

    monkeypatch.setattr(result_thread.sd, "_terminate", fake_terminate)
    monkeypatch.setattr(result_thread.sd, "_initialize", fake_initialize)
    monkeypatch.setattr(result_thread.sd, "query_hostapis", fake_query_hostapis)
    monkeypatch.setattr(result_thread.sd, "query_devices", fake_query_devices)
    monkeypatch.setattr(result_thread.sd, "InputStream", FakeInputStream)

    audio = thread._record_audio()

    assert refresh_calls == ["terminate", "initialize"]
    assert opened == [(new_default, 48000, 1440)]
    assert thread.sample_rate == 48000
    assert audio is not None
    assert len(audio) == 4 * 1440


def test_fallback_input_uses_its_own_native_sample_rate(monkeypatch):
    import result_thread
    from result_thread import ResultThread

    thread = ResultThread()
    thread.is_recording = True
    thread.is_running = True
    opened = []
    devices = [
        {
            "name": "Unavailable WASAPI default",
            "hostapi": 0,
            "max_input_channels": 1,
            "max_output_channels": 0,
            "default_samplerate": 48000.0,
        },
        {
            "name": "Available fallback microphone",
            "hostapi": 0,
            "max_input_channels": 1,
            "max_output_channels": 0,
            "default_samplerate": 44100.0,
        },
    ]

    class FakeInputStream:
        def __init__(self, *args, callback, device=None, samplerate=None, blocksize=None, **kwargs):
            opened.append((device, samplerate, blocksize))
            if device == 0:
                raise RuntimeError("default endpoint unavailable")
            self.callback = callback
            self.blocksize = blocksize

        def __enter__(self):
            frame = np.full((self.blocksize, 1), 9, dtype=np.int16)
            for _ in range(4):
                self.callback(frame, self.blocksize, None, None)
            thread.stop_recording()
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_query_devices(device=None, kind=None):
        if device is None and kind == "input":
            return devices[0]
        return devices if device is None else devices[device]

    monkeypatch.setattr(result_thread, "_refresh_input_devices", lambda: True)
    monkeypatch.setattr(
        result_thread.sd,
        "query_hostapis",
        lambda index=None: (
            [{"name": "Windows WASAPI", "default_input_device": 0}]
            if index is None
            else {"name": "Windows WASAPI", "default_input_device": 0}
        ),
    )
    monkeypatch.setattr(result_thread.sd, "query_devices", fake_query_devices)
    monkeypatch.setattr(result_thread.sd, "InputStream", FakeInputStream)

    audio = thread._record_audio()

    assert opened == [(0, 48000, 1440), (1, 44100, 1323)]
    assert thread.sample_rate == 44100
    assert audio is not None
    assert len(audio) == 4 * 1323


def test_c920_uses_its_live_wdm_endpoint(monkeypatch):
    import result_thread
    from result_thread import ResultThread

    thread = ResultThread()
    thread.is_recording = True
    thread.is_running = True
    opened = []
    devices = [
        {
            "name": "Microphone (HD Pro Webcam C920)",
            "hostapi": 0,
            "max_input_channels": 2,
            "max_output_channels": 0,
            "default_samplerate": 48000.0,
        },
        {
            "name": "Microphone (HD Pro Webcam C920)",
            "hostapi": 1,
            "max_input_channels": 2,
            "max_output_channels": 0,
            "default_samplerate": 32000.0,
        },
    ]
    hostapis = [
        {"name": "Windows WASAPI", "default_input_device": 0},
        {"name": "Windows WDM-KS", "default_input_device": 1},
    ]

    class FakeInputStream:
        def __init__(self, *args, callback, device=None, samplerate=None, blocksize=None, **kwargs):
            opened.append((device, samplerate, blocksize))
            self.callback = callback
            self.blocksize = blocksize

        def __enter__(self):
            frame = np.full((self.blocksize, 1), 200, dtype=np.int16)
            for _ in range(4):
                self.callback(frame, self.blocksize, None, None)
            thread.stop_recording()
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_query_devices(device=None, kind=None):
        if device is None and kind == "input":
            return devices[0]
        return devices if device is None else devices[device]

    monkeypatch.setattr(result_thread, "_refresh_input_devices", lambda: True)
    monkeypatch.setattr(
        result_thread.sd,
        "query_hostapis",
        lambda index=None: hostapis if index is None else hostapis[index],
    )
    monkeypatch.setattr(result_thread.sd, "query_devices", fake_query_devices)
    monkeypatch.setattr(result_thread.sd, "InputStream", FakeInputStream)

    audio = thread._record_audio()

    assert opened == [(1, 32000, 960)]
    assert thread.sample_rate == 32000
    assert audio is not None
    assert len(audio) == 4 * 960


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


def test_result_thread_transcribes_without_audio_retention_argument(monkeypatch):
    import result_thread
    from result_thread import ResultThread

    thread = ResultThread()
    captured = []
    results = []
    thread.resultSignal.connect(results.append)

    def record_audio():
        thread.sample_rate = 48000
        return np.ones(4800, dtype=np.int16)

    def transcribe(audio, sample_rate):
        captured.append((len(audio), sample_rate))
        return "captured"

    monkeypatch.setattr(thread, "_record_audio", record_audio)
    monkeypatch.setattr(result_thread, "transcribe", transcribe)

    thread.run()

    assert captured == [(4800, 48000)]
    assert results == ["captured"]


@pytest.mark.parametrize("transcription_outcome", ["empty", "error"])
def test_result_thread_never_persists_snippet_audio(
    tmp_path,
    monkeypatch,
    transcription_outcome,
):
    import result_thread
    from result_thread import ResultThread

    thread = ResultThread()

    def record_audio():
        thread.sample_rate = 48000
        return np.ones(4800, dtype=np.int16)

    def transcribe(_audio, sample_rate):
        assert sample_rate == 48000
        if transcription_outcome == "error":
            raise RuntimeError("expected test failure")
        return ""

    monkeypatch.setattr(result_thread, "_DEBUG_LOG", tmp_path / "debug.log")
    monkeypatch.setattr(thread, "_record_audio", record_audio)
    monkeypatch.setattr(result_thread, "transcribe", transcribe)

    thread.run()

    assert list(tmp_path.rglob("*.wav")) == []


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
