import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.fixture(autouse=True)
def no_real_microphone_mute_changes(monkeypatch):
    import result_thread

    monkeypatch.setattr(result_thread, "_unmute_input_device", lambda _device: None)


def test_record_audio_flushes_callback_frames_after_stop(monkeypatch):
    import result_thread
    from result_thread import ResultThread

    thread = ResultThread()
    thread.is_recording = True
    thread.is_running = True

    frame_size = 1440
    frames = [np.full((frame_size, 1), value, dtype=np.int16) for value in range(1, 6)]

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
            "name": "Microphone (5- Example USB Headset)",
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
        def __init__(
            self,
            *args,
            callback,
            device=None,
            samplerate=None,
            blocksize=None,
            **kwargs,
        ):
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
        def __init__(
            self,
            *args,
            callback,
            device=None,
            samplerate=None,
            blocksize=None,
            **kwargs,
        ):
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


def test_c920_uses_its_wasapi_endpoint_with_auto_convert(monkeypatch):
    import result_thread
    from result_thread import ResultThread

    thread = ResultThread()
    thread.is_recording = True
    thread.is_running = True
    opened = []
    wasapi_calls = []
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

    def fake_wasapi_settings(*, auto_convert=False):
        wasapi_calls.append(auto_convert)
        return ("wasapi", auto_convert)

    class FakeInputStream:
        def __init__(
            self,
            *args,
            callback,
            device=None,
            samplerate=None,
            blocksize=None,
            extra_settings=None,
            **kwargs,
        ):
            opened.append((device, samplerate, blocksize, extra_settings))
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
    monkeypatch.setattr(result_thread.sd, "WasapiSettings", fake_wasapi_settings)
    monkeypatch.setattr(result_thread.sd, "InputStream", FakeInputStream)

    audio = thread._record_audio()

    assert wasapi_calls == [True]
    assert opened == [(0, 48000, 1440, ("wasapi", True))]
    assert thread.sample_rate == 48000
    assert audio is not None
    assert len(audio) == 4 * 1440


def test_cancel_saves_one_recoverable_wav_without_transcription(
    tmp_path, monkeypatch
):
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
    recovery_path = tmp_path / "recoverable-snippet.wav"
    monkeypatch.setattr(result_thread, "snippet_recovery_path", lambda: recovery_path)
    monkeypatch.setattr(
        result_thread,
        "transcribe",
        lambda *_args, **_kwargs: pytest.fail(
            "cancelled audio must not be transcribed"
        ),
    )

    thread.run()

    assert cancelled == [True]
    assert results == []
    assert recovery_path.exists()
    with wave.open(str(recovery_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16000
        assert wav_file.getnframes() == 1600


def test_next_snippet_start_deletes_previous_recovery(tmp_path, monkeypatch):
    import result_thread
    from result_thread import ResultThread

    recovery_path = tmp_path / "recoverable-snippet.wav"
    recovery_path.write_bytes(b"previous cancelled audio")
    observed = []
    thread = ResultThread()

    def record_audio():
        observed.append(recovery_path.exists())
        thread.sample_rate = 16000
        return np.ones(1600, dtype=np.int16)

    monkeypatch.setattr(result_thread, "snippet_recovery_path", lambda: recovery_path)
    monkeypatch.setattr(thread, "_record_audio", record_audio)
    monkeypatch.setattr(result_thread, "transcribe", lambda *_args, **_kwargs: "done")

    thread.run()

    assert observed == [False]
    assert not recovery_path.exists()


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

    def transcribe(audio, sample_rate, *, status_callback):
        captured.append((len(audio), sample_rate))
        status_callback("retrying")
        return "captured"

    monkeypatch.setattr(thread, "_record_audio", record_audio)
    monkeypatch.setattr(result_thread, "transcribe", transcribe)

    thread.run()

    assert captured == [(4800, 48000)]
    assert results == ["captured"]


@pytest.mark.parametrize("transcription_outcome", ["empty", "success"])
def test_successful_snippet_never_persists_audio(
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

    def transcribe(_audio, sample_rate, *, status_callback):
        assert sample_rate == 48000
        return "done" if transcription_outcome == "success" else ""

    monkeypatch.setattr(result_thread, "_DEBUG_LOG", tmp_path / "debug.log")
    monkeypatch.setattr(thread, "_record_audio", record_audio)
    monkeypatch.setattr(result_thread, "transcribe", transcribe)

    thread.run()

    assert list(tmp_path.rglob("*.wav")) == []


@pytest.mark.parametrize("sample_rate", [48000, None])
def test_failed_transcription_saves_exact_audio_and_reports_recovery(
    tmp_path, monkeypatch, sample_rate
):
    import result_thread

    thread = result_thread.ResultThread()
    audio = np.arange(-2400, 2400, dtype=np.int16)
    recovery_path = tmp_path / "recoverable-snippet.wav"
    errors, results, statuses, attempts = [], [], [], []
    thread.errorSignal.connect(errors.append)
    thread.resultSignal.connect(results.append)
    thread.statusSignal.connect(statuses.append)

    def record_audio():
        thread.sample_rate = sample_rate
        return audio

    def transcribe(data, sample_rate, *, status_callback):
        attempts.append((data, sample_rate))
        raise RuntimeError("ElevenLabs HTTP 429: system_busy")

    monkeypatch.setattr(thread, "_record_audio", record_audio)
    monkeypatch.setattr(result_thread, "transcribe", transcribe)
    monkeypatch.setattr(result_thread, "snippet_recovery_path", lambda: recovery_path)

    thread.run()

    assert len(attempts) == 1  # Saving recovery must not make another paid request.
    assert attempts[0][0] is audio
    assert attempts[0][1] == (sample_rate or 16000)
    assert results == [""]
    assert statuses[-1] == "error"
    assert not thread.is_recording
    assert len(errors) == 1
    assert "ElevenLabs HTTP 429: system_busy" in errors[0]
    assert errors[0].startswith(f"Audio saved to {recovery_path}")
    assert "the next Snippet deletes it" in errors[0]
    assert list(tmp_path.glob("*.wav")) == [recovery_path]
    with wave.open(str(recovery_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == (sample_rate or 16000)
        assert wav_file.getnframes() == audio.size
        assert wav_file.readframes(audio.size) == audio.astype("<i2").tobytes()


@pytest.mark.parametrize("save_stage, save_error", [
    ("write", OSError("disk full")),
    ("write", wave.Error("invalid WAV")),
    ("mkdir", PermissionError("logs inaccessible")),
])
def test_recovery_save_failure_preserves_original_error_and_cleans_temp(
    tmp_path, monkeypatch, save_stage, save_error
):
    import result_thread

    thread = result_thread.ResultThread()
    recovery_path = tmp_path / "recoverable-snippet.wav"
    errors, results, statuses = [], [], []
    thread.errorSignal.connect(errors.append)
    thread.resultSignal.connect(results.append)
    thread.statusSignal.connect(statuses.append)
    thread.sample_rate = 48000
    monkeypatch.setattr(thread, "_record_audio", lambda: np.ones(4800, dtype=np.int16))
    monkeypatch.setattr(result_thread, "snippet_recovery_path", lambda: recovery_path)

    def transcribe(*_args, **_kwargs):
        raise RuntimeError("provider unavailable")

    def fail_write(*_args, **_kwargs):
        recovery_path.with_suffix(".tmp.wav").write_bytes(b"incomplete")
        raise save_error

    monkeypatch.setattr(result_thread, "transcribe", transcribe)
    if save_stage == "mkdir":
        def fail_mkdir(*_args, **_kwargs):
            raise save_error

        monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    else:
        monkeypatch.setattr(result_thread.wave, "open", fail_write)

    thread.run()

    assert errors == ["Could not save recovery audio.\nprovider unavailable"]
    assert results == [""]
    assert statuses[-1] == "error"
    assert not thread.is_recording
    assert list(tmp_path.glob("*.wav")) == []


def test_capture_failure_does_not_claim_audio_was_saved(tmp_path, monkeypatch):
    import result_thread

    thread = result_thread.ResultThread()
    errors = []
    thread.errorSignal.connect(errors.append)

    def fail_capture():
        raise RuntimeError("microphone unavailable")

    monkeypatch.setattr(thread, "_record_audio", fail_capture)
    monkeypatch.setattr(
        result_thread, "transcribe",
        lambda *_args, **_kwargs: pytest.fail("capture failed; do not transcribe"),
    )
    thread.run()

    assert errors == ["microphone unavailable"]
    assert list(tmp_path.rglob("*.wav")) == []


def test_recovery_failed_atomic_replace_preserves_previous_file(tmp_path, monkeypatch):
    import result_thread

    recovery_path = tmp_path / "recoverable-snippet.wav"
    recovery_path.write_bytes(b"previous recovery")
    monkeypatch.setattr(result_thread, "snippet_recovery_path", lambda: recovery_path)

    def fail_replace(*_args):
        raise PermissionError("file locked")

    monkeypatch.setattr(result_thread.os, "replace", fail_replace)

    assert result_thread._save_snippet_recovery(np.ones(1600, dtype=np.int16), 16000) is None
    assert recovery_path.read_bytes() == b"previous recovery"
    assert list(tmp_path.glob("*.wav")) == [recovery_path]


def test_stop_recording_logs_the_explicit_caller_reason(monkeypatch):
    import result_thread
    from result_thread import ResultThread

    messages = []
    monkeypatch.setattr(result_thread, "_debug", messages.append)
    thread = ResultThread()
    thread.is_recording = True

    thread.stop_recording(reason="hotkey toggle")

    assert thread.is_recording is False
    assert any(
        "hotkey toggle" in message and "was_recording=True" in message
        for message in messages
    )


@pytest.mark.parametrize("mute_error", [False, True])
def test_selected_microphone_is_unmuted_before_capture(monkeypatch, mute_error):
    import result_thread

    events = []
    thread = result_thread.ResultThread()
    thread.is_recording = True

    class Stream:
        def close(self):
            events.append("close")

        def __enter__(self):
            events.append("capture")
            thread.stop_recording()
            return self

        def __exit__(self, *_args):
            events.append("close")

    def unmute(device):
        events.append(("unmute", device))
        if mute_error:
            raise OSError("denied")

    monkeypatch.setattr(result_thread, "_refresh_input_devices", lambda: True)
    monkeypatch.setattr(result_thread, "_wasapi_default_input_device", lambda: 1)
    monkeypatch.setattr(result_thread, "_preferred_capture_device", lambda _device: 1)
    monkeypatch.setattr(result_thread, "_device_label", str)
    monkeypatch.setattr(result_thread, "_device_host_api", lambda _device: "Windows WASAPI")
    # The fallback actually opened is device 25, not the preferred device 1.
    monkeypatch.setattr(result_thread, "_open_input_stream",
                        lambda *_args: (Stream(), 25, 48000, 1440))
    monkeypatch.setattr(result_thread, "_unmute_input_device", unmute)
    if mute_error:
        with pytest.raises(RuntimeError, match="Could not confirm.*unmuted"):
            thread._record_audio()
        assert events == [("unmute", 25), "close"]
    else:
        thread._record_audio()
        assert events == [("unmute", 25), "capture", "close"]
