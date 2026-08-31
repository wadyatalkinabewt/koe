import time
import traceback
from datetime import datetime
from queue import Empty, Queue

import numpy as np
import sounddevice as sd
from PyQt5.QtCore import QMutex, QThread, pyqtSignal

from paths import logs_dir
from transcription import transcribe
from utils import ConfigManager

# Debug logging to file with rotation
_DEBUG_LOG = logs_dir() / "debug.log"
_DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
_MAX_LOG_SIZE = 1 * 1024 * 1024  # 1MB
_HOST_API_PRIORITY = {
    "Windows WASAPI": 0,
    "Windows DirectSound": 1,
    "MME": 2,
    "Windows WDM-KS": 3,
}
_WDM_COMPATIBILITY_MIC_NAMES = ("hd pro webcam c920",)


def _rotate_log_if_needed():
    """Rotate debug.log if it exceeds max size."""
    try:
        if _DEBUG_LOG.exists() and _DEBUG_LOG.stat().st_size > _MAX_LOG_SIZE:
            backup = _DEBUG_LOG.with_suffix(".log.1")
            if backup.exists():
                backup.unlink()
            _DEBUG_LOG.rename(backup)
    except OSError:
        pass


def _debug(msg: str):
    """Write debug message to file with timestamp."""
    _rotate_log_if_needed()
    timestamp = datetime.now().strftime("%H:%M:%S")
    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {msg}\n")
    except OSError:
        pass


def _normalise_sound_device(device):
    """Convert numeric device config strings to PortAudio device indexes."""
    if device in (None, "", "null"):
        return None
    if isinstance(device, str):
        try:
            return int(device)
        except ValueError:
            return device
    return device


def _input_device_candidates(preferred_device=None):
    """Return input devices to try, preferring current host defaults."""
    seen = set()
    candidates = []

    def add(device):
        key = ("default", None) if device is None else ("device", str(device))
        if key in seen:
            return
        seen.add(key)
        candidates.append(device)

    preferred_device = _normalise_sound_device(preferred_device)
    if preferred_device is not None:
        add(preferred_device)

    try:
        hostapis = list(sd.query_hostapis())
        devices = list(enumerate(sd.query_devices()))

        # Try each host API's current default before arbitrary endpoints. This
        # keeps a disconnected-but-still-enumerated microphone from beating the
        # webcam (or any other device) Windows has just promoted to default.
        for _api_index, hostapi in sorted(
            enumerate(hostapis),
            key=lambda item: _HOST_API_PRIORITY.get(item[1].get("name", ""), 99),
        ):
            default_input = hostapi.get("default_input_device")
            try:
                default_input = int(default_input)
            except (TypeError, ValueError):
                continue
            if default_input < 0 or default_input >= len(devices):
                continue
            if int(devices[default_input][1].get("max_input_channels") or 0) > 0:
                add(default_input)

        def sort_key(item):
            idx, info = item
            try:
                api_name = hostapis[int(info.get("hostapi", -1))].get("name", "")
            except (IndexError, TypeError, ValueError):
                api_name = ""
            return (
                _HOST_API_PRIORITY.get(api_name, 99),
                not bool(info.get("name")),
                str(info.get("name", "")).lower(),
                idx,
            )

        for idx, info in sorted(devices, key=sort_key):
            if int(info.get("max_input_channels") or 0) > 0:
                add(idx)
    except Exception as e:
        _debug(f"  Could not enumerate fallback input devices: {e}")

    # Retain the PortAudio-wide default as a final compatibility fallback. On
    # Windows this is commonly the legacy MME endpoint, so it must not precede
    # an available WASAPI device.
    add(None)

    return candidates


def _refresh_input_devices() -> bool:
    """Refresh PortAudio's device/default snapshot between snippet streams."""
    terminate = getattr(sd, "_terminate", None)
    initialize = getattr(sd, "_initialize", None)
    if not callable(terminate) or not callable(initialize):
        _debug("  Audio device refresh unavailable; using current PortAudio state")
        return False

    terminated = False
    try:
        # sounddevice initializes PortAudio once at import time. WASAPI default
        # changes made after Koe starts are not visible until that snapshot is
        # rebuilt. A ResultThread calls this before opening its only stream.
        terminate()
        terminated = True
        initialize()
        _debug("  Audio device list refreshed")
        return True
    except Exception as exc:
        _debug(f"  Audio device refresh failed: {exc}")
        if terminated:
            try:
                # Leave sounddevice usable if a transient refresh failed.
                initialize()
                _debug("  PortAudio reinitialized after refresh failure")
            except Exception as recovery_exc:
                _debug(f"  PortAudio recovery failed: {recovery_exc}")
        return False


def _wasapi_default_input_device():
    """Return the Windows WASAPI default capture device index when available."""
    try:
        for hostapi in sd.query_hostapis():
            if hostapi.get("name") != "Windows WASAPI":
                continue
            device = hostapi.get("default_input_device")
            if device is not None and int(device) >= 0:
                return int(device)
    except Exception as e:
        _debug(f"  Could not resolve WASAPI default input: {e}")
    return None


def _device_sample_rate(device, fallback: int = 48000) -> int:
    """Return an endpoint's native/default rate without forcing legacy resampling."""
    try:
        info = (
            sd.query_devices(kind="input")
            if device is None
            else sd.query_devices(device)
        )
        if isinstance(info, dict):
            rate = int(round(float(info.get("default_samplerate") or 0)))
            if rate > 0:
                return rate
    except Exception as e:
        _debug(f"  Could not read sample rate for {_device_label(device)}: {e}")
    return int(fallback)


def _device_label(device):
    if device is None:
        return "default"
    try:
        info = sd.query_devices(device)
        return f"{device} ({info.get('name', 'unknown')})"
    except Exception:
        return str(device)


def _device_host_api(device) -> str:
    try:
        info = sd.query_devices(device)
        hostapi = sd.query_hostapis(info.get("hostapi", -1))
        return str(hostapi.get("name") or "unknown")
    except Exception:
        return "unknown"


def _matching_input_device(reference_device, host_api_name: str):
    """Find the same named input through another Windows host API."""
    try:
        reference = sd.query_devices(reference_device)
        reference_name = " ".join(str(reference.get("name") or "").lower().split())
        if not reference_name:
            return None
        hostapis = list(sd.query_hostapis())
        for index, info in enumerate(sd.query_devices()):
            if int(info.get("max_input_channels") or 0) <= 0:
                continue
            try:
                candidate_host = hostapis[int(info.get("hostapi", -1))].get("name", "")
            except (IndexError, TypeError, ValueError):
                continue
            candidate_name = " ".join(str(info.get("name") or "").lower().split())
            if candidate_host == host_api_name and candidate_name == reference_name:
                return index
    except Exception as exc:
        _debug(f"  Could not resolve {host_api_name} compatibility input: {exc}")
    return None


def _preferred_capture_device(wasapi_default):
    """Resolve the live capture endpoint for the current Windows default."""
    if wasapi_default is None:
        return wasapi_default
    try:
        info = sd.query_devices(wasapi_default)
        name = str(info.get("name") or "").lower()
    except Exception:
        return wasapi_default

    # The C920's shared-mode PortAudio endpoints can open successfully while
    # delivering only digital silence on this Windows driver. Its matching
    # WDM-KS endpoint carries the real microphone signal.
    if any(marker in name for marker in _WDM_COMPATIBILITY_MIC_NAMES):
        compatibility_device = _matching_input_device(
            wasapi_default,
            "Windows WDM-KS",
        )
        if compatibility_device is not None:
            return compatibility_device
    return wasapi_default


def _open_input_stream(sound_device, callback, fallback_rate: int = 48000):
    """Open an input stream at the selected endpoint's native sample rate."""
    last_error = None
    for device in _input_device_candidates(sound_device):
        sample_rate = _device_sample_rate(device, fallback=fallback_rate)
        frame_size = max(1, int(round(sample_rate * 0.03)))
        try:
            stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
                blocksize=frame_size,
                device=device,
                callback=callback,
            )
            _debug(f"  Audio stream selected: {_device_label(device)}")
            return stream, device, sample_rate, frame_size
        except Exception as e:
            last_error = e
            _debug(f"  Audio stream failed for {_device_label(device)}: {e}")

    if last_error:
        raise last_error
    raise RuntimeError("No input audio devices found")


class ResultThread(QThread):
    """
    A thread class for handling audio recording, transcription, and result processing.

    This class manages the entire process of:
    1. Recording audio from the microphone
    2. Detecting speech and silence
    3. Saving the recorded audio as numpy array
    4. Transcribing the audio
    5. Emitting the transcription result

    Signals:
        statusSignal: Emits the current status of the thread (e.g., 'recording', 'transcribing', 'idle')
        resultSignal: Emits the transcription result
    """

    statusSignal = pyqtSignal(str)
    resultSignal = pyqtSignal(str)
    errorSignal = pyqtSignal(str)  # Emits error message for notifications
    cancelledSignal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.is_running = True
        self.sample_rate = None
        self.cancel_requested = False
        self.mutex = QMutex()

    def stop_recording(self, reason: str = "unspecified"):
        """Stop the current recording session."""
        self.mutex.lock()
        was_recording = self.is_recording
        self.is_recording = False
        self.mutex.unlock()
        _debug(
            f"ResultThread.stop_recording(reason={reason!r}, "
            f"was_recording={was_recording})"
        )

    def stop(self):
        """Stop the entire thread execution."""
        _debug("ResultThread.stop() called - setting is_running=False")
        self.mutex.lock()
        self.is_running = False
        self.mutex.unlock()
        # Don't emit statusSignal here - the status window may already be closed
        # by the cancel handler, and emitting to a closed window can crash.
        # Don't use wait() here - it blocks the main thread and prevents
        # signal processing, causing a deadlock when the worker thread
        # tries to emit resultSignal. Let the thread finish naturally.

    def cancel_recording(self):
        """Discard the active capture and exit without archiving or transcription."""
        _debug("ResultThread.cancel_recording() called")
        self.mutex.lock()
        self.cancel_requested = True
        self.is_running = False
        self.is_recording = False
        self.mutex.unlock()

    def run(self):
        """Main execution method for the thread."""
        _debug("ResultThread.run() STARTED")
        audio_data = (
            None  # Initialize before try so except handler can always reference it
        )
        try:
            if not self.is_running:
                _debug("  Early exit: is_running=False")
                if self.cancel_requested:
                    self.cancelledSignal.emit()
                return

            self.mutex.lock()
            self.is_recording = True
            self.mutex.unlock()

            _debug("  Emitting 'recording' status")
            self.statusSignal.emit("recording")
            ConfigManager.console_print("Recording...")
            _debug("  Starting _record_audio()")
            audio_data = self._record_audio()
            _debug(
                f"  _record_audio() returned: {type(audio_data)}, size={audio_data.size if audio_data is not None else 'None'}"
            )

            if self.cancel_requested:
                _debug("  Snippet cancelled: discarding capture before transcription")
                self.cancelledSignal.emit()
                return

            if not self.is_running:
                _debug("  Early exit after recording: is_running=False")
                # Emit empty result so status window gets closed properly
                self.resultSignal.emit("")
                return

            if audio_data is None:
                _debug("  Recording too short, emitting empty result")
                # Recording too short - emit empty result and close
                self.resultSignal.emit("")
                return

            _debug("  Emitting 'transcribing' status")
            self.statusSignal.emit("transcribing")
            ConfigManager.console_print("Transcribing...")

            # Time the transcription process
            _debug("  Starting transcription...")
            start_time = time.time()
            result = transcribe(
                audio_data,
                sample_rate=self.sample_rate or 16000,
            )
            end_time = time.time()

            transcription_time = end_time - start_time
            _debug(
                f"  Transcription done in {transcription_time:.2f}s, result length={len(result)}"
            )
            ConfigManager.console_print(
                f"Transcription completed in {transcription_time:.2f} seconds. Post-processed line: {result}"
            )

            # Always emit result after transcription completes, even if cancelled
            # (the user still deserves the clipboard copy and completion feedback)
            _debug("  Emitting result signal")
            self.resultSignal.emit(result)
            _debug("  Result signal emitted successfully")

        except Exception as e:
            _debug(f"  EXCEPTION: {e}")
            _debug(f"  Traceback: {traceback.format_exc()}")
            traceback.print_exc()

            error_msg = str(e) if str(e) else "Transcription failed"
            self.errorSignal.emit(error_msg)
            self.statusSignal.emit("error")
            self.resultSignal.emit("")
        finally:
            _debug("  Calling stop_recording()")
            self.stop_recording(reason="thread cleanup")
            _debug("ResultThread.run() FINISHED")

    def _record_audio(self):
        """
        Record audio from the microphone in memory.

        :return: numpy array of audio data, or None if the recording is too short
        """
        _debug("  _record_audio() entered")
        try:
            _refresh_input_devices()
            wasapi_default = _wasapi_default_input_device()
            preferred_device = _preferred_capture_device(wasapi_default)
            _debug(
                "  Windows default input: "
                f"device={_device_label(wasapi_default)}, "
                f"host_api={_device_host_api(wasapi_default)}"
            )
            _debug(
                "  Current capture target: "
                f"device={_device_label(preferred_device)}, "
                f"host_api={_device_host_api(preferred_device)}"
            )

            recording = []

            audio_queue: Queue[np.ndarray] = Queue()
            callback_error = [None]  # Mutable container to capture callback errors

            def audio_callback(indata, frames, time, status):
                try:
                    if status:
                        ConfigManager.console_print(f"Audio callback status: {status}")
                    if indata is None or len(indata) == 0:
                        return  # Skip empty frames
                    audio_queue.put(indata[:, 0].copy())
                except Exception as e:
                    callback_error[0] = str(e)

            def process_frame(frame):
                if frame is None or len(frame) == 0:
                    return

                frame = np.asarray(frame, dtype=np.int16)
                recording.append(frame)

            _debug("  Opening audio stream...")
            stream, selected_device, self.sample_rate, frame_size = _open_input_stream(
                preferred_device,
                audio_callback,
            )
            _debug(
                "  Capture configured: "
                f"device={_device_label(selected_device)}, "
                f"host_api={_device_host_api(selected_device)}, "
                f"sample_rate={self.sample_rate}, frame_size={frame_size}"
            )
            if selected_device != preferred_device:
                _debug(
                    f"  Using fallback input device: {_device_label(selected_device)}"
                )

            with stream:
                _debug("  Audio stream opened, entering recording loop")
                while self.is_running and self.is_recording:
                    # Check for callback errors
                    if callback_error[0]:
                        _debug(f"  Callback error: {callback_error[0]}")
                        break

                    # Use timeout so we can check is_recording flag regularly.
                    try:
                        frame = audio_queue.get(timeout=0.1)
                    except Empty:
                        continue

                    process_frame(frame)

                    # Drain any backlog so callback bursts do not overwrite audio.
                    while True:
                        try:
                            queued_frame = audio_queue.get_nowait()
                        except Empty:
                            break
                        process_frame(queued_frame)

                _debug(
                    "  Recording loop exited "
                    f"(is_running={self.is_running}, is_recording={self.is_recording}, "
                    f"callback_error={callback_error[0]!r})"
                )

                # Capture frames already delivered by the audio callback before
                # the stream closes. Without this, the last callback burst can be
                # dropped when the user presses the stop hotkey.
                flush_deadline = time.monotonic() + 0.25
                flushed_frames = 0
                while time.monotonic() < flush_deadline:
                    try:
                        frame = audio_queue.get(timeout=0.03)
                    except Empty:
                        continue
                    process_frame(frame)
                    flushed_frames += 1

                while True:
                    try:
                        frame = audio_queue.get_nowait()
                    except Empty:
                        break
                    process_frame(frame)
                    flushed_frames += 1

                if flushed_frames:
                    _debug(f"  Flushed {flushed_frames} queued audio frames after stop")

            _debug("  Audio stream closed")
            audio_data = (
                np.concatenate(recording).astype(np.int16, copy=False)
                if recording
                else np.array([], dtype=np.int16)
            )
            duration = len(audio_data) / self.sample_rate

            ConfigManager.console_print(
                f"Recording finished. Size: {audio_data.size} samples, Duration: {duration:.2f} seconds"
            )
            _debug(f"  Recording finished: {audio_data.size} samples, {duration:.2f}s")

            if (duration * 1000) < 100:
                ConfigManager.console_print("Discarded due to being too short.")
                _debug("  Recording too short, returning None")
                return None

            return audio_data
        except Exception as e:
            _debug(f"  _record_audio() EXCEPTION: {e}")
            _debug(f"  Traceback: {traceback.format_exc()}")
            raise
