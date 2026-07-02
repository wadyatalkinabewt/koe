import time
import traceback
import wave
import numpy as np
import sounddevice as sd
import webrtcvad
from PyQt5.QtCore import QThread, QMutex, pyqtSignal
from queue import Empty, Queue
from pathlib import Path
from datetime import datetime

from transcription import transcribe
from utils import ConfigManager

# Debug logging to file with rotation
_DEBUG_LOG = Path(__file__).parent.parent / "logs" / "debug.log"
_DEBUG_LOG.parent.mkdir(exist_ok=True)
_MAX_LOG_SIZE = 1 * 1024 * 1024  # 1MB
_MAX_TAIL_AUDIO_DEBUG_FILES = 5
_TAIL_AUDIO_DEBUG_SECONDS = 20
_HOST_API_PRIORITY = {
    "Windows WASAPI": 0,
    "Windows DirectSound": 1,
    "MME": 2,
    "Windows WDM-KS": 3,
}

def _rotate_log_if_needed():
    """Rotate debug.log if it exceeds max size."""
    try:
        if _DEBUG_LOG.exists() and _DEBUG_LOG.stat().st_size > _MAX_LOG_SIZE:
            backup = _DEBUG_LOG.with_suffix('.log.1')
            if backup.exists():
                backup.unlink()
            _DEBUG_LOG.rename(backup)
    except:
        pass

def _debug(msg: str):
    """Write debug message to file with timestamp."""
    _rotate_log_if_needed()
    timestamp = datetime.now().strftime("%H:%M:%S")
    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {msg}\n")
    except:
        pass


def _save_rolling_tail_audio(audio_data: np.ndarray, sample_rate: int):
    """Keep the final seconds of recent snippets for local cutoff debugging."""
    if audio_data is None or len(audio_data) == 0:
        return
    try:
        debug_dir = _DEBUG_LOG.parent / "snippet_tail_audio"
        debug_dir.mkdir(exist_ok=True)

        oldest = debug_dir / f"tail_{_MAX_TAIL_AUDIO_DEBUG_FILES}.wav"
        if oldest.exists():
            oldest.unlink()
        for i in range(_MAX_TAIL_AUDIO_DEBUG_FILES - 1, 0, -1):
            old = debug_dir / f"tail_{i}.wav"
            new = debug_dir / f"tail_{i+1}.wav"
            if old.exists():
                old.rename(new)

        sample_rate = int(sample_rate or 16000)
        tail_samples = sample_rate * _TAIL_AUDIO_DEBUG_SECONDS
        tail = np.asarray(audio_data[-tail_samples:], dtype=np.int16)
        path = debug_dir / "tail_1.wav"
        with wave.open(str(path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(tail.tobytes())
        _debug(f"  Saved tail audio debug: {path}")
    except Exception as e:
        _debug(f"  Failed to save tail audio debug: {e}")


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
    """Return input devices to try, preferring explicit/default then stable Windows APIs."""
    seen = set()
    candidates = []

    def add(device):
        key = ("default", None) if device is None else ("device", str(device))
        if key in seen:
            return
        seen.add(key)
        candidates.append(device)

    add(_normalise_sound_device(preferred_device))
    add(None)

    try:
        hostapis = sd.query_hostapis()
        devices = list(enumerate(sd.query_devices()))

        def sort_key(item):
            idx, info = item
            api_name = hostapis[info.get("hostapi", -1)].get("name", "")
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

    return candidates


def _device_label(device):
    if device is None:
        return "default"
    try:
        info = sd.query_devices(device)
        return f"{device} ({info.get('name', 'unknown')})"
    except Exception:
        return str(device)


def _open_input_stream(sample_rate: int, frame_size: int, sound_device, callback):
    """Open a compatible input stream, falling back when Windows default is stale."""
    last_error = None
    for device in _input_device_candidates(sound_device):
        try:
            stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype='int16',
                blocksize=frame_size,
                device=device,
                callback=callback,
            )
            _debug(f"  Audio stream selected: {_device_label(device)}")
            return stream, device
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

    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.is_running = True
        self.sample_rate = None
        self.mutex = QMutex()

    def stop_recording(self):
        """Stop the current recording session."""
        self.mutex.lock()
        self.is_recording = False
        self.mutex.unlock()

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

    def run(self):
        """Main execution method for the thread."""
        _debug("ResultThread.run() STARTED")
        audio_data = None  # Initialize before try so except handler can always reference it
        try:
            if not self.is_running:
                _debug("  Early exit: is_running=False")
                return

            self.mutex.lock()
            self.is_recording = True
            self.mutex.unlock()

            _debug("  Emitting 'recording' status")
            self.statusSignal.emit('recording')
            ConfigManager.console_print('Recording...')
            _debug("  Starting _record_audio()")
            audio_data = self._record_audio()
            _debug(f"  _record_audio() returned: {type(audio_data)}, size={audio_data.size if audio_data is not None else 'None'}")

            if not self.is_running:
                _debug("  Early exit after recording: is_running=False")
                # Save audio if recording was substantial (>2s) so it's not lost
                if audio_data is not None and audio_data.size > 0:
                    duration_sec = audio_data.size / (self.sample_rate or 16000)
                    if duration_sec > 2.0:
                        try:
                            import scipy.io.wavfile as wav
                            cancelled_path = _DEBUG_LOG.parent / f"failed_audio_cancelled_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                            wav.write(str(cancelled_path), self.sample_rate or 16000, audio_data)
                            _debug(f"  Saved cancelled recording ({duration_sec:.1f}s) to {cancelled_path}")
                        except Exception as save_err:
                            _debug(f"  Failed to save cancelled audio: {save_err}")
                # Emit empty result so status window gets closed properly
                self.resultSignal.emit('')
                return

            if audio_data is None:
                _debug("  Recording too short, emitting empty result")
                # Recording too short - emit empty result and close
                self.resultSignal.emit('')
                return

            _debug("  Emitting 'transcribing' status")
            self.statusSignal.emit('transcribing')
            ConfigManager.console_print('Transcribing...')
            _save_rolling_tail_audio(audio_data, self.sample_rate or 16000)

            # Time the transcription process
            _debug("  Starting transcription...")
            start_time = time.time()
            result = transcribe(audio_data, sample_rate=self.sample_rate or 16000)
            end_time = time.time()

            transcription_time = end_time - start_time
            _debug(f"  Transcription done in {transcription_time:.2f}s, result length={len(result)}")
            ConfigManager.console_print(f'Transcription completed in {transcription_time:.2f} seconds. Post-processed line: {result}')

            # If transcription returned empty/whitespace but we had substantial audio,
            # save the audio as a backup (possible silent engine failure)
            audio_duration = len(audio_data) / (self.sample_rate or 16000)
            if (not result or not result.strip()) and audio_duration > 2.0:
                _debug(f"  WARNING: Empty transcription for {audio_duration:.1f}s audio - saving backup")
                try:
                    failed_audio_path = _DEBUG_LOG.parent / f"failed_audio_empty_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                    import scipy.io.wavfile as wav
                    wav.write(str(failed_audio_path), self.sample_rate or 16000, audio_data)
                    _debug(f"  Saved empty-result audio to {failed_audio_path}")
                except Exception as save_err:
                    _debug(f"  Failed to save audio: {save_err}")

            # Always emit result after transcription completes, even if cancelled
            # (Snippet was already saved, user deserves the clipboard copy and beep)
            _debug("  Emitting result signal")
            self.resultSignal.emit(result)
            _debug("  Result signal emitted successfully")

        except Exception as e:
            _debug(f"  EXCEPTION: {e}")
            _debug(f"  Traceback: {traceback.format_exc()}")
            traceback.print_exc()

            # Save audio to disk if we have it (don't lose the recording)
            if audio_data is not None and len(audio_data) > 0:
                try:
                    failed_audio_path = _DEBUG_LOG.parent / f"failed_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                    import scipy.io.wavfile as wav
                    wav.write(str(failed_audio_path), self.sample_rate or 16000, audio_data)
                    _debug(f"  Saved failed audio to {failed_audio_path}")
                except Exception as save_err:
                    _debug(f"  Failed to save audio: {save_err}")

            error_msg = str(e) if str(e) else "Transcription failed"
            self.errorSignal.emit(error_msg)
            self.statusSignal.emit('error')
            self.resultSignal.emit('')
        finally:
            _debug("  Calling stop_recording()")
            self.stop_recording()
            _debug("ResultThread.run() FINISHED")

    def _record_audio(self):
        """
        Record audio from the microphone and save it to a temporary file.

        :return: numpy array of audio data, or None if the recording is too short
        """
        _debug("  _record_audio() entered")
        try:
            recording_options = ConfigManager.get_config_section('recording_options')
            self.sample_rate = recording_options.get('sample_rate') or 16000
            frame_duration_ms = 30  # 30ms frame duration for WebRTC VAD
            frame_size = int(self.sample_rate * (frame_duration_ms / 1000.0))
            silence_duration_ms = recording_options.get('silence_duration') or 900
            silence_frames = int(silence_duration_ms / frame_duration_ms)
            _debug(f"  Config loaded: sample_rate={self.sample_rate}, frame_size={frame_size}")

            # 150ms delay before starting VAD to avoid mistaking the sound of key pressing for voice
            initial_frames_to_skip = int(0.15 * self.sample_rate / frame_size)

            # Create VAD only for recording modes that use it
            recording_mode = recording_options.get('recording_mode') or 'continuous'
            vad = None
            if recording_mode in ('voice_activity_detection', 'continuous'):
                _debug("  Creating VAD...")
                vad = webrtcvad.Vad(2)  # VAD aggressiveness: 0 to 3, 3 being the most aggressive
                speech_detected = False
                silent_frame_count = 0
                _debug("  VAD created")

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

            def process_frame(frame, use_vad=True):
                nonlocal initial_frames_to_skip
                nonlocal speech_detected, silent_frame_count

                if frame is None or len(frame) == 0:
                    return False

                frame = np.asarray(frame, dtype=np.int16)
                recording.append(frame)

                # Avoid trying to detect voice in initial frames.
                if initial_frames_to_skip > 0:
                    initial_frames_to_skip -= 1
                    return False

                if vad and use_vad and len(frame) == frame_size:
                    if vad.is_speech(frame.tobytes(), self.sample_rate):
                        silent_frame_count = 0
                        if not speech_detected:
                            ConfigManager.console_print("Speech detected.")
                            speech_detected = True
                    else:
                        silent_frame_count += 1

                    if speech_detected and silent_frame_count > silence_frames:
                        _debug("  Silence detected, breaking loop")
                        return True

                return False

            _debug("  Opening audio stream...")
            stream, selected_device = _open_input_stream(
                self.sample_rate,
                frame_size,
                recording_options.get('sound_device'),
                audio_callback,
            )
            if selected_device != _normalise_sound_device(recording_options.get('sound_device')):
                _debug(f"  Using fallback input device: {_device_label(selected_device)}")

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

                    should_stop = process_frame(frame)

                    # Drain any backlog so callback bursts do not overwrite audio.
                    while True:
                        try:
                            queued_frame = audio_queue.get_nowait()
                        except Empty:
                            break
                        should_stop = process_frame(queued_frame) or should_stop

                    if should_stop:
                        break

                _debug("  Recording loop exited")

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
                    process_frame(frame, use_vad=False)
                    flushed_frames += 1

                while True:
                    try:
                        frame = audio_queue.get_nowait()
                    except Empty:
                        break
                    process_frame(frame, use_vad=False)
                    flushed_frames += 1

                if flushed_frames:
                    _debug(f"  Flushed {flushed_frames} queued audio frames after stop")

            _debug("  Audio stream closed")
            audio_data = np.concatenate(recording).astype(np.int16, copy=False) if recording else np.array([], dtype=np.int16)
            duration = len(audio_data) / self.sample_rate

            ConfigManager.console_print(f'Recording finished. Size: {audio_data.size} samples, Duration: {duration:.2f} seconds')
            _debug(f"  Recording finished: {audio_data.size} samples, {duration:.2f}s")

            min_duration_ms = recording_options.get('min_duration') or 100

            if (duration * 1000) < min_duration_ms:
                ConfigManager.console_print(f'Discarded due to being too short.')
                _debug("  Recording too short, returning None")
                return None

            return audio_data
        except Exception as e:
            _debug(f"  _record_audio() EXCEPTION: {e}")
            _debug(f"  Traceback: {traceback.format_exc()}")
            raise
