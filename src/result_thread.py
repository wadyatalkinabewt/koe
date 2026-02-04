import time
import traceback
import numpy as np
import sounddevice as sd
import tempfile
import wave
import webrtcvad
from PyQt5.QtCore import QThread, QMutex, pyqtSignal
from collections import deque
from threading import Event, Lock
from pathlib import Path
from datetime import datetime

from transcription import transcribe
from utils import ConfigManager

# Debug logging to file with rotation
_DEBUG_LOG = Path(__file__).parent.parent / "logs" / "debug.log"
_DEBUG_LOG.parent.mkdir(exist_ok=True)
_MAX_LOG_SIZE = 1 * 1024 * 1024  # 1MB

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

    def __init__(self, local_model=None):
        """
        Initialize the ResultThread.

        :param local_model: Local transcription model (if applicable)
        """
        super().__init__()
        self.local_model = local_model
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

            # Time the transcription process
            _debug("  Starting transcription...")
            start_time = time.time()
            result = transcribe(audio_data, self.local_model)
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

            audio_buffer = deque(maxlen=frame_size)
            recording = []

            data_ready = Event()
            buffer_lock = Lock()  # Protect audio_buffer access between threads
            callback_error = [None]  # Mutable container to capture callback errors

            def audio_callback(indata, frames, time, status):
                try:
                    if status:
                        ConfigManager.console_print(f"Audio callback status: {status}")
                    if indata is None or len(indata) == 0:
                        return  # Skip empty frames
                    with buffer_lock:
                        audio_buffer.extend(indata[:, 0])
                    data_ready.set()
                except Exception as e:
                    callback_error[0] = str(e)
                    data_ready.set()  # Unblock the main loop

            _debug("  Opening audio stream...")
            with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='int16',
                                blocksize=frame_size, device=recording_options.get('sound_device'),
                                callback=audio_callback):
                _debug("  Audio stream opened, entering recording loop")
                while self.is_running and self.is_recording:
                    # Check for callback errors
                    if callback_error[0]:
                        _debug(f"  Callback error: {callback_error[0]}")
                        break

                    # Use timeout so we can check is_recording flag regularly
                    if not data_ready.wait(timeout=0.1):
                        continue
                    data_ready.clear()

                    with buffer_lock:
                        if len(audio_buffer) < frame_size:
                            continue

                        # Save frame
                        frame = np.array(list(audio_buffer), dtype=np.int16)
                        audio_buffer.clear()
                    recording.extend(frame)

                    # Avoid trying to detect voice in initial frames
                    if initial_frames_to_skip > 0:
                        initial_frames_to_skip -= 1
                        continue

                    if vad:
                        if vad.is_speech(frame.tobytes(), self.sample_rate):
                            silent_frame_count = 0
                            if not speech_detected:
                                ConfigManager.console_print("Speech detected.")
                                speech_detected = True
                        else:
                            silent_frame_count += 1

                        if speech_detected and silent_frame_count > silence_frames:
                            _debug("  Silence detected, breaking loop")
                            break

                _debug("  Recording loop exited")

            _debug("  Audio stream closed")
            audio_data = np.array(recording, dtype=np.int16)
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
