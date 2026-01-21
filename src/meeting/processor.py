"""
Audio processor with VAD-based chunking for natural speech breaks.
Chunks at sentence boundaries (silence) rather than fixed time intervals.
"""

import threading
import time
import numpy as np
import webrtcvad
from typing import Optional, Callable
from dataclasses import dataclass
from .capture import AudioCapture, debug_log


@dataclass
class AudioChunk:
    """A chunk of audio ready for transcription."""
    mic_audio: np.ndarray          # User's microphone audio (16kHz mono)
    loopback_audio: Optional[np.ndarray]  # System audio (others)
    loopback_sample_rate: int      # Sample rate of loopback audio
    loopback_channels: int         # Number of channels in loopback
    timestamp: float               # When chunk started
    duration: float                # Chunk duration in seconds


class AudioProcessor:
    """
    Processes audio with VAD-based chunking for natural breaks.
    Chunks are emitted when silence is detected after minimum duration,
    or when maximum duration is reached.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        min_chunk_duration: float = 10.0,  # Minimum before looking for silence
        max_chunk_duration: float = 30.0,  # Force chunk even without silence
        silence_duration_ms: int = 600,    # Silence needed to trigger chunk
        on_chunk_ready: Optional[Callable[[AudioChunk], None]] = None
    ):
        self.sample_rate = sample_rate
        self.min_chunk_duration = min_chunk_duration
        self.max_chunk_duration = max_chunk_duration
        self.silence_duration_ms = silence_duration_ms
        self.on_chunk_ready = on_chunk_ready

        self.capture = AudioCapture(sample_rate=sample_rate)

        # VAD setup - 30ms frames at 16kHz = 480 samples per frame
        self._vad = webrtcvad.Vad(2)  # Aggressiveness 2 (0-3)
        self._frame_duration_ms = 30
        self._frame_size = int(sample_rate * self._frame_duration_ms / 1000)  # 480 samples
        self._silence_frames_needed = int(silence_duration_ms / self._frame_duration_ms)  # ~20 frames for 600ms

        # Buffers
        self._mic_buffer: list = []
        self._loopback_buffer: list = []
        self._vad_buffer: np.ndarray = np.array([], dtype=np.int16)  # For VAD frame processing
        self._chunk_start_time: float = 0
        self._silence_frame_count: int = 0

        # Threading
        self._running = False
        self._process_thread: Optional[threading.Thread] = None

        # Sample thresholds
        self._min_samples = int(sample_rate * min_chunk_duration)
        self._max_samples = int(sample_rate * max_chunk_duration)

    def start(self) -> bool:
        """Start audio processing."""
        debug_log("PROCESSOR START: Beginning")
        if self._running:
            debug_log("PROCESSOR START: Already running")
            return True

        if not self.capture.start():
            debug_log("PROCESSOR START: Capture failed to start")
            return False

        debug_log("PROCESSOR START: Capture started, setting up thread")
        self._running = True
        self._mic_buffer = []
        self._loopback_buffer = []
        self._vad_buffer = np.array([], dtype=np.int16)
        self._silence_frame_count = 0
        self._chunk_start_time = time.time()

        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        debug_log(f"PROCESSOR START: Thread created, starting...")
        self._process_thread.start()
        debug_log(f"PROCESSOR START: Thread started, is_alive={self._process_thread.is_alive()}")

        print("[Meeting Processor] Started with VAD-based chunking")
        return True

    def stop(self) -> Optional[AudioChunk]:
        """Stop processing and return any remaining audio as final chunk."""
        if not self._running:
            return None

        self._running = False
        self.capture.stop()

        if self._process_thread:
            self._process_thread.join(timeout=2.0)
            self._process_thread = None

        # Return remaining audio as final chunk
        final_chunk = self._create_chunk()
        print("[Meeting Processor] Stopped")
        return final_chunk

    def _process_loop(self):
        """Main processing loop - collects audio and emits chunks on silence."""
        try:
            debug_log("PROCESSOR: Loop starting with VAD")
            loop_count = 0

            while self._running:
                loop_count += 1

                # Collect audio from both sources
                mic_audio = self.capture.get_mic_audio(timeout=0.1)
                loopback_audio = self.capture.get_loopback_audio(timeout=0.01)

                if mic_audio is not None:
                    self._mic_buffer.append(mic_audio)
                    # Also add to VAD buffer for speech detection
                    self._vad_buffer = np.concatenate([self._vad_buffer, mic_audio])

                if loopback_audio is not None:
                    self._loopback_buffer.append(loopback_audio)

                # Calculate current buffer size
                mic_samples = sum(len(b) for b in self._mic_buffer)

                # Log periodically
                if loop_count % 50 == 0:
                    debug_log(f"PROCESSOR: loop={loop_count}, buffer={mic_samples}, silence_frames={self._silence_frame_count}")

                # Process VAD frames to detect silence
                should_chunk = False
                chunk_reason = ""

                # Process complete 30ms frames from VAD buffer
                while len(self._vad_buffer) >= self._frame_size:
                    frame = self._vad_buffer[:self._frame_size]
                    self._vad_buffer = self._vad_buffer[self._frame_size:]

                    # Check if this frame contains speech
                    try:
                        is_speech = self._vad.is_speech(frame.tobytes(), self.sample_rate)
                    except Exception:
                        is_speech = True  # Assume speech on error

                    if is_speech:
                        self._silence_frame_count = 0
                    else:
                        self._silence_frame_count += 1

                # Check chunking conditions
                if mic_samples >= self._max_samples:
                    # Force chunk at max duration
                    should_chunk = True
                    chunk_reason = "max_duration"
                elif mic_samples >= self._min_samples and self._silence_frame_count >= self._silence_frames_needed:
                    # Natural break: silence detected after minimum duration
                    should_chunk = True
                    chunk_reason = "silence_detected"

                if should_chunk:
                    debug_log(f"PROCESSOR: Chunk ready! {mic_samples} samples, reason={chunk_reason}")
                    chunk = self._create_chunk()
                    if chunk and self.on_chunk_ready:
                        debug_log("PROCESSOR: Calling on_chunk_ready callback")
                        self.on_chunk_ready(chunk)

                    # Reset for next chunk
                    self._chunk_start_time = time.time()
                    self._silence_frame_count = 0

                time.sleep(0.05)  # Small sleep to prevent CPU spin

            debug_log("PROCESSOR: Loop ended")
        except Exception as e:
            debug_log(f"PROCESSOR: Exception in loop: {e}")
            import traceback
            debug_log(traceback.format_exc())

    def _create_chunk(self) -> Optional[AudioChunk]:
        """Create a chunk from buffered audio."""
        if not self._mic_buffer:
            return None

        # Concatenate mic audio
        mic_audio = np.concatenate(self._mic_buffer)
        self._mic_buffer = []

        # Concatenate loopback audio (may be empty)
        loopback_audio = None
        if self._loopback_buffer:
            loopback_audio = np.concatenate(self._loopback_buffer)
            self._loopback_buffer = []

        # Clear VAD buffer too (start fresh for next chunk)
        self._vad_buffer = np.array([], dtype=np.int16)

        duration = len(mic_audio) / self.sample_rate

        return AudioChunk(
            mic_audio=mic_audio,
            loopback_audio=loopback_audio,
            loopback_sample_rate=self.capture.loopback_sample_rate,
            loopback_channels=self.capture.loopback_channels,
            timestamp=self._chunk_start_time,
            duration=duration
        )

    def is_running(self) -> bool:
        """Check if processor is running."""
        return self._running

    def get_buffer_info(self) -> tuple:
        """Get current buffer state for UI display."""
        mic_samples = sum(len(b) for b in self._mic_buffer)
        return (mic_samples, self._min_samples, self._max_samples, self._silence_frame_count)
