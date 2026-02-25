"""
Audio capture for meeting transcription.
Windows: PyAudioWPatch for WASAPI loopback support.
macOS: sounddevice with BlackHole virtual audio device.
Captures both microphone (user) and system audio (others) simultaneously.
"""

import sys
import threading
import queue
import numpy as np
from typing import Optional
from pathlib import Path

# Platform-conditional imports
if sys.platform == 'win32':
    import pyaudiowpatch as pyaudio
else:
    import sounddevice as sd

# Debug log file
DEBUG_LOG = str(Path(__file__).parent.parent.parent / "capture_debug.log")

def debug_log(msg):
    try:
        with open(DEBUG_LOG, "a") as f:
            f.write(f"{msg}\n")
    except:
        pass


class AudioCapture:
    """Captures audio from microphone and system loopback simultaneously."""

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._is_mac = sys.platform == 'darwin'

        self._recording = False
        self._mic_stream = None
        self._loopback_stream = None

        # Queues for audio data
        self.mic_queue: queue.Queue = queue.Queue()
        self.loopback_queue: queue.Queue = queue.Queue()

        # Track actual sample rates
        self.loopback_sample_rate: int = sample_rate
        self.loopback_channels: int = 1

        if not self._is_mac:
            self.p = pyaudio.PyAudio()

        # Find devices
        self.mic_device = self._find_default_mic()
        self.loopback_device = self._find_loopback_device()

    def _find_default_mic(self) -> Optional[dict]:
        """Find the default microphone device."""
        if self._is_mac:
            try:
                default_input = sd.query_devices(kind='input')
                print(f"[Meeting Audio] Default mic: {default_input['name']}")
                return default_input
            except Exception as e:
                print(f"[Meeting Audio] Error finding default mic: {e}")
                return None
        else:
            try:
                default_input = self.p.get_default_input_device_info()
                print(f"[Meeting Audio] Default mic: {default_input['name']}")
                return default_input
            except Exception as e:
                print(f"[Meeting Audio] Error finding default mic: {e}")
                return None

    def _find_loopback_device(self) -> Optional[dict]:
        """Find a loopback device for system audio capture."""
        if self._is_mac:
            return self._find_blackhole_device()
        else:
            return self._find_wasapi_loopback()

    def _find_blackhole_device(self) -> Optional[dict]:
        """Find BlackHole virtual audio device on macOS."""
        try:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if 'BlackHole' in dev['name'] and dev['max_input_channels'] > 0:
                    print(f"[Meeting Audio] Found BlackHole: {dev['name']}")
                    print(f"[Meeting Audio]   Rate: {dev['default_samplerate']}Hz, Channels: {dev['max_input_channels']}")
                    dev_copy = dict(dev)
                    dev_copy['index'] = i
                    return dev_copy

            print("[Meeting Audio] ERROR: BlackHole not found!")
            print("[Meeting Audio] Install with: brew install blackhole-2ch")
            print("[Meeting Audio] Then create a Multi-Output Device in Audio MIDI Setup")
            return None
        except Exception as e:
            print(f"[Meeting Audio] Error finding BlackHole device: {e}")
            return None

    def _find_wasapi_loopback(self) -> Optional[dict]:
        """Find a WASAPI loopback device for system audio capture (Windows)."""
        try:
            # ALWAYS use the default output's loopback (where meeting audio plays)
            try:
                default_loopback = self.p.get_default_wasapi_loopback()
                if default_loopback:
                    print(f"[Meeting Audio] Using default loopback: {default_loopback['name']}")
                    print(f"[Meeting Audio]   Rate: {default_loopback['defaultSampleRate']}Hz, Channels: {default_loopback['maxInputChannels']}")
                    return default_loopback
            except Exception as e:
                print(f"[Meeting Audio] get_default_wasapi_loopback failed: {e}")

            # Fallback: search for any loopback device
            wasapi_info = self.p.get_host_api_info_by_type(pyaudio.paWASAPI)

            for i in range(self.p.get_device_count()):
                device = self.p.get_device_info_by_index(i)
                if (device.get('hostApi') == wasapi_info['index'] and
                    device.get('isLoopbackDevice', False)):
                    print(f"[Meeting Audio] Fallback loopback: {device['name']}")
                    return device

            print("[Meeting Audio] ERROR: No loopback device found!")
            return None

        except Exception as e:
            print(f"[Meeting Audio] Error finding loopback device: {e}")
            return None

    # --- macOS callbacks (sounddevice) ---

    def _mic_callback_mac(self, indata, frames, time_info, status):
        """Callback for microphone audio on macOS (sounddevice)."""
        debug_log(f"MIC CALLBACK: recording={self._recording}, frames={frames}, status={status}")
        if self._recording:
            audio_data = indata.flatten().copy()
            self.mic_queue.put(audio_data)
            debug_log(f"MIC CALLBACK: queued {len(audio_data)} samples, queue size now {self.mic_queue.qsize()}")

    def _loopback_callback_mac(self, indata, frames, time_info, status):
        """Callback for loopback audio on macOS (sounddevice)."""
        if self._recording:
            self.loopback_queue.put(indata.flatten().copy())

    # --- Windows callbacks (PyAudioWPatch) ---

    def _mic_callback(self, in_data, frame_count, time_info, status):
        """Callback for microphone audio (Windows/PyAudioWPatch)."""
        debug_log(f"MIC CALLBACK: recording={self._recording}, frames={frame_count}, status={status}")
        if self._recording:
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            self.mic_queue.put(audio_data.copy())
            debug_log(f"MIC CALLBACK: queued {len(audio_data)} samples, queue size now {self.mic_queue.qsize()}")
        return (None, pyaudio.paContinue)

    def _loopback_callback(self, in_data, frame_count, time_info, status):
        """Callback for system loopback audio (Windows/PyAudioWPatch)."""
        if self._recording:
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            self.loopback_queue.put(audio_data.copy())
        return (None, pyaudio.paContinue)

    def start(self) -> bool:
        """Start capturing audio from both sources."""
        if self._recording:
            return True

        debug_log("START: Beginning audio capture setup")

        try:
            # Set recording flag FIRST so callbacks capture audio
            self._recording = True
            debug_log(f"START: Set _recording = True")

            if self._is_mac:
                return self._start_mac()
            else:
                return self._start_windows()

        except Exception as e:
            debug_log(f"START ERROR: {e}")
            print(f"[Meeting Audio] Error starting capture: {e}")
            self._recording = False
            self.stop()
            return False

    def _start_mac(self) -> bool:
        """Start audio capture on macOS using sounddevice."""
        # Start microphone stream
        if self.mic_device:
            mic_index = sd.default.device[0]  # Default input device index
            debug_log(f"START: Opening mic device {mic_index}: {self.mic_device['name']}")
            self._mic_stream = sd.InputStream(
                device=mic_index,
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16',
                blocksize=self.chunk_size,
                callback=self._mic_callback_mac
            )
            self._mic_stream.start()
            debug_log(f"START: Mic stream started")
            print("[Meeting Audio] Microphone stream started")

        # Start loopback stream (BlackHole)
        if self.loopback_device:
            bh_index = self.loopback_device['index']
            self.loopback_sample_rate = int(self.loopback_device['default_samplerate'])
            self.loopback_channels = int(self.loopback_device['max_input_channels'])

            debug_log(f"START: Opening BlackHole device {bh_index} ({self.loopback_channels}ch)")
            self._loopback_stream = sd.InputStream(
                device=bh_index,
                samplerate=self.loopback_sample_rate,
                channels=self.loopback_channels,
                dtype='int16',
                blocksize=self.chunk_size,
                callback=self._loopback_callback_mac
            )
            self._loopback_stream.start()
            debug_log(f"START: Loopback stream started")
            print(f"[Meeting Audio] BlackHole stream started ({self.loopback_sample_rate}Hz, {self.loopback_channels}ch)")

        debug_log("START: Audio capture setup complete")
        return True

    def _start_windows(self) -> bool:
        """Start audio capture on Windows using PyAudioWPatch."""
        # Start microphone stream with callback
        if self.mic_device:
            debug_log(f"START: Opening mic device {self.mic_device['index']}: {self.mic_device['name']}")
            self._mic_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=int(self.mic_device['index']),
                frames_per_buffer=self.chunk_size,
                stream_callback=self._mic_callback
            )
            self._mic_stream.start_stream()
            debug_log(f"START: Mic stream started, is_active={self._mic_stream.is_active()}")
            print("[Meeting Audio] Microphone stream started")

        # Start loopback stream with callback
        if self.loopback_device:
            self.loopback_sample_rate = int(self.loopback_device['defaultSampleRate'])
            # Use native channel count - some WASAPI devices require opening with all channels
            self.loopback_channels = int(self.loopback_device['maxInputChannels'])

            debug_log(f"START: Opening loopback device {self.loopback_device['index']} ({self.loopback_channels}ch)")
            self._loopback_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.loopback_channels,
                rate=self.loopback_sample_rate,
                input=True,
                input_device_index=int(self.loopback_device['index']),
                frames_per_buffer=self.chunk_size,
                stream_callback=self._loopback_callback
            )
            self._loopback_stream.start_stream()
            debug_log(f"START: Loopback stream started")
            print(f"[Meeting Audio] Loopback stream started ({self.loopback_sample_rate}Hz, {self.loopback_channels}ch)")

        debug_log("START: Audio capture setup complete")
        return True

    def stop(self):
        """Stop capturing audio."""
        self._recording = False
        debug_log("STOP: Stopping capture")

        if self._mic_stream:
            try:
                if self._is_mac:
                    self._mic_stream.stop()
                    self._mic_stream.close()
                else:
                    self._mic_stream.stop_stream()
                    self._mic_stream.close()
            except Exception:
                pass
            self._mic_stream = None

        if self._loopback_stream:
            try:
                if self._is_mac:
                    self._loopback_stream.stop()
                    self._loopback_stream.close()
                else:
                    self._loopback_stream.stop_stream()
                    self._loopback_stream.close()
            except Exception:
                pass
            self._loopback_stream = None

        print("[Meeting Audio] Capture stopped")

    def get_mic_audio(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get queued microphone audio."""
        chunks = []
        # First wait for at least one item (with timeout)
        try:
            chunk = self.mic_queue.get(timeout=timeout)
            chunks.append(chunk)
        except queue.Empty:
            return None

        # Then drain any additional items without blocking
        while True:
            try:
                chunk = self.mic_queue.get_nowait()
                chunks.append(chunk)
            except queue.Empty:
                break

        if chunks:
            return np.concatenate(chunks)
        return None

    def get_loopback_audio(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get queued loopback audio, converted to mono."""
        chunks = []
        # First wait for at least one item (with timeout)
        try:
            chunk = self.loopback_queue.get(timeout=timeout)
            chunks.append(chunk)
        except queue.Empty:
            return None

        # Then drain any additional items without blocking
        while True:
            try:
                chunk = self.loopback_queue.get_nowait()
                chunks.append(chunk)
            except queue.Empty:
                break

        if chunks:
            audio = np.concatenate(chunks)
            # Convert multi-channel to mono if needed
            if self.loopback_channels > 1:
                # Reshape to (samples, channels) and take mean
                audio = audio.reshape(-1, self.loopback_channels)
                # Use first 2 channels (left/right) for stereo content
                if self.loopback_channels >= 2:
                    audio = audio[:, :2].mean(axis=1)
                else:
                    audio = audio.mean(axis=1)
                audio = audio.astype(np.int16)
            return audio
        return None

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording

    def cleanup(self):
        """Clean up resources."""
        self.stop()
        if not self._is_mac and hasattr(self, 'p') and self.p:
            self.p.terminate()
            self.p = None
