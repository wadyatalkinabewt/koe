"""
Audio capture for Scribe.

Captures mic + Windows loopback (system audio) simultaneously, writing both
streams as raw WAV files to a temp dir. No live processing — transcription
happens after stop() in the app layer.

Mic is captured at 16kHz mono int16 (Whisper-ready). Loopback is captured at
the device's native rate/channels and downmixed/resampled later.
"""

import wave
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pyaudiowpatch as pyaudio


class AudioCapture:
    """Captures mic + system loopback to two WAV files in a temp dir."""

    def __init__(self, temp_dir: Path, mic_sample_rate: int = 16000, chunk_size: int = 1024):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.mic_sample_rate = mic_sample_rate
        self.chunk_size = chunk_size

        self.mic_path = self.temp_dir / "mic.wav"
        self.loopback_path = self.temp_dir / "loopback.wav"

        self._recording = False
        self._mic_stream = None
        self._loopback_stream = None
        self._mic_wav: Optional[wave.Wave_write] = None
        self._loopback_wav: Optional[wave.Wave_write] = None
        self._wav_lock = threading.Lock()

        self.loopback_sample_rate: int = 48000
        self.loopback_channels: int = 2

        self.p = pyaudio.PyAudio()
        self.mic_device = self._find_default_mic()
        self.loopback_device = self._find_loopback_device()

    def _find_default_mic(self) -> Optional[dict]:
        try:
            info = self.p.get_default_input_device_info()
            print(f"[Scribe] Default mic: {info['name']}")
            return info
        except Exception as e:
            print(f"[Scribe] Could not find default mic: {e}")
            return None

    def _find_loopback_device(self) -> Optional[dict]:
        try:
            default = self.p.get_default_wasapi_loopback()
            if default:
                print(f"[Scribe] Default loopback: {default['name']} "
                      f"({default['defaultSampleRate']}Hz, {default['maxInputChannels']}ch)")
                return default
        except Exception as e:
            print(f"[Scribe] get_default_wasapi_loopback failed: {e}")

        try:
            wasapi = self.p.get_host_api_info_by_type(pyaudio.paWASAPI)
            for i in range(self.p.get_device_count()):
                d = self.p.get_device_info_by_index(i)
                if d.get('hostApi') == wasapi['index'] and d.get('isLoopbackDevice', False):
                    print(f"[Scribe] Fallback loopback: {d['name']}")
                    return d
        except Exception as e:
            print(f"[Scribe] Loopback search failed: {e}")
        return None

    def _mic_callback(self, in_data, frame_count, time_info, status):
        if self._recording and self._mic_wav is not None:
            with self._wav_lock:
                try:
                    self._mic_wav.writeframes(in_data)
                except Exception:
                    pass
        return (None, pyaudio.paContinue)

    def _loopback_callback(self, in_data, frame_count, time_info, status):
        if self._recording and self._loopback_wav is not None:
            with self._wav_lock:
                try:
                    self._loopback_wav.writeframes(in_data)
                except Exception:
                    pass
        return (None, pyaudio.paContinue)

    def start(self) -> bool:
        if self._recording:
            return True
        if not self.mic_device or not self.loopback_device:
            print("[Scribe] Cannot start — missing mic or loopback device")
            return False

        try:
            # Open WAV files for writing BEFORE starting streams
            self._mic_wav = wave.open(str(self.mic_path), 'wb')
            self._mic_wav.setnchannels(1)
            self._mic_wav.setsampwidth(2)
            self._mic_wav.setframerate(self.mic_sample_rate)

            self.loopback_sample_rate = int(self.loopback_device['defaultSampleRate'])
            self.loopback_channels = int(self.loopback_device['maxInputChannels'])
            self._loopback_wav = wave.open(str(self.loopback_path), 'wb')
            self._loopback_wav.setnchannels(self.loopback_channels)
            self._loopback_wav.setsampwidth(2)
            self._loopback_wav.setframerate(self.loopback_sample_rate)

            self._recording = True

            self._mic_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.mic_sample_rate,
                input=True,
                input_device_index=int(self.mic_device['index']),
                frames_per_buffer=self.chunk_size,
                stream_callback=self._mic_callback,
            )
            self._mic_stream.start_stream()

            self._loopback_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.loopback_channels,
                rate=self.loopback_sample_rate,
                input=True,
                input_device_index=int(self.loopback_device['index']),
                frames_per_buffer=self.chunk_size,
                stream_callback=self._loopback_callback,
            )
            self._loopback_stream.start_stream()

            print(f"[Scribe] Recording started → {self.temp_dir}")
            return True

        except Exception as e:
            print(f"[Scribe] start() failed: {e}")
            self._recording = False
            self._cleanup_streams()
            self._close_wavs()
            return False

    def stop(self):
        if not self._recording:
            return
        self._recording = False
        self._cleanup_streams()
        self._close_wavs()
        print("[Scribe] Recording stopped")

    def _cleanup_streams(self):
        if self._mic_stream is not None:
            try:
                self._mic_stream.stop_stream()
                self._mic_stream.close()
            except Exception:
                pass
            self._mic_stream = None
        if self._loopback_stream is not None:
            try:
                self._loopback_stream.stop_stream()
                self._loopback_stream.close()
            except Exception:
                pass
            self._loopback_stream = None

    def _close_wavs(self):
        with self._wav_lock:
            if self._mic_wav is not None:
                try:
                    self._mic_wav.close()
                except Exception:
                    pass
                self._mic_wav = None
            if self._loopback_wav is not None:
                try:
                    self._loopback_wav.close()
                except Exception:
                    pass
                self._loopback_wav = None

    def is_recording(self) -> bool:
        return self._recording

    def cleanup(self):
        self.stop()
        if hasattr(self, 'p') and self.p is not None:
            try:
                self.p.terminate()
            except Exception:
                pass
            self.p = None


def load_wav_as_int16(path: Path) -> tuple[np.ndarray, int, int]:
    """Read a WAV file → (samples, sample_rate, channels). Samples are int16."""
    with wave.open(str(path), 'rb') as wf:
        sr = wf.getframerate()
        nchannels = wf.getnchannels()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    audio = np.frombuffer(raw, dtype=np.int16)
    return audio, sr, nchannels


def preprocess_loopback(audio: np.ndarray, sample_rate: int, channels: int,
                        target_rate: int = 16000, target_rms: float = 3000.0) -> np.ndarray:
    """Convert loopback audio → 16kHz mono int16, normalized for Whisper.

    Energy-preserving stereo→mono (sum/√n), polyphase resampling, RMS normalize.
    """
    from scipy.signal import resample_poly
    from math import gcd

    if audio.size == 0:
        return audio.astype(np.int16)

    audio_f = audio.astype(np.float32)

    # Stereo → mono (energy-preserving)
    if channels > 1:
        audio_f = audio_f.reshape(-1, channels)
        audio_f = audio_f.sum(axis=1) / (channels ** 0.5)

    # Resample to target rate
    if sample_rate != target_rate:
        g = gcd(sample_rate, target_rate)
        up = target_rate // g
        down = sample_rate // g
        audio_f = resample_poly(audio_f, up, down)

    # Normalize to target RMS (boost quiet system audio for Whisper)
    rms = np.sqrt(np.mean(audio_f ** 2))
    if rms > 1e-3:
        gain = min(target_rms / rms, 8.0)  # cap gain to avoid runaway on near-silent audio
        audio_f = audio_f * gain

    return np.clip(audio_f, -32768, 32767).astype(np.int16)
