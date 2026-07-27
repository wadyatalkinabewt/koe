"""
Audio capture for Scribe.

Captures mic + Windows loopback (system audio) simultaneously, writing both
streams as raw WAV files to a temp dir. No live processing — transcription
happens after stop() in the app layer.

Mic is captured at 16kHz mono int16 (API-ready). Loopback is captured at
the device's native rate/channels and downmixed/resampled later.
"""

import wave
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pyaudiowpatch as pyaudio


def _log(message: str) -> None:
    """Best-effort diagnostics must never break audio capture."""
    try:
        print(message)
    except (AttributeError, OSError, UnicodeError):
        pass


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
            _log(f"[Scribe] Default mic: {info['name']}")
            return info
        except Exception as e:
            _log(f"[Scribe] Could not find default mic: {e}")
            return None

    def _find_loopback_device(self) -> Optional[dict]:
        try:
            default = self.p.get_default_wasapi_loopback()
            if default:
                _log(f"[Scribe] Default loopback: {default['name']} "
                     f"({default['defaultSampleRate']}Hz, {default['maxInputChannels']}ch)")
                return default
        except Exception as e:
            _log(f"[Scribe] get_default_wasapi_loopback failed: {e}")

        try:
            wasapi = self.p.get_host_api_info_by_type(pyaudio.paWASAPI)
            for i in range(self.p.get_device_count()):
                d = self.p.get_device_info_by_index(i)
                if d.get('hostApi') == wasapi['index'] and d.get('isLoopbackDevice', False):
                    _log(f"[Scribe] Fallback loopback: {d['name']}")
                    return d
        except Exception as e:
            _log(f"[Scribe] Loopback search failed: {e}")
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
            _log("[Scribe] Cannot start - missing mic or loopback device")
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

            # Open both devices without starting either callback. Opening a WASAPI
            # loopback device can take noticeably longer than opening the mic; if
            # the mic starts during that work, frame zero no longer represents the
            # same point in time in both WAV files.
            self._mic_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.mic_sample_rate,
                input=True,
                input_device_index=int(self.mic_device['index']),
                frames_per_buffer=self.chunk_size,
                stream_callback=self._mic_callback,
                start=False,
            )

            self._loopback_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.loopback_channels,
                rate=self.loopback_sample_rate,
                input=True,
                input_device_index=int(self.loopback_device['index']),
                frames_per_buffer=self.chunk_size,
                stream_callback=self._loopback_callback,
                start=False,
            )

            self._recording = True
            self._loopback_stream.start_stream()
            self._mic_stream.start_stream()

            _log(f"[Scribe] Recording started -> {self.temp_dir}")
            return True

        except Exception as e:
            _log(f"[Scribe] start() failed: {e}")
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
        _log("[Scribe] Recording stopped")

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


def wav_has_meaningful_audio(
    path: Path,
    *,
    frame_ms: int = 20,
    rms_threshold: float = 12.0,
    min_active_seconds: float = 0.25,
) -> bool:
    """Return whether a WAV contains more than an effectively empty PCM stream.

    WASAPI loopback can produce a full-duration file containing only zero or
    +/-1 samples when no system audio played. Inspect short frames across every
    channel and require a small amount of sustained energy before treating that
    track as real meeting audio.
    """
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getsampwidth() != 2:
            raise ValueError("Scribe audio must use 16-bit PCM.")
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        if sample_rate <= 0 or channels <= 0:
            return False

        frames_per_chunk = max(1, int(sample_rate * frame_ms / 1000))
        required_active_frames = max(1, int(sample_rate * min_active_seconds))
        active_frames = 0

        while True:
            raw = wav_file.readframes(frames_per_chunk)
            if not raw:
                break
            samples = np.frombuffer(raw, dtype=np.int16)
            if samples.size == 0:
                continue
            frame_rms = float(
                np.sqrt(np.mean(samples.astype(np.float32) ** 2))
            )
            if frame_rms >= rms_threshold:
                active_frames += samples.size // channels
                if active_frames >= required_active_frames:
                    return True
    return False


def preprocess_audio_source(audio: np.ndarray, sample_rate: int, channels: int,
                            target_rate: int = 16000, target_rms: float = 3000.0) -> np.ndarray:
    """Convert one captured source to normalized 16 kHz mono int16.

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

    # Normalize to target RMS so quiet system audio remains intelligible.
    rms = np.sqrt(np.mean(audio_f ** 2))
    if rms > 1e-3:
        gain = min(target_rms / rms, 8.0)  # cap gain to avoid runaway on near-silent audio
        audio_f = audio_f * gain

    return np.clip(audio_f, -32768, 32767).astype(np.int16)


def write_mono_wav(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(audio, dtype=np.int16).reshape(-1)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(samples.tobytes())
    return path


def prepare_mono_meeting_mix(
    mic_path: Path,
    loopback_path: Path,
    mixed_path: Path,
    target_rate: int = 16000,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Overlay mic and loopback on one timeline and write one billable mono WAV.

    The returned source arrays are retained only for local speaker attribution;
    only ``mixed_path`` is sent to ElevenLabs.
    """
    mic_raw, mic_rate, mic_channels = load_wav_as_int16(mic_path)
    loop_raw, loop_rate, loop_channels = load_wav_as_int16(loopback_path)
    mic = preprocess_audio_source(mic_raw, mic_rate, mic_channels, target_rate=target_rate)
    loopback = preprocess_audio_source(
        loop_raw,
        loop_rate,
        loop_channels,
        target_rate=target_rate,
    )

    sample_count = max(mic.size, loopback.size)
    mic_aligned = np.zeros(sample_count, dtype=np.int16)
    loop_aligned = np.zeros(sample_count, dtype=np.int16)
    mic_aligned[:mic.size] = mic
    loop_aligned[:loopback.size] = loopback

    mixed = mic_aligned.astype(np.float32) + loop_aligned.astype(np.float32)
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 32767.0:
        mixed *= 32767.0 / peak
    write_mono_wav(mixed_path, np.clip(mixed, -32768, 32767), target_rate)
    return mic_aligned, loop_aligned, target_rate


def source_rms(audio: np.ndarray, start: float, end: float, sample_rate: int) -> float:
    """Return RMS for a timestamped source window."""
    if audio.size == 0 or end <= start:
        return 0.0
    first = max(0, min(audio.size, int(start * sample_rate)))
    last = max(first, min(audio.size, int(end * sample_rate)))
    if last <= first:
        return 0.0
    window = audio[first:last].astype(np.float32)
    return float(np.sqrt(np.mean(window ** 2))) if window.size else 0.0


def identify_microphone_speaker(
    segments: list[dict],
    mic_audio: np.ndarray,
    loopback_audio: np.ndarray,
    sample_rate: int,
) -> str | None:
    """Identify the diarized label whose speech aligns most strongly with mic input."""
    scores: dict[str, list[float]] = {}
    for segment in segments:
        label = str(segment.get("label") or "").strip()
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or start)
        if not label or end <= start:
            continue
        mic_rms = source_rms(mic_audio, start, end, sample_rate)
        loop_rms = source_rms(loopback_audio, start, end, sample_rate)
        if mic_rms < 80.0:
            continue
        duration = max(0.05, end - start)
        dominance = (mic_rms + 40.0) / (loop_rms + 40.0)
        weighted = dominance * min(duration, 20.0)
        scores.setdefault(label, []).append(weighted)

    if not scores:
        return None
    totals = {label: sum(values) for label, values in scores.items()}
    return max(totals, key=totals.get)
