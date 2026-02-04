"""
Transcription Client

Connects to the Koe transcription server. Falls back to local model if server unavailable.
Used by both Koe and Meeting Transcription mode.

For remote usage (laptop over Tailscale), set the WHISPER_SERVER_URL environment variable:
    export WHISPER_SERVER_URL=http://100.x.x.x:9876
"""

import os
import time
import base64
import json
import requests
import numpy as np
from typing import Optional, Tuple, Generator, Callable, List, Dict
from urllib.parse import urljoin
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Debug logging
_DEBUG_LOG = Path(__file__).parent.parent / "logs" / "debug.log"

def _debug(msg: str):
    """Write debug message to file with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] [client] {msg}\n")
    except:
        pass


@dataclass
class TranscriptionSegment:
    """A segment of transcribed audio."""
    text: str
    start: float
    end: float


@dataclass
class MeetingSegment:
    """A transcribed segment with speaker identification."""
    speaker: str
    text: str
    start: float
    end: float


# Support remote server via environment variable
DEFAULT_SERVER_URL = os.environ.get("WHISPER_SERVER_URL", "http://localhost:9876")

# API token for authentication (optional - if not set, no auth sent)
DEFAULT_API_TOKEN = os.environ.get("KOE_API_TOKEN")


def _validate_server_url(url: str) -> str:
    """Validate server URL has valid scheme and netloc.

    Args:
        url: The server URL to validate

    Returns:
        The validated URL (stripped of trailing slash)

    Raises:
        ValueError: If URL is malformed
    """
    from urllib.parse import urlparse
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Invalid WHISPER_SERVER_URL: must start with http:// or https:// (got '{url}')"
        )

    if not parsed.netloc or not parsed.hostname:
        raise ValueError(
            f"Invalid WHISPER_SERVER_URL: missing host (got '{url}')"
        )

    return url.rstrip("/")


class TranscriptionClient:
    """Client for the Whisper transcription server."""

    def __init__(self, server_url: str = DEFAULT_SERVER_URL, timeout: float = 60.0,
                 api_token: Optional[str] = DEFAULT_API_TOKEN):
        self.server_url = _validate_server_url(server_url)
        self.timeout = timeout  # Read timeout
        self.connect_timeout = 5.0  # Connection timeout (fail fast if server unreachable)
        self.api_token = api_token
        self._server_available: Optional[bool] = None

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for requests, including API token if configured."""
        headers = {}
        if self.api_token:
            headers["X-API-Token"] = self.api_token
        return headers

    def _save_failed_audio(self, audio_data: np.ndarray, sample_rate: int, reason: str):
        """Save audio to logs folder when transcription fails, so it can be recovered."""
        try:
            import wave
            logs_dir = Path(__file__).parent.parent / "logs"
            logs_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = logs_dir / f"failed_audio_{reason}_{timestamp}.wav"

            # Convert to int16 if needed
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32767).astype(np.int16)

            with wave.open(str(filename), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data.tobytes())

            _debug(f"  SAVED failed audio to {filename} ({len(audio_data)/sample_rate:.1f}s)")
        except Exception as e:
            _debug(f"  Failed to save audio: {e}")

    def is_server_available(self, force_check: bool = False) -> bool:
        """Check if the transcription server is running and ready."""
        if self._server_available is not None and not force_check:
            return self._server_available

        try:
            response = requests.get(
                f"{self.server_url}/status",
                timeout=2.0,
                headers=self._get_headers()
            )
            if response.status_code == 200:
                data = response.json()
                self._server_available = data.get("ready", False)
                return self._server_available
        except requests.RequestException:
            pass

        self._server_available = False
        return False

    def supports_long_audio(self) -> bool:
        """Check if the server supports efficient long audio transcription.

        Returns False if:
        - Server is not available
        - Server is running Parakeet without local attention enabled

        Long audio (>60s) without local attention will be extremely slow.
        """
        try:
            response = requests.get(
                f"{self.server_url}/status",
                timeout=2.0,
                headers=self._get_headers()
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("supports_long_audio", True)
        except requests.RequestException:
            pass
        return False

    def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        vad_filter: bool = True,
        filter_to_speaker: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        Transcribe audio using the server.

        Args:
            audio_data: int16 or float32 numpy array
            sample_rate: Sample rate of audio
            language: Language code (e.g., 'en', 'es')
            initial_prompt: Prompt to condition the model
            vad_filter: Whether to use voice activity detection
            filter_to_speaker: If set, only transcribe audio matching this enrolled speaker

        Returns:
            Tuple of (transcription text, success boolean)
        """
        _debug(f"transcribe() called: {len(audio_data)} samples, dtype={audio_data.dtype}")

        # Convert to int16 if needed
        if audio_data.dtype == np.float32:
            audio_int16 = (audio_data * 32768).astype(np.int16)
        elif audio_data.dtype == np.int16:
            audio_int16 = audio_data
        else:
            audio_int16 = audio_data.astype(np.int16)

        _debug(f"  Converting to base64 ({len(audio_int16) * 2} bytes)...")
        # Encode as base64
        audio_base64 = base64.b64encode(audio_int16.tobytes()).decode("utf-8")
        _debug(f"  Base64 encoded: {len(audio_base64)} chars")

        payload = {
            "audio_base64": audio_base64,
            "sample_rate": sample_rate,
            "vad_filter": vad_filter
        }
        if language:
            payload["language"] = language
        if initial_prompt:
            payload["initial_prompt"] = initial_prompt
        if filter_to_speaker:
            payload["filter_to_speaker"] = filter_to_speaker

        # Dynamic timeout based on audio length
        # Base 30s + 3x audio duration, capped at 900s (15 min)
        # WSL Parakeet can be slow on long recordings due to overhead
        audio_duration_sec = len(audio_data) / sample_rate
        dynamic_timeout = min(30.0 + (audio_duration_sec * 3), 900.0)

        # Check for long audio without local attention support
        # Without local attention, Parakeet has O(n²) attention which causes massive slowdown
        LONG_AUDIO_THRESHOLD = 60.0  # seconds
        if audio_duration_sec > LONG_AUDIO_THRESHOLD:
            if not self.supports_long_audio():
                _debug(f"  WARNING: Long audio ({audio_duration_sec:.1f}s) but server doesn't support it!")
                _debug(f"  Server likely missing local attention - transcription would be extremely slow")
                self._save_failed_audio(audio_data, sample_rate, "no_long_audio_support")
                return (
                    f"Server does not support long audio (>{LONG_AUDIO_THRESHOLD:.0f}s). "
                    "Restart the server to enable local attention. "
                    f"Audio saved to logs/failed_audio_no_long_audio_support_*.wav",
                    False
                )
            _debug(f"  Long audio ({audio_duration_sec:.1f}s) - server supports it, proceeding...")

        # Retry logic with exponential backoff
        max_retries = 2
        retry_delays = [1.0, 2.0]  # Exponential backoff: 1s, 2s
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    delay = retry_delays[attempt - 1]
                    _debug(f"  Retry {attempt}/{max_retries} after {delay}s delay...")
                    time.sleep(delay)

                _debug(f"  POST {self.server_url}/transcribe (connect={self.connect_timeout}s, read={dynamic_timeout:.1f}s for {audio_duration_sec:.1f}s audio)...")
                response = requests.post(
                    f"{self.server_url}/transcribe",
                    json=payload,
                    timeout=(self.connect_timeout, dynamic_timeout),  # (connect, read) timeouts
                    headers=self._get_headers()
                )
                _debug(f"  Response received: status={response.status_code}")

                if response.status_code == 200:
                    data = response.json()
                    text = data.get("text", "")
                    _debug(f"  Success: {len(text)} chars")
                    return text, True
                elif response.status_code == 401:
                    # Auth error - don't retry
                    _debug(f"  AUTH ERROR: Invalid or missing API token")
                    return "Authentication failed: invalid or missing API token", False
                else:
                    _debug(f"  Server error: {response.status_code}")
                    last_error = f"Server error: {response.status_code}"
                    # Don't retry on server errors (4xx, 5xx) except connection issues
                    if response.status_code >= 500:
                        continue  # Retry on 5xx errors
                    return last_error, False

            except (requests.Timeout, requests.ConnectionError) as e:
                # Retry on timeout and connection errors
                _debug(f"  {'TIMEOUT' if isinstance(e, requests.Timeout) else 'CONNECTION ERROR'} (attempt {attempt + 1}/{max_retries + 1}): {e}")
                last_error = str(e)
                if attempt == max_retries:
                    # All retries exhausted - save audio for recovery
                    reason = "timeout" if isinstance(e, requests.Timeout) else "connection_error"
                    self._save_failed_audio(audio_data, sample_rate, reason)
                    return f"{'Transcription timed out' if isinstance(e, requests.Timeout) else 'Connection error'} after {max_retries + 1} attempts", False
                continue

            except requests.RequestException as e:
                _debug(f"  REQUEST ERROR: {e}")
                self._save_failed_audio(audio_data, sample_rate, "connection_error")
                return f"Connection error: {e}", False

            except Exception as e:
                _debug(f"  UNEXPECTED ERROR: {type(e).__name__}: {e}")
                self._save_failed_audio(audio_data, sample_rate, "error")
                return f"Unexpected error: {e}", False

        # Should not reach here, but just in case
        self._save_failed_audio(audio_data, sample_rate, "unknown")
        return f"Failed after {max_retries + 1} attempts: {last_error}", False

    def transcribe_stream(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        vad_filter: bool = True,
        on_segment: Optional[Callable[[TranscriptionSegment], None]] = None
    ) -> Tuple[str, bool]:
        """
        Transcribe audio with streaming - calls on_segment for each segment as it arrives.

        Args:
            audio_data: int16 or float32 numpy array
            sample_rate: Sample rate of audio
            language: Language code (e.g., 'en', 'es')
            initial_prompt: Prompt to condition the model
            vad_filter: Whether to use voice activity detection
            on_segment: Callback called with each segment as it's transcribed

        Returns:
            Tuple of (full transcription text, success boolean)
        """
        # Convert to int16 if needed
        if audio_data.dtype == np.float32:
            audio_int16 = (audio_data * 32768).astype(np.int16)
        elif audio_data.dtype == np.int16:
            audio_int16 = audio_data
        else:
            audio_int16 = audio_data.astype(np.int16)

        # Encode as base64
        audio_base64 = base64.b64encode(audio_int16.tobytes()).decode("utf-8")

        payload = {
            "audio_base64": audio_base64,
            "sample_rate": sample_rate,
            "vad_filter": vad_filter
        }
        if language:
            payload["language"] = language
        if initial_prompt:
            payload["initial_prompt"] = initial_prompt

        try:
            response = requests.post(
                f"{self.server_url}/transcribe/stream",
                json=payload,
                timeout=self.timeout,
                stream=True,
                headers=self._get_headers()
            )

            if response.status_code != 200:
                return f"Server error: {response.status_code}", False

            full_text = ""
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]  # Remove "data: " prefix
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if data.get("done"):
                    full_text = data.get("full_text", "")
                elif "text" in data and on_segment:
                    segment = TranscriptionSegment(
                        text=data["text"],
                        start=data.get("start", 0.0),
                        end=data.get("end", 0.0)
                    )
                    on_segment(segment)

            return full_text, True

        except requests.Timeout:
            return "Transcription timed out", False
        except requests.RequestException as e:
            return f"Connection error: {e}", False

    def get_status(self) -> dict:
        """Get server status."""
        try:
            response = requests.get(
                f"{self.server_url}/status",
                timeout=2.0,
                headers=self._get_headers()
            )
            if response.status_code == 200:
                return response.json()
        except requests.RequestException:
            pass
        return {"status": "unavailable", "ready": False}

    def transcribe_meeting(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        vad_filter: bool = True,
        min_speakers: int = 1,
        max_speakers: int = 8,
        user_name: Optional[str] = None
    ) -> Tuple[List[MeetingSegment], bool]:
        """
        Transcribe audio with speaker diarization (for Scribe).

        Args:
            audio_data: int16 or float32 numpy array
            sample_rate: Sample rate of audio
            language: Language code (e.g., 'en', 'es')
            initial_prompt: Prompt to condition the model
            vad_filter: Whether to use voice activity detection
            min_speakers: Minimum expected speakers
            max_speakers: Maximum expected speakers
            user_name: Name for mic audio (if single speaker source)

        Returns:
            Tuple of (list of MeetingSegment, success boolean)
        """
        # Convert to int16 if needed
        if audio_data.dtype == np.float32:
            audio_int16 = (audio_data * 32768).astype(np.int16)
        elif audio_data.dtype == np.int16:
            audio_int16 = audio_data
        else:
            audio_int16 = audio_data.astype(np.int16)

        # Encode as base64
        audio_base64 = base64.b64encode(audio_int16.tobytes()).decode("utf-8")

        payload = {
            "audio_base64": audio_base64,
            "sample_rate": sample_rate,
            "vad_filter": vad_filter,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers
        }
        if language:
            payload["language"] = language
        if initial_prompt:
            payload["initial_prompt"] = initial_prompt
        if user_name:
            payload["user_name"] = user_name

        # Dynamic timeout based on audio length
        # Diarization takes longer: base 45s + 3x audio duration, capped at 300s
        audio_duration_sec = len(audio_data) / sample_rate
        dynamic_timeout = min(45.0 + (audio_duration_sec * 3), 300.0)

        try:
            response = requests.post(
                f"{self.server_url}/transcribe_meeting",
                json=payload,
                timeout=(self.connect_timeout, dynamic_timeout),
                headers=self._get_headers()
            )

            if response.status_code == 200:
                data = response.json()
                segments = [
                    MeetingSegment(
                        speaker=seg["speaker"],
                        text=seg["text"],
                        start=seg["start"],
                        end=seg["end"]
                    )
                    for seg in data.get("segments", [])
                ]
                return segments, True
            else:
                return [], False

        except requests.Timeout:
            return [], False
        except requests.RequestException as e:
            return [], False

    def reset_diarization(self) -> bool:
        """Reset diarization session (call at start of new meeting)."""
        try:
            response = requests.post(
                f"{self.server_url}/diarization/reset",
                timeout=5.0,
                headers=self._get_headers()
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def get_speakers(self) -> List[str]:
        """Get list of enrolled speakers."""
        try:
            response = requests.get(
                f"{self.server_url}/speakers",
                timeout=5.0,
                headers=self._get_headers()
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("speakers", [])
        except requests.RequestException:
            pass
        return []

    def is_diarization_available(self) -> bool:
        """Check if diarization is available on the server."""
        status = self.get_status()
        return status.get("diarization_available", False)

    def get_unenrolled_speakers(self) -> Dict[str, np.ndarray]:
        """Get unenrolled session speakers with their embeddings from the server.

        Returns:
            Dict mapping speaker labels (e.g., "Speaker 1") to numpy embeddings
        """
        try:
            response = requests.get(
                f"{self.server_url}/diarization/unenrolled",
                timeout=15.0,  # Increased for slow networks
                headers=self._get_headers()
            )
            if response.status_code == 200:
                data = response.json()
                speakers_data = data.get("speakers", {})

                # Decode base64 embeddings back to numpy arrays
                result = {}
                for label, emb_base64 in speakers_data.items():
                    embedding_bytes = base64.b64decode(emb_base64)
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    result[label] = embedding
                return result
        except requests.RequestException:
            pass
        return {}

    def get_failed_audio_files(self, max_age_minutes: int = 60) -> List[Path]:
        """Get list of failed audio files that could be retried.

        Args:
            max_age_minutes: Only return files created within this many minutes

        Returns:
            List of Path objects for failed audio files
        """
        logs_dir = Path(__file__).parent.parent / "logs"
        if not logs_dir.exists():
            return []

        # Find failed audio files
        failed_files = list(logs_dir.glob("failed_audio_*.wav"))

        # Filter by age
        cutoff_time = time.time() - (max_age_minutes * 60)
        recent_files = [
            f for f in failed_files
            if f.stat().st_mtime > cutoff_time
        ]

        # Sort by modification time (oldest first)
        recent_files.sort(key=lambda f: f.stat().st_mtime)
        return recent_files

    def retry_failed_audio(self, audio_path: Path, language: Optional[str] = None,
                          initial_prompt: Optional[str] = None) -> Tuple[str, bool]:
        """Retry transcription of a failed audio file.

        Args:
            audio_path: Path to the failed audio WAV file
            language: Language code for transcription
            initial_prompt: Initial prompt for transcription

        Returns:
            Tuple of (result_text, success). On success, deletes the failed audio file.
        """
        import wave

        try:
            # Load the WAV file
            with wave.open(str(audio_path), 'rb') as wf:
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                audio_bytes = wf.readframes(n_frames)
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)

            _debug(f"Retrying failed audio: {audio_path.name} ({len(audio_data)/sample_rate:.1f}s)")

            # Try to transcribe
            result, success = self.transcribe(
                audio_data, sample_rate=sample_rate,
                language=language, initial_prompt=initial_prompt
            )

            if success:
                # Delete the failed audio file on success
                try:
                    audio_path.unlink()
                    _debug(f"  Retry SUCCESS - deleted {audio_path.name}")
                except Exception as e:
                    _debug(f"  Retry succeeded but failed to delete file: {e}")
                return result, True
            else:
                _debug(f"  Retry FAILED: {result}")
                return result, False

        except Exception as e:
            _debug(f"  Retry ERROR: {e}")
            return f"Failed to load audio: {e}", False


# Convenience function for simple usage
_default_client: Optional[TranscriptionClient] = None


def get_client(server_url: str = DEFAULT_SERVER_URL) -> TranscriptionClient:
    """Get or create the default transcription client."""
    global _default_client
    if _default_client is None or _default_client.server_url != server_url:
        _default_client = TranscriptionClient(server_url)
    return _default_client


def transcribe_via_server(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    language: Optional[str] = None,
    server_url: str = DEFAULT_SERVER_URL
) -> Tuple[str, bool]:
    """
    Convenience function to transcribe via server.

    Returns (text, success). If success is False, text contains error message.
    """
    client = get_client(server_url)
    return client.transcribe(audio_data, sample_rate, language)


def is_server_running(server_url: str = DEFAULT_SERVER_URL) -> bool:
    """Check if the transcription server is running."""
    client = get_client(server_url)
    return client.is_server_available(force_check=True)
