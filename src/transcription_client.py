"""
Transcription Client

Connects to the Koe transcription server. Falls back to local model if server unavailable.
Used by both Koe and Meeting Transcription mode.

For remote usage (laptop over Tailscale), set the WHISPER_SERVER_URL environment variable:
    export WHISPER_SERVER_URL=http://100.x.x.x:9876
"""

import os
import base64
import json
import requests
import numpy as np
from typing import Optional, Tuple, Generator, Callable, List
from urllib.parse import urljoin
from dataclasses import dataclass


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


class TranscriptionClient:
    """Client for the Whisper transcription server."""

    def __init__(self, server_url: str = DEFAULT_SERVER_URL, timeout: float = 60.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self._server_available: Optional[bool] = None

    def is_server_available(self, force_check: bool = False) -> bool:
        """Check if the transcription server is running and ready."""
        if self._server_available is not None and not force_check:
            return self._server_available

        try:
            response = requests.get(
                f"{self.server_url}/status",
                timeout=2.0
            )
            if response.status_code == 200:
                data = response.json()
                self._server_available = data.get("ready", False)
                return self._server_available
        except requests.RequestException:
            pass

        self._server_available = False
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
        if filter_to_speaker:
            payload["filter_to_speaker"] = filter_to_speaker

        try:
            response = requests.post(
                f"{self.server_url}/transcribe",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("text", ""), True
            else:
                return f"Server error: {response.status_code}", False

        except requests.Timeout:
            return "Transcription timed out", False
        except requests.RequestException as e:
            return f"Connection error: {e}", False

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
                stream=True
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
            response = requests.get(f"{self.server_url}/status", timeout=2.0)
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

        try:
            response = requests.post(
                f"{self.server_url}/transcribe_meeting",
                json=payload,
                timeout=self.timeout
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
                timeout=5.0
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def get_speakers(self) -> List[str]:
        """Get list of enrolled speakers."""
        try:
            response = requests.get(
                f"{self.server_url}/speakers",
                timeout=5.0
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
