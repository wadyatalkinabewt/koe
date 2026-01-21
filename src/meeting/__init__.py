"""
Meeting Transcription Mode

Captures meeting audio (mic + system loopback) with speaker separation.
Uses the shared Whisper transcription server.
"""

from .capture import AudioCapture
from .processor import AudioProcessor, AudioChunk
from .transcript import TranscriptWriter
from .app import MeetingTranscriberApp

__all__ = [
    "AudioCapture",
    "AudioProcessor",
    "AudioChunk",
    "TranscriptWriter",
    "MeetingTranscriberApp",
]
