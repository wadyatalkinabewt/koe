"""
Meeting Transcription Mode

Captures meeting audio (mic + system loopback) with speaker separation.
Uses the shared Whisper transcription server.
"""

# Lazy imports to avoid loading Windows-only dependencies when only diarization is needed
def __getattr__(name):
    if name == "AudioCapture":
        from .capture import AudioCapture
        return AudioCapture
    elif name == "AudioProcessor":
        from .processor import AudioProcessor
        return AudioProcessor
    elif name == "AudioChunk":
        from .processor import AudioChunk
        return AudioChunk
    elif name == "TranscriptWriter":
        from .transcript import TranscriptWriter
        return TranscriptWriter
    elif name == "MeetingTranscriberApp":
        from .app import MeetingTranscriberApp
        return MeetingTranscriberApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AudioCapture",
    "AudioProcessor",
    "AudioChunk",
    "TranscriptWriter",
    "MeetingTranscriberApp",
]
