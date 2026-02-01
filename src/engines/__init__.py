"""
Koe Transcription Engines

Provides a unified interface for different transcription backends:
- Whisper (faster-whisper) - Default, well-tested
- Parakeet (NVIDIA NeMo) - ~50x faster, requires WSL on Windows
"""

from .base import (
    TranscriptionEngine,
    TranscriptionResult,
    TranscriptionSegment,
    ModelInfo,
    EngineNotAvailableError,
)
from .factory import (
    create_engine,
    get_available_engines,
    is_engine_available,
    get_all_models,
    get_engine_for_model,
    get_default_engine,
    get_engine_class,
    get_all_engines,
    register_engine,
)

__all__ = [
    # Base classes
    "TranscriptionEngine",
    "TranscriptionResult",
    "TranscriptionSegment",
    "ModelInfo",
    "EngineNotAvailableError",
    # Factory functions
    "create_engine",
    "get_available_engines",
    "is_engine_available",
    "get_all_models",
    "get_engine_for_model",
    "get_default_engine",
    "get_engine_class",
    "get_all_engines",
    "register_engine",
]
