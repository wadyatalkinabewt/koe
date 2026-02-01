"""
Base classes for transcription engines.

Provides a unified interface that all transcription backends must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class TranscriptionSegment:
    """A single segment of transcribed audio with timing information."""
    text: str
    start: float  # Start time in seconds
    end: float    # End time in seconds


@dataclass
class TranscriptionResult:
    """Result of a transcription operation."""
    text: str
    segments: List[TranscriptionSegment] = field(default_factory=list)
    duration_seconds: float = 0.0


@dataclass
class ModelInfo:
    """Information about a transcription model."""
    id: str
    name: str
    engine: str
    size_mb: int
    vram_mb: int
    description: str
    languages: List[str] = field(default_factory=lambda: ["en"])


class EngineNotAvailableError(Exception):
    """Raised when an engine is not available (missing dependencies)."""
    def __init__(self, engine_id: str, install_hint: str):
        self.engine_id = engine_id
        self.install_hint = install_hint
        super().__init__(f"Engine '{engine_id}' not available. {install_hint}")


class TranscriptionEngine(ABC):
    """
    Abstract base class for transcription engines.

    All transcription backends (Whisper, Parakeet, etc.) must implement this interface.
    """

    # Class attributes to be overridden by subclasses
    ENGINE_ID: str = "base"
    ENGINE_NAME: str = "Base Engine"

    def __init__(self):
        self._model = None
        self._model_name: Optional[str] = None
        self._device: Optional[str] = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._loaded

    @property
    def model_name(self) -> Optional[str]:
        """Get the name of the currently loaded model."""
        return self._model_name

    @property
    def device(self) -> Optional[str]:
        """Get the device the model is running on."""
        return self._device

    @abstractmethod
    def load(self, model_name: str, device: str = "auto", compute_type: str = "float16") -> bool:
        """
        Load a transcription model.

        Args:
            model_name: Name/ID of the model to load
            device: Device to load on ("auto", "cuda", "cpu")
            compute_type: Compute precision ("float16", "float32", "int8")

        Returns:
            True if loaded successfully, False otherwise
        """
        pass

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        vad_filter: bool = True,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio data.

        Args:
            audio: Audio data as numpy array (float32, normalized to [-1, 1])
            sample_rate: Sample rate of the audio
            language: Language code (e.g., "en") or None for auto-detect
            initial_prompt: Optional prompt to condition the transcription
            vad_filter: Whether to apply voice activity detection
            **kwargs: Engine-specific options

        Returns:
            TranscriptionResult with text and segments
        """
        pass

    @abstractmethod
    def get_supported_models(self) -> List[ModelInfo]:
        """
        Get list of models supported by this engine.

        Returns:
            List of ModelInfo for each supported model
        """
        pass

    def unload(self) -> None:
        """Unload the current model and free resources."""
        self._model = None
        self._model_name = None
        self._device = None
        self._loaded = False

    @classmethod
    def is_available(cls) -> bool:
        """
        Check if this engine is available (dependencies installed).

        Override in subclasses to check for specific dependencies.
        """
        return True

    @classmethod
    def get_install_hint(cls) -> str:
        """
        Get installation instructions for this engine.

        Override in subclasses to provide specific instructions.
        """
        return "Install required dependencies."
