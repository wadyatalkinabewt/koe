"""
Whisper transcription engine using faster-whisper.

This is the default engine, well-tested with good accuracy.
"""

from typing import List, Optional
import numpy as np

from .base import TranscriptionEngine, TranscriptionResult, TranscriptionSegment, ModelInfo
from .factory import register_engine


# Model metadata
WHISPER_MODELS = [
    ModelInfo(
        id="tiny",
        name="Whisper Tiny",
        engine="whisper",
        size_mb=75,
        vram_mb=500,
        description="Fastest, lowest accuracy. Good for testing.",
        languages=["multilingual"]
    ),
    ModelInfo(
        id="tiny.en",
        name="Whisper Tiny (English)",
        engine="whisper",
        size_mb=75,
        vram_mb=500,
        description="English-only variant of Tiny.",
        languages=["en"]
    ),
    ModelInfo(
        id="base",
        name="Whisper Base",
        engine="whisper",
        size_mb=150,
        vram_mb=600,
        description="Good balance of speed and accuracy for CPU.",
        languages=["multilingual"]
    ),
    ModelInfo(
        id="base.en",
        name="Whisper Base (English)",
        engine="whisper",
        size_mb=150,
        vram_mb=600,
        description="English-only variant of Base.",
        languages=["en"]
    ),
    ModelInfo(
        id="small",
        name="Whisper Small",
        engine="whisper",
        size_mb=500,
        vram_mb=1000,
        description="Better accuracy, still CPU-friendly.",
        languages=["multilingual"]
    ),
    ModelInfo(
        id="small.en",
        name="Whisper Small (English)",
        engine="whisper",
        size_mb=500,
        vram_mb=1000,
        description="English-only variant of Small.",
        languages=["en"]
    ),
    ModelInfo(
        id="medium",
        name="Whisper Medium",
        engine="whisper",
        size_mb=1500,
        vram_mb=2000,
        description="High accuracy, needs ~2GB VRAM.",
        languages=["multilingual"]
    ),
    ModelInfo(
        id="medium.en",
        name="Whisper Medium (English)",
        engine="whisper",
        size_mb=1500,
        vram_mb=2000,
        description="English-only variant of Medium.",
        languages=["en"]
    ),
    ModelInfo(
        id="large-v3",
        name="Whisper Large v3",
        engine="whisper",
        size_mb=3000,
        vram_mb=4000,
        description="Best accuracy, needs ~4GB VRAM. Recommended for GPU.",
        languages=["multilingual"]
    ),
    ModelInfo(
        id="large-v2",
        name="Whisper Large v2",
        engine="whisper",
        size_mb=3000,
        vram_mb=4000,
        description="Previous best model, still excellent.",
        languages=["multilingual"]
    ),
    ModelInfo(
        id="large-v1",
        name="Whisper Large v1",
        engine="whisper",
        size_mb=3000,
        vram_mb=4000,
        description="Original large model.",
        languages=["multilingual"]
    ),
    ModelInfo(
        id="large",
        name="Whisper Large",
        engine="whisper",
        size_mb=3000,
        vram_mb=4000,
        description="Alias for large-v3.",
        languages=["multilingual"]
    ),
]


@register_engine
class WhisperEngine(TranscriptionEngine):
    """
    Transcription engine using OpenAI Whisper via faster-whisper.

    faster-whisper is a CTranslate2 implementation that's significantly faster
    than the original OpenAI implementation while maintaining the same accuracy.
    """

    ENGINE_ID = "whisper"
    ENGINE_NAME = "Whisper (faster-whisper)"

    def __init__(self):
        super().__init__()
        self._compute_type = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if faster-whisper is installed."""
        try:
            import faster_whisper
            return True
        except ImportError:
            return False

    @classmethod
    def get_install_hint(cls) -> str:
        return "pip install faster-whisper nvidia-cudnn-cu12 nvidia-cublas-cu12"

    def load(self, model_name: str, device: str = "auto", compute_type: str = "float16") -> bool:
        """Load a Whisper model."""
        try:
            from faster_whisper import WhisperModel

            # Handle auto device selection
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

            # int8 requires CPU
            if compute_type == "int8":
                device = "cpu"

            print(f"[WhisperEngine] Loading model '{model_name}' on {device} ({compute_type})...")

            try:
                self._model = WhisperModel(model_name, device=device, compute_type=compute_type)
                self._device = device
            except Exception as e:
                print(f"[WhisperEngine] GPU load failed ({e}), falling back to CPU...")
                self._model = WhisperModel(model_name, device="cpu", compute_type=compute_type)
                self._device = "cpu"

            self._model_name = model_name
            self._compute_type = compute_type
            self._loaded = True

            print(f"[WhisperEngine] Model loaded successfully on {self._device}")
            return True

        except Exception as e:
            print(f"[WhisperEngine] Failed to load model: {e}")
            self._loaded = False
            return False

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        vad_filter: bool = True,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe audio using Whisper."""
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Ensure audio is float32 and normalized
        if audio.dtype != np.float32:
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            else:
                audio = audio.astype(np.float32)

        # Calculate duration
        duration = len(audio) / sample_rate

        # Additional options
        condition_on_previous = kwargs.get("condition_on_previous_text", False)
        hallucination_threshold = kwargs.get("hallucination_silence_threshold", 0.5)

        # Run transcription
        segments_iter, info = self._model.transcribe(
            audio=audio,
            language=language,
            initial_prompt=initial_prompt,
            vad_filter=vad_filter,
            condition_on_previous_text=condition_on_previous,
            hallucination_silence_threshold=hallucination_threshold,
        )

        # Collect segments
        segments = []
        full_text_parts = []

        for segment in segments_iter:
            segments.append(TranscriptionSegment(
                text=segment.text,
                start=segment.start,
                end=segment.end
            ))
            full_text_parts.append(segment.text)

        full_text = "".join(full_text_parts).strip()

        return TranscriptionResult(
            text=full_text,
            segments=segments,
            duration_seconds=duration
        )

    def get_supported_models(self) -> List[ModelInfo]:
        """Get list of supported Whisper models."""
        return WHISPER_MODELS.copy()

    def unload(self) -> None:
        """Unload the model and free GPU memory."""
        super().unload()
        self._compute_type = None
        # Force garbage collection to free GPU memory
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
