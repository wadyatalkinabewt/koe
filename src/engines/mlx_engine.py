"""
MLX Whisper transcription engine for Apple Silicon Macs.

Uses mlx-whisper which leverages Apple's MLX framework for GPU-accelerated
inference on Apple Silicon's unified memory architecture.

Install: pip install mlx-whisper
"""

from typing import List, Optional
import numpy as np

from .base import TranscriptionEngine, TranscriptionResult, TranscriptionSegment, ModelInfo
from .factory import register_engine


# Model metadata
MLX_MODELS = [
    ModelInfo(
        id="mlx-community/whisper-large-v3-turbo",
        name="Whisper Large v3 Turbo (MLX)",
        engine="mlx",
        size_mb=1600,
        vram_mb=3000,
        description="Fastest large model on Apple Silicon. Near-identical accuracy to large-v3.",
        languages=["multilingual"]
    ),
    ModelInfo(
        id="mlx-community/whisper-large-v3-mlx",
        name="Whisper Large v3 (MLX)",
        engine="mlx",
        size_mb=3000,
        vram_mb=4000,
        description="Best accuracy on Apple Silicon. Uses unified memory.",
        languages=["multilingual"]
    ),
    ModelInfo(
        id="mlx-community/whisper-medium-mlx",
        name="Whisper Medium (MLX)",
        engine="mlx",
        size_mb=1500,
        vram_mb=2000,
        description="Good balance of speed and accuracy.",
        languages=["multilingual"]
    ),
    ModelInfo(
        id="mlx-community/whisper-small-mlx",
        name="Whisper Small (MLX)",
        engine="mlx",
        size_mb=500,
        vram_mb=1000,
        description="Fast, moderate accuracy.",
        languages=["multilingual"]
    ),
    ModelInfo(
        id="mlx-community/whisper-base-mlx",
        name="Whisper Base (MLX)",
        engine="mlx",
        size_mb=150,
        vram_mb=600,
        description="Fastest, lowest accuracy.",
        languages=["multilingual"]
    ),
]

# Map short names to full MLX model IDs for convenience
_MODEL_ALIASES = {
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "base": "mlx-community/whisper-base-mlx",
}


@register_engine
class MLXWhisperEngine(TranscriptionEngine):
    """
    Transcription engine using mlx-whisper on Apple Silicon.

    mlx-whisper uses Apple's MLX framework which runs natively on the Apple
    Silicon GPU with unified memory. Benchmarks show ~15-20x realtime speed
    with large-v3-turbo on M2 Pro.
    """

    ENGINE_ID = "mlx"
    ENGINE_NAME = "Whisper (MLX - Apple Silicon)"

    def __init__(self):
        super().__init__()

    @classmethod
    def is_available(cls) -> bool:
        """Check if mlx-whisper is installed and we're on macOS."""
        import sys
        if sys.platform != 'darwin':
            return False
        try:
            import mlx_whisper
            return True
        except ImportError:
            return False

    @classmethod
    def get_install_hint(cls) -> str:
        return "pip install mlx-whisper (macOS Apple Silicon only)"

    def _resolve_model_name(self, model_name: str) -> str:
        """Resolve short model names to full MLX model IDs."""
        return _MODEL_ALIASES.get(model_name, model_name)

    def load(self, model_name: str, device: str = "auto", compute_type: str = "float16") -> bool:
        """Load an MLX Whisper model.

        The device and compute_type parameters are ignored since MLX automatically
        uses the Apple Silicon GPU with optimal precision.
        """
        try:
            import mlx_whisper

            resolved_name = self._resolve_model_name(model_name)
            print(f"[MLXWhisperEngine] Loading model '{resolved_name}' on Apple Silicon...")

            # mlx-whisper loads models lazily on first transcribe call,
            # but we can trigger a download/cache check by calling transcribe
            # with a tiny silent audio clip
            test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            try:
                mlx_whisper.transcribe(test_audio, path_or_hf_repo=resolved_name)
            except Exception:
                pass  # Some errors on silence are expected

            self._model_name = resolved_name
            self._device = "apple_silicon"
            self._loaded = True

            print(f"[MLXWhisperEngine] Model ready: {resolved_name}")
            return True

        except Exception as e:
            print(f"[MLXWhisperEngine] Failed to load model: {e}")
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
        """Transcribe audio using MLX Whisper."""
        if not self._loaded or self._model_name is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import mlx_whisper

        # Ensure audio is float32 and normalized
        if audio.dtype != np.float32:
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            else:
                audio = audio.astype(np.float32)

        duration = len(audio) / sample_rate

        # Build options
        transcribe_kwargs = {
            "path_or_hf_repo": self._model_name,
        }

        if language:
            transcribe_kwargs["language"] = language
        if initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt

        # mlx-whisper returns a dict with 'text' and 'segments'
        result = mlx_whisper.transcribe(audio, **transcribe_kwargs)

        # Extract segments
        segments = []
        for seg in result.get("segments", []):
            segments.append(TranscriptionSegment(
                text=seg.get("text", ""),
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0)
            ))

        full_text = result.get("text", "").strip()

        return TranscriptionResult(
            text=full_text,
            segments=segments,
            duration_seconds=duration
        )

    def get_supported_models(self) -> List[ModelInfo]:
        """Get list of supported MLX Whisper models."""
        return MLX_MODELS.copy()

    def unload(self) -> None:
        """Unload the model."""
        super().unload()
        import gc
        gc.collect()
