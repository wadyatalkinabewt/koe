"""
Parakeet transcription engine using NVIDIA NeMo.

~50x faster than Whisper with slightly better accuracy.
Requires WSL2 on Windows (native Windows not supported by NeMo).
"""

import sys
from typing import List, Optional
import numpy as np

from .base import TranscriptionEngine, TranscriptionResult, TranscriptionSegment, ModelInfo
from .factory import register_engine


# Model metadata
# Note: TDT models have CUDA graph incompatibility with CUDA 12.8+, use CTC models instead
PARAKEET_MODELS = [
    ModelInfo(
        id="nvidia/parakeet-ctc-0.6b",
        name="Parakeet CTC 0.6B",
        engine="parakeet",
        size_mb=600,
        vram_mb=2000,
        description="Fast and accurate. English only. ~50x faster than Whisper. Recommended.",
        languages=["en"]
    ),
    ModelInfo(
        id="nvidia/parakeet-ctc-1.1b",
        name="Parakeet CTC 1.1B",
        engine="parakeet",
        size_mb=1100,
        vram_mb=3000,
        description="Larger model, higher accuracy. English only.",
        languages=["en"]
    ),
    # TDT models disabled due to CUDA 12.8 incompatibility (cu_call returns 5 values, expects 6)
    # ModelInfo(
    #     id="nvidia/parakeet-tdt-0.6b-v2",
    #     name="Parakeet TDT 0.6B v2",
    #     engine="parakeet",
    #     size_mb=600,
    #     vram_mb=2000,
    #     description="Best accuracy but CUDA 12.8+ incompatible.",
    #     languages=["en"]
    # ),
]


def _is_wsl() -> bool:
    """Check if running inside WSL."""
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except:
        return False


def _check_wsl_available() -> bool:
    """Check if WSL with Ubuntu-22.04 is available (for Windows native).

    We check for WSL availability rather than a running server, since the server
    starts on demand when Koe launches.
    """
    import subprocess
    try:
        # Check if wsl command exists and Ubuntu-22.04 is installed
        result = subprocess.run(
            ["wsl", "--list", "--quiet"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            # WSL outputs UTF-16 LE on Windows, decode properly
            try:
                distros = result.stdout.decode('utf-16-le').lower()
            except UnicodeDecodeError:
                distros = result.stdout.decode('utf-8', errors='ignore').lower()
            return "ubuntu-22.04" in distros or "ubuntu" in distros
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@register_engine
class ParakeetEngine(TranscriptionEngine):
    """
    Transcription engine using NVIDIA Parakeet via NeMo toolkit.

    Parakeet is significantly faster than Whisper (~50x) with slightly better
    accuracy (6.05% vs 6.43% WER). However, it only supports English.

    On Windows, this requires WSL2 with GPU passthrough since NeMo doesn't
    support native Windows.
    """

    ENGINE_ID = "parakeet"
    ENGINE_NAME = "Parakeet (NVIDIA NeMo)"

    def __init__(self):
        super().__init__()
        self._processor = None

    @classmethod
    def is_available(cls) -> bool:
        """
        Check if Parakeet/NeMo is available.

        On Windows native, checks for WSL with Ubuntu.
        On Linux/WSL, checks for nemo_toolkit import.
        """
        if sys.platform == "win32":
            # On Windows, can only use via WSL - check if WSL is available
            return _check_wsl_available()

        # On Linux (including WSL), check for NeMo
        try:
            import nemo.collections.asr as nemo_asr
            return True
        except ImportError:
            return False

    @classmethod
    def get_install_hint(cls) -> str:
        if sys.platform == "win32":
            return (
                "Parakeet requires WSL2 on Windows. Setup:\n"
                "1. Install WSL2: wsl --install\n"
                "2. In WSL: pip install nemo_toolkit[asr]\n"
                "3. Run Koe server in WSL: python -m src.server"
            )
        return "pip install nemo_toolkit[asr]"

    def load(self, model_name: str, device: str = "auto", compute_type: str = "float16") -> bool:
        """Load a Parakeet model."""
        if sys.platform == "win32":
            raise RuntimeError(
                "Parakeet cannot be loaded on Windows native. "
                "Run the Koe server in WSL2 instead."
            )

        try:
            import nemo.collections.asr as nemo_asr
            import torch

            # Handle auto device selection
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            print(f"[ParakeetEngine] Loading model '{model_name}' on {device}...")

            # Load the ASR model from HuggingFace/NGC
            self._model = nemo_asr.models.ASRModel.from_pretrained(model_name)

            # Move to device
            if device == "cuda" and torch.cuda.is_available():
                self._model = self._model.cuda()
            else:
                self._model = self._model.cpu()
                device = "cpu"

            # Set to eval mode
            self._model.eval()

            self._model_name = model_name
            self._device = device
            self._loaded = True

            print(f"[ParakeetEngine] Model loaded successfully on {self._device}")
            return True

        except Exception as e:
            print(f"[ParakeetEngine] Failed to load model: {e}")
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
        """Transcribe audio using Parakeet."""
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Parakeet is English-only
        if language and language not in ("en", "english", None):
            print(f"[ParakeetEngine] Warning: Parakeet only supports English, ignoring language={language}")

        # Ensure audio is float32 and normalized
        if audio.dtype != np.float32:
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            else:
                audio = audio.astype(np.float32)

        # Calculate duration
        duration = len(audio) / sample_rate

        try:
            import torch
            import tempfile
            import soundfile as sf
            import os

            # NeMo expects audio files, so we need to save to temp file
            # (NeMo does support numpy arrays via transcribe() but file path is more reliable)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                sf.write(f.name, audio, sample_rate)

            try:
                # Transcribe
                with torch.no_grad():
                    # Use transcribe method which handles preprocessing
                    transcriptions = self._model.transcribe([temp_path])

                if transcriptions and len(transcriptions) > 0:
                    # NeMo returns list of transcriptions (one per file)
                    result = transcriptions[0]
                    if isinstance(result, str):
                        text = result
                    elif hasattr(result, 'text'):
                        # Hypothesis object
                        text = result.text
                    else:
                        text = str(result)
                else:
                    text = ""

            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass

            # Parakeet doesn't provide word-level timestamps in basic mode
            # Create a single segment for the whole audio
            segments = [TranscriptionSegment(
                text=text.strip(),
                start=0.0,
                end=duration
            )] if text.strip() else []

            return TranscriptionResult(
                text=text.strip(),
                segments=segments,
                duration_seconds=duration
            )

        except Exception as e:
            print(f"[ParakeetEngine] Transcription error: {e}")
            raise

    def get_supported_models(self) -> List[ModelInfo]:
        """Get list of supported Parakeet models."""
        return PARAKEET_MODELS.copy()

    def unload(self) -> None:
        """Unload the model and free GPU memory."""
        super().unload()
        self._processor = None
        # Force garbage collection to free GPU memory
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
