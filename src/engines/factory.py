"""
Engine factory for creating transcription engines.

Provides dynamic engine registration based on available dependencies.
"""

import sys
from typing import Dict, List, Optional, Type
from .base import TranscriptionEngine, ModelInfo, EngineNotAvailableError


# Registry of available engines (populated by register_engine)
_engine_registry: Dict[str, Type[TranscriptionEngine]] = {}


def register_engine(engine_class: Type[TranscriptionEngine]) -> Type[TranscriptionEngine]:
    """
    Register an engine class in the registry.

    Use as a decorator:
        @register_engine
        class MyEngine(TranscriptionEngine):
            ENGINE_ID = "my_engine"
    """
    _engine_registry[engine_class.ENGINE_ID] = engine_class
    return engine_class


def get_available_engines() -> List[str]:
    """
    Get list of engine IDs that are available (dependencies installed).

    Returns:
        List of engine IDs that can be used
    """
    available = []
    for engine_id, engine_class in _engine_registry.items():
        if engine_class.is_available():
            available.append(engine_id)
    return available


def is_engine_available(engine_id: str) -> bool:
    """
    Check if a specific engine is available.

    Args:
        engine_id: The engine ID to check

    Returns:
        True if engine is registered and dependencies are installed
    """
    if engine_id not in _engine_registry:
        return False
    return _engine_registry[engine_id].is_available()


def create_engine(engine_id: str) -> TranscriptionEngine:
    """
    Create an instance of the specified engine.

    Args:
        engine_id: The engine ID to instantiate

    Returns:
        An instance of the requested engine

    Raises:
        EngineNotAvailableError: If engine is not available
        ValueError: If engine ID is unknown
    """
    if engine_id not in _engine_registry:
        available = list(_engine_registry.keys())
        raise ValueError(f"Unknown engine '{engine_id}'. Available: {available}")

    engine_class = _engine_registry[engine_id]

    if not engine_class.is_available():
        raise EngineNotAvailableError(
            engine_id,
            engine_class.get_install_hint()
        )

    return engine_class()


def get_engine_class(engine_id: str) -> Optional[Type[TranscriptionEngine]]:
    """
    Get the class for a specific engine (without instantiating).

    Args:
        engine_id: The engine ID

    Returns:
        The engine class, or None if not found
    """
    return _engine_registry.get(engine_id)


def get_all_engines() -> Dict[str, Type[TranscriptionEngine]]:
    """
    Get all registered engines (available or not).

    Returns:
        Dictionary of engine_id -> engine_class
    """
    return dict(_engine_registry)


def get_all_models() -> List[ModelInfo]:
    """
    Get list of all models from all available engines.

    Returns:
        Combined list of ModelInfo from all engines
    """
    all_models = []
    for engine_id in get_available_engines():
        try:
            engine = create_engine(engine_id)
            all_models.extend(engine.get_supported_models())
        except Exception:
            pass
    return all_models


def get_engine_for_model(model_id: str) -> Optional[str]:
    """
    Determine which engine supports a given model ID.

    Args:
        model_id: The model ID to look up

    Returns:
        Engine ID that supports this model, or None
    """
    for engine_id in get_available_engines():
        try:
            engine = create_engine(engine_id)
            for model_info in engine.get_supported_models():
                if model_info.id == model_id:
                    return engine_id
        except Exception:
            pass
    return None


def get_default_engine() -> str:
    """
    Get the default engine ID (first available).

    Returns:
        Engine ID of the default engine

    Raises:
        RuntimeError: If no engines are available
    """
    available = get_available_engines()
    if not available:
        raise RuntimeError("No transcription engines available. Install faster-whisper or nemo_toolkit.")

    # Prefer whisper as default
    if "whisper" in available:
        return "whisper"

    return available[0]


# Import engines to register them (lazy import to avoid dependency errors)
def _register_engines():
    """Import engine modules to register them."""
    try:
        from . import whisper_engine
    except ImportError:
        pass

    try:
        from . import parakeet_engine
    except ImportError:
        pass

    try:
        from . import mlx_engine
    except ImportError:
        pass


# Register engines on module load
_register_engines()
