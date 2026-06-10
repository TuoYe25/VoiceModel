"""
Engine registry – allows dynamic registration and lookup of STT engine classes.
"""

from __future__ import annotations

from typing import Dict, Optional, Type

from .base_engine import BaseSTTEngine, EngineConfig, ModelInfo


_registry: Dict[str, Type[BaseSTTEngine]] = {}


def register_engine(name: str):
    """Decorator: register an engine class under a name (e.g. 'faster-whisper').

    Usage:
        @register_engine("faster-whisper")
        class FasterWhisperEngine(BaseSTTEngine):
            ...
    """
    def decorator(cls: Type[BaseSTTEngine]) -> Type[BaseSTTEngine]:
        _registry[name] = cls
        return cls
    return decorator


def get_engine(config: EngineConfig) -> BaseSTTEngine:
    """Instantiate the engine matching config.engine_type."""
    cls = _registry.get(config.engine_type)
    if cls is None:
        available = ", ".join(_registry.keys())
        raise ValueError(
            f"Unknown engine type '{config.engine_type}'. Available: {available}"
        )
    return cls(config)


def list_engines() -> Dict[str, list[ModelInfo]]:
    """Return {engine_name: [ModelInfo, ...]} for all registered engines."""
    result: Dict[str, list[ModelInfo]] = {}
    for name, cls in _registry.items():
        try:
            result[name] = cls.supported_models()
        except Exception:
            result[name] = []
    return result
