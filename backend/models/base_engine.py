"""
Base STT Engine abstraction layer.
All engines implement this interface, enabling a unified benchmarking and testing API.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EngineConfig:
    """Configuration for loading and running an STT engine."""
    model_id: str                          # e.g. "small", "large-v3", "Qwen3-ASR-1.7B"
    engine_type: str                       # "faster-whisper" | "qwen-asr" | "voxtral"
    language: Optional[str] = None         # ISO language code or None for auto-detect
    device: str = "cuda"                   # "cuda" | "cpu"
    device_index: int = 0                  # which physical GPU (when device="cuda")
    compute_type: str = "float16"          # "float16" | "int8" | "int8_float16"
    beam_size: int = 5
    vad_filter: bool = True                # Voice Activity Detection
    download_root: Optional[str] = None    # Custom model cache directory


@dataclass
class TranscriptionSegment:
    """A single transcribed segment with timing info."""
    start: float  # seconds
    end: float    # seconds
    text: str
    confidence: float = 1.0


@dataclass
class TranscriptionResult:
    """Complete transcription result from an engine."""
    text: str
    language: str
    segments: list[TranscriptionSegment] = field(default_factory=list)
    duration_seconds: float = 0.0          # audio duration
    processing_time_seconds: float = 0.0   # wall-clock inference time
    real_time_factor: float = 0.0          # processing_time / duration (lower = faster)
    peak_vram_mb: float = 0.0              # peak GPU memory during inference
    metadata: dict = field(default_factory=dict)


@dataclass
class ModelInfo:
    """Static information about a model for display / selection."""
    model_id: str
    engine_type: str
    display_name: str
    param_count: str        # e.g. "244M"
    vram_requirement: str   # e.g. "~2 GB"
    languages: list[str]
    description: str
    recommended_for: str    # "real-time", "batch", "edge", "multilingual"


class BaseSTTEngine(ABC):
    """Abstract base for all STT engine implementations."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self._model = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory (GPU/CPU). May download weights on first call."""
        ...

    @abstractmethod
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe an audio file. Returns structured result with metrics."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release GPU/CPU resources."""
        ...

    @staticmethod
    @abstractmethod
    def supported_models() -> list[ModelInfo]:
        """Return list of models this engine supports."""
        ...

    def benchmark_transcribe(self, audio_path: str, warmup: bool = False) -> TranscriptionResult:
        """Transcribe with additional benchmarking instrumentation."""
        if not self._loaded:
            self.load()

        # Optionally run a throwaway inference to warm GPU / CUDA context
        if warmup:
            self.transcribe(audio_path)

        result = self.transcribe(audio_path)

        return result

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(model={self.config.model_id}, device={self.config.device})>"
