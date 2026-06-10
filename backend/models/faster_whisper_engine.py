"""
Faster-Whisper engine – the recommended choice for NVIDIA GPU on Windows.
Uses CTranslate2 backend with int8 quantization for up to 4x speed over vanilla Whisper.

Key features:
- Int8 quantization cuts VRAM by ~40% with negligible accuracy loss
- Built-in Silero VAD to skip silence and reduce hallucination
- Supports all Whisper model sizes (tiny through large-v3)
- Batch processing for higher throughput
"""

from __future__ import annotations

import os
import time
from typing import Optional

import psutil
import numpy as np

from .base_engine import (
    BaseSTTEngine,
    EngineConfig,
    ModelInfo,
    TranscriptionResult,
    TranscriptionSegment,
)
from .registry import register_engine

# Attempt GPU memory tracking
try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False


# ─── Model catalogue ──────────────────────────────────────────────
_FASTER_WHISPER_MODELS = [
    ModelInfo(
        model_id="tiny",
        engine_type="faster-whisper",
        display_name="Whisper Tiny",
        param_count="39M",
        vram_requirement="~1 GB",
        languages=["multilingual (99+)"],
        description="Fastest, lowest resource. Good for real-time on edge.",
        recommended_for="real-time",
    ),
    ModelInfo(
        model_id="base",
        engine_type="faster-whisper",
        display_name="Whisper Base",
        param_count="74M",
        vram_requirement="~1 GB",
        languages=["multilingual (99+)"],
        description="Good balance for edge devices with limited GPU.",
        recommended_for="edge",
    ),
    ModelInfo(
        model_id="small",
        engine_type="faster-whisper",
        display_name="Whisper Small",
        param_count="244M",
        vram_requirement="~2 GB",
        languages=["multilingual (99+)"],
        description="Sweet spot: accuracy vs. resource for most edge GPUs.",
        recommended_for="edge",
    ),
    ModelInfo(
        model_id="medium",
        engine_type="faster-whisper",
        display_name="Whisper Medium",
        param_count="769M",
        vram_requirement="~5 GB",
        languages=["multilingual (99+)"],
        description="Strong accuracy for mid-range GPUs (RTX 3060+).",
        recommended_for="batch",
    ),
    ModelInfo(
        model_id="large-v3",
        engine_type="faster-whisper",
        display_name="Whisper Large-v3",
        param_count="1.55B",
        vram_requirement="~10 GB (2.5 GB int8)",
        languages=["multilingual (99+)"],
        description="Best accuracy. Needs 8+ GB VRAM (int8 recommended).",
        recommended_for="batch",
    ),
    ModelInfo(
        model_id="turbo",
        engine_type="faster-whisper",
        display_name="Whisper Turbo",
        param_count="809M",
        vram_requirement="~6 GB (2 GB int8)",
        languages=["multilingual (99+)"],
        description="large-v3 accuracy at 8x speed. Great for batch & near-real-time.",
        recommended_for="batch",
    ),
]


def _get_vram_usage_mb(device_index: int = 0) -> float:
    """Returns currently allocated GPU memory in MB, or 0."""
    if not _NVML_AVAILABLE:
        return 0.0
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 * 1024)
    except Exception:
        return 0.0


@register_engine("faster-whisper")
class FasterWhisperEngine(BaseSTTEngine):
    """faster-whisper engine backed by CTranslate2."""

    def __init__(self, config: EngineConfig):
        super().__init__(config)
        self._model_instance = None

    # ── abstract interface ──────────────────────────────────────

    def load(self) -> None:
        from faster_whisper import WhisperModel

        device = self.config.device
        compute = self.config.compute_type

        # Enforce int8 for large models unless user overrides
        if self.config.model_id in ("large-v3", "turbo") and compute == "float16":
            compute = "int8_float16"

        # faster-whisper API: `device` is a string ("cuda"/"cpu"),
        # `device_index` is a separate kwarg (int or list[int]).
        self._model_instance = WhisperModel(
            self.config.model_id,
            device=device,
            device_index=self.config.device_index,
            compute_type=compute,
            download_root=self.config.download_root,
            cpu_threads=os.cpu_count() or 4,
        )
        self._loaded = True
        print(f"[FasterWhisper] Loaded model '{self.config.model_id}' "
              f"on {device}:{self.config.device_index} ({compute})")

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        import soundfile as sf

        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        # Measure audio duration
        info = sf.info(audio_path)
        audio_duration = info.duration

        vram_before = _get_vram_usage_mb(self.config.device_index)
        t0 = time.perf_counter()

        segments_raw, lang_info = self._model_instance.transcribe(
            audio_path,
            language=self.config.language,
            beam_size=self.config.beam_size,
            vad_filter=self.config.vad_filter,
            vad_parameters=dict(
                threshold=0.5,
                min_speech_duration_ms=250,
            ),
        )

        segments = []
        full_text = ""
        for seg in segments_raw:
            segments.append(TranscriptionSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
                confidence=round(seg.avg_logprob, 4),
            ))
            full_text += seg.text

        elapsed = time.perf_counter() - t0
        vram_after = _get_vram_usage_mb(self.config.device_index)
        peak_vram = max(vram_after - vram_before, 0.0)

        return TranscriptionResult(
            text=full_text.strip(),
            language=lang_info.language,
            segments=segments,
            duration_seconds=round(audio_duration, 2),
            processing_time_seconds=round(elapsed, 3),
            real_time_factor=round(elapsed / max(audio_duration, 0.01), 3),
            peak_vram_mb=round(peak_vram, 1),
            metadata={
                "engine": "faster-whisper",
                "model": self.config.model_id,
                "compute_type": self.config.compute_type,
                "vad_filter": self.config.vad_filter,
            },
        )

    def unload(self) -> None:
        if self._model_instance is not None:
            # Close the model's underlying resources if available
            try:
                if hasattr(self._model_instance, "model") and self._model_instance.model is not None:
                    del self._model_instance.model
            except Exception:
                pass
            try:
                if hasattr(self._model_instance, "feature_extractor"):
                    del self._model_instance.feature_extractor
            except Exception:
                pass
            del self._model_instance
            self._model_instance = None
        self._loaded = False

        # Force CUDA memory release
        import gc
        gc.collect()

        # ctranslate2 uses its own CUDA allocator; try torch as fallback
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize(self.config.device_index)
        except Exception:
            pass

    @staticmethod
    def supported_models() -> list[ModelInfo]:
        return _FASTER_WHISPER_MODELS.copy()
