"""
Qwen3-ASR engine – state-of-the-art multilingual ASR from Alibaba Cloud.

Features:
- 1.7B & 0.6B model variants, ideal for edge deployment
- Excellent Chinese + 52 languages/dialects
- Streaming and batch modes via vLLM or Transformers backend
- Built-in language detection and forced alignment
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .base_engine import (
    BaseSTTEngine,
    EngineConfig,
    ModelInfo,
    TranscriptionResult,
    TranscriptionSegment,
)
from .registry import register_engine


_QWEN_ASR_MODELS = [
    ModelInfo(
        model_id="Qwen3-ASR-0.6B",
        engine_type="qwen-asr",
        display_name="Qwen3-ASR 0.6B",
        param_count="600M",
        vram_requirement="~4 GB",
        languages=["52 languages/dialects", "22 Chinese dialects"],
        description="Lightweight, excellent for edge. Great Chinese dialect support.",
        recommended_for="edge",
    ),
    ModelInfo(
        model_id="Qwen3-ASR-1.7B",
        engine_type="qwen-asr",
        display_name="Qwen3-ASR 1.7B",
        param_count="1.7B",
        vram_requirement="~8 GB",
        languages=["52 languages/dialects", "22 Chinese dialects"],
        description="SOTA open-source ASR. Best for Chinese & multilingual.",
        recommended_for="batch",
    ),
]


def _get_vram_usage_mb(device_index: int = 0) -> float:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 * 1024)
    except Exception:
        return 0.0


@register_engine("qwen-asr")
class Qwen3ASREngine(BaseSTTEngine):
    """Qwen3-ASR engine using HuggingFace Transformers or vLLM backend."""

    def __init__(self, config: EngineConfig):
        super().__init__(config)
        self._processor = None
        self._model = None

    # ── abstract interface ──────────────────────────────────────

    def load(self) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        target_device = (
            f"cuda:{self.config.device_index}" if self.config.device == "cuda" else "cpu"
        )
        self._model = Qwen3ASRModel.from_pretrained(
            self.config.model_id,
            dtype=torch.bfloat16,
            device_map=target_device,
        )
        self._loaded = True
        print(f"[Qwen3-ASR] Loaded model '{self.config.model_id}' on {target_device}")

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        import soundfile as sf

        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        info = sf.info(audio_path)
        audio_duration = info.duration

        vram_before = _get_vram_usage_mb(self.config.device_index)
        t0 = time.perf_counter()

        results = self._model.transcribe(
            audio=audio_path,
            language=self.config.language,
        )

        elapsed = time.perf_counter() - t0
        vram_after = _get_vram_usage_mb(self.config.device_index)
        peak_vram = max(vram_after - vram_before, 0.0)

        first = results[0] if results else None
        segments = []
        full_text = ""
        if first:
            full_text = first.text
            # Qwen3-ASR may return chunks; convert to segments
            for i, chunk in enumerate(first.chunks if hasattr(first, "chunks") else []):
                segments.append(TranscriptionSegment(
                    start=chunk.get("timestamp", (i, i + 1))[0] if isinstance(chunk, dict) else i,
                    end=chunk.get("timestamp", (i, i + 1))[1] if isinstance(chunk, dict) else i + 1,
                    text=chunk.get("text", "") if isinstance(chunk, dict) else str(chunk),
                ))

        return TranscriptionResult(
            text=full_text.strip(),
            language=first.language if first else "unknown",
            segments=segments,
            duration_seconds=round(audio_duration, 2),
            processing_time_seconds=round(elapsed, 3),
            real_time_factor=round(elapsed / max(audio_duration, 0.01), 3),
            peak_vram_mb=round(peak_vram, 1),
            metadata={
                "engine": "qwen-asr",
                "model": self.config.model_id,
            },
        )

    def unload(self) -> None:
        self._model = None
        self._loaded = False
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def supported_models() -> list[ModelInfo]:
        return _QWEN_ASR_MODELS.copy()
