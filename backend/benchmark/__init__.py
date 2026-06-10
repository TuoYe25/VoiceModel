"""
Benchmarking framework for STT models.

Measures:
- Real-Time Factor (RTF): processing_time / audio_duration
- Peak VRAM usage during inference
- Word Error Rate (WER) if ground truth provided
- CPU / GPU utilization during inference
- Model load time and model size on disk
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import psutil

from backend.models.base_engine import (
    BaseSTTEngine,
    EngineConfig,
    TranscriptionResult,
)
from backend.models.registry import get_engine


# ─── VRAM helpers ────────────────────────────────────────────
_GPU_UTIL_AVAILABLE = False
try:
    import GPUtil
    _GPU_UTIL_AVAILABLE = True
except ImportError:
    pass


def _gpu_stats(device_index: int = 0) -> dict:
    if not _GPU_UTIL_AVAILABLE:
        return {}
    try:
        gpus = GPUtil.getGPUs()
        if gpus and device_index < len(gpus):
            g = gpus[device_index]
            return {
                "gpu_index": device_index,
                "gpu_name": g.name,
                "gpu_utilization_pct": round(g.load * 100, 1),
                "gpu_memory_used_mb": round(g.memoryUsed, 1),
                "gpu_memory_total_mb": round(g.memoryTotal, 1),
                "gpu_temperature_c": g.temperature,
            }
    except Exception:
        pass
    return {}


# ─── WER calculation ─────────────────────────────────────────
def compute_wer(reference: str, hypothesis: str) -> float:
    """Simple Word Error Rate. Lower is better (0 = perfect match)."""
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    if not ref_words:
        return 0.0 if not hyp_words else float(len(hyp_words))

    # Levenshtein distance at word level
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,       # deletion
                d[i][j - 1] + 1,       # insertion
                d[i - 1][j - 1] + cost,  # substitution
            )

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


# ─── Benchmark result ────────────────────────────────────────
@dataclass
class BenchmarkResult:
    engine_name: str
    model_id: str
    audio_file: str
    audio_duration: float
    processing_time: float
    real_time_factor: float
    peak_vram_mb: float
    gpu_stats: dict = field(default_factory=dict)
    text_output: str = ""
    wer: Optional[float] = None
    model_load_time: float = 0.0
    cpu_percent: float = 0.0
    success: bool = True
    error: Optional[str] = None


# ─── Benchmark runner ────────────────────────────────────────
class BenchmarkRunner:
    """Runs a set of models against a set of audio files and collects metrics."""

    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_single(
        self,
        engine: BaseSTTEngine,
        audio_path: str,
        ground_truth_text: Optional[str] = None,
        warmup: bool = True,
    ) -> BenchmarkResult:
        """Benchmark a single engine+audio pair."""
        model_load_start = time.perf_counter()
        if not engine.is_loaded:
            engine.load()
        model_load_time = time.perf_counter() - model_load_start

        # Warmup
        if warmup:
            try:
                engine.transcribe(audio_path)
            except Exception:
                pass

        # Measure CPU usage during inference
        cpu_before = psutil.cpu_percent(interval=None)
        t0 = time.perf_counter()

        try:
            result = engine.transcribe(audio_path)
            success = True
            error = None
        except Exception as e:
            result = TranscriptionResult(text="", language="", duration_seconds=0.0)
            success = False
            error = str(e)

        elapsed = time.perf_counter() - t0
        cpu_after = psutil.cpu_percent(interval=None)

        gpu_stats = _gpu_stats(engine.config.device_index)

        # Compute WER if ground truth available
        wer = None
        if ground_truth_text and success:
            wer = compute_wer(ground_truth_text, result.text)

        # Get audio duration
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            audio_dur = info.duration
        except Exception:
            audio_dur = result.duration_seconds or 0.0

        return BenchmarkResult(
            engine_name=engine.config.engine_type,
            model_id=engine.config.model_id,
            audio_file=str(audio_path),
            audio_duration=round(audio_dur, 2),
            processing_time=round(elapsed, 3),
            real_time_factor=round(elapsed / max(audio_dur, 0.01), 3),
            peak_vram_mb=round(result.peak_vram_mb, 1),
            gpu_stats=gpu_stats,
            text_output=result.text[:200] if success else "",
            wer=round(wer, 4) if wer is not None else None,
            model_load_time=round(model_load_time, 2),
            cpu_percent=round((cpu_before + cpu_after) / 2, 1),
            success=success,
            error=error,
        )

    def run_benchmark(
        self,
        model_configs: list[EngineConfig],
        audio_files: list[str],
        ground_truths: Optional[dict[str, str]] = None,
        warmup: bool = True,
    ) -> list[BenchmarkResult]:
        """
        Run benchmarks across multiple model configs and audio files.

        Each model is isolated: a crash in one model won't lose results
        from previously-benchmarked models.

        Args:
            model_configs: List of engine configs to test
            audio_files: List of audio file paths
            ground_truths: Optional dict mapping audio_path -> reference text
            warmup: Whether to warm up each model
        """
        results: list[BenchmarkResult] = []

        for idx, cfg in enumerate(model_configs):
            print(f"\n{'='*60}")
            print(f"Benchmarking [{idx+1}/{len(model_configs)}]: {cfg.engine_type} / {cfg.model_id}")
            print(f"{'='*60}")

            engine = None
            try:
                engine = get_engine(cfg)

                for audio_path in audio_files:
                    gt = (ground_truths or {}).get(audio_path)
                    print(f"  → Audio: {os.path.basename(audio_path)}")

                    try:
                        result = self.run_single(engine, audio_path, ground_truth_text=gt, warmup=warmup)
                        results.append(result)

                        status = "OK" if result.success else "FAIL"
                        print(f"    {status} RTF={result.real_time_factor:.3f}, "
                              f"VRAM={result.peak_vram_mb:.0f}MB, "
                              f"WER={result.wer}, "
                              f"Time={result.processing_time:.2f}s")
                    except Exception as exc:
                        import traceback
                        print(f"    FAIL (audio error): {exc}")
                        results.append(BenchmarkResult(
                            engine_name=cfg.engine_type,
                            model_id=cfg.model_id,
                            audio_file=str(audio_path),
                            audio_duration=0.0,
                            processing_time=0.0,
                            real_time_factor=0.0,
                            peak_vram_mb=0.0,
                            success=False,
                            error=f"Audio error: {exc}",
                        ))
            except Exception as exc:
                import traceback
                print(f"  FAIL (model error): {exc}")
                # Record failure for this model without crashing
                for audio_path in audio_files:
                    results.append(BenchmarkResult(
                        engine_name=cfg.engine_type,
                        model_id=cfg.model_id,
                        audio_file=str(audio_path),
                        audio_duration=0.0,
                        processing_time=0.0,
                        real_time_factor=0.0,
                        peak_vram_mb=0.0,
                        success=False,
                        error=f"Model error: {exc}",
                    ))
            finally:
                if engine is not None:
                    try:
                        engine.unload()
                    except Exception:
                        pass

        return results

    def save_results(self, results: list[BenchmarkResult], filename: str = "benchmark.json") -> str:
        """Save results as JSON."""
        path = self.output_dir / filename
        data = []
        for r in results:
            data.append({
                "engine": r.engine_name,
                "model": r.model_id,
                "audio": os.path.basename(r.audio_file),
                "audio_duration_s": r.audio_duration,
                "processing_time_s": r.processing_time,
                "real_time_factor": r.real_time_factor,
                "peak_vram_mb": r.peak_vram_mb,
                "gpu": r.gpu_stats,
                "wer": r.wer,
                "model_load_time_s": r.model_load_time,
                "cpu_percent": r.cpu_percent,
                "success": r.success,
                "error": r.error,
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return str(path)

    def generate_report(self, results: list[BenchmarkResult]) -> str:
        """Generate a Markdown comparison report."""
        lines = [
            "# STT Model Benchmark Report\n",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            "---\n",
            "## Summary Table\n",
            "| Engine | Model | Audio | Duration | Proc Time | RTF | VRAM (MB) | WER | Success |",
            "|--------|-------|-------|----------|-----------|-----|-----------|-----|---------|",
        ]

        for r in results:
            wer_str = f"{r.wer:.4f}" if r.wer is not None else "N/A"
            status = "OK" if r.success else "FAIL"
            lines.append(
                f"| {r.engine_name} | {r.model_id} | {r.audio_file} | "
                f"{r.audio_duration:.1f}s | {r.processing_time:.2f}s | "
                f"{r.real_time_factor:.3f}x | {r.peak_vram_mb:.0f} | "
                f"{wer_str} | {status} |"
            )

        lines.append("\n---\n## Recommendations\n")
        lines.append(self._generate_recommendations(results))

        return "\n".join(lines)

    @staticmethod
    def _generate_recommendations(results: list[BenchmarkResult]) -> str:
        lines = []
        # Sort by WER (lowest), then RTF
        successful = [r for r in results if r.success]
        sorted_wer = sorted(successful, key=lambda r: r.wer if r.wer is not None else 999)
        sorted_speed = sorted(successful, key=lambda r: r.real_time_factor)

        if sorted_wer:
            best = sorted_wer[0]
            lines.append(f"**Highest Accuracy**: `{best.model_id}` (WER={best.wer})")

        if sorted_speed:
            best = sorted_speed[0]
            lines.append(f"**Fastest**: `{best.model_id}` (RTF={best.real_time_factor}x)")

        # Edge recommendation
        edge = [r for r in successful if r.peak_vram_mb < 4000 and r.real_time_factor < 1.0]
        if edge:
            lines.append(
                f"**Edge Recommended**: `{edge[0].model_id}` "
                f"(VRAM={edge[0].peak_vram_mb:.0f}MB, RTF={edge[0].real_time_factor:.3f}x)"
            )

        return "\n".join(lines)
