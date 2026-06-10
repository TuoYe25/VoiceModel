"""
ASR Hub – REST API server for the STT model testing platform.

Provides endpoints to:
- List available models with specs
- Transcribe audio files
- Run benchmarks across multiple models
- Stream real-time transcription results (SSE)
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

# Allow running from any directory: python backend/server.py or python server.py
_root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root_dir))
os.chdir(_root_dir)  # set working dir to project root

# Suppress HuggingFace symlink warning on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# Optionally set token for faster downloads: os.environ["HF_TOKEN"] = "your_token"

# ─── Inject CUDA DLLs alongside ctranslate2 ───────────────
# nvidia-* pip packages place cublas64_12.dll etc. inside
# site-packages/nvidia/*/bin/, which is NOT on the process DLL
# search path. Windows resolves dependent DLLs from the loading
# module's directory first, so we symlink / copy the needed DLLs
# into ctranslate2's package dir.
def _inject_cuda_dlls() -> None:
    import shutil
    import importlib.util as _u
    _ct2_dir = None
    try:
        _spec = _u.find_spec("ctranslate2")
        if _spec and _spec.origin:
            _ct2_dir = os.path.dirname(_spec.origin)
    except Exception:
        pass
    if not _ct2_dir:
        return
    _copied = 0
    for _pkg in ("nvidia.cublas", "nvidia.cuda_runtime", "nvidia.cudnn"):
        try:
            _spec = _u.find_spec(_pkg)
            if not _spec or not _spec.submodule_search_locations:
                continue
            for _loc in _spec.submodule_search_locations:
                _src = os.path.join(_loc, "bin")
                if not os.path.isdir(_src):
                    continue
                for _fn in os.listdir(_src):
                    if not _fn.endswith(".dll"):
                        continue
                    _src_path = os.path.join(_src, _fn)
                    _dst_path = os.path.join(_ct2_dir, _fn)
                    if os.path.isfile(_dst_path):
                        continue  # already present
                    try:
                        shutil.copy2(_src_path, _dst_path)
                        _copied += 1
                    except Exception:
                        pass
        except Exception:
            pass
    if _copied:
        print(f"[dll] Copied {_copied} CUDA DLL(s) into {_ct2_dir}", flush=True)

_inject_cuda_dlls()

# ─── GPU selection ─────────────────────────────────────────────
# Default = 1 (the 5070 Ti in your laptop). Override with `set ASR_GPU_INDEX=0`
# or pass ?device_index=0 in the request.
# NOTE: Do NOT set CUDA_VISIBLE_DEVICES – it remaps indices and conflicts
# with faster‑whisper’s `device_index` kwarg ("invalid device ordinal").
_DEFAULT_GPU_INDEX = int(os.environ.get("ASR_GPU_INDEX", "1"))
_INFERENCE_TIMEOUT_SEC = int(os.environ.get("ASR_TIMEOUT_SEC", "300"))

# Single shared executor so we don't block the event loop on future.result()
_inference_pool = ThreadPoolExecutor(max_workers=2)

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.models.base_engine import EngineConfig
from backend.models.registry import get_engine, list_engines
from backend.benchmark import BenchmarkRunner, BenchmarkResult

# ─── App Setup ────────────────────────────────────────────────
app = FastAPI(
    title="ASR Hub",
    description="Local STT/ASR model testing & benchmarking platform",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
def _shutdown_executor() -> None:
    _inference_pool.shutdown(wait=False)

# ─── Frontend path ────────────────────────────────────────────
_frontend_dir = _root_dir / "frontend"

# ─── State management ─────────────────────────────────────────
_active_engines: dict[str, object] = {}  # cache loaded engines by key
_upload_dir = _root_dir / "uploads"
_upload_dir.mkdir(exist_ok=True)


def _engine_key(engine_type: str, model_id: str, device: str) -> str:
    return f"{engine_type}:{model_id}:{device}"


def _ensure_wav(audio_path: Path) -> tuple[str, bool]:
    """Convert M4A / OGG / non‑soundfile formats to 16 kHz mono WAV.
    Returns (path_string, needs_cleanup).  The caller should delete the temp
    file when `needs_cleanup` is True."""
    ext = audio_path.suffix.lower()
    # soundfile can handle WAV, FLAC, OGG natively; skip conversion for known types
    if ext in (".wav", ".flac", ".ogg"):
        return str(audio_path), False

    converted = audio_path.with_suffix(".wav")
    try:
        from pydub import AudioSegment  # already in requirements.txt
        audio = AudioSegment.from_file(str(audio_path))
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(str(converted), format="wav")
        print(f"[audio] Converted {ext} → WAV ({converted.stat().st_size // 1024} KB)", flush=True)
        return str(converted), True
    except Exception:
        # If pydub/ffmpeg is missing, fall back to the original path and
        # let the engine’s built‑in loader try its luck.
        print(f"[audio] pydub conversion failed for {audio_path.name}, trying raw path", flush=True)
        return str(audio_path), False


# ─── REST Endpoints ───────────────────────────────────────────

@app.get("/api/models")
async def api_list_models():
    """List all registered engines and their supported models."""
    engines = list_engines()
    result = {}
    for eng_name, models in engines.items():
        result[eng_name] = [
            {
                "model_id": m.model_id,
                "display_name": m.display_name,
                "param_count": m.param_count,
                "vram_requirement": m.vram_requirement,
                "languages": m.languages,
                "description": m.description,
                "recommended_for": m.recommended_for,
            }
            for m in models
        ]
    return result


@app.post("/api/transcribe")
async def api_transcribe(
    file: UploadFile = File(...),
    engine_type: str = Query("faster-whisper"),
    model_id: str = Query("small"),
    language: Optional[str] = Query(None),
    device: str = Query("cuda"),
    device_index: Optional[int] = Query(None),
    compute_type: str = Query("int8"),
):
    """
    Transcribe an uploaded audio file using the specified engine & model.
    Returns: full text, segments with timestamps, and performance metrics.
    """
    # Save uploaded file
    ext = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    audio_path = _upload_dir / f"{int(time.time() * 1000)}{ext}"
    content = await file.read()
    audio_path.write_bytes(content)

    gpu_idx = device_index if device_index is not None else _DEFAULT_GPU_INDEX
    print(f"[transcribe] file={file.filename} engine={engine_type} model={model_id} "
          f"device={device}:{gpu_idx} compute={compute_type}", flush=True)

    # Convert non-WAV formats (M4A etc) so soundfile / the engine can read them
    audio_file, cleanup_wav = _ensure_wav(audio_path)

    try:
        config = EngineConfig(
            engine_type=engine_type,
            model_id=model_id,
            language=language,
            device=device,
            device_index=gpu_idx,
            compute_type=compute_type,
        )

        key = _engine_key(engine_type, model_id, f"{device}:{gpu_idx}")
        if key not in _active_engines:
            print(f"[transcribe] Loading {engine_type}/{model_id}...", flush=True)
            engine = get_engine(config)
            engine.load()
            _active_engines[key] = engine
            print(f"[transcribe] Model loaded.", flush=True)
        else:
            engine = _active_engines[key]

        print(f"[transcribe] Running inference on {os.path.basename(str(audio_path))}...", flush=True)
        # Run inference in a thread – non-blocking so the event loop stays alive
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(_inference_pool, engine.transcribe, audio_file)
        try:
            result = await asyncio.wait_for(fut, timeout=_INFERENCE_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Inference timed out after {_INFERENCE_TIMEOUT_SEC}s. "
                       f"The model may be stuck or the audio is too long.",
            )
        print(f"[transcribe] Done. text_len={len(result.text)} rtf={result.real_time_factor}", flush=True)

        return {
            "text": result.text,
            "language": result.language,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "confidence": s.confidence,
                }
                for s in result.segments
            ],
            "metrics": {
                "duration_seconds": result.duration_seconds,
                "processing_time_seconds": result.processing_time_seconds,
                "real_time_factor": result.real_time_factor,
                "peak_vram_mb": result.peak_vram_mb,
            },
            "metadata": result.metadata,
            "device": {"device": device, "device_index": gpu_idx},
        }
    except Exception as e:
        print(f"[transcribe] ERROR: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up original and any temp WAV
        try:
            audio_path.unlink()
        except Exception:
            pass
        if cleanup_wav:
            try:
                Path(audio_file).unlink()
            except Exception:
                pass


@app.post("/api/benchmark")
async def api_benchmark(
    file: UploadFile = File(...),
    ground_truth: Optional[str] = Form(None),
    models_json: Optional[str] = Form(None),
    device_index: Optional[int] = Query(None),
):
    """
    Run benchmark across multiple models on a single audio file.

    `models_json`: JSON string like:
    [{"engine_type":"faster-whisper","model_id":"small","compute_type":"int8"},
     {"engine_type":"faster-whisper","model_id":"medium","compute_type":"int8"}]
    """
    # Save audio
    ext = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    audio_path = _upload_dir / f"bench_{int(time.time() * 1000)}{ext}"
    content = await file.read()
    audio_path.write_bytes(content)

    gpu_idx = device_index if device_index is not None else _DEFAULT_GPU_INDEX
    print(f"[benchmark] file={file.filename} gpu_index={gpu_idx} models_json={models_json!r}", flush=True)

    # Convert non-WAV formats so soundfile / engines can read them
    audio_file, cleanup_wav = _ensure_wav(audio_path)

    try:
        # Parse model configs
        if models_json:
            model_list = json.loads(models_json)
        else:
            # Default: compare small/medium/turbo
            model_list = [
                {"engine_type": "faster-whisper", "model_id": "small", "compute_type": "int8"},
                {"engine_type": "faster-whisper", "model_id": "medium", "compute_type": "int8"},
                {"engine_type": "faster-whisper", "model_id": "turbo", "compute_type": "int8"},
            ]

        configs = [
            EngineConfig(
                engine_type=m.get("engine_type", "faster-whisper"),
                model_id=m["model_id"],
                compute_type=m.get("compute_type", "int8"),
                device=m.get("device", "cuda"),
                device_index=gpu_idx,
            )
            for m in model_list
        ]

        runner = BenchmarkRunner()
        # Run benchmark in a thread – non-blocking so the event loop stays alive
        # Per-model timeout: 8 min for large models (large-v3), min 5 min
        per_model_timeout = max(_INFERENCE_TIMEOUT_SEC, 480)
        bench_timeout = per_model_timeout * max(len(model_list), 1)

        loop = asyncio.get_running_loop()
        _bench_func = functools.partial(
            runner.run_benchmark,
            model_configs=configs,
            audio_files=[audio_file],
            ground_truths={audio_file: ground_truth} if ground_truth else None,
        )
        fut = loop.run_in_executor(_inference_pool, _bench_func)
        try:
            results = await asyncio.wait_for(fut, timeout=bench_timeout)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Benchmark timed out after {bench_timeout}s. "
                       f"Try fewer models or a shorter audio file.",
            )

        # Generate and return report
        report = runner.generate_report(results)

        return {
            "report": report,
            "results": [
                {
                    "engine": r.engine_name,
                    "model": r.model_id,
                    "audio_duration_s": r.audio_duration,
                    "processing_time_s": r.processing_time,
                    "real_time_factor": r.real_time_factor,
                    "peak_vram_mb": r.peak_vram_mb,
                    "wer": r.wer,
                    "success": r.success,
                    "error": r.error,
                }
                for r in results
            ],
            "device": {"device_index": gpu_idx},
        }
    except Exception as e:
        print(f"[benchmark] ERROR: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            audio_path.unlink()
        except Exception:
            pass
        if cleanup_wav:
            try:
                Path(audio_file).unlink()
            except Exception:
                pass


@app.get("/api/health")
async def health_check():
    """Health check with GPU status."""
    gpu_list = []
    default_index = _DEFAULT_GPU_INDEX
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_list.append({
                "index": i,
                "name": name,
                "vram_total_mb": round(mem.total / (1024 * 1024), 0),
                "vram_free_mb": round(mem.free / (1024 * 1024), 0),
                "vram_used_mb": round(mem.used / (1024 * 1024), 0),
                "utilization_pct": util.gpu,
            })
    except Exception as e:
        gpu_list = [{"error": str(e), "available": False}]

    active_gpu = next((g for g in gpu_list if g.get("index") == default_index), None)

    # Check installed engines
    engines_available = {}
    try:
        import faster_whisper  # noqa: F401
        engines_available["faster_whisper"] = True
    except ImportError:
        engines_available["faster_whisper"] = False

    try:
        import qwen_asr  # noqa: F401
        engines_available["qwen_asr"] = True
    except ImportError:
        engines_available["qwen_asr"] = False

    return {
        "status": "ok",
        "default_gpu_index": default_index,
        "active_gpu": active_gpu,
        "gpus": gpu_list,
        "engines_available": engines_available,
    }


@app.get("/api/recommendations")
async def api_recommendations(vram_mb: Optional[int] = Query(None)):
    """Get recommended models based on available VRAM."""
    all_models = list_engines()
    recs = []

    for eng_name, models in all_models.items():
        for m in models:
            recs.append({
                "engine": eng_name,
                "model_id": m.model_id,
                "display_name": m.display_name,
                "description": m.description,
                "vram_requirement": m.vram_requirement,
                "recommended_for": m.recommended_for,
                "param_count": m.param_count,
                "languages": m.languages,
            })

    # Rough VRAM filtering logic
    if vram_mb:
        def fits_vram(rec):
            req = rec["vram_requirement"].lower()
            if "~1" in req or "~2" in req:
                return vram_mb >= 2000
            if "~4" in req or "~5" in req:
                return vram_mb >= 6000
            if "~6" in req or "~8" in req:
                return vram_mb >= 8000
            if "~10" in req:
                return vram_mb >= 12000
            return True
        recs = [r for r in recs if fits_vram(r)]

    return {"recommendations": recs, "total": len(recs)}


# ─── Static frontend (mounted last so /api/* routes take priority) ──
class SPAStaticFiles(StaticFiles):
    """Serve static files; fall back to index.html for unknown paths (SPA)."""
    async def get_response(self, path, scope):
        try:
            return await super().get_response(path, scope)
        except Exception:
            # Fallback: serve index.html (for any unknown non-/api path)
            if not path.startswith("api"):
                return FileResponse(str(_frontend_dir / "index.html"))
            raise


app.mount("/", SPAStaticFiles(directory=str(_frontend_dir), html=True), name="frontend")


# ─── Startup ──────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
