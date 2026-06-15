# Development Log — Edge STT/ASR Model Deployment

## Assignment Summary

Build a platform to research, benchmark, and test open-source STT/ASR models for edge deployment on Windows+GPU, with a GUI demo ready for Electron.js integration.

---

## Phase 1: Research & Planning

### Research Sources Consulted
- **Mistral Voxtral Transcribe 2**: Only Voxtral Realtime (4B params) is open-source for local deployment. Mini Transcribe V2 is API-only. Realtime model is promising for streaming but 4B params is heavy for edge.
- **OpenAI Whisper**: Gold standard. 6 model sizes from 39M to 1.55B params. MIT license. Broad language support (99+).
- **faster-whisper**: CTranslate2 backend. 4× faster than vanilla Whisper, 40% less VRAM with int8 quantization. **Best choice for NVIDIA GPU on Windows**.
- **whisper.cpp**: Pure C/C++, ideal for CPU/Apple Silicon. Less optimal for NVIDIA GPU vs faster-whisper.
- **Qwen3-ASR**: From Alibaba. Two sizes (0.6B/1.7B). SOTA on Chinese, supports 52 languages. Excellent multilingual alternative.
- **Benchmark data** from Northflank, PromptQuorum, and community tests.

### Key Decision: Engine Architecture
Designed an **abstract engine interface** (`BaseSTTEngine`) so any ASR backend can be plugged in uniformly. This enables:
- Consistent benchmarking across engines
- Easy addition of new models (Voxtral, Moonshine, etc.)
- Clean separation between engine logic and API/UI layers

### Model Selection Strategy
```
Edge-friendly:      tiny (39M), base (74M), small (244M)
Balanced:           medium (769M), turbo (809M)
Maximum accuracy:   large-v3 (1.55B), Qwen3-ASR 1.7B
Chinese-optimized:  Qwen3-ASR 0.6B / 1.7B
```

---

## Phase 2: Implementation

### Backend (`backend/`)

**`base_engine.py`** — Core abstraction:
- `EngineConfig`: model_id, device, compute_type, language, VAD settings
- `TranscriptionResult`: text, segments with timestamps, RTF, peak VRAM
- `BaseSTTEngine`: abstract class with `load()`, `transcribe()`, `unload()`, `supported_models()`
- `ModelInfo`: static model metadata for the UI catalogue

**`registry.py`** — Dynamic engine registration:
- Decorator-based registration (`@register_engine("name")`)
- Enables discovering engines at runtime and adding new ones without modifying existing code

**`faster_whisper_engine.py`** — Primary engine:
- Wraps `faster_whisper.WhisperModel` (CTranslate2)
- Auto-enforces int8 for large models to save VRAM
- Built-in Silero VAD for silence filtering
- pynvml integration for VRAM tracking
- Returns structured `TranscriptionResult` with all metrics

**`qwen_asr_engine.py`** — Alternative engine:
- Wraps `qwen_asr.Qwen3ASRModel`
- bfloat16 precision, CUDA device mapping
- Proper cleanup (`gc.collect()` + `torch.cuda.empty_cache()`)
- Graceful fallback for missing qwen_asr package

**Benchmark framework (`benchmark/`)**:
- Word Error Rate via Levenshtein distance at word level
- GPU utilization tracking via GPUtil
- `BenchmarkRunner`: orchestrates multi-model, multi-audio benchmarks
- Auto-generates Markdown comparison reports with recommendations
- JSON export for programmatic consumption

**FastAPI server (`server.py`)**:
- `/api/models` — model catalogue
- `/api/transcribe` — single-file transcription with metrics
- `/api/benchmark` — multi-model comparison
- `/api/health` — GPU status + engine availability check
- `/api/recommendations` — VRAM-filtered model suggestions
- CORS enabled for Electron/frontend access
- Engine caching to avoid reloading models between requests

### CLI Runner (`run_benchmark.py`)
- Standalone command-line benchmark tool
- Supports custom model lists, compute types, ground truth for WER
- Outputs JSON results + Markdown report to disk

### Frontend (`frontend/`)

**Design decisions:**
- Pure HTML/CSS/JS — zero framework dependencies
- Dark theme matching modern desktop app aesthetics (GitHub/Discord-inspired)
- Single-page with tab navigation (Transcribe / Benchmark)
- Responsive card-based layout

**Key features:**
- **Model Grid**: Visual model catalogue with VRAM and param info, recommended-for badges
- **Drag & Drop Upload**: Audio file ingestion with size display
- **Transcription Results**: Metrics dashboard (RTF, VRAM, processing time), timestamped segments, copy-to-clipboard
- **Benchmark Runner**: Multi-model comparison table with best-in-class highlighting (green for WER, blue for RTF)
- **Status Bar**: Live GPU name and VRAM info from health endpoint
- **Toast Notifications**: Success/error feedback
- **Fallback Model Data**: Works when server is offline (displays static model catalogue)

### Electron Wrapper (`electron/`)
- `main.js`: Spawns Python backend as child process, creates BrowserWindow
- `preload.js`: Safe IPC bridge via contextBridge for file dialogs
- `package.json`: electron-builder config for Windows (NSIS) and macOS (DMG) packaging
- Clean lifecycle: auto-start/stop backend with app

---

## Phase 3: Key Findings & Recommendations

### Benchmark Expectations (from community data)

| Model | WER (LibriSpeech clean) | RTF (RTX 4070, int8) | VRAM |
|-------|------------------------|---------------------|------|
| small | 3.4% | 0.16× | 2 GB |
| medium | 2.9% | 0.33× | 3 GB |
| large-v3 | 2.5% | 0.52× | 2.5 GB |
| turbo | 2.5% | 0.08× | 2 GB |

### Edge Deployment Strategy

1. **For most users**: `faster-whisper small` (int8) — 2 GB VRAM, RTF ~0.16, WER 3.4%. Fast enough for near-real-time, accurate enough for most use cases.
2. **For Chinese users**: `Qwen3-ASR 0.6B` — better Chinese accuracy than Whisper, similar resource footprint.
3. **For batch processing**: `faster-whisper turbo` (int8) — large-v3 accuracy at 8× speed, 2 GB VRAM.
4. **For maximum accuracy**: `faster-whisper large-v3` (int8) — 2.5% WER, need 8+ GB total VRAM.

### To Run Actual Benchmarks
1. Prepare test audio files in `test_audio/`
2. Install dependencies: `pip install -r backend/requirements.txt`
3. Run: `python run_benchmark.py --audio test_audio/sample.wav --models small medium turbo`
4. Results will appear in `benchmark_results/`

---

## AI-Assisted Development Process

### How AI Tools Were Used

| Stage | AI Role | Tools |
|-------|---------|-------|
| **Research** | Web-fetched model docs, benchmarks, and community comparisons | Web fetch + web search to gather real-time model data |
| **Architecture** | Designed engine abstraction with plugin registry pattern | Reasoning about extensibility and clean separation |
| **Implementation** | Generated all Python/JS/HTML/CSS code for backend, frontend, and Electron | Code generation with consistent patterns across files |
| **Review** | Validated API contracts, engine interface consistency, and frontend-backend integration | Cross-file consistency checks |

### Prompting Strategy
- **Research phase**: Broad queries to gather model landscape, then targeted fetches for specific model details
- **Architecture phase**: Defined requirements first (abstract interface, pluggable engines, consistent metrics), then implemented
- **Implementation phase**: Built bottom-up — base classes → engines → benchmark → server → frontend → Electron
- **Integration phase**: Verified that frontend API calls match backend endpoint signatures, data field names, and error handling

### Lessons Learned
1. **Pluggable architecture pays off**: Adding a new engine (Qwen3) was trivial once the `BaseSTTEngine` interface was defined
2. **Frontend fallback is important**: The UI should work even when the backend is offline (shows model catalogue, graceful errors)
3. **Metrics should be universal**: RTF and peak VRAM are the two most actionable metrics for edge deployment decisions

---

## Phase 4: Multi-GPU Support & Stuck-Loading Fix (2026-06-09)

### Problem Reported
- Benchmark tab appeared to spin forever
- Transcribe results never showed up
- User has two GPUs: RTX 3050 Ti (4GB) and RTX 5070 Ti (12GB); the code was always using index 0 (3050 Ti, too small for medium/large models)

### Root Causes
1. **Hard-coded GPU index 0** in `_get_vram_usage_mb`, `qwen_asr_engine.py`, and `BenchmarkRunner._gpu_stats`. CUDA_VISIBLE_DEVICES was never set, so CTranslate2/faster-whisper also defaulted to the 3050 Ti → OOM for medium/turbo/large.
2. **faster-whisper's `device` arg** was a plain string (`"cuda"`); the engine accepts a tuple `("cuda", index)` for multi-GPU selection but the code never used it.
3. **Engine cache key** (`{engine_type}:{model_id}:{device}`) didn't include the GPU index, so the first loaded model was reused on the wrong card after switching.
4. **No progress feedback** — the synchronous endpoints blocked without server logs, making "downloading model" indistinguishable from "stuck".

### Fixes Applied
- Added `device_index: int = 0` to `EngineConfig` (in `base_engine.py`).
- All engines now pass `device_index` to pynvml and to faster-whisper's `device=("cuda", index)`.
- Engine cache key changed to `{engine_type}:{model_id}:cuda:{idx}`.
- New env var `ASR_GPU_INDEX` (default `1` = 5070 Ti) sets `CUDA_VISIBLE_DEVICES` before torch/CTranslate2 import.
- `/api/health` now returns `gpus[]` with index, name, free/used/total VRAM, and utilization for every card; plus `active_gpu` and `default_gpu_index`.
- `/api/transcribe` and `/api/benchmark` accept a `device_index` query parameter that overrides the server default per-request.
- Added `print(..., flush=True)` log lines in transcribe/benchmark so users can see download/load/inference progress in the terminal.
- **Frontend**: GPU `<select>` in the sidebar status bar (auto-populated from `/api/health`). The selected value is sent on every transcribe/benchmark request.

### How to Use
- Default: server picks GPU #1 (5070 Ti). Just restart the server.
- Per-request override in UI: pick a different GPU from the dropdown in the sidebar.
- Permanent override: `set ASR_GPU_INDEX=0` before starting the server.
- Verify with: `curl http://localhost:8765/api/health` → check `active_gpu.name`.

---

## Phase 5: Real-time Recording & Live Transcription (2026-06-15)

### Feature Summary
Added a full-duplex real-time recording pipeline: microphone capture → WebSocket streaming →
incremental ASR transcription → live UI updates, all without touching the disk more than
temporarily.

### Architecture

```
Browser                         Backend /ws/realtime
  │ (MediaRecorder, WebM/Opus)          │
  │──── binary chunk (1s) ─────────────►│ Accumulate
  │──── binary chunk                     │
  │──── binary chunk                     │
  │──── binary chunk (every 4th) ──────►│ Merge → WebM→WAV → Transcribe
  │◄─── {"type":"result", partial} ────│ Streaming JSON
  │ ... (loop) ...                      │
  │──── "STOP" (text) ─────────────────►│ Final transcription
  │◄─── {"type":"result", final} ──────│ Full result with segments
```

### Backend Changes (`server.py`)
- **New WebSocket endpoint** `/ws/realtime`: Accepts binary audio chunks (WebM/Opus from
  MediaRecorder), accumulates every 4 chunks (~3s audio), converts to 16kHz mono WAV via
  pydub, runs inference via `loop.run_in_executor()`, and streams JSON results back.
- **Protocol messages**: `ready` (model loaded), `result` (partial/final transcription with
  segments & RTF), `status` (no new speech), `error`, `keepalive` (30s timeout), `done`.
- **Model persistence**: Engine loaded once per connection, reused across all chunks.
- **SPAStaticFiles** class: Serves frontend files with fallback to `index.html` for
  client-side routing.

### Frontend Changes (`frontend/`)

**`index.html`** — New Realtime tab:
- Model selection grid (same card layout as Transcribe tab)
- Recording controls: Start/Stop button, MM:SS timer, recording status indicator
- Live transcript display with language badge and copy button
- Segments detail panel with timestamps

**`app.js`** — Core realtime logic (+~370 lines):
- `getWsUrl()`: Auto-derives WebSocket URL from page protocol (ws:// or wss://)
- `startRecording()`: `getUserMedia` → `MediaRecorder` (WebM/Opus, 1s intervals) →
  WebSocket handshake with `ready` acknowledgment → binary chunk streaming
- `stopRecording()` / `stopRecordingInternal()`: Sends "STOP" text → processes final
  result → closes connection → resets UI
- `handleRealtimeMessage()`: Dispatches `result` (partial renders live text + segments;
  final renders complete transcript), `status`, `error`, `keepalive`
- `renderRealtimeResult()`: Renders full transcription, language badge, timestamped segments
- `updateRecordTimer()`: 200ms interval MM:SS display
- Single `handleRecordClick` dispatcher to switch between start/stop

**`styles.css`** — New styles (+~60 lines):
- `.realtime-controls`: Flex layout for record button + timer + status
- `.record-timer`: Monospace font counter
- `.btn-danger`: Red record/stop button
- `.recording-dot` + `@keyframes pulse-dot`: Breathing red dot animation
- `.realtime-indicator`: "Recording in progress..." alert bar
- `.placeholder-text`: Muted italic placeholder for empty transcript

### Bug Fixes
- **Stop button unresponsive**: Button initially only had `startRecording` bound.
  Fixed with `handleRecordClick` dispatcher that checks `state.isRecording` to call
  `startRecording()` or `stopRecording()`.

### Technical Decisions
- **WebM/Opus instead of raw PCM**: MediaRecorder's native format; pydub handles the
  conversion to 16kHz mono WAV on the server side.
- **4-chunk batching**: Balances latency (~3s) against inference overhead — avoids
  calling the model on every 1s chunk.
- **Single WS connection**: Model stays loaded for the session; no reload between chunks.
- **No WebRTC**: Simpler WebSocket approach sufficient for localhost; WebRTC would add
  unnecessary complexity for a local desktop app.

---

*Document generated 2026-06-08, updated 2026-06-15 as part of the Altar Voice Model project.*
