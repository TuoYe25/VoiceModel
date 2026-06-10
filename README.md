# ASR Hub — Edge Device STT/ASR Model Deployment

> Open-source speech-to-text benchmarking & testing platform for edge devices (Windows + GPU).  
> Built for quick model evaluation with an Electron-ready desktop GUI.

## Quick Start

```bash
# 1. Install Python dependencies
cd backend
pip install -r requirements.txt

# 2. Start the API server
python server.py

# 3. Open the frontend
#    Option A: Open frontend/index.html directly in a browser
#    Option B: Use the Electron wrapper
cd electron && npm install && npm start
```

Then open `http://localhost:8765/docs` for the Swagger API docs, or `frontend/index.html` for the GUI.

---

## Project Architecture

```
Voice Model/
├── backend/                        # Python STT engine & API server
│   ├── models/
│   │   ├── base_engine.py          # Abstract engine interface
│   │   ├── faster_whisper_engine.py # CTranslate2-backed Whisper
│   │   ├── qwen_asr_engine.py       # Qwen3-ASR (HuggingFace)
│   │   └── registry.py             # Dynamic engine registry
│   ├── benchmark/
│   │   └── __init__.py             # Benchmark runner + WER + GPU metrics
│   ├── server.py                   # FastAPI REST server
│   └── requirements.txt
├── frontend/
│   ├── index.html                  # Single-page GUI
│   ├── styles.css                  # Dark theme styles
│   └── app.js                      # Client-side logic
├── electron/
│   ├── main.js                     # Electron main process
│   ├── preload.js                  # Context bridge
│   └── package.json                # Electron config
├── run_benchmark.py                # CLI benchmark runner
├── docs/
│   └── development-log.md          # AI-assisted development journal
└── README.md
```

---

## Supported Models

| Model | Params | VRAM | Speed | Languages | Best For |
|-------|--------|------|-------|-----------|----------|
| Whisper Tiny | 39M | ~1 GB | 10× real-time | 99+ | Real-time edge |
| Whisper Base | 74M | ~1 GB | 7× | 99+ | Low-resource edge |
| Whisper Small | 244M | ~2 GB | 4× | 99+ | **Edge sweet spot** |
| Whisper Medium | 769M | ~5 GB | 2× | 99+ | Mid-range GPU batch |
| Whisper Large-v3 | 1.55B | ~10 GB | 1× | 99+ | Best accuracy |
| Whisper Turbo | 809M | ~6 GB | 8× | 99+ | Fast + accurate |
| Qwen3-ASR 0.6B | 600M | ~4 GB | — | 52 | Excellent Chinese |
| Qwen3-ASR 1.7B | 1.7B | ~8 GB | — | 52 | SOTA multilingual |

*VRAM figures are for int8 quantization with faster-whisper; float16 requires ~40% more.*

---

## Benchmarking Methodology

The benchmark runner measures:

| Metric | Description |
|--------|-------------|
| **RTF** (Real-Time Factor) | Processing time ÷ audio duration. < 0.3 = real-time capable |
| **Peak VRAM** | Maximum GPU memory allocated during inference |
| **WER** (Word Error Rate) | Levenshtein distance at word level (needs ground truth) |
| **Model Load Time** | Time to initialize model on GPU |
| **CPU %** | Average CPU utilization during inference |

To run benchmarks via CLI:

```bash
python run_benchmark.py --audio test.wav --models small medium turbo
python run_benchmark.py --audio test.wav --ground-truth "expected text" --models small medium large-v3
```

---

## Engine Recommendations

### Windows + NVIDIA GPU (this project's target)
| Scenario | Recommended | Why |
|----------|------------|-----|
| Real-time transcription | `Whisper Small` (int8) | ~2 GB VRAM, RTF ~0.25 |
| Batch processing | `Whisper Turbo` (int8) | Large-v3 accuracy, 8× speed |
| Chinese speech | `Qwen3-ASR 0.6B` | Best Chinese WER, low VRAM |
| Highest accuracy | `Whisper Large-v3` (int8) | WER ~2.5%, ~2.5 GB VRAM |

### Key Finding
**faster-whisper** with int8 quantization is the best choice for Windows+NVIDIA GPU edge deployment. It provides 4× speedup and 40% VRAM savings over vanilla Whisper while maintaining identical accuracy.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | GPU status & engine availability |
| `GET` | `/api/models` | List all registered models |
| `POST` | `/api/transcribe` | Transcribe an audio file |
| `POST` | `/api/benchmark` | Run multi-model benchmark |
| `GET` | `/api/recommendations?vrram_mb=8192` | Get VRAM-appropriate models |

Full interactive docs at `http://localhost:8765/docs` after starting the server.

---

## Target Product Stack

The demo GUI is designed to be compatible with **Electron.js** on both **Windows** and **macOS**:

1. **Frontend**: Pure HTML/CSS/JS — no framework dependency, Electron-compatible
2. **Backend**: Python FastAPI — runs as a child process in Electron or standalone
3. **Packaging**: PyInstaller for Python backend → single .exe; electron-builder for the shell

For a production Electron app, the Python backend would be bundled via PyInstaller and launched by Electron's main process (see `electron/main.js` for the implementation).
