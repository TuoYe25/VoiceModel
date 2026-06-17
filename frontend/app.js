/**
 * ASR Hub — Frontend Application
 * 
 * Handles:
 * - Tab navigation: Transcribe / Benchmark
 * - Model selection with VRAM-aware recommendations
 * - Audio file upload (drag & drop + file picker)
 * - REST API calls to backend for transcription & benchmarking
 * - Real-time result display with metrics
 */

// Use relative paths so it works whether served from file:// or http://localhost:8765
const API_BASE = (window.location.protocol === "file:")
    ? "http://localhost:8765/api"
    : "/api";

// ── State ────────────────────────────────────────────────────
const state = {
  selectedModel: "small",
  selectedEngine: "faster-whisper",
  audioFile: null,
  benchAudioFile: null,
  benchModels: [],
  loadedModels: [],
  // Realtime recording
  selectedRealtimeModel: "small",
  selectedRealtimeEngine: "faster-whisper",
  mediaRecorder: null,
  realtimeWs: null,
  isRecording: false,
  recordStartTime: null,
  recordTimerInterval: null,
  finalTranscript: "",
};

// ── DOM helpers ──────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ── Toast ────────────────────────────────────────────────────
function showToast(message, type = "info") {
  const container = $("#toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(20px)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

// ── Fetch helper with timeout ─────────────────────────────────
async function fetchWithTimeout(url, options = {}, timeoutMs = 10000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    return res;
  } finally {
    clearTimeout(timeoutId);
  }
}

// ── Status Bar ───────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetchWithTimeout(`${API_BASE}/health`, {}, 8000);
    const data = await res.json();
    const dot = $("#statusDot");
    const text = $("#statusText");
    if (data.status === "ok") {
      dot.className = "status-dot connected";
      text.textContent = "Server connected";
      // Populate the GPU selector (if present)
      const sel = $("#gpuSelector");
      if (sel && Array.isArray(data.gpus) && data.gpus.length > 0) {
        sel.innerHTML = data.gpus
          .map(
            (g) =>
              `<option value="${g.index}" ${
                g.index === data.default_gpu_index ? "selected" : ""
              }>GPU #${g.index} — ${g.name} (${Math.round(
                (g.vram_total_mb || 0) / 1024
              )} GB)</option>`
          )
          .join("");
      }
    }
    return data;
  } catch (e) {
    $("#statusDot").className = "status-dot disconnected";
    $("#statusText").textContent = "Server offline";
    return null;
  }
}

function getSelectedGpuIndex() {
  const sel = $("#gpuSelector");
  if (!sel) return null;
  const v = sel.value;
  return v === "" || v == null ? null : Number(v);
}

// ── Tab Navigation ───────────────────────────────────────────
function initTabs() {
  $$(".nav-item").forEach((item) => {
    item.addEventListener("click", (e) => {
      e.preventDefault();
      const tab = item.dataset.tab;
      $$(".nav-item").forEach((i) => i.classList.remove("active"));
      item.classList.add("active");
      $$(".tab-panel").forEach((p) => p.classList.remove("active"));
      $(`#tab-${tab}`).classList.add("active");
    });
  });
}

// ── Model Loading ────────────────────────────────────────────
async function loadModels() {
  try {
    const res = await fetchWithTimeout(`${API_BASE}/models`, {}, 12000);
    state.loadedModels = await res.json();
    renderModelGrid();
    renderRealtimeModelGrid();
    renderBenchModelList();
  } catch (e) {
    console.error("Failed to load models:", e);
    showToast("Failed to load model list. Is the server running?", "error");
  }
}

function renderModelGrid() {
  const grid = $("#modelGrid");
  grid.innerHTML = "";

  const all = [];
  for (const [engine, models] of Object.entries(state.loadedModels)) {
    for (const m of models) {
      all.push({ ...m, engine });
    }
  }

  all.forEach((m) => {
    const card = document.createElement("div");
    card.className = "model-card";
    if (m.model_id === state.selectedModel) card.classList.add("selected");

    const recLabel = m.recommended_for === "edge" ? "Edge" : "Batch";
    const recClass = m.recommended_for === "edge" ? "" : "batch";

    card.innerHTML = `
      <div class="mc-name">${m.display_name}</div>
      <div class="mc-specs">
        <span>${m.param_count} params</span>
        <span>${m.vram_requirement} VRAM</span>
      </div>
      <span class="mc-recommended ${recClass}">${recLabel}</span>
    `;

    card.addEventListener("click", () => {
      $$("#modelGrid .model-card").forEach((c) => c.classList.remove("selected"));
      card.classList.add("selected");
      state.selectedModel = m.model_id;
      state.selectedEngine = m.engine;
    });

    grid.appendChild(card);
  });
}

function renderBenchModelList() {
  const container = $("#benchModelList");
  container.innerHTML = "";

  const all = [];
  for (const [engine, models] of Object.entries(state.loadedModels)) {
    for (const m of models) {
      all.push({ ...m, engine });
    }
  }

  // Only show faster-whisper by default for benchmarking
  const benchmarkModels = all.filter(
    (m) => m.engine === "faster-whisper"
  );

  benchmarkModels.forEach((m) => {
    const label = document.createElement("label");
    label.innerHTML = `
      <input type="checkbox" value="${m.model_id}" data-engine="${m.engine}"
        ${["small", "medium", "turbo"].includes(m.model_id) ? "checked" : ""} />
      <span>${m.display_name} <small style="color:var(--text-muted)">(${m.param_count})</small></span>
    `;
    container.appendChild(label);
  });

  // Update benchModels state on change
  container.addEventListener("change", () => {
    state.benchModels = [];
    container.querySelectorAll("input:checked").forEach((cb) => {
      state.benchModels.push({
        model_id: cb.value,
        engine_type: cb.dataset.engine,
        compute_type: "int8",
      });
    });
    updateBenchBtn();
  });

  // Initial state
  container.querySelectorAll("input:checked").forEach((cb) => {
    state.benchModels.push({
      model_id: cb.value,
      engine_type: cb.dataset.engine,
      compute_type: "int8",
    });
  });
}

function renderRealtimeModelGrid() {
  const grid = $("#realtimeModelGrid");
  if (!grid) return;
  grid.innerHTML = "";

  const all = [];
  for (const [engine, models] of Object.entries(state.loadedModels)) {
    for (const m of models) {
      all.push({ ...m, engine });
    }
  }

  all.forEach((m) => {
    const card = document.createElement("div");
    card.className = "model-card";
    if (m.model_id === state.selectedRealtimeModel) card.classList.add("selected");

    const recMap = { edge: "Edge", batch: "Batch", "real-time": "Real-time" };
    const recLabel = recMap[m.recommended_for] || m.recommended_for;
    const recClass = m.recommended_for === "edge" ? "" : m.recommended_for === "real-time" ? "real-time" : "batch";

    card.innerHTML = `
      <div class="mc-name">${m.display_name}</div>
      <div class="mc-specs">
        <span>${m.param_count} params</span>
        <span>${m.vram_requirement} VRAM</span>
      </div>
      <span class="mc-recommended ${recClass}">${recLabel}</span>
    `;

    card.addEventListener("click", () => {
      $$("#realtimeModelGrid .model-card").forEach((c) => c.classList.remove("selected"));
      card.classList.add("selected");
      state.selectedRealtimeModel = m.model_id;
      state.selectedRealtimeEngine = m.engine;
    });

    grid.appendChild(card);
  });
}

// ── Audio Upload ─────────────────────────────────────────────
function initUpload(zoneId, inputId, fileInfoId, clearBtnId, onFileSet) {
  const zone = $(zoneId);
  const input = $(inputId);
  const fileInfo = $(fileInfoId);
  const clearBtn = $(clearBtnId);

  zone.addEventListener("click", (e) => {
    if (e.target.tagName === "BUTTON") return;
    input.click();
  });

  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("dragover");
  });

  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));

  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("audio/")) {
      setFile(file, zone, fileInfo, onFileSet);
    }
  });

  input.addEventListener("change", () => {
    if (input.files[0]) {
      setFile(input.files[0], zone, fileInfo, onFileSet);
    }
  });

  clearBtn?.addEventListener("click", () => {
    onFileSet(null);
    zone.classList.remove("hidden");
    fileInfo.classList.add("hidden");
    input.value = "";
  });
}

function setFile(file, zone, fileInfo, onFileSet) {
  if (!file) return;
  zone.classList.add("hidden");
  fileInfo.classList.remove("hidden");
  const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
  fileInfo.querySelector(
    fileInfo.id === "fileInfo" ? ".file-name" : "span"
  ).textContent = `${file.name} (${sizeMB} MB)`;
  onFileSet(file);
}

// ── Transcribe ───────────────────────────────────────────────
async function transcribe() {
  if (!state.audioFile) return;

  const btn = $("#transcribeBtn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Transcribing...';

  try {
    const formData = new FormData();
    formData.append("file", state.audioFile);

    const params = new URLSearchParams({
      engine_type: state.selectedEngine,
      model_id: state.selectedModel,
      device: "cuda",
      compute_type: "int8",
    });
    const gpuIdx = getSelectedGpuIndex();
    if (gpuIdx != null) params.set("device_index", String(gpuIdx));

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 360_000); // 6 min

    const res = await fetch(`${API_BASE}/transcribe?${params}`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!res.ok) {
      let detail = "Transcription failed";
      try {
        // Read body as text first – then try JSON (avoids "body stream already read")
        const body = await res.text();
        try {
          const err = JSON.parse(body);
          detail = err.detail || detail;
        } catch {
          detail = `${detail}: ${body.substring(0, 200)}`;
        }
      } catch {
        /* ignore */
      }
      throw new Error(detail);
    }

    const data = await res.json();
    renderResult(data);
    showToast("Transcription complete!", "success");
  } catch (e) {
    const msg = e.name === "AbortError"
      ? "Transcribe timed out. Check server logs for details."
      : `Error: ${e.message}`;
    showToast(msg, "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5,3 19,12 5,21"/></svg> Transcribe`;
  }
}

function renderResult(data) {
  const card = $("#resultsCard");
  card.classList.remove("hidden");

  // Metrics
  const m = data.metrics;
  const rtfColor = m.real_time_factor < 0.3 ? "good" : "warn";
  const vramColor = m.peak_vram_mb < 4000 ? "good" : "warn";

  $("#metricsRow").innerHTML = `
    <div class="metric ${rtfColor}">
      <div class="m-value">${m.real_time_factor.toFixed(2)}x</div>
      <div class="m-label">Real-Time Factor</div>
    </div>
    <div class="metric">
      <div class="m-value">${m.processing_time_seconds.toFixed(2)}s</div>
      <div class="m-label">Processing Time</div>
    </div>
    <div class="metric ${vramColor}">
      <div class="m-value">${m.peak_vram_mb.toFixed(0)} MB</div>
      <div class="m-label">Peak VRAM</div>
    </div>
    <div class="metric">
      <div class="m-value">${m.duration_seconds.toFixed(1)}s</div>
      <div class="m-label">Audio Duration</div>
    </div>
  `;

  // Language badge
  $("#langBadge").textContent = `🌐 ${data.language}`;

  // Transcript text
  $("#transcriptText").textContent = data.text || "(No speech detected)";

  // Segments
  const segList = $("#segmentsList");
  if (data.segments && data.segments.length > 0) {
    segList.innerHTML = data.segments
      .map(
        (s) =>
          `<div class="segment-item">
            <span class="seg-time">${fmtTime(s.start)} → ${fmtTime(s.end)}</span>
            <span class="seg-text">${escapeHtml(s.text)}</span>
          </div>`
      )
      .join("");
  } else {
    segList.innerHTML = "<p style='color:var(--text-muted);padding:8px'>No segments</p>";
  }

  card.scrollIntoView({ behavior: "smooth" });
}

function fmtTime(sec) {
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(1);
  return `${m}:${s.padStart(4, "0")}`;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// ── Benchmark ────────────────────────────────────────────────
async function runBenchmark() {
  if (!state.benchAudioFile || state.benchModels.length === 0) return;

  const btn = $("#runBenchmarkBtn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Running benchmark...';

  try {
    const formData = new FormData();
    formData.append("file", state.benchAudioFile);
    formData.append("models_json", JSON.stringify(state.benchModels));

    const gt = $("#groundTruthInput").value.trim();
    if (gt) formData.append("ground_truth", gt);

    // Dynamic timeout: 8 min per model, max 45 min
    const benchTimeout = Math.min(
      Math.max(480_000, state.benchModels.length * 480_000),
      2_700_000
    );
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), benchTimeout);

    const res = await fetch(`${API_BASE}/benchmark?${(() => {
      const gpuIdx = getSelectedGpuIndex();
      return gpuIdx != null ? `device_index=${gpuIdx}` : "";
    })()}`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!res.ok) {
      let detail = `Benchmark failed (HTTP ${res.status})`;
      try {
        const body = await res.text();
        try {
          const err = JSON.parse(body);
          detail = err.detail || detail;
        } catch {
          detail = `${detail}: ${body.substring(0, 200)}`;
        }
      } catch {
        /* ignore */
      }
      throw new Error(detail);
    }

    const data = await res.json();
    renderBenchResults(data);
    showToast("Benchmark complete!", "success");
  } catch (e) {
    const msg = e.name === "AbortError"
      ? "Benchmark timed out. Try fewer models or a smaller audio file."
      : `Error: ${e.message}`;
    showToast(msg, "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22,2 15,22 11,13 2,9"/></svg> Run Benchmark`;
  }
}

function renderBenchResults(data) {
  const card = $("#benchResultsCard");
  card.classList.remove("hidden");

  // Table
  const tbody = $("#benchTableBody");
  const results = data.results || [];
  const minWER = Math.min(...results.filter((r) => r.wer != null).map((r) => r.wer));
  const minRTF = Math.min(...results.map((r) => r.real_time_factor));

  tbody.innerHTML = results
    .map(
      (r) => `
    <tr>
      <td><strong>${r.model}</strong></td>
      <td>${r.audio_duration_s?.toFixed(1) || "?"}s</td>
      <td>${r.processing_time_s?.toFixed(2) || "?"}s</td>
      <td class="${r.real_time_factor === minRTF ? "rtf-best" : ""}">${r.real_time_factor?.toFixed(2)}x</td>
      <td>${r.peak_vram_mb?.toFixed(0) || "?"} MB</td>
      <td class="${r.wer === minWER && r.wer != null ? "wer-best" : ""}">${r.wer != null ? (r.wer * 100).toFixed(2) + "%" : "N/A"}</td>
      <td>${r.success ? "✅" : `❌ ${r.error || ""}`}</td>
    </tr>`
    )
    .join("");

  // Report
  if (data.report) {
    $("#benchReport").innerHTML = `<pre style="font-size:13px;line-height:1.6;white-space:pre-wrap;color:var(--text-secondary)">${escapeHtml(data.report)}</pre>`;
  }

  card.scrollIntoView({ behavior: "smooth" });
}

function updateBenchBtn() {
  $("#runBenchmarkBtn").disabled = !state.benchAudioFile || state.benchModels.length === 0;
}

// ── Copy ─────────────────────────────────────────────────────
function initCopy() {
  $("#copyBtn").addEventListener("click", () => {
    const text = $("#transcriptText").textContent;
    navigator.clipboard.writeText(text).then(() => {
      showToast("Copied to clipboard!", "success");
    }).catch(() => {
      showToast("Failed to copy", "error");
    });
  });
}

// ── Realtime Recording ───────────────────────────────────────
function getWsUrl(path) {
  if (window.location.protocol === "file:") {
    return `ws://localhost:8765${path}`;
  }
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${window.location.host}${path}`;
}

async function startRecording() {
  if (state.isRecording) return;

  const btn = $("#recordBtn");
  const modelId = state.selectedRealtimeModel;
  const engineType = state.selectedRealtimeEngine;

  // Reset final transcript
  state.finalTranscript = "";
  $("#realtimeTranscriptText").innerHTML =
    '<span class="placeholder-text">Connecting...</span>';
  $("#realtimeLangBadge").textContent = "Live";
  $("#realtimeResultsCard").classList.remove("hidden");
  $("#realtimeSegmentsDetail").style.display = "none";
  $("#realtimeSegmentsList").innerHTML = "";

  btn.disabled = true;
  btn.innerHTML =
    '<span class="spinner"></span> Connecting...';

  try {
    // Request microphone
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 16000,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });

    // Build WebSocket URL
    const params = new URLSearchParams({
      engine_type: engineType,
      model_id: modelId,
      compute_type: "int8",
    });
    const gpuIdx = getSelectedGpuIndex();
    if (gpuIdx != null) params.set("device_index", String(gpuIdx));

    const wsUrl = `${getWsUrl("/ws/realtime")}?${params.toString()}`;
    const ws = new WebSocket(wsUrl);
    state.realtimeWs = ws;
    ws.binaryType = "blob";

    // WebSocket event handlers
    ws.onopen = () => {
      console.log("[realtime] WS connected");
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        handleRealtimeMessage(msg);
      } catch {
        // Binary message – ignore
      }
    };

    ws.onerror = (err) => {
      console.error("[realtime] WS error:", err);
      showToast("WebSocket connection error", "error");
    };

    ws.onclose = (event) => {
      console.log("[realtime] WS closed:", event.code, event.reason);
      if (state.isRecording) {
        stopRecordingInternal();
      }
    };

    // Wait for "ready" from server (or timeout)
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error("Server did not respond in time"));
      }, 30000);

      const origHandler = ws.onmessage;
      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === "ready") {
            clearTimeout(timeout);
            ws.onmessage = origHandler;
            resolve();
            return;
          }
          if (msg.type === "error") {
            clearTimeout(timeout);
            reject(new Error(msg.message));
            return;
          }
        } catch {
          // ignore parse errors
        }
        // Pass to original handler
        if (origHandler) origHandler(event);
      };
    });

    // Create MediaRecorder (WebM/Opus)
    let mimeType = "";
    for (const m of ["audio/webm;codecs=opus", "audio/webm"]) {
      if (MediaRecorder.isTypeSupported(m)) {
        mimeType = m;
        break;
      }
    }
    if (!mimeType) {
      throw new Error("Browser does not support audio recording (WebM/Opus)");
    }

    const recorder = new MediaRecorder(stream, { mimeType });
    state.mediaRecorder = recorder;

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0 && ws.readyState === WebSocket.OPEN) {
        ws.send(event.data);
      }
    };

    recorder.onstop = () => {
      // Send STOP signal
      if (ws.readyState === WebSocket.OPEN) {
        ws.send("STOP");
      }
      // Stop all tracks
      stream.getTracks().forEach((t) => t.stop());
    };

    // Start recording with 1-second chunks
    recorder.start(1000);
    state.isRecording = true;
    state.recordStartTime = Date.now();

    // Update UI
    btn.className = "btn btn-danger btn-lg";
    btn.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
        <rect x="6" y="4" width="12" height="16" rx="2"/>
      </svg>
      Stop Recording
    `;
    btn.disabled = false;

    $("#recordingIndicator").classList.remove("hidden");
    $("#recordStatus").textContent = "Recording";
    $("#recordStatus").className = "record-status recording";

    state.recordTimerInterval = setInterval(updateRecordTimer, 200);
    $("#realtimeTranscriptText").innerHTML =
      '<span class="placeholder-text">Listening...</span>';
  } catch (e) {
    console.error("[realtime] Start failed:", e);
    showToast(`Failed to start recording: ${e.message}`, "error");
    stopRecordingInternal();
  }
}

function stopRecording() {
  if (!state.isRecording) return;
  stopRecordingInternal();
}

function stopRecordingInternal() {
  state.isRecording = false;

  // Stop MediaRecorder
  if (state.mediaRecorder && state.mediaRecorder.state !== "inactive") {
    state.mediaRecorder.stop();
  }
  state.mediaRecorder = null;

  // Close WebSocket
  if (state.realtimeWs) {
    if (state.realtimeWs.readyState === WebSocket.OPEN) {
      state.realtimeWs.close(1000, "User stopped");
    }
    state.realtimeWs = null;
  }

  // Reset UI
  const btn = $("#recordBtn");
  btn.className = "btn btn-danger btn-lg";
  btn.innerHTML = `
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
      <circle cx="12" cy="12" r="8"/>
    </svg>
    Start Recording
  `;
  btn.disabled = false;

  $("#recordingIndicator").classList.add("hidden");
  $("#recordStatus").textContent = "";
  $("#recordStatus").className = "record-status";

  if (state.recordTimerInterval) {
    clearInterval(state.recordTimerInterval);
    state.recordTimerInterval = null;
  }
  state.recordStartTime = null;
}

function handleRealtimeMessage(msg) {
  switch (msg.type) {
    case "result":
      if (msg.final) {
        // Final result – show full transcript with metrics
        renderRealtimeResult(msg);
      } else {
        // Partial – update live text
        const el = $("#realtimeTranscriptText");
        el.textContent = msg.text || "(silence)";
        state.finalTranscript = msg.text || "";

        // Update segments
        if (msg.segments && msg.segments.length > 0) {
          const segDetail = $("#realtimeSegmentsDetail");
          segDetail.style.display = "block";
          $("#realtimeSegmentsList").innerHTML = msg.segments
            .map(
              (s) =>
                `<div class="segment-item">
                  <span class="seg-time">${fmtTime(s.start)} → ${fmtTime(s.end)}</span>
                  <span class="seg-text">${escapeHtml(s.text)}</span>
                </div>`
            )
            .join("");
        }
      }
      break;

    case "status":
      // No new speech in this chunk
      break;

    case "error":
      console.error("[realtime] Server error:", msg.message);
      showToast(`Transcription error: ${msg.message}`, "error");
      break;

    case "keepalive":
      // Server is still there
      break;

    case "ready":
      // Already handled during connection
      break;

    default:
      console.log("[realtime] Unknown message type:", msg.type);
  }
}

function renderRealtimeResult(data) {
  if (!data.text) {
    $("#realtimeTranscriptText").textContent = "(No speech detected)";
    return;
  }

  state.finalTranscript = data.text;
  $("#realtimeTranscriptText").textContent = data.text;
  $("#realtimeLangBadge").textContent = `🌐 ${data.language || "unknown"}`;

  if (data.segments && data.segments.length > 0) {
    const segDetail = $("#realtimeSegmentsDetail");
    segDetail.style.display = "block";
    $("#realtimeSegmentsList").innerHTML = data.segments
      .map(
        (s) =>
          `<div class="segment-item">
            <span class="seg-time">${fmtTime(s.start)} → ${fmtTime(s.end)}</span>
            <span class="seg-text">${escapeHtml(s.text)}</span>
          </div>`
      )
      .join("");
  }

  showToast("Recording complete!", "success");
}

function updateRecordTimer() {
  if (!state.recordStartTime) return;
  const elapsed = Math.floor((Date.now() - state.recordStartTime) / 1000);
  const mins = Math.floor(elapsed / 60);
  const secs = elapsed % 60;
  $("#recordTimer").textContent =
    `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

// ── Init ─────────────────────────────────────────────────────
async function init() {
  initTabs();

  // Upload zones
  initUpload(
    "#uploadZone",
    "#fileInput",
    "#fileInfo",
    "#clearFile",
    (file) => {
      state.audioFile = file;
      $("#transcribeBtn").disabled = !file;
    }
  );

  initUpload(
    "#benchUploadZone",
    "#benchFileInput",
    "#benchFileInfo",
    "#clearBenchFile",
    (file) => {
      state.benchAudioFile = file;
      updateBenchBtn();
    }
  );

  // Transcribe button
  $("#transcribeBtn").addEventListener("click", transcribe);

  // Benchmark button
  $("#runBenchmarkBtn").addEventListener("click", runBenchmark);

  // Realtime recording button – single handler, dispatches by state
  function handleRecordClick() {
    if (state.isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }
  $("#recordBtn").addEventListener("click", handleRecordClick);

  // Realtime copy button
  $("#realtimeCopyBtn").addEventListener("click", () => {
    const text = state.finalTranscript || $("#realtimeTranscriptText").textContent;
    navigator.clipboard.writeText(text).then(() => {
      showToast("Copied to clipboard!", "success");
    }).catch(() => {
      showToast("Failed to copy", "error");
    });
  });

  // Copy button
  initCopy();

  // Health check
  const health = await checkHealth();

  // Load model catalogue
  await loadModels();

  // If no models from the server, show a demo state
  if (Object.keys(state.loadedModels).length === 0) {
    console.log("Server returned empty model list, showing demo cache");
    state.loadedModels = getFallbackModels();
    renderModelGrid();
    renderRealtimeModelGrid();
    renderBenchModelList();
  }
}

function getFallbackModels() {
  return {
    "faster-whisper": [
      {
        model_id: "tiny", display_name: "Whisper Tiny",
        param_count: "39M", vram_requirement: "~1 GB",
        languages: ["multilingual (99+)"],
        description: "Fastest, lowest resource.",
        recommended_for: "real-time", engine: "faster-whisper",
      },
      {
        model_id: "base", display_name: "Whisper Base",
        param_count: "74M", vram_requirement: "~1 GB",
        languages: ["multilingual (99+)"],
        description: "Good for edge devices with limited GPU.",
        recommended_for: "edge", engine: "faster-whisper",
      },
      {
        model_id: "small", display_name: "Whisper Small",
        param_count: "244M", vram_requirement: "~2 GB",
        languages: ["multilingual (99+)"],
        description: "Sweet spot: accuracy vs. resource.",
        recommended_for: "edge", engine: "faster-whisper",
      },
      {
        model_id: "medium", display_name: "Whisper Medium",
        param_count: "769M", vram_requirement: "~5 GB",
        languages: ["multilingual (99+)"],
        description: "Strong accuracy for mid-range GPUs.",
        recommended_for: "batch", engine: "faster-whisper",
      },
      {
        model_id: "large-v3", display_name: "Whisper Large-v3",
        param_count: "1.55B", vram_requirement: "~10 GB (2.5 GB int8)",
        languages: ["multilingual (99+)"],
        description: "Best accuracy. Needs 8+ GB VRAM.",
        recommended_for: "batch", engine: "faster-whisper",
      },
      {
        model_id: "turbo", display_name: "Whisper Turbo",
        param_count: "809M", vram_requirement: "~6 GB (2 GB int8)",
        languages: ["multilingual (99+)"],
        description: "large-v3 accuracy at 8x speed.",
        recommended_for: "batch", engine: "faster-whisper",
      },
    ],
  };
}

// Kick off
document.addEventListener("DOMContentLoaded", init);
