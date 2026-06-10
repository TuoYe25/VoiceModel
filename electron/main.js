/**
 * Electron main process for ASR Hub desktop app.
 * 
 * Usage:
 *   npm install            # install electron dependencies
 *   npm start              # launch the app
 *   npm run dist           # package for distribution
 * 
 * Architecture:
 *   1. Electron spawns the Python backend (FastAPI) as a child process
 *   2. Backend serves REST API on localhost:8765
 *   3. Frontend (renderer) loads index.html and talks to backend
 * 
 * For Mac/Windows production builds, the Python backend can be bundled
 * with PyInstaller into a standalone executable.
 */

const { app, BrowserWindow, dialog, ipcMain } = require("electron");
const path = require("path");
const { spawn } = require("child_process");

let mainWindow = null;
let pythonProcess = null;

const BACKEND_PORT = 8765;
const BACKEND_URL = `http://localhost:${BACKEND_PORT}`;
const IS_DEV = process.env.NODE_ENV !== "production";

// ─── Python backend lifecycle ──────────────────────────────────
function startBackend() {
  const backendDir = path.join(__dirname, "..", "backend");
  const serverScript = path.join(backendDir, "server.py");

  const pythonCmd = process.platform === "win32" ? "python" : "python3";

  console.log(`[Electron] Starting Python backend: ${pythonCmd} ${serverScript}`);

  pythonProcess = spawn(pythonCmd, [serverScript], {
    cwd: path.join(__dirname, ".."),
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });

  pythonProcess.stdout.on("data", (data) => {
    console.log(`[Backend] ${data.toString().trim()}`);
  });

  pythonProcess.stderr.on("data", (data) => {
    console.error(`[Backend Error] ${data.toString().trim()}`);
  });

  pythonProcess.on("close", (code) => {
    console.log(`[Electron] Backend exited with code ${code}`);
    pythonProcess = null;
  });
}

function stopBackend() {
  if (pythonProcess) {
    console.log("[Electron] Stopping backend...");
    pythonProcess.kill();
    pythonProcess = null;
  }
}

// ─── Window creation ───────────────────────────────────────────
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    title: "ASR Hub — Edge STT Testing",
    backgroundColor: "#0d1117",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  // Load the frontend HTML
  const frontendPath = path.join(__dirname, "..", "frontend", "index.html");
  mainWindow.loadFile(frontendPath);

  if (IS_DEV) {
    mainWindow.webContents.openDevTools({ mode: "detach" });
  }

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

// ─── IPC handlers ──────────────────────────────────────────────
ipcMain.handle("get-backend-url", () => BACKEND_URL);

ipcMain.handle("select-audio-file", async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "Select Audio File",
    filters: [
      { name: "Audio Files", extensions: ["wav", "mp3", "flac", "ogg", "m4a", "aac"] },
    ],
    properties: ["openFile"],
  });
  if (result.canceled) return null;
  return result.filePaths[0];
});

// ─── App lifecycle ─────────────────────────────────────────────
app.whenReady().then(() => {
  startBackend();
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  stopBackend();
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  stopBackend();
});
