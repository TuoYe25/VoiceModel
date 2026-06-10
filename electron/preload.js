/**
 * Electron preload script — bridges main process ↔ renderer.
 * Exposes safe IPC methods to the frontend via contextBridge.
 */

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
  getBackendUrl: () => ipcRenderer.invoke("get-backend-url"),
  selectAudioFile: () => ipcRenderer.invoke("select-audio-file"),
  platform: process.platform,
});
