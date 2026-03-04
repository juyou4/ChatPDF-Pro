/**
 * Electron 主进程
 *
 * 功能：
 * - 单例模式（防多开冲突）
 * - 启动 Python 后端（ProcessManager）
 * - 创建 BrowserWindow 加载前端
 * - IPC 处理（apiBaseUrl, token, version 等）
 * - 优雅关闭
 */

import { app, BrowserWindow, ipcMain, dialog, shell } from 'electron';
import * as path from 'path';
import { ProcessManager, BackendInfo } from './process-manager';

// ---- 单例模式 ----
const gotTheLock = app.requestSingleInstanceLock();
if (!gotTheLock) {
  app.quit();
  process.exit(0);
}

let mainWindow: BrowserWindow | null = null;
let backendInfo: BackendInfo | null = null;
const processManager = new ProcessManager();

// ---- 单例：第二个实例启动时唤醒第一个 ----
app.on('second-instance', (_event, argv) => {
  if (mainWindow) {
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.focus();

    // 可选：解析 argv 中的 .pdf 路径，发送到 renderer 打开
    const pdfPath = argv.find((arg) => arg.endsWith('.pdf'));
    if (pdfPath) {
      mainWindow.webContents.send('open-pdf', pdfPath);
    }
  }
});

// ---- 创建主窗口 ----
function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    title: 'ChatPDF Pro',
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, 'preload.js'),
    },
    show: false, // 等后端就绪后再显示
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // 拦截外部链接，在系统浏览器中打开
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('http://') || url.startsWith('https://')) {
      shell.openExternal(url);
    }
    return { action: 'deny' };
  });
}

// ---- 加载前端 ----
function loadRenderer(): void {
  if (!mainWindow) return;

  if (app.isPackaged) {
    // 打包后：加载 extraResources/renderer/index.html
    const rendererPath = path.join(process.resourcesPath, 'renderer', 'index.html');
    mainWindow.loadFile(rendererPath);
  } else {
    // 开发模式：连接 Vite dev server
    mainWindow.loadURL('http://localhost:3000');
  }

  mainWindow.show();
}

// ---- 显示启动失败 UI ----
function showStartupError(error: Error): void {
  if (!mainWindow) return;

  const diagnostics = processManager.getDiagnostics();
  const errorHtml = `
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>ChatPDF - Startup Error</title>
      <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; padding: 40px; background: #f8f9fa; color: #333; }
        h1 { color: #dc3545; }
        pre { background: #e9ecef; padding: 16px; border-radius: 8px; overflow: auto; font-size: 13px; }
        button { padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; margin-right: 8px; font-size: 14px; }
        .btn-primary { background: #0d6efd; color: white; }
        .btn-secondary { background: #6c757d; color: white; }
      </style>
    </head>
    <body>
      <h1>Backend startup failed</h1>
      <p>${error.message}</p>
      <h3>Diagnostics</h3>
      <pre id="diag">${diagnostics}</pre>
      <button class="btn-primary" onclick="navigator.clipboard.writeText(document.getElementById('diag').textContent)">
        Copy diagnostics
      </button>
      <button class="btn-secondary" onclick="window.close()">Close</button>
      <h3>Possible solutions</h3>
      <ul>
        <li>Check if antivirus is blocking the application</li>
        <li>Try reinstalling ChatPDF Pro</li>
        <li>Check the log file listed above</li>
      </ul>
    </body>
    </html>
  `;

  mainWindow.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(errorHtml)}`);
  mainWindow.show();
}

// ---- IPC 处理 ----
function setupIPC(): void {
  ipcMain.handle('get-api-base-url', () => {
    return backendInfo?.baseUrl || 'http://127.0.0.1:8000';
  });

  ipcMain.handle('get-backend-token', () => {
    return backendInfo?.token || '';
  });

  ipcMain.handle('get-version', () => {
    return app.getVersion();
  });

  ipcMain.handle('open-data-dir', () => {
    shell.openPath(app.getPath('userData'));
  });

  ipcMain.handle('select-file', async (_event, options?: { filters?: Array<{ name: string; extensions: string[] }> }) => {
    if (!mainWindow) return null;
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ['openFile'],
      filters: options?.filters || [{ name: 'PDF Files', extensions: ['pdf'] }],
    });
    if (result.canceled || result.filePaths.length === 0) return null;
    return result.filePaths[0];
  });
}

// ---- 应用生命周期 ----
app.whenReady().then(async () => {
  setupIPC();
  createWindow();

  try {
    // 启动 Python 后端
    backendInfo = await processManager.start();
    console.log(`[Main] Backend ready at ${backendInfo.baseUrl}`);

    // 加载前端
    loadRenderer();
  } catch (error) {
    console.error('[Main] Failed to start backend:', error);
    showStartupError(error instanceof Error ? error : new Error(String(error)));
  }
});

app.on('window-all-closed', () => {
  // macOS 下保持运行直到 Cmd+Q
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', async () => {
  console.log('[Main] Shutting down...');
  await processManager.stop();
});

app.on('activate', () => {
  // macOS dock 点击重新创建窗口
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
    if (backendInfo) {
      loadRenderer();
    }
  }
});
