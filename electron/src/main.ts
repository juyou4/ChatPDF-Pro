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

import { app, BrowserWindow, ipcMain, dialog, shell, session } from 'electron';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { ProcessManager, BackendInfo } from './process-manager';

// ---- 单例模式 ----
const gotTheLock = app.requestSingleInstanceLock();
if (!gotTheLock) {
  app.quit();
  process.exit(0);
}

let mainWindow: BrowserWindow | null = null;
let backendInfo: BackendInfo | null = null;
let pendingStartupErrorUrl = '';
const processManager = new ProcessManager();

function isSafeExternalUrl(value: string): string | null {
  try {
    const parsed = new URL(value);
    return ['http:', 'https:'].includes(parsed.protocol) ? parsed.href : null;
  } catch {
    return null;
  }
}

function isTrustedRendererUrl(value: string): boolean {
  try {
    const parsed = new URL(value);
    if (app.isPackaged) {
      if (parsed.protocol !== 'file:') return false;
      const rendererRoot = path.resolve(process.resourcesPath, 'renderer');
      const filePath = path.resolve(fileURLToPath(parsed));
      return filePath === rendererRoot || filePath.startsWith(`${rendererRoot}${path.sep}`);
    }
    return parsed.origin === 'http://localhost:3000' || parsed.origin === 'http://127.0.0.1:3000';
  } catch {
    return false;
  }
}

function isAllowedNavigation(value: string): boolean {
  return isTrustedRendererUrl(value) || Boolean(pendingStartupErrorUrl && value === pendingStartupErrorUrl);
}

function escapeHtml(value: unknown): string {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function requireTrustedRenderer(event: { senderFrame: { url: string } }): void {
  if (!isTrustedRendererUrl(event.senderFrame.url)) {
    throw new Error('Untrusted renderer IPC request');
  }
}

async function openExternalSafely(value: string): Promise<void> {
  const safeUrl = isSafeExternalUrl(value);
  if (!safeUrl) throw new Error('Only http/https URLs are allowed');
  await shell.openExternal(safeUrl);
}

function configureSessionSecurity(): void {
  session.defaultSession.setPermissionRequestHandler((_webContents, _permission, callback) => callback(false));
  session.defaultSession.setPermissionCheckHandler(() => false);
}

function installBackendRequestAuthentication(info: BackendInfo): void {
  const backendOrigin = new URL(info.baseUrl).origin;
  session.defaultSession.webRequest.onBeforeSendHeaders(
    { urls: [`${backendOrigin}/*`] },
    (details, callback) => {
      callback({
        cancel: false,
        requestHeaders: { ...details.requestHeaders, 'X-ChatPDF-Token': info.token },
      });
    }
  );
}

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
      webSecurity: true,
      webviewTag: false,
      preload: path.join(__dirname, 'preload.js'),
    },
    show: false, // 等后端就绪后再显示
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  mainWindow.webContents.on('will-navigate', (event, url) => {
    if (isAllowedNavigation(url)) return;
    event.preventDefault();
    if (isSafeExternalUrl(url)) void openExternalSafely(url);
  });

  mainWindow.webContents.on('will-redirect', (event, url) => {
    if (isAllowedNavigation(url)) return;
    event.preventDefault();
    if (isSafeExternalUrl(url)) void openExternalSafely(url);
  });

  mainWindow.webContents.on('will-attach-webview', (event) => event.preventDefault());

  // 拦截外部链接，在系统浏览器中打开
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (isSafeExternalUrl(url)) void openExternalSafely(url);
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
      <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
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
      <p>${escapeHtml(error.message)}</p>
      <h3>Diagnostics</h3>
      <pre>${escapeHtml(diagnostics)}</pre>
      <h3>Possible solutions</h3>
      <ul>
        <li>Check if antivirus is blocking the application</li>
        <li>Try reinstalling ChatPDF Pro</li>
        <li>Check the log file listed above</li>
      </ul>
    </body>
    </html>
  `;

  pendingStartupErrorUrl = `data:text/html;charset=utf-8,${encodeURIComponent(errorHtml)}`;
  mainWindow.webContents.once('did-finish-load', () => {
    pendingStartupErrorUrl = '';
  });
  mainWindow.loadURL(pendingStartupErrorUrl);
  mainWindow.show();
}

// ---- IPC 处理 ----
function setupIPC(): void {
  ipcMain.handle('get-api-base-url', (event) => {
    requireTrustedRenderer(event);
    return backendInfo?.baseUrl || 'http://127.0.0.1:8000';
  });

  ipcMain.handle('open-external', async (event, url: string) => {
    requireTrustedRenderer(event);
    await openExternalSafely(url);
  });

  ipcMain.handle('get-version', (event) => {
    requireTrustedRenderer(event);
    return app.getVersion();
  });

  ipcMain.handle('open-data-dir', (event) => {
    requireTrustedRenderer(event);
    shell.openPath(app.getPath('userData'));
  });

  ipcMain.handle('select-file', async (event, options?: { filters?: Array<{ name: string; extensions: string[] }> }) => {
    requireTrustedRenderer(event);
    if (!mainWindow) return null;
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ['openFile'],
      filters: options?.filters || [{ name: 'PDF Files', extensions: ['pdf'] }],
    });
    if (result.canceled || result.filePaths.length === 0) return null;
    return result.filePaths[0];
  });

  ipcMain.handle('select-directory', async (event) => {
    requireTrustedRenderer(event);
    if (!mainWindow) return null;
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ['openDirectory', 'createDirectory'],
    });
    if (result.canceled || result.filePaths.length === 0) return null;
    return result.filePaths[0];
  });
}

// ---- 应用生命周期 ----
app.whenReady().then(async () => {
  configureSessionSecurity();
  setupIPC();
  createWindow();

  try {
    // 启动 Python 后端
    backendInfo = await processManager.start();
    installBackendRequestAuthentication(backendInfo);
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
