/**
 * Preload 脚本 - 安全 IPC 桥
 *
 * contextIsolation: true + nodeIntegration: false 下，
 * 只暴露必要的 API 给 renderer 进程。
 */

import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('chatpdfDesktop', {
  /** 后端 API base URL (http://127.0.0.1:<port>) */
  getApiBaseUrl: (): Promise<string> => ipcRenderer.invoke('get-api-base-url'),

  /** 应用版本 */
  getVersion: (): Promise<string> => ipcRenderer.invoke('get-version'),

  /** 是否为桌面模式 */
  isDesktop: true,

  /** 在系统文件管理器中打开目录 */
  openDataDir: (): Promise<void> => ipcRenderer.invoke('open-data-dir'),

  /** 在外部浏览器打开链接 */
  openExternal: (url: string): Promise<void> => ipcRenderer.invoke('open-external', url),

  /** 选择文件对话框 */
  selectFile: (options?: { filters?: Array<{ name: string; extensions: string[] }> }): Promise<string | null> =>
    ipcRenderer.invoke('select-file', options),

  /** 选择目录对话框 */
  selectDirectory: (): Promise<string | null> => ipcRenderer.invoke('select-directory'),
});
