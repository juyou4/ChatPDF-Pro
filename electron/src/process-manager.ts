/**
 * Python 后端进程管理器
 *
 * 负责：
 * - 查找可用端口（偏好端口列表 + 回退随机）
 * - 生成安全 token
 * - spawn Python 后端进程
 * - 健康检查轮询
 * - 崩溃重启（最多 3 次）
 * - 优雅关闭（跨平台 kill 进程树）
 */

import { ChildProcess, spawn, execSync } from 'child_process';
import { createServer } from 'net';
import { randomBytes } from 'crypto';
import { app } from 'electron';
import * as path from 'path';
import * as http from 'http';

// 偏好端口列表（高位冷门端口，避免与常见服务冲突）
const PREFERRED_PORTS = [39283, 39284, 39285];

export interface BackendInfo {
  port: number;
  token: string;
  baseUrl: string;
}

export class ProcessManager {
  private process: ChildProcess | null = null;
  private port: number = 0;
  private token: string = '';
  private restartCount: number = 0;
  private maxRestarts: number = 3;
  private isShuttingDown: boolean = false;
  private dataDir: string;

  constructor() {
    this.dataDir = app.getPath('userData');
  }

  /**
   * 启动后端并返回连接信息
   */
  async start(): Promise<BackendInfo> {
    this.port = await this.findPort();
    this.token = randomBytes(32).toString('hex');

    await this.spawnBackend();
    await this.waitForHealthy();

    return {
      port: this.port,
      token: this.token,
      baseUrl: `http://127.0.0.1:${this.port}`,
    };
  }

  /**
   * 优雅关闭后端进程
   */
  async stop(): Promise<void> {
    this.isShuttingDown = true;
    if (!this.process) return;

    const pid = this.process.pid;
    if (!pid) return;

    try {
      if (process.platform === 'win32') {
        // Windows: taskkill /T 杀进程树
        execSync(`taskkill /PID ${pid} /T /F`, { stdio: 'ignore' });
      } else {
        // macOS/Linux: 先发 SIGTERM，超时后 SIGKILL
        process.kill(-pid, 'SIGTERM');
        await new Promise<void>((resolve) => {
          const timeout = setTimeout(() => {
            try {
              process.kill(-pid, 'SIGKILL');
            } catch {
              // 进程已退出
            }
            resolve();
          }, 5000);

          this.process?.on('exit', () => {
            clearTimeout(timeout);
            resolve();
          });
        });
      }
    } catch {
      // 进程可能已退出
    }

    this.process = null;
  }

  /**
   * 获取后端连接信息
   */
  getInfo(): BackendInfo | null {
    if (!this.port || !this.token) return null;
    return {
      port: this.port,
      token: this.token,
      baseUrl: `http://127.0.0.1:${this.port}`,
    };
  }

  /**
   * 获取诊断信息（用于崩溃 UI）
   */
  getDiagnostics(): string {
    const lines = [
      `ChatPDF Desktop Diagnostics`,
      `===========================`,
      `OS: ${process.platform} ${process.arch}`,
      `Electron: ${process.versions.electron}`,
      `Node: ${process.versions.node}`,
      `Port: ${this.port}`,
      `Data Dir: ${this.dataDir}`,
      `Restart Count: ${this.restartCount}/${this.maxRestarts}`,
      `Log File: ${path.join(this.dataDir, 'logs', 'chatpdf-backend.log')}`,
    ];
    return lines.join('\n');
  }

  // ---- 私有方法 ----

  /**
   * 按偏好端口列表尝试，全部被占则回退系统分配
   */
  private async findPort(): Promise<number> {
    for (const port of PREFERRED_PORTS) {
      if (await this.isPortAvailable(port)) {
        return port;
      }
    }
    // 回退：系统分配随机端口
    return new Promise((resolve, reject) => {
      const server = createServer();
      server.listen(0, '127.0.0.1', () => {
        const addr = server.address();
        if (addr && typeof addr === 'object') {
          const port = addr.port;
          server.close(() => resolve(port));
        } else {
          server.close(() => reject(new Error('Failed to get port')));
        }
      });
      server.on('error', reject);
    });
  }

  private isPortAvailable(port: number): Promise<boolean> {
    return new Promise((resolve) => {
      const server = createServer();
      server.listen(port, '127.0.0.1', () => {
        server.close(() => resolve(true));
      });
      server.on('error', () => resolve(false));
    });
  }

  /**
   * spawn Python 后端进程
   */
  private async spawnBackend(): Promise<void> {
    const backendPath = this.getBackendPath();
    const env = {
      ...process.env,
      CHATPDF_MODE: 'desktop',
      CHATPDF_PORT: String(this.port),
      CHATPDF_DATA_DIR: this.dataDir,
      CHATPDF_BACKEND_TOKEN: this.token,
      CHATPDF_LOG_LEVEL: 'INFO',
    };

    console.log(`[ProcessManager] Starting backend: ${backendPath}`);
    console.log(`[ProcessManager] Port: ${this.port}, Data: ${this.dataDir}`);

    if (backendPath.endsWith('.exe') || !backendPath.includes('.')) {
      // PyInstaller 打包后的可执行文件
      this.process = spawn(backendPath, [], {
        env,
        stdio: ['ignore', 'pipe', 'pipe'],
        detached: process.platform !== 'win32', // Unix 需要 detach 以便 kill 进程组
      });
    } else {
      // 开发模式：用 python 运行
      this.process = spawn('python', [backendPath], {
        env,
        stdio: ['ignore', 'pipe', 'pipe'],
        cwd: path.dirname(backendPath),
      });
    }

    // 转发后端日志
    this.process.stdout?.on('data', (data: Buffer) => {
      console.log(`[Backend] ${data.toString().trim()}`);
    });
    this.process.stderr?.on('data', (data: Buffer) => {
      console.error(`[Backend] ${data.toString().trim()}`);
    });

    // 进程退出处理
    this.process.on('exit', (code, signal) => {
      console.log(`[ProcessManager] Backend exited: code=${code}, signal=${signal}`);
      if (!this.isShuttingDown && this.restartCount < this.maxRestarts) {
        this.restartCount++;
        const delay = this.restartCount * 2000; // 递增延迟
        console.log(`[ProcessManager] Restarting in ${delay}ms (attempt ${this.restartCount}/${this.maxRestarts})`);
        setTimeout(() => {
          this.spawnBackend().then(() => this.waitForHealthy()).catch(console.error);
        }, delay);
      }
    });
  }

  /**
   * 确定后端可执行文件路径
   */
  private getBackendPath(): string {
    if (app.isPackaged) {
      // 打包后：extraResources/backend/ 目录
      const resourcesPath = process.resourcesPath;
      if (process.platform === 'win32') {
        return path.join(resourcesPath, 'backend', 'desktop_entry.exe');
      }
      return path.join(resourcesPath, 'backend', 'desktop_entry');
    }
    // 开发模式：直接用 Python 脚本
    return path.join(__dirname, '..', '..', 'backend', 'desktop_entry.py');
  }

  /**
   * 等待后端健康检查通过
   */
  private waitForHealthy(timeoutMs: number = 30000): Promise<void> {
    return new Promise((resolve, reject) => {
      const startTime = Date.now();
      const interval = 500;

      const check = () => {
        if (Date.now() - startTime > timeoutMs) {
          reject(new Error(`Backend health check timeout after ${timeoutMs}ms`));
          return;
        }

        const req = http.get(`http://127.0.0.1:${this.port}/health`, (res) => {
          if (res.statusCode === 200) {
            resolve();
          } else {
            setTimeout(check, interval);
          }
        });

        req.on('error', () => {
          setTimeout(check, interval);
        });

        req.setTimeout(2000, () => {
          req.destroy();
          setTimeout(check, interval);
        });
      };

      check();
    });
  }
}
