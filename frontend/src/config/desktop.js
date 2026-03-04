/**
 * 桌面模式适配层
 *
 * 在 Electron 桌面模式下：
 * - 拦截 fetch 请求，自动添加后端 base URL 和安全 token
 * - 拦截 XMLHttpRequest，同理处理（用于文件上传等场景）
 * - 提供 capabilities 查询接口
 *
 * 在 Web 模式下（Vite 代理）：
 * - 不做任何修改，完全透明
 *
 * 使用方式：在 index.jsx 最顶部 import 此文件即可生效
 */

// 检测是否在 Electron 桌面模式下运行
const isDesktop = typeof window !== 'undefined' && window.chatpdfDesktop?.isDesktop === true;

let _apiBaseUrl = '';
let _backendToken = '';
let _initialized = false;

/**
 * 初始化桌面模式配置（从 Electron preload 获取连接信息）
 */
async function initDesktopMode() {
  if (!isDesktop || _initialized) return;

  try {
    _apiBaseUrl = await window.chatpdfDesktop.getApiBaseUrl();
    _backendToken = await window.chatpdfDesktop.getBackendToken();
    _initialized = true;
    console.log(`[Desktop] Connected to backend: ${_apiBaseUrl}`);
  } catch (err) {
    console.error('[Desktop] Failed to initialize:', err);
  }
}

/**
 * 拦截 fetch —— 为相对路径请求添加 base URL 和 token header
 */
if (isDesktop) {
  const originalFetch = window.fetch;

  window.fetch = async function (input, init) {
    // 确保已初始化
    if (!_initialized) {
      await initDesktopMode();
    }

    let url = typeof input === 'string' ? input : input instanceof Request ? input.url : String(input);

    // 只处理相对路径（即发往后端的请求）
    if (url.startsWith('/') || (!url.startsWith('http://') && !url.startsWith('https://'))) {
      url = `${_apiBaseUrl}${url.startsWith('/') ? '' : '/'}${url}`;

      // 构建新的 init，注入 token header
      const newInit = { ...init };
      newInit.headers = new Headers(newInit.headers || {});
      if (_backendToken) {
        newInit.headers.set('X-ChatPDF-Token', _backendToken);
      }

      return originalFetch.call(window, url, newInit);
    }

    // 外部请求（如 CDN 资源）不修改
    return originalFetch.call(window, input, init);
  };

  /**
   * 拦截 XMLHttpRequest —— 用于文件上传等非 fetch 场景
   */
  const OriginalXHR = window.XMLHttpRequest;
  const originalOpen = OriginalXHR.prototype.open;
  const originalSetRequestHeader = OriginalXHR.prototype.setRequestHeader;
  const originalSend = OriginalXHR.prototype.send;

  OriginalXHR.prototype.open = function (method, url, ...args) {
    // 保存原始 URL 以便在 send 时判断
    this._chatpdfUrl = url;
    this._chatpdfHasTokenHeader = false;

    if (typeof url === 'string' && (url.startsWith('/') || (!url.startsWith('http://') && !url.startsWith('https://')))) {
      const fullUrl = `${_apiBaseUrl}${url.startsWith('/') ? '' : '/'}${url}`;
      return originalOpen.call(this, method, fullUrl, ...args);
    }

    return originalOpen.call(this, method, url, ...args);
  };

  OriginalXHR.prototype.setRequestHeader = function (name, value) {
    if (typeof name === 'string' && name.toLowerCase() === 'x-chatpdf-token') {
      this._chatpdfHasTokenHeader = true;
    }
    return originalSetRequestHeader.call(this, name, value);
  };

  OriginalXHR.prototype.send = function (...args) {
    // 为后端请求注入 token；若业务代码已手动设置则不重复注入，避免 "token, token" 导致鉴权失败
    if (_backendToken && this._chatpdfUrl &&
        !this._chatpdfHasTokenHeader &&
        (this._chatpdfUrl.startsWith('/') || this._chatpdfUrl.startsWith(_apiBaseUrl))) {
      this.setRequestHeader('X-ChatPDF-Token', _backendToken);
    }
    return originalSend.call(this, ...args);
  };
}

/**
 * 获取后端能力信息
 */
export async function fetchCapabilities() {
  try {
    const res = await fetch('/capabilities');
    if (res.ok) return await res.json();
  } catch (err) {
    console.error('[Desktop] Failed to fetch capabilities:', err);
  }
  return null;
}

/**
 * 检查是否为桌面模式
 */
export { isDesktop };

/**
 * 立即启动初始化（import 时触发）
 */
if (isDesktop) {
  initDesktopMode();
}
