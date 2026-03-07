import { useState, useRef, useEffect, useCallback } from 'react';

/**
 * 内联 OCR 设置读取
 * 从 localStorage 中加载 OCR 配置
 */
const loadOCRSettings = () => {
  try {
    const raw = localStorage.getItem('ocrSettings');
    if (raw) {
      const parsed = JSON.parse(raw);
      const validModes = ['auto', 'always', 'never'];
      const validBackends = ['auto', 'tesseract', 'paddleocr', 'mistral', 'mineru', 'doc2x'];
      return {
        mode: validModes.includes(parsed.mode) ? parsed.mode : 'auto',
        backend: validBackends.includes(parsed.backend) ? parsed.backend : 'auto',
      };
    }
  } catch { /* ignore */ }
  return { mode: 'auto', backend: 'auto' };
};

// API base URL
const API_BASE_URL = '';

/**
 * 解析后端错误响应，尽量提取可读错误信息
 */
const getUploadErrorMessage = (xhr) => {
  if (xhr.status === 401) {
    return '桌面后端鉴权失败，请重启 ChatPDF Pro 后重试';
  }

  const fallback = `Upload failed (HTTP ${xhr.status || 'unknown'})`;
  const raw = xhr.responseText;
  if (!raw) return fallback;

  try {
    const parsed = JSON.parse(raw);
    if (typeof parsed?.detail === 'string' && parsed.detail.trim()) {
      return parsed.detail;
    }
    if (Array.isArray(parsed?.detail)) {
      const msgs = parsed.detail
        .map((item) => {
          if (typeof item === 'string') return item;
          if (item && typeof item.msg === 'string') return item.msg;
          return null;
        })
        .filter(Boolean);
      if (msgs.length > 0) return msgs.join('；');
    }
    if (typeof parsed?.message === 'string' && parsed.message.trim()) {
      return parsed.message;
    }
  } catch {
    // ignore JSON parse error
  }

  return raw.slice(0, 300) || fallback;
};

/**
 * 文档状态管理 Hook
 * 管理文档上传、docId、docInfo、会话历史等状态和逻辑
 *
 * @param {Object} options - 配置选项
 * @param {Function} options.getEmbeddingConfig - 获取 embedding 配置（compositeKey + provider）
 * @param {Function} options.setMessages - 设置消息列表（跨域状态）
 * @param {Function} options.setCurrentPage - 设置当前 PDF 页码（跨域状态）
 * @param {Function} options.setScreenshots - 设置截图列表（跨域状态）
 * @param {Function} options.setIsLoading - 设置加载状态（跨域状态）
 * @param {Function} options.setSelectedText - 设置选中文本（跨域状态）
 */
export function useDocumentState({
  getEmbeddingConfig,
  getChatCredentials,
  getProviderById,
  setMessages,
  setCurrentPage,
  setScreenshots,
  setIsLoading,
  setSelectedText,
} = {}) {
  // 文档核心状态
  const [docId, setDocId] = useState(null);
  const [docInfo, setDocInfo] = useState(null);

  // 上传状态
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState('uploading');

  // 会话历史
  const [history, setHistory] = useState([]);

  // 存储信息
  const [storageInfo, setStorageInfo] = useState(null);

  // 速览（Overview）状态
  const [overview, setOverview] = useState(null);
  const [overviewLoading, setOverviewLoading] = useState(false);
  const [overviewError, setOverviewError] = useState(null);

  // 文件输入引用
  const fileInputRef = useRef(null);

  /**
   * 获取存储信息
   */
  const fetchStorageInfo = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/storage_info`);
      if (res.ok) setStorageInfo(await res.json());
    } catch (e) {
      console.error(e);
    }
  }, []);

  /**
   * 加载会话历史
   */
  const loadHistory = useCallback(() => {
    const s = localStorage.getItem('chatHistory');
    if (s) setHistory(JSON.parse(s));
  }, []);

  /**
   * 文件上传处理
   */
  const handleFileUpload = useCallback(async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setIsUploading(true);
    setUploadProgress(0);
    setUploadStatus('uploading');

    const formData = new FormData();
    formData.append('file', file);

    // 获取 embedding 配置（直接从 DefaultsContext 读取 compositeKey，不依赖 ModelContext）
    const embeddingConfig = getEmbeddingConfig?.();
    if (!embeddingConfig || embeddingConfig.isValid === false) {
      let reasonHint = '请先在设置中选择可用的 Embedding 模型并配置 API Key。';
      if (embeddingConfig?.reason === 'model_not_found') {
        reasonHint = `当前默认 Embedding 模型不存在或已下线：${embeddingConfig.providerId || ''}:${embeddingConfig.modelId || ''}。\n请重新选择可用的 Embedding 模型。`;
      } else if (embeddingConfig?.reason === 'wrong_type') {
        reasonHint = `当前默认模型类型是 ${embeddingConfig.modelType || 'unknown'}，不是 Embedding。\n请切换到 Embedding 模型后再上传。`;
      } else if (embeddingConfig?.reason === 'provider_missing') {
        reasonHint = `当前默认模型对应的 Provider 不存在：${embeddingConfig.providerId || 'unknown'}。\n请在模型服务中重新配置。`;
      }
      alert(`${reasonHint}\n\n路径：右上角设置 → 模型服务 → EMBEDDING`);
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
      return;
    }

    formData.append('embedding_model', embeddingConfig.compositeKey);
    if (embeddingConfig.providerId !== 'local') {
      if (!embeddingConfig.provider?.apiKey) {
        alert(`请先为 ${embeddingConfig.provider?.name || embeddingConfig.providerId} 配置 API Key`);
        setIsUploading(false);
        return;
      }
      formData.append('embedding_api_key', embeddingConfig.provider.apiKey);
      formData.append('embedding_api_host', embeddingConfig.provider.apiHost);
    }

    // OCR 设置
    const ocrSettings = loadOCRSettings();
    formData.append('enable_ocr', ocrSettings.mode || 'auto');
    formData.append('ocr_backend', ocrSettings.backend || 'auto');

    // 桌面模式显式注入后端地址和 token，避免 XHR 拦截初始化时序导致 401
    let uploadUrl = `${API_BASE_URL}/upload`;
    let backendToken = '';
    const isDesktop = typeof window !== 'undefined' && window.chatpdfDesktop?.isDesktop === true;
    if (isDesktop) {
      try {
        const desktopBase = await window.chatpdfDesktop.getApiBaseUrl();
        const normalizedBase = desktopBase ? desktopBase.replace(/\/$/, '') : '';
        uploadUrl = normalizedBase ? `${normalizedBase}/upload` : '/upload';
        backendToken = (await window.chatpdfDesktop.getBackendToken()) || '';
      } catch (e) {
        console.warn('[Upload] 获取桌面后端配置失败，回退默认请求路径', e);
      }
    }

    try {
      const data = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.upload.addEventListener('progress', (e) => {
          if (e.lengthComputable) setUploadProgress(Math.round((e.loaded / e.total) * 70));
        });
        xhr.addEventListener('load', () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            setUploadStatus('processing');
            setUploadProgress(75);
            try {
              resolve(JSON.parse(xhr.responseText));
            } catch (e) {
              reject(e);
            }
          } else {
            reject(new Error(getUploadErrorMessage(xhr)));
          }
        });
        xhr.addEventListener('error', () => reject(new Error('Network error')));
        xhr.open('POST', uploadUrl);
        if (backendToken) {
          xhr.setRequestHeader('X-ChatPDF-Token', backendToken);
        }
        xhr.send(formData);
      });

      setDocId(data.doc_id);

      // 获取文档详细信息
      const dres = await fetch(`${API_BASE_URL}/document/${data.doc_id}?t=${Date.now()}`);
      const ddata = await dres.json();
      const full = { ...ddata, ...data };
      setDocInfo(full);

      // 构建上传成功消息
      let uploadMsg = `✅ 文档《${data.filename}》上传成功！共 ${data.total_pages} 页。`;
      if (data.ocr_used) {
        uploadMsg += `\n🔍 已使用 OCR（${data.ocr_backend || '自动'}）处理部分页面。`;
      }
      setMessages?.([{ type: 'system', content: uploadMsg }]);
    } catch (error) {
      alert(`上传失败: ${error.message}`);
    } finally {
      setTimeout(() => {
        setIsUploading(false);
        setUploadProgress(0);
        setUploadStatus('uploading');
      }, 500);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  }, [getEmbeddingConfig, setMessages]);

  /**
   * 开始新对话（重置文档和相关状态）
   */
  const startNewChat = useCallback(() => {
    setDocId(null);
    setDocInfo(null);
    setMessages?.([]);
    setCurrentPage?.(1);
    setSelectedText?.('');
    setScreenshots?.([]);
  }, [setMessages, setCurrentPage, setSelectedText, setScreenshots]);

  /**
   * 加载历史会话
   */
  const loadSession = useCallback(async (s) => {
    setIsLoading?.(true);
    try {
      const res = await fetch(`${API_BASE_URL}/document/${s.docId}?t=${Date.now()}`);
      if (res.ok) {
        setDocId(s.docId);
        setDocInfo(await res.json());
        // 恢复历史消息：确保不存在"永久流式中"的脏消息（页面关闭时可能残留 isStreaming:true）
        const restoredMessages = (s.messages || []).map((m) =>
          m.isStreaming ? { ...m, isStreaming: false } : m
        );
        setMessages?.(restoredMessages);
        setCurrentPage?.(1);
      }
    } catch (e) {
      // 静默处理
    } finally {
      setIsLoading?.(false);
    }
  }, [setMessages, setCurrentPage, setIsLoading]);

  /**
   * 删除历史会话
   */
  const deleteSession = useCallback((sid) => {
    if (!window.confirm('确定要删除这个对话吗？')) return;
    const h = JSON.parse(localStorage.getItem('chatHistory') || '[]');
    const next = h.filter(x => x.id !== sid);
    localStorage.setItem('chatHistory', JSON.stringify(next));
    setHistory(next);
    if (sid === docId) {
      setDocId(null);
      setDocInfo(null);
      setMessages?.([]);
    }
  }, [docId, setMessages]);

  /**
   * 保存当前会话到历史
   * 需要外部传入 messages 和 docInfo，因为这些可能来自其他 hook
   */
  const saveCurrentSession = useCallback((messages) => {
    if (!docId || !docInfo) return;
    const h = JSON.parse(localStorage.getItem('chatHistory') || '[]');
    const idx = h.findIndex(x => x.id === docId);
    const data = {
      id: docId,
      docId,
      filename: docInfo.filename,
      messages,
      updatedAt: Date.now(),
      createdAt: idx >= 0 ? h[idx].createdAt : Date.now(),
    };
    if (idx >= 0) h[idx] = data;
    else h.unshift(data);
    const lim = h.slice(0, 50);
    localStorage.setItem('chatHistory', JSON.stringify(lim));
    setHistory(lim);
  }, [docId, docInfo]);

  /**
   * 获取速览数据
   * @param {string} depth - 速览深度: brief(简介) / standard(标准) / detailed(详细)
   */
  const fetchOverview = useCallback(async (depth = 'standard') => {
    if (!docId) {
      setOverviewError('请先上传文档');
      return;
    }

    const chatCredentials = getChatCredentials?.();
    const chatProvider = chatCredentials?.providerId || 'openai';
    const chatModel = chatCredentials?.modelId || 'gpt-4o';
    const chatApiKey = chatCredentials?.apiKey || '';
    const chatProviderFull = getProviderById?.(chatProvider);

    if (getChatCredentials && !chatApiKey && chatProvider !== 'local' && chatProvider !== 'ollama') {
      setOverviewError(`请先为 ${chatProviderFull?.name || chatProvider} 配置 API Key`);
      return;
    }

    setOverviewLoading(true);
    setOverviewError(null);

    try {
      const params = new URLSearchParams({ depth });
      const headers = {};
      if (chatCredentials) {
        headers['X-ChatPDF-Provider'] = chatProvider;
        headers['X-ChatPDF-Model'] = chatModel;
        if (chatApiKey) {
          headers['X-ChatPDF-Api-Key'] = chatApiKey;
        }
        if (chatProviderFull?.apiHost) {
          headers['X-ChatPDF-Api-Host'] = chatProviderFull.apiHost;
        }
      }

      const res = await fetch(`${API_BASE_URL}/documents/${docId}/overview?${params}`, {
        headers,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({ detail: '获取速览失败' }));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setOverview(data);
    } catch (err) {
      console.error('获取速览失败:', err);
      setOverviewError(err.message || '获取速览失败');
    } finally {
      setOverviewLoading(false);
    }
  }, [docId, getChatCredentials, getProviderById]);

  // 初始化时加载历史
  useEffect(() => {
    loadHistory();
    fetchStorageInfo();
  }, [loadHistory, fetchStorageInfo]);

  return {
    // 文档状态
    docId,
    setDocId,
    docInfo,
    setDocInfo,

    // 上传状态
    isUploading,
    uploadProgress,
    uploadStatus,

    // 会话历史
    history,
    setHistory,

    // 存储信息
    storageInfo,

    // 速览（Overview）状态
    overview,
    setOverview,
    overviewLoading,
    overviewError,
    fetchOverview,

    // 引用
    fileInputRef,

    // 方法
    handleFileUpload,
    startNewChat,
    loadSession,
    deleteSession,
    saveCurrentSession,
    loadHistory,
    fetchStorageInfo,
  };
}
