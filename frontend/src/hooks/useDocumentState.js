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
 * 文档状态管理 Hook
 * 管理文档上传、docId、docInfo、会话历史等状态和逻辑
 *
 * @param {Object} options - 配置选项
 * @param {Function} options.getCurrentProvider - 获取当前 embedding provider
 * @param {Function} options.getCurrentEmbeddingModel - 获取当前 embedding 模型
 * @param {Function} options.setMessages - 设置消息列表（跨域状态）
 * @param {Function} options.setCurrentPage - 设置当前 PDF 页码（跨域状态）
 * @param {Function} options.setScreenshots - 设置截图列表（跨域状态）
 * @param {Function} options.setIsLoading - 设置加载状态（跨域状态）
 * @param {Function} options.setSelectedText - 设置选中文本（跨域状态）
 */
export function useDocumentState({
  getCurrentProvider,
  getCurrentEmbeddingModel,
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

    // 获取 embedding 配置
    const provider = getCurrentProvider?.();
    const emodel = getCurrentEmbeddingModel?.();
    if (emodel && provider) {
      const compositeKey = `${provider.id}:${emodel.id}`;
      formData.append('embedding_model', compositeKey);
      if (provider.id !== 'local') {
        if (!provider.apiKey) {
          alert(`请先为 ${provider.name} 配置 API Key`);
          setIsUploading(false);
          return;
        }
        formData.append('embedding_api_key', provider.apiKey);
        formData.append('embedding_api_host', provider.apiHost);
      }
    } else {
      formData.append('embedding_model', 'local:all-MiniLM-L6-v2');
    }

    // OCR 设置
    const ocrSettings = loadOCRSettings();
    formData.append('enable_ocr', ocrSettings.mode || 'auto');
    formData.append('ocr_backend', ocrSettings.backend || 'auto');

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
            reject(new Error('Upload failed'));
          }
        });
        xhr.addEventListener('error', () => reject(new Error('Network error')));
        xhr.open('POST', `${API_BASE_URL}/upload`);
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
  }, [getCurrentProvider, getCurrentEmbeddingModel, setMessages]);

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
        setMessages?.(s.messages || []);
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
