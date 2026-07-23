import { useState, useRef, useEffect, useCallback } from 'react';
import { VALID_PARSE_ROUTES } from '../utils/parseRouteUtils';

const VALID_PAGE_OCR_BACKENDS = ['auto', 'tesseract', 'paddleocr', 'mistral'];
const LEGACY_PAGE_OCR_BACKENDS = ['mineru', 'doc2x'];
const DEFAULT_OCR_SETTINGS = {
  mode: 'auto',
  backend: 'auto',
  parseRoute: 'auto',
  mineruFigureEnhance: true,
  figureRenderMode: 'raw',
};

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
      const validFigureRenderModes = ['raw', 'yolo'];
      const settings = {
        mode: validModes.includes(parsed.mode) ? parsed.mode : 'auto',
        backend: VALID_PAGE_OCR_BACKENDS.includes(parsed.backend) ? parsed.backend : 'auto',
        parseRoute: VALID_PARSE_ROUTES.includes(parsed.parseRoute) ? parsed.parseRoute : 'auto',
        mineruFigureEnhance: typeof parsed.mineruFigureEnhance === 'boolean' ? parsed.mineruFigureEnhance : true,
        figureRenderMode: validFigureRenderModes.includes(parsed.figureRenderMode)
          ? parsed.figureRenderMode
          : 'raw',
      };
      if (LEGACY_PAGE_OCR_BACKENDS.includes(parsed.backend)) {
        localStorage.setItem('ocrSettings', JSON.stringify({
          ...parsed,
          backend: 'auto',
          parseRoute: settings.parseRoute,
        }));
      }
      return settings;
    }
  } catch { /* ignore */ }
  return { ...DEFAULT_OCR_SETTINGS };
};

// API base URL
const API_BASE_URL = '';
const UPLOAD_PROGRESS_CAP = 42;
const PROCESSING_PROGRESS_CAP = 88;

const buildDocumentInfoUrl = (docId, includeContent) => (
  `${API_BASE_URL}/document/${encodeURIComponent(docId)}?include_content=${includeContent ? 'true' : 'false'}&t=${Date.now()}`
);

const fetchDocumentInfo = async (docId) => {
  const metadataResponse = await fetch(buildDocumentInfoUrl(docId, false));
  if (!metadataResponse.ok) {
    return { response: metadataResponse, data: null };
  }

  const metadata = await metadataResponse.json();
  if (metadata.pdf_url) {
    return { response: metadataResponse, data: metadata };
  }

  const contentResponse = await fetch(buildDocumentInfoUrl(docId, true));
  if (!contentResponse.ok) {
    return { response: contentResponse, data: null };
  }
  return { response: contentResponse, data: await contentResponse.json() };
};

const getOverviewParseIdentity = (documentInfo) => {
  const manifest = documentInfo?.parse_manifest || {};
  const generation = String(manifest.generation || '').trim();
  const sourceHash = String(manifest.source_hash || '').trim();
  return `g=${encodeURIComponent(generation || 'legacy')}:s=${encodeURIComponent(sourceHash || 'legacy')}`;
};

const CHAT_RELIABLE_TERMINAL_STATES = new Set([
  'completed',
  'recovered_retry',
  'evidence_fallback',
  'degraded',
  'failed',
  'interrupted',
]);

const normalizeParseIdentity = (value) => {
  const nested = value?.parse_identity && typeof value.parse_identity === 'object'
    ? value.parse_identity
    : {};
  return {
    generation: String(
      value?.parse_generation
      || value?.parseGeneration
      || value?.generation
      || nested.parse_generation
      || nested.generation
      || ''
    ).trim(),
    sourceHash: String(
      value?.document_source_hash
      || value?.documentSourceHash
      || value?.source_hash
      || nested.document_source_hash
      || nested.source_hash
      || ''
    ).trim(),
  };
};

const getDocumentParseIdentity = (documentInfo) => normalizeParseIdentity(
  documentInfo?.parse_manifest || {}
);

const hasCompleteParseIdentity = (identity) => Boolean(
  identity?.generation && identity?.sourceHash
);

const parseIdentitiesMatch = (left, right) => (
  hasCompleteParseIdentity(left)
  && hasCompleteParseIdentity(right)
  && left.generation === right.generation
  && left.sourceHash === right.sourceHash
);

const getMessageTurnStatus = (message) => String(
  message?.turnStatus
  || message?.turn_status
  || message?.answerStatus
  || message?.answer_status
  || ''
).trim().toLowerCase();

const normalizeRestoredSessionMessages = (session, documentInfo) => {
  const messages = Array.isArray(session?.messages) ? session.messages : [];
  const currentIdentity = getDocumentParseIdentity(documentInfo);
  const sessionIdentity = normalizeParseIdentity(session || {});
  const sessionMatchesCurrent = parseIdentitiesMatch(sessionIdentity, currentIdentity);

  return messages.map((message) => {
    if (!message || typeof message !== 'object') return message;
    const messageIdentity = normalizeParseIdentity(message);
    const hasMessageIdentity = hasCompleteParseIdentity(messageIdentity);
    const messageMatchesCurrent = parseIdentitiesMatch(messageIdentity, currentIdentity)
      || (!hasMessageIdentity && sessionMatchesCurrent);
    const staleIdentity = hasCompleteParseIdentity(currentIdentity) && !messageMatchesCurrent;
    const next = {
      ...message,
      ...(messageMatchesCurrent
        ? {
          parse_generation: currentIdentity.generation,
          document_source_hash: currentIdentity.sourceHash,
        }
        : {}),
      ...(staleIdentity ? { parseIdentityStale: true } : {}),
    };

    if (message.type !== 'assistant') return next;

    const turnStatus = getMessageTurnStatus(message);
    const interrupted = staleIdentity
      || message.isStreaming === true
      || ['streaming', 'cancelled', 'aborted'].includes(turnStatus)
      || !CHAT_RELIABLE_TERMINAL_STATES.has(turnStatus);
    return {
      ...next,
      isStreaming: false,
      turnStatus: interrupted ? 'interrupted' : turnStatus,
    };
  });
};

const isKeylessLocalProvider = (provider) => ['local', 'ollama'].includes(String(provider || '').trim().toLowerCase());

const getOverviewTextIdentity = (credentials) => {
  const provider = String(credentials?.providerId || '').trim();
  const model = String(credentials?.modelId || '').trim();
  const host = String(credentials?.apiHost || '').trim().replace(/\/$/, '');
  const available = Boolean(credentials?.apiKey) || isKeylessLocalProvider(provider);
  return `p=${encodeURIComponent(provider || 'default')}:m=${encodeURIComponent(model || 'default')}:h=${encodeURIComponent(host)}:a=${available ? 'ready' : 'unavailable'}`;
};

const getOverviewVisualIdentity = (credentials) => {
  const provider = String(credentials?.providerId || '').trim();
  const model = String(credentials?.modelId || '').trim();
  const host = String(credentials?.apiHost || '').trim().replace(/\/$/, '');
  const visionCapable = credentials?.isVisionCapable === true;
  const available = visionCapable && (Boolean(credentials?.apiKey) || isKeylessLocalProvider(provider));
  const strategy = String(credentials?.strategy || 'balanced').trim();
  const localProvider = String(credentials?.local?.providerId || '').trim();
  const localModel = String(credentials?.local?.modelId || '').trim();
  const localHost = String(credentials?.local?.apiHost || '').trim().replace(/\/$/, '');
  const localAvailable = credentials?.local?.isVisionCapable === true
    && (Boolean(credentials?.local?.apiKey) || isKeylessLocalProvider(localProvider));
  return `s=${encodeURIComponent(strategy)}:p=${encodeURIComponent(provider || 'default')}:m=${encodeURIComponent(model || 'default')}:h=${encodeURIComponent(host)}:e=${visionCapable ? 'vision' : 'disabled'}:a=${available ? 'ready' : 'unavailable'}:lp=${encodeURIComponent(localProvider)}:lm=${encodeURIComponent(localModel)}:lh=${encodeURIComponent(localHost)}:la=${localAvailable ? 'ready' : 'unavailable'}`;
};

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
  getVisualCredentials,
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
  const [uploadFileInfo, setUploadFileInfo] = useState(null);
  const uploadProcessingTimerRef = useRef(null);
  const uploadResetTimerRef = useRef(null);
  const uploadEpochRef = useRef(0);

  // 会话历史
  const [history, setHistory] = useState([]);

  // 存储信息
  const [storageInfo, setStorageInfo] = useState(null);

  // 速览（Overview）状态
  const [overview, setOverview] = useState(null);
  const [overviewLoading, setOverviewLoading] = useState(false);
  const [overviewError, setOverviewError] = useState(null);
  const overviewCacheRef = useRef(new Map());
  const overviewInflightRef = useRef(new Map());
  const currentOverviewKeyRef = useRef('');
  const overviewEpochRef = useRef(0);
  const overviewObservedContextRef = useRef(null);
  const overviewContextRef = useRef({ docId: '', parseIdentity: '', textIdentity: '', visualIdentity: '' });
  const overviewParseIdentity = getOverviewParseIdentity(docInfo);
  const overviewTextIdentity = getOverviewTextIdentity(getChatCredentials?.());
  const overviewVisualIdentity = getOverviewVisualIdentity(getVisualCredentials?.());
  overviewContextRef.current = {
    docId: docId || '',
    parseIdentity: overviewParseIdentity,
    textIdentity: overviewTextIdentity,
    visualIdentity: overviewVisualIdentity,
  };
  // 文件输入引用
  const fileInputRef = useRef(null);

  const clearUploadProcessingTimer = useCallback(() => {
    if (uploadProcessingTimerRef.current) {
      window.clearInterval(uploadProcessingTimerRef.current);
      uploadProcessingTimerRef.current = null;
    }
  }, []);

  const startUploadProcessingTimer = useCallback(() => {
    clearUploadProcessingTimer();
    setUploadStatus('extracting');
    setUploadProgress((prev) => Math.max(prev, UPLOAD_PROGRESS_CAP + 3));

    uploadProcessingTimerRef.current = window.setInterval(() => {
      setUploadProgress((prev) => {
        if (prev >= PROCESSING_PROGRESS_CAP) return PROCESSING_PROGRESS_CAP;
        if (prev >= 76) {
          setUploadStatus('finalizing');
          return Math.min(PROCESSING_PROGRESS_CAP, prev + 0.35);
        }
        if (prev >= 58) {
          setUploadStatus('indexing');
          return Math.min(PROCESSING_PROGRESS_CAP, prev + 0.8);
        }
        setUploadStatus('extracting');
        return Math.min(PROCESSING_PROGRESS_CAP, prev + 1.6);
      });
    }, 700);
  }, [clearUploadProcessingTimer]);

  const buildOverviewKey = useCallback((
    currentDocId,
    depth,
    currentDocInfo = docInfo,
    visualIdentity = overviewVisualIdentity,
    textIdentity = overviewTextIdentity,
  ) => {
    if (!currentDocId) return '';
    return `${currentDocId}:${getOverviewParseIdentity(currentDocInfo)}:${textIdentity}:${visualIdentity}:${depth}`;
  }, [docInfo, overviewTextIdentity, overviewVisualIdentity]);

  const clearOverviewEntriesForDocument = useCallback((currentDocId) => {
    if (!currentDocId) return;
    const prefix = `${currentDocId}:`;
    [...overviewCacheRef.current.keys()].forEach((key) => {
      if (key.startsWith(prefix)) overviewCacheRef.current.delete(key);
    });
    [...overviewInflightRef.current.keys()].forEach((key) => {
      if (key.startsWith(prefix)) overviewInflightRef.current.delete(key);
    });
  }, []);

  const invalidateOverviewEpoch = useCallback((currentDocId, {
    clearDocumentCache = false,
    clearAllCaches = false,
  } = {}) => {
    overviewEpochRef.current += 1;
    if (clearAllCaches) {
      overviewCacheRef.current.clear();
      overviewInflightRef.current.clear();
    } else if (clearDocumentCache) {
      clearOverviewEntriesForDocument(currentDocId);
    }
    currentOverviewKeyRef.current = '';
    setOverview(null);
    setOverviewError(null);
    setOverviewLoading(false);
    return overviewEpochRef.current;
  }, [clearOverviewEntriesForDocument]);

  const clearOverviewCache = useCallback((currentDocId = docId) => {
    invalidateOverviewEpoch(currentDocId, {
      clearDocumentCache: Boolean(currentDocId),
      clearAllCaches: !currentDocId,
    });
  }, [docId, invalidateOverviewEpoch]);

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
  const handleFileUpload = useCallback(async (event, options = {}) => {
    const file = event.target.files[0];
    if (!file) return;
    const uploadEpoch = ++uploadEpochRef.current;
    if (uploadResetTimerRef.current) {
      window.clearTimeout(uploadResetTimerRef.current);
      uploadResetTimerRef.current = null;
    }

    setIsUploading(true);
    setUploadProgress(0);
    setUploadStatus('uploading');
    setUploadFileInfo({
      name: file.name || 'PDF 文档',
      size: Number(file.size) || 0,
    });
    clearUploadProcessingTimer();
    // 同一 PDF 重传会复用 docId，必须在上传开始时丢弃旧解析代际的速览结果。
    invalidateOverviewEpoch(docId, { clearDocumentCache: Boolean(docId) });

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
      setUploadFileInfo(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
      return;
    }

    formData.append('embedding_model', embeddingConfig.compositeKey);
    formData.append('embedding_provider', embeddingConfig.providerId || '');
    const embeddingProviderId = String(embeddingConfig.providerId || '').trim();
    const embeddingApiHost = String(embeddingConfig.provider?.apiHost || '').trim();
    if (!isKeylessLocalProvider(embeddingProviderId)) {
      if (!embeddingConfig.provider?.apiKey) {
        alert(`请先为 ${embeddingConfig.provider?.name || embeddingConfig.providerId} 配置 API Key`);
        setIsUploading(false);
        setUploadFileInfo(null);
        if (fileInputRef.current) fileInputRef.current.value = '';
        return;
      }
      formData.append('embedding_api_key', embeddingConfig.provider.apiKey);
    }
    if (embeddingApiHost) {
      formData.append('embedding_api_host', embeddingApiHost);
    }

    // OCR 设置
    const ocrSettings = loadOCRSettings();
    formData.append('enable_ocr', ocrSettings.mode || 'auto');
    formData.append('ocr_backend', ocrSettings.backend || 'auto');
    const requestedParseRoute = VALID_PARSE_ROUTES.includes(options.parseRoute)
      ? options.parseRoute
      : ocrSettings.parseRoute || 'auto';
    formData.append('parse_route', requestedParseRoute);

    // 桌面模式显式使用后端地址；鉴权 header 由 Electron 主进程注入。
    let uploadUrl = `${API_BASE_URL}/upload`;
    const isDesktop = typeof window !== 'undefined' && window.chatpdfDesktop?.isDesktop === true;
    if (isDesktop) {
      try {
        const desktopBase = await window.chatpdfDesktop.getApiBaseUrl();
        const normalizedBase = desktopBase ? desktopBase.replace(/\/$/, '') : '';
        uploadUrl = normalizedBase ? `${normalizedBase}/upload` : '/upload';
      } catch (e) {
        console.warn('[Upload] 获取桌面后端配置失败，回退默认请求路径', e);
      }
    }

    try {
      const data = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.upload.addEventListener('progress', (e) => {
          if (e.lengthComputable) {
            setUploadProgress(Math.min(UPLOAD_PROGRESS_CAP, Math.round((e.loaded / e.total) * UPLOAD_PROGRESS_CAP)));
          }
        });
        xhr.upload.addEventListener('load', () => {
          startUploadProcessingTimer();
        });
        xhr.addEventListener('load', () => {
          clearUploadProcessingTimer();
          if (xhr.status >= 200 && xhr.status < 300) {
            setUploadStatus('finalizing');
            setUploadProgress(90);
            try {
              resolve(JSON.parse(xhr.responseText));
            } catch (e) {
              reject(e);
            }
          } else {
            reject(new Error(getUploadErrorMessage(xhr)));
          }
        });
        xhr.addEventListener('error', () => {
          clearUploadProcessingTimer();
          reject(new Error('Network error'));
        });
        xhr.open('POST', uploadUrl);
        xhr.send(formData);
      });

      setDocId(data.doc_id);
      setUploadStatus('loading_document');
      setUploadProgress(93);

      // 获取文档详细信息
      const { response: dres, data: ddata } = await fetchDocumentInfo(data.doc_id);
      if (!dres.ok || !ddata) {
        throw new Error('文档元数据加载失败');
      }
      const full = { ...ddata, ...data };
      setDocInfo(full);
      const uploadedManifest = full?.parse_manifest || data?.parse_manifest || {};
      const isMinerUAccepted = uploadedManifest?.resolved_route === 'mineru'
        && full?.parse_ready !== true
        && uploadedManifest?.status !== 'ready';
      setUploadStatus(isMinerUAccepted ? 'accepted' : 'ready');
      setUploadProgress(100);

      // 构建上传成功消息
      let uploadMsg = `✅ 文档《${data.filename}》上传成功！共 ${data.total_pages} 页。`;
      if (uploadedManifest?.resolved_route === 'mineru' && uploadedManifest?.status !== 'ready') {
        uploadMsg += '\n⏳ 已选择 MinerU 全程解析，正文、阅读、大纲、翻译、速览和问答索引会在同一解析结果发布后可用。';
      }
      if (data.ocr_used) {
        uploadMsg += `\n🔍 已使用 OCR（${data.ocr_backend || '自动'}）处理部分页面。`;
      }
      const ocrStatus = String(data.ocr_status || '').trim().toLowerCase();
      if (data.ocr_warning || ['partial_success', 'failed', 'unavailable'].includes(ocrStatus)) {
        const failedPageCount = Array.isArray(data.ocr_failed_pages)
          ? data.ocr_failed_pages.length
          : 0;
        const fallbackWarning = ocrStatus === 'partial_success'
          ? `OCR 部分完成${failedPageCount > 0 ? `，${failedPageCount} 页未识别` : ''}`
          : 'OCR 未完成，已保留原始提取文本';
        uploadMsg += `\n⚠️ ${data.ocr_warning || fallbackWarning}`;
      }
      if (data.indexing_status && data.indexing_status !== 'ready') {
        uploadMsg += '\n⏳ 检索索引正在后台准备中，PDF 阅读可先使用；首次问答可能需要稍等。';
      }
      setMessages?.([{ type: 'system', content: uploadMsg }]);
    } catch (error) {
      alert(`上传失败: ${error.message}`);
    } finally {
      clearUploadProcessingTimer();
      uploadResetTimerRef.current = window.setTimeout(() => {
        if (uploadEpochRef.current !== uploadEpoch) return;
        setIsUploading(false);
        setUploadProgress(0);
        setUploadStatus('uploading');
        setUploadFileInfo(null);
        uploadResetTimerRef.current = null;
      }, 500);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  }, [
    clearUploadProcessingTimer,
    docId,
    getEmbeddingConfig,
    invalidateOverviewEpoch,
    setMessages,
    startUploadProcessingTimer,
  ]);

  useEffect(() => {
    return () => {
      clearUploadProcessingTimer();
      if (uploadResetTimerRef.current) window.clearTimeout(uploadResetTimerRef.current);
    };
  }, [clearUploadProcessingTimer]);

  /**
   * 开始新对话（重置文档和相关状态）
   */
  const startNewChat = useCallback(() => {
    invalidateOverviewEpoch(docId);
    setDocId(null);
    setDocInfo(null);
    setMessages?.([]);
    setCurrentPage?.(1);
    setSelectedText?.('');
    setScreenshots?.([]);
  }, [docId, invalidateOverviewEpoch, setMessages, setCurrentPage, setSelectedText, setScreenshots]);

  /**
   * 加载历史会话
   */
  const loadSession = useCallback(async (s) => {
    setIsLoading?.(true);
    try {
      const { response, data } = await fetchDocumentInfo(s.docId);
      if (response.ok && data) {
        // 重开当前会话时保留已展示的速览；只有实际切换文档才让旧请求失效。
        if (String(s.docId || '') !== String(docId || '')) {
          invalidateOverviewEpoch(docId);
        }
        setDocId(s.docId);
        setDocInfo(data);
        // 旧版没有可靠终态的助手消息一律按中断恢复；解析身份不匹配的
        // 历史仍可展示，但会被标记为 stale，不能进入新代际的上下文。
        const restoredMessages = normalizeRestoredSessionMessages(s, data);
        setMessages?.(restoredMessages);
        setCurrentPage?.(1);
      }
    } catch (e) {
      // 静默处理
    } finally {
      setIsLoading?.(false);
    }
  }, [docId, invalidateOverviewEpoch, setMessages, setCurrentPage, setIsLoading]);

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
      invalidateOverviewEpoch(docId, { clearDocumentCache: true });
      setDocId(null);
      setDocInfo(null);
      setMessages?.([]);
    }
  }, [docId, invalidateOverviewEpoch, setMessages]);

  /**
   * 保存当前会话到历史
   * 需要外部传入 messages 和 docInfo，因为这些可能来自其他 hook
   */
  const saveCurrentSession = useCallback((messages) => {
    if (!docId || !docInfo) return;
    const h = JSON.parse(localStorage.getItem('chatHistory') || '[]');
    const idx = h.findIndex(x => x.id === docId);
    const parseIdentity = getDocumentParseIdentity(docInfo);
    const data = {
      id: docId,
      docId,
      filename: docInfo.filename,
      messages,
      ...(hasCompleteParseIdentity(parseIdentity)
        ? {
          parse_generation: parseIdentity.generation,
          document_source_hash: parseIdentity.sourceHash,
          parse_identity: {
            parse_generation: parseIdentity.generation,
            document_source_hash: parseIdentity.sourceHash,
          },
        }
        : {}),
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
   * @param {{force?: boolean}} options - force=true 时绕过本地和后端缓存重新生成
   */
  const fetchOverview = useCallback(async (depth = 'standard', options = {}) => {
    if (!docId) {
      setOverviewError('请先上传文档');
      return;
    }
    const force = Boolean(options?.force);
    const requestDocId = docId;
    const requestParseIdentity = overviewParseIdentity;
    const requestTextIdentity = overviewTextIdentity;
    const requestVisualIdentity = overviewVisualIdentity;
    const requestEpoch = overviewEpochRef.current;
    const isCurrentOverviewRequest = () => {
      const currentContext = overviewContextRef.current;
      return (
        overviewEpochRef.current === requestEpoch
        && currentContext.docId === requestDocId
        && currentContext.parseIdentity === requestParseIdentity
        && currentContext.textIdentity === requestTextIdentity
        && currentContext.visualIdentity === requestVisualIdentity
      );
    };

    // A caller can retain an older callback across a document or parse-route
    // switch. Check before cache hits and local loading/error updates too,
    // not only after the network response.
    if (!isCurrentOverviewRequest()) return undefined;

    const overviewKey = buildOverviewKey(
      requestDocId,
      depth,
      docInfo,
      requestVisualIdentity,
      requestTextIdentity,
    );
    if (!force && currentOverviewKeyRef.current === overviewKey && overview) {
      return overview;
    }

    const cachedOverview = overviewCacheRef.current.get(overviewKey);
    if (!force && cachedOverview) {
      currentOverviewKeyRef.current = overviewKey;
      setOverview(cachedOverview);
      setOverviewError(null);
      return cachedOverview;
    }

    const inflightEntry = overviewInflightRef.current.get(overviewKey);
    if (!force && inflightEntry?.epoch === requestEpoch) {
      return inflightEntry.promise;
    }
    if (inflightEntry) {
      overviewInflightRef.current.delete(overviewKey);
    }

    const chatCredentials = getChatCredentials?.();
    const chatProvider = chatCredentials?.providerId || 'openai';
    const chatModel = chatCredentials?.modelId || 'gpt-4o';
    const chatApiKey = chatCredentials?.apiKey || '';
    const chatProviderFull = getProviderById?.(chatProvider);
    const visualCredentials = getVisualCredentials?.() || null;
    const useDedicatedVisualModel = visualCredentials?.source === 'dedicated';
    const visualProvider = useDedicatedVisualModel ? visualCredentials?.providerId || '' : '';
    const visualModel = useDedicatedVisualModel ? visualCredentials?.modelId || '' : '';
    const visualApiKey = visualCredentials?.apiKey || '';
    const visualProviderFull = useDedicatedVisualModel ? getProviderById?.(visualProvider) : null;

    if (getChatCredentials && !chatApiKey && chatProvider !== 'local' && chatProvider !== 'ollama') {
      setOverviewError(`请先为 ${chatProviderFull?.name || chatProvider} 配置 API Key`);
      return;
    }

    if (force) {
      overviewCacheRef.current.delete(overviewKey);
      currentOverviewKeyRef.current = '';
    }
    setOverviewLoading(true);
    setOverviewError(null);

    const requestPromise = (async () => {
      const params = new URLSearchParams({ depth });
      if (force) {
        params.set('force', 'true');
      }
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
      if (visualCredentials) {
        headers['X-ChatPDF-Visual-Strategy'] = visualCredentials.strategy || 'balanced';
        headers['X-ChatPDF-Visual-Enabled'] = String(visualCredentials.isVisionCapable === true);
        if (useDedicatedVisualModel) {
          headers['X-ChatPDF-Visual-Provider'] = visualProvider;
          headers['X-ChatPDF-Visual-Model'] = visualModel;
          if (visualApiKey) {
            headers['X-ChatPDF-Visual-Api-Key'] = visualApiKey;
          }
          if (visualProviderFull?.apiHost) {
            headers['X-ChatPDF-Visual-Api-Host'] = visualProviderFull.apiHost;
          }
        }
        if (visualCredentials.local?.providerId && visualCredentials.local?.modelId) {
          headers['X-ChatPDF-Local-Visual-Provider'] = visualCredentials.local.providerId;
          headers['X-ChatPDF-Local-Visual-Model'] = visualCredentials.local.modelId;
          if (visualCredentials.local.apiKey) {
            headers['X-ChatPDF-Local-Visual-Api-Key'] = visualCredentials.local.apiKey;
          }
          if (visualCredentials.local.apiHost) {
            headers['X-ChatPDF-Local-Visual-Api-Host'] = visualCredentials.local.apiHost;
          }
        }
      }

      const res = await fetch(`${API_BASE_URL}/documents/${requestDocId}/overview?${params}`, {
        headers,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({ detail: '获取速览失败' }));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      if (isCurrentOverviewRequest()) {
        const generationStatus = data?.ai_meta?.generation_status
          || (data?.ai_meta?.fallback ? 'fallback' : 'completed');
        const cacheable = generationStatus === 'completed' && !data?.ai_meta?.fallback;
        if (cacheable) {
          overviewCacheRef.current.set(overviewKey, data);
          currentOverviewKeyRef.current = overviewKey;
        } else {
          overviewCacheRef.current.delete(overviewKey);
          currentOverviewKeyRef.current = '';
        }
        setOverview(data);
      }
      return data;
    })();

    overviewInflightRef.current.set(overviewKey, {
      epoch: requestEpoch,
      promise: requestPromise,
    });

    try {
      return await requestPromise;
    } catch (err) {
      console.error('获取速览失败:', err);
      if (isCurrentOverviewRequest()) {
        setOverviewError(err.message || '获取速览失败');
      }
      throw err;
    } finally {
      const activeEntry = overviewInflightRef.current.get(overviewKey);
      if (activeEntry?.promise === requestPromise) {
        overviewInflightRef.current.delete(overviewKey);
      }
      if (isCurrentOverviewRequest()) {
        setOverviewLoading(false);
      }
    }
  }, [
    buildOverviewKey,
    docId,
    docInfo,
    getChatCredentials,
    getVisualCredentials,
    getProviderById,
    overview,
    overviewParseIdentity,
    overviewTextIdentity,
    overviewVisualIdentity,
  ]);

  useEffect(() => {
    const nextContext = {
      docId: docId || '',
      parseIdentity: overviewParseIdentity,
      textIdentity: overviewTextIdentity,
      visualIdentity: overviewVisualIdentity,
    };
    const previousContext = overviewObservedContextRef.current;
    overviewObservedContextRef.current = nextContext;
    if (!previousContext) return;

    const changedDocument = previousContext.docId !== nextContext.docId;
    const changedParseGeneration = (
      previousContext.docId === nextContext.docId
      && previousContext.parseIdentity !== nextContext.parseIdentity
    );
    const changedVisualModel = (
      previousContext.docId === nextContext.docId
      && previousContext.parseIdentity === nextContext.parseIdentity
      && previousContext.textIdentity === nextContext.textIdentity
      && previousContext.visualIdentity !== nextContext.visualIdentity
    );
    const changedTextModel = (
      previousContext.docId === nextContext.docId
      && previousContext.parseIdentity === nextContext.parseIdentity
      && previousContext.textIdentity !== nextContext.textIdentity
    );
    if (!changedDocument && !changedParseGeneration && !changedTextModel && !changedVisualModel) return;

    invalidateOverviewEpoch(previousContext.docId || nextContext.docId, {
      clearDocumentCache: changedParseGeneration,
    });
  }, [
    docId,
    invalidateOverviewEpoch,
    overviewParseIdentity,
    overviewTextIdentity,
    overviewVisualIdentity,
  ]);

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
    uploadFileInfo,

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
    clearOverviewCache,

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
