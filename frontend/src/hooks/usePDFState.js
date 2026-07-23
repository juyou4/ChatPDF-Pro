import { useState, useCallback, useEffect, useRef } from 'react';
import { useDebouncedValue } from './useDebouncedValue';

// API base URL
const API_BASE_URL = '';

const idleSearchStatus = () => ({
  state: 'idle',
  message: '',
  errorCode: '',
  fallbackReason: '',
  resultCount: 0,
});

const getParseIdentityPart = (documentIdentity, key) => {
  const prefix = `${key}=`;
  const rawPart = String(documentIdentity || '')
    .split(':')
    .find((part) => part.startsWith(prefix));
  if (!rawPart) return '';
  const rawValue = rawPart.slice(prefix.length);
  try {
    const value = decodeURIComponent(rawValue).trim();
    return value === 'legacy' ? '' : value;
  } catch {
    return rawValue.trim() === 'legacy' ? '' : rawValue.trim();
  }
};

const getSearchErrorMessage = (payload, fallback) => {
  const detail = payload?.detail;
  const rawMessage = typeof detail === 'string'
    ? detail
    : (typeof payload?.error === 'string' ? payload.error : fallback);
  const message = String(rawMessage || fallback || '文档检索失败').trim();
  return message.length > 180 ? `${message.slice(0, 180)}...` : message;
};

const isKeylessLocalProvider = (providerId) => ['local', 'ollama'].includes(
  String(providerId || '').trim().toLowerCase()
);

/**
 * PDF 查看器状态管理 Hook
 * 管理 PDF 页码、缩放、搜索、高亮、文本选择等状态
 *
 * 将 PDF 相关状态从 ChatPDF 主组件中提取出来，
 * 使 PDF 状态变更仅触发 PDF 查看器区域的重渲染。
 *
 * @param {Object} options - 配置选项
 * @param {string|null} options.docId - 当前文档 ID
 * @param {Object|null} options.docInfo - 当前文档信息
 * @param {string} options.documentIdentity - 主解析代际标识
 * @param {string} options.parseGeneration - 当前解析 generation
 * @param {string} options.documentSourceHash - 当前文档 source hash
 * @param {boolean} options.useRerank - 是否启用重排
 * @param {string} options.rerankerModel - 重排模型名称
 * @param {Function} options.getRerankCredentials - 获取重排凭证
 * @param {Function} options.getEmbeddingConfig - 获取 embedding 配置
 * @param {string} options.embeddingApiKey - 旧签名兼容字段，不再用于搜索鉴权
 * @param {string} options.apiKey - 旧签名兼容字段，不再用于搜索鉴权
 */
export function usePDFState({
  docId = null,
  docInfo = null,
  documentIdentity = '',
  parseGeneration = '',
  documentSourceHash = '',
  useRerank = false,
  rerankerModel = 'BAAI/bge-reranker-base',
  getRerankCredentials,
  getEmbeddingConfig,
  embeddingApiKey: _embeddingApiKey = '',
  apiKey: _apiKey = '',
} = {}) {
  // ========== 页码与缩放 ==========
  const [currentPage, setCurrentPage] = useState(1);
  const [pdfScale, setPdfScale] = useState(1.0);

  // 使用防抖值实现缩放防抖（150ms），避免频繁触发 PDF 重渲染
  const debouncedScale = useDebouncedValue(pdfScale, 150);

  // ========== 文本选择 ==========
  const [selectedText, setSelectedText] = useState('');
  const [showTextMenu, setShowTextMenu] = useState(false);
  const [menuPosition, setMenuPosition] = useState({ x: 0, y: 0 });

  // ========== 搜索状态 ==========
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [currentResultIndex, setCurrentResultIndex] = useState(0);
  const [isSearching, setIsSearching] = useState(false);
  const [searchHistory, setSearchHistory] = useState([]);
  const [searchStatus, setSearchStatus] = useState(idleSearchStatus);

  // ========== 高亮状态 ==========
  const [activeHighlight, setActiveHighlight] = useState(null);

  // ========== Refs ==========
  const pdfContainerRef = useRef(null);
  const searchEpochRef = useRef(0);
  const searchAbortRef = useRef(null);

  const getSearchEmbeddingRequest = useCallback(() => {
    if (typeof getEmbeddingConfig !== 'function') {
      return {
        isValid: false,
        errorCode: 'embedding_config_unavailable',
        errorMessage: '当前页面未接入 Embedding 配置，请刷新后重试',
      };
    }

    const embeddingConfig = getEmbeddingConfig();
    if (!embeddingConfig?.isValid) {
      let message = '请先在模型设置里选择可用的 Embedding 模型';
      if (embeddingConfig?.reason === 'model_not_found') {
        message = '当前默认 Embedding 模型不存在或已下线，请重新选择后再搜索';
      } else if (embeddingConfig?.reason === 'wrong_type') {
        message = '当前默认模型不是 Embedding 类型，请切换后再搜索';
      } else if (embeddingConfig?.reason === 'provider_missing') {
        message = '当前默认 Embedding Provider 不存在，请在模型服务中重新配置';
      }
      return {
        isValid: false,
        errorCode: embeddingConfig?.reason || 'embedding_config_invalid',
        errorMessage: message,
      };
    }

    const providerId = String(embeddingConfig.providerId || '').trim();
    const provider = embeddingConfig.provider || null;
    const providerApiKey = String(provider?.apiKey || '').trim();
    if (!isKeylessLocalProvider(providerId) && !providerApiKey) {
      return {
        isValid: false,
        errorCode: 'embedding_api_key_missing',
        errorMessage: `请先为 ${provider?.name || providerId} 配置 Embedding API Key`,
      };
    }

    return {
      isValid: true,
      apiKey: providerApiKey || null,
      embeddingModel: embeddingConfig.compositeKey || '',
      embeddingProvider: providerId || null,
      embeddingApiHost: String(provider?.apiHost || '').trim() || null,
    };
  }, [getEmbeddingConfig]);

  const getSearchRerankRequest = useCallback(() => {
    if (!useRerank) {
      return { isValid: true, enabled: false };
    }
    if (typeof getRerankCredentials !== 'function') {
      return {
        isValid: false,
        enabled: true,
        errorCode: 'rerank_config_unavailable',
        errorMessage: '当前页面未接入 Rerank 配置，请刷新后重试',
      };
    }

    const rerankConfig = getRerankCredentials();
    if (!rerankConfig) {
      return {
        isValid: false,
        enabled: true,
        errorCode: 'rerank_model_missing',
        errorMessage: '请先选择可用的 Rerank 模型后再搜索',
      };
    }
    if (rerankConfig.isValid === false) {
      return {
        isValid: false,
        enabled: true,
        errorCode: rerankConfig.reason || 'rerank_config_invalid',
        errorMessage: rerankConfig.errorMessage || '当前 Rerank 配置无效，请重新检查',
      };
    }

    return {
      isValid: true,
      enabled: true,
      providerId: rerankConfig.providerId || null,
      modelId: rerankConfig.modelId || null,
      apiKey: rerankConfig.apiKey || null,
      rerankEndpoint: rerankConfig.rerankEndpoint || null,
    };
  }, [getRerankCredentials, useRerank]);

  // ========== 文档切换时重置搜索状态 ==========
  useEffect(() => {
    searchEpochRef.current += 1;
    searchAbortRef.current?.abort();
    searchAbortRef.current = null;
    setSearchQuery('');
    setSearchResults([]);
    setCurrentResultIndex(0);
    setActiveHighlight(null);
    setIsSearching(false);
    setSearchStatus(idleSearchStatus());
    if (docId) {
      const stored = JSON.parse(localStorage.getItem(`search_history_${docId}`) || '[]');
      setSearchHistory(stored);
    } else {
      setSearchHistory([]);
    }
    return () => {
      searchEpochRef.current += 1;
      searchAbortRef.current?.abort();
      searchAbortRef.current = null;
    };
  }, [docId, documentIdentity]);

  // ========== 高亮自动消失定时器 ==========
  useEffect(() => {
    if (!activeHighlight) return;
    const duration = activeHighlight.source === 'citation' ? 4000 : 2500;
    const timer = setTimeout(() => setActiveHighlight(null), duration);
    return () => clearTimeout(timer);
  }, [activeHighlight]);

  // ========== 搜索方法 ==========

  /**
   * 内部方法：聚焦到指定搜索结果
   */
  const focusResultInternal = useCallback((idx, res) => {
    if (!res || !res.length) return;
    const i = ((idx % res.length) + res.length) % res.length;
    const t = res[i];
    const p = Math.max(1, Math.min(t.page || 1, docInfo?.total_pages || 1));
    setCurrentResultIndex(i);
    setCurrentPage(p);
    setActiveHighlight({ page: p, text: t.chunk || '', at: Date.now() });
  }, [docInfo]);

  /**
   * 执行文档搜索
   * @param {string} [cq] - 可选的搜索查询，不传则使用当前 searchQuery
   */
  const handleSearch = useCallback(async (cq) => {
    if (!docId) {
      alert('请先上传文档');
      return;
    }
    const q = (cq ?? searchQuery).trim();
    if (!q) {
      searchEpochRef.current += 1;
      searchAbortRef.current?.abort();
      searchAbortRef.current = null;
      setSearchResults([]);
      setCurrentResultIndex(0);
      setActiveHighlight(null);
      setIsSearching(false);
      setSearchStatus(idleSearchStatus());
      return;
    }

    const requestEpoch = searchEpochRef.current + 1;
    searchEpochRef.current = requestEpoch;
    searchAbortRef.current?.abort();
    searchAbortRef.current = null;

    const expectedGeneration = String(
      parseGeneration || getParseIdentityPart(documentIdentity, 'g')
    ).trim();
    const expectedSourceHash = String(
      documentSourceHash || getParseIdentityPart(documentIdentity, 's')
    ).trim();

    const searchEmbeddingRequest = getSearchEmbeddingRequest();
    if (!searchEmbeddingRequest.isValid) {
      setSearchQuery(q);
      setSearchResults([]);
      setCurrentResultIndex(0);
      setActiveHighlight(null);
      setIsSearching(false);
      setSearchStatus({
        state: 'error',
        message: searchEmbeddingRequest.errorMessage,
        errorCode: searchEmbeddingRequest.errorCode,
        fallbackReason: '',
        resultCount: 0,
      });
      return;
    }

    const ctrl = new AbortController();
    searchAbortRef.current = ctrl;
    setIsSearching(true);
    setSearchQuery(q);
    setSearchResults([]);
    setCurrentResultIndex(0);
    setActiveHighlight(null);
    setSearchStatus(idleSearchStatus());
    const searchContext = {
      docId,
      documentIdentity,
      parseGeneration: expectedGeneration,
      sourceHash: expectedSourceHash,
      epoch: requestEpoch,
    };
    const isCurrentSearch = () => (
      searchEpochRef.current === searchContext.epoch
      && searchAbortRef.current === ctrl
    );

    // 获取重排凭证
    const rerankRequest = getSearchRerankRequest();
    if (!rerankRequest.isValid) {
      setSearchQuery(q);
      setSearchResults([]);
      setCurrentResultIndex(0);
      setActiveHighlight(null);
      setIsSearching(false);
      setSearchStatus({
        state: 'error',
        message: rerankRequest.errorMessage,
        errorCode: rerankRequest.errorCode,
        fallbackReason: '',
        resultCount: 0,
      });
      return;
    }

    const {
      providerId: rp,
      modelId: rm,
      apiKey: rk,
      rerankEndpoint,
    } = rerankRequest;
    let timedOut = false;
    const tid = setTimeout(() => {
      timedOut = true;
      ctrl.abort();
    }, 45000);

    try {
      const res = await fetch(`${API_BASE_URL}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: ctrl.signal,
        body: JSON.stringify({
          doc_id: docId,
          query: q,
          embedding_api_key: searchEmbeddingRequest.apiKey,
          top_k: 5,
          candidate_k: 20,
          use_rerank: useRerank,
          reranker_model: useRerank ? (rm || rerankerModel) : undefined,
          rerank_provider: useRerank ? rp : undefined,
          rerank_api_key: useRerank ? rk : undefined,
          rerank_endpoint: useRerank ? rerankEndpoint : undefined,
          embedding_model: searchEmbeddingRequest.embeddingModel,
          embedding_provider: searchEmbeddingRequest.embeddingProvider,
          embedding_api_host: searchEmbeddingRequest.embeddingApiHost,
          parse_generation: expectedGeneration && expectedSourceHash ? expectedGeneration : undefined,
          document_source_hash: expectedGeneration && expectedSourceHash ? expectedSourceHash : undefined,
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!isCurrentSearch()) return;
      if (!res.ok) {
        setSearchStatus({
          state: res.status === 409 ? 'stale' : 'error',
          message: getSearchErrorMessage(data, `文档检索失败（HTTP ${res.status}）`),
          errorCode: String(data?.error_code || `http_${res.status}`),
          fallbackReason: '',
          resultCount: 0,
        });
        return;
      }

      const responseIdentity = data?.parse_identity && typeof data.parse_identity === 'object'
        ? data.parse_identity
        : {};
      const responseGeneration = String(
        data?.parse_generation || responseIdentity.generation || ''
      ).trim();
      const responseSourceHash = String(
        data?.document_source_hash || data?.source_hash || responseIdentity.source_hash || ''
      ).trim();
      if (
        expectedGeneration
        && expectedSourceHash
        && (
          responseGeneration !== expectedGeneration
          || responseSourceHash !== expectedSourceHash
        )
      ) {
        setSearchStatus({
          state: 'stale',
          message: '文档解析已更新，已丢弃旧检索结果，请重新搜索',
          errorCode: 'parse_identity_mismatch',
          fallbackReason: '',
          resultCount: 0,
        });
        return;
      }

      const hasValidResults = Array.isArray(data?.results);
      const results = hasValidResults ? data.results : [];
      const degraded = Boolean(
        !hasValidResults || data?.degraded || data?.retrieval_degraded || data?.error
      );
      const fallbackReason = String(data?.fallback_reason || '').trim();
      setSearchResults(results);
      if (degraded) {
        const serviceMessage = getSearchErrorMessage(
          data,
          hasValidResults ? '主检索服务暂不可用' : '检索服务返回了无效响应'
        );
        setSearchStatus({
          state: 'degraded',
          message: results.length
            ? `${serviceMessage}，当前展示 ${results.length} 条可用的降级结果`
            : `${serviceMessage}；当前没有可展示结果，这不代表文档中没有匹配内容`,
          errorCode: String(data?.error_code || 'retrieval_degraded'),
          fallbackReason,
          resultCount: results.length,
        });
      } else if (results.length) {
        setSearchStatus({
          state: 'ok',
          message: '',
          errorCode: '',
          fallbackReason: '',
          resultCount: results.length,
        });
      } else {
        setSearchStatus({
          state: 'empty',
          message: '未找到匹配内容',
          errorCode: '',
          fallbackReason: '',
          resultCount: 0,
        });
        alert('未找到结果');
      }

      if (results.length) {
        // 聚焦到第一个结果（降级结果仍然可用，但由 searchStatus 明示来源）。
        focusResultInternal(0, results);
        setSearchHistory((previous) => {
          const previousItems = Array.isArray(previous) ? previous : [];
          const next = [q, ...previousItems.filter((item) => item !== q)].slice(0, 8);
          try {
            localStorage.setItem(`search_history_${docId}`, JSON.stringify(next));
          } catch {
            // 本地存储不可用不应影响搜索结果。
          }
          return next;
        });
      }
    } catch (e) {
      if (!isCurrentSearch()) return;
      setSearchResults([]);
      setCurrentResultIndex(0);
      setActiveHighlight(null);
      setSearchStatus({
        state: 'error',
        message: timedOut ? '文档检索超时，请稍后重试' : '文档检索失败，请稍后重试',
        errorCode: timedOut ? 'search_timeout' : 'search_request_failed',
        fallbackReason: '',
        resultCount: 0,
      });
    } finally {
      clearTimeout(tid);
      if (isCurrentSearch()) {
        searchAbortRef.current = null;
        setIsSearching(false);
      }
    }
  }, [
    docId,
    documentIdentity,
    documentSourceHash,
    focusResultInternal,
    getSearchEmbeddingRequest,
    getSearchRerankRequest,
    getRerankCredentials,
    parseGeneration,
    rerankerModel,
    searchQuery,
    useRerank,
  ]);

  const dismissSearchStatus = useCallback(() => {
    setSearchStatus(idleSearchStatus());
  }, []);

  /**
   * 聚焦到指定搜索结果（公开方法，默认使用当前 searchResults）
   * @param {number} idx - 结果索引
   * @param {Array} [res] - 可选的结果数组
   */
  const focusResult = useCallback((idx, res) => {
    focusResultInternal(idx, res || searchResults);
  }, [focusResultInternal, searchResults]);

  /**
   * 处理引用点击，跳转到对应页面并高亮
   * @param {Object} citation - 引用信息
   */
  const handleCitationClick = useCallback((c) => {
    if (!c || typeof c !== 'object') return;
    const rawAnchor = c.citation_anchor && typeof c.citation_anchor === 'object'
      ? c.citation_anchor
      : {};
    const rawPage = rawAnchor.page ?? c.page_range?.[0] ?? c.page;
    const targetPage = Number(rawPage);
    if (!Number.isInteger(targetPage) || targetPage <= 0) return;

    const span = rawAnchor.span && typeof rawAnchor.span === 'object'
      ? rawAnchor.span
      : (c.citation_span && typeof c.citation_span === 'object' ? c.citation_span : {});
    const citationAnchor = {
      blockId: rawAnchor.block_id || c.block_id || c.evidence_block_id || '',
      bbox: rawAnchor.bbox || c.bbox || c.figure_bbox || null,
      rects: rawAnchor.rects || c.rects || [],
      coordinateSpace: rawAnchor.coordinate_space || c.coordinate_space || '',
      pageSize: rawAnchor.page_size || c.page_size || null,
      parseGeneration: rawAnchor.parse_generation || c.parse_generation || '',
    };
    const text = c.highlight_text || span.text || c.display_text || '';
    const hasGeometry = Boolean(
      citationAnchor.blockId
      || citationAnchor.bbox
      || citationAnchor.rects?.length
    );

    setActiveHighlight(null);
    setCurrentPage(targetPage);
    if (text || hasGeometry) {
      setActiveHighlight({
        page: targetPage,
        text,
        startPhrase: c.start_phrase || span.start_phrase || '',
        endPhrase: c.end_phrase || span.end_phrase || '',
        alignmentStatus: c.alignment_status || span.alignment_status || '',
        parseGeneration: citationAnchor.parseGeneration,
        citationAnchor,
        source: 'citation',
        at: Date.now(),
      });
    }
  }, []);

  /**
   * 格式化相似度分数
   * @param {Object} r - 搜索结果项
   * @returns {number} 相似度百分比
   */
  const formatSimilarity = useCallback((r) => {
    if (r?.similarity_percent !== undefined) return r.similarity_percent;
    const s = typeof r?.score === 'number' ? r.score : 0;
    return Math.round((1 / (1 + Math.max(s, 0))) * 10000) / 100;
  }, []);

  /**
   * 渲染高亮片段
   * @param {string} snip - 文本片段
   * @param {Array} hls - 高亮区域列表
   */
  const renderHighlightedSnippet = useCallback((snip, hls = []) => {
    if (!snip) return '...';
    if (!hls.length) return snip;
    const ord = [...hls].sort((a, b) => a.start - b.start);
    const parts = [];
    let cur = 0;
    ord.forEach((h, i) => {
      const s = Math.max(0, Math.min(snip.length, h.start || 0));
      const e = Math.max(s, Math.min(snip.length, h.end || 0));
      if (s > cur) parts.push(snip.slice(cur, s));
      // 注意：此处返回纯文本标记，实际 JSX 渲染由调用方处理
      parts.push({ key: i, text: snip.slice(s, e), isHighlight: true });
      cur = e;
    });
    if (cur < snip.length) parts.push(snip.slice(cur));
    return parts;
  }, []);

  /**
   * 重置所有 PDF 状态（用于新建对话等场景）
   */
  const resetPDFState = useCallback(() => {
    searchEpochRef.current += 1;
    searchAbortRef.current?.abort();
    searchAbortRef.current = null;
    setCurrentPage(1);
    setPdfScale(1.0);
    setSelectedText('');
    setShowTextMenu(false);
    setSearchQuery('');
    setSearchResults([]);
    setCurrentResultIndex(0);
    setActiveHighlight(null);
    setIsSearching(false);
    setSearchHistory([]);
    setSearchStatus(idleSearchStatus());
  }, []);

  return {
    // 页码与缩放
    currentPage,
    setCurrentPage,
    pdfScale,
    setPdfScale,
    debouncedScale,

    // 文本选择
    selectedText,
    setSelectedText,
    showTextMenu,
    setShowTextMenu,
    menuPosition,
    setMenuPosition,

    // 搜索状态
    searchQuery,
    setSearchQuery,
    searchResults,
    setSearchResults,
    currentResultIndex,
    setCurrentResultIndex,
    isSearching,
    searchStatus,
    searchHistory,
    setSearchHistory,

    // 高亮
    activeHighlight,
    setActiveHighlight,

    // Refs
    pdfContainerRef,

    // 方法
    handleSearch,
    dismissSearchStatus,
    focusResult,
    handleCitationClick,
    formatSimilarity,
    renderHighlightedSnippet,
    resetPDFState,
  };
}
