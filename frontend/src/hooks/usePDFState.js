import { useState, useCallback, useEffect, useRef } from 'react';
import { useDebouncedValue } from './useDebouncedValue';

// API base URL
const API_BASE_URL = '';

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
 * @param {boolean} options.useRerank - 是否启用重排
 * @param {string} options.rerankerModel - 重排模型名称
 * @param {Function} options.getRerankCredentials - 获取重排凭证
 * @param {string} options.embeddingApiKey - embedding API Key
 * @param {string} options.apiKey - 通用 API Key
 */
export function usePDFState({
  docId = null,
  docInfo = null,
  useRerank = false,
  rerankerModel = 'BAAI/bge-reranker-base',
  getRerankCredentials,
  embeddingApiKey = '',
  apiKey = '',
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

  // ========== 高亮状态 ==========
  const [activeHighlight, setActiveHighlight] = useState(null);

  // ========== Refs ==========
  const pdfContainerRef = useRef(null);

  // ========== 文档切换时重置搜索状态 ==========
  useEffect(() => {
    setSearchQuery('');
    setSearchResults([]);
    setCurrentResultIndex(0);
    setActiveHighlight(null);
    if (docId) {
      const stored = JSON.parse(localStorage.getItem(`search_history_${docId}`) || '[]');
      setSearchHistory(stored);
    } else {
      setSearchHistory([]);
    }
  }, [docId]);

  // ========== 高亮自动消失定时器 ==========
  useEffect(() => {
    if (!activeHighlight) return;
    const duration = activeHighlight.source === 'citation' ? 4000 : 2500;
    const timer = setTimeout(() => setActiveHighlight(null), duration);
    return () => clearTimeout(timer);
  }, [activeHighlight]);

  // ========== 搜索方法 ==========

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
      setSearchResults([]);
      setCurrentResultIndex(0);
      setActiveHighlight(null);
      return;
    }

    setIsSearching(true);
    setSearchQuery(q);

    // 获取重排凭证
    const { providerId: rp, modelId: rm, apiKey: rk } = getRerankCredentials?.() || {};
    const ctrl = new AbortController();
    const tid = setTimeout(() => ctrl.abort(), 45000);

    try {
      const res = await fetch(`${API_BASE_URL}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: ctrl.signal,
        body: JSON.stringify({
          doc_id: docId,
          query: q,
          api_key: embeddingApiKey || apiKey,
          top_k: 5,
          candidate_k: 20,
          use_rerank: useRerank,
          reranker_model: useRerank ? (rm || rerankerModel) : undefined,
          rerank_provider: useRerank ? rp : undefined,
          rerank_api_key: useRerank ? rk : undefined,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        const results = Array.isArray(data.results) ? data.results : [];
        setSearchResults(results);
        if (results.length) {
          // 聚焦到第一个结果
          focusResultInternal(0, results);
          // 更新搜索历史
          if (!searchHistory.includes(q)) {
            setSearchHistory(prev => [q, ...prev.filter(x => x !== q)].slice(0, 8));
          }
        } else {
          alert('未找到结果');
        }
      }
    } catch (e) {
      // 静默处理（超时或取消）
    } finally {
      clearTimeout(tid);
      setIsSearching(false);
    }
  }, [docId, searchQuery, embeddingApiKey, apiKey, useRerank, rerankerModel, getRerankCredentials, searchHistory]);

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
    if (!c?.page_range) return;
    const tp = c.page_range[0];
    if (typeof tp === 'number' && tp > 0) {
      setActiveHighlight(null);
      setCurrentPage(tp);
      if (c.highlight_text) {
        setTimeout(() => setActiveHighlight({
          page: tp,
          text: c.highlight_text,
          source: 'citation',
        }), 400);
      }
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
    searchHistory,
    setSearchHistory,

    // 高亮
    activeHighlight,
    setActiveHighlight,

    // Refs
    pdfContainerRef,

    // 方法
    handleSearch,
    focusResult,
    handleCitationClick,
    formatSimilarity,
    renderHighlightedSnippet,
    resetPDFState,
  };
}
