import React, { useState, useCallback, useEffect, useRef } from 'react';
import { ChevronDown, ChevronRight, FileText, ExternalLink, Image } from 'lucide-react';

/**
 * 证据面板：展示每个检索到的文档块
 * 被引用的证据排在前面（默认展开），未被引用的折叠
 *
 * Props:
 *   citations: Array - 引文列表 [{ref, group_id, page_range, highlight_text}, ...]
 *   docId: string - 文档 ID，用于按需加载页面缩略图
 *   onCitationClick: Function - 点击引文跳转 PDF
 *   activeRef: number|null - 当前高亮的引文编号（双向联动）
 *   onRefHover: Function - 鼠标悬停引文编号时的回调
 */
export default function EvidencePanel({ citations, docId, onCitationClick, activeRef, onRefHover }) {
  const [expandedRefs, setExpandedRefs] = useState(new Set());
  const [panelCollapsed, setPanelCollapsed] = useState(false);
  // thumbnail cache: page -> {url, loading, error}
  const [thumbnails, setThumbnails] = useState({});
  const fetchingRef = useRef(new Set());

  const toggleRef = useCallback((ref) => {
    setExpandedRefs(prev => {
      const next = new Set(prev);
      if (next.has(ref)) next.delete(ref);
      else next.add(ref);
      return next;
    });
  }, []);

  // Fetch thumbnail for a given page (lazy, cached)
  const fetchThumbnail = useCallback((page) => {
    if (!docId || !page || thumbnails[page] || fetchingRef.current.has(page)) return;
    fetchingRef.current.add(page);
    setThumbnails(prev => ({ ...prev, [page]: { loading: true } }));
    fetch(`/api/document/${docId}/thumbnail/${page}`)
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then(data => {
        setThumbnails(prev => ({ ...prev, [page]: { url: data.thumbnail, loading: false } }));
      })
      .catch(() => {
        setThumbnails(prev => ({ ...prev, [page]: { error: true, loading: false } }));
        fetchingRef.current.delete(page);
      });
  }, [docId, thumbnails]);

  if (!citations || citations.length === 0) return null;

  // 分组：有 highlight_text 的排前面
  const cited = citations.filter(c => c.highlight_text);
  const uncited = citations.filter(c => !c.highlight_text);

  const renderCitation = (c, defaultOpen) => {
    const ref = c.ref;
    const isExpanded = defaultOpen ? !expandedRefs.has(ref) : expandedRefs.has(ref);
    const isActive = activeRef === ref;
    const pageLabel = c.page_range
      ? c.page_range[0] === c.page_range[1]
        ? `P${c.page_range[0]}`
        : `P${c.page_range[0]}-${c.page_range[1]}`
      : '';
    // Use the first page of the citation range for thumbnail
    const thumbPage = c.page_range?.[0];
    const thumb = thumbPage ? thumbnails[thumbPage] : null;

    return (
      <div
        key={ref}
        className={`border rounded-lg transition-all duration-200 ${
          isActive
            ? 'border-blue-400 bg-blue-50/50 ring-1 ring-blue-200'
            : 'border-gray-200 hover:border-gray-300'
        }`}
        onMouseEnter={() => onRefHover?.(ref)}
        onMouseLeave={() => onRefHover?.(null)}
      >
        {/* 头部 */}
        <button
          onClick={() => {
            toggleRef(ref);
            // Lazy-load thumbnail when expanding
            if (!isExpanded && thumbPage) fetchThumbnail(thumbPage);
          }}
          className="w-full flex items-center gap-2 px-3 py-2 text-left text-xs"
        >
          {isExpanded
            ? <ChevronDown className="w-3.5 h-3.5 text-gray-400 flex-shrink-0" />
            : <ChevronRight className="w-3.5 h-3.5 text-gray-400 flex-shrink-0" />
          }
          <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-blue-100 text-blue-700 text-[10px] font-bold flex-shrink-0">
            {ref}
          </span>
          <span className="text-gray-500 truncate flex-1">
            {c.group_id || `来源 ${ref}`}
          </span>
          {pageLabel && (
            <span className="text-[10px] text-gray-400 flex-shrink-0">{pageLabel}</span>
          )}
          {c.highlight_text && (
            <FileText className="w-3 h-3 text-green-500 flex-shrink-0" title="已匹配到原文" />
          )}
        </button>

        {/* 展开内容 */}
        {isExpanded && (
          <div className="px-3 pb-2.5 pt-0">
            {/* 页面缩略图 */}
            {thumbPage && docId && (
              <div className="mb-2">
                {thumb?.loading && (
                  <div className="w-full h-20 bg-gray-100 rounded animate-pulse flex items-center justify-center">
                    <Image className="w-4 h-4 text-gray-300" />
                  </div>
                )}
                {thumb?.url && (
                  <img
                    src={thumb.url}
                    alt={`页面 ${thumbPage} 预览`}
                    className="w-full rounded border border-gray-100 cursor-pointer hover:opacity-90 transition-opacity"
                    style={{ maxHeight: '120px', objectFit: 'contain', objectPosition: 'top' }}
                    onClick={(e) => { e.stopPropagation(); onCitationClick?.(c); }}
                    title={`点击跳转到 ${pageLabel}`}
                  />
                )}
              </div>
            )}
            {c.highlight_text ? (
              <div className="text-xs text-gray-700 leading-relaxed bg-yellow-50/60 border border-yellow-100 rounded px-2.5 py-2">
                <mark className="bg-yellow-200/70 rounded px-0.5">{c.highlight_text}</mark>
              </div>
            ) : (
              <div className="text-xs text-gray-400 italic">未匹配到精确引文</div>
            )}
            {c.page_range && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onCitationClick?.(c);
                }}
                className="mt-1.5 inline-flex items-center gap-1 text-[10px] text-blue-500 hover:text-blue-700 transition-colors"
              >
                <ExternalLink className="w-3 h-3" />
                跳转到 {pageLabel}
              </button>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="mt-2 ml-2">
      <button
        onClick={() => setPanelCollapsed(prev => !prev)}
        className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-700 transition-colors mb-1.5"
      >
        {panelCollapsed
          ? <ChevronRight className="w-3.5 h-3.5" />
          : <ChevronDown className="w-3.5 h-3.5" />
        }
        <span className="font-medium">引用来源</span>
        <span className="text-gray-400">({citations.length})</span>
      </button>
      {!panelCollapsed && (
        <div className="flex flex-col gap-1.5 max-w-lg">
          {cited.map(c => renderCitation(c, true))}
          {uncited.map(c => renderCitation(c, false))}
        </div>
      )}
    </div>
  );
}
