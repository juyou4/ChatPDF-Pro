import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { Expand, ImageOff, LocateFixed, RefreshCw, X } from 'lucide-react';
import {
  buildChatVisualAttachmentUrl,
  normalizeChatVisualAttachment,
  normalizeChatVisualAttachments,
} from '../utils/visualAttachmentUtils';

const DocumentFigure = ({ attachment, docId, darkMode = false, onLocate }) => {
  const normalized = useMemo(
    () => normalizeChatVisualAttachment(attachment),
    [attachment],
  );
  const [imageUrl, setImageUrl] = useState('');
  const [loadState, setLoadState] = useState('loading');
  const [reloadKey, setReloadKey] = useState(0);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    if (!normalized || !docId) {
      setLoadState('error');
      return () => {};
    }
    const endpoint = buildChatVisualAttachmentUrl(docId, normalized);
    if (!endpoint) {
      setLoadState('error');
      return () => {};
    }
    const controller = new AbortController();
    let objectUrl = '';
    setImageUrl('');
    setLoadState('loading');
    fetch(endpoint, { signal: controller.signal, cache: 'force-cache' })
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const contentType = String(response.headers.get('content-type') || '').toLowerCase();
        if (!contentType.startsWith('image/jpeg')) throw new Error('invalid image content type');
        return response.blob();
      })
      .then((blob) => {
        if (controller.signal.aborted) return;
        objectUrl = URL.createObjectURL(blob);
        setImageUrl(objectUrl);
        setLoadState('ready');
      })
      .catch((error) => {
        if (error?.name !== 'AbortError') setLoadState('error');
      });
    return () => {
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [docId, normalized, reloadKey]);

  const locate = useCallback(() => {
    if (!normalized || typeof onLocate !== 'function') return;
    onLocate({
      asset_id: normalized.asset_id,
      figure_id: normalized.figure_id,
      page: normalized.page,
      page_range: [normalized.page, normalized.page],
      bbox: normalized.bbox,
      figure_bbox: normalized.bbox,
      coordinate_space: normalized.coordinate_space,
      parse_generation: normalized.parse_generation,
      highlight_text: normalized.caption,
    });
  }, [normalized, onLocate]);

  if (!normalized) return null;
  const aspectRatio = normalized.width && normalized.height
    ? `${normalized.width} / ${normalized.height}`
    : '16 / 10';
  const statusLabel = normalized.evidence_mode === 'vlm_verified' ? '视觉核验' : '解析图表';

  const lightbox = expanded && imageUrl && typeof document !== 'undefined'
    ? createPortal(
      <div
        className="fixed inset-0 z-[120] flex items-center justify-center bg-black/70 p-8 backdrop-blur-sm"
        role="dialog"
        aria-modal="true"
        aria-label={normalized.caption || normalized.label}
        onClick={() => setExpanded(false)}
      >
        <button
          type="button"
          className="absolute right-6 top-6 grid h-10 w-10 place-items-center rounded-lg bg-white/90 text-gray-700 shadow-sm transition-colors hover:bg-white"
          onClick={() => setExpanded(false)}
          aria-label="关闭大图"
          title="关闭"
        >
          <X className="h-5 w-5" />
        </button>
        <img
          src={imageUrl}
          alt={normalized.caption || normalized.label}
          className="max-h-full max-w-full object-contain"
          onClick={(event) => event.stopPropagation()}
        />
      </div>,
      document.body,
    )
    : null;

  return (
    <figure className={`not-prose mt-3 overflow-hidden rounded-lg border ${darkMode ? 'border-white/10 bg-[#252525]' : 'border-[#e9e5df] bg-white'}`}>
      <div
        className={`relative flex w-full items-center justify-center overflow-hidden ${darkMode ? 'bg-black/20' : 'bg-[#f7f7f6]'}`}
        style={{ aspectRatio, maxHeight: '430px', minHeight: '150px' }}
      >
        {loadState === 'ready' && imageUrl ? (
          <button
            type="button"
            className="h-full w-full cursor-zoom-in"
            onClick={() => setExpanded(true)}
            aria-label="放大查看文献图表"
            title="放大查看"
          >
            <img
              src={imageUrl}
              alt={normalized.caption || normalized.label}
              className="h-full w-full object-contain"
            />
          </button>
        ) : loadState === 'error' ? (
          <div className={`flex flex-col items-center gap-2 text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            <ImageOff className="h-5 w-5" />
            <span>图表暂时无法加载</span>
            <button
              type="button"
              className={`grid h-8 w-8 place-items-center rounded-lg transition-colors ${darkMode ? 'hover:bg-white/10' : 'hover:bg-black/5'}`}
              onClick={() => setReloadKey((value) => value + 1)}
              aria-label="重新加载图表"
              title="重新加载"
            >
              <RefreshCw className="h-4 w-4" />
            </button>
          </div>
        ) : (
          <div className={`h-8 w-8 animate-pulse rounded-lg ${darkMode ? 'bg-white/10' : 'bg-black/5'}`} aria-label="正在加载图表" />
        )}
        {loadState === 'ready' && (
          <div className="absolute right-2 top-2 flex gap-1">
            <button
              type="button"
              className="grid h-8 w-8 place-items-center rounded-lg border border-black/5 bg-white/90 text-gray-700 shadow-sm transition-colors hover:bg-white"
              onClick={() => setExpanded(true)}
              aria-label="放大查看文献图表"
              title="放大查看"
            >
              <Expand className="h-4 w-4" />
            </button>
            <button
              type="button"
              className="grid h-8 w-8 place-items-center rounded-lg border border-black/5 bg-white/90 text-gray-700 shadow-sm transition-colors hover:bg-white"
              onClick={locate}
              aria-label={`定位到第 ${normalized.page} 页`}
              title={`定位到第 ${normalized.page} 页`}
            >
              <LocateFixed className="h-4 w-4" />
            </button>
          </div>
        )}
      </div>
      <figcaption className="flex items-start justify-between gap-4 px-3.5 py-3">
        <div className="min-w-0">
          <div className={`text-[11px] font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            {normalized.label} · 第 {normalized.page} 页 · {statusLabel}
          </div>
          {normalized.caption && (
            <p className={`mt-1 text-[13px] leading-5 ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
              {normalized.caption}
            </p>
          )}
        </div>
        <button
          type="button"
          className={`grid h-8 w-8 shrink-0 place-items-center rounded-lg transition-colors ${darkMode ? 'text-gray-300 hover:bg-white/10' : 'text-gray-600 hover:bg-black/5'}`}
          onClick={locate}
          aria-label={`在 PDF 中定位第 ${normalized.page} 页`}
          title="在 PDF 中定位"
        >
          <LocateFixed className="h-4 w-4" />
        </button>
      </figcaption>
      {lightbox}
    </figure>
  );
};

export const DocumentVisualAttachments = ({ attachments, docId, darkMode = false, onLocate }) => {
  const normalized = useMemo(
    () => normalizeChatVisualAttachments(attachments),
    [attachments],
  );
  if (!normalized.length) return null;
  return (
    <div className="mt-3 space-y-3" data-testid="document-visual-attachments">
      {normalized.map((attachment) => (
        <DocumentFigure
          key={attachment.attachment_id}
          attachment={attachment}
          docId={docId}
          darkMode={darkMode}
          onLocate={onLocate}
        />
      ))}
    </div>
  );
};

export default DocumentFigure;
