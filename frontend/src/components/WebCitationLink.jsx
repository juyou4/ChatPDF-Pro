import React, { useState } from 'react';
import { ExternalLink } from 'lucide-react';

/**
 * WebCitationLink — 联网搜索来源引用徽章
 *
 * 渲染为上标 [N]，悬浮时显示 Tooltip（标题 + URL + 摘要）。
 */
const WebCitationLink = React.memo(({ refNumber, source }) => {
  const [visible, setVisible] = useState(false);

  if (!source) {
    return <span className="text-gray-400 text-xs">[{refNumber}]</span>;
  }

  const { title, url, snippet } = source;
  const displayTitle = title || url || `来源 ${refNumber}`;
  const hostname = url ? (() => { try { return new URL(url).hostname; } catch { return url; } })() : '';

  return (
    <span className="relative inline-block">
      <button
        type="button"
        className="inline-flex items-center justify-center min-w-[1.5em] px-1 py-0 mx-0.5 text-xs font-semibold text-blue-600 bg-blue-50 border border-blue-200 rounded hover:bg-blue-100 hover:text-blue-700 hover:border-blue-300 cursor-pointer transition-colors duration-150 align-baseline leading-tight"
        style={{ fontSize: '0.8em', verticalAlign: 'super' }}
        onMouseEnter={() => setVisible(true)}
        onMouseLeave={() => setVisible(false)}
        onFocus={() => setVisible(true)}
        onBlur={() => setVisible(false)}
        onClick={() => url && window.open(url, '_blank', 'noopener,noreferrer')}
        title={displayTitle}
      >
        {refNumber}
      </button>
      {visible && (
        <span
          className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-1.5 w-64 rounded-lg shadow-lg bg-white border border-gray-200 p-3 text-left pointer-events-none"
          style={{ fontSize: '12px' }}
        >
          <span className="block font-semibold text-gray-800 line-clamp-2 leading-tight mb-1">
            {displayTitle}
          </span>
          {hostname && (
            <span className="flex items-center gap-0.5 text-blue-500 mb-1.5 truncate">
              <ExternalLink className="w-2.5 h-2.5 flex-shrink-0" />
              <span className="truncate text-[11px]">{hostname}</span>
            </span>
          )}
          {snippet && (
            <span className="block text-gray-500 text-[11px] leading-snug line-clamp-3">
              {snippet.length > 150 ? snippet.slice(0, 150) + '…' : snippet}
            </span>
          )}
          {/* 向下的小三角 */}
          <span className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-gray-200" />
        </span>
      )}
    </span>
  );
});

WebCitationLink.displayName = 'WebCitationLink';

export default WebCitationLink;
