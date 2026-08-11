import React from 'react';

/**
 * WebCitationLink — 联网搜索来源引用徽章
 *
 * 渲染为轻量上标编号，直接打开对应的外部来源。
 */
const WebCitationLink = React.memo(({ refNumber, source }) => {
  const descriptionId = React.useId();

  if (!source) {
    return <span className="text-gray-400 text-xs">[{refNumber}]</span>;
  }

  const { title, url } = source;
  const displayTitle = title || url || `来源 ${refNumber}`;
  const hostname = url ? (() => { try { return new URL(url).hostname; } catch { return url; } })() : '';

  const description = [displayTitle, hostname].filter(Boolean).join('，');

  return (
    <span className="inline-block">
      <a
        href={url || undefined}
        target="_blank"
        rel="noreferrer"
        title={displayTitle}
        aria-describedby={descriptionId}
        className="mx-0.5 inline-flex min-w-[0.9em] items-center justify-center align-super text-[0.74em] font-semibold leading-none tabular-nums text-[#9A4B36] transition-colors duration-150 hover:text-[#743725] focus-visible:rounded-[3px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#E99A7D]/80 dark:text-[#F0A58A] dark:hover:text-[#FFD0BE]"
      >
        {refNumber}
      </a>
      <span id={descriptionId} className="sr-only">{description}</span>
    </span>
  );
});

WebCitationLink.displayName = 'WebCitationLink';

export default WebCitationLink;
