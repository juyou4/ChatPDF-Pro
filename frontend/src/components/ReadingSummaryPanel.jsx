import React, { memo, useMemo } from 'react';
import { AlertCircle, CheckCircle2, Loader2, RefreshCw } from 'lucide-react';

const AtomIcon = ({ className = '' }) => (
  <svg
    className={className}
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <circle cx="12" cy="12" r="1" />
    <path d="M20.2 20.2c2.04-2.03.02-7.36-4.5-11.9-4.54-4.52-9.87-6.54-11.9-4.5-2.04 2.03-.02 7.36 4.5 11.9 4.54 4.52 9.87 6.54 11.9 4.5Z" />
    <path d="M15.7 15.7c4.52-4.54 6.54-9.87 4.5-11.9-2.03-2.04-7.36-.02-11.9 4.5-4.52 4.54-6.54 9.87-4.5 11.9 2.03 2.04 7.36.02 11.9-4.5Z" />
  </svg>
);

const getSourceMeta = (source) => {
  if (source === 'ai') return { label: 'AI 结构化', tone: 'ai' };
  if (source === 'fallback' || source === 'client_fallback') return { label: '基础兜底', tone: 'fallback' };
  if (source) return { label: '基础目录', tone: 'fallback' };
  return { label: '待生成', tone: 'muted' };
};

function ReadingSummaryPanel({
  items = [],
  loading = false,
  error = '',
  activeNodeId = null,
  visitedNodeIds = [],
  onJump,
  onRetry,
  source = '',
  retrying = false,
  darkMode = false,
}) {
  const topItems = useMemo(() => (Array.isArray(items) ? items.filter((item) => item?.title) : []), [items]);
  const visitedSet = useMemo(() => new Set(visitedNodeIds || []), [visitedNodeIds]);
  const isFallback = source && source !== 'ai';
  const sourceMeta = getSourceMeta(source);

  if (loading) {
    return (
      <div className={`h-full flex items-center justify-center text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
        生成总结中
      </div>
    );
  }

  if (error && topItems.length === 0) {
    return (
      <div className={`px-4 py-3 text-sm rounded-xl ${darkMode ? 'bg-red-500/10 text-red-300' : 'bg-red-50 text-red-600'}`}>
        {error}
      </div>
    );
  }

  if (topItems.length === 0) {
    return (
      <div className={`h-full flex flex-col items-center justify-center text-center px-8 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
        <AtomIcon className="w-8 h-8 mb-3 opacity-60" />
        <div className="text-sm font-medium">暂无 AI 总结</div>
      </div>
    );
  }

  const pageChip = (page) => {
    if (!page) return null;
    return (
      <span className={`ml-2 shrink-0 rounded-md px-1.5 py-0.5 text-[10px] font-semibold tabular-nums ${
        darkMode ? 'text-gray-400 bg-white/[0.07]' : 'text-gray-400 bg-gray-100/90'
      }`}>
        P{page}
      </span>
    );
  };

  // 二级及更深节点：轻卡片，hover 微浮起
  const renderChild = (item, depth = 0) => {
    const isActive = item.id === activeNodeId;
    const isVisited = !isActive && visitedSet.has(item.id);
    const children = Array.isArray(item.children) ? item.children.filter((child) => child?.title) : [];
    return (
      <div key={item.id || item.title} className={depth > 0 ? 'pl-3' : ''}>
        <button
          type="button"
          onClick={() => onJump?.(item)}
          className={`group w-full rounded-xl border p-3 text-left transition-all duration-200 hover:-translate-y-[1px] ${
            isActive
              ? (darkMode
                ? 'border-white/10 bg-white/10 backdrop-blur-md shadow-[0_10px_25px_rgba(0,0,0,0.2)]'
                : 'border-white/80 bg-white/90 backdrop-blur-md shadow-[0_10px_25px_rgba(0,0,0,0.06)]')
              : isVisited
                ? (darkMode
                  ? 'border-amber-400/20 bg-amber-400/[0.05] hover:shadow-[0_10px_24px_-18px_rgba(0,0,0,0.6)]'
                  : 'border-amber-200/70 bg-amber-50/50 hover:shadow-[0_10px_24px_-18px_rgba(180,120,20,0.25)]')
                : (darkMode
                  ? 'border-white/[0.08] bg-white/[0.04] hover:border-white/[0.14] hover:bg-white/[0.06] hover:shadow-[0_10px_24px_-18px_rgba(0,0,0,0.6)]'
                  : 'border-gray-200/70 bg-white/85 shadow-[0_1px_2px_rgba(15,23,42,0.04)] hover:border-gray-300/80 hover:bg-white hover:shadow-[0_10px_24px_-14px_rgba(15,23,42,0.18)]')
          }`}
        >
          <div className="flex items-start">
            <span className="flex min-w-0 flex-1 items-start gap-1.5">
              {isActive && <CheckCircle2 className="mt-[3px] h-3.5 w-3.5 shrink-0 text-[#9333ea]" strokeWidth={2.5} />}
              {isVisited && <span className="mt-[6px] h-[7px] w-[7px] shrink-0 rounded-full bg-amber-400" />}
              <span className={`min-w-0 text-[12.5px] font-semibold leading-snug ${darkMode ? 'text-gray-100' : 'text-gray-800'}`}>
                {item.title}
              </span>
            </span>
            {pageChip(item.page)}
          </div>
          {item.summary && (
            <p className={`mt-1.5 text-[12px] leading-relaxed ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              {item.summary}
            </p>
          )}
        </button>
        {children.length > 0 && (
          <div className={`mt-1.5 space-y-1.5 border-l pl-2 ${darkMode ? 'border-white/[0.08]' : 'border-gray-200/80'}`}>
            {children.map((child) => renderChild(child, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  // 顶层节点：分区头 + 摘要 + 子卡片
  const renderSection = (item) => {
    const isActive = item.id === activeNodeId;
    const isVisited = !isActive && visitedSet.has(item.id);
    const children = Array.isArray(item.children) ? item.children.filter((child) => child?.title) : [];
    return (
      <section key={item.id || item.title} className="relative pl-3.5">
        <span
          className={`absolute left-0 top-1 bottom-1 w-[3px] rounded-full ${darkMode ? 'bg-white/[0.12]' : 'bg-gray-200'}`}
          aria-hidden="true"
        />
        <button
          type="button"
          onClick={() => onJump?.(item)}
          className={`group w-full text-left transition-all duration-300 ${
            isActive
              ? (darkMode
                ? 'rounded-2xl p-3 bg-white/10 backdrop-blur-md border border-white/10 shadow-[0_10px_25px_rgba(0,0,0,0.2)]'
                : 'rounded-2xl p-3 bg-white/85 backdrop-blur-md border border-white/80 shadow-[0_10px_25px_rgba(0,0,0,0.06)]')
              : ''
          }`}
        >
          <div className="flex items-start">
            <span className="flex min-w-0 flex-1 items-center gap-1.5">
              {isActive ? (
                <CheckCircle2 className="h-4 w-4 shrink-0 text-[#9333ea]" strokeWidth={2.5} />
              ) : isVisited ? (
                <span className="h-[8px] w-[8px] shrink-0 rounded-full bg-amber-400" />
              ) : (
                <AtomIcon className={`h-3.5 w-3.5 shrink-0 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`} />
              )}
              <span className={`min-w-0 text-[13.5px] font-bold leading-snug transition-colors ${
                darkMode ? 'text-gray-50 group-hover:text-white' : 'text-gray-900 group-hover:text-black'
              }`}>
                {item.title}
              </span>
            </span>
            {pageChip(item.page)}
          </div>
          {item.summary && (
            <p className={`mt-1.5 text-[12px] leading-relaxed ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              {item.summary}
            </p>
          )}
        </button>
        {children.length > 0 && (
          <div className="mt-2.5 space-y-1.5">
            {children.map((child) => renderChild(child))}
          </div>
        )}
      </section>
    );
  };

  return (
    <div className="h-full overflow-y-auto pr-1 pl-0.5">
      <div className="space-y-5 pb-4">
        <div className="flex items-center justify-between gap-2">
          <span className={`inline-flex items-center rounded-full border px-2.5 py-1 text-[11px] font-semibold ${
            sourceMeta.tone === 'ai'
              ? (darkMode ? 'border-white/10 bg-white/[0.06] text-gray-200' : 'border-slate-200 bg-white/85 text-slate-600')
              : sourceMeta.tone === 'fallback'
                ? (darkMode ? 'border-amber-400/20 bg-amber-400/10 text-amber-100' : 'border-amber-100 bg-amber-50 text-amber-700')
                : (darkMode ? 'border-white/10 bg-white/[0.04] text-gray-500' : 'border-gray-200 bg-gray-50 text-gray-500')
          }`}>
            {sourceMeta.label}
          </span>
        </div>
        {error && (
          <div className={`rounded-2xl border px-3.5 py-3 text-[12px] ${
            darkMode ? 'border-red-400/30 bg-red-400/10 text-red-200' : 'border-red-200 bg-red-50 text-red-700'
          }`}>
            {error}
          </div>
        )}
        {isFallback && (
          <div className={`rounded-2xl border px-3.5 py-3 text-[12px] leading-relaxed ${
            darkMode
              ? 'border-amber-400/30 bg-amber-400/10 text-amber-100'
              : 'border-amber-200 bg-amber-50 text-amber-900'
          }`}>
            <div className="flex items-start gap-2">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              <div className="min-w-0 flex-1">
                <div className="font-semibold">AI 总结未生成，当前展示基础目录。</div>
                <div className={`mt-1 ${darkMode ? 'text-amber-100/70' : 'text-amber-800/75'}`}>
                  配置模型后可重新生成更完整的中文结构化总结。
                </div>
              </div>
            </div>
            {onRetry && (
              <button
                type="button"
                onClick={onRetry}
                disabled={retrying}
                className={`mt-2 inline-flex items-center gap-1.5 rounded-xl px-2.5 py-1.5 text-[12px] font-semibold transition-all disabled:opacity-60 ${
                  darkMode
                    ? 'bg-white/10 hover:bg-white/15 text-amber-50'
                    : 'bg-white/80 hover:bg-white text-amber-900 shadow-sm'
                }`}
              >
                <RefreshCw className={`h-3.5 w-3.5 ${retrying ? 'animate-spin' : ''}`} />
                重新生成
              </button>
            )}
          </div>
        )}
        {topItems.map((item) => renderSection(item))}
      </div>
    </div>
  );
}

export default memo(ReadingSummaryPanel);
