import React, { memo, useMemo } from 'react';
import { BookOpen, CheckCircle2 } from 'lucide-react';
import BreatheLoader from './BreatheLoader';

const getOutlineSourceMeta = (source) => {
  if (source === 'ai') return { label: 'AI 章节树', tone: 'ai' };
  if (source === 'toc') return { label: 'PDF 书签', tone: 'toc' };
  if (source === 'mineru') return { label: 'MinerU 章节结构', tone: 'toc' };
  if (source === 'heuristic') return { label: '启发式大纲', tone: 'fallback' };
  return { label: '基础大纲', tone: 'muted' };
};

function DocumentOutline({
  outline = [],
  loading = false,
  error = '',
  source = '',
  currentPage = 1,
  activeBlockId = null,
  activeNodeId = null,
  visitedNodeIds = [],
  onJump,
  darkMode = false,
}) {
  const items = useMemo(() => {
    if (!Array.isArray(outline)) return [];
    const result = [];
    const walk = (nodes, level = 1, parentKey = '') => {
      nodes.forEach((item, index) => {
        if (!item || !item.title) return;
        const nodeId = item.id || item.section_id || `${parentKey}${index}`;
        const evidenceIds = item.evidence?.block_ids || item.evidence_block_ids || (item.first_block ? [item.first_block] : []);
        result.push({
          ...item,
          id: nodeId,
          key: nodeId || `${item.page || 1}-${index}`,
          level: Math.max(1, Math.min(Number(item.level) || level, 6)),
          page: Number(item.page || item.evidence?.primary_page) || 1,
          evidenceIds,
        });
        if (Array.isArray(item.children) && item.children.length > 0) {
          walk(item.children, level + 1, `${nodeId}-`);
        }
      });
    };
    walk(outline);
    return result;
  }, [outline]);
  const visitedSet = useMemo(() => new Set(visitedNodeIds || []), [visitedNodeIds]);
  const sourceMeta = getOutlineSourceMeta(source);

  if (loading) {
    return (
      <div
        className={`flex h-full flex-col items-center justify-center gap-3 text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}
        role="status"
        aria-live="polite"
      >
        <BreatheLoader className={darkMode ? 'text-[#FFA07A]' : 'text-[#D97A5D]'} />
        <span>生成大纲中</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`px-4 py-3 text-sm rounded-xl ${darkMode ? 'bg-red-500/10 text-red-300' : 'bg-red-50 text-red-600'}`}>
        {error}
      </div>
    );
  }

  if (items.length === 0) {
    return (
      <div className={`h-full flex flex-col items-center justify-center text-center px-8 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
        <BookOpen className="w-8 h-8 mb-3 opacity-60" />
        <div className="text-sm font-medium">暂无大纲</div>
      </div>
    );
  }

  const titleClassByLevel = (level, isActive) => {
    if (isActive) {
      return `text-[13px] font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`;
    }
    if (level <= 1) {
      return `text-[13px] font-semibold ${darkMode ? 'text-gray-100 group-hover:text-white' : 'text-gray-800 group-hover:text-gray-950'}`;
    }
    if (level === 2) {
      return `text-[12.5px] font-medium ${darkMode ? 'text-gray-300 group-hover:text-gray-100' : 'text-gray-600 group-hover:text-gray-900'}`;
    }
    return `text-[12px] ${darkMode ? 'text-gray-400 group-hover:text-gray-200' : 'text-gray-500 group-hover:text-gray-800'}`;
  };

  return (
    <div className="h-full overflow-y-auto custom-scrollbar pr-1 pl-0.5">
      <div className="pb-4">
        <div className="mb-3 flex items-center justify-between gap-2">
          <span className={`inline-flex items-center rounded-full border px-2.5 py-1 text-[11px] font-semibold ${
            sourceMeta.tone === 'ai'
              ? (darkMode ? 'border-white/10 bg-white/[0.06] text-gray-200' : 'border-slate-200 bg-white/85 text-slate-600')
              : sourceMeta.tone === 'toc'
                ? (darkMode ? 'border-emerald-400/20 bg-emerald-400/10 text-emerald-100' : 'border-emerald-100 bg-emerald-50 text-emerald-700')
                : sourceMeta.tone === 'fallback'
                  ? (darkMode ? 'border-amber-400/20 bg-amber-400/10 text-amber-100' : 'border-amber-100 bg-amber-50 text-amber-700')
                  : (darkMode ? 'border-white/10 bg-white/[0.04] text-gray-500' : 'border-gray-200 bg-gray-50 text-gray-500')
          }`}>
            {sourceMeta.label}
          </span>
        </div>
        {items.map((item) => {
          const isActive = item.id === activeNodeId
            || item.first_block === activeBlockId
            || item.evidenceIds.includes(activeBlockId)
            || (!activeNodeId && !activeBlockId && item.page === currentPage);
          const isVisited = !isActive && visitedSet.has(item.id);
          const guides = Math.max(0, item.level - 1);
          return (
            <button
              key={item.key}
              type="button"
              onClick={() => onJump?.(item)}
              title={item.title}
              className={`group relative w-full flex items-stretch pr-2 text-left transition-all duration-300 ${
                isActive
                  ? (darkMode
                    ? 'my-1 rounded-2xl bg-white/10 backdrop-blur-md border border-white/10 shadow-[0_10px_25px_rgba(0,0,0,0.2)]'
                    : 'my-1 rounded-2xl bg-white/85 backdrop-blur-md border border-white/80 shadow-[0_10px_25px_rgba(0,0,0,0.06)]')
                  : isVisited
                    ? (darkMode ? 'rounded-lg bg-amber-400/[0.06]' : 'rounded-lg bg-amber-50/60')
                    : (darkMode ? 'rounded-lg hover:bg-white/[0.05]' : 'rounded-lg hover:bg-white/40')
              }`}
            >
              <span className="w-2.5 shrink-0" aria-hidden="true" />
              {Array.from({ length: guides }).map((_, i) => (
                <span
                  key={i}
                  aria-hidden="true"
                  className={`w-3.5 shrink-0 self-stretch ${
                    isActive ? '' : `border-l ${darkMode ? 'border-white/[0.08]' : 'border-gray-200/90'}`
                  }`}
                />
              ))}
              <span className="flex min-w-0 flex-1 items-center gap-2 py-[7px]">
                <span className="flex w-3.5 shrink-0 items-center justify-center" aria-hidden="true">
                  {isActive ? (
                    <CheckCircle2 className="h-3.5 w-3.5 text-[#f85d25]" strokeWidth={2.5} />
                  ) : isVisited ? (
                    <span className="h-[7px] w-[7px] rounded-full bg-amber-400" />
                  ) : (
                    <span
                      className={`rounded-full ${
                        item.level <= 1
                          ? `h-[6px] w-[6px] ${darkMode ? 'bg-gray-500' : 'bg-gray-400'}`
                          : `h-[4px] w-[4px] ${darkMode ? 'bg-gray-600' : 'bg-gray-300'}`
                      }`}
                    />
                  )}
                </span>
                <span className={`min-w-0 flex-1 leading-snug line-clamp-2 ${titleClassByLevel(item.level, isActive)}`}>
                  {item.title}
                </span>
                <span
                  className={`ml-1 shrink-0 text-[10px] font-medium tabular-nums transition-colors ${
                    isActive
                      ? (darkMode ? 'text-gray-300' : 'text-gray-500')
                      : (darkMode ? 'text-gray-600 group-hover:text-gray-400' : 'text-gray-300 group-hover:text-gray-400')
                  }`}
                >
                  {item.page}
                </span>
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export default memo(DocumentOutline);
