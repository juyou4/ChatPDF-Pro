import React, { memo, useMemo } from 'react';
import { BookOpen } from 'lucide-react';
import { LayoutGroup, motion } from 'framer-motion';
import BreatheLoader from './BreatheLoader';

const getOutlineSourceMeta = (source) => {
  if (source === 'ai') return { label: 'AI 章节树', tone: 'ai' };
  if (source === 'toc') return { label: 'PDF 书签', tone: 'toc' };
  if (source === 'mineru') return { label: 'MinerU 章节结构', tone: 'toc' };
  if (source === 'heuristic') return { label: '启发式大纲', tone: 'fallback' };
  return { label: '基础大纲', tone: 'muted' };
};

// 同一页常有「章 + 若干节」。旧逻辑把它们全部做成选中卡，叠在一起又吵又冗余。
// 这里只挑一条定位点：点过的章节优先，否则当前页最浅的标题，再否则上一节。
export function resolvePrimaryOutlineItem(items, { activeNodeId, activeBlockId, currentPage } = {}) {
  if (!Array.isArray(items) || items.length === 0) return null;

  if (activeNodeId) {
    const exact = items.find((item) => item.id === activeNodeId);
    if (exact) return exact;
  }

  if (activeBlockId) {
    const matches = items.filter((item) => (
      item.first_block === activeBlockId
      || (Array.isArray(item.evidenceIds) && item.evidenceIds.includes(activeBlockId))
    ));
    if (matches.length === 1) return matches[0];
    if (matches.length > 1) {
      return matches.reduce((best, item) => (
        Number(item.level) > Number(best.level) ? item : best
      ));
    }
  }

  const page = Math.max(1, Number(currentPage) || 1);
  const onPage = items.filter((item) => Number(item.page) === page);
  if (onPage.length > 0) {
    return onPage.reduce((best, item) => (
      Number(item.level) < Number(best.level) ? item : best
    ));
  }

  for (let index = items.length - 1; index >= 0; index -= 1) {
    if (Number(items[index].page) < page) return items[index];
  }
  return null;
}

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
  const primaryItem = useMemo(
    () => resolvePrimaryOutlineItem(items, { activeNodeId, activeBlockId, currentPage }),
    [activeBlockId, activeNodeId, currentPage, items]
  );
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
      return `text-[13px] font-semibold ${darkMode ? 'text-gray-100' : 'text-[#2c2723]'}`;
    }
    if (level <= 1) {
      return `text-[13px] font-semibold ${darkMode ? 'text-gray-200 group-hover:text-white' : 'text-[#3f3a35] group-hover:text-[#2c2723]'}`;
    }
    if (level === 2) {
      return `text-[12.5px] font-medium ${darkMode ? 'text-gray-400 group-hover:text-gray-200' : 'text-[#5c564f] group-hover:text-[#2c2723]'}`;
    }
    return `text-[12px] ${darkMode ? 'text-gray-500 group-hover:text-gray-300' : 'text-[#8a7f74] group-hover:text-[#2c2723]'}`;
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
        <LayoutGroup id="pdf-outline-active">
          {items.map((item) => {
            const isActive = primaryItem?.key === item.key;
            const isVisited = !isActive && visitedSet.has(item.id);
            const guides = Math.max(0, item.level - 1);
            return (
              <button
                key={item.key}
                type="button"
                onClick={() => onJump?.(item)}
                title={item.title}
                aria-current={isActive ? 'true' : undefined}
                className={`group relative isolate flex w-full items-stretch rounded-[14px] pr-2 text-left transition-colors duration-200 ${
                  isActive
                    ? 'z-10'
                    : (darkMode ? 'hover:bg-white/[0.05]' : 'hover:bg-white/55')
                }`}
              >
                {isActive && (
                  <motion.div
                    layoutId="pdf-outline-active-card"
                    aria-hidden="true"
                    className={`absolute inset-0 z-0 rounded-[14px] ${
                      darkMode
                        ? 'bg-white/10 ring-1 ring-inset ring-white/[0.08] shadow-[0_4px_12px_rgba(0,0,0,0.16)]'
                        : 'bg-white ring-1 ring-inset ring-black/[0.025] shadow-[0_4px_12px_rgba(31,41,55,0.08),0_1px_2px_rgba(31,41,55,0.04)]'
                    }`}
                    transition={{ type: 'tween', duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
                  />
                )}
                <span className="relative z-10 w-2.5 shrink-0" aria-hidden="true" />
                {Array.from({ length: guides }).map((_, i) => (
                  <span
                    key={i}
                    aria-hidden="true"
                    className={`relative z-10 w-3.5 shrink-0 self-stretch ${
                      isActive ? '' : `border-l ${darkMode ? 'border-white/[0.08]' : 'border-[#e8e4df]'}`
                    }`}
                  />
                ))}
                <span className="relative z-10 flex min-w-0 flex-1 items-center gap-2 py-[7px]">
                  <span className="flex w-3.5 shrink-0 items-center justify-center" aria-hidden="true">
                    <span
                      className={`rounded-full ${
                        isActive
                          ? `h-[6px] w-[6px] ${darkMode ? 'bg-gray-100' : 'bg-[#1a1a1a]'}`
                          : isVisited
                            ? `h-[6px] w-[6px] ${darkMode ? 'bg-gray-500' : 'bg-[#b8aea4]'}`
                            : item.level <= 1
                              ? `h-[6px] w-[6px] ${darkMode ? 'bg-gray-500' : 'bg-[#c4bbb3]'}`
                              : `h-[4px] w-[4px] ${darkMode ? 'bg-gray-600' : 'bg-[#d8d0c8]'}`
                      }`}
                    />
                  </span>
                  <span className={`min-w-0 flex-1 leading-snug line-clamp-2 ${titleClassByLevel(item.level, isActive)}`}>
                    {item.title}
                  </span>
                  <span
                    className={`ml-1 shrink-0 text-[10px] font-medium tabular-nums transition-colors ${
                      isActive
                        ? (darkMode ? 'text-gray-400' : 'text-[#8a7f74]')
                        : (darkMode ? 'text-gray-600 group-hover:text-gray-400' : 'text-[#c4bbb3] group-hover:text-[#8a7f74]')
                    }`}
                  >
                    {item.page}
                  </span>
                </span>
              </button>
            );
          })}
        </LayoutGroup>
      </div>
    </div>
  );
}

export default memo(DocumentOutline);
