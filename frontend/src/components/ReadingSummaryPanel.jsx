import React, { memo, useEffect, useMemo, useState } from 'react';
import { AlertCircle, CheckCircle2, ChevronDown, RefreshCw } from 'lucide-react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import BreatheLoader from './BreatheLoader';

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
  if (source === 'ai') return { label: 'AI 快速精读', tone: 'ai' };
  if (source === 'fallback' || source === 'client_fallback') return { label: '章节结构导览', tone: 'fallback' };
  if (source) return { label: '基础目录', tone: 'fallback' };
  return { label: '待生成', tone: 'muted' };
};

const normalizeGenerationError = (value) => {
  const text = String(value || '').trim();
  if (!text) return '';
  if (/JSONDecodeError|Expecting\s+(?:value|.*delimiter)|line\s+\d+\s+column\s+\d+/i.test(text)) {
    return '模型返回的结构化结果格式不完整，已保留基础结果';
  }
  if (/low-quality reading outline|章节数字缺少对应证据/i.test(text)) {
    return '个别章节未通过证据校验，已保留基础章节提要';
  }
  return text.length > 180 ? `${text.slice(0, 180)}...` : text;
};

const normalizeReadingSummary = (value) => String(value || '')
  .replace(/^(?:当前解析结果未提取到可供总结的正文。|该小节未提取到独立正文。)\s*/, '')
  .trim();

const splitFindingLead = (value) => {
  const text = String(value || '').trim();
  const match = text.match(/^(.{2,18}?[：:])\s*(.+)$/);
  return match ? { lead: match[1], body: match[2] } : { lead: '', body: text };
};

const formatEvidencePages = (item, maxSegments = 3) => {
  const exactPages = Array.isArray(item?.evidence?.pages)
    ? [...new Set(item.evidence.pages
      .map((value) => Number(value))
      .filter((value) => Number.isInteger(value) && value > 0))]
      .sort((left, right) => left - right)
    : [];
  if (exactPages.length === 0) {
    const start = Number(item?.page_start || item?.page || 0);
    const end = Number(item?.page_end || start);
    if (!Number.isInteger(start) || start <= 0) return null;
    const label = Number.isInteger(end) && end > start ? `P${start}-${end}` : `P${start}`;
    return { label, fullLabel: label };
  }

  const ranges = [];
  let rangeStart = exactPages[0];
  let previous = exactPages[0];
  for (const page of exactPages.slice(1)) {
    if (page === previous + 1) {
      previous = page;
      continue;
    }
    ranges.push([rangeStart, previous]);
    rangeStart = page;
    previous = page;
  }
  ranges.push([rangeStart, previous]);
  const rangeLabels = ranges.map(([start, end]) => (start === end ? `P${start}` : `P${start}-${end}`));
  const fullLabel = rangeLabels.join(', ');
  const omitted = Math.max(0, rangeLabels.length - maxSegments);
  return {
    fullLabel,
    label: omitted > 0
      ? `${rangeLabels.slice(0, maxSegments).join(', ')} +${omitted}`
      : fullLabel,
  };
};

const nodeHasExpandableDetail = (item) => {
  const study = item?.study || {};
  return Boolean(
    study.purpose
    || study.method_or_argument
    || study.caveats_or_connections
    || study.findings?.length
    || study.learning_path?.length
    || item?.children?.length
  );
};

const findNodePath = (items, targetId, path = []) => {
  for (const item of items || []) {
    const nextPath = [...path, item.id];
    if (item.id === targetId) return nextPath;
    const childPath = findNodePath(item.children, targetId, nextPath);
    if (childPath.length) return childPath;
  }
  return [];
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
  generationError = '',
  retrying = false,
  meta = {},
  darkMode = false,
}) {
  const topItems = useMemo(() => (Array.isArray(items) ? items.filter((item) => item?.title) : []), [items]);
  const visitedSet = useMemo(() => new Set(visitedNodeIds || []), [visitedNodeIds]);
  const [expandedIds, setExpandedIds] = useState(() => new Set());
  const reduceMotion = useReducedMotion();
  const isFallback = source === 'fallback' || source === 'client_fallback';
  const sourceMeta = getSourceMeta(source);
  const displayGenerationError = normalizeGenerationError(generationError);
  const coverage = meta?.section_coverage || {};
  const generation = meta?.generation || {};
  const bodyMissing = Math.max(0, Number(coverage.body_expected || 0) - Number(coverage.body_summarized || 0));
  const appendixMissing = Math.max(0, Number(coverage.appendix_expected || 0) - Number(coverage.appendix_summarized || 0));
  const reportedFallbackSections = Math.max(0, Number(generation.fallback_sections || 0));
  const incompleteSectionCount = Math.max(reportedFallbackSections, bodyMissing + appendixMissing);
  const hasPartialGeneration = source === 'ai' && incompleteSectionCount > 0;
  const samplingBySection = useMemo(() => new Map(
    (Array.isArray(meta?.sampling?.sections) ? meta.sampling.sections : [])
      .filter((item) => item?.source_section_id)
      .map((item) => [String(item.source_section_id), item])
  ), [meta?.sampling?.sections]);

  useEffect(() => {
    const overview = topItems.find((item) => item.type === 'overview');
    const defaultId = overview?.id || topItems[0]?.id;
    setExpandedIds(new Set(defaultId ? [defaultId] : []));
  }, [topItems]);

  useEffect(() => {
    if (!activeNodeId) return;
    const path = findNodePath(topItems, activeNodeId);
    if (!path.length) return;
    const ancestorIds = path.slice(0, -1);
    if (!ancestorIds.length) return;
    setExpandedIds((previous) => new Set([...previous, ...ancestorIds]));
  }, [activeNodeId, topItems]);

  if (loading) {
    return (
      <div
        className={`flex h-full flex-col items-center justify-center gap-3 text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}
        role="status"
        aria-live="polite"
      >
        <BreatheLoader className={darkMode ? 'text-[#FFA07A]' : 'text-[#D97A5D]'} />
        <span>生成快速精读中</span>
      </div>
    );
  }

  if (error && topItems.length === 0) {
    return (
      <div className={`rounded-xl px-4 py-3 text-sm ${darkMode ? 'bg-red-500/10 text-red-300' : 'bg-red-50 text-red-600'}`}>
        {error}
      </div>
    );
  }

  if (topItems.length === 0) {
    return (
      <div className={`flex h-full flex-col items-center justify-center px-8 text-center ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
        <AtomIcon className="mb-3 h-8 w-8 opacity-60" />
        <div className="text-sm font-medium">暂无 AI 总结</div>
      </div>
    );
  }

  // 自由展开：各章节独立开合，不做手风琴互斥
  const toggleExpanded = (event, itemId) => {
    event.stopPropagation();
    setExpandedIds((previous) => {
      const next = new Set(previous);
      if (next.has(itemId)) next.delete(itemId);
      else next.add(itemId);
      return next;
    });
  };

  const pageChip = (item) => {
    if (item?.type === 'overview') return null;
    const pages = formatEvidencePages(item);
    if (!pages) return null;
    return (
      <span
        className={`max-w-[108px] shrink-0 truncate rounded-md px-1.5 py-0.5 text-[10px] font-semibold tabular-nums ${
          darkMode ? 'bg-white/[0.07] text-gray-400' : 'bg-gray-100/90 text-gray-400'
        }`}
        title={pages.fullLabel}
      >
        {pages.label}
      </span>
    );
  };

  const renderStudy = (item) => {
    const study = item.study || {};
    const rows = [
      ['问题', study.purpose],
      ['方法 / 论证', study.method_or_argument],
      ['边界 / 联系', study.caveats_or_connections],
    ].filter(([, value]) => value);
    const findings = Array.isArray(study.findings) ? study.findings.filter(Boolean) : [];
    const learningPath = Array.isArray(study.learning_path) ? study.learning_path.filter(Boolean) : [];
    if (!rows.length && !findings.length && !learningPath.length) return null;
    const findingsLabel = item.type === 'overview' ? '关键认识' : '关键结论';

    return (
      <div className="mt-2.5 pl-0.5">
        {findings.length > 0 && (
          <div>
            <div className={`text-[10.5px] font-semibold leading-none ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>{findingsLabel}</div>
            <ul className={`mt-2 space-y-2.5 text-[12px] leading-[1.7] ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
              {findings.map((finding, index) => {
                const { lead, body } = splitFindingLead(finding);
                return (
                  <li key={`${item.id}-finding-${index}`} className="flex items-start gap-2">
                    <span
                      aria-hidden="true"
                      className={`mt-[9px] h-1 w-1 shrink-0 rounded-full ${darkMode ? 'bg-[#E9A28C]/70' : 'bg-[#D97A5D]/70'}`}
                    />
                    <span className="min-w-0 flex-1 text-pretty">
                      {lead && (
                        <span className={darkMode ? 'font-medium text-gray-100' : 'font-medium text-gray-800'}>
                          {lead}{' '}
                        </span>
                      )}
                      {body}
                    </span>
                  </li>
                );
              })}
            </ul>
          </div>
        )}
        {rows.length > 0 && (
          <dl className={`${findings.length ? 'mt-3' : ''} space-y-3`}>
            {rows.map(([label, value]) => (
              <div key={label}>
                <dt className={`text-[10.5px] font-semibold leading-none ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>{label}</dt>
                <dd className={`mt-1 text-pretty text-[12px] leading-[1.65] ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>{value}</dd>
              </div>
            ))}
          </dl>
        )}
        {learningPath.length > 0 && (
          <div className="mt-3">
            <div className={`text-[10.5px] font-semibold leading-none ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>阅读顺序</div>
            <ol className={`mt-2 space-y-1.5 text-[12px] leading-[1.65] ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
              {learningPath.map((step, index) => (
                <li key={`${item.id}-path-${index}`} className="flex items-start gap-2.5">
                  <span className="w-3.5 shrink-0 font-semibold tabular-nums text-[#D97A5D]">{index + 1}</span>
                  <span className="min-w-0 flex-1 text-pretty">{step}</span>
                </li>
              ))}
            </ol>
          </div>
        )}
      </div>
    );
  };

  const renderNode = (item, depth = 0) => {
    const isActive = item.id === activeNodeId;
    const isVisited = !isActive && visitedSet.has(item.id);
    const children = Array.isArray(item.children) ? item.children.filter((child) => child?.title) : [];
    const expanded = expandedIds.has(item.id);
    const hasDetail = nodeHasExpandableDetail(item);
    const isTop = depth === 0;
    const displaySummary = normalizeReadingSummary(item.summary);
    const parsedKeywordConfidence = Number(item.keyword_confidence);
    const hasKeywordConfidence = item.keyword_confidence !== undefined
      && item.keyword_confidence !== null
      && Number.isFinite(parsedKeywordConfidence);
    const keywords = isFallback
      && (!hasKeywordConfidence || parsedKeywordConfidence >= 0.55)
      && Array.isArray(item.keywords)
      ? item.keywords.filter(Boolean).slice(0, 4)
      : [];
    const sampling = samplingBySection.get(String(item.source_section_id || ''));
    const showSamplingNotice = source === 'ai' && sampling?.review_recommended;
    const study = item.study || {};
    const hasStudyDetail = Boolean(
      study.purpose
      || study.method_or_argument
      || study.caveats_or_connections
      || study.findings?.length
      || study.learning_path?.length
    );
    const previewSummary = expanded && hasStudyDetail ? '' : displaySummary;
    const hasPreview = showSamplingNotice || keywords.length > 0 || Boolean(previewSummary);

    return (
      <section key={item.id || item.title}>
        {/* 顶层：白卡浮在浅灰底上做分隔；激活态用描边+标题变色（不再用图标） */}
        <motion.div
          className={`group relative transition-colors duration-200 motion-reduce:transition-none ${
            isTop
              ? `rounded-[14px] px-3 py-2.5 ${
                isActive
                  ? (darkMode ? 'bg-white/[0.08]' : 'bg-white shadow-[0_6px_18px_rgba(30,35,45,0.09)]')
                  : (darkMode ? 'bg-white/[0.04] hover:bg-white/[0.06]' : 'bg-white/85 shadow-[0_1px_3px_rgba(30,35,45,0.05)] hover:bg-white')
              }`
              : `rounded-xl px-2 py-2 ${
                isActive
                  ? (darkMode ? 'bg-white/[0.08]' : 'bg-white/90 shadow-[0_5px_16px_rgba(30,35,45,0.06)]')
                  : expanded
                    ? (darkMode ? 'bg-white/[0.035]' : 'bg-[#D97A5D]/[0.025]')
                    : (darkMode ? 'hover:bg-white/[0.04]' : 'hover:bg-white/65')
              }`
          }`}
          whileHover={!reduceMotion && isTop && !expanded ? { y: -1 } : undefined}
          transition={reduceMotion
            ? { duration: 0 }
            : { type: 'spring', stiffness: 420, damping: 34, mass: 0.55 }}
        >
          <AnimatePresence initial={false}>
            {isTop && expanded && hasDetail && (
              <motion.span
                key={`${item.id}-expanded-accent`}
                aria-hidden="true"
                className={`pointer-events-none absolute bottom-3 left-0 top-3 w-0.5 origin-center rounded-r-full ${
                  darkMode ? 'bg-[#E9A28C]/70' : 'bg-[#D97A5D]/75'
                }`}
                initial={reduceMotion ? false : { opacity: 0, scaleY: 0.35 }}
                animate={{ opacity: 1, scaleY: 1 }}
                exit={reduceMotion ? { opacity: 0 } : { opacity: 0, scaleY: 0.35 }}
                transition={reduceMotion ? { duration: 0 } : { duration: 0.18, ease: [0.22, 1, 0.36, 1] }}
              />
            )}
          </AnimatePresence>
          <div className="flex items-start gap-1.5">
            <button
              type="button"
              onClick={() => onJump?.(item)}
              className="flex min-w-0 flex-1 items-start gap-2 rounded-md text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/30"
            >
              {/* 顶层不放前导图标，标题直接起头；子级保留小灰点表结构 */}
              {!isTop && (
                isActive ? (
                  <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-[#f85d25]" strokeWidth={2.5} />
                ) : (
                  <span className={`mt-[6px] h-1.5 w-1.5 shrink-0 rounded-full ${
                    isVisited
                      ? (darkMode ? 'bg-gray-500' : 'bg-gray-400')
                      : (darkMode ? 'bg-gray-600' : 'bg-gray-300')
                  }`} />
                )
              )}
              <span className={`block text-pretty font-semibold leading-snug ${
                isTop ? 'text-[13px]' : 'text-[12px]'
              } ${
                isTop && isActive
                  ? (darkMode ? 'text-[#E9A28C]' : 'text-[#B95F43]')
                  : (darkMode ? 'text-gray-100' : 'text-gray-800')
              }`}>
                {item.title}
              </span>
            </button>
            <div className="flex shrink-0 items-center gap-0.5">
              {pageChip(item)}
              {hasDetail && (
                <button
                  type="button"
                  onClick={(event) => toggleExpanded(event, item.id)}
                  className={`rounded-lg p-1.5 transition-[color,background-color,transform] duration-200 active:scale-[0.96] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/30 ${
                    expanded
                      ? (darkMode ? 'text-[#E9A28C]' : 'text-[#B95F43]')
                      : (darkMode ? 'text-gray-500 hover:bg-white/[0.08] hover:text-gray-200' : 'text-gray-400 hover:bg-gray-100 hover:text-gray-700')
                  }`}
                  aria-expanded={expanded}
                  aria-label={`${expanded ? '收起' : '展开'} ${item.title}`}
                  title={expanded ? '收起' : '展开'}
                >
                  <motion.span
                    className="flex"
                    initial={false}
                    animate={{ rotate: expanded ? 180 : 0 }}
                    transition={reduceMotion
                      ? { duration: 0 }
                      : { type: 'spring', stiffness: 430, damping: 30, mass: 0.45 }}
                  >
                    <ChevronDown className="h-3.5 w-3.5" />
                  </motion.span>
                </button>
              )}
            </div>
          </div>

          {hasPreview && (
            <button
              type="button"
              onClick={() => onJump?.(item)}
              className={`mt-1 block w-full rounded-md pr-1 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/30 ${isTop ? 'pl-0.5' : 'pl-6'}`}
            >
              {showSamplingNotice && (
                <span
                  className={`mt-1 inline-flex rounded-md px-1.5 py-0.5 text-[9.5px] font-medium tabular-nums ${
                    darkMode ? 'bg-amber-400/10 text-amber-200/80' : 'bg-amber-50 text-amber-700'
                  }`}
                  title={`本节共 ${sampling.total_blocks} 个内容块，选取 ${sampling.selected_blocks} 个代表块生成精简摘要`}
                >
                  长章节采样 {sampling.block_coverage_pct}%
                </span>
              )}
              {keywords.length > 0 && (
                <span className="mt-1.5 flex flex-wrap gap-1">
                  {keywords.map((keyword) => (
                    <span
                      key={`${item.id}-${keyword}`}
                      className={`max-w-full truncate rounded-md px-1.5 py-0.5 text-[9.5px] font-medium ${
                        darkMode ? 'bg-white/[0.06] text-gray-400' : 'bg-gray-100/80 text-gray-500'
                      }`}
                    >
                      {keyword}
                    </span>
                  ))}
                </span>
              )}
              {previewSummary && (
                <span
                  className={`mt-1 block text-pretty text-[12px] font-normal leading-[1.65] ${
                    hasDetail && !expanded ? 'line-clamp-3' : ''
                  } ${
                  darkMode ? 'text-gray-300' : 'text-gray-600'
                  }`}
                  title={hasDetail && !expanded ? previewSummary : undefined}
                >
                  {previewSummary}
                </span>
              )}
            </button>
          )}
        </motion.div>

        {/* 详情按需挂载，只动画合成层，避免长章节展开时逐帧重排高度。 */}
        {hasDetail && expanded && (
          <motion.div
            className={`${isTop ? 'pb-2.5 pl-3.5 pr-3' : 'pb-2 pl-7 pr-2'}`}
            initial={reduceMotion ? false : { opacity: 0, y: -3 }}
            animate={{ opacity: 1, y: 0 }}
            transition={reduceMotion ? { duration: 0 } : { duration: 0.12, ease: [0.22, 1, 0.36, 1] }}
          >
            {renderStudy(item)}
            {children.length > 0 && (
              <div className={`mt-3 space-y-1 border-l pl-2 ${darkMode ? 'border-white/[0.08]' : 'border-gray-200/90'}`}>
                {children.map((child) => renderNode(child, depth + 1))}
              </div>
            )}
          </motion.div>
        )}
      </section>
    );
  };

  return (
    <div className="custom-scrollbar h-full overflow-y-auto pl-0.5 pr-1">
      <div className="space-y-3 pb-4">
        <div className="flex flex-wrap items-center justify-between gap-2 px-0.5">
          <span className={`inline-flex items-center rounded-full border px-2.5 py-1 text-[11px] font-semibold ${
            sourceMeta.tone === 'ai'
              ? (darkMode ? 'border-white/10 bg-white/[0.06] text-gray-200' : 'border-slate-200 bg-white/85 text-slate-600')
              : sourceMeta.tone === 'fallback'
                ? (darkMode ? 'border-amber-400/20 bg-amber-400/10 text-amber-100' : 'border-amber-100 bg-amber-50 text-amber-700')
                : (darkMode ? 'border-white/10 bg-white/[0.04] text-gray-500' : 'border-gray-200 bg-gray-50 text-gray-500')
          }`}>
            {sourceMeta.label}
          </span>
          {coverage.body_expected > 0 && (
            <span className={`text-[10.5px] font-medium tabular-nums ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
              {isFallback ? `正文 ${coverage.body_expected} 章` : `正文 ${coverage.body_summarized}/${coverage.body_expected}`}
              {coverage.appendix_expected > 0
                ? (isFallback
                  ? ` · 附录 ${coverage.appendix_expected} 章`
                  : ` · 附录 ${coverage.appendix_summarized}/${coverage.appendix_expected}`)
                : ''}
            </span>
          )}
        </div>

        {hasPartialGeneration && (
          <div className={`flex items-center justify-between gap-2 rounded-xl px-3 py-2 text-[11px] ${
            darkMode ? 'bg-amber-400/[0.08] text-amber-100/80' : 'bg-amber-50/80 text-amber-800'
          }`}>
            <span>{incompleteSectionCount} 个章节未生成完整 AI 提要，已保留章节定位。</span>
            {onRetry && (
              <button
                type="button"
                onClick={onRetry}
                disabled={retrying}
                className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-lg transition-colors disabled:opacity-60 ${
                  darkMode ? 'hover:bg-white/10 hover:text-amber-50' : 'hover:bg-amber-100/80 hover:text-amber-950'
                }`}
                aria-label="重新生成 AI 快速精读"
                title="重新生成"
              >
                <RefreshCw className={`h-3.5 w-3.5 ${retrying ? 'animate-spin' : ''}`} />
              </button>
            )}
          </div>
        )}
        {error && (
          <div className={`rounded-xl border px-3.5 py-3 text-[12px] ${
            darkMode ? 'border-red-400/30 bg-red-400/10 text-red-200' : 'border-red-200 bg-red-50 text-red-700'
          }`}>
            {error}
          </div>
        )}
        {isFallback && (
          <div className={`rounded-xl border px-3.5 py-3 text-[12px] leading-relaxed ${
            darkMode
              ? 'border-amber-400/30 bg-amber-400/10 text-amber-100'
              : 'border-amber-200 bg-amber-50 text-amber-900'
          }`}>
            <div className="flex items-start gap-2">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              <div className="min-w-0 flex-1">
                <div className="font-semibold">AI 快速精读未生成，当前显示章节结构导览。</div>
                <div className={`mt-1 ${darkMode ? 'text-amber-100/70' : 'text-amber-800/75'}`}>
                  {displayGenerationError
                    ? `模型生成失败：${displayGenerationError}`
                    : '配置 API Key 后可生成简洁的中文快速精读。'}
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
                    ? 'bg-white/10 text-amber-50 hover:bg-white/15'
                    : 'bg-white/80 text-amber-900 shadow-sm hover:bg-white'
                }`}
              >
                <RefreshCw className={`h-3.5 w-3.5 ${retrying ? 'animate-spin' : ''}`} />
                生成 AI 精读
              </button>
            )}
          </div>
        )}
        <div className="space-y-2">
          {topItems.map((item) => renderNode(item))}
        </div>
      </div>
    </div>
  );
}

export default memo(ReadingSummaryPanel);
