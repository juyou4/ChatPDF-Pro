import React, { useId, useState } from 'react';
import { ChevronDown, ChevronUp, LocateFixed, Quote, TriangleAlert } from 'lucide-react';
import {
  getCriticConfidenceTier,
  getCriticIssueTypeMeta,
  sanitizeCriticText,
  shouldShowCriticConfidence,
} from '../utils/answerCriticUtils';

const CRITIC_NOTICE_VARIANTS = {
  hallucination: {
    title: '答案需核对',
    description: '自审发现部分内容可能超出当前证据，请谨慎参考。',
    icon: TriangleAlert,
    container: 'border-[#eadbd4] bg-[#fffdfc] text-[#423d3a] dark:border-[#FFA07A]/18 dark:bg-white/[0.025] dark:text-gray-200',
    summary: 'hover:bg-[#fff8f4] dark:hover:bg-[#FFA07A]/[0.045]',
    iconBox: 'border-[#f0d2c3] bg-[#fff3ed] text-[#b85f47] dark:border-[#FFA07A]/20 dark:bg-[#FFA07A]/10 dark:text-[#ffc7b4]',
    detail: 'text-[#655c57] dark:text-gray-300',
    muted: 'text-[#8c6b5c] dark:text-[#e7aa93]',
    dot: 'bg-[#cf8063]',
    divider: 'border-[#f0e3dc] dark:border-white/10',
    action: 'border-[#ead2c6] bg-[#fff8f4] text-[#a9533c] hover:border-[#dfbba9] hover:bg-[#fff0e8] dark:border-[#FFA07A]/20 dark:bg-[#FFA07A]/10 dark:text-[#ffc7b4] dark:hover:bg-[#FFA07A]/15',
  },
  citation: {
    title: '引用待补充',
    description: '部分事实陈述缺少引用编号 [n]，请结合证据面板核对。',
    icon: Quote,
    container: 'border-[#e8dfd2] bg-[#fffefb] text-[#403d38] dark:border-amber-300/15 dark:bg-white/[0.025] dark:text-gray-200',
    summary: 'hover:bg-[#fffaf1] dark:hover:bg-amber-300/[0.045]',
    iconBox: 'border-[#ecd9b9] bg-[#fff8e9] text-[#b97830] dark:border-amber-300/15 dark:bg-amber-300/10 dark:text-amber-200',
    detail: 'text-[#625d55] dark:text-gray-300',
    muted: 'text-[#8a735c] dark:text-amber-200/70',
    dot: 'bg-[#d49a55]',
    divider: 'border-[#eee5d9] dark:border-white/10',
    action: 'border-[#e6d6bd] bg-[#fffaf1] text-[#9c652c] hover:border-[#dac09a] hover:bg-[#fff4df] dark:border-amber-300/15 dark:bg-amber-300/10 dark:text-amber-200 dark:hover:bg-amber-300/15',
  },
};

const CRITIC_VISIBLE_DETAIL_LINES = 2;

/**
 * 自审提示条。默认只显示一行摘要，detailLines 由 buildCriticDetailLines 产出。
 *
 * 之前的实现分别渲染 critic.reason 和 critic.issues[0]，而后端在判定幻觉时令
 * reason = issues[0]，导致同一句话固定显示两遍，其余 issue 反而不可见。
 */
const AnswerCriticNotice = ({ critic, variant, detailLines, onLocateClaim }) => {
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [showAllLines, setShowAllLines] = useState(false);
  const detailsId = useId();
  const style = CRITIC_NOTICE_VARIANTS[variant];
  if (!style || !critic) return null;

  const lines = Array.isArray(detailLines) ? detailLines : [];
  const suggestion = sanitizeCriticText(critic.suggestion);
  // suggestion 常与 reason 同源，已在 detailLines 里出现过就不再重复渲染。
  const showSuggestion = Boolean(suggestion) && !lines.some((line) => line.text === suggestion);
  const showConfidence = shouldShowCriticConfidence(critic);
  const tier = showConfidence ? getCriticConfidenceTier(critic.confidence) : null;
  const Icon = style.icon;
  const visibleLines = showAllLines ? lines : lines.slice(0, CRITIC_VISIBLE_DETAIL_LINES);
  const collapsedCount = Math.max(0, lines.length - CRITIC_VISIBLE_DETAIL_LINES);
  const hasDetails = lines.length > 0 || showSuggestion;
  const issueCount = lines.length + (showSuggestion ? 1 : 0);

  const toggleDetails = () => {
    if (!hasDetails) return;
    if (detailsOpen) setShowAllLines(false);
    setDetailsOpen((current) => !current);
  };

  return (
    <aside
      aria-label={style.title}
      data-variant={variant}
      className={`mb-2 overflow-hidden rounded-[10px] border text-xs ${style.container}`}
    >
      <button
        type="button"
        onClick={toggleDetails}
        disabled={!hasDetails}
        aria-expanded={hasDetails ? detailsOpen : undefined}
        aria-controls={hasDetails ? detailsId : undefined}
        className={`flex min-h-9 w-full items-center gap-2 px-2.5 py-1.5 text-left transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[#FFA07A]/35 disabled:cursor-default ${style.summary}`}
      >
        <span className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-[6px] border ${style.iconBox}`}>
          <Icon size={12} strokeWidth={2} aria-hidden="true" />
        </span>
        <h4 className="shrink-0 text-[12px] font-semibold leading-5 text-[#363330] dark:text-gray-100">
          {style.title}
        </h4>
        <span className={`min-w-0 flex-1 truncate text-[11px] ${style.muted}`}>
          {style.description}
        </span>
        {issueCount > 0 && (
          <span className={`shrink-0 text-[10.5px] tabular-nums ${style.muted}`}>
            {issueCount} 项
          </span>
        )}
        {showConfidence && (
          <span className={`shrink-0 text-[10.5px] font-medium tabular-nums ${tier.className}`}>
            核对 {(critic.confidence * 100).toFixed(0)}% · {tier.label}
          </span>
        )}
        {hasDetails && (
          <ChevronDown
            size={13}
            strokeWidth={2}
            className={`shrink-0 transition-transform duration-300 ${detailsOpen ? 'rotate-180' : ''} ${style.muted}`}
            aria-hidden="true"
          />
        )}
      </button>

      {hasDetails && (
        <div
          id={detailsId}
          className={`grid transition-[grid-template-rows,opacity,visibility] duration-250 ease-[cubic-bezier(0.22,1,0.36,1)] motion-reduce:transition-none ${
            detailsOpen
              ? 'visible grid-rows-[1fr] opacity-100'
              : 'invisible pointer-events-none grid-rows-[0fr] opacity-0'
          }`}
          aria-hidden={!detailsOpen}
          inert={detailsOpen ? undefined : ''}
        >
          <div className="min-h-0 overflow-hidden">
            <div className={`mx-2.5 border-t pb-2.5 pt-2 ${style.divider}`}>
              {visibleLines.length > 0 && (
                <ul className={`space-y-1.5 leading-[1.55] ${style.detail}`}>
            {visibleLines.map((line) => {
              const typeMeta = line.issueType ? getCriticIssueTypeMeta(line.issueType) : null;
              return (
                    <li key={line.text} className="grid grid-cols-[6px_minmax(0,1fr)] gap-1.5">
                  <span className={`mt-[7px] h-1 w-1 rounded-full ${style.dot}`} aria-hidden="true" />
                  <div className="min-w-0">
                    {typeMeta && (
                      <span className={`mr-1.5 text-[10.5px] font-semibold ${typeMeta.className}`}>
                        {typeMeta.label}
                      </span>
                    )}
                    <span>{line.text}</span>
                    {line.claimSpan && (
                      <button
                        type="button"
                        onClick={() => onLocateClaim && onLocateClaim(line.claimSpan)}
                        disabled={!onLocateClaim}
                        title={line.claimSpan}
                        className={`ml-2 inline-flex items-center gap-1 rounded-[6px] border px-1.5 py-0.5 text-[10.5px] font-medium transition-[background-color,border-color,transform] duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#FFA07A]/35 active:scale-[0.98] disabled:cursor-default disabled:opacity-[0.55] ${style.action}`}
                      >
                        <LocateFixed size={11} strokeWidth={2} aria-hidden="true" />
                        定位原句
                      </button>
                    )}
                  </div>
                </li>
              );
            })}
                </ul>
              )}
              {lines.length > CRITIC_VISIBLE_DETAIL_LINES && (
          <button
            type="button"
                  onClick={() => setShowAllLines((current) => !current)}
                  aria-expanded={showAllLines}
                  className={`mt-1.5 inline-flex items-center gap-1 text-[10.5px] font-medium transition-[color,transform] duration-200 hover:text-[#8c4d2f] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#FFA07A]/35 active:translate-y-px dark:hover:text-[#ffc7b4] ${style.muted}`}
          >
                  {showAllLines ? '收起问题' : `查看其余 ${collapsedCount} 项`}
                  {showAllLines
              ? <ChevronUp size={12} strokeWidth={2} aria-hidden="true" />
              : <ChevronDown size={12} strokeWidth={2} aria-hidden="true" />}
          </button>
              )}
              {showSuggestion && (
                <div className={`mt-2 flex gap-2 border-t pt-2 leading-[1.55] ${style.detail} ${style.divider}`}>
            <span className={`shrink-0 text-[10.5px] font-semibold ${style.muted}`}>建议</span>
            <span>{suggestion}</span>
          </div>
              )}
            </div>
          </div>
        </div>
      )}
    </aside>
  );
};

export default AnswerCriticNotice;
