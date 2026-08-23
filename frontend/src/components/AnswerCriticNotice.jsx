import React, { useId, useState } from 'react';
import { ChevronDown, ChevronUp, LocateFixed, Quote, TriangleAlert } from 'lucide-react';
import {
  formatEvidenceRefs,
  getCriticConfidenceTier,
  getCriticIssueTypeMeta,
  sanitizeCriticText,
  shouldShowCriticConfidence,
} from '../utils/answerCriticUtils';

const NOTICE_TONE = {
  title: 'text-[#3f3a35] dark:text-gray-100',
  body: 'text-[#4a453f] dark:text-gray-300',
  muted: 'text-[#8a827b] dark:text-gray-400',
  rule: 'border-[#e6e1db] dark:border-white/10',
  focus: 'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#cfc6bd]/70 dark:focus-visible:ring-white/20',
};

const CRITIC_NOTICE_VARIANTS = {
  hallucination: {
    title: '答案需核对',
    description: '部分内容可能超出当前证据，请谨慎参考。',
    icon: TriangleAlert,
  },
  citation: {
    title: '引用待补充',
    description: '部分事实陈述还缺少引用编号。',
    fullSummaryDescription: '有章节结论还没绑到阅读证据。',
    icon: Quote,
  },
  overreach: {
    title: '结论过强',
    description: '部分表述超出当前证据支持范围。',
    icon: TriangleAlert,
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
  const NoticeIcon = typeof style.icon === 'function' ? style.icon : Quote;
  const visibleLines = showAllLines ? lines : lines.slice(0, CRITIC_VISIBLE_DETAIL_LINES);
  const collapsedCount = Math.max(0, lines.length - CRITIC_VISIBLE_DETAIL_LINES);
  const hasDetails = lines.length > 0 || showSuggestion;
  const issueCount = lines.length + (showSuggestion ? 1 : 0);
  const headerDescription = variant === 'citation' && critic.answer_mode === 'full_document_summary'
    ? (style.fullSummaryDescription || style.description)
    : style.description;

  const toggleDetails = () => {
    if (!hasDetails) return;
    if (detailsOpen) setShowAllLines(false);
    setDetailsOpen((current) => !current);
  };

  return (
    <aside
      aria-label={style.title}
      data-variant={variant}
      className="text-[12.5px]"
    >
      <button
        type="button"
        onClick={toggleDetails}
        disabled={!hasDetails}
        aria-expanded={hasDetails ? detailsOpen : undefined}
        aria-controls={hasDetails ? detailsId : undefined}
        className={`-ml-1 flex w-full items-start gap-2 rounded-[8px] px-1 py-1 text-left transition-colors duration-200 hover:bg-[#f6f3f1] disabled:cursor-default dark:hover:bg-white/[0.03] ${NOTICE_TONE.focus}`}
      >
        <NoticeIcon
          size={14}
          strokeWidth={1.8}
          className={`mt-[3px] shrink-0 ${NOTICE_TONE.muted}`}
          aria-hidden="true"
        />
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
            <h4 className={`text-[13px] font-medium leading-5 ${NOTICE_TONE.title}`}>
              {style.title}
            </h4>
            {issueCount > 0 && (
              <span className={`text-[11px] tabular-nums ${NOTICE_TONE.muted}`}>
                {issueCount} 项
              </span>
            )}
            {showConfidence && (
              <span className={`text-[11px] tabular-nums ${NOTICE_TONE.muted}`}>
                核对 {(critic.confidence * 100).toFixed(0)}% · {tier.label}
              </span>
            )}
          </span>
          <span className={`mt-0.5 block text-[12px] leading-5 ${NOTICE_TONE.muted}`}>
            {headerDescription}
          </span>
        </span>
        {hasDetails && (
          <ChevronDown
            size={14}
            strokeWidth={1.8}
            className={`mt-1 shrink-0 transition-transform duration-300 ${detailsOpen ? 'rotate-180' : ''} ${NOTICE_TONE.muted}`}
            aria-hidden="true"
          />
        )}
      </button>

      {hasDetails && detailsOpen && (
        <div
          id={detailsId}
          className="grid grid-rows-[1fr] opacity-100 transition-[grid-template-rows,opacity] duration-250 ease-[cubic-bezier(0.22,1,0.36,1)] motion-reduce:transition-none"
        >
          <div className="min-h-0 overflow-hidden">
            <div className={`ml-[18px] border-l pb-1 pl-3.5 pt-2 ${NOTICE_TONE.rule}`}>
              {visibleLines.length > 0 && (
                <ul className={`space-y-2.5 leading-6 ${NOTICE_TONE.body}`}>
                  {visibleLines.map((line) => {
                    const typeMeta = line.issueType ? getCriticIssueTypeMeta(line.issueType) : null;
                    const evidenceRefs = formatEvidenceRefs(line.evidenceRefs);
                    return (
                      <li key={line.text} className="min-w-0">
                        {typeMeta && (
                          <span className={`mr-1.5 text-[11.5px] font-medium ${NOTICE_TONE.title}`}>
                            {typeMeta.label}
                          </span>
                        )}
                        <span className="line-clamp-3" title={line.text}>{line.text}</span>
                        {(line.claimSpan || evidenceRefs) && (
                          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1">
                            {evidenceRefs && (
                              <span className={`text-[11.5px] tabular-nums ${NOTICE_TONE.muted}`}>
                                应对 {evidenceRefs}
                              </span>
                            )}
                            {line.claimSpan && (
                              <button
                                type="button"
                                onClick={() => onLocateClaim && onLocateClaim(line.claimSpan)}
                                disabled={!onLocateClaim}
                                title={line.claimSpan}
                                className={`inline-flex items-center gap-1 text-[11.5px] transition-colors duration-200 hover:text-[#3f3a35] active:translate-y-px disabled:cursor-default disabled:opacity-[0.55] dark:hover:text-gray-100 ${NOTICE_TONE.muted} ${NOTICE_TONE.focus}`}
                              >
                                <LocateFixed size={12} strokeWidth={1.8} aria-hidden="true" />
                                定位原句
                              </button>
                            )}
                          </div>
                        )}
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
                  className={`mt-2 inline-flex items-center gap-1 text-[11.5px] transition-colors duration-200 hover:text-[#3f3a35] active:translate-y-px dark:hover:text-gray-100 ${NOTICE_TONE.muted} ${NOTICE_TONE.focus}`}
                >
                  {showAllLines ? '收起问题' : `查看其余 ${collapsedCount} 项`}
                  {showAllLines
                    ? <ChevronUp size={12} strokeWidth={1.8} aria-hidden="true" />
                    : <ChevronDown size={12} strokeWidth={1.8} aria-hidden="true" />}
                </button>
              )}
              {showSuggestion && (
                <div className={`mt-2.5 flex gap-2 border-t pt-2.5 leading-6 ${NOTICE_TONE.body} ${NOTICE_TONE.rule}`}>
                  <span className={`shrink-0 text-[11px] ${NOTICE_TONE.muted}`}>建议</span>
                  <span className="line-clamp-4" title={suggestion}>{suggestion}</span>
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
