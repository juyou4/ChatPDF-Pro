import React, { useState } from 'react';
import {
  getCriticConfidenceTier,
  getCriticIssueTypeMeta,
  sanitizeCriticText,
  shouldShowCriticConfidence,
} from '../utils/answerCriticUtils';

const CRITIC_NOTICE_VARIANTS = {
  hallucination: {
    title: '答案自审检测到潜在幻觉',
    container: 'bg-orange-50 border-orange-200 text-orange-700',
    detail: 'text-orange-600/80',
    muted: 'text-orange-600/70',
    withIcon: true,
  },
  citation: {
    title: '部分事实陈述缺少引用编号 [n]，请结合证据面板核对。',
    container: 'bg-amber-50 border-amber-200 text-amber-700',
    detail: 'text-amber-600/80',
    muted: 'text-amber-600/70',
    withIcon: false,
  },
};

const CRITIC_VISIBLE_DETAIL_LINES = 2;

/**
 * 自审提示框。detailLines 由 buildCriticDetailLines 产出（已去重、已清洗占位符）。
 *
 * 之前的实现分别渲染 critic.reason 和 critic.issues[0]，而后端在判定幻觉时令
 * reason = issues[0]，导致同一句话固定显示两遍，其余 issue 反而不可见。
 */
const AnswerCriticNotice = ({ critic, variant, detailLines, onLocateClaim }) => {
  const [expanded, setExpanded] = useState(false);
  const style = CRITIC_NOTICE_VARIANTS[variant];
  if (!style || !critic) return null;

  const lines = Array.isArray(detailLines) ? detailLines : [];
  const visibleLines = expanded ? lines : lines.slice(0, CRITIC_VISIBLE_DETAIL_LINES);
  const hiddenCount = lines.length - visibleLines.length;
  const suggestion = sanitizeCriticText(critic.suggestion);
  // suggestion 常与 reason 同源，已在 detailLines 里出现过就不再重复渲染。
  const showSuggestion = Boolean(suggestion) && !lines.some((line) => line.text === suggestion);
  const showConfidence = shouldShowCriticConfidence(critic);
  const tier = showConfidence ? getCriticConfidenceTier(critic.confidence) : null;

  return (
    <div className={`mb-2 px-3 py-2 rounded-lg border text-xs flex items-start gap-1.5 ${style.container}`}>
      {style.withIcon && (
        <svg className="w-4 h-4 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
        </svg>
      )}
      <div className="flex-1 min-w-0">
        <div className={variant === 'hallucination' ? 'font-medium' : undefined}>{style.title}</div>
        {visibleLines.length > 0 && (
          <ul className={`mt-0.5 space-y-0.5 ${style.detail}`}>
            {visibleLines.map((line) => {
              const typeMeta = line.issueType ? getCriticIssueTypeMeta(line.issueType) : null;
              return (
                <li key={line.text}>
                  {lines.length > 1 ? '· ' : ''}
                  {typeMeta && (
                    <span className={`font-medium ${typeMeta.className}`}>[{typeMeta.label}] </span>
                  )}
                  {line.text}
                  {line.claimSpan && (
                    <button
                      type="button"
                      onClick={() => onLocateClaim && onLocateClaim(line.claimSpan)}
                      disabled={!onLocateClaim}
                      title={line.claimSpan}
                      className={`ml-1 underline underline-offset-2 ${style.muted} ${onLocateClaim ? '' : 'cursor-default no-underline'}`}
                    >
                      定位原句
                    </button>
                  )}
                </li>
              );
            })}
          </ul>
        )}
        {hiddenCount > 0 && (
          <button
            type="button"
            onClick={() => setExpanded(true)}
            className={`mt-0.5 underline underline-offset-2 ${style.muted}`}
          >
            还有 {hiddenCount} 条问题
          </button>
        )}
        {showSuggestion && (
          <div className={`mt-0.5 ${style.detail}`}>建议：{suggestion}</div>
        )}
        {showConfidence && (
          <div className={`mt-0.5 ${style.muted}`}>
            自审评分:{' '}
            <span className={`font-medium ${tier.className}`}>
              {(critic.confidence * 100).toFixed(0)}%（{tier.label}）
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

export default AnswerCriticNotice;
