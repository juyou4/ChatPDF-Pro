import React, { useState } from 'react';
import {
  Brain,
  Check,
  ChevronDown,
  Copy,
  Database,
  Image as ImageIcon,
  Loader2,
  MessageCircle,
  RefreshCw,
  Scan,
  ScanText,
  Sparkles,
  X,
} from 'lucide-react';
import { motion } from 'framer-motion';
import { buildCriticDetailLines, getFullDocumentSummaryCoverage, hasCitationRisk, locateTextInElement } from '../utils/answerCriticUtils';
import { shouldStreamAssistantContent } from '../utils/messageRenderUtils';
import AnswerCriticNotice from './AnswerCriticNotice';
import AssistantModelIdentity from './AssistantModelIdentity';
import BlurText from './BlurText';
import { DocumentVisualAttachments } from './DocumentFigure';
import DocumentUploadNotice, { resolveDocumentUploadNotice } from './DocumentUploadNotice';
import EvidencePanel from './EvidencePanel';
import MindmapView from './MindmapView';
import StreamingMarkdown from './StreamingMarkdown';
import ThinkingBlock from './ThinkingBlock';
import {
  chatMessageRowPropsAreEqual,
  resolveTableVisualVerificationState,
} from '../utils/chatMessageRowMemo';

const BLUR_TEXT_STATUS_PROFILES = {
  light: {
    delay: 36,
    maxDelay: 240,
    stepDuration: 0.16,
    from: { filter: 'blur(3px)', opacity: 0, y: -4 },
    to: [
      { filter: 'blur(1px)', opacity: 0.62, y: 0.6 },
      { filter: 'blur(0px)', opacity: 1, y: 0 },
    ],
  },
  medium: {
    delay: 50,
    maxDelay: 420,
    stepDuration: 0.22,
    from: { filter: 'blur(6px)', opacity: 0, y: -8 },
    to: [
      { filter: 'blur(2.3px)', opacity: 0.55, y: 1 },
      { filter: 'blur(0px)', opacity: 1, y: 0 },
    ],
  },
  strong: {
    delay: 64,
    maxDelay: 560,
    stepDuration: 0.28,
    from: { filter: 'blur(9px)', opacity: 0, y: -12 },
    to: [
      { filter: 'blur(3.8px)', opacity: 0.48, y: 1.5 },
      { filter: 'blur(0px)', opacity: 1, y: 0 },
    ],
  },
};

const MEMORY_KIND_LABELS = {
  working: '工作记忆',
  profile: '画像',
  doc_fact: '文档事实',
  episodic: '对话摘要',
  consolidated: '压缩事实',
  graph: '图谱',
};

const MemoryHitsBadge = ({ hits, meta }) => {
  const [expanded, setExpanded] = useState(false);
  if (!Array.isArray(hits) || hits.length === 0) return null;

  return (
    <div className="mt-3 border-t border-gray-100 pt-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-xs text-emerald-600 hover:text-emerald-800 transition-colors font-medium"
      >
        <Brain className="w-3.5 h-3.5" />
        <span>记忆命中 ({hits.length})</span>
        {typeof meta?.used_tokens === 'number' && meta?.token_budget ? (
          <span className="rounded-full bg-emerald-50 px-1.5 py-0.5 text-[10px] text-emerald-700 tabular-nums">
            {meta.used_tokens}/{meta.token_budget} token
          </span>
        ) : null}
        {meta?.truncated && <span className="rounded-full bg-emerald-50 px-1.5 py-0.5 text-[10px] text-emerald-700">已截断</span>}
        <ChevronDown className={`w-3 h-3 transition-transform ${expanded ? 'rotate-180' : ''}`} />
      </button>
      {expanded && (
        <div className="mt-2 space-y-2">
          {meta && (
            <div className="rounded-xl border border-emerald-100/80 bg-white px-2.5 py-2 text-[11px] text-gray-600">
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                {typeof meta.retrieved_count === 'number' && (
                  <span>召回 <b className="text-gray-800 tabular-nums">{meta.retrieved_count}</b></span>
                )}
                {typeof meta.selected_count === 'number' && (
                  <span>注入 <b className="text-gray-800 tabular-nums">{meta.selected_count}</b></span>
                )}
                {typeof meta.used_tokens === 'number' && (
                  <span>
                    占用 <b className="text-gray-800 tabular-nums">{meta.used_tokens}</b>
                    {meta.token_budget ? <span className="text-gray-400"> / {meta.token_budget}</span> : null} token
                  </span>
                )}
                {meta.strategy && <span className="text-gray-400">{meta.strategy}</span>}
              </div>
              {Array.isArray(meta.selected_kinds) && meta.selected_kinds.length > 0 && (
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {Object.entries(
                    meta.selected_kinds.reduce((acc, kind) => {
                      acc[kind] = (acc[kind] || 0) + 1;
                      return acc;
                    }, {})
                  ).map(([kind, count]) => (
                    <span key={kind} className="rounded-full bg-gray-50 px-1.5 py-0.5 text-[10px] text-gray-600">
                      {MEMORY_KIND_LABELS[kind] || kind} ×{count}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}
          {hits.map((hit, i) => (
            <div key={hit.id || `${hit.memory_kind}-${i}`} className="rounded-xl border border-emerald-100 bg-emerald-50/60 p-2.5">
              <div className="flex items-center gap-2 text-[11px] text-emerald-700">
                <span className="rounded-full bg-white px-1.5 py-0.5 font-semibold">{MEMORY_KIND_LABELS[hit.memory_kind] || hit.memory_kind || '记忆'}</span>
                {hit.memory_scope && <span>{hit.memory_scope === 'profile' ? '全局' : '当前文档'}</span>}
                {typeof hit.score === 'number' && <span>score {hit.score.toFixed(2)}</span>}
              </div>
              <div className="mt-1 text-xs font-medium text-gray-800">{hit.title || '记忆条目'}</div>
              <div className="mt-1 text-xs leading-5 text-gray-600">{hit.summary || hit.content}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const TABLE_VISUAL_VERIFICATION_STATUS_META = {
  pending: {
    label: '正在核验',
    Icon: Loader2,
    className: 'border-sky-200 bg-sky-50 text-sky-700',
    iconClassName: 'animate-spin',
  },
  confirmed: {
    label: '核验确认',
    Icon: Check,
    className: 'border-emerald-200 bg-emerald-50 text-emerald-700',
    iconClassName: '',
  },
  conflict: {
    label: '发现冲突',
    Icon: Scan,
    className: 'border-amber-200 bg-amber-50 text-amber-700',
    iconClassName: '',
  },
  indeterminate: {
    label: '无法确定',
    Icon: ScanText,
    className: 'border-slate-200 bg-slate-50 text-slate-700',
    iconClassName: '',
  },
  failed: {
    label: '核验失败',
    Icon: X,
    className: 'border-rose-200 bg-rose-50 text-rose-700',
    iconClassName: '',
  },
  skipped: {
    label: '未调用',
    Icon: ScanText,
    className: 'border-slate-200 bg-slate-50 text-slate-600',
    iconClassName: '',
  },
  stale: {
    label: '结果已失效',
    Icon: ScanText,
    className: 'border-slate-200 bg-slate-50 text-slate-600',
    iconClassName: '',
  },
};

export const resolveTableVisualVerificationStatus = (verification) => {
  const state = resolveTableVisualVerificationState(verification);
  return TABLE_VISUAL_VERIFICATION_STATUS_META[state]
    ? { state, ...TABLE_VISUAL_VERIFICATION_STATUS_META[state] }
    : null;
};

const getTableVisualVerificationDetail = (verification, state) => {
  const explicitDetail = [
    verification?.summary,
    verification?.message,
    verification?.note,
    verification?.table_caption,
    verification?.table_id,
  ].find((value) => typeof value === 'string' && value.trim());
  if (explicitDetail) return explicitDetail.trim();
  if (state === 'pending') return '正在比对表格截图与结构化单元格';
  if (state === 'confirmed') return '视觉结果与结构化表格证据一致';
  if (state === 'conflict') return '视觉结果与结构化表格证据存在差异';
  if (state === 'indeterminate') return '图像证据不足以作出可靠判断';
  if (state === 'stale') return '文档解析结果已更新，请按需重新核验';
  if (state === 'skipped') {
    const skippedReason = String(verification?.skipped_reason || verification?.reason || '').trim();
    const reasonLabels = {
      mode_off: '设置为关闭，未发送表格截图',
      model_not_vision_capable: '当前模型不支持图片输入',
      missing_visual_model: '未配置可用的图表理解模型',
      visual_policy_model_unavailable: '视觉策略没有可调用的模型',
      missing_pdf: '找不到原始 PDF，无法生成截图',
      no_table_target: '没有找到明确的表格目标',
      ambiguous_table_target: '表格目标有歧义，未猜测调用',
      no_segments: '没有足够的表格证据触发核验',
      not_risky: '结构化证据通过风险检查，未发送截图',
      already_present: '已复用本轮已有的视觉核验结果',
    };
    return reasonLabels[skippedReason] || '本次未发送表格截图';
  }
  return '本次视觉核验未完成';
};

const TableVisualVerificationStatus = ({ verification }) => {
  const status = resolveTableVisualVerificationStatus(verification);
  if (!status) return null;

  const { Icon } = status;
  const detail = getTableVisualVerificationDetail(verification, status.state);
  return (
    <div
      className={`mt-3 flex min-w-0 items-center gap-2 rounded-lg border px-2.5 py-2 text-xs ${status.className}`}
      role="status"
      aria-live={status.state === 'pending' ? 'polite' : 'off'}
    >
      <Icon className={`h-3.5 w-3.5 shrink-0 ${status.iconClassName}`} />
      <span className="shrink-0 font-medium">表格视觉核验</span>
      <span className="shrink-0">{status.label}</span>
      <span className="min-w-0 truncate opacity-80" title={detail}>{detail}</span>
    </div>
  );
};

const CHAT_TURN_STATUS_META = {
  truncated: {
    Icon: MessageCircle,
    text: '回答达到模型输出上限，当前内容可能不完整，不会用于后续对话上下文或记忆',
    className: 'border-amber-200 bg-amber-50 text-amber-700',
  },
  interrupted: {
    Icon: MessageCircle,
    text: '生成已停止，当前内容可能不完整，不会用于后续对话上下文',
    className: 'border-slate-200 bg-slate-50 text-slate-600',
  },
  degraded: {
    Icon: X,
    text: '模型服务异常，当前为降级回答，不会用于后续对话上下文',
    className: 'border-rose-200 bg-rose-50 text-rose-700',
  },
  evidence_fallback: {
    Icon: ScanText,
    text: '模型生成失败，当前内容由文档证据兜底，不会用于后续对话上下文',
    className: 'border-amber-200 bg-amber-50 text-amber-700',
  },
  recovered_retry: {
    Icon: Sparkles,
    text: '首次生成异常，已自动重试并完成回答',
    className: 'border-emerald-200 bg-emerald-50 text-emerald-700',
  },
};

const ChatTurnStatusNotice = ({ status, stale = false }) => {
  const normalized = String(status || '').trim().toLowerCase();
  const meta = stale
    ? {
      Icon: ScanText,
      text: '文档解析结果已更新，此回答仅作历史参考，不会用于后续对话上下文',
      className: 'border-slate-200 bg-slate-50 text-slate-600',
    }
    : CHAT_TURN_STATUS_META[normalized];
  if (!meta) return null;
  const { Icon } = meta;
  return (
    <div
      className={`mt-2 flex items-start gap-1.5 rounded-lg border px-2.5 py-2 text-xs ${meta.className}`}
      role="status"
    >
      <Icon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
      <span>{meta.text}</span>
    </div>
  );
};

export { chatMessageRowPropsAreEqual, isDocumentUploadNoticeMessage } from '../utils/chatMessageRowMemo';

function resolveLiveParseStatus(msg, liveParseStatus, runtime) {
  const documentUploadNotice = msg.type === 'system'
    ? resolveDocumentUploadNotice(msg)
    : null;
  if (!documentUploadNotice) return { documentUploadNotice: null, liveParseStatus: null };
  const noticeDocId = String(documentUploadNotice?.docId || '');
  const currentFilename = String(runtime?.docFilename || '').trim().toLocaleLowerCase();
  const currentPageCount = Number(runtime?.docPageCount || 0);
  const noticePageCount = Number(documentUploadNotice?.pageCount || 0);
  const matchesLegacyCurrentDocument = Boolean(
    !noticeDocId
    && runtime?.docId
    && currentFilename
    && String(documentUploadNotice?.filename || '').trim().toLocaleLowerCase() === currentFilename
    && (!currentPageCount || !noticePageCount || currentPageCount === noticePageCount)
  );
  const noticeBelongsToCurrentDocument = noticeDocId
    ? noticeDocId === String(runtime?.docId || '')
    : matchesLegacyCurrentDocument;
  return {
    documentUploadNotice,
    liveParseStatus: noticeBelongsToCurrentDocument ? liveParseStatus : null,
  };
}

function ChatMessageRowInner({
  msg,
  idx,
  isLatest,
  isStreaming,
  liveParseStatus,
  copied,
  liked,
  remembered,
  disliked,
  conflictRecoveryStatus,
  currentEmbeddingLabel,
  indexedEmbeddingLabel,
  ragIndexBusy,
  runtime,
}) {
  const {
    darkMode,
    enableBlurReveal,
    blurIntensity,
    reduceMotion,
    messageStyle,
    reasoningEffort,
    apiProvider,
    docId,
    confirmRegenerateMessage,
    streamingThinkingRef,
    streamingContentRef,
    handleCitationClick,
    handleDocumentAwareCitationClick,
    copyMessage,
    regenerateMessage,
    saveToMemory,
    confirmAction,
    activeCitationRef,
    setActiveCitationRef,
    setFeedbackTarget,
    handleRebuildMinerURagIndex,
  } = runtime;

  const noticeState = resolveLiveParseStatus(msg, liveParseStatus, runtime);
  if (noticeState.documentUploadNotice) {
    return (
      <div className="flex w-full items-start">
        <DocumentUploadNotice
          notice={noticeState.documentUploadNotice}
          liveParseStatus={noticeState.liveParseStatus}
        />
      </div>
    );
  }

  const hasThinking = typeof msg.thinking === 'string' && msg.thinking.trim().length > 0;
  const hasRetrievalProgress = Array.isArray(msg.retrievalProgress) && msg.retrievalProgress.length > 0;
  const hasAgentTrace = msg.type === 'assistant' && msg.agentTrace && msg.agentTrace.enabled;
  const isStreamingCurrentMessage = isStreaming === true
    || shouldStreamAssistantContent(msg, runtime?.streamingMessageId);
  const blurTextStatusProfile = BLUR_TEXT_STATUS_PROFILES[blurIntensity]
    || BLUR_TEXT_STATUS_PROFILES.medium;
  const hasWebSearchActivity = msg.type === 'assistant' && Boolean(
    msg.webSearchStatus
    || (Array.isArray(msg.webSearchSources) && msg.webSearchSources.length > 0)
    || (Array.isArray(msg.webSearchReads) && msg.webSearchReads.length > 0)
    || msg.webSearchAudit?.requested
  );
  const hasReasoningStatusNotice = msg.type === 'assistant' && Boolean(
    msg.reasoningResolution?.fallback
    || (msg.reasoningResolution?.enabled && msg.reasoningResolution?.output_observed === false)
  );
  const shouldShowThinking = hasThinking || hasRetrievalProgress || hasAgentTrace || hasWebSearchActivity || hasReasoningStatusNotice || (
    isStreamingCurrentMessage && (
      reasoningEffort !== 'off'
      || !msg.content
      || !msg.content.trim()
    )
  );
  const shouldStreamContent = isStreamingCurrentMessage;
  const isEmbeddingIdentityConflict = msg.type === 'assistant' && msg.embeddingIdentityConflict === true;
  const selectedEmbeddingConfig = isEmbeddingIdentityConflict
    ? (runtime.getEmbeddingConfig?.() || {})
    : null;
  const resolvedCurrentEmbeddingLabel = currentEmbeddingLabel || (
    selectedEmbeddingConfig?.isValid
      ? `${selectedEmbeddingConfig.modelId || selectedEmbeddingConfig.compositeKey || '当前模型'} · ${selectedEmbeddingConfig.provider?.name || selectedEmbeddingConfig.providerId || '当前 Provider'}`
      : '当前设置不可用'
  );
  const resolvedIndexedEmbeddingLabel = indexedEmbeddingLabel
    || `${runtime.ragIndexStatus?.embedding_model || '未知模型'} · ${runtime.ragIndexStatus?.embedding_provider || '未知 Provider'}`;
  const criticDetailLines = buildCriticDetailLines(msg.answerCritic);
  const criticAnswerScopeId = `critic-answer-${msg.id ?? idx}`;
  const handleLocateClaim = (claimSpan) => {
    const scope = document.querySelector(`[data-critic-answer="${criticAnswerScopeId}"]`);
    if (scope) locateTextInElement(scope, claimSpan);
  };
  const animateEnter = !reduceMotion && isLatest && !isStreamingCurrentMessage && !msg.answerStarted;

  return (
    <motion.div
      initial={animateEnter ? { opacity: 0, y: 10 } : false}
      animate={{ opacity: 1, y: 0 }}
      className={`content-typography flex flex-col ${msg.type === 'user' ? 'items-end' : 'items-start'}`}
      style={{ fontSize: 'var(--content-font-size, 14px)' }}
    >
      <div className={`${msg.type === 'user'
        ? 'w-fit max-w-[78%] min-w-0 px-[18px] py-[10px] message-bubble-user'
        : messageStyle === 'bubble'
          ? 'max-w-[88%] min-w-0 px-5 py-4 message-bubble-ai overflow-hidden'
          : 'w-full max-w-full min-w-0 bg-transparent shadow-none p-0 text-gray-800 dark:text-gray-50 overflow-hidden'
      }`}
        style={msg.type !== 'user' && messageStyle !== 'bubble' ? { contain: 'inline-size' } : undefined}
      >
        {msg.type === 'assistant' && (
          <AssistantModelIdentity
            model={msg.model || msg.used_model}
            providerId={msg.provider || msg.used_provider || apiProvider}
            darkMode={darkMode}
          />
        )}
        {shouldShowThinking && (
          <ThinkingBlock
            content={msg.thinking}
            isStreaming={isStreamingCurrentMessage}
            answerStarted={Boolean(msg.answerStarted)}
            answerGenerating={Boolean(msg.answerGenerating)}
            darkMode={darkMode}
            thinkingMs={msg.thinkingMs || 0}
            streamingRef={isStreamingCurrentMessage ? streamingThinkingRef : undefined}
            retrievalProgress={msg.retrievalProgress || []}
            agentTrace={msg.agentTrace || null}
            liveMessageId={isStreamingCurrentMessage ? msg.id : null}
            reasoningResolution={msg.reasoningResolution || null}
            thinkingLive={Boolean(msg.thinkingLive)}
            webSearchActivity={hasWebSearchActivity ? {
              sources: msg.webSearchSources || [],
              reads: msg.webSearchReads || [],
              status: msg.webSearchStatus || null,
              audit: msg.webSearchAudit || null,
              query: msg.webSearchQuery || '',
            } : null}
          />
        )}
        {msg.hasImage && (
          <div className="mb-2 rounded-lg overflow-hidden border border-white/20">
            <div className="bg-black/10 p-2 flex items-center gap-2 text-xs">
              <ImageIcon className="w-3 h-3" /> Image attached
            </div>
          </div>
        )}
        {msg.maxRelevanceScore !== null && msg.maxRelevanceScore !== undefined && msg.maxRelevanceScore >= 0 && msg.maxRelevanceScore < 0.3 && !msg.isStreaming && (
          <div className="mb-2 px-3 py-2 rounded-lg bg-amber-50 border border-amber-200 text-amber-700 text-xs flex items-center gap-1.5">
            <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" /></svg>
            <span>检索到的内容与您的问题相关性较低，回答可能不够准确，请谨慎参考。</span>
          </div>
        )}
        {msg.answerCritic && !msg.isStreaming && msg.answerCritic.has_hallucination && (
          <AnswerCriticNotice
            critic={msg.answerCritic}
            variant="hallucination"
            detailLines={criticDetailLines}
            onLocateClaim={handleLocateClaim}
          />
        )}
        {msg.answerCritic && !msg.isStreaming && !msg.answerCritic.has_hallucination && hasCitationRisk(msg.answerCritic) && (
          <AnswerCriticNotice
            critic={msg.answerCritic}
            variant="citation"
            detailLines={criticDetailLines}
            onLocateClaim={handleLocateClaim}
          />
        )}
        {msg.answerGenerating && isStreamingCurrentMessage && !(msg.content && String(msg.content).trim()) && (
          enableBlurReveal ? (
            <BlurText
              text="正在生成回答..."
              delay={blurTextStatusProfile.delay}
              maxDelay={blurTextStatusProfile.maxDelay}
              animateBy="letters"
              direction="top"
              animationFrom={blurTextStatusProfile.from}
              animationTo={blurTextStatusProfile.to}
              stepDuration={blurTextStatusProfile.stepDuration}
              easing={[0.22, 1, 0.36, 1]}
              className={`mb-2 text-[12px] ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}
            />
          ) : (
            <div className={`mb-2 text-[12px] ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              正在生成回答...
            </div>
          )
        )}
        {isEmbeddingIdentityConflict ? (
          <section className={`mt-1 rounded-[16px] border px-4 py-3.5 ${
            conflictRecoveryStatus === 'completed'
              ? (darkMode ? 'border-emerald-400/20 bg-emerald-400/[0.08] text-emerald-100' : 'border-emerald-200 bg-emerald-50/70 text-emerald-900')
              : (darkMode ? 'border-amber-300/20 bg-amber-300/[0.08] text-amber-50' : 'border-amber-200 bg-amber-50/75 text-amber-950')
          }`}>
            <div className="flex min-w-0 items-start gap-3">
              <div className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-[11px] ${
                conflictRecoveryStatus === 'completed'
                  ? (darkMode ? 'bg-emerald-300/15 text-emerald-200' : 'bg-emerald-100 text-emerald-700')
                  : (darkMode ? 'bg-amber-200/15 text-amber-100' : 'bg-amber-100 text-amber-700')
              }`}>
                {conflictRecoveryStatus === 'completed' ? <Check className="h-4 w-4" /> : <Database className="h-4 w-4" />}
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-[13px] font-semibold">
                  {conflictRecoveryStatus === 'completed' ? '问答索引已同步' : '问答索引需要同步'}
                </p>
                <p className={`mt-1 text-[12px] leading-5 ${darkMode ? 'text-gray-300' : 'text-amber-900/75'}`}>
                  {conflictRecoveryStatus === 'completed'
                    ? '已按当前 Embedding 配置重建问答索引，可重新提问。MinerU 的解析结果、阅读结构和速览内容没有被改动。'
                    : 'MinerU 解析结果仍然有效。后端曾以不同的 Embedding 身份重建问答索引，保护校验因此停止了本次检索。'}
                </p>
              </div>
            </div>
            {conflictRecoveryStatus !== 'completed' && (
              <>
                <div className={`mt-3 grid gap-1.5 border-t pt-3 text-[11px] ${darkMode ? 'border-white/10 text-gray-400' : 'border-amber-200/80 text-amber-900/65'}`}>
                  <div className="flex min-w-0 items-center justify-between gap-3">
                    <span className="shrink-0">当前索引</span>
                    <span className="min-w-0 truncate text-right font-medium" title={resolvedIndexedEmbeddingLabel}>{resolvedIndexedEmbeddingLabel}</span>
                  </div>
                  <div className="flex min-w-0 items-center justify-between gap-3">
                    <span className="shrink-0">当前设置</span>
                    <span className="min-w-0 truncate text-right font-medium" title={resolvedCurrentEmbeddingLabel}>{resolvedCurrentEmbeddingLabel}</span>
                  </div>
                </div>
                <div className="mt-3 flex items-center justify-between gap-3">
                  <span className={`text-[11px] ${darkMode ? 'text-gray-400' : 'text-amber-900/60'}`}>只重建问答索引，不会重新上传 PDF</span>
                  <button
                    type="button"
                    onClick={() => handleRebuildMinerURagIndex({ forceEmbeddingRebuild: true, conflictMessageId: msg.id })}
                    disabled={ragIndexBusy || conflictRecoveryStatus === 'rebuilding'}
                    className={`inline-flex shrink-0 items-center gap-1.5 rounded-full px-3 py-1.5 text-[11px] font-semibold transition-all focus-visible:outline-none focus-visible:ring-2 disabled:cursor-not-allowed disabled:opacity-60 ${
                      darkMode
                        ? 'bg-white/10 text-white hover:bg-white/15 focus-visible:ring-amber-200/50'
                        : 'bg-[#3b3a38] text-white shadow-[0_3px_8px_rgba(41,37,36,0.18)] hover:bg-[#262523] focus-visible:ring-amber-500/35'
                    }`}
                  >
                    <RefreshCw className={`h-3.5 w-3.5 ${conflictRecoveryStatus === 'rebuilding' ? 'animate-spin' : ''}`} />
                    {conflictRecoveryStatus === 'rebuilding' ? '同步中' : conflictRecoveryStatus === 'failed' ? '重新尝试' : '按当前配置重建'}
                  </button>
                </div>
              </>
            )}
          </section>
        ) : (
          <div data-critic-answer={criticAnswerScopeId}>
            <StreamingMarkdown
              content={msg.content}
              isStreaming={shouldStreamContent}
              enableBlurReveal={enableBlurReveal}
              blurIntensity={blurIntensity}
              citations={msg.citations || null}
              onCitationClick={handleDocumentAwareCitationClick}
              streamingRef={shouldStreamContent ? streamingContentRef : undefined}
              webSearchSources={msg.webSearchSources || null}
              suppressInitialDots={
                Boolean(msg.answerGenerating)
                || (isStreamingCurrentMessage && !(msg.content && String(msg.content).trim()))
              }
            />
          </div>
        )}
        {msg.type === 'assistant' && !msg.isStreaming && msg.visualAttachments?.length > 0 && (
          <DocumentVisualAttachments
            attachments={msg.visualAttachments}
            docId={docId}
            darkMode={darkMode}
            onLocate={handleCitationClick}
          />
        )}
        {msg.type === 'assistant' && !msg.isStreaming && !isEmbeddingIdentityConflict && (
          <ChatTurnStatusNotice
            status={msg.turnStatus || msg.turn_status || msg.answerStatus || msg.answer_status}
            stale={msg.parseIdentityStale === true}
          />
        )}
        {msg.type === 'assistant' && !msg.isStreaming && msg.memoryHits && msg.memoryHits.length > 0 && (
          <MemoryHitsBadge hits={msg.memoryHits} meta={msg.memoryMeta} />
        )}
      </div>
      {msg.type === 'assistant' && !msg.isStreaming && msg.visualVerification && (
        <TableVisualVerificationStatus verification={msg.visualVerification} />
      )}
      {msg.type === 'assistant' && !msg.isStreaming && msg.citations && msg.citations.length > 0 && (
        <EvidencePanel
          citations={msg.citations}
          docId={docId}
          onCitationClick={handleDocumentAwareCitationClick}
          activeRef={activeCitationRef}
          onRefHover={setActiveCitationRef}
        />
      )}
      {msg.type === 'assistant' && !msg.isStreaming && msg.mindmapMarkdown && (
        <MindmapView markdown={msg.mindmapMarkdown} />
      )}
      {msg.type === 'assistant' && !msg.isStreaming && (
        <div className="flex flex-wrap items-center gap-1.5 mt-3 ml-2">
          <button onClick={() => copyMessage(msg.content, msg.id || idx)} className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-500 hover:text-gray-700 transition-colors" title="复制">
            {copied ? (
              <svg className="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>
            ) : (<Copy className="w-4 h-4" />)}
          </button>
          <button onClick={async () => {
            if (!confirmRegenerateMessage || await confirmAction({
              title: '重新生成这条回答',
              description: '会按当前模型和检索设置再生成一次，原回答会被替换。',
              confirmLabel: '重新生成',
            })) regenerateMessage(idx);
          }} className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-500 hover:text-gray-700 transition-colors" title="重新生成">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>
          </button>
          <button onClick={() => saveToMemory(idx, 'liked')} className={`p-1.5 rounded-lg hover:bg-gray-100 transition-colors ${liked ? 'text-pink-500' : 'text-gray-500 hover:text-gray-700'}`} title="点赞并记忆">
            <svg className="w-4 h-4" fill={liked ? 'currentColor' : 'none'} stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" /></svg>
          </button>
          <button onClick={() => setFeedbackTarget({ idx, msg })} className={`p-1.5 rounded-lg hover:bg-gray-100 transition-colors ${disliked ? 'text-orange-500' : 'text-gray-500 hover:text-gray-700'}`} title="点踩并反馈">
            <svg className="w-4 h-4" fill={disliked ? 'currentColor' : 'none'} stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018a2 2 0 01.485.06l3.76.94m-7 10v5a2 2 0 002 2h.096c.5 0 .905-.405.905-.904 0-.715.211-1.413.608-2.008L17 13V4m-7 10h2m5-10h2a2 2 0 012 2v6a2 2 0 01-2 2h-2.5" /></svg>
          </button>
          <button onClick={() => saveToMemory(idx, 'manual')} className={`p-1.5 rounded-lg hover:bg-gray-100 transition-colors ${remembered ? 'text-purple-500' : 'text-gray-500 hover:text-gray-700'}`} title="记住这个">
            <Brain className={`w-4 h-4 ${remembered ? 'fill-current' : ''}`} />
          </button>
          {msg.qaScore != null && (
            <span className={`ml-1.5 text-[10px] px-2 py-1 rounded-full font-medium ${msg.qaScore >= 0.7 ? 'bg-green-50 text-green-600' : msg.qaScore >= 0.4 ? 'bg-yellow-50 text-yellow-600' : 'bg-red-50 text-red-600'}`} title={`回答置信度: ${(msg.qaScore * 100).toFixed(0)}%`}>
              {(msg.qaScore * 100).toFixed(0)}%
            </span>
          )}
          {msg.answerCertainty?.label && (() => {
            const label = String(msg.answerCertainty.label);
            const fullDocumentCoverage = getFullDocumentSummaryCoverage(msg.answerCertainty);
            const styles = {
              Certain: 'bg-green-50 text-green-700 border-green-200',
              Partial: 'bg-amber-50 text-amber-700 border-amber-200',
              Unsure: 'bg-orange-50 text-orange-700 border-orange-200',
              Refused: 'bg-slate-100 text-slate-600 border-slate-200',
            };
            const titles = {
              Certain: '证据充分且引用覆盖较好',
              Partial: '部分证据充分，请核对关键结论',
              Unsure: '当前证据或引用覆盖不足，需核对关键结论',
              Refused: '模型判定文档证据不足以作答',
            };
            const zh = { Certain: '较确定', Partial: '部分确定', Unsure: '需核对', Refused: '已拒答' };
            return (
              <>
                <span
                  className={`ml-1.5 text-[10px] px-2 py-1 rounded-full font-medium border ${styles[label] || styles.Unsure}`}
                  title={titles[label] || '回答确定性'}
                >
                  {zh[label] || label}
                </span>
                {fullDocumentCoverage && (
                  <span
                    className={`ml-1.5 text-[10px] px-2 py-1 rounded-full font-medium border ${fullDocumentCoverage.complete
                      ? 'bg-sky-50 text-sky-700 border-sky-200'
                      : 'bg-amber-50 text-amber-700 border-amber-200'}`}
                    title={fullDocumentCoverage.title}
                  >
                    {fullDocumentCoverage.text}
                  </span>
                )}
              </>
            );
          })()}
        </div>
      )}
      {msg.type === 'assistant' && !msg.isStreaming && msg.followupQuestions && msg.followupQuestions.length > 0 && (
        <div className="flex flex-wrap gap-2 mt-3 ml-2">
          {msg.followupQuestions.map((q, qi) => (
            <button
              key={qi}
              onClick={() => {
                const textarea = document.querySelector('textarea');
                if (textarea) {
                  const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
                  nativeInputValueSetter.call(textarea, q);
                  textarea.dispatchEvent(new Event('input', { bubbles: true }));
                  textarea.focus();
                }
              }}
              className="text-xs px-3 py-1.5 rounded-full border border-gray-200 bg-white text-gray-600 shadow-[0_1px_3px_rgba(30,30,35,0.05)] transition-all duration-200 hover:border-[#FFA07A] hover:bg-[#FFF4EF] hover:text-[#B85F47] active:scale-95 cursor-pointer dark:border-white/10 dark:bg-white/[0.06] dark:text-gray-300 dark:shadow-none dark:hover:border-[#FFA07A]/40 dark:hover:bg-[#FFA07A]/10 dark:hover:text-[#FFDCCF]"
            >
              {q}
            </button>
          ))}
        </div>
      )}
    </motion.div>
  );
}

const ChatMessageRow = React.memo(ChatMessageRowInner, chatMessageRowPropsAreEqual);
ChatMessageRow.displayName = 'ChatMessageRow';

export default ChatMessageRow;
