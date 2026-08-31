import React, { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { AlertCircle, CheckCircle2, Clock3, Loader2, RefreshCw, Route, Upload, X } from 'lucide-react';
import { resolveDocumentParseState } from '../utils/parseRouteUtils';

const STATE_STYLES = {
  ready: {
    Icon: CheckCircle2,
    light: 'border-emerald-200/80 bg-[#f5fcf8]/95 text-emerald-700',
    dark: 'border-emerald-400/20 bg-[#16251f]/95 text-emerald-300',
    icon: 'bg-emerald-500/10',
  },
  processing: {
    Icon: Loader2,
    light: 'border-[#ed8c68]/25 bg-[#fff8f5]/95 text-[#b85f47]',
    dark: 'border-[#FFA07A]/25 bg-[#28201e]/95 text-[#FFA07A]',
    icon: 'bg-[#ed8c68]/10',
    iconClassName: 'animate-spin',
  },
  awaiting_publish: {
    // 这是等待用户修正/发布索引的停顿态，不是远端仍在运行。
    Icon: Clock3,
    light: 'border-amber-200/90 bg-[#fffaf0]/95 text-amber-700',
    dark: 'border-amber-300/20 bg-[#2a261b]/95 text-amber-300',
    icon: 'bg-amber-500/10',
  },
  // 能力已开放但覆盖面残缺：不能用 ready 的绿色勾，那会把降级说成完成。
  partial_ready: {
    Icon: AlertCircle,
    light: 'border-amber-200/90 bg-[#fffaf0]/95 text-amber-700',
    dark: 'border-amber-300/20 bg-[#2a261b]/95 text-amber-300',
    icon: 'bg-amber-500/10',
  },
  failed: {
    Icon: AlertCircle,
    light: 'border-rose-200/90 bg-[#fff7f7]/95 text-rose-700',
    dark: 'border-rose-300/20 bg-[#2b1d20]/95 text-rose-300',
    icon: 'bg-rose-500/10',
  },
  publish_failed: {
    Icon: AlertCircle,
    light: 'border-rose-200/90 bg-[#fff7f7]/95 text-rose-700',
    dark: 'border-rose-300/20 bg-[#2b1d20]/95 text-rose-300',
    icon: 'bg-rose-500/10',
  },
  cancelled: {
    Icon: AlertCircle,
    light: 'border-amber-200/90 bg-[#fffaf0]/95 text-amber-700',
    dark: 'border-amber-300/20 bg-[#2a261b]/95 text-amber-300',
    icon: 'bg-amber-500/10',
  },
  not_started: {
    Icon: RefreshCw,
    light: 'border-slate-200/90 bg-white/95 text-slate-600',
    dark: 'border-white/10 bg-[#20232a]/95 text-gray-300',
    icon: 'bg-slate-500/10',
  },
};

export const PARSE_NOTICE_SEEN_STORAGE_KEY = 'chatpdf_parse_notice_seen_v1';
const MAX_SEEN_PARSE_NOTICES = 80;

const resolveNoticeStorage = (storage) => {
  if (storage) return storage;
  if (typeof window === 'undefined') return null;
  return window.localStorage;
};

export const loadSeenParseNoticeKeys = (storage) => {
  try {
    const parsed = JSON.parse(resolveNoticeStorage(storage)?.getItem(PARSE_NOTICE_SEEN_STORAGE_KEY) || '[]');
    return new Set(Array.isArray(parsed) ? parsed.filter((item) => typeof item === 'string' && item) : []);
  } catch {
    return new Set();
  }
};

export const rememberParseNoticeKey = (noticeKey, storage) => {
  if (!noticeKey) return;
  const targetStorage = resolveNoticeStorage(storage);
  if (!targetStorage) return;
  try {
    const current = [...loadSeenParseNoticeKeys(targetStorage)];
    const next = [...current.filter((key) => key !== noticeKey), noticeKey].slice(-MAX_SEEN_PARSE_NOTICES);
    targetStorage.setItem(PARSE_NOTICE_SEEN_STORAGE_KEY, JSON.stringify(next));
  } catch {
    // 本地存储不可用时仍允许本次提醒正常展示。
  }
};

export const TASK_PILL_NOTICE_MOTION = {
  initial: {
    opacity: 0,
    y: -10,
  },
  animate: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.16,
      ease: [0.22, 1, 0.36, 1],
    },
  },
  exit: {
    opacity: 0,
    y: -10,
    transition: {
      duration: 0.12,
      ease: [0.4, 0, 1, 1],
    },
  },
};

const DocumentParseStatusBar = React.memo(({
  documentId = '',
  manifest,
  parseReady,
  deepParseStatus,
  ragIndexStatus,
  darkMode = false,
  onOpenProcessing,
  onRetry,
  onRetryIndex,
  onRefresh,
  onChooseRoute,
  suppressed = false,
  className = '',
}) => {
  const parseState = resolveDocumentParseState({ manifest, parseReady, deepParseStatus, ragIndexStatus });
  const ragPublishFailed = parseState.state === 'awaiting_publish'
    && ragIndexStatus?.status === 'failed';
  const state = ragPublishFailed
    ? {
        ...parseState,
        state: 'publish_failed',
        statusLabel: '问答索引发布失败',
        detail: ragIndexStatus?.error || '版面解析结果已保留，可在后台任务中重试发布',
      }
    : parseState;
  const style = STATE_STYLES[state.state] || STATE_STYLES.processing;
  const StateIcon = style.Icon;
  const canRetry = state.resolvedRoute === 'mineru'
    && ['failed', 'cancelled', 'not_started'].includes(state.state)
    && onRetry;
  const canRetryIndex = state.resolvedRoute === 'mineru' && state.state === 'publish_failed' && onRetryIndex;
  const statusSyncFailed = state.state === 'failed'
    && ['status_sync_failed', 'downstream_task_stalled'].includes(
      String(state.errorCode || deepParseStatus?.error_code || deepParseStatus?.stage || '').trim().toLowerCase(),
    );
  const canRefresh = statusSyncFailed && onRefresh;
  const retryLabel = deepParseStatus?.resume_available && deepParseStatus?.resume_kind === 'result_download'
    ? '重试结果下载'
    : ['not_started', 'cancelled'].includes(state.state)
      ? '开始解析'
    : '重试 MinerU';
  const canInspect = state.resolvedRoute === 'mineru'
    && ['processing', 'awaiting_publish', 'publish_failed'].includes(state.state)
    && onOpenProcessing;
  // 进行中的解析进度已经在聊天区上传卡片里展示，右上角弹窗只保留失败/重建等提醒。
  const isParseProgressNotice = state.state === 'processing';
  const shouldOfferRouteChange = ['failed', 'cancelled'].includes(state.state) && onChooseRoute;
  const noticeKey = [
    documentId,
    manifest?.generation,
    manifest?.source_hash,
    state.state,
    state.errorCode,
    deepParseStatus?.stage,
  ].map((value) => String(value || '')).join('|');
  const [dismissedNoticeKey, setDismissedNoticeKey] = useState('');
  const seenNoticeKeysRef = useRef(null);
  const shownNoticeKeyRef = useRef('');
  if (!seenNoticeKeysRef.current) {
    seenNoticeKeysRef.current = loadSeenParseNoticeKeys();
  }
  const wasSeenBeforeThisMount = (
    seenNoticeKeysRef.current.has(noticeKey)
    && shownNoticeKeyRef.current !== noticeKey
  );
  useEffect(() => {
    if (suppressed) setDismissedNoticeKey(noticeKey);
  }, [noticeKey, suppressed]);
  const visible = !isParseProgressNotice
    && !suppressed
    && dismissedNoticeKey !== noticeKey
    && !wasSeenBeforeThisMount;

  useEffect(() => {
    if (!visible || shownNoticeKeyRef.current === noticeKey) return;
    shownNoticeKeyRef.current = noticeKey;
    seenNoticeKeysRef.current.add(noticeKey);
    rememberParseNoticeKey(noticeKey);
  }, [noticeKey, visible]);

  return (
    <div
      data-testid="parse-notification-layer"
      data-motion-origin="task-pill"
      className={`pointer-events-none absolute right-0 top-[3.25rem] z-40 flex w-[330px] max-w-[calc(100vw-2.5rem)] justify-end ${className}`}
    >
      <AnimatePresence initial={false}>
        {visible && (
          <motion.aside
            key={noticeKey}
            role="status"
            aria-live={state.state === 'ready' ? 'off' : 'polite'}
            initial={TASK_PILL_NOTICE_MOTION.initial}
            animate={TASK_PILL_NOTICE_MOTION.animate}
            exit={TASK_PILL_NOTICE_MOTION.exit}
            className={`pointer-events-auto relative w-full rounded-[16px] border p-3 backdrop-blur-xl ${
              darkMode
                ? `${style.dark} shadow-[0_18px_46px_-18px_rgba(0,0,0,0.72),0_4px_12px_rgba(0,0,0,0.3)]`
                : `${style.light} shadow-[0_18px_44px_-18px_rgba(78,64,56,0.34),0_4px_12px_rgba(78,64,56,0.1)]`
            }`}
          >
              <div className="flex items-start gap-2.5">
                <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-[9px] ${style.icon}`}>
                  <StateIcon className={`h-4 w-4 ${style.iconClassName || ''}`} />
                </div>
              <div className="min-w-0 flex-1">
                <div className="flex min-w-0 items-center gap-2 text-[12px]">
                  <span className="inline-flex shrink-0 items-center gap-1.5 font-bold">
                    <Route className="h-3.5 w-3.5" />
                    {state.routeLabel}
                  </span>
                  <span className="h-3 w-px shrink-0 bg-current opacity-20" aria-hidden="true" />
                  <span className="truncate font-semibold">{state.statusLabel}</span>
                </div>
                <p className="mt-1 line-clamp-2 text-[11px] leading-4 opacity-70">{state.detail}</p>
                {(canInspect || canRetry || canRetryIndex || shouldOfferRouteChange) && (
                  <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
                    {canInspect && (
                      <button
                        type="button"
                        onClick={onOpenProcessing}
                        className={`rounded-[8px] px-2.5 py-1.5 text-[11px] font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/35 ${darkMode ? 'hover:bg-white/10' : 'bg-white/55 hover:bg-white'}`}
                      >
                        {state.state === 'publish_failed' ? '查看任务' : '查看进度'}
                      </button>
                    )}
                    {canRefresh && (
                      <button
                        type="button"
                        onClick={onRefresh}
                        className={`inline-flex items-center gap-1.5 rounded-[8px] px-2.5 py-1.5 text-[11px] font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/35 ${darkMode ? 'bg-white/10 hover:bg-white/15' : 'bg-white/65 hover:bg-white'}`}
                      >
                        <RefreshCw className="h-3.5 w-3.5" />
                        刷新状态
                      </button>
                    )}
                    {canRetry && !canRefresh && (
                      <button
                        type="button"
                        onClick={onRetry}
                        className={`inline-flex items-center gap-1.5 rounded-[8px] px-2.5 py-1.5 text-[11px] font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-300/45 ${darkMode ? 'bg-rose-300/10 hover:bg-rose-300/15' : 'bg-white/65 hover:bg-white'}`}
                      >
                        <RefreshCw className="h-3.5 w-3.5" />
                        {retryLabel}
                      </button>
                    )}
                    {canRetryIndex && (
                      <button
                        type="button"
                        onClick={onRetryIndex}
                        className={`inline-flex items-center gap-1.5 rounded-[8px] px-2.5 py-1.5 text-[11px] font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-300/45 ${darkMode ? 'bg-rose-300/10 hover:bg-rose-300/15' : 'bg-white/65 hover:bg-white'}`}
                      >
                        <RefreshCw className="h-3.5 w-3.5" />
                        重试发布
                      </button>
                    )}
                    {shouldOfferRouteChange && (
                      <button
                        type="button"
                        onClick={onChooseRoute}
                        className={`inline-flex items-center gap-1.5 rounded-[8px] px-2.5 py-1.5 text-[11px] font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/35 ${darkMode ? 'bg-white/10 hover:bg-white/15' : 'bg-white/65 hover:bg-white'}`}
                      >
                        <Upload className="h-3.5 w-3.5" />
                        重新上传
                      </button>
                    )}
                  </div>
                )}
              </div>
              <button
                type="button"
                aria-label="关闭解析提醒"
                title="关闭"
                onClick={() => setDismissedNoticeKey(noticeKey)}
                className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-[8px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-current/25 ${darkMode ? 'hover:bg-white/10' : 'hover:bg-black/[0.05]'}`}
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>
    </div>
  );
});

export default DocumentParseStatusBar;
