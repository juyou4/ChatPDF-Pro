import { memo, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import {
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  Loader2,
  MoreHorizontal,
  RotateCcw,
  Sparkles,
  Trash2,
  X,
} from 'lucide-react';
import MinerUScanLoader from './MinerUScanLoader';
import { useMinerUElapsedClock } from '../hooks/useMinerUElapsedClock';
import {
  applyMinerUElapsedToProgress,
  canonicalizeMinerUProgressStage,
  formatMinerUElapsed,
} from '../utils/mineruProgressUtils';

const VISIBLE_TASK_STATES = new Set(['running', 'failed', 'recommended']);

const getTaskProgress = (progress) => {
  if (!progress || typeof progress !== 'object') return null;
  const percent = Number(progress.percent);
  if (!Number.isFinite(percent)) return null;
  const normalizedPercent = Math.max(0, Math.min(100, Math.round(percent)));
  return {
    percent: normalizedPercent,
    label: progress.label || `${normalizedPercent}%`,
    ariaLabel: progress.ariaLabel || `进度 ${normalizedPercent}%`,
  };
};

export const TASK_PILL_PANEL_MOTION = {
  initial: {
    opacity: 0,
    y: -10,
  },
  animate: {
    opacity: 1,
    y: 0,
  },
  exit: {
    opacity: 0,
    y: -10,
  },
  transition: {
    duration: 0.16,
    ease: [0.22, 1, 0.36, 1],
  },
};

export const getVisibleBackgroundTasks = (items, hasDocument) => {
  if (!hasDocument || !Array.isArray(items)) return [];
  return items.filter((item) => VISIBLE_TASK_STATES.has(item?.state));
};

const canonicalizeTaskStatus = (status) => {
  const text = String(status || '').trim();
  if (/^(预估|远端)\s*\d+%$/.test(text)) return '';
  return text;
};

const backgroundTaskSignature = (item) => {
  if (!item || typeof item !== 'object') return '';
  const progress = item.progress && typeof item.progress === 'object' ? item.progress : {};
  const events = Array.isArray(item.events) ? item.events : [];
  const lastEvent = events[events.length - 1] || {};
  return [
    item.id,
    item.state,
    canonicalizeTaskStatus(item.status),
    item.desc,
    item.title,
    item.parseGeneration,
    item.actionLabel,
    canonicalizeMinerUProgressStage(progress.stage),
    progress.estimated === false ? progress.percent : '',
    events.length,
    lastEvent.stage,
    lastEvent.status,
  ].map((value) => String(value ?? '')).join('|');
};

export const stabilizeBackgroundTaskItems = (previous, next) => {
  if (previous === next) return previous ?? next ?? [];
  if (!Array.isArray(next)) return [];
  if (
    Array.isArray(previous)
    && previous.length === next.length
    && previous.every((item, index) => {
      const incoming = next[index];
      const prevGen = String(item?.parseGeneration || '');
      const nextGen = String(incoming?.parseGeneration || '');
      const hydrated = nextGen ? incoming : { ...incoming, parseGeneration: prevGen };
      return backgroundTaskSignature(item) === backgroundTaskSignature(hydrated);
    })
  ) {
    return previous;
  }
  return next;
};

export const getBackgroundTaskSummary = (items) => {
  const tasks = Array.isArray(items) ? items : [];
  const failedCount = tasks.filter((item) => item.state === 'failed').length;
  const runningCount = tasks.filter((item) => item.state === 'running').length;
  const recommendedCount = tasks.filter((item) => item.state === 'recommended').length;
  const minerUProgress = getTaskProgress(
    tasks.find((item) => item?.id === 'deep_parse' && item?.state === 'running')?.progress
  );

  if (failedCount > 0) return { state: 'failed', label: '任务异常', count: failedCount };
  if (minerUProgress) return { state: 'running', label: `解析 ${minerUProgress.percent}%`, count: runningCount };
  if (runningCount > 0) return { state: 'running', label: `任务 ${runningCount}`, count: runningCount };
  if (recommendedCount > 0) return { state: 'recommended', label: `建议 ${recommendedCount}`, count: recommendedCount };
  return { state: 'idle', label: '任务', count: 0 };
};

const TASK_STYLES = {
  running: {
    Icon: Loader2,
    lightIconClassName: 'animate-spin text-[#ed8c68]',
    darkIconClassName: 'animate-spin text-[#FFA07A]',
    lightStatusClassName: 'text-[#B85F47]',
    darkStatusClassName: 'text-[#FFA07A]',
  },
  failed: {
    Icon: AlertCircle,
    lightIconClassName: 'text-[#B85F47]',
    darkIconClassName: 'text-[#FFA07A]',
    lightStatusClassName: 'text-[#B85F47]',
    darkStatusClassName: 'text-[#FFA07A]',
  },
  recommended: {
    Icon: Sparkles,
    lightIconClassName: 'text-amber-500',
    darkIconClassName: 'text-amber-300',
    lightStatusClassName: 'text-amber-700',
    darkStatusClassName: 'text-amber-300',
  },
};

const EVENT_STAGE_LABELS = {
  queued: '排队',
  upload: '上传',
  submit_mineru: '提交 MinerU',
  poll: '等待解析',
  download: '下载结果',
  normalize: '整理结构',
  publish_block_index: '发布阅读块',
  build_rag: '构建问答索引',
  rag_index: '问答索引',
  embedding: 'Embedding 服务',
  stalled: '无进展终止',
  publish_visual_assets: '发布视觉资产',
  downstream_ai: '下游 AI',
  ready: '已就绪',
  failed: '失败',
  cancelled: '已取消',
  restart_recovery: '重启恢复',
};

const SHORTFALL_LABELS = {
  quality_gate_failed: '解析质量门未通过',
  claim_support_shortfall: '部分结论缺少足够证据',
  partial_quality: '部分章节未达到质量门',
  degraded_result: '已返回降级结果',
  worker_interrupted: '服务重启导致任务中断',
  generation_failed: '生成阶段失败',
  embedding_quota_exhausted: 'Embedding 余额或额度不足',
  embedding_auth_failed: 'Embedding 凭证无效',
  embedding_model_unavailable: 'Embedding 模型不可用',
  embedding_rate_limited: 'Embedding 请求被限流',
  embedding_network_error: 'Embedding 网络不可达',
  rag_index_stalled: '问答索引长时间无进展',
  rag_index_quality_failed: '问答索引质量校验失败',
  rag_index_storage_failed: '问答索引写入失败',
  downstream_task_stalled: '下游任务长时间无进展',
  task_start_failed: '任务启动失败',
  status_sync_failed: '状态同步失败',
  mineru_quota_exhausted: 'MinerU 余额或额度不足',
  mineru_rate_limited: 'MinerU 请求被限流',
  mineru_file_rejected: 'MinerU 拒绝了文件',
  mineru_service_unavailable: 'MinerU 服务暂时不可用',
  mineru_endpoint_invalid: 'MinerU 服务地址不可用',
  mineru_response_invalid: 'MinerU 返回格式异常',
  mineru_result_expired: 'MinerU 结果已过期',
  mineru_stalled: 'MinerU 长时间无进展',
};

const formatEventTime = (value) => {
  const timestamp = Number(value);
  if (!Number.isFinite(timestamp) || timestamp <= 0) return '';
  try {
    return new Date(timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return '';
  }
};

const splitDiagnosticMessage = (value) => {
  const text = String(value || '').trim();
  const match = text.match(/^(.*?)[:：]\s*`?([A-Za-z][A-Za-z0-9_]+)`?\s*$/);
  if (!match) return { message: text, code: '' };
  return { message: match[1].trim(), code: match[2] };
};

const parseClockSeedMs = (value) => {
  if (value === null || value === undefined || value === '') return 0;
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value > 0 && value < 1e11 ? value * 1000 : value;
  }
  const parsed = Date.parse(String(value));
  return Number.isFinite(parsed) ? parsed : 0;
};

const MinerUElapsedSuffix = ({ active, resetKey, seedMs }) => {
  const elapsedSeconds = useMinerUElapsedClock(active, resetKey, seedMs);
  const label = formatMinerUElapsed(elapsedSeconds);
  if (!label) return null;
  return <>{` · 已耗时 ${label}`}</>;
};

const MinerURunningProgress = ({ item, darkMode }) => {
  const elapsedSeconds = useMinerUElapsedClock(
    true,
    [String(item?.id || ''), String(item?.parseGeneration || '')].join(':'),
    parseClockSeedMs(item?.startedAt),
  );
  const taskProgress = getTaskProgress(
    applyMinerUElapsedToProgress(item?.progress, elapsedSeconds),
  );
  if (!taskProgress) return null;
  return (
    <div className="mt-2 flex items-center gap-2">
      <div
        role="progressbar"
        aria-label={`${item.title}：${taskProgress.ariaLabel}`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={taskProgress.percent}
        className={`relative h-1.5 min-w-0 flex-1 overflow-hidden rounded-full ${darkMode ? 'bg-white/10' : 'bg-[#eee8e3]'}`}
      >
        <div
          className="mineru-progress-fill absolute inset-y-0 left-0 rounded-full bg-[#D97A5D]"
          style={{ transform: `scaleX(${taskProgress.percent / 100})` }}
        />
      </div>
      <span className={`shrink-0 text-[10px] font-semibold tabular-nums ${darkMode ? 'text-[#FFA07A]' : 'text-[#B85F47]'}`}>
        {taskProgress.label}
      </span>
    </div>
  );
};

const BackgroundTaskPanel = memo(({
  items = [],
  autoEnabled,
  onAutoEnabledChange,
  onClose,
  onClearCache,
  canClearCache,
  onRollbackRagIndex,
  canRollbackRagIndex,
  darkMode = false,
}) => {
  const [showMaintenanceMenu, setShowMaintenanceMenu] = useState(false);
  const [expandedTaskIds, setExpandedTaskIds] = useState(() => new Set());
  const summary = useMemo(() => getBackgroundTaskSummary(items), [items]);

  return (
    <motion.div
      onClick={(event) => event.stopPropagation()}
      initial={TASK_PILL_PANEL_MOTION.initial}
      animate={TASK_PILL_PANEL_MOTION.animate}
      exit={TASK_PILL_PANEL_MOTION.exit}
      transition={TASK_PILL_PANEL_MOTION.transition}
      className={`absolute right-0 top-[3.25rem] w-[330px] overflow-hidden rounded-[22px] border shadow-[0_20px_48px_rgba(76,60,43,0.14)] backdrop-blur-xl ${
        darkMode
          ? 'border-white/10 bg-[#1f2329]/95 text-gray-100'
          : 'border-[#E7E1D9] bg-[#fffefd]/95 text-[#24272d]'
      }`}
      role="dialog"
      aria-label="后台任务"
    >
      <div className={`flex items-start justify-between gap-3 border-b px-4 pb-3 pt-4 ${darkMode ? 'border-white/[0.08]' : 'border-[#eee8e3]'}`}>
        <div>
          <div className="text-[14px] font-bold">后台任务</div>
          <div className="mt-1 flex items-center gap-1.5">
            {summary.state !== 'idle' && (
              <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                summary.state === 'failed'
                  ? (darkMode ? 'bg-[#D97A5D]/15 text-[#FFA07A]' : 'bg-[#F8EBE6] text-[#B85F47]')
                  : summary.state === 'running'
                    ? (darkMode ? 'bg-[#D97A5D]/15 text-[#FFA07A]' : 'bg-[#FFF4EF] text-[#B85F47]')
                    : (darkMode ? 'bg-amber-400/15 text-amber-200' : 'bg-[#F6EFE4] text-[#9A7048]')
              }`}>
                {summary.label}
              </span>
            )}
            {summary.state === 'idle' && (
              <span className={`text-[11px] ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                当前没有运行或待处理任务
              </span>
            )}
          </div>
        </div>
        <div className="relative flex items-center gap-1">
          <button
            type="button"
            onClick={() => setShowMaintenanceMenu((current) => !current)}
            aria-label="后台任务更多操作"
            title="更多操作"
            className={`rounded-full p-1.5 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/35 ${darkMode ? 'text-gray-400 hover:bg-white/10 hover:text-gray-200' : 'text-gray-400 hover:bg-gray-100 hover:text-gray-600'}`}
          >
            <MoreHorizontal className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={onClose}
            aria-label="关闭后台任务"
            title="关闭"
            className={`rounded-full p-1.5 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/35 ${darkMode ? 'text-gray-400 hover:bg-white/10 hover:text-gray-200' : 'text-gray-400 hover:bg-gray-100 hover:text-gray-600'}`}
          >
            <X className="h-4 w-4" />
          </button>

          {showMaintenanceMenu && (
            <div className={`absolute right-8 top-9 z-10 w-44 rounded-[12px] border p-1.5 shadow-lg ${darkMode ? 'border-white/10 bg-[#292d34]' : 'border-gray-100 bg-white'}`}>
              {canRollbackRagIndex && onRollbackRagIndex && (
                <button
                  type="button"
                  onClick={() => {
                    setShowMaintenanceMenu(false);
                    onRollbackRagIndex();
                  }}
                  className={`flex w-full items-center gap-2 rounded-[9px] px-3 py-2 text-left text-[11px] font-semibold transition-colors ${darkMode ? 'text-gray-300 hover:bg-white/10' : 'text-gray-600 hover:bg-gray-50'}`}
                >
                  <RotateCcw className="h-3.5 w-3.5" />
                  回退到本地问答索引
                </button>
              )}
              <button
                type="button"
                onClick={() => {
                  setShowMaintenanceMenu(false);
                  onClearCache?.();
                }}
                disabled={!canClearCache}
                className={`flex w-full items-center gap-2 rounded-[9px] px-3 py-2 text-left text-[11px] font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${darkMode ? 'text-gray-300 hover:bg-white/10' : 'text-gray-600 hover:bg-gray-50'}`}
              >
                <Trash2 className="h-3.5 w-3.5" />
                清理当前文档缓存
              </button>
            </div>
          )}
        </div>
      </div>

      <div className={`flex items-center justify-between px-4 py-3 ${darkMode ? 'bg-white/[0.025]' : 'bg-[#fbfaf8]'}`}>
        <div>
          <div className="text-[12px] font-semibold">自动后台处理</div>
          <div className={`mt-0.5 text-[10px] ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>总结、大纲与翻译按当前阅读设置执行</div>
        </div>
        <button
          type="button"
          role="switch"
          aria-checked={autoEnabled}
          aria-label="自动后台处理"
          onClick={() => onAutoEnabledChange?.(!autoEnabled)}
          className={`relative h-6 w-11 shrink-0 rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/35 ${autoEnabled ? 'bg-[#D97A5D]' : darkMode ? 'bg-white/15' : 'bg-gray-200'}`}
        >
          <span className={`absolute left-0 top-1 h-4 w-4 rounded-full bg-white shadow-sm transition-transform ${autoEnabled ? 'translate-x-6' : 'translate-x-1'}`} />
        </button>
      </div>

      <div className={`max-h-[360px] overflow-y-auto px-4 ${darkMode ? 'divide-white/[0.08]' : 'divide-[#eee8e3]'} divide-y`}>
        {items.length === 0 ? (
          <div className="flex flex-col items-center px-4 py-8 text-center">
            <CheckCircle2 className={`h-7 w-7 ${darkMode ? 'text-gray-600' : 'text-gray-300'}`} />
            <div className={`mt-2 text-[12px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>没有后台任务</div>
            <div className={`mt-1 max-w-[220px] text-[10px] leading-4 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
              手动生成请前往总结、大纲、速览或阅读面板
            </div>
          </div>
        ) : items.map((item) => {
          const taskStyle = TASK_STYLES[item.state] || TASK_STYLES.running;
          const TaskIcon = taskStyle.Icon;
          const showAction = item.actionLabel && (!item.busy || item.actionLabel === '取消');
          const taskProgress = getTaskProgress(item.progress);
          const showMinerUScanLoader = item.id === 'deep_parse' && item.state === 'running';
          const events = Array.isArray(item.events) ? item.events : [];
          const shortfallCode = String(item.shortfall?.code || '').trim();
          const hasDetails = events.length > 0 || Boolean(shortfallCode);
          const expanded = expandedTaskIds.has(item.id);
          const diagnostic = splitDiagnosticMessage(item.desc);
          return (
            <div
              key={item.id}
              className={`flex items-start gap-3 py-3.5 ${
                item.state === 'failed'
                  ? `my-2 rounded-[14px] border px-2.5 ${darkMode ? 'border-[#D97A5D]/20 bg-[#D97A5D]/10' : 'border-[#F0DDD6] bg-[#FBF4F1]'}`
                  : ''
              }`}
            >
              {showMinerUScanLoader ? (
                <MinerUScanLoader
                  size={20}
                  className={`mt-0.5 ${darkMode ? 'text-[#FFA07A]' : 'text-[#ed8c68]'}`}
                />
              ) : (
                <TaskIcon className={`mt-0.5 h-4 w-4 shrink-0 ${darkMode ? taskStyle.darkIconClassName : taskStyle.lightIconClassName}`} />
              )}
              <div className="min-w-0 flex-1">
                <div className="flex min-w-0 items-center gap-2">
                  <span className="truncate text-[12px] font-bold">{item.title}</span>
                  <span className={`shrink-0 text-[10px] font-semibold ${darkMode ? taskStyle.darkStatusClassName : taskStyle.lightStatusClassName}`}>{item.status}</span>
                  {hasDetails && (
                    <button
                      type="button"
                      onClick={() => setExpandedTaskIds((previous) => {
                        const next = new Set(previous);
                        if (next.has(item.id)) next.delete(item.id);
                        else next.add(item.id);
                        return next;
                      })}
                      className={`ml-auto shrink-0 rounded-full p-1 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/35 ${darkMode ? 'text-gray-500 hover:bg-white/10 hover:text-gray-200' : 'text-gray-400 hover:bg-gray-100 hover:text-gray-700'}`}
                      aria-expanded={expanded}
                      aria-label={`${expanded ? '收起' : '展开'} ${item.title}处理详情`}
                      title={expanded ? '收起处理详情' : '查看处理详情'}
                    >
                      <ChevronDown className={`h-3.5 w-3.5 transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`} />
                    </button>
                  )}
                </div>
                <div className={`mt-1 text-[10px] leading-4 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  {diagnostic.message}
                  {item.id === 'deep_parse' && item.state === 'running' && (
                    <MinerUElapsedSuffix
                      active
                      resetKey={[
                        String(item.id || ''),
                        String(item.parseGeneration || ''),
                      ].join(':')}
                      seedMs={parseClockSeedMs(item.startedAt)}
                    />
                  )}
                  {diagnostic.code && (
                    <code className={`mt-1.5 block w-fit rounded-[7px] px-1.5 py-0.5 font-mono text-[10px] font-semibold tracking-tight ${
                      darkMode ? 'bg-white/5 text-[#FFC4B0]' : 'bg-white/80 text-[#8A5A48]'
                    }`}>
                      {diagnostic.code}
                    </code>
                  )}
                </div>
                {item.id === 'deep_parse' && item.state === 'running' ? (
                  <MinerURunningProgress item={item} darkMode={darkMode} />
                ) : taskProgress ? (
                  <div className="mt-2 flex items-center gap-2">
                    <div
                      role="progressbar"
                      aria-label={`${item.title}：${taskProgress.ariaLabel}`}
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-valuenow={taskProgress.percent}
                      className={`relative h-1.5 min-w-0 flex-1 overflow-hidden rounded-full ${darkMode ? 'bg-white/10' : 'bg-[#eee8e3]'}`}
                    >
                      <div
                        className="mineru-progress-fill absolute inset-y-0 left-0 rounded-full bg-[#D97A5D]"
                        style={{ transform: `scaleX(${taskProgress.percent / 100})` }}
                      />
                    </div>
                    <span className={`shrink-0 text-[10px] font-semibold tabular-nums ${darkMode ? 'text-[#FFA07A]' : 'text-[#B85F47]'}`}>
                      {taskProgress.label}
                    </span>
                  </div>
                ) : null}
                {expanded && hasDetails && (
                  <div className={`mt-2.5 space-y-1.5 rounded-[10px] border px-2.5 py-2 text-[10px] ${darkMode ? 'border-white/[0.08] bg-white/[0.03] text-gray-400' : 'border-[#eee8e3] bg-[#fbfaf8] text-gray-500'}`}>
                    {shortfallCode && (
                      <div className={`${darkMode ? 'text-amber-200' : 'text-amber-700'} font-semibold`}>
                        {SHORTFALL_LABELS[shortfallCode] || shortfallCode}
                        {Number.isFinite(Number(item.shortfall?.count)) && ` · ${item.shortfall.count} 项`}
                      </div>
                    )}
                    {events.slice(-8).map((event) => {
                      const stage = EVENT_STAGE_LABELS[event?.stage] || event?.stage || '处理中';
                      const eventStatus = event?.status === 'failed'
                        ? '失败'
                        : event?.status === 'cancelled'
                          ? '已取消'
                          : event?.status === 'succeeded'
                            ? '完成'
                            : event?.status === 'partial' || event?.status === 'degraded'
                              ? '降级'
                              : '进行中';
                      const timeLabel = formatEventTime(event?.timestamp);
                      return (
                        <div key={event?.event_id || `${event?.sequence}-${event?.stage}`} className="flex items-center gap-2">
                          <span className="w-4 shrink-0 text-right tabular-nums opacity-60">{event?.sequence || ''}</span>
                          <span className="min-w-0 flex-1 truncate">{stage}</span>
                          <span className="shrink-0 font-medium">{eventStatus}</span>
                          {timeLabel && <span className="shrink-0 tabular-nums opacity-60">{timeLabel}</span>}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
              {showAction && (
                <button
                  type="button"
                  onClick={() => item.onAction?.()}
                  disabled={item.disabled}
                  className={`shrink-0 rounded-full px-2.5 py-1.5 text-[10px] font-semibold transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/35 disabled:cursor-not-allowed disabled:opacity-40 active:scale-[0.98] ${
                    item.state === 'failed'
                      ? 'bg-[#D97A5D] text-white shadow-[0_4px_10px_rgba(160,76,55,0.2)] hover:bg-[#c66b50]'
                      : darkMode
                        ? 'bg-white/[0.07] text-gray-200 hover:bg-white/10'
                        : 'bg-[#fff4ef] text-[#B85F47] hover:bg-[#ffe8df]'
                  }`}
                >
                  {item.actionLabel}
                </button>
              )}
            </div>
          );
        })}
      </div>
    </motion.div>
  );
});

export default BackgroundTaskPanel;
