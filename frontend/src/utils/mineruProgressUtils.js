const STAGE_LABELS = {
  queued: '等待开始解析',
  waiting_for_slot: '等待 MinerU 处理名额',
  waiting_for_document_lock: '等待当前文档任务结束',
  requesting_upload: '正在申请上传链接',
  uploading: '正在上传 PDF 到 MinerU',
  resuming: '正在恢复 MinerU 任务',
  resuming_result_download: '正在重新获取解析结果',
  polling: 'MinerU 正在解析',
  mineru_parsing: 'MinerU 正在解析',
  downloading: '正在下载解析结果',
  retrying_download: '正在重试结果下载',
  building_index: '正在构建阅读结构',
  preparing_rag_index: '正在准备问答索引',
  building_rag_index: '正在构建问答索引',
  rebuilding_rag_index: '正在构建问答索引',
  building_vector_index: '正在生成向量索引',
  validating_vector_index: '正在校验向量索引',
  preparing_semantic_index: '正在准备语义索引',
  building_semantic_index: '正在生成语义索引',
  validating_semantic_index: '正在校验语义索引',
  publishing_rag_index: '正在发布问答索引',
  awaiting_rag_index: '等待问答索引发布',
  rag_index_failed: '问答索引发布失败',
  download_failed: '解析结果下载失败',
  queue_full: '任务排队已满',
  start_failed: '任务启动失败',
  mineru_quality_failed: 'MinerU 结果质量校验失败',
  mineru_response_invalid: 'MinerU 返回格式异常',
  status_sync_failed: '解析状态同步失败',
  stalled: '任务长时间无进展',
};

const STAGE_ESTIMATES = {
  queued: 2,
  waiting_for_slot: 4,
  waiting_for_document_lock: 7,
  requesting_upload: 10,
  uploading: 18,
  mineru_parsing: 0,
  resuming: 20,
  resuming_result_download: 80,
  downloading: 82,
  retrying_download: 82,
  building_index: 84,
  preparing_rag_index: 88,
  building_rag_index: 89,
  rebuilding_rag_index: 89,
  building_vector_index: 89,
  validating_vector_index: 93,
  preparing_semantic_index: 94,
  building_semantic_index: 95,
  validating_semantic_index: 97,
  publishing_rag_index: 98,
  awaiting_rag_index: 96,
};

const STAGE_ESTIMATE_WINDOWS = {
  building_index: [84, 87, 18],
  building_rag_index: [89, 92, 35],
  rebuilding_rag_index: [89, 92, 35],
  building_vector_index: [89, 92, 35],
  building_semantic_index: [95, 96, 30],
  publishing_rag_index: [98, 99, 12],
};

// MinerU 远端解析可能十几分钟都停在同一 stage。不能用 updated_at 当窗口起点，
// 轮询每次改这个字段会把进度条打回 floor，看起来像卡片在抖。
// 预估从 0 开始爬升，mineru_parsing / polling 用同一条曲线，避免两阶段来回跳。
const LONG_RUNNING_PARSE_WINDOW = [0, 78, 180];
const LONG_RUNNING_PARSE_STAGES = {
  mineru_parsing: LONG_RUNNING_PARSE_WINDOW,
  polling: LONG_RUNNING_PARSE_WINDOW,
};

const LIVE_STATUSES = new Set(['queued', 'running', 'processing', 'pending', 'waiting']);
const RAG_INDEX_BUILD_STAGES = new Set([
  'building_rag_index',
  'rebuilding_rag_index',
  'building_vector_index',
]);
const liveElapsedAnchors = new Map();
const liveStageAnchors = new Map();
const livePercentHighWater = new Map();

export const resetMinerUElapsedAnchors = () => {
  liveElapsedAnchors.clear();
  liveStageAnchors.clear();
  livePercentHighWater.clear();
};

export const canonicalizeMinerUProgressStage = (stage) => {
  const value = String(stage || '').trim().toLowerCase();
  if (!value || value === 'polling' || value === 'mineru_parsing') return 'mineru_parsing';
  if (RAG_INDEX_BUILD_STAGES.has(value)) return 'building_rag_index';
  return value;
};

const isLongRunningParseStage = (stage, status = '') => {
  const normalized = String(stage || '').trim().toLowerCase();
  if (LONG_RUNNING_PARSE_STAGES[normalized]) return true;
  return !normalized && LIVE_STATUSES.has(String(status || '').trim().toLowerCase());
};

const applyHighWaterPercent = (key, percent) => {
  if (!key || percent === null || percent === undefined) return percent;
  const previous = livePercentHighWater.get(key);
  const next = previous === undefined ? percent : Math.max(previous, percent);
  livePercentHighWater.set(key, next);
  return next;
};

const resolveParseGeneration = (task, fallback) => String(
  fallback?.parse_generation
  || fallback?.parse_manifest?.generation
  || task?.parse_generation
  || task?.parse_manifest?.generation
  || ''
).trim();

const isStaleParseTask = (task, fallback) => {
  const fallbackGeneration = String(
    fallback?.parse_generation
    || fallback?.parse_manifest?.generation
    || ''
  ).trim();
  if (!fallbackGeneration) return false;
  const taskGeneration = String(
    task?.parse_generation
    || task?.parse_manifest?.generation
    || ''
  ).trim();
  return taskGeneration !== fallbackGeneration;
};

const resolveLiveAnchorKey = (task, fallback) => {
  const generation = resolveParseGeneration(task, fallback);
  const docId = String(task?.doc_id || fallback?.doc_id || '').trim();
  const jobId = String(task?.job_id || task?.task_id || fallback?.job_id || '').trim();
  if (docId && generation) return `${docId}:${generation}`;
  if (generation) return `gen:${generation}`;
  if (docId && jobId) return `${docId}:${jobId}`;
  if (jobId) return `job:${jobId}`;
  if (docId) return `doc:${docId}`;
  return '';
};

const clampPercent = (value) => {
  if (value === null || value === undefined || value === '') return null;
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return null;
  return Math.max(0, Math.min(100, Math.round(numeric)));
};

const parseTimestamp = (value) => {
  if (value === null || value === undefined || value === '') return null;
  if (typeof value === 'number' && Number.isFinite(value)) {
    const millis = value > 0 && value < 1e11 ? value * 1000 : value;
    return Number.isFinite(millis) ? millis : null;
  }
  const text = String(value).trim();
  if (!text) return null;
  // 截断到毫秒：`2026-08-17T01:41:22.123456` 在部分浏览器是 Invalid Date。
  const normalized = text
    .replace(' ', 'T')
    .replace(/(\.\d{3})\d+/, '$1');
  const timestamp = Date.parse(normalized);
  return Number.isFinite(timestamp) ? timestamp : null;
};

const resolveLiveElapsedSeconds = (task, fallback, nowMs) => {
  const status = String(task?.status || fallback?.status || '').trim().toLowerCase();
  const key = resolveLiveAnchorKey(task, fallback);
  if (!key || !LIVE_STATUSES.has(status)) {
    if (key) liveElapsedAnchors.delete(key);
    return null;
  }
  const staleTask = isStaleParseTask(task, fallback);
  const runStarted = parseTimestamp(
    staleTask
      ? (fallback?.started_at || fallback?.created_at)
      : (
        task?.started_at
        || fallback?.started_at
        || task?.created_at
        || fallback?.created_at
      )
  );
  const validRunStarted = runStarted !== null && runStarted <= nowMs + 2000
    ? runStarted
    : null;
  const previous = liveElapsedAnchors.get(key);
  // 上一次解析的锚点如果早于这次 started_at，必须丢掉，否则会接着上次的秒数走。
  if (!Number.isFinite(previous) || (validRunStarted !== null && previous < validRunStarted - 1500)) {
    const startedAt = validRunStarted !== null ? Math.min(validRunStarted, nowMs) : nowMs;
    liveElapsedAnchors.set(key, startedAt);
    return Math.max(0, Math.floor((nowMs - startedAt) / 1000));
  }
  if (nowMs < previous) {
    return 0;
  }
  return Math.max(0, Math.floor((nowMs - previous) / 1000));
};

const resolveElapsedSeconds = (task, backendProgress, fallback, nowMs) => {
  const staleTask = isStaleParseTask(task, fallback);
  const started = parseTimestamp(
    staleTask ? fallback?.started_at : (task?.started_at || fallback?.started_at)
  );
  const created = parseTimestamp(
    staleTask ? fallback?.created_at : (task?.created_at || fallback?.created_at)
  );
  const timestamp = started !== null
    ? started
    : (created !== null && nowMs - created <= 120000 && created <= nowMs + 2000 ? created : null);
  const fromTimestamp = timestamp === null || timestamp > nowMs + 2000
    ? null
    : Math.max(0, Math.floor((nowMs - timestamp) / 1000));
  const rawReported = staleTask ? null : backendProgress?.elapsed_seconds;
  const reported = rawReported === null || rawReported === undefined || rawReported === ''
    ? null
    : Number(rawReported);
  let normalizedReported = Number.isFinite(reported) && reported >= 0
    ? Math.floor(reported)
    : null;
  if (fromTimestamp !== null && normalizedReported !== null && normalizedReported > fromTimestamp + 8) {
    normalizedReported = null;
  }
  const liveElapsed = resolveLiveElapsedSeconds(task, fallback, nowMs);
  const candidates = [fromTimestamp, normalizedReported, liveElapsed]
    .filter((value) => value !== null && value !== undefined);
  if (!candidates.length) return null;
  return Math.max(...candidates);
};

const resolveStageWindow = (stage) => {
  const normalized = String(stage || '').trim().toLowerCase();
  return STAGE_ESTIMATE_WINDOWS[normalized]
    || STAGE_ESTIMATE_WINDOWS[canonicalizeMinerUProgressStage(normalized)]
    || null;
};

const resolveStageElapsedSeconds = (task, fallback, stage, nowMs) => {
  const canonicalStage = canonicalizeMinerUProgressStage(stage);
  const reportedStageStarted = parseTimestamp(
    task?.stage_started_at || fallback?.stage_started_at
  );
  const validReported = reportedStageStarted !== null && reportedStageStarted <= nowMs + 2000
    ? Math.min(reportedStageStarted, nowMs)
    : null;
  const parseKey = resolveLiveAnchorKey(task, fallback);
  const key = parseKey ? `${parseKey}:${canonicalStage}` : '';
  let startedAt = validReported;
  if (key) {
    const previous = liveStageAnchors.get(key);
    if (startedAt === null) {
      startedAt = Number.isFinite(previous) ? previous : nowMs;
    } else if (Number.isFinite(previous)) {
      // 轮询可能把 updated_at 写进 stage_started_at。只允许锚点前移到更早，不能被刷新成“现在”。
      startedAt = Math.min(previous, startedAt);
    }
    liveStageAnchors.set(key, startedAt);
  }
  if (startedAt === null) return 0;
  return Math.max(0, (nowMs - startedAt) / 1000);
};

export const estimateStageWindowPercent = (stage, stageElapsedSeconds) => {
  const window = resolveStageWindow(stage);
  if (!window) return null;
  const [floor, ceiling, timeConstant] = window;
  const elapsed = Math.max(0, Number(stageElapsedSeconds) || 0);
  const estimate = floor + (ceiling - floor) * (1 - Math.exp(-elapsed / timeConstant));
  return clampPercent(Math.max(floor, Math.min(ceiling, estimate)));
};

const deriveStageWindowPercent = (task, fallback, stage, nowMs) => {
  if (!resolveStageWindow(stage)) return null;
  return estimateStageWindowPercent(
    stage,
    resolveStageElapsedSeconds(task, fallback, stage, nowMs),
  );
};

const deriveLongRunningParsePercent = (task, fallback, stage, nowMs) => {
  const window = LONG_RUNNING_PARSE_STAGES[stage];
  if (!window) return null;
  const [floor, ceiling, timeConstant] = window;
  const elapsed = resolveElapsedSeconds(
    task,
    task?.progress && typeof task.progress === 'object' ? task.progress : null,
    fallback,
    nowMs,
  ) ?? 0;
  const estimate = floor + (ceiling - floor) * (1 - Math.exp(-elapsed / timeConstant));
  return clampPercent(Math.max(floor, Math.min(ceiling, estimate)));
};

const deriveFallbackPercent = (task, fallback, stage, status, nowMs) => {
  const remotePercent = clampPercent(task?.remote_progress_percent);
  if (stage === 'polling' && remotePercent !== null) {
    return Math.max(0, Math.min(78, Math.round(remotePercent * 0.78)));
  }
  if (stage === 'polling') {
    const attempt = Math.max(0, Number.parseInt(task?.poll_attempt, 10) || 0);
    const attemptPercent = Math.max(0, Math.min(75, Math.round(50 * (1 - Math.exp(-attempt / 23)))));
    return Math.max(attemptPercent, deriveLongRunningParsePercent(task, fallback, stage, nowMs) ?? attemptPercent);
  }
  if (status === 'ready' || status === 'partial_ready') return 100;
  const longRunningPercent = deriveLongRunningParsePercent(task, fallback, stage, nowMs);
  if (longRunningPercent !== null) return longRunningPercent;
  const stageWindowPercent = deriveStageWindowPercent(task, fallback, stage, nowMs);
  if (stageWindowPercent !== null) return stageWindowPercent;
  return STAGE_ESTIMATES[stage] ?? (status === 'running' ? 0 : 2);
};

export const estimateLongRunningParsePercent = (elapsedSeconds) => {
  const [, ceiling, timeConstant] = LONG_RUNNING_PARSE_WINDOW;
  const elapsed = Math.max(0, Number(elapsedSeconds) || 0);
  const estimate = ceiling * (1 - Math.exp(-elapsed / timeConstant));
  return clampPercent(Math.max(0, Math.min(ceiling, estimate)));
};

export const applyMinerUElapsedToProgress = (progress, elapsedSeconds) => {
  if (!progress || typeof progress !== 'object') return progress;
  const elapsed = Number.isFinite(Number(elapsedSeconds))
    ? Math.max(0, Math.floor(Number(elapsedSeconds)))
    : progress.elapsedSeconds;
  const elapsedLabel = formatMinerUElapsed(elapsed);
  const stageLabel = progress.stageLabel || 'MinerU 正在解析';
  const longRunning = isLongRunningParseStage(progress.stage, 'running');
  const snapshotElapsed = Number(progress.elapsedSeconds);
  const snapshotStageElapsed = Number(progress.stageElapsedSeconds);
  const elapsedDelta = Number.isFinite(snapshotElapsed)
    ? Math.max(0, Number(elapsed) - snapshotElapsed)
    : 0;
  const stageElapsed = (
    Number.isFinite(snapshotStageElapsed) ? snapshotStageElapsed : 0
  ) + elapsedDelta;
  const windowPercent = progress.estimated !== false
    ? estimateStageWindowPercent(progress.stage, stageElapsed)
    : null;
  const percent = longRunning && progress.estimated !== false
    ? estimateLongRunningParsePercent(elapsed)
    : windowPercent === null
      ? progress.percent
      : clampPercent(Math.max(progress.percent ?? 0, windowPercent));
  const label = progress.estimated === false && progress.remotePercent != null
    ? `远端 ${progress.remotePercent}%`
    : `预估 ${percent}%`;
  return {
    ...progress,
    percent,
    elapsedSeconds: elapsed,
    stageElapsedSeconds: Math.max(0, Math.floor(stageElapsed)),
    elapsedLabel,
    label,
    summaryLabel: `${percent}%`,
    detail: `${stageLabel}${elapsedLabel ? ` · 已耗时 ${elapsedLabel}` : ''}`,
    ariaLabel: `${stageLabel}，${label}${elapsedLabel ? `，已耗时 ${elapsedLabel}` : ''}`,
  };
};

export const formatMinerUElapsed = (seconds) => {
  const value = Number(seconds);
  if (!Number.isFinite(value) || value < 0) return '';
  const total = Math.floor(value);
  if (total < 60) return `${total} 秒`;
  const minutes = Math.floor(total / 60);
  const remaining = total % 60;
  if (minutes < 60) return remaining > 0 ? `${minutes} 分 ${remaining} 秒` : `${minutes} 分`;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return remainingMinutes > 0 ? `${hours} 小时 ${remainingMinutes} 分` : `${hours} 小时`;
};

export const getMinerUProgressPresentation = (task = {}, fallback = {}, nowMs = Date.now()) => {
  const status = String(task?.status || fallback?.status || '').trim().toLowerCase();
  const backendProgress = task?.progress && typeof task.progress === 'object' ? task.progress : null;
  const stage = String(task?.stage || backendProgress?.stage || fallback?.stage || '').trim().toLowerCase();
  const backendPercent = clampPercent(backendProgress?.percent);
  const remotePercent = clampPercent(backendProgress?.remote_percent ?? task?.remote_progress_percent);
  const estimated = backendProgress
    ? backendProgress.estimated !== false
    : remotePercent === null;
  const longRunning = isLongRunningParseStage(stage, status);
  const liveStagePercent = estimated && !longRunning
    ? deriveStageWindowPercent(task, fallback, stage, nowMs)
    : null;
  const fallbackPercent = deriveFallbackPercent(task, fallback, stage, status, nowMs);
  // 旧后端仍可能把 mineru_parsing 钉在 22%。预估长阶段只信本地耗时曲线，否则永远从 22% 起跳。
  const estimatedBackendPercent = estimated && longRunning ? null : backendPercent;
  let percent = estimated
    ? clampPercent(Math.max(estimatedBackendPercent ?? 0, fallbackPercent ?? 0, liveStagePercent ?? 0))
    : clampPercent(backendPercent ?? fallbackPercent ?? liveStagePercent ?? 0);
  const elapsedSeconds = resolveElapsedSeconds(task, backendProgress, fallback, nowMs);
  const stageElapsedSeconds = resolveStageWindow(stage)
    ? Math.max(0, Math.floor(resolveStageElapsedSeconds(task, fallback, stage, nowMs)))
    : elapsedSeconds;
  const anchorKey = resolveLiveAnchorKey(task, fallback);
  const highWaterKey = anchorKey
    ? `${anchorKey}:${canonicalizeMinerUProgressStage(stage)}`
    : '';
  if (LIVE_STATUSES.has(status) && highWaterKey) {
    percent = applyHighWaterPercent(highWaterKey, percent);
  }
  const stageLabel = STAGE_LABELS[stage] || (status === 'queued' ? '等待开始解析' : 'MinerU 正在解析');
  const elapsedLabel = formatMinerUElapsed(elapsedSeconds);
  if (['failed', 'cancelled'].includes(status) || (
    stage === 'awaiting_rag_index'
    && task?.rag_index?.status === 'failed'
  )) {
    return {
      percent: null,
      estimated: false,
      remotePercent: null,
      elapsedSeconds,
      stageElapsedSeconds: 0,
      elapsedLabel,
      stage,
      stageLabel,
      label: status === 'cancelled' ? '已取消' : '已失败',
      summaryLabel: status === 'cancelled' ? '已取消' : '失败',
      detail: stageLabel,
      ariaLabel: `${stageLabel}，${status === 'cancelled' ? '已取消' : '已失败'}`,
    };
  }
  const label = remotePercent !== null && !estimated
    ? `远端 ${remotePercent}%`
    : `预估 ${percent}%`;
  const detail = `${stageLabel}${elapsedLabel ? ` · 已耗时 ${elapsedLabel}` : ''}`;

  return {
    percent,
    estimated,
    remotePercent,
    elapsedSeconds,
    stageElapsedSeconds,
    elapsedLabel,
    stage,
    stageLabel,
    label,
    summaryLabel: `${percent}%`,
    detail,
    ariaLabel: `${stageLabel}，${label}${elapsedLabel ? `，已耗时 ${elapsedLabel}` : ''}`,
  };
};
