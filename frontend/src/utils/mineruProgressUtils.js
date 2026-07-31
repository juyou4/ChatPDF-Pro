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
};

const STAGE_ESTIMATES = {
  queued: 2,
  waiting_for_slot: 4,
  waiting_for_document_lock: 7,
  requesting_upload: 10,
  uploading: 18,
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

const clampPercent = (value) => {
  if (value === null || value === undefined || value === '') return null;
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return null;
  return Math.max(0, Math.min(100, Math.round(numeric)));
};

const parseTimestamp = (value) => {
  if (!value) return null;
  const timestamp = new Date(value).getTime();
  return Number.isFinite(timestamp) ? timestamp : null;
};

const resolveElapsedSeconds = (task, backendProgress, fallback, nowMs) => {
  const started = parseTimestamp(
    task?.started_at
    || fallback?.started_at
    || task?.created_at
    || fallback?.created_at
  );
  const fromTimestamp = started === null
    ? null
    : Math.max(0, Math.floor((nowMs - started) / 1000));
  const rawReported = backendProgress?.elapsed_seconds;
  const reported = rawReported === null || rawReported === undefined || rawReported === ''
    ? null
    : Number(rawReported);
  const normalizedReported = Number.isFinite(reported) && reported >= 0
    ? Math.floor(reported)
    : null;
  if (fromTimestamp !== null && normalizedReported !== null) {
    return Math.max(fromTimestamp, normalizedReported);
  }
  return fromTimestamp ?? normalizedReported;
};

const deriveStageWindowPercent = (task, fallback, stage, nowMs) => {
  const window = STAGE_ESTIMATE_WINDOWS[stage];
  if (!window) return null;
  const [floor, ceiling, timeConstant] = window;
  const stageStarted = parseTimestamp(
    task?.stage_started_at
    || fallback?.stage_started_at
    || task?.updated_at
    || fallback?.updated_at
  );
  if (stageStarted === null) return floor;
  const stageElapsed = Math.max(0, (nowMs - stageStarted) / 1000);
  const estimate = floor + (ceiling - floor) * (1 - Math.exp(-stageElapsed / timeConstant));
  return clampPercent(Math.max(floor, Math.min(ceiling, estimate)));
};

const deriveFallbackPercent = (task, fallback, stage, status, nowMs) => {
  const remotePercent = clampPercent(task?.remote_progress_percent);
  if (stage === 'polling' && remotePercent !== null) {
    return Math.max(24, Math.min(78, Math.round(24 + remotePercent * 0.54)));
  }
  if (stage === 'polling') {
    const attempt = Math.max(0, Number.parseInt(task?.poll_attempt, 10) || 0);
    return Math.max(25, Math.min(75, Math.round(25 + 50 * (1 - Math.exp(-attempt / 23)))));
  }
  if (status === 'ready' || status === 'partial_ready') return 100;
  const stageWindowPercent = deriveStageWindowPercent(task, fallback, stage, nowMs);
  if (stageWindowPercent !== null) return stageWindowPercent;
  return STAGE_ESTIMATES[stage] ?? (status === 'running' ? 22 : 2);
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
  const liveStagePercent = estimated
    ? deriveStageWindowPercent(task, fallback, stage, nowMs)
    : null;
  const fallbackPercent = deriveFallbackPercent(task, fallback, stage, status, nowMs);
  const percent = clampPercent(Math.max(
    backendPercent ?? fallbackPercent ?? 0,
    liveStagePercent ?? 0
  ));
  const elapsedSeconds = resolveElapsedSeconds(task, backendProgress, fallback, nowMs);
  const stageLabel = STAGE_LABELS[stage] || (status === 'queued' ? '等待开始解析' : 'MinerU 正在解析');
  const elapsedLabel = formatMinerUElapsed(elapsedSeconds);
  const label = remotePercent !== null && !estimated
    ? `远端 ${remotePercent}%`
    : `预估 ${percent}%`;
  const detail = `${stageLabel}${elapsedLabel ? ` · 已耗时 ${elapsedLabel}` : ''}`;

  return {
    percent,
    estimated,
    remotePercent,
    elapsedSeconds,
    elapsedLabel,
    stage,
    stageLabel,
    label,
    summaryLabel: `${percent}%`,
    detail,
    ariaLabel: `${stageLabel}，${label}${elapsedLabel ? `，已耗时 ${elapsedLabel}` : ''}`,
  };
};
