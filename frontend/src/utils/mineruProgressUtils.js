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
  building_rag_index: '正在构建问答索引',
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
  building_index: 87,
  building_rag_index: 93,
  awaiting_rag_index: 96,
};

const clampPercent = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return null;
  return Math.max(0, Math.min(100, Math.round(numeric)));
};

const resolveElapsedSeconds = (task, backendProgress) => {
  const reported = Number(backendProgress?.elapsed_seconds);
  if (Number.isFinite(reported) && reported >= 0) return Math.floor(reported);
  const startedAt = task?.started_at || task?.created_at;
  if (!startedAt) return null;
  const started = new Date(startedAt).getTime();
  if (!Number.isFinite(started)) return null;
  return Math.max(0, Math.floor((Date.now() - started) / 1000));
};

const deriveFallbackPercent = (task, stage, status) => {
  const remotePercent = clampPercent(task?.remote_progress_percent);
  if (stage === 'polling' && remotePercent !== null) {
    return Math.max(24, Math.min(78, Math.round(24 + remotePercent * 0.54)));
  }
  if (stage === 'polling') {
    const attempt = Math.max(0, Number.parseInt(task?.poll_attempt, 10) || 0);
    return Math.max(25, Math.min(75, Math.round(25 + 50 * (1 - Math.exp(-attempt / 23)))));
  }
  if (status === 'ready' || status === 'partial_ready') return 100;
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

export const getMinerUProgressPresentation = (task = {}, fallback = {}) => {
  const status = String(task?.status || fallback?.status || '').trim().toLowerCase();
  const stage = String(task?.stage || fallback?.stage || '').trim().toLowerCase();
  const backendProgress = task?.progress && typeof task.progress === 'object' ? task.progress : null;
  const backendPercent = clampPercent(backendProgress?.percent);
  const remotePercent = clampPercent(backendProgress?.remote_percent ?? task?.remote_progress_percent);
  const percent = backendPercent ?? deriveFallbackPercent(task, stage, status);
  const estimated = backendProgress
    ? backendProgress.estimated !== false
    : remotePercent === null;
  const elapsedSeconds = resolveElapsedSeconds(task, backendProgress);
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
    stageLabel,
    label,
    summaryLabel: `${percent}%`,
    detail,
    ariaLabel: `${stageLabel}，${label}${elapsedLabel ? `，已耗时 ${elapsedLabel}` : ''}`,
  };
};
