export const VALID_PARSE_ROUTES = ['auto', 'local', 'mineru'];
export const DEFAULT_PARSE_ROUTE = 'mineru';

export const PARSE_ROUTE_OPTIONS = [
  {
    value: 'mineru',
    label: 'MinerU 深度解析',
    shortLabel: 'MinerU',
    description: '默认路线，统一生成正文、阅读结构、问答索引、大纲、总结、翻译与速览',
  },
  {
    value: 'local',
    label: '本地解析',
    shortLabel: '本地',
    description: '在本机处理 PDF，首次使用时下载本地解析组件',
  },
];

const ROUTE_LABELS = {
  auto: '自动选择',
  local: '本地解析',
  mineru: 'MinerU 深度解析',
};

const normalizeRoute = (route) => {
  const normalized = String(route || '').trim().toLowerCase();
  return VALID_PARSE_ROUTES.includes(normalized) ? normalized : '';
};

const getStorage = (storage) => {
  if (storage) return storage;
  if (typeof window === 'undefined') return null;
  return window.localStorage;
};

export const getParseRouteLabel = (route, fallback = '自动选择') => (
  ROUTE_LABELS[normalizeRoute(route)] || fallback
);

export const loadStoredParseRoute = (storage) => {
  try {
    const target = getStorage(storage);
    if (!target) return DEFAULT_PARSE_ROUTE;
    const parsed = JSON.parse(target.getItem('ocrSettings') || '{}');
    return normalizeRoute(parsed.parseRoute) === 'local' ? 'local' : DEFAULT_PARSE_ROUTE;
  } catch {
    return DEFAULT_PARSE_ROUTE;
  }
};

export const saveStoredParseRoute = (route, storage) => {
  const normalized = normalizeRoute(route) === 'local' ? 'local' : DEFAULT_PARSE_ROUTE;
  try {
    const target = getStorage(storage);
    if (!target) return normalized;
    const parsed = JSON.parse(target.getItem('ocrSettings') || '{}');
    target.setItem('ocrSettings', JSON.stringify({ ...parsed, parseRoute: normalized }));
  } catch {
    // localStorage 不可用时仍返回已校验的选择，当前上传可以继续使用。
  }
  return normalized;
};

export const shouldPollMinerUStatus = ({
  status,
  primaryMinerURoute = false,
  routePending = false,
  routeFailed = false,
  routeCancelled = false,
  publishFailed = false,
  awaitingPublish = false,
  notStarted = false,
} = {}) => {
  if (publishFailed || awaitingPublish || notStarted) return false;
  const normalizedStatus = String(status || '').trim().toLowerCase();
  if (['queued', 'running'].includes(normalizedStatus)) return true;
  return Boolean(
    primaryMinerURoute
    && routePending
    && !routeFailed
    && !routeCancelled
  );
};

const RAG_PUBLISH_STAGES = new Set([
  'awaiting_rag_index',
  'preparing_rag_index',
  'building_rag_index',
  'rebuilding_rag_index',
  'building_vector_index',
  'validating_vector_index',
  'preparing_semantic_index',
  'building_semantic_index',
  'validating_semantic_index',
  'publishing_rag_index',
]);
const STAGE_LABELS = {
  requesting_upload: '正在准备上传',
  queued_mineru: '等待 MinerU 开始解析',
  waiting_for_slot: '等待 MinerU 处理名额',
  waiting_for_document_lock: '等待文档任务释放',
  resuming: '正在恢复解析任务',
  uploading: '正在上传到 MinerU',
  polling: 'MinerU 正在解析',
  mineru_parsing: 'MinerU 正在解析',
  downloading: '正在下载解析结果',
  retrying_download: '正在重试结果下载',
  building_index: '正在构建问答索引',
  building_vector_index: '正在生成向量索引',
  validating_vector_index: '正在校验向量索引',
  preparing_semantic_index: '正在准备语义索引',
  building_semantic_index: '正在生成语义索引',
  validating_semantic_index: '正在校验语义索引',
  publishing_rag_index: '正在发布问答索引',
  queue_full: '任务排队已满',
  download_failed: '解析结果下载失败',
  mineru_quality_failed: 'MinerU 结果质量校验失败',
  mineru_response_invalid: 'MinerU 返回格式异常',
  building_rag_index: '正在构建问答索引',
  rag_index_failed: '问答索引发布失败',
  stalled: '任务长时间无进展',
  status_sync_failed: '解析状态同步失败',
  awaiting_rag_index: '等待问答索引发布',
  queued: '等待开始解析',
};

export const resolveDocumentParseState = ({ manifest, parseReady, deepParseStatus, ragIndexStatus } = {}) => {
  const requestedRoute = normalizeRoute(manifest?.requested_route || manifest?.route) || DEFAULT_PARSE_ROUTE;
  const resolvedRoute = normalizeRoute(manifest?.resolved_route) || (requestedRoute === 'auto' ? '' : requestedRoute);
  const manifestStatus = String(manifest?.status || '').trim().toLowerCase();
  const deepStatus = String(deepParseStatus?.status || '').trim().toLowerCase();
  const statusManifest = deepParseStatus?.parse_manifest;
  const statusHasIdentity = Boolean(statusManifest?.generation || statusManifest?.source_hash);
  const statusMatchesManifest = !statusHasIdentity || (
    String(statusManifest?.generation || '') === String(manifest?.generation || '')
    && String(statusManifest?.source_hash || '') === String(manifest?.source_hash || '')
  );
  const deepParseOwnsPrimaryRoute = resolvedRoute === 'mineru'
    && Boolean(manifest?.metadata?.full_route)
    && !manifest?.metadata?.legacy_inferred
    && statusMatchesManifest;
  const knownDeepStatuses = new Set(['', 'idle', 'pending', 'queued', 'running', 'ready', 'partial_ready', 'failed', 'cancelled']);
  const deepStatusUnknown = Boolean(
    deepParseOwnsPrimaryRoute
    && deepStatus
    && !knownDeepStatuses.has(deepStatus)
  );
  const rawStatus = deepStatusUnknown
    ? 'failed'
    : deepParseOwnsPrimaryRoute
    ? (['failed', 'cancelled'].includes(manifestStatus) ? manifestStatus : deepStatus || manifestStatus)
    : manifestStatus;
  // A manifest can remain at ``building_rag_index`` while the worker has
  // already published a terminal error (for example after a status-sync
  // failure). Prefer the terminal worker fields so the UI does not show a
  // stale stage or an empty explanation.
  const deepIsTerminal = ['failed', 'cancelled', 'partial_ready', 'ready'].includes(deepStatus);
  const deepIsAwaitingPublish = String(deepParseStatus?.stage || '').trim().toLowerCase() === 'awaiting_rag_index';
  const stage = String(
    deepStatusUnknown
      ? 'status_sync_failed'
      : deepParseOwnsPrimaryRoute && (deepIsTerminal || deepIsAwaitingPublish) && deepParseStatus?.stage
      ? deepParseStatus.stage
      : manifest?.stage || (deepParseOwnsPrimaryRoute ? deepParseStatus?.stage : '') || ''
  ).trim().toLowerCase();
  const error = String(
    deepStatusUnknown
      ? '解析服务返回了无法识别的任务状态，已暂停轮询；请刷新状态或重试'
      : deepParseOwnsPrimaryRoute && (deepIsTerminal || deepIsAwaitingPublish) && deepParseStatus?.error
      ? deepParseStatus.error
      : manifest?.error || (deepParseOwnsPrimaryRoute ? deepParseStatus?.error : '') || ''
  ).trim();
  const errorCode = String(
    deepParseStatus?.error_code
      || manifest?.metadata?.error_code
      || ragIndexStatus?.error_code
      || ''
  ).trim().toLowerCase();
  const ragIndexStatusValue = String(ragIndexStatus?.status || '').trim().toLowerCase();
  const ragIndexFailure = ragIndexStatusValue === 'failed'
    && (
      String(manifest?.stage || '').trim().toLowerCase() === 'awaiting_rag_index'
      || ragIndexStatus?.preserve_parse === true
      || deepParseStatus?.rag_index_failure?.preserve_parse === true
      || manifest?.metadata?.rag_index_failure?.preserve_parse === true
    );
  const hasExplicitReadyState = typeof parseReady === 'boolean';
  const ready = hasExplicitReadyState
    ? parseReady
    : (rawStatus === 'ready' || manifest?.status === 'ready') && stage !== 'awaiting_rag_index';

  const routeLabel = requestedRoute === 'auto' && resolvedRoute
    ? `自动 -> ${getParseRouteLabel(resolvedRoute)}`
    : getParseRouteLabel(resolvedRoute || requestedRoute);

  if (resolvedRoute === 'mineru' && manifest?.metadata?.full_route && ragIndexFailure) {
    return {
      requestedRoute,
      resolvedRoute,
      routeLabel,
      state: 'publish_failed',
      errorCode,
      statusLabel: '问答索引发布失败',
      detail: ragIndexStatus?.error
        || manifest?.metadata?.rag_index_failure?.message
        || 'MinerU 版面解析已完成，但问答索引未发布；修正 Embedding 配置后可重试。',
    };
  }

  if (rawStatus === 'failed') {
    return {
      requestedRoute,
      resolvedRoute,
      routeLabel,
      state: 'failed',
      errorCode,
      statusLabel: stage === 'status_sync_failed'
        ? '状态同步失败'
        : stage === 'stalled'
          ? '任务已停止等待'
          : '解析失败',
      detail: error || (
        stage === 'status_sync_failed'
          ? '解析状态连续同步失败，已暂停轮询；恢复网络后可刷新或重试'
          : stage === 'stalled'
            ? '任务长时间没有进展，已停止等待；请检查网络或服务后重试'
            : '主解析未完成，请重试或重新选择上传路线'
      ),
    };
  }

  if (rawStatus === 'cancelled') {
    return {
      requestedRoute,
      resolvedRoute,
      routeLabel,
      state: 'cancelled',
      errorCode,
      statusLabel: '解析已取消',
      detail: '重新上传时可以继续使用 MinerU，或切换为本地解析',
    };
  }

  if (
    rawStatus === 'pending'
    && resolvedRoute === 'mineru'
    && !['queued', 'running'].includes(deepStatus)
    && !['queued_mineru', 'waiting_for_slot', 'waiting_for_document_lock', 'uploading', 'polling', 'mineru_parsing', 'downloading', 'building_index', 'building_rag_index'].includes(stage)
    && stage !== 'awaiting_rag_index'
  ) {
    return {
      requestedRoute,
      resolvedRoute,
      routeLabel,
      state: 'not_started',
      errorCode,
      statusLabel: '尚未开始解析',
      detail: 'MinerU 解析任务尚未启动，可重新开始解析',
    };
  }

  // 后端在部分页面解析失败但产物仍可发布时给出 partial_ready。能力确实全开，
  // 但覆盖面是残缺的 —— 之前这里没有分支，它要么被当成「全部能力已就绪」（降级
  // 完全不可见），要么落到最后的 processing 分支一直显示「解析中」。
  if (rawStatus === 'partial_ready' || stage === 'partial_ready') {
    return {
      requestedRoute,
      resolvedRoute,
      routeLabel,
      state: 'partial_ready',
      errorCode,
      statusLabel: '部分页面未解析成功',
      detail: error || '阅读与问答已开放，但仅覆盖解析成功的页面',
    };
  }

  if (ready) {
    return {
      requestedRoute,
      resolvedRoute,
      routeLabel,
      state: 'ready',
      errorCode,
      statusLabel: '全部能力已就绪',
      detail: resolvedRoute === 'mineru' ? '正文、阅读与问答均来自 MinerU' : '正文、阅读与问答使用本地主路线',
    };
  }

  if (stage === 'awaiting_rag_index' || RAG_PUBLISH_STAGES.has(stage)) {
    return {
      requestedRoute,
      resolvedRoute,
      routeLabel,
      state: 'awaiting_publish',
      errorCode,
      statusLabel: '等待问答索引发布',
      detail: '版面解析已完成，索引发布后会统一开放阅读、翻译、速览和问答',
    };
  }

  return {
    requestedRoute,
    resolvedRoute,
    routeLabel,
    state: 'processing',
    errorCode,
    statusLabel: STAGE_LABELS[stage] || (resolvedRoute === 'mineru' ? 'MinerU 全程解析中' : '正在准备文档'),
    detail: resolvedRoute === 'mineru'
      ? '完成后将统一开放阅读、翻译、速览和问答'
      : '解析完成后即可开始阅读和提问',
  };
};
