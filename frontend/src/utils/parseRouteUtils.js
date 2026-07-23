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
} = {}) => {
  const normalizedStatus = String(status || '').trim().toLowerCase();
  if (['queued', 'running'].includes(normalizedStatus)) return true;
  return Boolean(
    primaryMinerURoute
    && routePending
    && !routeFailed
    && !routeCancelled
  );
};

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
  building_index: '正在构建问答索引',
  building_rag_index: '正在构建问答索引',
  awaiting_rag_index: '等待问答索引发布',
  queued: '等待开始解析',
};

export const resolveDocumentParseState = ({ manifest, parseReady, deepParseStatus } = {}) => {
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
  const rawStatus = deepParseOwnsPrimaryRoute
    ? (['failed', 'cancelled'].includes(manifestStatus) ? manifestStatus : deepStatus || manifestStatus)
    : manifestStatus;
  const stage = String(
    manifest?.stage || (deepParseOwnsPrimaryRoute ? deepParseStatus?.stage : '') || ''
  ).trim().toLowerCase();
  const error = String(
    manifest?.error || (deepParseOwnsPrimaryRoute ? deepParseStatus?.error : '') || ''
  ).trim();
  const hasExplicitReadyState = typeof parseReady === 'boolean';
  const ready = hasExplicitReadyState
    ? parseReady
    : (rawStatus === 'ready' || manifest?.status === 'ready') && stage !== 'awaiting_rag_index';

  const routeLabel = requestedRoute === 'auto' && resolvedRoute
    ? `自动 -> ${getParseRouteLabel(resolvedRoute)}`
    : getParseRouteLabel(resolvedRoute || requestedRoute);

  if (rawStatus === 'failed') {
    return {
      requestedRoute,
      resolvedRoute,
      routeLabel,
      state: 'failed',
      statusLabel: '解析失败',
      detail: error || '主解析未完成，请重试或重新选择上传路线',
    };
  }

  if (rawStatus === 'cancelled') {
    return {
      requestedRoute,
      resolvedRoute,
      routeLabel,
      state: 'cancelled',
      statusLabel: '解析已取消',
      detail: '重新上传时可以继续使用 MinerU，或切换为本地解析',
    };
  }

  if (ready) {
    return {
      requestedRoute,
      resolvedRoute,
      routeLabel,
      state: 'ready',
      statusLabel: '全部能力已就绪',
      detail: resolvedRoute === 'mineru' ? '正文、阅读与问答均来自 MinerU' : '正文、阅读与问答使用本地主路线',
    };
  }

  if (stage === 'awaiting_rag_index') {
    return {
      requestedRoute,
      resolvedRoute,
      routeLabel,
      state: 'awaiting_publish',
      statusLabel: '等待问答索引发布',
      detail: '版面解析已完成，索引发布后会统一开放阅读、翻译、速览和问答',
    };
  }

  return {
    requestedRoute,
    resolvedRoute,
    routeLabel,
    state: 'processing',
    statusLabel: STAGE_LABELS[stage] || (resolvedRoute === 'mineru' ? 'MinerU 全程解析中' : '正在准备文档'),
    detail: resolvedRoute === 'mineru'
      ? '完成后将统一开放阅读、翻译、速览和问答'
      : '解析完成后即可开始阅读和提问',
  };
};
