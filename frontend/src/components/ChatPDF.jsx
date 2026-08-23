import React, { useState, useRef, useEffect, useMemo, useCallback, lazy, Suspense } from 'react';
import { createPortal } from 'react-dom';
import {
  ArrowRight,
  ArrowUpDown,
  ArrowUpRight,
  Brain,
  Check,
  ChevronDown,
  ChevronRight,
  Copy,
  Database,
  Files,
  Globe,
  GripVertical,
  History,
  ListFilter,
  Loader2,
  Menu,
  MessageSquare,
  Moon,
  Paperclip,
  Plus,
  RefreshCw,
  Scan,
  ScanText,
  Settings,
  SlidersHorizontal,
  Sparkles,
  Sun,
  Trash2,
  Type,
  Upload,
  X,
} from 'lucide-react';
import { motion, AnimatePresence, useReducedMotion } from 'framer-motion';
import { MorphIcon } from 'morphicons/react';
import { ArrowUp as ArrowUpIcon, Pause as PauseIcon } from 'lucide';
import pdfFiletypeIcon from '../assets/images/pdf-filetype.svg';
import { supportsVision } from '../utils/visionDetectorUtils';
import ScreenshotPreview from './ScreenshotPreview';
import TextSelectionToolbar from './TextSelectionToolbar';
import ChatMessageRow from './ChatMessageRow';
import PdfWorkspacePane, { preloadPDFViewer } from './PdfWorkspacePane';
import DocumentParseStatusBar from './DocumentParseStatusBar';
import { useProvider } from '../contexts/ProviderContext';
import { useModel } from '../contexts/ModelContext';
import { useDefaults } from '../contexts/DefaultsContext';
import { useCapabilities } from '../contexts/CapabilitiesContext';
const loadEmbeddingSettings = () => import('./EmbeddingSettings');
const preloadEmbeddingSettings = () => { void loadEmbeddingSettings().catch(() => {}); };
const EmbeddingSettings = lazy(loadEmbeddingSettings);
const OCRSettingsPanel = lazy(() => import('./OCRSettingsPanel'));
const GlobalSettings = lazy(() => import('./GlobalSettings'));
import { TriStateToggle, VisualVerificationMode } from './RetrievalTuningControls';
const ChatSettings = lazy(() => import('./ChatSettings'));
const OverviewPanel = lazy(() => import('./OverviewPanel'));
import { useGlobalSettings } from '../contexts/GlobalSettingsContext';
import { useChatParams } from '../contexts/ChatParamsContext';
import { useReadingSettings } from '../contexts/ReadingSettingsContext';
import { useDebouncedLocalStorage } from '../hooks/useDebouncedLocalStorage';
import { useUIState } from '../hooks/useUIState';
import { useDocumentState } from '../hooks/useDocumentState';
import { useMessageState } from '../hooks/useMessageState';
import { mergeRecordIfChanged, startVisiblePoll } from '../utils/visiblePoll';
import { usePDFState } from '../hooks/usePDFState';
import { useScreenshotState } from '../hooks/useScreenshotState';
import PresetQuestions from './PresetQuestions';
import ModelQuickSwitch from './ModelQuickSwitch';
import ChatContextIndicator from './ChatContextIndicator';
import VirtualMessageList from './VirtualMessageList';
import WebSearchButton from './WebSearchButton';
import DocumentOutline from './DocumentOutline';
import ReadingAnalysisPanel from './ReadingAnalysisPanel';
import ReadingSummaryPanel from './ReadingSummaryPanel';
import SettingsSegmentedControl from './SettingsSegmentedControl';
import SettingsRange from './SettingsRange';
import ParseRouteSelect from './ParseRouteSelect';
import LocalParserInstallDialog from './LocalParserInstallDialog';
import SessionDeleteDialog from './SessionDeleteDialog';
import ConfirmDialog, { useConfirmDialog } from './ConfirmDialog';
import BackgroundTaskPanel, {
  getBackgroundTaskSummary,
  getVisibleBackgroundTasks,
  stabilizeBackgroundTaskItems,
} from './BackgroundTaskPanel';
import {
  loadStoredParseRoute,
  resolveDocumentParseState,
  saveStoredParseRoute,
  shouldPollMinerUStatus,
} from '../utils/parseRouteUtils';
import { getMinerUProgressPresentation } from '../utils/mineruProgressUtils';
import { stabilizeLiveParseStatus } from '../utils/chatMessageRowMemo';
import { shouldStreamAssistantContent } from '../utils/messageRenderUtils';
import {
  blockIndexMatchesParseContext,
  buildPretranslateAutoIdentity,
  executePretranslateBatches,
  selectPendingPretranslateBlocks,
  shouldForcePretranslateRequest,
} from '../utils/pretranslateUtils';
import {
  createDocumentNote,
  createDocumentHighlight,
  DEFAULT_DOCUMENT_HIGHLIGHT_COLOR,
  getDocumentHighlightFingerprint,
  normalizeDocumentHighlightColor,
  normalizeDocumentHighlightStyle,
  readDocumentHighlights,
  readDocumentNotes,
  writeDocumentHighlights,
  writeDocumentNotes,
} from '../utils/documentHighlightUtils';
import {
  buildSelectionSearchUrl,
  openExternalHttpUrl,
  writePlainTextToClipboard,
} from '../utils/selectionToolUtils';
import {
  hasSelectionText,
  normalizePdfSelection,
} from '../utils/pdfSelectionUtils';

const isLoopbackApiHost = (value) => {
  const input = String(value || '').trim();
  if (!input || input.startsWith('/')) return true;
  try {
    const parsed = new URL(input.includes('://') ? input : `http://${input}`);
    const host = parsed.hostname.toLowerCase().replace(/^\[|\]$/g, '').replace(/\.$/, '');
    return host === 'localhost'
      || host.endsWith('.localhost')
      || host === '::1'
      || host === '::'
      || host === '0.0.0.0'
      || host.startsWith('127.');
  } catch {
    return false;
  }
};

const KEYLESS_LOCAL_PROVIDERS = new Set(['local', 'ollama']);
const EMPTY_ID_LIST = Object.freeze([]);
const isKeylessLocalProvider = (providerId) => KEYLESS_LOCAL_PROVIDERS.has(
  String(providerId || '').trim().toLowerCase()
);
const getMissingEmbeddingApiKeyMessage = (providerName) => `请先为 ${providerName || '当前 Embedding Provider'} 配置 Embedding API Key`;

const buildProviderApiEndpoint = (apiHost, endpointPath) => {
  const rawEndpoint = String(endpointPath || '').trim();
  if (!rawEndpoint) return '';
  if (/^https?:\/\//i.test(rawEndpoint)) return rawEndpoint;
  const rawHost = String(apiHost || '').trim();
  if (!rawHost) return '';
  const normalizedHost = rawHost.replace(/\/+$/, '');
  const normalizedPath = `/${rawEndpoint.replace(/^\/+/, '')}`;
  try {
    const hostUrl = new URL(normalizedHost);
    const hostPath = hostUrl.pathname.replace(/\/+$/, '');
    if (hostPath && (hostPath === normalizedPath || hostPath.endsWith(normalizedPath))) {
      return normalizedHost;
    }
    if (hostPath && normalizedPath.startsWith(`${hostPath}/`)) {
      hostUrl.pathname = normalizedPath;
      return hostUrl.toString().replace(/\/$/, '');
    }
  } catch {
    // 保留旧配置的字符串拼接兜底，真正请求仍由后端安全校验。
  }
  return `${normalizedHost}${normalizedPath}`;
};

const getUsageTokenSummary = (usage) => {
  if (!usage || typeof usage !== 'object') return null;
  const prompt = usage.prompt_tokens ?? usage.input_tokens ?? usage.promptTokenCount ?? usage.inputTokenCount ?? null;
  const completion = usage.completion_tokens ?? usage.output_tokens ?? usage.candidatesTokenCount ?? usage.outputTokenCount ?? null;
  const total = usage.total_tokens ?? usage.totalTokenCount ?? (
    Number.isFinite(prompt) && Number.isFinite(completion) ? prompt + completion : null
  );
  if (!Number.isFinite(total) && !Number.isFinite(prompt) && !Number.isFinite(completion)) return null;
  return { prompt, completion, total, estimated: Boolean(usage.estimated), cost: usage.cost || null };
};

const SendPauseIconSwap = ({ isPaused }) => (
  <span className="chat-send-icon-swap" data-state={isPaused ? 'pause' : 'send'} aria-hidden="true">
    <MorphIcon
      icon={isPaused ? PauseIcon : ArrowUpIcon}
      size={16}
      strokeWidth={2.15}
      spring="snappy"
      reducedMotion="user"
    />
  </span>
);

const SummaryIcon = ({ className = '' }) => (
  <svg
    className={className}
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <path d="M15 4H7" />
    <path d="m18 16 3 3-3 3" />
    <path d="M3 4v13a2 2 0 0 0 2 2h16" />
    <path d="M7 14h7" />
    <path d="M7 9h12" />
  </svg>
);

// GraphRAG 构建/查询端点使用本常量。其他 fetch 依赖均已在各自 hooks 内部维护同名常量。
const API_BASE_URL = '';
const SIDEBAR_DEFAULT_WIDTH = 280;
const SIDEBAR_MIN_WIDTH = 240;
const SIDEBAR_MAX_WIDTH = 420;
const MAIN_PANEL_MIN_WIDTH = 780;
const SIDEBAR_KEYBOARD_STEP = 16;

const UPLOAD_STATUS_META = {
  uploading: {
    label: 'Uploading',
    title: '正在上传文档...',
    desc: '文件传输完成后会进入本地解析和索引阶段',
  },
  extracting: {
    label: 'Extracting',
    title: '正在提取 PDF 文本...',
    desc: '检测页面文本、OCR 和版面清理，这一步耗时取决于页数',
  },
  indexing: {
    label: 'Preparing',
    title: '正在准备阅读视图...',
    desc: '整理页面文本和结构信息，检索索引会在后台继续完成',
  },
  finalizing: {
    label: 'Finalizing',
    title: '正在保存解析结果...',
    desc: '马上进入阅读界面，增强索引会后台准备',
  },
  loading_document: {
    label: 'Loading',
    title: '正在加载文档信息...',
    desc: '即将打开阅读界面',
  },
  accepted: {
    label: 'Accepted',
    title: '文件已接收，MinerU 解析已开始',
    desc: '100% 表示文件传输完成；主解析进度会在工作区继续显示',
  },
  ready: {
    label: 'Ready',
    title: '文档准备完成',
    desc: '正在进入阅读界面',
  },
};

const formatUploadFileSize = (bytes) => {
  const size = Number(bytes);
  if (!Number.isFinite(size) || size <= 0) return '';
  if (size >= 1024 * 1024) return `${(size / (1024 * 1024)).toFixed(1)} MB`;
  if (size >= 1024) return `${Math.round(size / 1024)} KB`;
  return `${Math.round(size)} B`;
};

export const UploadDocumentCard = ({
  darkMode,
  isUploading,
  uploadProgress,
  uploadStatus,
  uploadStatusMeta,
  uploadFileInfo,
  parseRoute,
  onParseRouteChange,
  onSelect,
  onWarmup,
}) => {
  const progress = Math.max(0, Math.min(100, Math.round(Number(uploadProgress) || 0)));
  const fileSize = formatUploadFileSize(uploadFileInfo?.size);
  const isReady = uploadStatus === 'ready';

  return (
    <AnimatePresence initial={false} mode="wait">
      {isUploading ? (
        <motion.div
          key="upload-progress"
          role="status"
          aria-live="polite"
          aria-label={`${uploadStatusMeta.title} ${progress}%`}
          title={uploadStatusMeta.desc}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -4 }}
          transition={{ duration: 0.24, ease: [0.22, 1, 0.36, 1] }}
          className={`upload-document-card relative min-h-[108px] w-full overflow-hidden rounded-[24px] border ${
            darkMode
              ? 'border-white/[0.08] bg-[#24272d]'
              : 'border-[#eadfd8] bg-white'
          }`}
        >
          <motion.div
            aria-hidden="true"
            className={`absolute inset-y-0 left-0 w-full origin-left ${darkMode ? 'bg-[#FFA07A]/[0.08]' : 'bg-[#FFF4EF]'}`}
            initial={false}
            animate={{ scaleX: progress / 100 }}
            transition={{ duration: 0.45, ease: 'linear' }}
          />
          <div
            aria-hidden="true"
            className={`absolute bottom-4 left-5 right-5 z-20 h-[5px] overflow-hidden rounded-full ${
              darkMode ? 'bg-white/[0.10]' : 'bg-[#ECE8E5]'
            }`}
          >
            <motion.div
              className={`absolute inset-y-0 -left-5 w-[calc(100%+40px)] origin-left ${darkMode ? 'bg-[#FFA07A]' : 'bg-[#D97A5D]'}`}
              initial={false}
              animate={{ scaleX: progress / 100 }}
              transition={{ duration: 0.45, ease: 'linear' }}
            />
          </div>

          <div className="relative z-10 flex min-h-[108px] items-center gap-4 px-5 pb-8 pt-4">
            <img
              src={pdfFiletypeIcon}
              alt=""
              aria-hidden="true"
              draggable="false"
              className="h-12 w-12 shrink-0 select-none"
            />

            <div className="min-w-0 flex-1">
              <div className={`truncate text-[15px] font-semibold ${darkMode ? 'text-gray-100' : 'text-gray-900'}`}>
                {uploadFileInfo?.name || 'PDF 文档'}
              </div>
              <div className="mt-1 flex min-h-[18px] items-center gap-1.5 text-[12px]">
                <AnimatePresence initial={false} mode="wait">
                  <motion.span
                    key={uploadStatus}
                    initial={{ opacity: 0, y: 7 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -7 }}
                    transition={{ duration: 0.2 }}
                    className={isReady ? 'font-medium text-emerald-600 dark:text-emerald-300' : 'font-medium text-[#B85F47] dark:text-[#FFA07A]'}
                  >
                    {uploadStatusMeta.title}
                  </motion.span>
                </AnimatePresence>
                {fileSize && (
                  <>
                    <span className={darkMode ? 'text-gray-600' : 'text-gray-300'}>•</span>
                    <span className={darkMode ? 'text-gray-400' : 'text-gray-500'}>{fileSize}</span>
                  </>
                )}
              </div>
            </div>

            <div className="flex min-w-[72px] shrink-0 flex-col items-end">
              <span className={`text-[17px] font-semibold tabular-nums ${darkMode ? 'text-gray-100' : 'text-gray-800'}`}>{progress}%</span>
              <span className={`mt-0.5 text-[9px] font-semibold uppercase tracking-[0.12em] ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                {uploadStatusMeta.label}
              </span>
            </div>
          </div>
        </motion.div>
      ) : (
        <motion.div
          key="upload-idle"
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -4 }}
          onPointerEnter={onWarmup}
          onFocusCapture={onWarmup}
          className={`upload-document-card flex min-h-[108px] w-full items-center gap-4 rounded-[24px] p-5 text-left transition-[box-shadow,background-color] ${
            darkMode
              ? 'border border-white/[0.08] bg-white/[0.055]'
              : 'bg-white'
          }`}
        >
          <div className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-full ${darkMode ? 'bg-[#FFA07A]/10 text-[#FFA07A]' : 'bg-[#FFF4EF] text-[#D97A5D]'}`}>
            <Upload className="h-5 w-5" />
          </div>
          <div className="min-w-0 flex-1">
            <button
              type="button"
              onClick={onSelect}
              className="block max-w-full text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/30"
            >
              <span className={`block text-[15px] font-bold ${darkMode ? 'text-gray-100' : 'text-gray-800'}`}>上传 PDF 文档</span>
              <span className={`mt-0.5 block text-[12px] ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>解析完成后即可速览、翻译和提问</span>
            </button>
            <div className={`mt-1.5 flex max-w-full items-center gap-1.5 text-[11px] ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
              <ScanText className="h-3.5 w-3.5 shrink-0" />
              <span className="shrink-0">解析路线</span>
              <ParseRouteSelect
                value={parseRoute}
                onChange={onParseRouteChange}
                darkMode={darkMode}
              />
            </div>
          </div>
          <motion.button
            type="button"
            onClick={onSelect}
            whileTap={{ scale: 0.98 }}
            className="accent-cta shrink-0 rounded-full px-4 py-2 text-[12px] font-bold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/35"
          >
            选择文件
          </motion.button>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

const buildClientReadingFallback = (blockIndex) => {
  const firstBlock = blockIndex?.pages?.[0]?.blocks?.find((block) => block?.block_id);
  return {
    source: 'client_fallback',
    items: [{
      id: 'summary_unavailable',
      type: 'fallback',
      title: 'AI 总结暂不可用',
      summary: '当前无法生成结构化总结；PDF 章节目录请切换到“大纲”。',
      page: firstBlock ? 1 : 1,
      first_block: firstBlock?.block_id || null,
      evidence_block_ids: firstBlock?.block_id ? [firstBlock.block_id] : [],
      evidence: {
        block_ids: firstBlock?.block_id ? [firstBlock.block_id] : [],
        pages: [1],
        primary_page: 1,
      },
      children: [],
    }],
  };
};

const SECTION_ANCHOR_BLOCK_TYPES = new Set(['heading', 'paragraph']);
const TRANSLATABLE_READING_BLOCK_TYPES = new Set(['heading', 'paragraph', 'caption', 'table', 'formula', 'code']);
const PUBLICATION_HEADER_RE = /\b(vol\.?|no\.?|pp\.?|transactions?|journal|proceedings|conference|copyright|authorized|licensed|downloaded|doi|issn|isbn|technical\s+report)\b/i;

const isTranslatableReadingBlock = (block) => {
  const type = block?.type || 'paragraph';
  const text = String(block?.text || '').trim();
  return Boolean(block?.block_id) && text.length > 1 && TRANSLATABLE_READING_BLOCK_TYPES.has(type);
};

const isValidBlockTranslationText = (value) => {
  const text = String(value || '').trim();
  if (!text) return false;
  if (/^\s*[{[]?['"]?(?:error|_used_provider|_used_model|_usage_meta|fallback_used|raw_usage|completion_tokens|prompt_tokens)['"]?\s*[:=]/s.test(text)) {
    return false;
  }
  if (text.includes('_used_provider') && text.includes('_usage_meta')) return false;
  if (text.includes('completion_tokens') && text.includes('prompt_tokens')) return false;
  return true;
};

const filterValidBlockTranslations = (items = {}) => {
  const result = {};
  Object.entries(items || {}).forEach(([blockId, item]) => {
    if (item && isValidBlockTranslationText(item.translation)) {
      result[blockId] = item;
    }
  });
  return result;
};

const sanitizeTranslationError = (message, fallback = '段落翻译失败，请稍后重试或更换翻译模型') => {
  const text = String(message || '').trim();
  if (!text) return fallback;
  if (
    text.includes('模型未返回译文正文')
    || text.includes('模型返回了无效译文')
    || text.includes('no usable content')
    || text.includes('empty')
    || text.includes('Expecting value')
    || text.includes('JSONDecodeError')
  ) {
    return '模型没有返回可用译文，已保留成功缓存；可稍后补齐失败段落或更换翻译模型';
  }
  if (text.includes('429') || /rate limit|too many requests/i.test(text)) {
    return '翻译请求触发限速，已保留成功缓存；请降低并发或稍后补齐失败段落';
  }
  if (text.includes('timeout') || text.includes('timed out') || text.includes('超时')) {
    return '翻译请求超时，已保留成功缓存；可稍后继续补齐';
  }
  return text.length > 180 ? `${text.slice(0, 180)}...` : text;
};

const isMinerUFullRoute = (manifest) => {
  const route = String(
    manifest?.resolved_route || manifest?.requested_route || manifest?.route || ''
  ).trim().toLowerCase();
  return route === 'mineru';
};

const getParseIdentity = (manifest) => {
  const route = String(
    manifest?.resolved_route || manifest?.requested_route || manifest?.route || ''
  ).trim().toLowerCase();
  const generation = String(manifest?.generation || '').trim();
  const sourceHash = String(manifest?.source_hash || '').trim();
  return `r=${encodeURIComponent(route || 'legacy')}:g=${encodeURIComponent(generation || 'legacy')}:s=${encodeURIComponent(sourceHash || 'legacy')}`;
};

const getStatusParseIdentity = (status) => getParseIdentity(status?.parse_manifest);

const buildOutlineCacheKey = (docId, parseIdentity, blockIndex) => (
  `${docId || ''}:${parseIdentity || ''}:${blockIndex?.block_index_hash || blockIndex?.block_index_revision || blockIndex?.visual_supplement_revision || ''}`
);

const matchesOutlineGenerationFailure = (data, providerId, modelId) => {
  const meta = data?.meta || {};
  return Boolean(meta.generation_error)
    && String(meta.provider || '').toLowerCase() === String(providerId || '').toLowerCase()
    && String(meta.model || '') === String(modelId || '');
};

const isReusableOutlineResult = (data, kind, providerId, modelId) => {
  if (!data) return false;
  if (['ai', 'ai_partial'].includes(data.source)) {
    return String(data.provider || '').toLowerCase() === String(providerId || '').toLowerCase()
      && String(data.model || '') === String(modelId || '');
  }
  if (kind === 'section' && ['toc', 'bookmark', 'mineru'].includes(data.source)) return true;
  if (kind === 'reading') return false;
  return matchesOutlineGenerationFailure(data, providerId, modelId);
};

const getOutlineResultNotice = (data, fallbackMessage) => {
  if (data?.source === 'ai_partial') {
    return '部分章节未完成 AI 总结，当前保留已生成内容；可手动重新生成补齐。';
  }
  return data?.meta?.generation_error ? fallbackMessage : '';
};

const isMinerUParseGateError = (error) => {
  if (Number(error?.status) !== 409) return false;
  return /mineru|全程解析/i.test(String(error?.message || ''));
};

const getMinerUParsePendingNotice = (manifest, deepParseStatus) => {
  const status = String(manifest?.status || '').trim().toLowerCase();
  const stage = String(manifest?.stage || deepParseStatus?.stage || '').trim().toLowerCase();
  const error = String(manifest?.error || deepParseStatus?.error || '').trim();

  if (status === 'failed') {
    if (deepParseStatus?.resume_available && deepParseStatus?.resume_kind === 'result_download') {
      return 'MinerU 已完成远端解析，但结果下载连接中断。重试会复用原任务，不会重新上传 PDF。';
    }
    return error || 'MinerU 全程解析失败，请重新上传后选择 MinerU 路线重试';
  }
  if (status === 'cancelled') return 'MinerU 全程解析已取消，请重新上传后选择 MinerU 路线重试';
  if (stage === 'awaiting_rag_index') {
    return 'MinerU 版面解析已完成，正在等待问答索引发布后统一开放阅读、翻译、速览和问答';
  }
  return 'MinerU 全程解析中，完成后将自动加载阅读结构、大纲、翻译、速览和问答';
};

const normalizeOutlineText = (value) => String(value || '')
  .toLowerCase()
  .replace(/[^a-z0-9\u4e00-\u9fff]+/g, ' ')
  .trim()
  .replace(/\s+/g, ' ');

const stripOutlinePrefix = (value) => {
  let text = normalizeOutlineText(value);
  text = text.replace(/^(?:\d+\s+){1,4}/, '').trim();
  const parts = text.split(/\s+/).filter(Boolean);
  if (parts.length >= 2 && /^([ivxlcm]+|[a-z])$/i.test(parts[0])) {
    text = parts.slice(1).join(' ');
  }
  return text;
};

const outlineTextVariants = (value) => {
  const normalized = normalizeOutlineText(value);
  const stripped = stripOutlinePrefix(value);
  return [...new Set([normalized, stripped].filter(Boolean))];
};

const outlineTitleMatchScore = (title, text) => {
  const titleVariants = outlineTextVariants(title);
  const textVariants = outlineTextVariants(text);
  if (!titleVariants.length || !textVariants.length) return 0;
  if (titleVariants.some((titleText) => textVariants.includes(titleText))) return 4;
  for (const titleText of titleVariants) {
    if (titleText.length < 4) continue;
    for (const blockText of textVariants) {
      if (blockText.startsWith(titleText) || titleText.startsWith(blockText)) return 3;
      if (blockText.includes(titleText)) return 2;
    }
  }
  return 0;
};

const isPublicationHeaderBlock = (block) => {
  const text = String(block?.text || '').replace(/\s+/g, ' ').trim();
  if (!text) return true;
  if (block?.type === 'artifact') return true;
  return PUBLICATION_HEADER_RE.test(text) && text.split(/\s+/).length <= 18;
};

const isUsableSectionAnchor = (block, title, { allowLoose = false } = {}) => {
  if (!block?.block_id || !SECTION_ANCHOR_BLOCK_TYPES.has(block.type || 'paragraph')) return false;
  if (isPublicationHeaderBlock(block)) return false;
  const score = outlineTitleMatchScore(title, block.text);
  if (score >= 3) return true;
  if ((block.type || 'paragraph') === 'heading' && score >= 2) return true;
  if (!allowLoose || score < 2) return false;
  const titleVariants = outlineTextVariants(title).filter((item) => item.length >= 4);
  const blockVariants = outlineTextVariants(block.text);
  return titleVariants.some((titleText) => blockVariants.some((blockText) => blockText.startsWith(titleText)));
};

const GENERIC_OUTLINE_TITLES = new Set(['全文', '全文书签', 'full text', 'fulltext', 'document', 'article', 'paper', 'contents', 'content']);

const flattenOutlineNodes = (items = []) => {
  const result = [];
  const walk = (nodes) => {
    (nodes || []).forEach((node) => {
      if (!node) return;
      result.push(node);
      if (node.children?.length) walk(node.children);
    });
  };
  walk(items);
  return result;
};

const isUsefulOutline = (items = [], source = '') => {
  const flat = flattenOutlineNodes(items);
  if (flat.length === 0) return false;
  if (flat.length === 1) {
    const title = String(flat[0]?.title || '').trim().toLowerCase().replace(/\s+/g, ' ');
    if (GENERIC_OUTLINE_TITLES.has(title)) return false;
    return source !== 'toc' && flat[0]?.source !== 'toc';
  }
  return true;
};

const ReadingDocumentIcon = ({ size = 17 }) => (
  <svg
    aria-hidden="true"
    focusable="false"
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="currentColor"
  >
    <path d="M17.75 2.001a2.25 2.25 0 0 1 2.245 2.096L20 4.25v15.498a2.25 2.25 0 0 1-2.096 2.245l-.154.005H6.25a2.25 2.25 0 0 1-2.245-2.096L4 19.75V4.251a2.25 2.25 0 0 1 2.096-2.245l.154-.005zm0 1.5H6.25a.75.75 0 0 0-.743.648l-.007.102v15.498c0 .38.282.694.648.743l.102.007h11.5a.75.75 0 0 0 .743-.648l.007-.102V4.251a.75.75 0 0 0-.648-.743zm-5.502 9.496a.75.75 0 0 1 .102 1.494l-.102.006H7.75a.75.75 0 0 1-.102-1.493l.102-.007zM16.25 10a.75.75 0 0 1 .102 1.493l-.102.007h-8.5a.75.75 0 0 1-.102-1.494L7.75 10zm0-2.999a.75.75 0 0 1 .102 1.493l-.102.007h-8.5a.75.75 0 0 1-.102-1.493L7.75 7z" />
  </svg>
);

const SETTINGS_SECTIONS = [
  { id: 'common', label: '常用', description: '模型与主要工作配置', Icon: Settings },
  { id: 'reading', label: '阅读', description: '翻译、速览与阅读行为', Icon: ReadingDocumentIcon },
  { id: 'retrieval', label: '检索', description: '召回、证据与代理策略', Icon: ListFilter },
  { id: 'interface', label: '界面', description: '字号、工具栏与回答呈现', Icon: SlidersHorizontal },
  { id: 'storage', label: '存储', description: '文件位置与缓存管理', Icon: Database },
];

const SettingsSwitch = ({ checked, onChange, label, darkMode }) => (
  <button
    type="button"
    role="switch"
    aria-checked={checked}
    aria-label={label}
    onClick={() => onChange(!checked)}
    className={`settings-toggle relative h-6 w-11 shrink-0 rounded-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/35 ${
      checked ? 'settings-toggle-active' : darkMode ? 'bg-white/15' : 'bg-gray-200'
    }`}
  >
    <span className={`settings-toggle-thumb absolute left-0 top-1 h-4 w-4 rounded-full bg-white ${checked ? 'translate-x-6' : 'translate-x-1'}`} />
  </button>
);

const SettingsFeatureRow = ({ Icon, title, description, checked, onChange, statusLabel, statusTone = 'muted', darkMode }) => (
  <div className="settings-feature-row grid grid-cols-[30px_minmax(0,1fr)_auto] items-center gap-3.5 px-5 py-4">
    <div className={`settings-feature-icon flex h-[30px] w-[30px] items-center justify-center ${
      checked
        ? darkMode ? 'text-[#ffb49a]' : 'text-[#B85F47]'
        : darkMode ? 'text-gray-500' : 'text-gray-400'
    }`}>
      <Icon size={18} strokeWidth={2} />
    </div>
    <div className="min-w-0">
      <div className="flex flex-wrap items-center gap-2">
        <h4 className={`text-[13px] font-semibold leading-snug ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>{title}</h4>
        {statusLabel ? <span className="settings-feature-status" data-tone={statusTone}>{statusLabel}</span> : null}
      </div>
      <p className={`mt-1 text-[11px] leading-snug ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>{description}</p>
    </div>
    <SettingsSwitch checked={checked} onChange={onChange} label={title} darkMode={darkMode} />
  </div>
);

const SettingsCheckRow = ({ title, description, hint, checked, onChange, darkMode }) => (
  <label className="settings-row flex items-start space-x-3.5 group cursor-pointer p-1 rounded-2xl">
    <div className={`w-5 h-5 rounded-[6px] flex items-center justify-center shrink-0 mt-0.5 transition-transform group-hover:scale-105 ${checked ? 'bg-[#F0653A] text-white shadow-[0_4px_12px_rgba(240,101,58,0.28)] settings-check-pop' : 'border-2 border-gray-300 bg-transparent'}`}>
      {checked && <Check size={13} strokeWidth={3.5} className="settings-check-mark" />}
    </div>
    <div className="flex min-w-0 flex-1 flex-col">
      <div className="flex items-center justify-between gap-2">
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <h4 className={`text-[14px] font-semibold leading-snug ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>{title}</h4>
          {hint ? <span className="settings-feature-status" data-tone="muted">{hint}</span> : null}
        </div>
        <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="hidden" />
      </div>
      {description ? (
        <p className={`mt-0.5 text-[12px] leading-snug font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
          {description}
        </p>
      ) : null}
    </div>
  </label>
);

const ChatPDF = () => {
  // ========== Context Hooks ==========
  const { getProviderById } = useProvider();
  const { getModelById, allModels } = useModel();
  const { getDefaultModel } = useDefaults();
  const { hasLocalRerank } = useCapabilities();
  const globalSettings = useGlobalSettings();
  const {
    setReasoningEffort,
    reasoningEffort,
    streamOutput,
    setStreamOutput,
    contextCount,
    enableMemory,
    globalScale,
    setGlobalScale,
  } = globalSettings;
  const {
    sendShortcut, confirmDeleteMessage, confirmRegenerateMessage, messageStyle, messageFontSize, setMessageFontSize, codeCollapsible, codeWrappable, codeShowLineNumbers,
    overrideNumericTable, setOverrideNumericTable,
    overrideAnswerCritic, setOverrideAnswerCritic,
    overrideLLMQueryRewrite, setOverrideLLMQueryRewrite,
    overrideBM25Synonyms, setOverrideBM25Synonyms,
    numericTableVisualVerification, setNumericTableVisualVerification,
    visualModelKey, setVisualModelKey,
    visualStrategy, setVisualStrategy,
    localVisualModelKey, setLocalVisualModelKey,
    cheapModel, setCheapModel,
    cheapModelProvider, setCheapModelProvider,
    setCheapModelEndpoint,
    mathEngine,
    mathEnableSingleDollar,
  } = useChatParams();
  const {
    aiAutoProcess,
    setAiAutoProcess,
    autoOutlineSummary,
    setAutoOutlineSummary,
    autoPretranslate: enableHoverPretranslate,
    setAutoPretranslate,
    blockSummary,
    setBlockSummary,
    pretranslateConcurrency,
    setPretranslateConcurrency,
    overviewDefaultDepth,
    setOverviewDefaultDepth,
  } = useReadingSettings();
  const shouldAutoPretranslate = aiAutoProcess && enableHoverPretranslate;
  // 走 ref 读取：翻译那个 useCallback 的依赖数组很大，
  // 把一个开关塞进去会让它频繁重建，进而拖累依赖它的一串 memo。
  const blockSummaryRef = useRef(blockSummary);
  useEffect(() => {
    blockSummaryRef.current = blockSummary;
  }, [blockSummary]);

  // ========== 设置状态 - 使用防抖 localStorage 写入（需求 8.1） ==========
  const [apiKey, setApiKey] = useDebouncedLocalStorage('apiKey', '');
  const [apiProvider, setApiProvider] = useDebouncedLocalStorage('apiProvider', 'openai');
  const [model, setModel] = useDebouncedLocalStorage('model', 'gpt-4o');
  const [enableVectorSearch, setEnableVectorSearch] = useDebouncedLocalStorage('enableVectorSearch', true);
  // 一次性迁移：旧版默认 enableVectorSearch=false，需强制升级为 true
  useEffect(() => {
    if (!localStorage.getItem('_migrated_vectorSearch_v1')) {
      setEnableVectorSearch(true);
      localStorage.setItem('_migrated_vectorSearch_v1', '1');
    }
  }, []);
  const [enableScreenshot, setEnableScreenshot] = useDebouncedLocalStorage('enableScreenshot', true);
  const [streamSpeed, setStreamSpeed] = useDebouncedLocalStorage('streamSpeed', 'normal');
  const [enableBlurReveal, setEnableBlurReveal] = useDebouncedLocalStorage('enableBlurReveal', true);
  const [blurIntensity, setBlurIntensity] = useDebouncedLocalStorage('blurIntensity', 'medium');
  // v2 曾强制关闭模糊渐显。只修正受该迁移影响的用户一次，之后仍尊重手动设置。
  useEffect(() => {
    if (
      localStorage.getItem('_migrated_stream_reveal_v2') &&
      !localStorage.getItem('_restored_stream_reveal_v3')
    ) {
      setEnableBlurReveal(true);
      setBlurIntensity('medium');
      localStorage.setItem('_restored_stream_reveal_v3', '1');
    }
  }, []);
  const [searchEngine, setSearchEngine] = useDebouncedLocalStorage('searchEngine', 'google');
  const [searchEngineUrl, setSearchEngineUrl] = useDebouncedLocalStorage('searchEngineUrl', 'https://www.google.com/search?q={query}');
  const [toolbarSize, setToolbarSize] = useDebouncedLocalStorage('toolbarSize', 'normal');
  const [useRerankSetting, setUseRerankSetting] = useDebouncedLocalStorage('useRerank', true);
  const [rerankerModel, setRerankerModel] = useDebouncedLocalStorage('rerankerModel', 'BAAI/bge-reranker-base');
  const [enableGraphRAG, setEnableGraphRAG] = useDebouncedLocalStorage('enableGraphRAG', false);
  const [enableAgentRetrieval, setEnableAgentRetrieval] = useDebouncedLocalStorage('enableAgentRetrieval', false);
  const [forceAgentRetrieval, setForceAgentRetrieval] = useDebouncedLocalStorage('chatpdf_force_agent_retrieval', false);
  const [enableJiebaBM25, setEnableJiebaBM25] = useDebouncedLocalStorage('enableJiebaBM25', true);
  const [numExpandContextChunk, setNumExpandContextChunk] = useDebouncedLocalStorage('numExpandContextChunk', 1);

  // 不需要持久化的设置状态
  const [availableModels, setAvailableModels] = useState({});
  const [availableEmbeddingModels, setAvailableEmbeddingModels] = useState({});
  const [documentHighlights, setDocumentHighlights] = useState([]);
  const [documentNotes, setDocumentNotes] = useState([]);
  const [annotationTool, setAnnotationTool] = useState(null);
  const [annotationColor, setAnnotationColor] = useState(DEFAULT_DOCUMENT_HIGHLIGHT_COLOR);
  const [selectedSavedHighlightId, setSelectedSavedHighlightId] = useState('');
  const [autoAnnotationRevision, setAutoAnnotationRevision] = useState(0);
  const [pendingUserNoteRevealId, setPendingUserNoteRevealId] = useState('');
  const [sidebarMode, setSidebarMode] = useState('history');
  const [isSidebarResizing, setIsSidebarResizing] = useState(false);
  const [settingsSection, setSettingsSection] = useState('common');
  const [pendingSettingsPanel, setPendingSettingsPanel] = useState(null);
  const [showRetrievalTuning, setShowRetrievalTuning] = useState(false);
  const [blockIndex, setBlockIndex] = useState(null);
  const [blockIndexLoading, setBlockIndexLoading] = useState(false);
  const [blockIndexError, setBlockIndexError] = useState('');
  const [blockIndexReloadKey, setBlockIndexReloadKey] = useState(0);
  const [readingOutline, setReadingOutline] = useState(null);
  const [readingOutlineLoading, setReadingOutlineLoading] = useState(false);
  const [readingOutlineError, setReadingOutlineError] = useState('');
  const [readingOutlineFallbackNotice, setReadingOutlineFallbackNotice] = useState('');
  const [readingOutlineReloadKey, setReadingOutlineReloadKey] = useState(0);
  const [sectionOutline, setSectionOutline] = useState(null);
  const [sectionOutlineLoading, setSectionOutlineLoading] = useState(false);
  const [sectionOutlineError, setSectionOutlineError] = useState('');
  const [sectionOutlineFallbackNotice, setSectionOutlineFallbackNotice] = useState('');
  const [sectionOutlineReloadKey, setSectionOutlineReloadKey] = useState(0);
  const [activeReadingNodeId, setActiveReadingNodeId] = useState(null);
  const [visitedReadingNodeIds, setVisitedReadingNodeIds] = useState(new Set());
  const [activeSectionNodeId, setActiveSectionNodeId] = useState(null);
  const [visitedSectionNodeIds, setVisitedSectionNodeIds] = useState(new Set());
  const [blockTranslations, setBlockTranslations] = useState({});
  const [blockTranslateLoading, setBlockTranslateLoading] = useState(false);
  const [blockTranslateError, setBlockTranslateError] = useState('');
  const [translatingBlockIds, setTranslatingBlockIds] = useState(new Set());
  const [blockTranslationsLoaded, setBlockTranslationsLoaded] = useState(false);
  const [blockTranslationsLoadedIdentity, setBlockTranslationsLoadedIdentity] = useState('');
  const [pretranslateProgress, setPretranslateProgress] = useState({ running: false, done: 0, total: 0 });
  const [failedTranslationBlockIds, setFailedTranslationBlockIds] = useState(new Set());
  const [pretranslateNotice, setPretranslateNotice] = useState('');
  const [pretranslateError, setPretranslateError] = useState('');
  const [showAiProcessingPanel, setShowAiProcessingPanel] = useState(false);
  const [deepParseStatus, setDeepParseStatus] = useState(null);
  const [downstreamTaskStatuses, setDownstreamTaskStatuses] = useState({});
  const [parseIdentityHydration, setParseIdentityHydration] = useState({
    docId: '',
    settled: true,
  });
  const [deepParseNotice, setDeepParseNotice] = useState('');
  const [ragIndexStatus, setRagIndexStatus] = useState(null);
  const [ragIndexBusy, setRagIndexBusy] = useState(false);
  const [ragIndexNotice, setRagIndexNotice] = useState('');
  const [ragIndexError, setRagIndexError] = useState('');
  const [embeddingConflictRecovery, setEmbeddingConflictRecovery] = useState({
    messageId: null,
    status: 'idle',
  });
  const [selectedParseRoute, setSelectedParseRoute] = useState(loadStoredParseRoute);
  const [isLocalParserInstallOpen, setIsLocalParserInstallOpen] = useState(false);
  const [hoveredReadingBlockId, setHoveredReadingBlockId] = useState(null);
  const [pinnedReadingBlockId, setPinnedReadingBlockId] = useState(null);
  const [readingJumpPulseToken, setReadingJumpPulseToken] = useState(0);
  const [readerNavigationRequest, setReaderNavigationRequest] = useState(null);
  const pretranslateRunRef = useRef(0);
  const pretranslateStartedDocRef = useRef(null);
  const pretranslateAbortRef = useRef(null);
  const blockTranslationEpochRef = useRef(0);
  const visualSupplementRevisionRef = useRef('');
  const parseContextRef = useRef({ docId: '', parseIdentity: '', epoch: 0 });
  const deepParseStatusRequestRef = useRef({ sequence: 0, controller: null });
  const prevShouldAutoPretranslateRef = useRef(shouldAutoPretranslate);
  const readingOutlineRequestRef = useRef(0);
  const readingOutlineForceRef = useRef(false);
  const readingOutlineCacheRef = useRef(new Map());
  const sectionOutlineRequestRef = useRef(0);
  const sectionOutlineForceRef = useRef(false);
  const sectionOutlineCacheRef = useRef(new Map());
  const streamConfigMigratedRef = useRef(false);
  const uploadStartsNewChatRef = useRef(true);
  const pendingLocalParserUploadRef = useRef(null);
  const selectedPdfHighlightAnchorRef = useRef(null);
  const pendingAutoAnnotationRef = useRef(null);
  const sidebarResizeRef = useRef({ active: false, pointerId: null, startX: 0, startWidth: SIDEBAR_DEFAULT_WIDTH, maxWidth: SIDEBAR_MAX_WIDTH });

  // 兼容旧配置：streamSpeed 和 streamOutput 过去分别持久化，
  // 会留下“速度仍开启，但隐藏的流式开关已关闭”的冲突状态。
  // 首次加载时按 streamSpeed 纠正一次，避免用户看到长时间空白后整段出现。
  useEffect(() => {
    if (streamConfigMigratedRef.current) return;
    const expectedStreamOutput = streamSpeed !== 'off';
    if (streamOutput !== expectedStreamOutput) {
      setStreamOutput(expectedStreamOutput);
    }
    streamConfigMigratedRef.current = true;
  }, [streamSpeed, streamOutput, setStreamOutput]);

  // ========== UI 状态 Hook（需求 1.3） ==========
  const {
    isNarrowDesktop,
    showSidebar, setShowSidebar,
    isHeaderExpanded, setIsHeaderExpanded,
    sidebarWidth, setSidebarWidth,
    pdfPanelWidth, setPdfPanelWidth,
    darkMode, setDarkMode,
    showSettings, setShowSettings,
    showEmbeddingSettings, setShowEmbeddingSettings,
    showOCRSettings, setShowOCRSettings,
    showGlobalSettings, setShowGlobalSettings,
    showChatSettings, setShowChatSettings,
    enableThinking, setEnableThinking,
    rightPanelMode, setRightPanelMode,
    overviewDepth, setOverviewDepth,
  } = useUIState();
  const { confirm: confirmAction, confirmDialogProps } = useConfirmDialog();
  const reduceMotion = useReducedMotion();

  // 译文只在一个地方落地：'panel' = 右侧阅读区的「页面翻译」卡片，
  // 'dock' = PDF 右侧吸附栏。两者都要固定占屏幕，同时开着就是把同一份内容画两遍。
  const [translationSurface, setTranslationSurface] = useState('panel');

  useEffect(() => {
    if (!isNarrowDesktop || !showSidebar) return undefined;
    const handleEscape = (event) => {
      if (event.key === 'Escape') setShowSidebar(false);
    };
    window.addEventListener('keydown', handleEscape);
    return () => window.removeEventListener('keydown', handleEscape);
  }, [isNarrowDesktop, setShowSidebar, showSidebar]);

  useEffect(() => {
    if (isNarrowDesktop && (pdfPanelWidth < 38 || pdfPanelWidth > 62)) {
      setPdfPanelWidth(50);
    }
  }, [isNarrowDesktop, pdfPanelWidth, setPdfPanelWidth]);

  // 首页空闲时预热 PDF 阅读器（pdf.js 很大）。打开文档时不应再等这块 JS。
  useEffect(() => {
    if (typeof window.requestIdleCallback === 'function') {
      const idleId = window.requestIdleCallback(preloadPDFViewer, { timeout: 800 });
      return () => window.cancelIdleCallback?.(idleId);
    }
    const timerId = window.setTimeout(preloadPDFViewer, 120);
    return () => window.clearTimeout(timerId);
  }, []);

  // 设置中心打开后利用空闲帧预热模型服务代码与初始状态，避免点击时才解析懒加载模块。
  useEffect(() => {
    if (!showSettings || showEmbeddingSettings) return undefined;

    if (typeof window.requestIdleCallback === 'function') {
      const idleId = window.requestIdleCallback(preloadEmbeddingSettings, { timeout: 600 });
      return () => window.cancelIdleCallback?.(idleId);
    }

    const timerId = window.setTimeout(preloadEmbeddingSettings, 80);
    return () => window.clearTimeout(timerId);
  }, [showEmbeddingSettings, showSettings]);

  useEffect(() => {
    if (overviewDefaultDepth && overviewDefaultDepth !== overviewDepth) {
      setOverviewDepth(overviewDefaultDepth);
    }
  }, [overviewDefaultDepth, overviewDepth, setOverviewDepth]);

  useEffect(() => {
    if (!showOCRSettings) setSelectedParseRoute(loadStoredParseRoute());
  }, [showOCRSettings]);

  const handleOverviewDepthChange = useCallback((nextDepth) => {
    setOverviewDepth(nextDepth);
    setOverviewDefaultDepth(nextDepth);
  }, [setOverviewDefaultDepth, setOverviewDepth]);

  // ========== 模型/凭证辅助函数 ==========
  const getEmbeddingConfig = useCallback(() => {
    const emk = getDefaultModel('embeddingModel');
    if (!emk) {
      return { isValid: false, reason: 'not_selected' };
    }

    const [pid, ...rest] = emk.split(':');
    const modelId = rest.join(':');
    const provider = getProviderById(pid);
    if (!provider) {
      return { isValid: false, reason: 'provider_missing', compositeKey: emk, providerId: pid, modelId };
    }

    const modelObj = modelId ? getModelById(modelId, pid) : null;
    if (!modelObj) {
      return { isValid: false, reason: 'model_not_found', compositeKey: emk, providerId: pid, modelId, provider };
    }
    if (modelObj.type !== 'embedding') {
      return {
        isValid: false,
        reason: 'wrong_type',
        compositeKey: emk,
        providerId: pid,
        modelId,
        modelType: modelObj.type,
        provider,
      };
    }

    return {
      isValid: true,
      compositeKey: emk,
      providerId: pid,
      modelId,
      model: modelObj,
      provider,
    };
  }, [getDefaultModel, getProviderById, getModelById]);

  const getEmbeddingCredentialState = useCallback(() => {
    const config = getEmbeddingConfig();
    if (!config.isValid) {
      return {
        isValid: false,
        reason: String(config.reason || 'embedding_config_invalid'),
        providerId: String(config.providerId || '').trim(),
        providerName: String(config.provider?.name || config.providerId || '').trim(),
        apiKey: '',
        apiHost: '',
        isMissingApiKey: false,
      };
    }

    const providerId = String(config.providerId || '').trim();
    const providerName = String(config.provider?.name || providerId || '').trim();
    const apiKey = isKeylessLocalProvider(providerId)
      ? ''
      : String(config.provider?.apiKey || '').trim();

    return {
      isValid: true,
      providerId,
      providerName,
      apiKey,
      apiHost: String(config.provider?.apiHost || '').trim(),
      isMissingApiKey: !isKeylessLocalProvider(providerId) && !apiKey,
    };
  }, [getEmbeddingConfig]);

  const getMissingEmbeddingCredential = useCallback(() => {
    const credentials = getEmbeddingCredentialState();
    if (!credentials.isValid) {
      const messages = {
        model_not_found: '当前默认 Embedding 模型不存在或已下线，请重新选择',
        wrong_type: '当前默认模型不是 Embedding 类型，请切换后重试',
        provider_missing: '当前默认 Embedding Provider 不存在，请在模型服务中重新配置',
      };
      return {
        ...credentials,
        message: messages[credentials.reason] || '请先在模型设置里选择可用的 Embedding 模型',
      };
    }
    if (!credentials.isMissingApiKey) return null;
    return {
      ...credentials,
      message: getMissingEmbeddingApiKeyMessage(credentials.providerName || credentials.providerId),
    };
  }, [getEmbeddingCredentialState]);

  const getEmbeddingApiKey = useCallback(() => {
    const credentials = getEmbeddingCredentialState();
    return credentials.isValid ? credentials.apiKey : '';
  }, [getEmbeddingCredentialState]);

  const preparedMemoryEmbeddingRef = useRef('');
  useEffect(() => {
    if (!enableMemory) {
      preparedMemoryEmbeddingRef.current = '';
      return undefined;
    }
    const config = getEmbeddingConfig();
    const credentials = getEmbeddingCredentialState();
    if (!config.isValid || !credentials.isValid || credentials.isMissingApiKey) {
      return undefined;
    }

    const payload = {
      embedding_api_key: credentials.apiKey || '',
      embedding_model: config.compositeKey || config.modelId || '',
      embedding_provider: config.providerId || '',
      embedding_api_host: credentials.apiHost || '',
    };
    const signature = [
      payload.embedding_provider,
      payload.embedding_model,
      payload.embedding_api_host,
      payload.embedding_api_key,
    ].join('\u0000');
    if (preparedMemoryEmbeddingRef.current === signature) {
      return undefined;
    }
    preparedMemoryEmbeddingRef.current = signature;

    let active = true;
    let retryTimer = null;
    const prepare = async (attempt = 0) => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/memory/embedding/prepare`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (response.ok) return;
        // Frontend and backend are separate dev processes. During a backend
        // restart the new endpoint can briefly be 404/5xx; retry that transient
        // state, but do not loop on a genuinely invalid embedding configuration.
        if (response.status !== 404 && response.status < 500) {
          if (preparedMemoryEmbeddingRef.current === signature) {
            preparedMemoryEmbeddingRef.current = '';
          }
          return;
        }
        throw new Error(`HTTP ${response.status}`);
      } catch {
        if (!active || preparedMemoryEmbeddingRef.current !== signature) return;
        if (attempt < 4) {
          retryTimer = window.setTimeout(() => prepare(attempt + 1), 1500);
        } else {
          preparedMemoryEmbeddingRef.current = '';
        }
      }
    };
    void prepare();

    return () => {
      active = false;
      if (retryTimer !== null) window.clearTimeout(retryTimer);
    };
  }, [enableMemory, getEmbeddingConfig, getEmbeddingCredentialState]);

  const getCurrentChatModel = useCallback(() => {
    const chatKey = getDefaultModel('assistantModel');
    if (chatKey) {
      const [pid, mid] = chatKey.split(':');
      return { providerId: pid, modelId: mid };
    }
    return { providerId: apiProvider, modelId: model };
  }, [getDefaultModel, apiProvider, model]);

  const getChatCredentials = useCallback(() => {
    const chatKey = getDefaultModel('assistantModel');
    const { providerId, modelId } = getCurrentChatModel();
    const provider = getProviderById(providerId);
    if (chatKey) {
      return { providerId, modelId, apiKey: provider?.apiKey || '', apiHost: provider?.apiHost || '' };
    }
    return { providerId, modelId, apiKey: provider?.apiKey || apiKey, apiHost: provider?.apiHost || '' };
  }, [getDefaultModel, getCurrentChatModel, getProviderById, apiKey]);

  const getVisualCredentials = useCallback(() => {
    const chatCredentials = getChatCredentials?.() || {};
    const chatProvider = chatCredentials.providerId || 'openai';
    const chatModel = chatCredentials.modelId || 'gpt-4o';
    const useDedicatedModel = Boolean(visualModelKey && visualModelKey !== 'follow_chat' && visualModelKey.includes(':'));
    const separator = useDedicatedModel ? visualModelKey.indexOf(':') : -1;
    const providerId = useDedicatedModel ? visualModelKey.slice(0, separator) : chatProvider;
    const modelId = useDedicatedModel ? visualModelKey.slice(separator + 1) : chatModel;
    const provider = getProviderById?.(providerId);
    const modelObject = getModelById?.(modelId, providerId) || { id: modelId, providerId };
    const useLocalModel = Boolean(localVisualModelKey && localVisualModelKey !== 'none' && localVisualModelKey.includes(':'));
    const localSeparator = useLocalModel ? localVisualModelKey.indexOf(':') : -1;
    const localProviderId = useLocalModel ? localVisualModelKey.slice(0, localSeparator) : '';
    const localModelId = useLocalModel ? localVisualModelKey.slice(localSeparator + 1) : '';
    const localProvider = useLocalModel ? getProviderById?.(localProviderId) : null;
    const localModelObject = useLocalModel
      ? getModelById?.(localModelId, localProviderId) || { id: localModelId, providerId: localProviderId }
      : null;
    const strongIsVisionCapable = supportsVision(modelObject);
    const localIsVisionCapable = useLocalModel && supportsVision(localModelObject);
    return {
      providerId,
      modelId,
      apiKey: provider?.apiKey || (useDedicatedModel ? '' : chatCredentials.apiKey || ''),
      apiHost: provider?.apiHost || '',
      isVisionCapable: strongIsVisionCapable,
      policyVisionCapable: strongIsVisionCapable || localIsVisionCapable,
      source: useDedicatedModel ? 'dedicated' : 'follow_chat',
      strategy: visualStrategy || 'balanced',
      local: useLocalModel ? {
        providerId: localProviderId,
        modelId: localModelId,
        apiKey: localProvider?.apiKey || '',
        apiHost: localProvider?.apiHost || '',
        isVisionCapable: localIsVisionCapable,
      } : null,
    };
  }, [getChatCredentials, getModelById, getProviderById, localVisualModelKey, visualModelKey, visualStrategy]);

  const getChatRequestConfig = useCallback(() => {
    const chatCredentials = getChatCredentials?.();
    const chatProvider = chatCredentials?.providerId || 'openai';
    const chatModel = chatCredentials?.modelId || 'gpt-4o';
    const chatApiKey = chatCredentials?.apiKey || '';
    const chatProviderFull = getProviderById?.(chatProvider);
    const providerLower = String(chatProvider || '').toLowerCase();
    const canCallModel = Boolean(chatApiKey) || providerLower === 'local' || providerLower === 'ollama';
    const headers = { 'Content-Type': 'application/json' };

    headers['X-ChatPDF-Provider'] = chatProvider;
    headers['X-ChatPDF-Model'] = chatModel;
    if (chatCredentials) {
      if (chatApiKey) {
        headers['X-ChatPDF-Api-Key'] = chatApiKey;
      }
      if (chatProviderFull?.apiHost) {
        headers['X-ChatPDF-Api-Host'] = chatProviderFull.apiHost;
      }
    }

    return {
      headers,
      providerId: chatProvider,
      modelId: chatModel,
      providerName: chatProviderFull?.name || chatProvider,
      canCallModel,
    };
  }, [getChatCredentials, getProviderById]);

  const getCurrentRerankModel = useCallback(() => {
    const rrk = getDefaultModel('rerankModel');
    if (rrk) {
      const [pid, ...rest] = rrk.split(':');
      return {
        providerId: pid,
        modelId: rest.join(':'),
        compositeKey: rrk,
        source: 'selected',
      };
    }
    // 没有配置 rerank 模型时，仅在本地 rerank 可用时才 fallback 到本地
    if (hasLocalRerank) {
      return {
        providerId: 'local',
        modelId: 'BAAI/bge-reranker-base',
        compositeKey: 'local:BAAI/bge-reranker-base',
        source: 'fallback_local',
      };
    }
    return null;
  }, [getDefaultModel, hasLocalRerank]);

  const getRerankCredentials = useCallback(() => {
    const rerankModel = getCurrentRerankModel();
    if (!rerankModel) return null;
    const { providerId, modelId, source } = rerankModel;
    const provider = getProviderById(providerId);
    if (!provider) {
      return {
        isValid: false,
        reason: 'provider_missing',
        providerId,
        modelId,
        providerName: providerId,
        errorMessage: '当前默认 Rerank Provider 不存在，请在模型服务中重新配置',
      };
    }

    const modelObj = getModelById(modelId, providerId);
    if (!modelObj && source !== 'fallback_local') {
      return {
        isValid: false,
        reason: 'model_not_found',
        providerId,
        modelId,
        providerName: provider.name || providerId,
        errorMessage: '当前默认 Rerank 模型不存在或已下线，请重新选择后再搜索',
      };
    }
    if (modelObj?.type && modelObj.type !== 'rerank') {
      return {
        isValid: false,
        reason: 'wrong_type',
        providerId,
        modelId,
        modelType: modelObj.type,
        providerName: provider.name || providerId,
        errorMessage: '当前默认模型不是 Rerank 类型，请切换后再搜索',
      };
    }

    const apiHost = String(provider?.apiHost || '').trim();
    const rerankEndpoint = buildProviderApiEndpoint(
      apiHost,
      provider?.apiConfig?.rerankEndpoint || '',
    );
    const apiKey = isKeylessLocalProvider(providerId) ? '' : String(provider?.apiKey || '').trim();
    if (!isKeylessLocalProvider(providerId) && !apiKey) {
      return {
        isValid: false,
        reason: 'api_key_missing',
        providerId,
        modelId,
        providerName: provider.name || providerId,
        errorMessage: `请先为 ${provider.name || providerId} 配置 Rerank API Key`,
      };
    }
    if (!isKeylessLocalProvider(providerId) && !rerankEndpoint) {
      return {
        isValid: false,
        reason: 'endpoint_missing',
        providerId,
        modelId,
        providerName: provider.name || providerId,
        errorMessage: `当前 Provider 未配置可用的 Rerank Endpoint：${provider.name || providerId}`,
      };
    }

    return {
      isValid: true,
      providerId,
      modelId,
      providerName: provider.name || providerId,
      apiKey,
      apiHost,
      rerankEndpoint,
    };
  }, [getCurrentRerankModel, getProviderById, getModelById]);

  const defaultModelOverview = useMemo(() => {
    const resolveModel = (type) => {
      const key = getDefaultModel(type);
      if (!key) {
        return {
          modelName: type === 'rerankModel' ? '未选择重排模型' : '尚未选择模型',
          providerName: '未配置服务商',
          state: 'missing',
          statusLabel: type === 'rerankModel' ? '可选' : '未选择',
        };
      }

      const separatorIndex = key.indexOf(':');
      const providerId = separatorIndex > 0 ? key.slice(0, separatorIndex) : '';
      const modelId = separatorIndex > 0 ? key.slice(separatorIndex + 1) : key;
      const provider = providerId ? getProviderById(providerId) : null;
      const selectedModel = providerId ? getModelById(modelId, providerId) : null;
      const isLocal = providerId === 'local';
      const isReady = Boolean(
        isLocal
        || (provider?.enabled !== false && String(provider?.apiKey || '').trim())
      );

      return {
        modelName: selectedModel?.name || modelId,
        providerName: provider?.name || providerId || '默认服务',
        state: isLocal ? 'local' : isReady ? 'ready' : 'needs_setup',
        statusLabel: isLocal ? '本地' : isReady ? '已配置' : '需配置',
      };
    };

    return {
      assistant: resolveModel('assistantModel'),
      embedding: resolveModel('embeddingModel'),
      rerank: resolveModel('rerankModel'),
    };
  }, [getDefaultModel, getModelById, getProviderById]);

  const currentChatModelObj = useMemo(() => {
    const chatKey = getDefaultModel('assistantModel');
    if (!chatKey || !chatKey.includes(':')) return null;
    const [pid, mid] = chatKey.split(':');
    return getModelById(mid, pid);
  }, [getDefaultModel, getModelById]);

  const isVisionCapable = useMemo(() => {
    const current = currentChatModelObj || {
      id: getCurrentChatModel().modelId,
      providerId: getCurrentChatModel().providerId,
    };
    return supportsVision(current);
  }, [currentChatModelObj, getCurrentChatModel]);

  const visualModelOptions = useMemo(() => {
    const chatCredentials = getChatCredentials?.() || {};
    const followingLabel = isVisionCapable
      ? `跟随对话模型（${chatCredentials.modelId || '当前模型'}）`
      : `跟随对话模型（${chatCredentials.modelId || '当前模型'}，未检测到视觉能力）`;
    const candidates = (allModels || [])
      .filter((candidate) => candidate?.type === 'chat' && supportsVision(candidate))
      .map((candidate) => ({
        value: `${candidate.providerId}:${candidate.id}`,
        label: `${getProviderById?.(candidate.providerId)?.name || candidate.providerId} · ${candidate.name || candidate.id}`,
      }));
    const seen = new Set();
    return [
      { value: 'follow_chat', label: followingLabel },
      ...candidates.filter((candidate) => !seen.has(candidate.value) && seen.add(candidate.value)),
    ];
  }, [allModels, getChatCredentials, getProviderById, isVisionCapable]);

  const localVisualModelOptions = useMemo(() => {
    const candidates = (allModels || [])
      .filter((candidate) => (
        candidate?.type === 'chat'
        && ['local', 'ollama'].includes(String(candidate.providerId || '').toLowerCase())
        && isLoopbackApiHost(getProviderById?.(candidate.providerId)?.apiHost)
        && supportsVision(candidate)
      ))
      .map((candidate) => ({
        value: `${candidate.providerId}:${candidate.id}`,
        label: `${getProviderById?.(candidate.providerId)?.name || candidate.providerId} · ${candidate.name || candidate.id}`,
      }));
    const seen = new Set();
    return [
      { value: 'none', label: '未配置本地视觉模型' },
      ...candidates.filter((candidate) => !seen.has(candidate.value) && seen.add(candidate.value)),
    ];
  }, [allModels, getProviderById]);

  const cheapModelKey = cheapModelProvider && cheapModel
    ? `${cheapModelProvider}:${cheapModel}`
    : '';
  const currentChatProviderId = getChatCredentials?.()?.providerId || '';

  const cheapModelOptions = useMemo(() => {
    const candidates = (allModels || [])
      .filter((candidate) => candidate?.type === 'chat')
      .filter((candidate) => !currentChatProviderId || candidate.providerId === currentChatProviderId)
      .filter((candidate) => {
        const provider = getProviderById?.(candidate.providerId);
        if (!provider || provider.enabled === false) return false;
        return isKeylessLocalProvider(candidate.providerId)
          || Boolean(String(provider.apiKey || '').trim());
      })
      .map((candidate) => {
        const provider = getProviderById?.(candidate.providerId);
        return {
          value: `${candidate.providerId}:${candidate.id}`,
          label: `${provider?.name || candidate.providerId} · ${candidate.name || candidate.id}`,
          providerId: candidate.providerId,
          modelId: candidate.id,
        };
      });
    const seen = new Set();
    return [
      { value: '', label: '跟随后端配置' },
      ...candidates.filter((candidate) => !seen.has(candidate.value) && seen.add(candidate.value)),
    ];
  }, [allModels, currentChatProviderId, getProviderById]);

  const cheapModelSelectionAvailable = !cheapModelKey
    || cheapModelOptions.some((option) => option.value === cheapModelKey);

  const cheapModelUnavailableLabel = cheapModelKey && !cheapModelSelectionAvailable
    ? `已保存：${cheapModelProvider} · ${cheapModel}（当前配置中不可用）`
    : undefined;

  const handleCheapModelChange = useCallback((value) => {
    if (!value) {
      setCheapModel('');
      setCheapModelProvider('');
      setCheapModelEndpoint('');
      return;
    }

    const separatorIndex = value.indexOf(':');
    if (separatorIndex <= 0) return;

    const providerId = value.slice(0, separatorIndex);
    const modelId = value.slice(separatorIndex + 1);
    const provider = getProviderById?.(providerId);

    setCheapModelProvider(providerId);
    setCheapModel(modelId);
    setCheapModelEndpoint(provider?.apiHost || '');
  }, [getProviderById, setCheapModel, setCheapModelEndpoint, setCheapModelProvider]);
  const hasLocalVisualModel = localVisualModelKey !== 'none'
    && localVisualModelOptions.some((option) => option.value === localVisualModelKey);
  const visualPolicyReady = visualStrategy === 'privacy'
    ? hasLocalVisualModel
    : (isVisionCapable || visualModelKey !== 'follow_chat' || hasLocalVisualModel);
  const visualCredentials = useMemo(() => getVisualCredentials?.() || {}, [getVisualCredentials]);
  const strongVisualModelAvailable = visualCredentials.isVisionCapable === true
    && (Boolean(visualCredentials.apiKey) || isKeylessLocalProvider(visualCredentials.providerId));
  const visualModelSummary = useMemo(() => {
    const selectedStrong = visualModelOptions.find((option) => option.value === visualModelKey);
    const selectedLocal = localVisualModelOptions.find((option) => option.value === localVisualModelKey);
    if (visualStrategy === 'privacy') {
      return hasLocalVisualModel
        ? `仅本地 · ${selectedLocal?.label || '本地视觉模型'}`
        : '未配置本地视觉模型';
    }
    if (strongVisualModelAvailable && visualModelKey === 'follow_chat') {
      const chatCredentials = getChatCredentials?.() || {};
      return `跟随对话模型 · ${chatCredentials.modelId || '当前模型'}`;
    }
    if (strongVisualModelAvailable) {
      return selectedStrong?.label || '独立视觉模型';
    }
    if (hasLocalVisualModel) {
      return `本地视觉模型 · ${selectedLocal?.label || '本地视觉模型'}（强视觉模型不可用）`;
    }
    return visualModelKey === 'follow_chat'
      ? '当前对话模型不支持视觉，尚未配置本地模型'
      : '已保存的视觉模型当前不可用';
  }, [
    getChatCredentials,
    hasLocalVisualModel,
    localVisualModelKey,
    localVisualModelOptions,
    strongVisualModelAvailable,
    visualModelKey,
    visualModelOptions,
    visualStrategy,
  ]);

  // ========== 文档状态 Hook（需求 1.1） ==========
  // useDocumentState 内部管理 docId/docInfo，需要其他 Hook 的 setter 函数
  // setter 函数通过 ref 桥接，避免 Hook 调用顺序问题
  const messageSettersRef = useRef({});
  const pdfSettersRef = useRef({});
  const screenshotSettersRef = useRef({});

  const documentState = useDocumentState({
    getEmbeddingConfig,
    getChatCredentials,
    getVisualCredentials,
    getProviderById,
    setMessages: (...args) => messageSettersRef.current.setMessages?.(...args),
    setCurrentPage: (...args) => pdfSettersRef.current.setCurrentPage?.(...args),
    setScreenshots: (...args) => screenshotSettersRef.current.setScreenshots?.(...args),
    setIsLoading: (...args) => messageSettersRef.current.setIsLoading?.(...args),
    setSelectedText: (...args) => pdfSettersRef.current.setSelectedText?.(...args),
  });
  const {
    docId, setDocId,
    docInfo, setDocInfo,
    isUploading, uploadProgress, uploadStatus, uploadFileInfo,
    history, storageInfo,
    fileInputRef,
    handleFileUpload, startNewChat, loadSession, deleteSession,
    saveCurrentSession, flushCurrentSession, fetchStorageInfo,
    overview, overviewLoading, overviewError, fetchOverview, clearOverviewCache,
  } = documentState;
  const [pendingSessionDelete, setPendingSessionDelete] = useState(null);
  const requestSessionDelete = useCallback((session) => {
    const sessionId = String(session?.id || '').trim();
    if (!sessionId) return;
    if (!confirmDeleteMessage) {
      deleteSession(sessionId);
      return;
    }
    setPendingSessionDelete({
      id: sessionId,
      filename: session?.filename || '未命名文档',
    });
  }, [confirmDeleteMessage, deleteSession]);
  const confirmSessionDelete = useCallback(() => {
    const sessionId = String(pendingSessionDelete?.id || '').trim();
    if (sessionId) deleteSession(sessionId);
    setPendingSessionDelete(null);
  }, [deleteSession, pendingSessionDelete]);

  const [crossDocumentIds, setCrossDocumentIds] = useState([]);
  const [crossDocumentMenuOpen, setCrossDocumentMenuOpen] = useState(false);
  const [crossDocumentCandidates, setCrossDocumentCandidates] = useState([]);
  const [crossDocumentLoading, setCrossDocumentLoading] = useState(false);
  const historyDocumentCandidates = useMemo(() => (
    (history || [])
      .filter((item) => item?.docId && String(item.docId) !== String(docId || ''))
      .map((item) => ({
        doc_id: String(item.docId),
        filename: item.filename || '未命名文档',
        parse_ready: true,
        upload_time: item.updatedAt || item.createdAt || 0,
      }))
  ), [docId, history]);
  const crossDocumentOptions = useMemo(() => {
    const byId = new Map();
    [...crossDocumentCandidates, ...historyDocumentCandidates].forEach((candidate) => {
      const candidateId = String(candidate?.doc_id || '').trim();
      if (!candidateId || candidateId === String(docId || '')) return;
      const existing = byId.get(candidateId);
      if (!existing) byId.set(candidateId, { ...candidate, doc_id: candidateId });
    });
    return [...byId.values()].slice(0, 8);
  }, [crossDocumentCandidates, docId, historyDocumentCandidates]);
  const loadCrossDocumentCandidates = useCallback(async () => {
    if (!docId) return;
    setCrossDocumentLoading(true);
    try {
      const params = new URLSearchParams({ exclude_doc_id: String(docId), limit: '8' });
      const response = await fetch(`${API_BASE_URL}/documents/recall?${params.toString()}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json();
      setCrossDocumentCandidates(Array.isArray(payload?.candidates) ? payload.candidates : []);
    } catch {
      // Local session history remains a useful offline fallback for the picker.
      setCrossDocumentCandidates([]);
    } finally {
      setCrossDocumentLoading(false);
    }
  }, [docId]);
  const toggleCrossDocument = useCallback((candidateId) => {
    const normalizedId = String(candidateId || '').trim();
    if (!normalizedId || normalizedId === String(docId || '')) return;
    setCrossDocumentIds((previous) => (
      previous.includes(normalizedId)
        ? previous.filter((item) => item !== normalizedId)
        : [...previous, normalizedId].slice(0, 4)
    ));
  }, [docId]);
  const toggleCrossDocumentMenu = useCallback(() => {
    setCrossDocumentMenuOpen((open) => {
      if (!open) void loadCrossDocumentCandidates();
      return !open;
    });
  }, [loadCrossDocumentCandidates]);
  useEffect(() => {
    setCrossDocumentIds((previous) => previous.filter((item) => item !== String(docId || '')));
    setCrossDocumentMenuOpen(false);
  }, [docId]);

  const openUploadHome = useCallback(() => {
    uploadStartsNewChatRef.current = false;
    setSelectedParseRoute(loadStoredParseRoute());
    setSidebarMode('history');
    startNewChat();
  }, [startNewChat]);

  const ensureLocalParserReady = useCallback(async () => {
    try {
      const response = await fetch('/api/runtime/addons/local-parser/status');
      if (response.ok) {
        const status = await response.json();
        if (status?.ready) return true;
      }
    } catch {
      // The install dialog presents the retriable backend status to the user.
    }
    setIsLocalParserInstallOpen(true);
    return false;
  }, []);

  const handleUploadRouteChange = useCallback((route) => {
    const nextRoute = saveStoredParseRoute(route);
    setSelectedParseRoute(nextRoute);
    if (nextRoute === 'local') {
      void ensureLocalParserReady();
    } else {
      pendingLocalParserUploadRef.current = null;
      setIsLocalParserInstallOpen(false);
    }
  }, [ensureLocalParserReady]);

  const handleChooseUploadFile = useCallback(async (startsNewChat = true) => {
    preloadPDFViewer();
    const shouldStartNewChat = typeof startsNewChat === 'boolean' ? startsNewChat : true;
    if (selectedParseRoute === 'local') {
      const localReady = await ensureLocalParserReady();
      if (!localReady) {
        pendingLocalParserUploadRef.current = shouldStartNewChat;
        return;
      }
    }
    uploadStartsNewChatRef.current = shouldStartNewChat;
    fileInputRef.current?.click();
  }, [ensureLocalParserReady, fileInputRef, selectedParseRoute]);

  const handleLocalParserInstallClose = useCallback(() => {
    pendingLocalParserUploadRef.current = null;
    setIsLocalParserInstallOpen(false);
  }, []);

  const handleLocalParserReady = useCallback(() => {
    setIsLocalParserInstallOpen(false);
    const shouldStartNewChat = pendingLocalParserUploadRef.current;
    pendingLocalParserUploadRef.current = null;
    if (typeof shouldStartNewChat !== 'boolean') return;
    uploadStartsNewChatRef.current = shouldStartNewChat;
    fileInputRef.current?.click();
  }, [fileInputRef]);

  const handleUploadInputChange = useCallback((event) => {
    if (!event.target.files?.[0]) return;
    if (uploadStartsNewChatRef.current) startNewChat();
    handleFileUpload(event, { parseRoute: selectedParseRoute });
  }, [handleFileUpload, selectedParseRoute, startNewChat]);
  const docInfoParseIdentity = getParseIdentity(docInfo?.parse_manifest);
  const deepParseStatusMatchesDocument = Boolean(
    deepParseStatus
    && (!deepParseStatus.doc_id || String(deepParseStatus.doc_id) === String(docId || ''))
    && (
      !docInfo?.parse_manifest
      || !deepParseStatus?.parse_manifest
      || getStatusParseIdentity(deepParseStatus) === docInfoParseIdentity
    )
  );
  const currentDeepParseStatus = deepParseStatusMatchesDocument ? deepParseStatus : null;
  const documentParseManifest = currentDeepParseStatus?.parse_manifest || docInfo?.parse_manifest || null;
  const documentParseIdentity = getParseIdentity(documentParseManifest);
  const currentTranslationCredentials = getChatCredentials?.() || {};
  const translationProviderId = String(currentTranslationCredentials.providerId || 'openai');
  const translationModelId = String(currentTranslationCredentials.modelId || 'gpt-4o');
  const pretranslateAutoIdentity = buildPretranslateAutoIdentity({
    docId,
    parseIdentity: documentParseIdentity,
    providerId: translationProviderId,
    modelId: translationModelId,
  });
  const blockIndexMatchesCurrentParse = blockIndexMatchesParseContext({
    blockIndex,
    docId,
    parseManifest: documentParseManifest,
  });
  if (
    parseContextRef.current.docId !== (docId || '')
    || parseContextRef.current.parseIdentity !== documentParseIdentity
  ) {
    parseContextRef.current = {
      docId: docId || '',
      parseIdentity: documentParseIdentity,
      epoch: parseContextRef.current.epoch + 1,
    };
  }
  const isCurrentParseContext = useCallback((context) => (
    parseContextRef.current.docId === context.docId
    && parseContextRef.current.parseIdentity === context.parseIdentity
    && parseContextRef.current.epoch === context.epoch
  ), []);

  useEffect(() => {
    blockTranslationEpochRef.current += 1;
    pretranslateRunRef.current += 1;
    pretranslateAbortRef.current?.abort();
    pretranslateAbortRef.current = null;
    pretranslateStartedDocRef.current = null;
    setBlockTranslations({});
    setBlockTranslationsLoaded(false);
    setBlockTranslationsLoadedIdentity('');
    setFailedTranslationBlockIds(new Set());
    setTranslatingBlockIds(new Set());
    setBlockTranslateLoading(false);
    setBlockTranslateError('');
    setPretranslateNotice('');
    setPretranslateError('');
    setPretranslateProgress({ running: false, done: 0, total: 0 });
  }, [translationModelId, translationProviderId]);
  const documentParseReady = typeof currentDeepParseStatus?.parse_ready === 'boolean'
    ? currentDeepParseStatus.parse_ready
    : typeof docInfo?.parse_ready === 'boolean'
      ? docInfo.parse_ready
      : documentParseManifest?.status === 'ready';
  const currentParseStatus = String(
    currentDeepParseStatus?.status || documentParseManifest?.status || ''
  ).trim().toLowerCase();
  const isMinerUFullRouteFailed = isMinerUFullRoute(documentParseManifest)
    && currentParseStatus === 'failed';
  const canResumeMinerUResultDownload = isMinerUFullRouteFailed
    && currentDeepParseStatus?.resume_available === true
    && currentDeepParseStatus?.resume_kind === 'result_download';
  const isMinerUFullRouteCancelled = isMinerUFullRoute(documentParseManifest)
    && currentParseStatus === 'cancelled';
  const isMinerUFullRoutePending = isMinerUFullRoute(documentParseManifest)
    && (!documentParseReady || documentParseManifest?.status !== 'ready');
  const minerUParsePendingNotice = getMinerUParsePendingNotice(documentParseManifest, currentDeepParseStatus);
  const isDocumentParseIdentityHydrating = Boolean(
    docId
    && (
      parseIdentityHydration.docId !== String(docId)
      || parseIdentityHydration.settled !== true
    )
  );
  const isChatInteractionLocked = isMinerUFullRoutePending || isDocumentParseIdentityHydrating;
  const chatInteractionLockedNotice = isMinerUFullRoutePending
    ? minerUParsePendingNotice
    : '正在同步文档解析状态，请稍候...';
  const isLegacyParseManifest = Boolean(documentParseManifest?.metadata?.legacy_inferred);
  const primaryParseRoute = String(documentParseManifest?.resolved_route || '').trim().toLowerCase();
  const isNewLocalPrimaryRoute = Boolean(
    documentParseManifest && !isLegacyParseManifest && primaryParseRoute === 'local'
  );
  const isNewMinerUPrimaryRoute = Boolean(
    documentParseManifest && !isLegacyParseManifest && primaryParseRoute === 'mineru'
  );
  const shouldPollDeepParseStatus = shouldPollMinerUStatus({
    status: currentDeepParseStatus?.status,
    primaryMinerURoute: isNewMinerUPrimaryRoute,
    routePending: isMinerUFullRoutePending,
    routeFailed: isMinerUFullRouteFailed,
    routeCancelled: isMinerUFullRouteCancelled,
  });
  const canUseLegacyMinerUActions = !documentParseManifest || isLegacyParseManifest;
  const canPublishPendingMinerURag = isNewMinerUPrimaryRoute
    && documentParseManifest?.stage === 'awaiting_rag_index';
  const requiresMinerURagSource = Boolean(
    isNewMinerUPrimaryRoute
    || canPublishPendingMinerURag
    || currentDeepParseStatus?.active_mineru
  );
  const minerUActionLockedNotice = isNewLocalPrimaryRoute
    ? '当前文档已按本地路线完成解析；如需 MinerU，请重新上传时选择 MinerU 全程解析'
    : '当前文档由 MinerU 全程路线管理；请等待统一发布，或重新上传时选择其他路线';
  const [pendingHistoryId, setPendingHistoryId] = useState(null);
  const handleHistorySessionClick = useCallback(async (item) => {
    setPendingHistoryId(item.id);
    try {
      await loadSession(item);
    } finally {
      setPendingHistoryId((currentId) => currentId === item.id ? null : currentId);
    }
  }, [loadSession]);

  // ========== PDF 状态 Hook（需求 1.1） ==========
  const pdfState = usePDFState({
    docId,
    docInfo,
    documentIdentity: documentParseIdentity,
    parseGeneration: documentParseManifest?.generation || '',
    documentSourceHash: documentParseManifest?.source_hash || '',
    useRerank: useRerankSetting,
    rerankerModel,
    getRerankCredentials,
    getEmbeddingConfig,
    embeddingApiKey: getEmbeddingApiKey(),
    apiKey,
  });
  const {
    currentPage, setCurrentPage,
    pdfScale, setPdfScale,
    selectedText, setSelectedText,
    showTextMenu, setShowTextMenu,
    searchQuery, setSearchQuery,
    searchResults,
    currentResultIndex,
    isSearching,
    searchStatus,
    searchHistory,
    activeHighlight, setActiveHighlight,
    pdfContainerRef,
    handleSearch, dismissSearchStatus, focusResult, handleCitationClick,
    formatSimilarity, renderHighlightedSnippet,
  } = pdfState;

  useEffect(() => {
    setDocumentHighlights(readDocumentHighlights(docId));
    setDocumentNotes(readDocumentNotes(docId));
    setPendingUserNoteRevealId('');
    setAnnotationTool(null);
    setSelectedSavedHighlightId('');
    setAutoAnnotationRevision(0);
    selectedPdfHighlightAnchorRef.current = null;
    pendingAutoAnnotationRef.current = null;
  }, [docId]);

  useEffect(() => {
    let cancelled = false;
    const requestContext = {
      docId: docId || '',
      parseIdentity: documentParseIdentity,
      epoch: parseContextRef.current.epoch,
    };
    const isActiveRequest = () => !cancelled && isCurrentParseContext(requestContext);
    blockTranslationEpochRef.current += 1;

    if (!docId) {
      setBlockIndex(null);
      setBlockIndexError('');
      setBlockIndexLoading(false);
      setReadingOutline(null);
      setReadingOutlineError('');
      setReadingOutlineFallbackNotice('');
      setReadingOutlineLoading(false);
      setSectionOutline(null);
      setSectionOutlineError('');
      setSectionOutlineFallbackNotice('');
      setSectionOutlineLoading(false);
      setActiveReadingNodeId(null);
      setVisitedReadingNodeIds(new Set());
      setActiveSectionNodeId(null);
      setVisitedSectionNodeIds(new Set());
      setBlockTranslations({});
      setBlockTranslateError('');
      setPretranslateNotice('');
      setPretranslateError('');
      setBlockTranslateLoading(false);
      setTranslatingBlockIds(new Set());
      setBlockTranslationsLoaded(false);
      setBlockTranslationsLoadedIdentity('');
      setPretranslateProgress({ running: false, done: 0, total: 0 });
      setFailedTranslationBlockIds(new Set());
      setDeepParseStatus(null);
      setDownstreamTaskStatuses({});
      setDeepParseNotice('');
      setRagIndexError('');
      setHoveredReadingBlockId(null);
      setPinnedReadingBlockId(null);
      pretranslateRunRef.current += 1;
      pretranslateAbortRef.current?.abort();
      pretranslateAbortRef.current = null;
      pretranslateStartedDocRef.current = null;
      setSidebarMode('history');
      return () => {};
    }

    setBlockIndex(null);
    setBlockIndexError('');
    setBlockIndexLoading(true);
    setReadingOutline(null);
    setReadingOutlineError('');
    setReadingOutlineFallbackNotice('');
    setReadingOutlineLoading(false);
    setSectionOutline(null);
    setSectionOutlineError('');
    setSectionOutlineFallbackNotice('');
    setSectionOutlineLoading(false);
    setActiveReadingNodeId(null);
    setVisitedReadingNodeIds(new Set());
    setActiveSectionNodeId(null);
    setVisitedSectionNodeIds(new Set());
    setBlockTranslations({});
    setBlockTranslateError('');
    setPretranslateNotice('');
    setPretranslateError('');
    setBlockTranslateLoading(false);
    setTranslatingBlockIds(new Set());
    setBlockTranslationsLoaded(false);
    setBlockTranslationsLoadedIdentity('');
    setPretranslateProgress({ running: false, done: 0, total: 0 });
    setFailedTranslationBlockIds(new Set());
    setDeepParseStatus(null);
    setDownstreamTaskStatuses({});
    setDeepParseNotice('');
    setRagIndexStatus(null);
    setRagIndexNotice('');
    setRagIndexError('');
    setRagIndexBusy(false);
    setEmbeddingConflictRecovery({ messageId: null, status: 'idle' });
    setHoveredReadingBlockId(null);
    setPinnedReadingBlockId(null);
    pretranslateRunRef.current += 1;
    pretranslateAbortRef.current?.abort();
    pretranslateAbortRef.current = null;
    pretranslateStartedDocRef.current = null;

    if (isMinerUFullRoutePending) {
      setBlockIndexError(minerUParsePendingNotice);
      setBlockIndexLoading(false);
      return () => {
        cancelled = true;
      };
    }

    fetch(`${API_BASE_URL}/documents/${docId}/blocks?t=${Date.now()}`)
      .then(async (res) => {
        if (!res.ok) {
          const detail = await res.json().catch(() => ({}));
          const error = new Error(detail?.detail || `HTTP ${res.status}`);
          error.status = res.status;
          throw error;
        }
        return res.json();
      })
      .then((data) => {
        if (!isActiveRequest()) return;
        if (!blockIndexMatchesParseContext({
          blockIndex: data,
          docId: requestContext.docId,
          parseManifest: documentParseManifest,
        })) {
          throw new Error('阅读结构与当前解析路线不一致');
        }
        setBlockIndex(data);
      })
      .catch((error) => {
        if (isActiveRequest()) {
          console.warn('[ImmersiveReading] blocks 加载失败', error);
          setBlockIndexError(
            isMinerUFullRoutePending || isMinerUParseGateError(error)
              ? minerUParsePendingNotice
              : '大纲加载失败'
          );
        }
      })
      .finally(() => {
        if (isActiveRequest()) setBlockIndexLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [
    docId,
    documentParseIdentity,
    blockIndexReloadKey,
    documentParseManifest,
    isCurrentParseContext,
    isMinerUFullRoutePending,
    minerUParsePendingNotice,
  ]);

  useEffect(() => {
    setParseIdentityHydration({
      docId: String(docId || ''),
      settled: !docId,
    });
  }, [docId]);

  const settleParseIdentityHydration = useCallback((requestDocId) => {
    const normalizedDocId = String(requestDocId || '');
    setParseIdentityHydration((current) => (
      current.docId === normalizedDocId && current.settled !== true
        ? { ...current, settled: true }
        : current
    ));
  }, []);

  const invalidateDeepParseStatusRequest = useCallback(() => {
    deepParseStatusRequestRef.current.sequence += 1;
    deepParseStatusRequestRef.current.controller?.abort();
    deepParseStatusRequestRef.current.controller = null;
  }, []);

  const refreshDeepParseStatus = useCallback(async () => {
    if (!docId) return null;
    const requestContext = {
      docId,
      parseIdentity: documentParseIdentity,
      epoch: parseContextRef.current.epoch,
    };
    const requestSequence = deepParseStatusRequestRef.current.sequence + 1;
    deepParseStatusRequestRef.current.sequence = requestSequence;
    deepParseStatusRequestRef.current.controller?.abort();
    const controller = new AbortController();
    deepParseStatusRequestRef.current.controller = controller;
    const releaseRequest = () => {
      if (deepParseStatusRequestRef.current.sequence === requestSequence) {
        deepParseStatusRequestRef.current.controller = null;
      }
    };
    let res;
    try {
      res = await fetch(
        `${API_BASE_URL}/documents/${requestContext.docId}/deep-parse/status?t=${Date.now()}`,
        { signal: controller.signal, cache: 'no-store' }
      );
    } catch (error) {
      releaseRequest();
      if (error?.name === 'AbortError') return null;
      if (isCurrentParseContext(requestContext)) {
        settleParseIdentityHydration(requestContext.docId);
      }
      throw error;
    }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      if (
        deepParseStatusRequestRef.current.sequence === requestSequence
        && isCurrentParseContext(requestContext)
      ) {
        settleParseIdentityHydration(requestContext.docId);
      }
      releaseRequest();
      throw new Error(data?.detail || `HTTP ${res.status}`);
    }
    if (
      deepParseStatusRequestRef.current.sequence !== requestSequence
      || !isCurrentParseContext(requestContext)
      || (data?.doc_id && String(data.doc_id) !== String(requestContext.docId))
    ) {
      releaseRequest();
      return null;
    }
    settleParseIdentityHydration(requestContext.docId);
    setDeepParseStatus(data);
    if (data?.parse_manifest || typeof data?.parse_ready === 'boolean') {
      setDocInfo((current) => {
        if (!current) return current;
        const currentDocId = current.doc_id || current.docId || current.id;
        if (currentDocId && String(currentDocId) !== String(requestContext.docId)) return current;
        if (
          current?.parse_manifest
          && getParseIdentity(current.parse_manifest) !== requestContext.parseIdentity
        ) {
          return current;
        }
        const next = {
          ...current,
          ...(data.parse_manifest ? { parse_manifest: data.parse_manifest } : {}),
          ...(typeof data.parse_ready === 'boolean' ? { parse_ready: data.parse_ready } : {}),
        };
        if (
          next.parse_ready === current.parse_ready
          && getParseIdentity(next.parse_manifest) === getParseIdentity(current.parse_manifest)
        ) {
          return current;
        }
        return next;
      });
    }
    if (data?.rag_index) {
      setRagIndexStatus(data.rag_index);
      setRagIndexError(
        data.rag_index.status === 'failed'
          ? data.rag_index.error || '问答索引处理失败'
          : ''
      );
    }
    releaseRequest();
    return data;
  }, [
    docId,
    documentParseIdentity,
    isCurrentParseContext,
    setDocInfo,
    settleParseIdentityHydration,
  ]);

  const refreshRagIndexStatus = useCallback(async () => {
    if (!docId) return null;
    const requestContext = {
      docId,
      parseIdentity: documentParseIdentity,
      epoch: parseContextRef.current.epoch,
    };
    const res = await fetch(`${API_BASE_URL}/documents/${requestContext.docId}/rag-index/status?t=${Date.now()}`);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data?.detail || `HTTP ${res.status}`);
    if (
      !isCurrentParseContext(requestContext)
      || (data?.doc_id && String(data.doc_id) !== String(requestContext.docId))
    ) {
      return null;
    }
    setRagIndexStatus(data);
    setRagIndexError(data?.status === 'failed' ? data.error || '问答索引处理失败' : '');
    return data;
  }, [docId, documentParseIdentity, isCurrentParseContext]);

  const refreshDownstreamTaskStatuses = useCallback(async () => {
    if (!docId) return {};
    const requestContext = {
      docId,
      parseIdentity: documentParseIdentity,
      epoch: parseContextRef.current.epoch,
    };
    const purposes = ['overview', 'reading_outline', 'section_outline'];
    const responses = await Promise.all(purposes.map(async (purpose) => {
      try {
        const response = await fetch(
          `${API_BASE_URL}/documents/${requestContext.docId}/ai-tasks/${purpose}?t=${Date.now()}`,
          { cache: 'no-store' },
        );
        if (!response.ok) return [purpose, null];
        return [purpose, await response.json().catch(() => null)];
      } catch {
        return [purpose, null];
      }
    }));
    if (!isCurrentParseContext(requestContext)) return {};
    const next = Object.fromEntries(responses.filter(([, value]) => value));
    setDownstreamTaskStatuses((prev) => (
      JSON.stringify(prev) === JSON.stringify(next) ? prev : next
    ));
    return next;
  }, [docId, documentParseIdentity, isCurrentParseContext]);

  useEffect(() => {
    if (!docId) return () => {};
    refreshDownstreamTaskStatuses().catch(() => {});
    if (!showAiProcessingPanel) return () => {};
    return startVisiblePoll(
      () => refreshDownstreamTaskStatuses().catch(() => {}),
      3000,
      { immediate: false },
    );
  }, [docId, refreshDownstreamTaskStatuses, showAiProcessingPanel]);

  const refreshReadingBlocksAfterDeepParse = useCallback(() => {
    // MinerU 发布后旧 block id 不再可信，先让所有旧翻译请求失效。
    blockTranslationEpochRef.current += 1;
    pretranslateRunRef.current += 1;
    pretranslateAbortRef.current?.abort();
    pretranslateAbortRef.current = null;
    pretranslateStartedDocRef.current = null;
    setBlockTranslations({});
    setBlockTranslationsLoaded(false);
    setBlockTranslationsLoadedIdentity('');
    setFailedTranslationBlockIds(new Set());
    setTranslatingBlockIds(new Set());
    setBlockTranslateError('');
    setBlockTranslateLoading(false);
    setPretranslateNotice('');
    setPretranslateError('');
    setPretranslateProgress({ running: false, done: 0, total: 0 });
    setReadingOutline(null);
    setSectionOutline(null);
    readingOutlineCacheRef.current.clear();
    sectionOutlineCacheRef.current.clear();
    setReadingOutlineFallbackNotice('');
    setSectionOutlineFallbackNotice('');
    setReadingOutlineReloadKey((value) => value + 1);
    setSectionOutlineReloadKey((value) => value + 1);
    setActiveReadingNodeId(null);
    setVisitedReadingNodeIds(new Set());
    setActiveSectionNodeId(null);
    setVisitedSectionNodeIds(new Set());
    setPinnedReadingBlockId(null);
    setHoveredReadingBlockId(null);
    setBlockIndexReloadKey((value) => value + 1);
    // 深度解析完成后后端已让 logical_figures 缓存失效，这里同步清掉前端的
    // 速览缓存，避免用户已经打开过速览时仍显示深度解析前的旧图表结果。
    clearOverviewCache?.(docId);
  }, [clearOverviewCache, docId]);

  const refreshReadingBlocksAfterVisualSupplement = useCallback(() => {
    // A visual supplement is additive: it appends figure evidence but keeps
    // the parser identity and existing text block ids intact. Reload the index
    // so the new evidence is visible, while allowing any page/full-document
    // translation already in progress to finish and stay recoverable.
    setBlockIndexReloadKey((value) => value + 1);
  }, []);

  useEffect(() => {
    const revision = String(
      overview?.visual_supplement_revision
      || overview?.figure_meta?.visual_supplement_revision
      || ''
    ).trim();
    if (!docId || !revision) return;
    const token = `${docId}:${documentParseIdentity}:${revision}`;
    if (visualSupplementRevisionRef.current === token) return;
    visualSupplementRevisionRef.current = token;
    refreshReadingBlocksAfterVisualSupplement();
  }, [docId, documentParseIdentity, overview, refreshReadingBlocksAfterVisualSupplement]);

  useEffect(() => {
    return invalidateDeepParseStatusRequest;
  }, [invalidateDeepParseStatusRequest]);

  useEffect(() => {
    let cancelled = false;
    if (!docId) return () => {};
    if (!shouldPollDeepParseStatus) {
      refreshDeepParseStatus().catch((error) => {
        if (!cancelled) {
          console.warn('[DeepParse] 状态加载失败', error);
        }
      });
      refreshRagIndexStatus().catch((error) => {
        if (!cancelled) {
          console.warn('[RagIndex] 状态加载失败', error);
        }
      });
    }
    return () => {
      cancelled = true;
    };
  }, [docId, refreshDeepParseStatus, refreshRagIndexStatus, shouldPollDeepParseStatus]);

  useEffect(() => {
    if (!docId || !shouldPollDeepParseStatus) {
      return () => {};
    }
    return startVisiblePoll(async () => {
      try {
        const data = await refreshDeepParseStatus();
        if (!data) return true;
        if (['ready', 'partial_ready'].includes(data.status) && data.active_mineru && data.parse_ready === true) {
          setDeepParseNotice(
            data.status === 'partial_ready'
              ? 'MinerU 已完成，部分页面未识别出可用文本；阅读、问答和速览将基于其余页面'
              : data.recommend_rag_index_rebuild
              ? 'MinerU 深度解析完成，阅读结构、大纲与速览图表均已刷新；建议重建问答索引以启用结构化表格证据'
              : 'MinerU 深度解析完成，阅读结构、大纲与速览图表均已刷新'
          );
          refreshReadingBlocksAfterDeepParse();
          return false;
        }
        if (['ready', 'partial_ready'].includes(data.status) && data.active_mineru) {
          setDeepParseNotice(getMinerUParsePendingNotice(data.parse_manifest, data));
          return false;
        }
        if (data.status === 'failed') {
          setDeepParseNotice(data.error || 'MinerU 深度解析失败');
          return false;
        }
        if (data.status === 'cancelled') {
          setDeepParseNotice('MinerU 深度解析已取消');
          return false;
        }
        return true;
      } catch (error) {
        setDeepParseNotice(error.message || 'MinerU 深度解析状态同步失败');
        return true;
      }
    }, 2500);
  }, [
    docId,
    documentParseIdentity,
    refreshDeepParseStatus,
    refreshReadingBlocksAfterDeepParse,
    shouldPollDeepParseStatus,
  ]);

  const handleStartMinerUDeepParse = useCallback(async (options = {}) => {
    const retryFullRoute = options?.retryFullRoute === true;
    const canRetryFullRoute = retryFullRoute && isNewMinerUPrimaryRoute && isMinerUFullRouteFailed;
    const canResumeResultDownload = canRetryFullRoute
      && currentDeepParseStatus?.resume_available === true
      && currentDeepParseStatus?.resume_kind === 'result_download';
    if (!canUseLegacyMinerUActions && !canRetryFullRoute) {
      setDeepParseNotice(minerUActionLockedNotice);
      return;
    }
    if (!docId || ['queued', 'running'].includes(currentDeepParseStatus?.status)) return;
    const activeMinerU = Boolean(currentDeepParseStatus?.active_mineru);
    const parseTarget = currentDeepParseStatus?.access_mode === 'direct'
      ? 'MinerU 官方 API'
      : '你配置的 MinerU Worker 服务';
    const confirmed = await confirmAction(
      canResumeResultDownload
        ? {
          title: '重新获取解析结果',
          description: '会复用 MinerU 已完成的远端任务，只重新下载结果，不会再上传 PDF。',
          confirmLabel: '继续下载',
          tone: 'caution',
        }
        : canRetryFullRoute
        ? {
          title: '重新解析这份文档',
          description: `会把当前 PDF 重新上传到${parseTarget}。成功后，下面这些内容会一起换成新结果。`,
          impacts: ['正文', '阅读', '问答索引', '大纲', '总结', '翻译', '速览'],
          confirmLabel: '继续解析',
          tone: 'caution',
        }
        : activeMinerU
        ? {
          title: '重新运行深度解析',
          description: `会把当前 PDF 上传到${parseTarget}，并替换阅读块、大纲、翻译缓存和速览图表。`,
          impacts: ['阅读块', '大纲', '翻译缓存', '速览'],
          confirmLabel: '继续解析',
          tone: 'caution',
        }
        : {
          title: '开始 MinerU 深度解析',
          description: `会把当前 PDF 上传到${parseTarget}，生成带坐标的结构化结果，速览图表也会一起升级。`,
          confirmLabel: '开始解析',
          tone: 'caution',
        }
    );
    if (!confirmed) return;

    setDeepParseNotice('');
    const requestContext = {
      docId,
      parseIdentity: documentParseIdentity,
      epoch: parseContextRef.current.epoch,
    };
    invalidateDeepParseStatusRequest();
    try {
      const res = await fetch(`${API_BASE_URL}/documents/${requestContext.docId}/deep-parse`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider: 'mineru', force: activeMinerU || canRetryFullRoute }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.detail || `HTTP ${res.status}`);
      if (
        !isCurrentParseContext(requestContext)
        || (data?.doc_id && String(data.doc_id) !== String(requestContext.docId))
      ) return;
      setDeepParseStatus(data);
      if (['ready', 'partial_ready'].includes(data.status) && data.active_mineru && data.parse_ready === true) {
        setDeepParseNotice(
          data.status === 'partial_ready'
            ? 'MinerU 已完成，部分页面未识别出可用文本；阅读、问答和速览将基于其余页面'
            : data.recommend_rag_index_rebuild
            ? 'MinerU 深度解析已就绪，阅读结构、大纲与速览图表均已刷新；建议重建问答索引以启用结构化表格证据'
            : 'MinerU 深度解析已就绪，阅读结构、大纲与速览图表均已刷新'
        );
        refreshReadingBlocksAfterDeepParse();
      } else if (['ready', 'partial_ready'].includes(data.status) && data.active_mineru) {
        setDeepParseNotice(getMinerUParsePendingNotice(data.parse_manifest, data));
      } else if (data.resume_kind === 'result_download' || data.stage === 'resuming_result_download') {
        setDeepParseNotice('正在重新获取 MinerU 已完成的解析结果，不会重新上传 PDF');
      } else {
        setDeepParseNotice('MinerU 深度解析已开始，可继续阅读，完成后阅读结构与速览图表会自动刷新');
      }
    } catch (error) {
      if (!isCurrentParseContext(requestContext)) return;
      setDeepParseNotice(error.message || 'MinerU 深度解析启动失败');
      setDeepParseStatus((prev) => ({ ...(prev || {}), status: 'failed', error: error.message || '启动失败' }));
    }
  }, [
    canUseLegacyMinerUActions,
    currentDeepParseStatus?.access_mode,
    currentDeepParseStatus?.active_mineru,
    currentDeepParseStatus?.resume_available,
    currentDeepParseStatus?.resume_kind,
    currentDeepParseStatus?.status,
    docId,
    documentParseIdentity,
    invalidateDeepParseStatusRequest,
    isCurrentParseContext,
    isMinerUFullRouteFailed,
    isNewMinerUPrimaryRoute,
    minerUActionLockedNotice,
    confirmAction,
    refreshReadingBlocksAfterDeepParse,
  ]);

  const handleCancelMinerUDeepParse = useCallback(async () => {
    if (!docId || !['queued', 'running'].includes(currentDeepParseStatus?.status)) return;
    const requestContext = {
      docId,
      parseIdentity: documentParseIdentity,
      epoch: parseContextRef.current.epoch,
    };
    invalidateDeepParseStatusRequest();
    try {
      const res = await fetch(`${API_BASE_URL}/documents/${requestContext.docId}/deep-parse/cancel`, {
        method: 'POST',
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.detail || `HTTP ${res.status}`);
      if (
        !isCurrentParseContext(requestContext)
        || (data?.doc_id && String(data.doc_id) !== String(requestContext.docId))
      ) return;
      setDeepParseStatus(data);
      setDeepParseNotice('MinerU 深度解析已取消');
    } catch (error) {
      if (!isCurrentParseContext(requestContext)) return;
      setDeepParseNotice(error.message || 'MinerU 深度解析取消失败');
    }
  }, [
    currentDeepParseStatus?.status,
    docId,
    documentParseIdentity,
    invalidateDeepParseStatusRequest,
    isCurrentParseContext,
  ]);

  const handleRebuildMinerURagIndex = useCallback(async (options = {}) => {
    const forceEmbeddingRebuild = options?.forceEmbeddingRebuild === true;
    const conflictMessageId = options?.conflictMessageId ?? null;
    const rebuildLocalIndex = Boolean(
      !requiresMinerURagSource && (
        (forceEmbeddingRebuild && primaryParseRoute !== 'mineru')
        || (ragIndexStatus?.upgrade_required && ragIndexStatus?.index_source !== 'mineru')
      )
    );
    if (isNewLocalPrimaryRoute && !rebuildLocalIndex) {
      setRagIndexNotice('当前文档已固定为本地解析路线，不能单独切换到 MinerU 问答索引');
      return;
    }
    if (isNewMinerUPrimaryRoute && !canPublishPendingMinerURag && !forceEmbeddingRebuild && !ragIndexStatus?.upgrade_required) {
      setRagIndexNotice('当前 MinerU 全程解析会统一管理问答索引，无需单独重建');
      return;
    }
    if (!docId || ragIndexBusy) return;
    if (!rebuildLocalIndex && !canPublishPendingMinerURag && !currentDeepParseStatus?.active_mineru && !forceEmbeddingRebuild) {
      const message = '请先完成 MinerU 深度解析';
      setRagIndexNotice(message);
      setRagIndexError(message);
      return;
    }
    const embedConfig = getEmbeddingConfig?.() || {};
    if (!embedConfig.isValid) {
      const message = '请先在模型设置里选择可用的 Embedding 模型';
      setRagIndexNotice(message);
      setRagIndexError(message);
      return;
    }
    const missingEmbeddingCredential = getMissingEmbeddingCredential();
    if (missingEmbeddingCredential) {
      setRagIndexNotice(missingEmbeddingCredential.message);
      setRagIndexError(missingEmbeddingCredential.message);
      return;
    }
    const embeddingCredentials = getEmbeddingCredentialState();
    const embeddingModel = embedConfig.compositeKey;
    const embeddingProvider = embedConfig.providerId || '';
    const embeddingApiKeyValue = embeddingCredentials.apiKey || '';
    const embeddingApiHost = embeddingCredentials.apiHost || '';

    setRagIndexBusy(true);
    if (conflictMessageId !== null) {
      setEmbeddingConflictRecovery({ messageId: conflictMessageId, status: 'rebuilding' });
    }
    setRagIndexError('');
    setRagIndexNotice(forceEmbeddingRebuild
      ? '正在评估按当前 Embedding 配置同步问答索引的成本...'
      : rebuildLocalIndex ? '正在评估本地问答索引升级成本...' : '正在评估 MinerU 问答索引重建成本...');
    try {
      const estimateRes = await fetch(`${API_BASE_URL}/documents/${docId}/rag-index/rebuild`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ estimate_only: true, source: rebuildLocalIndex ? 'local' : 'mineru' }),
      });
      const estimateData = await estimateRes.json().catch(() => ({}));
      if (!estimateRes.ok) {
        const detail = estimateData?.detail;
        throw new Error(typeof detail === 'string' ? detail : detail?.message || `HTTP ${estimateRes.status}`);
      }
      if (!estimateData.can_rebuild) {
        const failures = (estimateData.quality_failures || []).join('、') || '质量门未通过';
        throw new Error(`${rebuildLocalIndex ? '当前正文' : 'MinerU 结果'}暂不能重建问答索引：${failures}`);
      }
      const estimate = estimateData.estimate || {};
      const confirmed = await confirmAction({
        title: forceEmbeddingRebuild ? '按当前模型重建问答索引' : rebuildLocalIndex ? '升级本地问答索引' : '重建 MinerU 问答索引',
        description: [
          forceEmbeddingRebuild
            ? '当前问答索引曾用其他 Embedding 身份构建。将保留 MinerU 解析结果，只按当前配置重建向量索引。'
            : rebuildLocalIndex
              ? '将使用当前解析路线的正文阅读块升级问答索引。'
              : '将使用 MinerU 结构化结果重建问答索引。',
          `预计重新嵌入约 ${estimate.estimated_embedding_tokens || 0} tokens，约 ${estimate.estimated_chunk_count || 0} 个分块，表格 ${estimate.structured_table_count || 0} 个，大约 1-3 分钟。`,
          '历史对话中的引用可能发生偏移。阅读侧翻译、大纲和速览不受影响。重建期间旧问答索引仍可使用。',
        ],
        confirmLabel: '开始重建',
        tone: 'caution',
      });
      if (!confirmed) {
        setRagIndexNotice('问答索引尚未发布，可稍后继续');
        setRagIndexError('');
        if (conflictMessageId !== null) {
          setEmbeddingConflictRecovery({ messageId: conflictMessageId, status: 'idle' });
        }
        return;
      }
      setRagIndexNotice(forceEmbeddingRebuild
        ? '正在按当前 Embedding 配置同步问答索引...'
        : rebuildLocalIndex ? '正在升级本地问答索引...' : '正在重建 MinerU 问答索引...');
      const rebuildRes = await fetch(`${API_BASE_URL}/documents/${docId}/rag-index/rebuild`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          embedding_model: embeddingModel,
          embedding_provider: embeddingProvider,
          embedding_api_key: embeddingApiKeyValue,
          embedding_api_host: embeddingApiHost,
          source: rebuildLocalIndex ? 'local' : 'mineru',
        }),
      });
      const rebuildData = await rebuildRes.json().catch(() => ({}));
      if (!rebuildRes.ok) {
        const detail = rebuildData?.detail;
        throw new Error(typeof detail === 'string' ? detail : detail?.message || `HTTP ${rebuildRes.status}`);
      }
      setRagIndexStatus(rebuildData.rag_index || null);
      setRagIndexError('');
      setRagIndexNotice(
        forceEmbeddingRebuild
          ? '问答索引已按当前 Embedding 配置同步，可重新提问'
          : rebuildLocalIndex
          ? '本地问答索引已升级，正文、表格与页码检索已按当前阅读结构重建'
          : 'MinerU 问答索引已重建，表格问答会优先使用结构化证据'
      );
      if (conflictMessageId !== null) {
        setEmbeddingConflictRecovery({ messageId: conflictMessageId, status: 'completed' });
      }
      if (!rebuildLocalIndex && !forceEmbeddingRebuild) {
        const status = await refreshDeepParseStatus();
        if (status?.active_mineru && status?.parse_ready === true) {
          setDeepParseNotice('MinerU 问答索引已发布，阅读结构、大纲、翻译、速览和问答现已同步切换');
          refreshReadingBlocksAfterDeepParse();
        }
      }
    } catch (error) {
      const message = error.message || '问答索引重建失败，已保留原索引';
      setRagIndexNotice(message);
      setRagIndexError(message);
      if (conflictMessageId !== null) {
        setEmbeddingConflictRecovery({ messageId: conflictMessageId, status: 'failed' });
      }
    } finally {
      setRagIndexBusy(false);
    }
  }, [
    canPublishPendingMinerURag,
    currentDeepParseStatus?.active_mineru,
    docId,
    getEmbeddingCredentialState,
    getEmbeddingConfig,
    getMissingEmbeddingCredential,
    isNewLocalPrimaryRoute,
    isNewMinerUPrimaryRoute,
    primaryParseRoute,
    ragIndexBusy,
    ragIndexStatus?.index_source,
    ragIndexStatus?.upgrade_required,
    refreshDeepParseStatus,
    refreshReadingBlocksAfterDeepParse,
    requiresMinerURagSource,
    confirmAction,
  ]);

  const handleRollbackRagIndex = useCallback(async () => {
    if (!canUseLegacyMinerUActions) {
      setRagIndexNotice(minerUActionLockedNotice);
      return;
    }
    if (!docId || ragIndexBusy) return;
    const confirmed = await confirmAction({
      title: '回退到本地问答索引',
      description: '阅读侧 MinerU 解析不受影响，只会把问答索引换回本地 PDF 解析结果。',
      confirmLabel: '确认回退',
      tone: 'caution',
    });
    if (!confirmed) return;
    setRagIndexBusy(true);
    setRagIndexNotice('正在回退问答索引...');
    try {
      const res = await fetch(`${API_BASE_URL}/documents/${docId}/rag-index/rollback`, { method: 'POST' });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.detail || `HTTP ${res.status}`);
      setRagIndexStatus(data.rag_index || null);
      setRagIndexNotice('已回退到本地问答索引');
      refreshDeepParseStatus().catch(() => {});
    } catch (error) {
      setRagIndexNotice(error.message || '问答索引回退失败');
    } finally {
      setRagIndexBusy(false);
    }
  }, [canUseLegacyMinerUActions, confirmAction, docId, minerUActionLockedNotice, ragIndexBusy, refreshDeepParseStatus]);

  useEffect(() => {
    let cancelled = false;
    const requestContext = {
      docId: docId || '',
      parseIdentity: documentParseIdentity,
      epoch: parseContextRef.current.epoch,
    };
    const isActiveRequest = () => !cancelled && isCurrentParseContext(requestContext);
    if (!docId || !blockIndex) {
      setReadingOutline(null);
      setReadingOutlineError(isMinerUFullRoutePending ? minerUParsePendingNotice : '');
      setReadingOutlineFallbackNotice('');
      setReadingOutlineLoading(false);
      return () => {};
    }

    const requestId = readingOutlineRequestRef.current + 1;
    readingOutlineRequestRef.current = requestId;
    const { headers, canCallModel, providerId, modelId } = getChatRequestConfig();
    const visualCredentials = getVisualCredentials();
    const shouldForce = canCallModel && readingOutlineForceRef.current;
    readingOutlineForceRef.current = false;
    const shouldAutoGenerate = canCallModel && aiAutoProcess && autoOutlineSummary;
    const cacheKey = buildOutlineCacheKey(docId, documentParseIdentity, blockIndex);
    const cachedOutline = readingOutlineCacheRef.current.get(cacheKey);
    if (
      !shouldForce
      && cachedOutline
      && isReusableOutlineResult(cachedOutline, 'reading', providerId, modelId)
    ) {
      setReadingOutline(cachedOutline);
      setReadingOutlineError('');
      setReadingOutlineFallbackNotice(
        getOutlineResultNotice(cachedOutline, 'AI 总结生成失败，当前显示本地基础结果')
      );
      setReadingOutlineLoading(false);
      return () => {
        cancelled = true;
      };
    }

    setReadingOutlineLoading(true);
    setReadingOutlineError('');
    setReadingOutlineFallbackNotice('');

    const requestOutline = async (method, force = false) => {
      const url = method === 'POST'
        ? `${API_BASE_URL}/documents/${docId}/reading-outline`
        : `${API_BASE_URL}/documents/${docId}/reading-outline?t=${Date.now()}`;
      const res = await fetch(url, method === 'POST' ? {
        method,
        headers,
        body: JSON.stringify({
          force,
          visual_strategy: visualCredentials.strategy,
          visual_enabled: visualCredentials.policyVisionCapable,
          visual_provider: visualCredentials.providerId,
          visual_model: visualCredentials.modelId,
          visual_api_key: visualCredentials.apiKey,
          visual_api_host: visualCredentials.apiHost,
          local_visual_provider: visualCredentials.local?.providerId || '',
          local_visual_model: visualCredentials.local?.modelId || '',
          local_visual_api_key: visualCredentials.local?.apiKey || '',
          local_visual_api_host: visualCredentials.local?.apiHost || '',
        }),
      } : { method });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.detail || `HTTP ${res.status}`);
      return data;
    };

    (async () => {
      let data;
      if (shouldForce) {
        data = await requestOutline('POST', true);
      } else {
        data = await requestOutline('GET');
        if (shouldAutoGenerate && !isReusableOutlineResult(data, 'reading', providerId, modelId)) {
          data = await requestOutline('POST', false);
        }
      }
      return data;
    })()
      .then((data) => {
        if (isActiveRequest() && readingOutlineRequestRef.current === requestId) {
          readingOutlineCacheRef.current.set(cacheKey, data);
          setReadingOutline(data);
          setReadingOutlineFallbackNotice(
            getOutlineResultNotice(data, 'AI 总结生成失败，当前显示本地基础结果')
          );
        }
      })
      .catch((error) => {
        if (isActiveRequest() && readingOutlineRequestRef.current === requestId) {
          console.warn('[ImmersiveReading] AI 大纲加载失败', error);
          setReadingOutline(buildClientReadingFallback(blockIndex));
          setReadingOutlineError('');
          setReadingOutlineFallbackNotice(
            shouldForce || shouldAutoGenerate ? 'AI 总结生成失败，当前显示本地基础结果' : ''
          );
        }
      })
      .finally(() => {
        if (isActiveRequest() && readingOutlineRequestRef.current === requestId) {
          setReadingOutlineLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [
    docId,
    documentParseIdentity,
    blockIndex,
    getChatRequestConfig,
    getVisualCredentials,
    isCurrentParseContext,
    readingOutlineReloadKey,
    aiAutoProcess,
    autoOutlineSummary,
    isMinerUFullRoutePending,
    minerUParsePendingNotice,
  ]);

  const handleRegenerateReadingOutline = useCallback(() => {
    if (isMinerUFullRoutePending) {
      setReadingOutlineError(minerUParsePendingNotice);
      return;
    }
    const { canCallModel, providerName } = getChatRequestConfig();
    if (!canCallModel) {
      setReadingOutlineError(`请先为 ${providerName} 配置 API Key 后再重新生成`);
      return;
    }
    readingOutlineForceRef.current = true;
    readingOutlineCacheRef.current.clear();
    setReadingOutlineFallbackNotice('');
    setReadingOutlineReloadKey((value) => value + 1);
  }, [getChatRequestConfig, isMinerUFullRoutePending, minerUParsePendingNotice]);

  const handleRegenerateSectionOutline = useCallback(() => {
    if (isMinerUFullRoutePending) {
      setSectionOutlineError(minerUParsePendingNotice);
      return;
    }
    const { canCallModel, providerName } = getChatRequestConfig();
    if (!canCallModel) {
      setSectionOutlineError(`请先为 ${providerName} 配置 API Key 后再重新生成`);
      return;
    }
    sectionOutlineForceRef.current = true;
    setSectionOutlineReloadKey((value) => value + 1);
  }, [getChatRequestConfig, isMinerUFullRoutePending, minerUParsePendingNotice]);

  useEffect(() => {
    let cancelled = false;
    const requestContext = {
      docId: docId || '',
      parseIdentity: documentParseIdentity,
      epoch: parseContextRef.current.epoch,
    };
    const isActiveRequest = () => !cancelled && isCurrentParseContext(requestContext);
    if (!docId || !blockIndex) {
      setSectionOutline(null);
      setSectionOutlineError(isMinerUFullRoutePending ? minerUParsePendingNotice : '');
      setSectionOutlineFallbackNotice('');
      setSectionOutlineLoading(false);
      return () => {};
    }

    const requestId = sectionOutlineRequestRef.current + 1;
    sectionOutlineRequestRef.current = requestId;
    const { headers, canCallModel, providerId, modelId } = getChatRequestConfig();
    const shouldForce = canCallModel && sectionOutlineForceRef.current;
    sectionOutlineForceRef.current = false;
    const shouldAutoGenerate = canCallModel && aiAutoProcess && autoOutlineSummary;
    const cacheKey = buildOutlineCacheKey(docId, documentParseIdentity, blockIndex);
    const cachedOutline = sectionOutlineCacheRef.current.get(cacheKey);
    if (
      !shouldForce
      && cachedOutline
      && (!shouldAutoGenerate || isReusableOutlineResult(cachedOutline, 'section', providerId, modelId))
    ) {
      setSectionOutline(cachedOutline);
      setSectionOutlineError('');
      setSectionOutlineFallbackNotice(
        getOutlineResultNotice(cachedOutline, 'AI 章节大纲生成失败，当前显示文档基础结构')
      );
      setSectionOutlineLoading(false);
      return () => {
        cancelled = true;
      };
    }

    setSectionOutlineLoading(true);
    setSectionOutlineError('');
    setSectionOutlineFallbackNotice('');

    const requestOutline = async (method, force = false) => {
      const url = method === 'POST'
        ? `${API_BASE_URL}/documents/${docId}/section-outline`
        : `${API_BASE_URL}/documents/${docId}/section-outline?t=${Date.now()}`;
      const res = await fetch(url, method === 'POST' ? {
        method,
        headers,
        body: JSON.stringify({ force }),
      } : { method });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.detail || `HTTP ${res.status}`);
      return data;
    };

    (async () => {
      let data;
      if (shouldForce) {
        data = await requestOutline('POST', true);
      } else {
        data = await requestOutline('GET');
        if (shouldAutoGenerate && !isReusableOutlineResult(data, 'section', providerId, modelId)) {
          data = await requestOutline('POST', false);
        }
      }
      return data;
    })()
      .then((data) => {
        if (isActiveRequest() && sectionOutlineRequestRef.current === requestId) {
          sectionOutlineCacheRef.current.set(cacheKey, data);
          setSectionOutline(data);
          setSectionOutlineFallbackNotice(
            getOutlineResultNotice(data, 'AI 章节大纲生成失败，当前显示文档基础结构')
          );
        }
      })
      .catch((error) => {
        if (isActiveRequest() && sectionOutlineRequestRef.current === requestId) {
          console.warn('[ImmersiveReading] 章节大纲加载失败，回退启发式大纲', error);
          setSectionOutline(null);
          setSectionOutlineError('');
          setSectionOutlineFallbackNotice(
            shouldForce || shouldAutoGenerate ? 'AI 章节大纲生成失败，当前显示文档基础结构' : ''
          );
        }
      })
      .finally(() => {
        if (isActiveRequest() && sectionOutlineRequestRef.current === requestId) {
          setSectionOutlineLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [
    docId,
    documentParseIdentity,
    blockIndex,
    getChatRequestConfig,
    isCurrentParseContext,
    sectionOutlineReloadKey,
    aiAutoProcess,
    autoOutlineSummary,
    isMinerUFullRoutePending,
    minerUParsePendingNotice,
  ]);

  useEffect(() => {
    let cancelled = false;
    const translationEpoch = blockTranslationEpochRef.current;
    const requestContext = {
      docId: docId || '',
      parseIdentity: documentParseIdentity,
      epoch: parseContextRef.current.epoch,
    };
    const isActiveRequest = () => (
      !cancelled
      && blockTranslationEpochRef.current === translationEpoch
      && isCurrentParseContext(requestContext)
    );
    if (!docId || !blockIndex || !blockIndexMatchesCurrentParse) {
      setBlockTranslations({});
      setBlockTranslationsLoaded(false);
      setBlockTranslationsLoadedIdentity('');
      setFailedTranslationBlockIds(new Set());
      setBlockTranslateError('');
      setPretranslateNotice('');
      setPretranslateError('');
      return () => {};
    }
    setBlockTranslationsLoaded(false);
    setBlockTranslationsLoadedIdentity('');
    setFailedTranslationBlockIds(new Set());
    setBlockTranslateError('');
    setPretranslateNotice('');
    setPretranslateError('');

    const { headers } = getChatRequestConfig();
    fetch(`${API_BASE_URL}/documents/${docId}/blocks/translations?target_lang=zh&t=${Date.now()}`, {
      headers,
    })
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (isActiveRequest()) {
          setBlockTranslations(filterValidBlockTranslations(data?.items || {}));
          const restoredFailures = [
            ...(Array.isArray(data?.failed_block_ids) ? data.failed_block_ids : []),
            ...(Array.isArray(data?.skipped_block_ids) ? data.skipped_block_ids : []),
          ].filter(Boolean);
          setFailedTranslationBlockIds(new Set(restoredFailures));
          if (restoredFailures.length > 0) {
            setBlockTranslateError(`有 ${restoredFailures.length} 个段落尚未完成翻译，可继续补齐`);
          }
        }
      })
      .catch((error) => {
        if (isActiveRequest()) {
          console.warn('[ImmersiveReading] 翻译缓存加载失败', error);
        }
      })
      .finally(() => {
        if (isActiveRequest()) {
          setBlockTranslationsLoaded(true);
          setBlockTranslationsLoadedIdentity(pretranslateAutoIdentity);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [
    blockIndex,
    blockIndexMatchesCurrentParse,
    docId,
    documentParseIdentity,
    getChatRequestConfig,
    isCurrentParseContext,
    pretranslateAutoIdentity,
    translationModelId,
    translationProviderId,
  ]);

  const blockMap = useMemo(() => {
    const result = {};
    (blockIndex?.pages || []).forEach((page) => {
      (page.blocks || []).forEach((block) => {
        if (block?.block_id) {
          result[block.block_id] = { ...block, page: Number(page.page) || 1 };
        }
      });
    });
    return result;
  }, [blockIndex]);

  const resolveSectionOutlineAnchor = useCallback((item) => {
    if (!item) return null;
    const title = item.title || item.label || item.name || '';
    const targetPage = Number(item.evidence?.primary_page || item.page || 0) || null;
    const blocks = Object.values(blockMap);
    const rawIds = [
      item.first_block,
      item.block_id,
      ...(item.evidence?.block_ids || []),
      ...(item.evidence_block_ids || []),
    ].filter(Boolean);

    const findByTitle = ({ page = null, headingOnly = false, allowLoose = false } = {}) => {
      for (const block of blocks) {
        if (page && Number(block.page) !== Number(page)) continue;
        if (headingOnly && block.type !== 'heading') continue;
        if (isUsableSectionAnchor(block, title, { allowLoose })) {
          return block.block_id;
        }
      }
      return null;
    };

    return (
      findByTitle({ page: targetPage, headingOnly: true })
      || findByTitle({ headingOnly: true })
      || rawIds.find((blockId) => isUsableSectionAnchor(blockMap[blockId], title))
      || findByTitle({ page: targetPage, allowLoose: true })
      || findByTitle({ allowLoose: true })
      || blocks.find((block) => (
        targetPage
        && Number(block.page) === Number(targetPage)
        && block.type === 'heading'
        && !isPublicationHeaderBlock(block)
      ))?.block_id
      || rawIds.find((blockId) => {
        const block = blockMap[blockId];
        return block?.block_id && block.type !== 'artifact' && !isPublicationHeaderBlock(block);
      })
      || null
    );
  }, [blockMap]);

  const readingOutlineItems = useMemo(() => readingOutline?.items || [], [readingOutline]);
  const sectionOutlineItems = useMemo(() => sectionOutline?.items || [], [sectionOutline]);
  const blockOutlineItems = useMemo(() => blockIndex?.outline || [], [blockIndex]);
  const pdfOutlineItems = useMemo(() => {
    if (isUsefulOutline(sectionOutlineItems, sectionOutline?.source)) return sectionOutlineItems;
    if (isUsefulOutline(blockOutlineItems)) return blockOutlineItems;
    return [];
  }, [blockOutlineItems, sectionOutline?.source, sectionOutlineItems]);
  const pdfOutlineSource = isUsefulOutline(sectionOutlineItems, sectionOutline?.source)
    ? sectionOutline?.source
    : (
      isUsefulOutline(blockOutlineItems)
        ? (blockOutlineItems.some((item) => item?.source === 'toc') ? 'toc' : 'heuristic')
        : ''
    );
  const pdfOutlineLoading = blockIndexLoading || (sectionOutlineLoading && pdfOutlineItems.length === 0);
  const pdfOutlineError = blockIndexError || (pdfOutlineItems.length === 0 ? sectionOutlineError : '');

  const readingOutlineFlat = useMemo(() => {
    const result = [];
    const walk = (nodes, level = 1) => {
      (nodes || []).forEach((node) => {
        if (!node) return;
        result.push({ ...node, level });
        if (Array.isArray(node.children) && node.children.length > 0) {
          walk(node.children, level + 1);
        }
      });
    };
    walk(readingOutlineItems);
    return result;
  }, [readingOutlineItems]);

  const readingNodeById = useMemo(() => {
    const result = {};
    readingOutlineFlat.forEach((item) => {
      if (item?.id) result[item.id] = item;
    });
    return result;
  }, [readingOutlineFlat]);

  const blockToReadingNode = useMemo(() => {
    const result = {};
    readingOutlineFlat.forEach((item) => {
      const ids = item.evidence?.block_ids || item.evidence_block_ids || [];
      ids.forEach((blockId) => {
        if (!result[blockId]) result[blockId] = item;
      });
    });
    return result;
  }, [readingOutlineFlat]);

  const activeReadingNode = activeReadingNodeId ? readingNodeById[activeReadingNodeId] : null;
  const focusedReadingBlockIds = useMemo(() => {
    const ids = activeReadingNode?.evidence?.block_ids || activeReadingNode?.evidence_block_ids || [];
    const primaryBlockId = activeReadingNode?.first_block || ids[0] || pinnedReadingBlockId;
    return primaryBlockId ? [primaryBlockId] : EMPTY_ID_LIST;
  }, [activeReadingNode, pinnedReadingBlockId]);
  const translatingBlockIdList = useMemo(
    () => (translatingBlockIds.size ? Array.from(translatingBlockIds) : EMPTY_ID_LIST),
    [translatingBlockIds],
  );

  const activeReadingBlockId = hoveredReadingBlockId || focusedReadingBlockIds[0] || pinnedReadingBlockId;

  const allTranslatableReadingBlocks = useMemo(() => {
    if (!blockIndexMatchesCurrentParse) return [];
    return (blockIndex?.pages || []).flatMap((page) => {
      const pageNumber = Number(page.page) || 1;
      return (page.blocks || [])
        .filter(isTranslatableReadingBlock)
        .map((block) => ({ ...block, page: pageNumber }));
    });
  }, [blockIndex, blockIndexMatchesCurrentParse]);

  const translatedReadingBlockCount = useMemo(() => {
    return allTranslatableReadingBlocks.filter((block) => blockTranslations[block.block_id]).length;
  }, [allTranslatableReadingBlocks, blockTranslations]);
  const pendingReadingBlockCount = Math.max(0, allTranslatableReadingBlocks.length - translatedReadingBlockCount);
  const failedReadingBlockCount = useMemo(() => {
    if (!failedTranslationBlockIds || failedTranslationBlockIds.size === 0) return 0;
    const validIds = new Set(allTranslatableReadingBlocks.map((block) => block.block_id));
    return [...failedTranslationBlockIds].filter((blockId) => validIds.has(blockId)).length;
  }, [allTranslatableReadingBlocks, failedTranslationBlockIds]);

  useEffect(() => {
    if (pretranslateProgress.running || pretranslateProgress.coverageLocked) return;
    const total = allTranslatableReadingBlocks.length;
    const done = Math.min(total, translatedReadingBlockCount);
    setPretranslateProgress((prev) => (
      prev.done === done && prev.total === total && !prev.force
        ? prev
        : { ...prev, running: false, done, total, force: false }
    ));
  }, [
    allTranslatableReadingBlocks.length,
    pretranslateProgress.coverageLocked,
    pretranslateProgress.running,
    translatedReadingBlockCount,
  ]);

  useEffect(() => {
    if (
      !pretranslateProgress.running
      || !docId
      || !blockIndex
      || !blockIndexMatchesCurrentParse
      || allTranslatableReadingBlocks.length === 0
    ) {
      return () => {};
    }

    let cancelled = false;
    const translationEpoch = blockTranslationEpochRef.current;
    const requestContext = {
      docId: docId || '',
      parseIdentity: documentParseIdentity,
      epoch: parseContextRef.current.epoch,
    };
    const isActiveRequest = () => (
      !cancelled
      && blockTranslationEpochRef.current === translationEpoch
      && isCurrentParseContext(requestContext)
    );
    const pollCachedTranslations = async () => {
      try {
        const { headers } = getChatRequestConfig();
        const res = await fetch(
          `${API_BASE_URL}/documents/${requestContext.docId}/blocks/translations?target_lang=zh&t=${Date.now()}`,
          { headers },
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (!isActiveRequest()) return;

        const items = data?.items || {};
        setBlockTranslations((prev) => mergeRecordIfChanged(prev, items));
        const cachedFailures = [
          ...(Array.isArray(data?.failed_block_ids) ? data.failed_block_ids : []),
          ...(Array.isArray(data?.skipped_block_ids) ? data.skipped_block_ids : []),
        ].filter(Boolean);
        setFailedTranslationBlockIds((prev) => {
          const next = new Set(cachedFailures);
          if (prev.size === next.size && cachedFailures.every((id) => prev.has(id))) return prev;
          return next;
        });
        const done = allTranslatableReadingBlocks.filter((block) => items[block.block_id]).length;
        setPretranslateProgress((prev) => {
          if (!prev.running || prev.force || prev.retryFailed) return prev;
          const nextDone = Math.max(prev.done || 0, Math.min(done, prev.total || allTranslatableReadingBlocks.length));
          return nextDone === prev.done ? prev : { ...prev, done: nextDone };
        });
      } catch (error) {
        if (isActiveRequest()) {
          console.warn('[ImmersiveReading] 预翻译进度同步失败', error);
        }
      }
    };

    const stop = startVisiblePoll(pollCachedTranslations, 2500, { immediate: false });
    pollCachedTranslations();
    return () => {
      cancelled = true;
      stop();
    };
  }, [
    allTranslatableReadingBlocks,
    blockIndex,
    blockIndexMatchesCurrentParse,
    docId,
    documentParseIdentity,
    getChatRequestConfig,
    isCurrentParseContext,
    pretranslateProgress.running,
    translationModelId,
    translationProviderId,
  ]);

  const currentPageBlocks = useMemo(() => {
    const page = blockIndex?.pages?.find((item) => Number(item.page) === Number(currentPage));
    const blocks = page?.blocks || [];
    return blocks
      .filter((block) => {
        const type = block?.type || 'paragraph';
        const text = String(block?.text || '').trim();
        return text.length > 1 && ['heading', 'paragraph', 'caption', 'figure', 'table', 'formula', 'code'].includes(type);
      })
      .slice(0, 24);
  }, [blockIndex, currentPage]);

  const currentPageReadingNotes = useMemo(() => {
    return readingOutlineFlat.filter((item) => {
      const pages = item.evidence?.pages || [];
      if (pages.includes(Number(currentPage))) return true;
      return Number(item.page) === Number(currentPage);
    });
  }, [currentPage, readingOutlineFlat]);

  const currentPageUserNotes = useMemo(() => (
    documentNotes.filter((item) => Number(item.page) === Number(currentPage))
  ), [currentPage, documentNotes]);

  const handleUserNoteReveal = useCallback((noteId) => {
    setPendingUserNoteRevealId((currentId) => currentId === noteId ? '' : currentId);
  }, []);

  const handleUserNoteClick = useCallback((note) => {
    if (!note) return;
    const targetPage = Math.max(1, Number(note.page) || 1);
    setCurrentPage(targetPage);
    const hasSelectionAnchor = note.anchor_type !== 'page' && Boolean(note.text || note.rects?.length);
    if (!hasSelectionAnchor) return;
    setActiveHighlight({
      page: targetPage,
      text: note.text,
      source: 'note',
      at: Date.now(),
      citationAnchor: {
        rects: (note.rects || []).map((rect) => [
          rect.left,
          rect.top,
          rect.left + rect.width,
          rect.top + rect.height,
        ]),
        coordinateSpace: 'pdf_top_left_points',
      },
    });
  }, [setActiveHighlight, setCurrentPage]);

  const handleDeleteUserNote = useCallback((noteId) => {
    const targetNote = documentNotes.find((item) => item.id === noteId);
    if (!targetNote) return;

    const nextNotes = documentNotes.filter((item) => item.id !== noteId);
    const selectionFingerprint = getDocumentHighlightFingerprint(targetNote);
    const hasRemainingLinkedNote = Boolean(selectionFingerprint) && nextNotes.some(
      (item) => getDocumentHighlightFingerprint(item) === selectionFingerprint
    );
    const nextHighlights = selectionFingerprint && !hasRemainingLinkedNote
      ? documentHighlights.filter((item) => getDocumentHighlightFingerprint(item) !== selectionFingerprint)
      : documentHighlights;

    setDocumentNotes(nextNotes);
    if (!writeDocumentNotes(docId, nextNotes)) {
      console.warn('[DocumentNote] 笔记已从当前界面移除，但持久化写入失败');
    }
    if (nextHighlights !== documentHighlights) {
      setDocumentHighlights(nextHighlights);
      if (!writeDocumentHighlights(docId, nextHighlights)) {
        console.warn('[DocumentHighlight] 关联标注已从当前界面移除，但持久化写入失败');
      }
    }

    if (selectedSavedHighlightId && !nextHighlights.some((item) => item.id === selectedSavedHighlightId)) {
      selectedPdfHighlightAnchorRef.current = null;
      pendingAutoAnnotationRef.current = null;
      setSelectedSavedHighlightId('');
      setShowTextMenu(false);
      setSelectedText('');
    }
  }, [docId, documentHighlights, documentNotes, selectedSavedHighlightId, setSelectedText, setShowTextMenu]);

  const handleSaveUserNote = useCallback(async ({ id, note, page }) => {
    if (!docId) throw new Error('请先打开文档');
    const content = String(note || '').trim();
    if (!content) throw new Error('笔记内容不能为空');
    const targetPage = Math.max(1, Math.floor(Number(page) || currentPage));
    const updatedAt = Date.now();
    let savedNote;
    let nextNotes;

    if (id) {
      const existingNote = documentNotes.find((item) => item.id === id);
      if (!existingNote) throw new Error('这条笔记已不存在');
      savedNote = {
        ...existingNote,
        note: content,
        page: targetPage,
        updated_at: updatedAt,
      };
      nextNotes = documentNotes.map((item) => item.id === id ? savedNote : item);
    } else {
      savedNote = createDocumentNote({
        text: '',
        note: content,
        page: targetPage,
        rects: [],
        anchorType: 'page',
        now: updatedAt,
      });
      if (!savedNote) throw new Error('笔记内容不能为空');
      nextNotes = [...documentNotes, savedNote];
    }

    setDocumentNotes(nextNotes);
    setPendingUserNoteRevealId(savedNote.id);
    if (!writeDocumentNotes(docId, nextNotes)) {
      console.warn('[DocumentNote] 笔记已显示，但持久化写入失败');
    }
    return savedNote;
  }, [currentPage, docId, documentNotes]);

  const handleOutlineJump = useCallback((item) => {
    if (!item) return;
    const targetPage = item.evidence?.primary_page || item.page;
    const firstBlock = item.first_block || item.evidence?.block_ids?.[0] || item.evidence_block_ids?.[0] || null;
    if (targetPage) {
      const page = Math.max(1, Number(targetPage) || 1);
      setCurrentPage(page);
      setReaderNavigationRequest((current) => ({
        revision: (current?.revision || 0) + 1,
        page,
        blockId: firstBlock,
      }));
    }
    setActiveReadingNodeId(item.id || null);
    if (item.id) {
      setVisitedReadingNodeIds((prev) => {
        const next = new Set(prev);
        next.add(item.id);
        return next;
      });
    }
    setPinnedReadingBlockId(firstBlock);
    if (firstBlock) {
      setReadingJumpPulseToken((value) => value + 1);
    }
    setHoveredReadingBlockId(null);
  }, [setCurrentPage]);

  const handleSectionOutlineJump = useCallback((item) => {
    if (!item) return;
    const firstBlock = resolveSectionOutlineAnchor(item);
    const resolvedBlock = firstBlock ? blockMap[firstBlock] : null;
    const targetPage = resolvedBlock?.page || item.evidence?.primary_page || item.page;
    if (targetPage) {
      const page = Math.max(1, Number(targetPage) || 1);
      setCurrentPage(page);
      setReaderNavigationRequest((current) => ({
        revision: (current?.revision || 0) + 1,
        page,
        blockId: firstBlock,
      }));
    }
    const nodeId = item.id || item.section_id || null;
    setActiveReadingNodeId(null);
    setActiveSectionNodeId(nodeId);
    if (nodeId) {
      setVisitedSectionNodeIds((prev) => {
        const next = new Set(prev);
        next.add(nodeId);
        return next;
      });
    }
    setPinnedReadingBlockId(firstBlock);
    if (firstBlock) {
      setReadingJumpPulseToken((value) => value + 1);
    }
    setHoveredReadingBlockId(null);
  }, [blockMap, resolveSectionOutlineAnchor, setCurrentPage]);

  const handleReadingBlockHover = useCallback((block) => {
    setHoveredReadingBlockId(block?.block_id || null);
  }, []);

  const handleReadingBlockClick = useCallback((block) => {
    const blockId = block?.block_id || null;
    const node = blockId ? blockToReadingNode[blockId] : null;
    if (node) {
      handleOutlineJump(node);
      return;
    }
    setActiveReadingNodeId(null);
    setPinnedReadingBlockId(blockId);
  }, [blockToReadingNode, handleOutlineJump]);

  const translateReadingBlocks = useCallback(async (blocksToTranslate, options = {}) => {
    const selectedBlocks = (blocksToTranslate || []).filter((block) => block?.block_id);
    if (!docId || selectedBlocks.length === 0) return null;
    if (isMinerUFullRoutePending) {
      setBlockTranslateError(minerUParsePendingNotice);
      return null;
    }
    const blockIds = selectedBlocks.map((block) => block.block_id);
    const translationEpoch = options.translationEpoch ?? blockTranslationEpochRef.current;
    const canApply = () => (
      blockTranslationEpochRef.current === translationEpoch
      && (typeof options.isCurrent !== 'function' || options.isCurrent())
    );
    const staleResult = () => (
      options.returnRaw ? { stale: true, items: {}, failed_block_ids: [] } : null
    );
    if (!canApply()) return staleResult();

    const chatCredentials = getChatCredentials?.();
    const chatProvider = chatCredentials?.providerId || 'openai';
    const chatModel = chatCredentials?.modelId || 'gpt-4o';
    const chatApiKey = chatCredentials?.apiKey || '';
    const chatProviderFull = getProviderById?.(chatProvider);

    if (getChatCredentials && !chatApiKey && chatProvider !== 'local' && chatProvider !== 'ollama') {
      const message = `请先为 ${chatProviderFull?.name || chatProvider} 配置 API Key`;
      setBlockTranslateError(message);
      return options.returnRaw ? { items: {}, failed_block_ids: [], error: message } : null;
    }

    if (options.showPanelLoading) {
      setBlockTranslateLoading(true);
    }
    setBlockTranslateError('');
    setTranslatingBlockIds((prev) => {
      const next = new Set(prev);
      blockIds.forEach((blockId) => next.add(blockId));
      return next;
    });

    try {
      const headers = { 'Content-Type': 'application/json' };
      if (chatCredentials) {
        headers['X-ChatPDF-Provider'] = chatProvider;
        headers['X-ChatPDF-Model'] = chatModel;
        if (chatApiKey) {
          headers['X-ChatPDF-Api-Key'] = chatApiKey;
        }
        if (chatProviderFull?.apiHost) {
          headers['X-ChatPDF-Api-Host'] = chatProviderFull.apiHost;
        }
      }

      const endpoint = options.bulk ? 'pretranslate' : 'translate';
      const requestBody = {
        block_ids: blockIds,
        target_lang: 'zh',
        force: Boolean(options.force),
        with_summary: Boolean(blockSummaryRef.current),
      };
      if (options.bulk) {
        requestBody.concurrency = pretranslateConcurrency;
      }

      const res = await fetch(`${API_BASE_URL}/documents/${docId}/blocks/${endpoint}`, {
        method: 'POST',
        headers,
        signal: options.signal,
        body: JSON.stringify(requestBody),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({ detail: '段落翻译失败' }));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      if (!canApply()) return staleResult();
      const translatedItems = filterValidBlockTranslations(data?.items || {});
      const translatedIds = Object.keys(translatedItems);
      const invalidReturnedIds = Object.keys(data?.items || {}).filter((blockId) => !translatedItems[blockId]);
      setBlockTranslations((prev) => mergeRecordIfChanged(prev, translatedItems));
      const failedBlockIds = [
        ...(Array.isArray(data?.failed_block_ids) ? data.failed_block_ids : []),
        ...(Array.isArray(data?.skipped_block_ids) ? data.skipped_block_ids : []),
        ...invalidReturnedIds,
      ].filter((blockId, index, arr) => blockId && arr.indexOf(blockId) === index);
      setFailedTranslationBlockIds((prev) => {
        const next = new Set(prev);
        translatedIds.forEach((blockId) => next.delete(blockId));
        failedBlockIds.forEach((blockId) => next.add(blockId));
        return next;
      });
      if (failedBlockIds.length > 0) {
        setBlockTranslateError(`有 ${failedBlockIds.length} 个段落暂未翻译成功，可稍后补齐`);
      }
      const normalizedData = {
        ...(data || {}),
        items: translatedItems,
        failed_block_ids: failedBlockIds,
        skipped_block_ids: Array.isArray(data?.skipped_block_ids) ? data.skipped_block_ids : [],
      };
      return options.returnRaw ? normalizedData : translatedItems;
    } catch (error) {
      if (error?.name === 'AbortError') {
        return null;
      }
      if (canApply()) {
        const message = sanitizeTranslationError(error.message);
        setBlockTranslateError(message);
        return options.returnRaw ? { items: {}, failed_block_ids: [], error: message } : null;
      }
      return staleResult();
    } finally {
      if (canApply()) {
        setTranslatingBlockIds((prev) => {
          const next = new Set(prev);
          blockIds.forEach((blockId) => next.delete(blockId));
          return next;
        });
        if (options.showPanelLoading) {
          setBlockTranslateLoading(false);
        }
      }
    }
  }, [
    docId,
    getChatCredentials,
    getProviderById,
    isMinerUFullRoutePending,
    minerUParsePendingNotice,
    pretranslateConcurrency,
  ]);

  const handleTranslateCurrentPage = useCallback(() => {
    translateReadingBlocks(currentPageBlocks, { showPanelLoading: true });
  }, [currentPageBlocks, translateReadingBlocks]);

  const handleRetranslateReadingBlock = useCallback(async (block) => {
    if (!block?.block_id) return;
    setPretranslateNotice(`正在重译 ${block.block_id}`);
    const data = await translateReadingBlocks([block], {
      force: true,
      returnRaw: true,
    });
    if (data?.stale) return;
    const translated = data?.items?.[block.block_id]?.translation;
    if (isValidBlockTranslationText(translated)) {
      setPretranslateNotice(`已重译 ${block.block_id}`);
      setBlockTranslateError('');
      return;
    }
    setPretranslateNotice(`重译失败：${block.block_id} 可稍后再试`);
    setBlockTranslateError(sanitizeTranslationError(data?.error || '模型没有返回可用译文'));
  }, [translateReadingBlocks]);

  const pretranslateReadingDocument = useCallback(async ({ force = false, retryFailed = false } = {}) => {
    if (!blockIndexMatchesCurrentParse) {
      setPretranslateNotice('阅读结构正在同步当前解析路线，请稍后重试');
      setPretranslateError('');
      return;
    }
    if (!docId || allTranslatableReadingBlocks.length === 0) {
      setPretranslateProgress({ running: false, done: 0, total: 0 });
      setPretranslateNotice('当前文档还没有可缓存的文本块');
      setPretranslateError('');
      return;
    }

    const { canCallModel, providerName } = getChatRequestConfig();
    if (!canCallModel) {
      const message = `请先为 ${providerName} 配置 API Key，再开启悬浮预翻译`;
      setBlockTranslateError(message);
      setPretranslateNotice(`请先为 ${providerName} 配置 API Key`);
      setPretranslateError(message);
      return;
    }

    const failedIds = new Set(failedTranslationBlockIds || []);
    const pendingBlocks = selectPendingPretranslateBlocks({
      blocks: allTranslatableReadingBlocks,
      translations: blockTranslations,
      failedBlockIds: failedIds,
      force,
      retryFailed,
    });
    const total = allTranslatableReadingBlocks.length;
    const failedCachedCount = retryFailed
      ? allTranslatableReadingBlocks.filter((block) => (
        failedIds.has(block.block_id) && blockTranslations?.[block.block_id]
      )).length
      : 0;
    const initialDone = force
      ? 0
      : Math.max(0, translatedReadingBlockCount - failedCachedCount);
    const requestForce = shouldForcePretranslateRequest({ force, retryFailed });

    if (pendingBlocks.length === 0) {
      setBlockTranslateError('');
      setPretranslateNotice('全文翻译缓存已是最新');
      setPretranslateError('');
      setPretranslateProgress({ running: false, done: translatedReadingBlockCount, total });
      return;
    }

    const runId = pretranslateRunRef.current + 1;
    pretranslateRunRef.current = runId;
    pretranslateAbortRef.current?.abort();
    const abortController = new AbortController();
    pretranslateAbortRef.current = abortController;
    const translationEpoch = blockTranslationEpochRef.current;
    const requestContext = {
      docId: docId || '',
      parseIdentity: documentParseIdentity,
      epoch: parseContextRef.current.epoch,
    };
    const isRunCurrent = () => (
      pretranslateRunRef.current === runId
      && !abortController.signal.aborted
      && blockTranslationEpochRef.current === translationEpoch
      && isCurrentParseContext(requestContext)
    );
    setBlockTranslateError('');
    setPretranslateError('');
    setPretranslateNotice(`正在缓存 ${pendingBlocks.length} 个段落译文`);
    setPretranslateProgress((prev) => ({
      running: true,
      done: initialDone,
      total,
      force: Boolean(force),
      retryFailed: Boolean(retryFailed),
      coverageLocked: Boolean(prev.coverageLocked || force || retryFailed),
    }));

    const result = await executePretranslateBatches({
      blocks: pendingBlocks,
      isCurrent: isRunCurrent,
      translateBatch: (batch) => translateReadingBlocks(batch, {
        force: requestForce,
        bulk: true,
        returnRaw: true,
        signal: abortController.signal,
        translationEpoch,
        isCurrent: isRunCurrent,
      }),
      onBatchStart: ({ batchNumber, batchCount }) => {
        if (isRunCurrent() && batchCount > 1) {
          setPretranslateNotice(`正在缓存第 ${batchNumber}/${batchCount} 批 · 共 ${pendingBlocks.length} 个段落`);
        }
      },
      onBatchComplete: ({ batchNumber, batchCount, successfulCount }) => {
        if (!isRunCurrent()) return;
        const done = Math.min(total, force ? successfulCount : initialDone + successfulCount);
        setPretranslateProgress((prev) => ({
          ...prev,
          running: true,
          done: Math.max(prev.done || 0, done),
          total,
          force: Boolean(force),
          batch: batchNumber,
          batches: batchCount,
        }));
      },
    });

    if (result?.stale || pretranslateRunRef.current !== runId) return;
    pretranslateAbortRef.current = null;

    if (result?.aborted || abortController.signal.aborted) {
      pretranslateStartedDocRef.current = pretranslateAutoIdentity;
      setPretranslateProgress((prev) => ({
        ...prev,
        running: false,
        coverageLocked: Boolean(prev.coverageLocked || prev.force || prev.retryFailed),
        force: false,
        retryFailed: false,
      }));
      setBlockTranslateError('');
      setPretranslateNotice('已取消，已完成的译文缓存会保留');
      setPretranslateError('');
      return;
    }

    const successfulIds = new Set(result?.successfulBlockIds || []);
    const aggregatedFailedIds = new Set(result?.failedBlockIds || []);
    setFailedTranslationBlockIds((prev) => {
      const next = new Set(prev);
      successfulIds.forEach((blockId) => next.delete(blockId));
      aggregatedFailedIds.forEach((blockId) => next.add(blockId));
      return next;
    });

    const done = Math.min(total, force ? successfulIds.size : initialDone + successfulIds.size);
    const batchError = String(result?.error || '').trim();
    const coverageLocked = Boolean(
      (force || retryFailed) && (batchError || aggregatedFailedIds.size > 0)
    );
    setPretranslateProgress((prev) => ({
      ...prev,
      running: false,
      done: Math.max(prev.done || 0, done),
      total,
      force: false,
      retryFailed: false,
      coverageLocked,
    }));

    if (batchError) {
      pretranslateStartedDocRef.current = pretranslateAutoIdentity;
      setPretranslateError(batchError);
      setBlockTranslateError(batchError);
      setPretranslateNotice('缓存请求中断，已完成的译文已保留，可稍后继续补齐');
    } else if (aggregatedFailedIds.size > 0) {
      pretranslateStartedDocRef.current = pretranslateAutoIdentity;
      setPretranslateError('');
      setBlockTranslateError(`有 ${aggregatedFailedIds.size} 个段落暂未翻译成功，已完成 ${done}/${total}，可稍后继续补齐`);
      setPretranslateNotice(`部分完成：${done}/${total}`);
    } else {
      setPretranslateError('');
      setBlockTranslateError('');
      setPretranslateNotice(`缓存完成：${done}/${total}`);
    }
  }, [
    allTranslatableReadingBlocks,
    blockIndexMatchesCurrentParse,
    blockTranslations,
    docId,
    documentParseIdentity,
    failedTranslationBlockIds,
    getChatRequestConfig,
    isCurrentParseContext,
    pretranslateAutoIdentity,
    translatedReadingBlockCount,
    translateReadingBlocks,
  ]);

  const cancelPretranslateReadingDocument = useCallback(() => {
    const wasRunning = pretranslateProgress.running;
    pretranslateRunRef.current += 1;
    pretranslateStartedDocRef.current = pretranslateAutoIdentity || null;
    pretranslateAbortRef.current?.abort();
    pretranslateAbortRef.current = null;
    setPretranslateProgress((prev) => (
      prev.running
        ? {
          ...prev,
          running: false,
          coverageLocked: Boolean(prev.coverageLocked || prev.force || prev.retryFailed),
          force: false,
          retryFailed: false,
        }
        : prev
    ));
    setTranslatingBlockIds(new Set());
    if (wasRunning) {
      setBlockTranslateError('');
      setPretranslateNotice('已取消，已完成的译文缓存会保留');
      setPretranslateError('');
    }
  }, [pretranslateAutoIdentity, pretranslateProgress.running]);

  const handleStartPretranslate = useCallback((options = {}) => {
    const retryFailed = options.force
      ? false
      : (options.retryFailed ?? failedReadingBlockCount > 0);
    pretranslateReadingDocument({ ...options, retryFailed });
  }, [failedReadingBlockCount, pretranslateReadingDocument]);

  // 打开「逐段要点」之前翻好的块，缓存里 summary 是空的。
  // 这里只补那一次要点调用，复用已有译文，不重跑翻译。
  const [summaryBackfillRunning, setSummaryBackfillRunning] = useState(false);
  const handleBackfillSummaries = useCallback(async () => {
    if (!docId || summaryBackfillRunning) return;
    setSummaryBackfillRunning(true);
    setPretranslateNotice('正在为已翻译的段落补要点…');
    try {
      const headers = { 'Content-Type': 'application/json' };
      const { chatApiKey, chatProviderFull } = getChatRequestConfig();
      if (chatApiKey) headers['X-ChatPDF-Api-Key'] = chatApiKey;
      if (chatProviderFull?.apiHost) headers['X-ChatPDF-Api-Host'] = chatProviderFull.apiHost;

      const res = await fetch(`${API_BASE_URL}/documents/${docId}/blocks/backfill-summaries`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ target_lang: 'zh', concurrency: pretranslateConcurrency }),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      const filled = Object.keys(data?.items || {}).length;
      setBlockTranslations((prev) => mergeRecordIfChanged(prev, data?.items || {}));
      setPretranslateNotice(
        filled > 0
          ? `已为 ${filled} 段补上要点`
          : '没有需要补要点的段落'
      );
    } catch (error) {
      setPretranslateNotice('');
      setBlockTranslateError(sanitizeTranslationError(error?.message, '补齐要点失败，请稍后重试'));
    } finally {
      setSummaryBackfillRunning(false);
    }
  }, [docId, getChatRequestConfig, pretranslateConcurrency, summaryBackfillRunning]);

  useEffect(() => {
    const wasEnabled = prevShouldAutoPretranslateRef.current;
    prevShouldAutoPretranslateRef.current = shouldAutoPretranslate;
    if (wasEnabled && !shouldAutoPretranslate) {
      cancelPretranslateReadingDocument();
    }
  }, [cancelPretranslateReadingDocument, shouldAutoPretranslate]);

  useEffect(() => {
    if (
      !shouldAutoPretranslate
      || !docId
      || !blockIndex
      || !blockIndexMatchesCurrentParse
      || !blockTranslationsLoaded
      || blockTranslationsLoadedIdentity !== pretranslateAutoIdentity
      || !pretranslateAutoIdentity
    ) return;
    if (pretranslateStartedDocRef.current === pretranslateAutoIdentity) return;
    const { canCallModel } = getChatRequestConfig();
    if (!canCallModel) return;

    const hasPendingBlocks = allTranslatableReadingBlocks.some((block) => !blockTranslations[block.block_id]);
    if (!hasPendingBlocks) {
      pretranslateStartedDocRef.current = pretranslateAutoIdentity;
      setPretranslateProgress({
        running: false,
        done: allTranslatableReadingBlocks.length,
        total: allTranslatableReadingBlocks.length,
      });
      return;
    }

    pretranslateStartedDocRef.current = pretranslateAutoIdentity;
    pretranslateReadingDocument();
  }, [
    allTranslatableReadingBlocks,
    blockIndex,
    blockIndexMatchesCurrentParse,
    blockTranslations,
    blockTranslationsLoaded,
    blockTranslationsLoadedIdentity,
    docId,
    shouldAutoPretranslate,
    getChatRequestConfig,
    pretranslateAutoIdentity,
    pretranslateReadingDocument,
  ]);

  // ========== 截图状态 Hook（需求 1.1） ==========
  // textareaRef 来自 useMessageState（后续初始化），通过代理 ref 桥接
  const textareaRefProxy = useRef(null);
  const screenshotState = useScreenshotState({
    pdfContainerRef,
    textareaRef: textareaRefProxy,
    isVisionCapable,
    sendMessage: (...args) => messageSettersRef.current.sendMessage?.(...args),
  });
  const {
    screenshots,
    setScreenshots,
    isSelectingArea, setIsSelectingArea,
    handleAreaSelected, handleSelectionCancel,
    handleScreenshotAction, handleScreenshotClose,
  } = screenshotState;
  const handleParseAwareScreenshotAction = useCallback((...args) => {
    if (isChatInteractionLocked) {
      setDeepParseNotice(chatInteractionLockedNotice);
      return;
    }
    handleScreenshotAction(...args);
  }, [chatInteractionLockedNotice, handleScreenshotAction, isChatInteractionLocked]);

  // ========== 消息状态 Hook（需求 1.2） ==========
  const messageState = useMessageState({
    docId,
    parseGeneration: documentParseManifest?.generation || '',
    documentSourceHash: documentParseManifest?.source_hash || '',
    parseIdentityReady: !isDocumentParseIdentityHydrating,
    screenshots,
    setScreenshots,
    selectedText,
    getChatCredentials,
    getVisualCredentials,
    getCurrentChatModel,
    getProviderById,
    streamSpeed,
    enableVectorSearch,
    embeddingApiKey: getEmbeddingApiKey(),
    getEmbeddingConfig,
    enableGraphRAG,
    enableAgentRetrieval,
    forceAgentRetrieval,
    enableJiebaBM25,
    numExpandContextChunk,
    enableBlurReveal,
    blurIntensity,
    enableMathHydration: mathEngine !== 'none',
    enableSingleDollarMath: mathEnableSingleDollar !== false,
    globalSettings,
  });
  const {
    messages, setMessages,
    isLoading, setIsLoading,
    hasInput, setHasInput,
    streamingMessageId,
    lastCallInfo,
    copiedMessageId,
    likedMessages, rememberedMessages,
    textareaRef,
    sendMessage: sendMessageInternal, handleStop,
    regenerateMessage: regenerateMessageInternal, copyMessage, saveToMemory,
    setInputValue,
    invalidateVisualVerificationState,
    // ref 直写模式：流式输出期间直接更新 DOM
    streamingContentRef,
    streamingThinkingRef,
    subscribeContentCommittedPrefix,
    subscribeContentDisplayedText,
  } = messageState;
  const hasEmbeddingDependentChatFeatures = enableVectorSearch
    || enableGraphRAG
    || enableAgentRetrieval
    || forceAgentRetrieval;
  const sendMessage = useCallback((overrides = {}) => {
    if (isChatInteractionLocked) {
      setDeepParseNotice(chatInteractionLockedNotice);
      return;
    }
    if (hasEmbeddingDependentChatFeatures) {
      const missingEmbeddingCredential = getMissingEmbeddingCredential();
      if (missingEmbeddingCredential) {
        alert(missingEmbeddingCredential.message);
        return;
      }
    }
    const scopedDocIds = Array.isArray(overrides?.docIds)
      ? overrides.docIds
      : crossDocumentIds;
    return sendMessageInternal({ ...overrides, docIds: scopedDocIds });
  }, [
    chatInteractionLockedNotice,
    crossDocumentIds,
    getMissingEmbeddingCredential,
    hasEmbeddingDependentChatFeatures,
    isChatInteractionLocked,
    sendMessageInternal,
  ]);
  const regenerateMessage = useCallback((...args) => {
    if (hasEmbeddingDependentChatFeatures) {
      const missingEmbeddingCredential = getMissingEmbeddingCredential();
      if (missingEmbeddingCredential) {
        alert(missingEmbeddingCredential.message);
        return;
      }
    }
    return regenerateMessageInternal(...args);
  }, [getMissingEmbeddingCredential, hasEmbeddingDependentChatFeatures, regenerateMessageInternal]);

  // 双向联动：当前高亮的引文编号
  const [activeCitationRef, setActiveCitationRef] = useState(null);

  // 用户反馈
  const [feedbackTarget, setFeedbackTarget] = useState(null); // {idx, msg}
  const [dislikedMessages, setDislikedMessages] = useState(new Set());

  // GraphRAG 构建状态：per-docId，不持久化（后端重启后内存实例丢失，由 stats 查询恢复）
  const [graphragStatus, setGraphragStatus] = useState('unknown'); // unknown | idle | building | built | error
  const [graphragStats, setGraphragStats] = useState(null); // { num_nodes, num_edges, num_docs, num_chunks }
  const [graphragError, setGraphragError] = useState('');
  const [graphragProgress, setGraphragProgress] = useState(null); // { stage, progress, last_error }

  // 将 setter 函数注册到 ref 桥接对象，供 useDocumentState 和 useScreenshotState 使用
  messageSettersRef.current = { setMessages, setIsLoading, setInputValue, sendMessage };
  pdfSettersRef.current = { setCurrentPage, setSelectedText };
  screenshotSettersRef.current = { setScreenshots: screenshotState.setScreenshots };
  // 同步 textareaRef 代理，使截图 Hook 能正确聚焦输入框
  textareaRefProxy.current = textareaRef?.current ?? null;

  // ========== Refs ==========
  const chatPaneRef = useRef(null);

  // ========== 副作用 ==========
  useEffect(() => {
    fetchAvailableModels();
    fetchAvailableEmbeddingModels();
  }, []);

  // 注意：原有的 localStorage 批量写入 useEffect 已被 useDebouncedLocalStorage 替代
  // lastCallInfo 仍需单独处理
  useEffect(() => {
    if (lastCallInfo) localStorage.setItem('lastCallInfo', JSON.stringify(lastCallInfo));
  }, [lastCallInfo]);

  useEffect(() => {
    if (Object.keys(availableModels).length === 0) return;
    const providerModels = availableModels[apiProvider]?.models;
    if (providerModels && !providerModels[model]) {
      const first = Object.keys(providerModels)[0];
      if (first) setModel(first);
    }
  }, [availableModels, apiProvider]);

  // 流式中跳过会话写入；结束、切文档时再落盘。秒表不再抬到这里。
  useEffect(() => {
    if (!docId || !docInfo) return undefined;
    if (streamingMessageId) return undefined;
    saveCurrentSession(messages);
    return undefined;
  }, [docId, docInfo, messages, saveCurrentSession, streamingMessageId]);

  useEffect(() => () => {
    flushCurrentSession?.();
  }, [docId, flushCurrentSession]);

  // ========== GraphRAG 构建 / 状态查询 ==========
  // 切换文档时先重置状态，然后查询是否已有图谱实例在后端内存中。
  // 注意：后端 `_graphrag_instances` 只在进程内存里存活，重启后会丢；磁盘上的
  // `data/graphrag/<doc_id>/` 目录存在但没有自动 reload 逻辑，所以这里只能拿
  // 到「本次进程已构建」的状态。
  useEffect(() => {
    if (!docId || isMinerUFullRoutePending) {
      setGraphragStatus('unknown');
      setGraphragStats(null);
      setGraphragError('');
      setGraphragProgress(null);
      return;
    }
    let cancelled = false;
    (async () => {
      setGraphragStatus('idle');
      setGraphragStats(null);
      setGraphragError('');
      setGraphragProgress(null);
      try {
        const res = await fetch(`${API_BASE_URL}/document/${docId}/graphrag/stats`);
        if (cancelled) return;
        if (res.ok) {
          const data = await res.json();
          const stats = data.stats || {};
          setGraphragStats(stats);
          // 从 build_meta 恢复进度/错误信息
          const meta = stats.build_meta || {};
          if (meta.status === 'done') {
            setGraphragStatus('built');
          } else if (meta.status === 'failed') {
            setGraphragStatus('error');
            setGraphragError(meta.last_error || '构建失败');
          }
          setGraphragProgress(meta);
        }
        // 404 表示未构建 → 保持 'idle'
      } catch (e) {
        if (!cancelled) console.warn('[GraphRAG] stats 查询失败', e);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [docId, isMinerUFullRoutePending]);

  const handleBuildGraphRAG = useCallback(async (opts = {}) => {
    if (!docId) { alert('请先上传文档'); return; }
    if (isMinerUFullRoutePending) {
      setDeepParseNotice(minerUParsePendingNotice);
      return;
    }
    const { providerId: chatProvider, modelId: chatModel, apiKey: chatApiKey } = getChatCredentials?.() || {};
    if (!chatApiKey && chatProvider !== 'ollama' && chatProvider !== 'local') {
      alert('请先配置对话模型的 API Key');
      return;
    }
    if (!chatModel) {
      alert('请先选择对话模型');
      return;
    }

    const chatProviderFull = getProviderById?.(chatProvider);
    const embedConfig = getEmbeddingConfig?.() || {};
    if (!embedConfig.isValid) {
      alert('请先在模型设置里选择可用的 Embedding 模型');
      return;
    }
    const missingEmbeddingCredential = getMissingEmbeddingCredential();
    if (missingEmbeddingCredential) {
      alert(missingEmbeddingCredential.message);
      return;
    }
    const embeddingCredentials = getEmbeddingCredentialState();

    setGraphragStatus('building');
    setGraphragError('');

    const embedModel = embedConfig.compositeKey || '';
    const embedProvider = embedConfig.providerId || '';
    const embedApiKey = embeddingCredentials.apiKey || '';
    const embedApiHost = embeddingCredentials.apiHost || '';

    const body = {
      api_key: chatApiKey,
      model: chatModel,
      api_provider: chatProviderFull?.provider || chatProvider,
      api_host: chatProviderFull?.apiHost || '',
      embedding_model: embedModel,
      embedding_provider: embedProvider,
      embedding_api_key: embedApiKey,
      embedding_api_host: embedApiHost,
      force_rebuild: opts.forceRebuild || false,
    };

    // 轮询 progress API
    let pollInterval = null;
    const startPolling = () => {
      pollInterval = setInterval(async () => {
        if (typeof document !== 'undefined' && document.hidden) return;
        try {
          const res = await fetch(`${API_BASE_URL}/document/${docId}/graphrag/progress`);
          if (!res.ok) return;
          const data = await res.json();
          const prog = data.progress || {};
          setGraphragProgress((prev) => (
            prev?.status === prog.status
            && prev?.percent === prog.percent
            && prev?.last_error === prog.last_error
              ? prev
              : prog
          ));
          if (prog.status === 'done') {
            clearInterval(pollInterval);
            // 构建完成，查询 stats
            const statsRes = await fetch(`${API_BASE_URL}/document/${docId}/graphrag/stats`);
            if (statsRes.ok) {
              const statsData = await statsRes.json();
              setGraphragStats(statsData.stats || null);
            }
            setGraphragStatus('built');
          } else if (prog.status === 'failed') {
            clearInterval(pollInterval);
            setGraphragError(prog.last_error || '构建失败');
            setGraphragStatus('error');
          }
        } catch (e) {
          // 轮询失败不中断
        }
      }, 2000);
    };

    try {
      startPolling();
      const res = await fetch(`${API_BASE_URL}/document/${docId}/graphrag/build`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        clearInterval(pollInterval);
        const detail = (await res.json()).detail || '构建失败';
        throw new Error(detail);
      }
      const data = await res.json();
      if (data.loaded_from_disk || data.stats) {
        clearInterval(pollInterval);
        setGraphragStats(data.stats || null);
        setGraphragProgress(data.stats?.build_meta || null);
        setGraphragStatus('built');
      }
    } catch (e) {
      clearInterval(pollInterval);
      console.error('[GraphRAG] 构建失败', e);
      setGraphragError(String(e?.message || e));
      setGraphragStatus('error');
    }
  }, [
    docId,
    getChatCredentials,
    getEmbeddingCredentialState,
    getProviderById,
    getEmbeddingConfig,
    getMissingEmbeddingCredential,
    isMinerUFullRoutePending,
    minerUParsePendingNotice,
  ]);

  // ========== 数据获取函数（useCallback 包裹，稳定引用） ==========
  const fetchAvailableModels = useCallback(async () => {
    try {
      const res = await fetch('/models');
      const data = await res.json();
      setAvailableModels(data);
    } catch (e) { console.error(e); }
  }, []);

  const fetchAvailableEmbeddingModels = useCallback(async () => {
    try {
      const res = await fetch('/embedding_models');
      if (res.ok) setAvailableEmbeddingModels(await res.json());
    } catch (e) { console.error(e); }
  }, []);

  // ========== 划词工具栏相关函数 ==========
  const handleAnnotationToolChange = useCallback((tool) => {
    const nextTool = tool === 'highlight' || tool === 'underline' ? tool : null;
    if (!nextTool) pendingAutoAnnotationRef.current = null;
    setAnnotationTool(nextTool);
  }, []);

  const handleAnnotationColorChange = useCallback((color) => {
    setAnnotationColor(normalizeDocumentHighlightColor(color));
  }, []);

  const queueAutoAnnotation = useCallback((text, anchor = null) => {
    const normalizedText = String(text || '').trim();
    if (!normalizedText || !annotationTool) return false;

    setSelectedSavedHighlightId('');
    selectedPdfHighlightAnchorRef.current = anchor;
    pendingAutoAnnotationRef.current = {
      text: normalizedText,
      color: annotationColor,
      style: annotationTool,
    };
    setSelectedText(normalizedText);
    setShowTextMenu(true);
    setAutoAnnotationRevision((value) => value + 1);
    return true;
  }, [annotationColor, annotationTool, setSelectedText, setShowTextMenu]);

  const handleTextSelection = useCallback(() => {
    const selection = window.getSelection();
    const normalized = normalizePdfSelection({
      selection,
      root: pdfContainerRef.current,
      fallbackPage: currentPage,
    });
    if (normalized?.text) {
      if (queueAutoAnnotation(normalized.text, normalized.anchor)) return;
      setSelectedSavedHighlightId('');
      selectedPdfHighlightAnchorRef.current = normalized.anchor;
      setSelectedText(normalized.text);
      setShowTextMenu(true);
    }
  }, [currentPage, pdfContainerRef, queueAutoAnnotation, setSelectedText, setShowTextMenu]);

  const handleCloseToolbar = useCallback(() => {
    selectedPdfHighlightAnchorRef.current = null;
    pendingAutoAnnotationRef.current = null;
    setSelectedSavedHighlightId('');
    setShowTextMenu(false);
    setSelectedText('');
  }, [setSelectedText, setShowTextMenu]);

  // PDFViewer 的文本选择回调（useCallback 稳定引用，避免 PDFViewer 不必要重渲染）
  const handlePdfTextSelect = useCallback((text, anchor = null) => {
    if (text) {
      if (queueAutoAnnotation(text, anchor)) return;
      setSelectedSavedHighlightId('');
      selectedPdfHighlightAnchorRef.current = anchor;
      setSelectedText(text);
      setShowTextMenu(true);
    }
  }, [queueAutoAnnotation, setSelectedText, setShowTextMenu]);

  const handleToggleSidebar = useCallback(() => {
    setShowSidebar((prev) => !prev);
  }, [setShowSidebar]);

  // ModelQuickSwitch 的思考模式切换回调（useCallback 稳定引用）
  const handleThinkingChange = useCallback((enabled) => {
    setEnableThinking(enabled);
  }, [setEnableThinking]);

  const handleCopy = useCallback(async () => {
    await writePlainTextToClipboard(selectedText);
    return { message: '已复制到剪贴板' };
  }, [selectedText]);

  const handleHighlight = useCallback((color, style = 'highlight') => {
    if (!docId || !selectedText.trim()) return { message: '没有可高亮的文字', tone: 'error' };
    // Guard against accidental event objects from onClick wiring.
    const resolvedColor = normalizeDocumentHighlightColor(
      typeof color === 'string' ? color : DEFAULT_DOCUMENT_HIGHLIGHT_COLOR
    );
    const resolvedStyle = normalizeDocumentHighlightStyle(style);
    const annotationLabel = resolvedStyle === 'underline' ? '下划线' : '高亮';
    const anchor = selectedPdfHighlightAnchorRef.current;
    const newHighlight = createDocumentHighlight({
      text: selectedText,
      page: Number(anchor?.page) || currentPage,
      rects: anchor?.rects || [],
      pageRects: anchor?.page_rects || [],
      color: resolvedColor,
      style: resolvedStyle,
    });
    if (!newHighlight) return { message: '高亮创建失败', tone: 'error' };

    const fingerprint = getDocumentHighlightFingerprint(newHighlight);
    // Same text+geometry with a different color is treated as an update.
    const existingIndex = documentHighlights.findIndex(
      (item) => getDocumentHighlightFingerprint(item) === fingerprint
    );
    let nextHighlights;
    let message = `已添加${annotationLabel}`;
    if (existingIndex >= 0) {
      const existing = documentHighlights[existingIndex];
      if (
        normalizeDocumentHighlightColor(existing.color) === newHighlight.color
        && normalizeDocumentHighlightStyle(existing.style) === newHighlight.style
      ) {
        window.getSelection()?.removeAllRanges?.();
        handleCloseToolbar();
        return { message: `该区域已有${annotationLabel}` };
      }
      nextHighlights = documentHighlights.map((item, index) => (
        index === existingIndex
          ? { ...item, color: newHighlight.color, style: newHighlight.style }
          : item
      ));
      message = `已更新${annotationLabel}`;
    } else {
      nextHighlights = [...documentHighlights, newHighlight];
    }
    setDocumentHighlights(nextHighlights);
    if (!writeDocumentHighlights(docId, nextHighlights)) {
      console.warn('[DocumentHighlight] 高亮已显示，但持久化写入失败');
    }

    window.getSelection()?.removeAllRanges?.();
    handleCloseToolbar();
    return { message };
  }, [currentPage, docId, documentHighlights, handleCloseToolbar, selectedText]);

  useEffect(() => {
    const pending = pendingAutoAnnotationRef.current;
    if (!pending || autoAnnotationRevision === 0) return;
    if (pending.text !== selectedText) {
      pendingAutoAnnotationRef.current = null;
      return;
    }

    pendingAutoAnnotationRef.current = null;
    void handleHighlight(pending.color, pending.style);
  }, [autoAnnotationRevision, handleHighlight, selectedText]);

  const handleSavedHighlightClick = useCallback((highlight) => {
    const text = String(highlight?.text || '').trim();
    if (!text) return;
    const page = Math.max(1, Number(highlight.page) || currentPage);
    selectedPdfHighlightAnchorRef.current = {
      page,
      rects: Array.isArray(highlight.rects) ? highlight.rects : [],
      page_rects: Array.isArray(highlight.page_rects) ? highlight.page_rects : [],
      coordinate_space: 'pdf_top_left_points',
    };
    pendingAutoAnnotationRef.current = null;
    setSelectedSavedHighlightId(highlight.id || '');
    setSelectedText(text);
    setShowTextMenu(true);
  }, [currentPage, setSelectedText, setShowTextMenu]);

  const handleDeleteSelectedAnnotation = useCallback(() => {
    const targetHighlight = documentHighlights.find((item) => item.id === selectedSavedHighlightId);
    if (!targetHighlight) return { message: '没有可删除的标注', tone: 'error' };

    const selectionFingerprint = getDocumentHighlightFingerprint(targetHighlight);
    const relatedNotes = selectionFingerprint
      ? documentNotes.filter((item) => getDocumentHighlightFingerprint(item) === selectionFingerprint)
      : [];
    const nextHighlights = documentHighlights.filter((item) => item.id !== targetHighlight.id);
    const nextNotes = relatedNotes.length > 0
      ? documentNotes.filter((item) => !relatedNotes.some((note) => note.id === item.id))
      : documentNotes;
    const annotationLabel = normalizeDocumentHighlightStyle(targetHighlight.style) === 'underline'
      ? '下划线'
      : '高亮';

    setDocumentHighlights(nextHighlights);
    if (!writeDocumentHighlights(docId, nextHighlights)) {
      console.warn('[DocumentHighlight] 标注已从当前界面移除，但持久化写入失败');
    }
    if (nextNotes !== documentNotes) {
      setDocumentNotes(nextNotes);
      if (!writeDocumentNotes(docId, nextNotes)) {
        console.warn('[DocumentNote] 关联笔记已从当前界面移除，但持久化写入失败');
      }
    }

    setPendingUserNoteRevealId('');
    window.getSelection()?.removeAllRanges?.();
    handleCloseToolbar();
    return { message: relatedNotes.length > 0 ? `已删除${annotationLabel}及关联笔记` : `已删除${annotationLabel}` };
  }, [docId, documentHighlights, documentNotes, handleCloseToolbar, selectedSavedHighlightId]);

  const handleAddNote = useCallback(async (noteContent) => {
    if (!docId || !selectedText.trim()) throw new Error('没有可关联的原文');
    const anchor = selectedPdfHighlightAnchorRef.current;
    const newNote = createDocumentNote({
      text: selectedText,
      note: noteContent,
      page: Number(anchor?.page) || currentPage,
      rects: anchor?.rects || [],
      pageRects: anchor?.page_rects || [],
    });
    if (!newNote) throw new Error('笔记内容不能为空');

    const nextNotes = [...documentNotes, newNote];
    setDocumentNotes(nextNotes);
    setPendingUserNoteRevealId(newNote.id);
    if (!writeDocumentNotes(docId, nextNotes)) {
      console.warn('[DocumentNote] 笔记已显示，但持久化写入失败');
    }
    setRightPanelMode('analysis');
    setActiveHighlight({
      page: newNote.page,
      text: newNote.text,
      source: 'note',
      at: Date.now(),
      citationAnchor: {
        rects: newNote.rects.map((rect) => [
          rect.left,
          rect.top,
          rect.left + rect.width,
          rect.top + rect.height,
        ]),
        coordinateSpace: 'pdf_top_left_points',
      },
    });
    window.getSelection()?.removeAllRanges?.();
    handleCloseToolbar();
    return { message: '笔记已保存' };
  }, [currentPage, docId, documentNotes, handleCloseToolbar, selectedText, setActiveHighlight, setRightPanelMode]);

  const handleAIExplain = useCallback(() => {
    if (isMinerUFullRoutePending) {
      setDeepParseNotice(minerUParsePendingNotice);
      return { message: minerUParsePendingNotice, tone: 'error' };
    }
    const input = `请解释这段话：\n\n"${selectedText}"`;
    setRightPanelMode('chat');
    handleCloseToolbar();
    return sendMessage({ input, interactionMode: 'selection' });
  }, [
    handleCloseToolbar,
    isMinerUFullRoutePending,
    minerUParsePendingNotice,
    selectedText,
    sendMessage,
    setRightPanelMode,
  ]);

  const handleTranslate = useCallback(() => {
    if (isMinerUFullRoutePending) {
      setDeepParseNotice(minerUParsePendingNotice);
      return { message: minerUParsePendingNotice, tone: 'error' };
    }
    const input = `请将以下内容翻译成中文：\n\n"${selectedText}"`;
    setRightPanelMode('chat');
    handleCloseToolbar();
    return sendMessage({ input, interactionMode: 'selection' });
  }, [
    handleCloseToolbar,
    isMinerUFullRoutePending,
    minerUParsePendingNotice,
    selectedText,
    sendMessage,
    setRightPanelMode,
  ]);

  const handleWebSearch = useCallback(async () => {
    const url = buildSelectionSearchUrl({
      engine: searchEngine,
      customUrl: searchEngineUrl,
      query: selectedText,
    });
    await openExternalHttpUrl(url);
    handleCloseToolbar();
  }, [handleCloseToolbar, searchEngine, searchEngineUrl, selectedText]);

  const handleShare = useCallback(async () => {
    const shareText = `📄 来自《${docInfo?.filename || '文档'}》第 ${currentPage} 页：\n\n"${selectedText}"\n\n--- ChatPDF Pro ---`;
    if (typeof navigator.share === 'function') {
      try {
        await navigator.share({
          title: `${docInfo?.filename || '文档'} · 第 ${currentPage} 页`,
          text: shareText,
        });
        return { message: '已打开系统分享' };
      } catch (error) {
        if (error?.name === 'AbortError') return null;
      }
    }
    await writePlainTextToClipboard(shareText);
    return { message: '分享内容已复制' };
  }, [docInfo, currentPage, selectedText]);

  // ========== 搜索导航 ==========
  const goToNextResult = useCallback(() => {
    if (!searchResults.length) return;
    focusResult(currentResultIndex + 1);
  }, [searchResults.length, currentResultIndex, focusResult]);

  const goToPrevResult = useCallback(() => {
    if (!searchResults.length) return;
    focusResult(currentResultIndex - 1);
  }, [searchResults.length, currentResultIndex, focusResult]);

  const clearSearchHistory = useCallback(() => {
    if (!docId) return;
    localStorage.removeItem(`search_history_${docId}`);
    // searchHistory 由 usePDFState 管理，需要通过 pdfState 清除
    pdfState.setSearchHistory?.([]);
  }, [docId, pdfState]);

  const handleClearDocumentAICache = useCallback(async () => {
    if (!docId) return;
    if (isMinerUFullRoutePending) {
      setDeepParseNotice(minerUParsePendingNotice);
      return;
    }
    if (!await confirmAction({
      title: '清理当前文档缓存',
      description: '只清理当前文档的 AI 辅助缓存，不会删除原始 PDF、向量索引或对话历史。',
      confirmLabel: '确认清理',
      tone: 'danger',
    })) {
      return;
    }
    handleStop();
    blockTranslationEpochRef.current += 1;
    pretranslateRunRef.current += 1;
    pretranslateAbortRef.current?.abort();
    pretranslateAbortRef.current = null;
    setPretranslateProgress((current) => ({ ...current, running: false }));
    setTranslatingBlockIds(new Set());
    setBlockTranslateLoading(false);
    try {
      const res = await fetch(`${API_BASE_URL}/documents/${docId}/ai-cache`, { method: 'DELETE' });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({ detail: '清理缓存失败' }));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }
      const cleared = await res.json().catch(() => null);
      if (!cleared || cleared.doc_id !== docId) {
        throw new Error('后端未返回有效的缓存清理结果');
      }
      const removedCaches = Array.isArray(cleared.removed) ? cleared.removed : [];
      if (!removedCaches.includes('table_visual_verification')) {
        throw new Error('后端未确认表格视觉核验缓存已清理');
      }
      invalidateVisualVerificationState();
      readingOutlineCacheRef.current.clear();
      sectionOutlineCacheRef.current.clear();
      setBlockIndexReloadKey((value) => value + 1);
      setReadingOutline(null);
      setReadingOutlineError('');
      setReadingOutlineFallbackNotice('');
      setReadingOutlineReloadKey((prev) => prev + 1);
      setSectionOutline(null);
      setSectionOutlineError('');
      setSectionOutlineFallbackNotice('');
      setSectionOutlineReloadKey((prev) => prev + 1);
      setBlockTranslations({});
      setFailedTranslationBlockIds(new Set());
      setBlockTranslateError('');
      setPretranslateError('');
      setBlockTranslationsLoaded(false);
      setBlockTranslationsLoadedIdentity('');
      setPretranslateProgress({ running: false, done: 0, total: allTranslatableReadingBlocks.length });
      setPretranslateNotice(allTranslatableReadingBlocks.length > 0 ? '缓存已清理，可重新开始全文缓存' : '');
      pretranslateStartedDocRef.current = null;
      clearOverviewCache?.(docId);
      setShowAiProcessingPanel(false);
      alert('当前文档 AI 缓存已清理');
    } catch (error) {
      alert(error.message || '清理缓存失败');
    }
  }, [
    allTranslatableReadingBlocks.length,
    clearOverviewCache,
    confirmAction,
    docId,
    handleStop,
    invalidateVisualVerificationState,
    isMinerUFullRoutePending,
    minerUParsePendingNotice,
  ]);

  const handleRegenerateOverview = useCallback(() => {
    if (!docId) return;
    if (isMinerUFullRoutePending) {
      setDeepParseNotice(minerUParsePendingNotice);
      return;
    }
    fetchOverview?.(overviewDepth, { force: true }).catch(() => {});
    setRightPanelMode('overview');
  }, [
    docId,
    fetchOverview,
    isMinerUFullRoutePending,
    minerUParsePendingNotice,
    overviewDepth,
    setRightPanelMode,
  ]);

  // ========== 预设问题（useMemo 缓存计算结果） ==========
  const showPresetQuestions = useMemo(() => docId && messages.filter(
    msg => msg.type === 'user' || msg.type === 'assistant'
  ).length === 0, [docId, messages]);

  const handlePresetSelect = useCallback((query) => {
    if (isChatInteractionLocked) {
      setDeepParseNotice(chatInteractionLockedNotice);
      return;
    }
    return sendMessage({ input: query, interactionMode: 'preset' });
  }, [chatInteractionLockedNotice, isChatInteractionLocked, sendMessage]);

  // ========== 懒加载设置面板关闭回调（useCallback 稳定引用） ==========
  const openSettings = useCallback(() => {
    setSettingsSection('common');
    setShowSettings(true);
    fetchStorageInfo();
  }, [fetchStorageInfo, setShowSettings]);
  const handleSettingsSectionChange = useCallback((section) => {
    setSettingsSection(section);
    if (section === 'storage') fetchStorageInfo();
  }, [fetchStorageInfo]);
  const handleOpenEmbeddingSettings = useCallback(() => {
    preloadEmbeddingSettings();
    setPendingSettingsPanel('embedding');
    setShowSettings(false);
  }, [setShowSettings]);
  const handleSettingsModalExitComplete = useCallback(() => {
    if (pendingSettingsPanel !== 'embedding') return;
    setShowEmbeddingSettings(true);
    setPendingSettingsPanel(null);
  }, [pendingSettingsPanel, setShowEmbeddingSettings]);
  const handleEmbeddingSettingsClose = useCallback(() => {
    setPendingSettingsPanel('settings');
    setShowEmbeddingSettings(false);
  }, [setShowEmbeddingSettings]);
  const handleEmbeddingSettingsExitComplete = useCallback(() => {
    if (pendingSettingsPanel !== 'settings') return;
    setSettingsSection('common');
    setShowSettings(true);
    setPendingSettingsPanel(null);
  }, [pendingSettingsPanel, setShowSettings]);

  // 设置中心与模型服务面板通过两个 AnimatePresence 串联。动画被系统
  // 打断、窗口切换或页面负载抖动时，Framer Motion 可能无法触发退出回调；
  // 兜底计时器保证不会停在“两个面板都关闭、pending 仍存在”的死状态。
  useEffect(() => {
    if (!pendingSettingsPanel || showSettings || showEmbeddingSettings) return undefined;
    const timerId = window.setTimeout(() => {
      if (pendingSettingsPanel === 'embedding') {
        setShowEmbeddingSettings(true);
      } else if (pendingSettingsPanel === 'settings') {
        setSettingsSection('common');
        setShowSettings(true);
      }
      setPendingSettingsPanel(null);
    }, 450);
    return () => window.clearTimeout(timerId);
  }, [pendingSettingsPanel, setShowEmbeddingSettings, setShowSettings, showEmbeddingSettings, showSettings]);
  const handleGlobalSettingsClose = useCallback(() => { setShowGlobalSettings(false); setSettingsSection('common'); setShowSettings(true); }, [setShowGlobalSettings, setShowSettings]);
  const handleChatSettingsClose = useCallback(() => { setShowChatSettings(false); setSettingsSection('common'); setShowSettings(true); }, [setShowChatSettings, setShowSettings]);
  const handleOCRSettingsClose = useCallback(() => { setShowOCRSettings(false); setSettingsSection('common'); setShowSettings(true); }, [setShowOCRSettings, setShowSettings]);

  const getSidebarMaxWidth = useCallback((handleElement) => {
    const container = handleElement?.parentElement;
    if (!container) return SIDEBAR_MAX_WIDTH;
    const containerStyle = window.getComputedStyle(container);
    const horizontalPadding = (Number.parseFloat(containerStyle.paddingLeft) || 0)
      + (Number.parseFloat(containerStyle.paddingRight) || 0);
    const availableWidth = container.getBoundingClientRect().width
      - horizontalPadding
      - handleElement.getBoundingClientRect().width
      - MAIN_PANEL_MIN_WIDTH;
    return Math.max(SIDEBAR_MIN_WIDTH, Math.min(SIDEBAR_MAX_WIDTH, availableWidth));
  }, []);

  const handleSidebarResizeStart = useCallback((event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.setPointerCapture?.(event.pointerId);
    sidebarResizeRef.current = {
      active: true,
      pointerId: event.pointerId,
      startX: event.clientX,
      startWidth: sidebarWidth,
      maxWidth: getSidebarMaxWidth(event.currentTarget),
    };
    setIsSidebarResizing(true);
  }, [getSidebarMaxWidth, sidebarWidth]);

  const handleSidebarResizeMove = useCallback((event) => {
    const resizeState = sidebarResizeRef.current;
    if (!resizeState.active || resizeState.pointerId !== event.pointerId) return;
    event.preventDefault();
    const nextWidth = Math.round(Math.max(
      SIDEBAR_MIN_WIDTH,
      Math.min(resizeState.maxWidth, resizeState.startWidth + event.clientX - resizeState.startX),
    ));
    setSidebarWidth(nextWidth);
  }, [setSidebarWidth]);

  const handleSidebarResizeEnd = useCallback((event) => {
    const resizeState = sidebarResizeRef.current;
    if (!resizeState.active || resizeState.pointerId !== event.pointerId) return;
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    sidebarResizeRef.current = { ...resizeState, active: false, pointerId: null };
    setIsSidebarResizing(false);
  }, []);

  const handleSidebarResizeKeyDown = useCallback((event) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    event.stopPropagation();
    const maxWidth = getSidebarMaxWidth(event.currentTarget);
    const nextWidth = event.key === 'Home'
      ? SIDEBAR_MIN_WIDTH
      : event.key === 'End'
        ? maxWidth
        : sidebarWidth + (event.key === 'ArrowRight' ? SIDEBAR_KEYBOARD_STEP : -SIDEBAR_KEYBOARD_STEP);
    setSidebarWidth(Math.round(Math.max(SIDEBAR_MIN_WIDTH, Math.min(maxWidth, nextWidth))));
  }, [getSidebarMaxWidth, setSidebarWidth, sidebarWidth]);

  // ========== 根容器点击回调（useCallback 稳定引用） ==========
  const handleRootClick = useCallback((e) => {
    if (!showTextMenu) return;
    const selection = window.getSelection();
    const hasActiveSelection = hasSelectionText(selection);
    if (hasActiveSelection) return;
    if (!e.target.closest('.text-selection-toolbar-container')) {
      handleCloseToolbar();
    }
  }, [showTextMenu, handleCloseToolbar]);

  const deepParseStatusValue = String(currentDeepParseStatus?.status || '').trim().toLowerCase();
  const deepParseStage = currentDeepParseStatus?.stage || documentParseManifest?.stage;
  const resolvedDocumentParseState = useMemo(() => resolveDocumentParseState({
    manifest: documentParseManifest,
    parseReady: documentParseReady,
    deepParseStatus: currentDeepParseStatus,
  }), [currentDeepParseStatus, documentParseManifest, documentParseReady]);
  const deepParseFailed = isMinerUFullRouteFailed || deepParseStatusValue === 'failed';
  const fullRouteAlreadyPublished = isNewMinerUPrimaryRoute
    && ['ready', 'partial_ready'].includes(resolvedDocumentParseState.state);
  const deepParseRunning = !fullRouteAlreadyPublished && (
    ['queued', 'running'].includes(deepParseStatusValue) || Boolean(
      isNewMinerUPrimaryRoute
      && isMinerUFullRoutePending
      && !deepParseFailed
      && !isMinerUFullRouteCancelled
    )
  );
  // 秒表和预估百分比的 1Hz 跳动留在上传卡片 / 任务面板内部。
  // 这里只拍阶段快照；预估百分比按耗时在卡片里重算，避免轮询换对象把卡片抖起来。
  const deepParseProgress = useMemo(() => (
    deepParseRunning
      ? getMinerUProgressPresentation(currentDeepParseStatus, {
        status: deepParseStatusValue || 'running',
        stage: deepParseStage,
        doc_id: docId,
        parse_generation: docInfo?.parse_manifest?.generation || documentParseManifest?.generation,
      })
      : null
  ), [
    currentDeepParseStatus,
    deepParseRunning,
    deepParseStage,
    deepParseStatusValue,
    docId,
    documentParseManifest?.generation,
    docInfo?.parse_manifest?.generation,
  ]);
  const documentUploadParseStatusRaw = useMemo(() => {
    if (resolvedDocumentParseState.resolvedRoute !== 'mineru') return null;
    if (['failed', 'publish_failed'].includes(resolvedDocumentParseState.state)) {
      return {
        status: 'failed',
        title: resolvedDocumentParseState.state === 'publish_failed' ? '问答索引发布失败' : 'MinerU 解析失败',
        description: resolvedDocumentParseState.detail || '解析未完成，可在右上角任务面板中重试。',
      };
    }
    if (resolvedDocumentParseState.state === 'cancelled') {
      return {
        status: 'warning',
        title: 'MinerU 解析已取消',
        description: resolvedDocumentParseState.detail,
      };
    }
    if (resolvedDocumentParseState.state === 'partial_ready') {
      return {
        status: 'warning',
        title: 'MinerU 部分完成',
        description: resolvedDocumentParseState.detail,
      };
    }
    if (resolvedDocumentParseState.state === 'ready') {
      const droppedHeadings = Number(currentDeepParseStatus?.silently_dropped_heading_count || 0);
      const outlineIncomplete = Boolean(currentDeepParseStatus?.structure_degraded);
      return {
        status: 'complete',
        title: 'MinerU 解析完成',
        description: outlineIncomplete
          ? `阅读、速览、翻译和问答已就绪。有 ${droppedHeadings} 个标题未进入大纲，已按 MinerU 版面块发布。`
          : '阅读、速览、大纲、翻译和问答已全部就绪。',
      };
    }
    if (resolvedDocumentParseState.state === 'awaiting_publish') {
      return {
        status: 'warning',
        title: 'MinerU 解析已完成',
        description: resolvedDocumentParseState.detail,
      };
    }
    if (resolvedDocumentParseState.state === 'processing' && deepParseProgress) {
      return {
        status: 'processing',
        title: 'MinerU 全程解析',
        description: deepParseProgress.stageLabel,
        progress: deepParseProgress,
        parseGeneration: String(
          docInfo?.parse_manifest?.generation
          || documentParseManifest?.generation
          || currentDeepParseStatus?.parse_generation
          || ''
        ),
        startedAt: currentDeepParseStatus?.started_at || currentDeepParseStatus?.created_at,
      };
    }
    return null;
  }, [
    currentDeepParseStatus?.silently_dropped_heading_count,
    currentDeepParseStatus?.structure_degraded,
    deepParseProgress,
    docInfo?.parse_manifest?.generation,
    currentDeepParseStatus?.started_at,
    currentDeepParseStatus?.created_at,
    resolvedDocumentParseState,
  ]);
  const documentUploadParseStatusRef = useRef(null);
  const documentUploadParseStatus = stabilizeLiveParseStatus(
    documentUploadParseStatusRef.current,
    documentUploadParseStatusRaw,
  );
  documentUploadParseStatusRef.current = documentUploadParseStatus;

  // ========== 虚拟消息列表渲染回调（useCallback 稳定引用） ==========
  const handleDocumentAwareCitationClick = useCallback(async (citation) => {
    if (!citation || typeof citation !== 'object') return;
    setActiveCitationRef(citation?.ref ?? null);
    const targetDocId = String(citation.doc_id || '').trim();
    if (targetDocId && targetDocId !== String(docId || '')) {
      const session = (history || []).find((item) => String(item?.docId || '') === targetDocId)
        || { id: targetDocId, docId: targetDocId, messages: [] };
      await loadSession(session);
      window.setTimeout(() => handleCitationClick(citation), 180);
      return;
    }
    handleCitationClick(citation);
  }, [docId, handleCitationClick, history, loadSession, setActiveCitationRef]);

  const messageRowRuntime = useMemo(() => ({
    darkMode,
    enableBlurReveal,
    blurIntensity,
    reduceMotion,
    messageStyle,
    reasoningEffort,
    apiProvider,
    docId,
    docFilename: String(docInfo?.filename || ''),
    docPageCount: Number(docInfo?.total_pages || docInfo?.data?.total_pages || 0),
    confirmRegenerateMessage,
    streamingThinkingRef,
    streamingContentRef,
    subscribeContentCommittedPrefix,
    subscribeContentDisplayedText,
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
    getEmbeddingConfig,
  }), [
    darkMode,
    enableBlurReveal,
    blurIntensity,
    reduceMotion,
    messageStyle,
    reasoningEffort,
    apiProvider,
    docId,
    docInfo?.filename,
    docInfo?.total_pages,
    docInfo?.data?.total_pages,
    confirmRegenerateMessage,
    subscribeContentCommittedPrefix,
    subscribeContentDisplayedText,
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
    getEmbeddingConfig,
  ]);

  const openAiProcessingPanel = useCallback(() => {
    setShowAiProcessingPanel(true);
  }, []);
  const closeAiProcessingPanel = useCallback(() => {
    setShowAiProcessingPanel(false);
  }, []);
  const retryMinerUFromParseNotice = useCallback(() => {
    handleStartMinerUDeepParse({ retryFullRoute: true });
  }, [handleStartMinerUDeepParse]);

  const renderMessage = useCallback((msg, idx) => {
    const isEmbeddingIdentityConflict = msg.type === 'assistant' && msg.embeddingIdentityConflict === true;
    const conflictRecoveryStatus = (
      isEmbeddingIdentityConflict && embeddingConflictRecovery.messageId === msg.id
        ? embeddingConflictRecovery.status
        : 'idle'
    );
    const selectedEmbeddingConfig = isEmbeddingIdentityConflict
      ? (getEmbeddingConfig?.() || {})
      : null;
    const currentEmbeddingLabel = selectedEmbeddingConfig?.isValid
      ? `${selectedEmbeddingConfig.modelId || selectedEmbeddingConfig.compositeKey || '当前模型'} · ${selectedEmbeddingConfig.provider?.name || selectedEmbeddingConfig.providerId || '当前 Provider'}`
      : '当前设置不可用';
    const indexedEmbeddingLabel = `${ragIndexStatus?.embedding_model || '未知模型'} · ${ragIndexStatus?.embedding_provider || '未知 Provider'}`;
    return (
      <ChatMessageRow
        msg={msg}
        idx={idx}
        isLatest={idx === messages.length - 1}
        isStreaming={shouldStreamAssistantContent(msg, streamingMessageId)}
        liveParseStatus={documentUploadParseStatus}
        copied={copiedMessageId === (msg.id || idx)}
        liked={likedMessages.has(idx)}
        remembered={rememberedMessages.has(idx)}
        disliked={dislikedMessages.has(idx)}
        conflictRecoveryStatus={conflictRecoveryStatus}
        currentEmbeddingLabel={currentEmbeddingLabel}
        indexedEmbeddingLabel={indexedEmbeddingLabel}
        ragIndexBusy={ragIndexBusy}
        runtime={messageRowRuntime}
      />
    );
  }, [
    copiedMessageId,
    dislikedMessages,
    documentUploadParseStatus,
    embeddingConflictRecovery,
    getEmbeddingConfig,
    likedMessages,
    messageRowRuntime,
    messages.length,
    ragIndexBusy,
    ragIndexStatus?.embedding_model,
    ragIndexStatus?.embedding_provider,
    rememberedMessages,
    streamingMessageId,
  ]);

  // ========== 反馈提交 ==========
  const handleFeedbackSubmit = useCallback(async (issueTypes, detail) => {
    if (!feedbackTarget) return;
    const { idx, msg } = feedbackTarget;
    const prevMsg = messages[idx - 1];
    try {
      await fetch('/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          doc_id: docId || '',
          message_idx: idx,
          feedback_type: 'dislike',
          issue_types: issueTypes,
          detail,
          question: prevMsg?.type === 'user' ? prevMsg.content : '',
          answer: (msg.content || '').slice(0, 500),
          model: msg.model || '',
        }),
      });
      setDislikedMessages(prev => new Set(prev).add(idx));
    } catch (e) {
      console.error('反馈提交失败', e);
    }
    setFeedbackTarget(null);
  }, [feedbackTarget, messages, docId]);

  const sidebarTabs = [
    { id: 'history', label: '会话', Icon: History },
    { id: 'summary', label: '总结', Icon: SummaryIcon },
    { id: 'outline', label: '大纲', Icon: ListFilter },
  ];
  const activeSidebarTabIndex = Math.max(0, sidebarTabs.findIndex((item) => item.id === sidebarMode));
  const rightPanelTabs = [
    { id: 'overview', label: '速览' },
    { id: 'analysis', label: '阅读' },
    { id: 'chat', label: '对话' },
  ];
  const activeRightPanelTabIndex = Math.max(0, rightPanelTabs.findIndex((item) => item.id === rightPanelMode));
  const aiProcessingItems = useMemo(() => {
    const suppressDependentTasks = isMinerUFullRoutePending;
    const overviewTask = downstreamTaskStatuses.overview || {};
    const summaryTask = downstreamTaskStatuses.reading_outline || {};
    const outlineTask = downstreamTaskStatuses.section_outline || {};
    const taskState = (task, fallback) => {
      if (task.status === 'failed') return 'failed';
      if (['partial', 'degraded', 'fallback'].includes(task.status)) return 'recommended';
      return fallback;
    };
    const summaryState = suppressDependentTasks
      ? 'idle'
      : readingOutlineLoading
        ? 'running'
        : readingOutlineError
          ? 'failed'
          : readingOutlineFallbackNotice
            ? 'recommended'
            : taskState(summaryTask, 'idle');
    const outlineState = suppressDependentTasks
      ? 'idle'
      : sectionOutlineLoading
          ? 'running'
          : sectionOutlineError
            ? 'failed'
            : sectionOutlineFallbackNotice
              ? 'recommended'
              : taskState(outlineTask, 'idle');
    const overviewState = suppressDependentTasks
      ? 'idle'
        : overviewLoading
          ? 'running'
          : overviewError
            ? 'failed'
          : taskState(overviewTask, 'idle');
    const translationState = suppressDependentTasks
      ? 'idle'
      : pretranslateProgress.running
        ? 'running'
        : pretranslateError || failedReadingBlockCount > 0
          ? 'failed'
          : 'idle';
    const deepParseStatusText = deepParseFailed
      ? '失败'
      : isMinerUFullRouteCancelled || deepParseStatusValue === 'cancelled'
        ? '已取消'
        : deepParseRunning
       ? (deepParseProgress?.stageLabel || (currentDeepParseStatus?.poll_attempt && currentDeepParseStatus?.poll_total
        ? `等待处理 ${currentDeepParseStatus.poll_attempt}/${currentDeepParseStatus.poll_total}`
        : deepParseStage === 'requesting_upload'
          ? '申请上传中'
          : deepParseStage === 'uploading'
            ? '上传中'
            : deepParseStage === 'polling'
              ? '等待处理'
              : deepParseStage === 'downloading'
                ? '下载结果中'
                : deepParseStage === 'retrying_download'
                  ? '重试下载中'
                  : deepParseStage === 'resuming_result_download'
                    ? '恢复结果中'
                : deepParseStage === 'building_index'
                  ? '重建索引中'
                  : '解析中'))
        : deepParseStatusValue === 'partial_ready' && currentDeepParseStatus?.active_mineru
          ? '部分完成'
          : currentDeepParseStatus?.status === 'ready' && currentDeepParseStatus?.active_mineru
          ? '已完成'
          : currentDeepParseStatus?.configured === false
            ? '未配置'
            : '按需解析';

    const deepParseRecommended = Boolean(
      !isMinerUFullRoutePending
      && !currentDeepParseStatus?.active_mineru
      && !deepParseRunning
      && !deepParseFailed
      && currentDeepParseStatus?.configured !== false
      && currentDeepParseStatus?.recommend_deep_parse
    );
    const deepParseState = deepParseFailed
      ? 'failed'
      : deepParseRunning
        ? 'running'
        : deepParseRecommended
          ? 'recommended'
          : 'idle';
    const canCancelDeepParse = ['queued', 'running'].includes(deepParseStatusValue);
    const canRetryFullRoute = isNewMinerUPrimaryRoute && deepParseFailed;
    const ragIndexSource = ragIndexStatus?.index_source || (ragIndexStatus?.ready ? 'pdf_native' : '');
    const ragIndexIsMinerU = ragIndexSource === 'mineru';
    const ragIndexUpgradeRequired = Boolean(ragIndexStatus?.upgrade_required);
    const ragIndexRecommended = Boolean(
      ragIndexUpgradeRequired || canPublishPendingMinerURag || (
        isLegacyParseManifest
        && currentDeepParseStatus?.active_mineru
        && currentDeepParseStatus?.recommend_rag_index_rebuild
        && !ragIndexBusy
      )
    );
    const persistentRagIndexError = ragIndexStatus?.status === 'failed'
      ? ragIndexStatus?.error || '问答索引处理失败'
      : '';
    const ragIndexTaskError = ragIndexError || persistentRagIndexError;
    const ragIndexState = ragIndexTaskError
      ? 'failed'
      : ragIndexBusy
        ? 'running'
        : ragIndexRecommended
          ? 'recommended'
          : 'idle';
    const ragIndexStatusText = ragIndexTaskError
      ? '处理失败'
      : canPublishPendingMinerURag
      ? '待发布'
      : ragIndexBusy
      ? '处理中'
      : ragIndexUpgradeRequired
        ? '建议升级'
      : !ragIndexStatus?.ready
        ? '未就绪'
        : ragIndexIsMinerU
          ? 'MinerU'
          : ragIndexRecommended
            ? '建议重建'
            : '本地';
    const ragIndexDesc = ragIndexUpgradeRequired
      ? '当前仍是旧版问答索引；升级后会按阅读块顺序重建正文，并隔离表格行与参考文献污染'
      : canPublishPendingMinerURag
      ? 'MinerU 版面结果已就绪，使用当前 Embedding 配置发布问答索引后会统一开放全部能力'
      : !currentDeepParseStatus?.active_mineru
      ? '先完成 MinerU 深度解析后，才能把问答索引升级为同源结构化结果'
      : ragIndexIsMinerU
        ? `问答检索已使用 MinerU 结构化结果${ragIndexStatus?.table_chunk_count ? `，${ragIndexStatus.table_chunk_count} 个结构化表格块` : ''}`
        : currentDeepParseStatus?.recommend_rag_index_reason
          || '当前问答仍使用本地 PDF 解析索引，建议重建为 MinerU 结构化问答索引以改善表格和双栏问答';

    const minerUActionItems = isNewLocalPrimaryRoute ? [] : [
      {
        id: 'deep_parse',
        title: 'MinerU 深度解析',
        state: deepParseState,
        desc: deepParseFailed
          ? currentDeepParseStatus?.error || documentParseManifest?.error || deepParseNotice || 'MinerU 深度解析失败'
          : isMinerUFullRoutePending
          ? minerUParsePendingNotice
          : currentDeepParseStatus?.active_mineru
          ? `当前阅读结构、大纲与速览图表均来自 MinerU${currentDeepParseStatus?.block_count ? `，${currentDeepParseStatus.block_count} 个块` : ''}${currentDeepParseStatus?.figure_count ? `，${currentDeepParseStatus.figure_count} 个图表` : ''}${deepParseStatusValue === 'partial_ready' ? '；部分页面未形成可用正文' : ''}`
          : deepParseRunning && deepParseProgress?.stageLabel
            ? deepParseProgress.stageLabel
          : deepParseRunning && currentDeepParseStatus?.message
            ? currentDeepParseStatus.message
            : currentDeepParseStatus?.status === 'failed' && currentDeepParseStatus?.error
              ? currentDeepParseStatus.error
              : deepParseRecommended
                ? currentDeepParseStatus.recommend_reason
                : `用 MinerU 重建带坐标的阅读块、大纲和速览图表，手动触发才会上传 PDF${currentDeepParseStatus?.access_mode === 'direct' ? '到官方 API' : '到 Worker'}`,
        status: deepParseStatusText,
        progress: deepParseProgress,
        startedAt: currentDeepParseStatus?.started_at || currentDeepParseStatus?.created_at,
        parseGeneration: String(
          docInfo?.parse_manifest?.generation
          || documentParseManifest?.generation
          || currentDeepParseStatus?.parse_generation
          || ''
        ),
        events: currentDeepParseStatus?.events,
        shortfall: currentDeepParseStatus?.shortfall,
        busy: deepParseRunning,
        actionLabel: deepParseRunning && canCancelDeepParse
          ? '取消'
          : deepParseFailed
            ? (canResumeMinerUResultDownload ? '重试下载' : '重试')
            : deepParseRecommended
              ? '开始解析'
              : null,
        onAction: deepParseRunning && canCancelDeepParse
          ? handleCancelMinerUDeepParse
          : canRetryFullRoute
            ? () => handleStartMinerUDeepParse({ retryFullRoute: true })
            : canUseLegacyMinerUActions
              ? handleStartMinerUDeepParse
              : undefined,
        disabled: !docId || currentDeepParseStatus?.configured === false || (!canUseLegacyMinerUActions && !canRetryFullRoute && !canCancelDeepParse),
      },
      ...(isLegacyParseManifest || canPublishPendingMinerURag || ragIndexUpgradeRequired ? [{
        id: 'rag_index',
        title: '问答索引',
        state: ragIndexState,
        desc: ragIndexTaskError || ragIndexNotice || ragIndexDesc,
        status: ragIndexStatusText,
        busy: ragIndexBusy,
        actionLabel: ragIndexState === 'failed'
          ? (canPublishPendingMinerURag ? '重试发布' : '重试重建')
          : ragIndexUpgradeRequired
            ? '升级'
          : canPublishPendingMinerURag
            ? '发布'
            : currentDeepParseStatus?.active_mineru
              ? '重建'
              : null,
        onAction: handleRebuildMinerURagIndex,
        disabled: !docId || ragIndexBusy || (!ragIndexUpgradeRequired && !canPublishPendingMinerURag && !currentDeepParseStatus?.active_mineru),
      }] : []),
    ];

    const localRagUpgradeItems = isNewLocalPrimaryRoute && ragIndexUpgradeRequired
      ? [{
        id: 'rag_index',
        title: '问答索引',
        state: ragIndexState,
        desc: ragIndexTaskError || ragIndexNotice || ragIndexDesc,
        status: ragIndexStatusText,
        busy: ragIndexBusy,
        actionLabel: ragIndexState === 'failed' ? '重试升级' : '升级',
        onAction: handleRebuildMinerURagIndex,
        disabled: !docId || ragIndexBusy,
      }]
      : [];

    return [
      ...minerUActionItems,
      ...localRagUpgradeItems,
      {
        id: 'summary',
        title: 'AI 总结',
        state: summaryState,
        desc: readingOutlineError || readingOutlineFallbackNotice || '左侧总结栏的结构化论文梳理',
        status: summaryState === 'failed' ? '生成失败' : summaryState === 'recommended' ? '已降级' : '生成中',
        events: summaryTask.events,
        shortfall: summaryTask.shortfall,
        busy: readingOutlineLoading,
        actionLabel: ['failed', 'recommended'].includes(summaryState) ? '重试' : null,
        onAction: handleRegenerateReadingOutline,
        disabled: readingOutlineLoading || !docId || isMinerUFullRoutePending,
      },
      {
        id: 'outline',
        title: '章节大纲',
        state: outlineState,
        desc: sectionOutlineError || sectionOutlineFallbackNotice || '左侧大纲栏的原文章节树',
        status: outlineState === 'failed' ? '生成失败' : outlineState === 'recommended' ? '已降级' : '生成中',
        events: outlineTask.events,
        shortfall: outlineTask.shortfall,
        busy: sectionOutlineLoading,
        actionLabel: ['failed', 'recommended'].includes(outlineState) ? '重试' : null,
        onAction: handleRegenerateSectionOutline,
        disabled: sectionOutlineLoading || !docId || isMinerUFullRoutePending,
      },
      {
        id: 'overview',
        title: '速览',
        state: overviewState,
        desc: overviewError || `当前默认详细度：${overviewDepth === 'brief' ? '简略' : overviewDepth === 'detailed' ? '详细' : '标准'}`,
        status: overviewState === 'failed' ? '生成失败' : '生成中',
        events: overviewTask.events,
        shortfall: overviewTask.shortfall,
        busy: overviewLoading,
        actionLabel: overviewState === 'failed' ? '重试' : null,
        onAction: handleRegenerateOverview,
        disabled: overviewLoading || !docId || isMinerUFullRoutePending,
      },
      {
        id: 'translation',
        title: '悬浮翻译',
        state: translationState,
        desc: pretranslateError || pretranslateNotice || (failedReadingBlockCount > 0
          ? `有 ${failedReadingBlockCount} 个段落尚未完成，可继续补齐`
          : '预缓存段落翻译，悬浮时直接显示'),
        status: pretranslateProgress.running
          ? `${Math.min(pretranslateProgress.done, pretranslateProgress.total)}/${pretranslateProgress.total || allTranslatableReadingBlocks.length}`
          : failedReadingBlockCount > 0
            ? `失败 ${failedReadingBlockCount}`
            : '缓存失败',
        busy: pretranslateProgress.running,
        actionLabel: pretranslateProgress.running
          ? '取消'
          : translationState === 'failed'
            ? (failedReadingBlockCount > 0 ? '补齐失败' : '重试')
            : null,
        onAction: pretranslateProgress.running
          ? cancelPretranslateReadingDocument
          : handleStartPretranslate,
        disabled: !docId || isMinerUFullRoutePending || allTranslatableReadingBlocks.length === 0,
      },
    ];
  }, [
    allTranslatableReadingBlocks.length,
    canPublishPendingMinerURag,
    canResumeMinerUResultDownload,
    canUseLegacyMinerUActions,
    cancelPretranslateReadingDocument,
    deepParseFailed,
    deepParseNotice,
    deepParseProgress,
    docId,
    deepParseRunning,
    deepParseStage,
    deepParseStatusValue,
    downstreamTaskStatuses,
    documentParseManifest?.error,
    documentParseManifest?.stage,
    currentDeepParseStatus?.active_mineru,
    currentDeepParseStatus?.access_mode,
    currentDeepParseStatus?.recommend_deep_parse,
    currentDeepParseStatus?.recommend_rag_index_rebuild,
    currentDeepParseStatus?.recommend_rag_index_reason,
    currentDeepParseStatus?.recommend_reason,
    currentDeepParseStatus?.block_count,
    currentDeepParseStatus?.figure_count,
    currentDeepParseStatus?.configured,
    currentDeepParseStatus?.message,
    currentDeepParseStatus?.created_at,
    currentDeepParseStatus?.started_at,
    currentDeepParseStatus?.poll_attempt,
    currentDeepParseStatus?.poll_total,
    currentDeepParseStatus?.progress,
    currentDeepParseStatus?.remote_progress_percent,
    currentDeepParseStatus?.events,
    currentDeepParseStatus?.shortfall,
    currentDeepParseStatus?.stage,
    currentDeepParseStatus?.status,
    failedReadingBlockCount,
    handleRebuildMinerURagIndex,
    handleStartPretranslate,
    handleRegenerateOverview,
    handleRegenerateReadingOutline,
    handleRegenerateSectionOutline,
    handleCancelMinerUDeepParse,
    handleStartMinerUDeepParse,
    overviewDepth,
    overviewError,
    overviewLoading,
    pretranslateProgress.done,
    pretranslateProgress.running,
    pretranslateProgress.total,
    pretranslateError,
    pretranslateNotice,
    ragIndexBusy,
    ragIndexError,
    ragIndexNotice,
    ragIndexStatus?.index_source,
    ragIndexStatus?.error,
    ragIndexStatus?.ready,
    ragIndexStatus?.status,
    ragIndexStatus?.table_chunk_count,
    ragIndexStatus?.upgrade_required,
    requiresMinerURagSource,
    readingOutlineError,
    readingOutlineFallbackNotice,
    readingOutlineLoading,
    sectionOutlineError,
    sectionOutlineFallbackNotice,
    sectionOutlineLoading,
    isMinerUFullRoutePending,
    isMinerUFullRouteFailed,
    isMinerUFullRouteCancelled,
    isLegacyParseManifest,
    isNewLocalPrimaryRoute,
    isNewMinerUPrimaryRoute,
    minerUParsePendingNotice,
  ]);

  const uploadStatusMeta = UPLOAD_STATUS_META[uploadStatus] || UPLOAD_STATUS_META.uploading;
  const uploadProgressLabel = Math.max(0, Math.min(100, Math.round(Number(uploadProgress) || 0)));
  const showDocumentWorkspace = Boolean(docId && !isUploading);
  const hasDockedSelectionToolbar = showDocumentWorkspace;

  useEffect(() => {
    if (showDocumentWorkspace) preloadPDFViewer();
  }, [showDocumentWorkspace]);

  // 吸附栏只存在于 PDFViewer 里（需要 showDocumentWorkspace && pdf_url）。
  // 没有阅读器却残留 'dock'，右侧翻译卡会被过滤掉、左边又没有吸附栏 —— 两边都看不到译文，
  // 而窄条还理直气壮写着「译文在 PDF 吸附栏」。所以阅读器不在场时强制回落。
  const canDockTranslation = showDocumentWorkspace && Boolean(docInfo?.pdf_url);
  useEffect(() => {
    if (canDockTranslation) return;
    setTranslationSurface((current) => (current === 'panel' ? current : 'panel'));
  }, [canDockTranslation]);

  // 工具栏渲染在 PDF 面板内部（PDFViewer 的正常流里），不再是浮在应用最顶层的 fixed 层。
  const selectionToolbarNode = useMemo(() => (
    hasDockedSelectionToolbar ? (
      <TextSelectionToolbar
        selectedText={selectedText}
        darkMode={darkMode}
        onCopy={handleCopy}
        annotationTool={annotationTool}
        annotationColor={annotationColor}
        onAnnotationToolChange={handleAnnotationToolChange}
        onAnnotationColorChange={handleAnnotationColorChange}
        canDeleteAnnotation={Boolean(selectedSavedHighlightId)}
        onDeleteAnnotation={handleDeleteSelectedAnnotation}
        onAddNote={handleAddNote}
        onAIExplain={handleAIExplain}
        onTranslate={handleTranslate}
        onWebSearch={handleWebSearch}
        onShare={handleShare}
        size={toolbarSize}
      />
    ) : null
  ), [
    annotationColor,
    annotationTool,
    darkMode,
    handleAIExplain,
    handleAddNote,
    handleAnnotationColorChange,
    handleAnnotationToolChange,
    handleCopy,
    handleDeleteSelectedAnnotation,
    handleShare,
    handleTranslate,
    handleWebSearch,
    hasDockedSelectionToolbar,
    selectedSavedHighlightId,
    selectedText,
    toolbarSize,
  ]);
  const canRollbackCurrentRagIndex = Boolean(
    canUseLegacyMinerUActions
    && ragIndexStatus?.index_source === 'mineru'
    && ragIndexStatus?.can_rollback
    && !ragIndexBusy
  );
  const visibleBackgroundTasksRaw = useMemo(
    () => getVisibleBackgroundTasks(aiProcessingItems, showDocumentWorkspace),
    [aiProcessingItems, showDocumentWorkspace]
  );
  const visibleBackgroundTasksRef = useRef(visibleBackgroundTasksRaw);
  const visibleBackgroundTasks = stabilizeBackgroundTaskItems(
    visibleBackgroundTasksRef.current,
    visibleBackgroundTasksRaw,
  );
  visibleBackgroundTasksRef.current = visibleBackgroundTasks;
  const backgroundTaskSummary = useMemo(
    () => getBackgroundTaskSummary(visibleBackgroundTasks),
    [visibleBackgroundTasks]
  );
  const backgroundTaskPillLabel = backgroundTaskSummary.state === 'running'
    ? `任务 ${Math.max(1, backgroundTaskSummary.count || 0)}`
    : backgroundTaskSummary.label;
  const activeSettingsSectionMeta = SETTINGS_SECTIONS.find((section) => section.id === settingsSection) || SETTINGS_SECTIONS[0];
  const ActiveSettingsSectionIcon = activeSettingsSectionMeta.Icon;
  const baseRetrievalEnabledCount = [enableVectorSearch, enableJiebaBM25].filter(Boolean).length;
  const graphRagStatusLabel = !enableGraphRAG
    ? '未启用'
    : graphragStatus === 'built'
      ? '已就绪'
      : graphragStatus === 'building'
        ? `构建中${graphragProgress?.progress > 0 ? ` ${graphragProgress.progress}%` : ''}`
        : graphragStatus === 'error'
          ? '需要处理'
          : docId
            ? '待构建'
            : '等待文档';
  const agentModeLabel = !enableAgentRetrieval
    ? '未启用'
    : forceAgentRetrieval
      ? '全部问题'
      : '按需触发';

  useEffect(() => {
    if (!showDocumentWorkspace) setShowAiProcessingPanel(false);
  }, [showDocumentWorkspace]);

  // ========== 渲染 ==========
  return (
    <div
      className={`chatpdf-shell ${docId ? 'chatpdf-shell--document' : 'chatpdf-shell--welcome'} ${darkMode ? 'dark chatpdf-shell--dark' : ''} h-screen w-full overflow-hidden transition-colors duration-300 ${darkMode ? 'bg-[#111318] text-gray-200' : 'text-[var(--color-text-main)]'}`}
      onClick={handleRootClick}
    >
      <LocalParserInstallDialog
        open={isLocalParserInstallOpen}
        darkMode={darkMode}
        onClose={handleLocalParserInstallClose}
        onReady={handleLocalParserReady}
      />
      <SessionDeleteDialog
        session={pendingSessionDelete}
        darkMode={darkMode}
        onClose={() => setPendingSessionDelete(null)}
        onConfirm={confirmSessionDelete}
      />
      <ConfirmDialog {...confirmDialogProps} darkMode={darkMode} />

      {/* 统一应用外壳：2K 屏基本铺满，超宽屏保留上限；阅读态继续使用紧凑间距。 */}
      <div className={`relative mx-auto flex h-full w-full max-w-[2400px] ${docId ? 'px-3 py-3' : 'px-3 py-4 sm:px-4 sm:py-5'}`}>

      {/* 侧边栏（历史记录） */}
      {isNarrowDesktop && showSidebar && (
        <button
          type="button"
          aria-label="关闭侧边栏"
          onClick={() => setShowSidebar(false)}
          className="absolute inset-0 z-40 bg-black/20 backdrop-blur-[2px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[#FFA07A]/35"
        />
      )}
      <motion.div
        initial={false}
        animate={{ width: showSidebar ? sidebarWidth : 0, opacity: showSidebar ? 1 : 0 }}
        transition={{ duration: isSidebarResizing ? 0 : 0.2, ease: "easeInOut" }}
        style={{ pointerEvents: showSidebar ? 'auto' : 'none' }}
        className={`flex flex-col overflow-hidden rounded-[28px] ${
          isNarrowDesktop
            ? `absolute left-3 z-50 h-auto ${docId ? 'bottom-3 top-3' : 'bottom-4 top-4 sm:bottom-5 sm:left-4 sm:top-5'}`
            : 'z-20 h-full flex-shrink-0'
        } ${darkMode ? 'border border-white/[0.07] bg-[#191c21]/95 shadow-[0_20px_44px_-24px_rgba(0,0,0,0.72)] backdrop-blur-3xl backdrop-saturate-150' : 'bg-[#26272b] shadow-[0_20px_44px_-20px_rgba(28,30,34,0.5)]'}`}
      >
        <div className="mx-auto flex h-full flex-col items-stretch relative" style={{ width: sidebarWidth, minWidth: sidebarWidth }}>
          <div className="px-5 pt-6 pb-4 flex items-center justify-between mb-1">
            <div className="flex items-center gap-2.5 font-bold text-lg tracking-tight pl-1">
              <svg
                aria-hidden="true"
                viewBox="0 0 24 24"
                className="h-7 w-7 flex-shrink-0 text-[#FFA07A]"
              >
                <path
                  fill="currentColor"
                  d="M11 1v1H7a3 3 0 0 0-3 3v3a5 5 0 0 0 5 5h6a5 5 0 0 0 5-5V5a3 3 0 0 0-3-3h-4V1zM6 5a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v3a3 3 0 0 1-3 3H9a3 3 0 0 1-3-3zm3.5 4a1.5 1.5 0 1 0 0-3a1.5 1.5 0 0 0 0 3m5 0a1.5 1.5 0 1 0 0-3a1.5 1.5 0 0 0 0 3M6 22a6 6 0 0 1 12 0h2a8 8 0 1 0-16 0z"
                />
              </svg>
              <span className="text-gray-100">ChatPDF</span>
            </div>
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => setDarkMode(!darkMode)}
                aria-label={darkMode ? '切换到浅色模式' : '切换到深色模式'}
                title={darkMode ? '浅色模式' : '深色模式'}
                className={`rounded-full border p-2 transition-[background-color,border-color,color,transform] duration-200 active:scale-95 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#FFA07A]/35 ${darkMode ? 'border-white/[0.08] bg-white/[0.035] text-gray-400 hover:border-[#FFA07A]/25 hover:bg-[#FFA07A]/10 hover:text-[#FFD5C7]' : 'border-transparent text-gray-400 hover:bg-white/10 hover:text-gray-200'}`}
              >
                {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
              </button>
            </div>
          </div>

          <div className="px-4 mb-4 flex justify-center">
            <button
              onClick={openUploadHome}
              className="tanya-btn w-full"
            >
              <Plus className="w-4 h-4 opacity-70" />
              <span>上传 PDF</span>
            </button>
            <input ref={fileInputRef} type="file" accept=".pdf" onChange={handleUploadInputChange} className="hidden" />
          </div>

          <div className="px-4 mb-3">
            <div className="relative grid grid-cols-3 overflow-hidden rounded-[20px] border border-white/[0.12] bg-[#202126] p-1 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
              <motion.div
                className="absolute bottom-1 left-1 top-1 rounded-[16px] bg-[#484a52] ring-1 ring-inset ring-white/[0.14] shadow-[0_5px_14px_rgba(0,0,0,0.34),inset_0_1px_0_rgba(255,255,255,0.14)]"
                initial={false}
                style={{ width: 'calc((100% - 0.5rem) / 3)' }}
                animate={{ x: `${activeSidebarTabIndex * 100}%` }}
                transition={{ type: 'spring', stiffness: 430, damping: 34, mass: 0.72 }}
              />
              {sidebarTabs.map(({ id, label, Icon }) => {
                const isActive = sidebarMode === id;
                return (
                  <motion.button
                    key={id}
                    type="button"
                    onClick={() => setSidebarMode(id)}
                    whileTap={{ scale: 0.97 }}
                    className={`relative z-10 flex h-8 items-center justify-center gap-1.5 rounded-[16px] text-[12px] font-semibold transition-colors duration-200 ${
                      isActive ? 'text-white' : 'text-gray-500 hover:text-gray-300'
                    }`}
                  >
                    <Icon className={`h-3.5 w-3.5 transition-all duration-200 ${isActive ? 'scale-105 text-[#FFA07A]' : 'scale-100 opacity-80'}`} />
                    <span>{label}</span>
                  </motion.button>
                );
              })}
            </div>
          </div>

          {sidebarMode === 'history' ? (
            /* 与总结/大纲一致：内容统一放浅色内嵌卡，切 tab 不再明暗跳变 */
            <div className={`flex-1 min-h-0 mx-3 mb-2 p-2.5 overflow-hidden rounded-[16px] ${darkMode ? 'bg-white/[0.03]' : 'bg-[#f5f4f2]'}`}>
              <div className="h-full overflow-y-auto custom-scrollbar">
                <h2 className={`mb-2.5 pl-2 pt-1 text-[11px] font-bold tracking-wider ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  会话历史
                </h2>
                <ul className="space-y-1 relative">
                  {history.map((item, idx) => {
                    const isActive = item.id === (pendingHistoryId ?? docId);
                    return (
                      <li key={item.id ?? item.docId ?? idx}>
                        <motion.div
                          onClick={() => handleHistorySessionClick(item)}
                          className={`relative isolate w-full flex items-center justify-between px-3 py-2.5 rounded-[12px] cursor-pointer group transition-colors duration-200 ${
                            isActive
                              ? (darkMode ? 'z-10 text-white font-bold' : 'z-10 text-gray-900 font-bold')
                              : (darkMode ? 'text-gray-400 font-medium hover:bg-white/5 hover:text-gray-200' : 'text-gray-600 font-medium hover:bg-white/80 hover:text-gray-900')
                          }`}
                        >
                          {isActive && (
                            <motion.div
                              layoutId="history-active-session-card"
                              aria-hidden="true"
                              initial={false}
                              className={`absolute inset-0 z-0 rounded-[12px] will-change-transform ${
                                darkMode
                                  ? 'bg-white/10 ring-1 ring-inset ring-white/[0.08] shadow-[0_4px_12px_rgba(0,0,0,0.16)]'
                                  : 'bg-white ring-1 ring-inset ring-black/[0.025] shadow-[0_4px_12px_rgba(31,41,55,0.08),0_1px_2px_rgba(31,41,55,0.04)]'
                              }`}
                              transition={{ type: 'tween', duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
                            />
                          )}
                          <div className="relative z-10 flex items-center gap-2.5 overflow-hidden">
                            <MessageSquare
                              size={16}
                              className={`${isActive ? 'text-[#B85F47]' : 'text-gray-400'} transition-colors duration-200 flex-shrink-0`}
                              strokeWidth={isActive ? 2.5 : 2}
                            />
                            <span className="text-[13px] truncate">{item.filename}</span>
                          </div>
                          <button
                            onClick={(e) => { e.stopPropagation(); requestSessionDelete(item); }}
                            aria-label={`删除会话 ${item.filename}`}
                            title="删除会话"
                            className={`relative z-10 opacity-0 group-hover:opacity-100 p-1 rounded-full text-gray-400 transition-all flex-shrink-0 ${darkMode ? 'hover:bg-red-500/15 hover:text-red-400' : 'hover:bg-red-50 hover:text-red-500'}`}
                          >
                            <Trash2 size={15} strokeWidth={2} />
                          </button>
                        </motion.div>
                      </li>
                    );
                  })}
                </ul>
              </div>
            </div>
          ) : sidebarMode === 'summary' ? (
            /* 长文阅读内容放浅色内嵌卡：深色栏只当外框，可读性优先 */
            <div className={`flex-1 min-h-0 mx-3 mb-2 p-2.5 overflow-hidden rounded-[16px] ${darkMode ? 'bg-white/[0.03]' : 'bg-[#f5f4f2]'}`}>
              <ReadingSummaryPanel
                items={readingOutlineItems}
                loading={readingOutlineLoading}
                error={readingOutlineError}
                activeNodeId={activeReadingNodeId}
                visitedNodeIds={[...visitedReadingNodeIds]}
                onJump={handleOutlineJump}
                onRetry={handleRegenerateReadingOutline}
                source={readingOutline?.source || ''}
                generationError={readingOutline?.meta?.generation_error || ''}
                retrying={readingOutlineLoading}
                meta={readingOutline?.meta || {}}
                darkMode={darkMode}
              />
            </div>
          ) : (
            <div className={`flex-1 min-h-0 mx-3 mb-2 p-2.5 overflow-hidden rounded-[16px] ${darkMode ? 'bg-white/[0.03]' : 'bg-[#f5f4f2]'}`}>
              <DocumentOutline
                outline={pdfOutlineItems}
                loading={pdfOutlineLoading}
                error={pdfOutlineError}
                source={pdfOutlineSource}
                currentPage={currentPage}
                activeBlockId={activeReadingBlockId}
                activeNodeId={activeSectionNodeId}
                visitedNodeIds={[...visitedSectionNodeIds]}
                onJump={handleSectionOutlineJump}
                darkMode={darkMode}
              />
            </div>
          )}

          <div className="px-4 py-4 border-t border-white/[0.06]">
            <button
              onClick={openSettings}
              className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-[14px] transition-all duration-300 text-gray-400 font-medium hover:bg-white/5 hover:text-gray-200"
            >
              <Settings size={17} className="text-gray-500" strokeWidth={2} />
              <span className="text-[13px]">设置中心</span>
            </button>
          </div>
        </div>
      </motion.div>

      {showSidebar && !isNarrowDesktop && (
        <div
          role="separator"
          aria-label="调整侧边栏宽度"
          aria-orientation="vertical"
          aria-valuemin={SIDEBAR_MIN_WIDTH}
          aria-valuemax={SIDEBAR_MAX_WIDTH}
          aria-valuenow={Math.round(sidebarWidth)}
          tabIndex={0}
          title="拖拽调整侧边栏宽度，双击恢复默认"
          onPointerDown={handleSidebarResizeStart}
          onPointerMove={handleSidebarResizeMove}
          onPointerUp={handleSidebarResizeEnd}
          onPointerCancel={handleSidebarResizeEnd}
          onKeyDown={handleSidebarResizeKeyDown}
          onDoubleClick={() => setSidebarWidth(SIDEBAR_DEFAULT_WIDTH)}
          className={`group relative z-30 flex h-full flex-shrink-0 touch-none select-none cursor-col-resize items-center justify-center focus-visible:outline-none ${docId ? 'w-3' : 'w-4'}`}
        >
          <div className={`flex h-12 w-3 items-center justify-center rounded-full border transition-all duration-200 group-focus-visible:ring-2 group-focus-visible:ring-[#FFA07A]/40 ${
            isSidebarResizing
              ? 'scale-110 accent-control shadow-[0_7px_18px_rgba(184,95,71,0.18)]'
              : darkMode
                ? 'border-white/10 bg-[#30333a] text-gray-500 group-hover:border-white/20 group-hover:bg-[#3a3e46] group-hover:text-[#FFA07A]'
                : 'border-white/90 bg-white/85 text-gray-400 shadow-[0_5px_14px_rgba(60,56,52,0.16)] group-hover:text-[#B85F47]'
          }`}>
            <GripVertical className="h-4 w-3" />
          </div>
        </div>
      )}

      {/* 主内容区域：统一承托 PDF 预览与对话面板的工作台底板 */}
      <div className={`workspace-surface flex-1 min-w-0 flex flex-col h-full relative rounded-[28px] overflow-hidden ${isSidebarResizing ? 'transition-none' : 'transition-[background-color,border-color,box-shadow] duration-200 ease-out'} ${darkMode ? 'workspace-surface-dark' : ''}`}>
        {/* 侧边栏展开按钮 (未打开文档时显示) */}
        {!showSidebar && (!docId || isNarrowDesktop) && (
          <button
            onClick={() => setShowSidebar(true)}
            type="button"
            aria-label="显示侧边栏"
            className={`absolute top-4 left-4 z-20 p-2 backdrop-blur-md shadow-sm rounded-full hover:scale-105 transition-all border ${darkMode ? 'bg-white/10 text-gray-300 border-white/10 hover:bg-white/20' : 'bg-white/80 text-gray-700 border-white/50 hover:bg-white'}`}
            title="显示侧边栏"
          >
            <Menu className="w-5 h-5" />
          </button>
        )}

        {/* 内容区域：阅读态用更小的内边距 */}
        <div className={`flex-1 flex overflow-hidden ${showDocumentWorkspace ? 'p-3 gap-3' : `p-5 gap-4 ${isNarrowDesktop ? 'pt-14' : ''}`}`}>
          {/* 左侧：PDF 预览。parse/chat 状态变化时由 memo 挡住 PDFViewer。 */}
          {showDocumentWorkspace ? (
            <PdfWorkspacePane
              darkMode={darkMode}
              pdfPanelWidth={pdfPanelWidth}
              docInfo={docInfo}
              currentPage={currentPage}
              onPageChange={setCurrentPage}
              pdfScale={pdfScale}
              onPdfScaleChange={setPdfScale}
              pdfContainerRef={pdfContainerRef}
              activeHighlight={activeHighlight}
              documentHighlights={documentHighlights}
              onSavedHighlightClick={handleSavedHighlightClick}
              isSelectingArea={isSelectingArea}
              onAreaSelected={handleAreaSelected}
              onSelectionCancel={handleSelectionCancel}
              onTextSelect={handlePdfTextSelect}
              onToggleSidebar={handleToggleSidebar}
              blockIndex={blockIndex}
              activeReadingBlockId={activeReadingBlockId}
              focusedReadingBlockIds={focusedReadingBlockIds}
              focusPulseToken={readingJumpPulseToken}
              navigationRequest={readerNavigationRequest}
              onBlockHover={handleReadingBlockHover}
              onBlockClick={handleReadingBlockClick}
              blockTranslations={blockTranslations}
              translatingBlockIds={translatingBlockIdList}
              hasDockedSelectionToolbar={hasDockedSelectionToolbar}
              selectionToolbar={selectionToolbarNode}
              translationSurface={translationSurface}
              onTranslationSurfaceChange={setTranslationSurface}
              onTextSelection={handleTextSelection}
              searchStatus={searchStatus}
              onDismissSearchStatus={dismissSearchStatus}
            />
          ) : (
            /* 空状态：左栏工作台（问候 + 上传 + 功能 + 最近文件），不再留大面积空白 */
            <div className="flex-1 min-w-0 flex flex-col overflow-y-auto custom-scrollbar px-3 pt-4">
              <div className="mb-6 px-1">
                <h1 className={`text-[32px] font-extrabold tracking-tight leading-tight ${darkMode ? 'text-gray-100' : 'text-gray-900'}`}>开始阅读论文</h1>
                <p className={`text-[14px] mt-1.5 font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>上传 PDF，AI 帮你速览、解析和对话</p>
              </div>

              <UploadDocumentCard
                darkMode={darkMode}
                isUploading={isUploading}
                uploadProgress={uploadProgressLabel}
                uploadStatus={uploadStatus}
                uploadStatusMeta={uploadStatusMeta}
                uploadFileInfo={uploadFileInfo}
                parseRoute={selectedParseRoute}
                onParseRouteChange={handleUploadRouteChange}
                onSelect={() => handleChooseUploadFile(true)}
                onWarmup={preloadPDFViewer}
              />

              <div className="mt-6 px-1">
                <h3 className={`text-[13px] font-bold mb-3 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>支持功能</h3>
                <div className="grid grid-cols-5 gap-2">
                  {[
                    { Icon: Sparkles, label: '智能速览' },
                    { Icon: ListFilter, label: '章节大纲' },
                    { Icon: Globe, label: '划词翻译' },
                    { Icon: Database, label: '表格问答' },
                    { Icon: ArrowUpRight, label: '引用溯源' },
                  ].map(({ Icon, label }) => (
                    <div key={label} className="flex flex-col items-center gap-2">
                      <div className={`w-12 h-12 rounded-full flex items-center justify-center ${darkMode ? 'bg-white/[0.06] text-gray-300' : 'bg-white text-gray-600 shadow-[var(--shadow-sm)]'}`}>
                        <Icon className="w-[18px] h-[18px]" />
                      </div>
                      <span className={`text-[11px] font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>{label}</span>
                    </div>
                  ))}
                </div>
              </div>

              {history.length === 0 && (
                <div className="mt-7 px-1 pb-4">
                  <h3 className={`text-[13px] font-bold mb-3 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>工作流程</h3>
                  <div className={`rounded-[18px] border p-5 space-y-4 ${darkMode ? 'border-white/[0.07] bg-white/[0.045] shadow-[inset_0_1px_0_rgba(255,255,255,0.025)]' : 'border-transparent bg-white shadow-[var(--shadow-sm)]'}`}>
                    {[
                      { n: '1', title: '上传文档', desc: 'PDF 自动解析文字、表格与版面结构' },
                      { n: '2', title: '生成速览', desc: 'AI 提炼大纲、总结和关键结论' },
                      { n: '3', title: '开始对话', desc: '基于原文提问，回答附引用可溯源' },
                    ].map(({ n, title, desc }) => (
                      <div key={n} className="flex items-start gap-3.5">
                        <div className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-[12px] font-bold ${darkMode ? 'bg-[#FFA07A]/12 text-[#FFD1C1]' : 'bg-[#FFF4EF] text-[#B85F47]'}`}>{n}</div>
                        <div className="min-w-0 pt-0.5">
                          <div className={`text-[13px] font-semibold ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>{title}</div>
                          <div className={`text-[12px] mt-0.5 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>{desc}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {history.length > 0 && (
                <div className="mt-7 px-1 pb-4">
                  <h3 className={`text-[13px] font-bold mb-3 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>最近文件</h3>
                  <div className="grid grid-cols-2 gap-2.5">
                    {history.slice(0, 6).map((item) => (
                      <button
                        key={item.id}
                        onClick={() => loadSession(item)}
                        onPointerEnter={preloadPDFViewer}
                        className={`flex items-center gap-3 p-3.5 rounded-[16px] text-left transition-all hover:-translate-y-0.5 ${darkMode ? 'bg-white/[0.04] hover:bg-white/[0.07]' : 'bg-white shadow-[var(--shadow-sm)] hover:shadow-[var(--shadow-card)]'}`}
                      >
                        <div className={`w-9 h-9 rounded-[10px] flex items-center justify-center shrink-0 ${darkMode ? 'bg-purple-500/10 text-purple-300' : 'bg-purple-50 text-purple-500'}`}>
                          <MessageSquare size={15} />
                        </div>
                        <div className="min-w-0">
                          <div className={`text-[13px] font-semibold truncate ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>{item.filename}</div>
                          <div className={`text-[11px] mt-0.5 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>点击继续对话</div>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* 阅读区与对话区共用的可拖拽分隔器 */}
          <div
            role="separator"
            aria-label="调整阅读区与对话区宽度"
            aria-orientation="vertical"
            aria-valuemin={isNarrowDesktop ? 38 : 30}
            aria-valuemax={isNarrowDesktop ? 62 : 70}
            aria-valuenow={Math.round(pdfPanelWidth)}
            tabIndex={0}
            title="拖拽或使用左右方向键调整宽度，双击恢复均分"
            className="group relative z-20 -ml-2 flex h-full w-4 flex-shrink-0 touch-none select-none cursor-col-resize items-center justify-center focus-visible:outline-none"
            onPointerDown={(event) => {
              event.preventDefault();
              const startX = event.clientX;
              const startWidth = pdfPanelWidth;
              const containerWidth = event.currentTarget?.parentElement?.offsetWidth || window.innerWidth;
              const minWidth = isNarrowDesktop ? 38 : 30;
              const maxWidth = isNarrowDesktop ? 62 : 70;
              const handlePointerMove = (moveEvent) => {
                const deltaX = moveEvent.clientX - startX;
                const deltaPercent = (deltaX / containerWidth) * 100;
                const newWidth = Math.max(minWidth, Math.min(maxWidth, startWidth + deltaPercent));
                setPdfPanelWidth(newWidth);
              };
              const handlePointerUp = () => {
                document.removeEventListener('pointermove', handlePointerMove);
                document.removeEventListener('pointerup', handlePointerUp);
                document.removeEventListener('pointercancel', handlePointerUp);
              };
              document.addEventListener('pointermove', handlePointerMove);
              document.addEventListener('pointerup', handlePointerUp);
                document.addEventListener('pointercancel', handlePointerUp);
            }}
            onDoubleClick={() => setPdfPanelWidth(50)}
            onKeyDown={(event) => {
              if (!['ArrowLeft', 'ArrowRight', 'Home'].includes(event.key)) return;
              event.preventDefault();
              const minWidth = isNarrowDesktop ? 38 : 30;
              const maxWidth = isNarrowDesktop ? 62 : 70;
              const nextWidth = event.key === 'Home'
                ? 50
                : pdfPanelWidth + (event.key === 'ArrowRight' ? 2 : -2);
              setPdfPanelWidth(Math.max(minWidth, Math.min(maxWidth, nextWidth)));
            }}
          >
            <div className={`flex h-12 w-3 items-center justify-center rounded-full border transition-all duration-200 group-hover:scale-105 group-active:scale-110 group-focus-visible:ring-2 group-focus-visible:ring-[#FFA07A]/40 ${
              darkMode
                ? 'border-white/10 bg-[#30333a] text-gray-500 group-hover:border-white/20 group-hover:bg-[#3a3e46] group-hover:text-[#FFA07A]'
                : 'border-white/90 bg-white/90 text-gray-400 shadow-[0_5px_14px_rgba(60,56,52,0.16)] group-hover:text-[#B85F47]'
            }`}>
              <GripVertical className="h-4 w-3" />
            </div>
          </div>

          {/* 右侧：聊天/速览/解析区域 */}
          <motion.div
            initial={false}
            animate={{ opacity: 1, x: 0 }}
            className={`workspace-pane workspace-pane-chat relative flex flex-col overflow-hidden min-w-0 ${darkMode ? 'workspace-pane-dark' : ''}`}
            style={{ width: `calc(${100 - pdfPanelWidth}% - 2rem)`, minWidth: '350px' }}
          >
            {showDocumentWorkspace && <div className="absolute right-5 top-5 z-30">
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation();
                  setShowAiProcessingPanel((prev) => !prev);
                }}
                className={`relative z-50 inline-flex items-center gap-2 rounded-full border px-3 py-2 text-[12px] font-semibold shadow-[0_10px_24px_rgba(148,163,184,0.14),inset_0_1px_0_rgba(255,255,255,0.9)] transition-all hover:-translate-y-0.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/35 ${
                  darkMode
                    ? 'border-white/10 bg-white/[0.06] text-gray-200 hover:bg-white/[0.09]'
                    : 'border-white/70 bg-white/75 text-gray-600 hover:text-[#ed8c68]'
                }`}
                aria-label={`查看后台任务：${backgroundTaskPillLabel}`}
                title={backgroundTaskPillLabel}
              >
                <span className={`flex h-2 w-2 rounded-full ${
                  backgroundTaskSummary.state === 'failed'
                    ? 'bg-[#D97A5D]'
                    : backgroundTaskSummary.state === 'running'
                    ? 'bg-[#ed8c68]'
                    : backgroundTaskSummary.state === 'recommended'
                      ? 'bg-amber-400'
                      : darkMode ? 'bg-gray-600' : 'bg-gray-300'
                }`} />
                {backgroundTaskPillLabel}
              </button>

              <AnimatePresence initial={false}>
                {showAiProcessingPanel && (
                  <BackgroundTaskPanel
                    key="background-task-panel"
                    items={visibleBackgroundTasks}
                    autoEnabled={aiAutoProcess}
                    onAutoEnabledChange={setAiAutoProcess}
                    onClose={closeAiProcessingPanel}
                    onClearCache={handleClearDocumentAICache}
                    canClearCache={Boolean(docId) && !isMinerUFullRoutePending}
                    onRollbackRagIndex={handleRollbackRagIndex}
                    canRollbackRagIndex={canRollbackCurrentRagIndex}
                    darkMode={darkMode}
                  />
                )}
              </AnimatePresence>
              <DocumentParseStatusBar
                documentId={docId}
                manifest={documentParseManifest}
                parseReady={documentParseReady}
                deepParseStatus={currentDeepParseStatus}
                ragIndexStatus={ragIndexStatus}
                darkMode={darkMode}
                onOpenProcessing={openAiProcessingPanel}
                onRetry={retryMinerUFromParseNotice}
                onChooseRoute={openUploadHome}
                suppressed={showAiProcessingPanel}
              />
            </div>}

            {/* 顶部导航：速览 / 阅读 / 对话 */}
              <div className={`pt-6 pb-2 flex shrink-0 ${isNarrowDesktop ? 'justify-start pl-4' : 'justify-center'}`}>
                <div
                  role="group"
                  aria-label="右侧面板视图"
                  className={`relative isolate grid grid-cols-3 rounded-[20px] border p-1 backdrop-blur-xl ${isNarrowDesktop ? 'w-[210px]' : 'w-[240px]'} ${
                    darkMode
                      ? 'border-white/[0.07] bg-white/[0.04] shadow-[inset_0_1px_1px_rgba(255,255,255,0.04)]'
                      : 'border-black/[0.03] bg-[#f5f5f6]/90 shadow-[inset_0_1px_1px_rgba(17,24,39,0.035)]'
                  }`}
                >
                  <motion.div
                    aria-hidden="true"
                    className={`absolute bottom-1 left-1 top-1 z-0 rounded-[16px] will-change-transform ${
                      darkMode
                        ? 'bg-[#383a41] ring-1 ring-inset ring-white/[0.10] shadow-[0_7px_18px_rgba(0,0,0,0.28),0_1px_3px_rgba(0,0,0,0.22),inset_0_1px_0_rgba(255,255,255,0.10)]'
                        : 'bg-white ring-1 ring-inset ring-black/[0.035] shadow-[0_7px_18px_rgba(31,41,55,0.12),0_1px_3px_rgba(31,41,55,0.08),inset_0_1px_0_rgba(255,255,255,0.98)]'
                    }`}
                    initial={false}
                    style={{ width: 'calc((100% - 0.5rem) / 3)' }}
                    animate={{ x: `${activeRightPanelTabIndex * 100}%` }}
                    transition={{ type: 'spring', stiffness: 360, damping: 28, mass: 0.72 }}
                  />

                  {rightPanelTabs.map(({ id, label }) => {
                    const isActive = rightPanelMode === id;
                    return (
                      <motion.button
                        key={id}
                        type="button"
                        aria-pressed={isActive}
                        onClick={() => setRightPanelMode(id)}
                        whileTap={{ scale: 0.97 }}
                        className={`relative z-10 h-9 rounded-[16px] text-[13px] font-semibold transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[#8b95a3]/60 ${
                          isActive
                            ? (darkMode ? 'text-white' : 'text-[#bd6a4e]')
                            : (darkMode ? 'text-gray-400 hover:text-gray-200' : 'text-gray-400 hover:text-gray-600')
                        }`}
                      >
                        {label}
                      </motion.button>
                    );
                  })}
                </div>
              </div>

              {/* 内容区域：根据模式显示速览或对话 */}
            <div className="flex-1 overflow-hidden flex flex-col min-w-0">
              {rightPanelMode === 'overview' ? (
                <Suspense fallback={
                  <div className="flex-1 flex items-center justify-center text-gray-400">
                    <Loader2 className="w-6 h-6 animate-spin mr-2" />
                    加载中...
                  </div>
                }>
                  <OverviewPanel
                    docId={docId}
                    overview={overview}
                    loading={overviewLoading}
                    error={overviewError}
                    depth={overviewDepth}
                    parseRoute={primaryParseRoute}
                    onDepthChange={handleOverviewDepthChange}
                    onFetch={isMinerUFullRoutePending ? undefined : fetchOverview}
                    availabilityState={isMinerUFullRouteFailed
                      ? 'failed'
                      : isMinerUFullRouteCancelled
                        ? 'cancelled'
                        : isMinerUFullRoutePending ? 'processing' : ''}
                    availabilityMessage={isMinerUFullRoutePending ? minerUParsePendingNotice : ''}
                    availabilityActionLabel={isMinerUFullRouteFailed
                      ? (canResumeMinerUResultDownload ? '重试结果下载' : '重试 MinerU')
                      : isMinerUFullRouteCancelled ? '重新上传' : ''}
                    onAvailabilityAction={isMinerUFullRouteFailed
                      ? () => handleStartMinerUDeepParse({ retryFullRoute: true })
                      : isMinerUFullRouteCancelled ? openUploadHome : undefined}
                  />
                </Suspense>
              ) : rightPanelMode === 'analysis' ? (
                <ReadingAnalysisPanel
                  blocks={currentPageBlocks}
                  translations={blockTranslations}
                  translatingBlockIds={translatingBlockIdList}
                  loading={blockTranslateLoading}
                  error={blockTranslateError}
                  notice={pretranslateNotice}
                  pretranslateProgress={pretranslateProgress}
                  currentPage={currentPage}
                  activeBlockId={activeReadingBlockId}
                  notes={currentPageReadingNotes}
                  userNotes={currentPageUserNotes}
                  revealUserNoteId={pendingUserNoteRevealId}
                  onUserNoteReveal={handleUserNoteReveal}
                  activeNodeId={activeReadingNodeId}
                  visitedNodeIds={[...visitedReadingNodeIds]}
                  onTranslate={handleTranslateCurrentPage}
                  onRetranslateBlock={handleRetranslateReadingBlock}
                  onPretranslate={handleStartPretranslate}
                  onBackfillSummaries={blockSummary ? handleBackfillSummaries : undefined}
                  backfillingSummaries={summaryBackfillRunning}
                  onBlockHover={handleReadingBlockHover}
                  onBlockClick={handleReadingBlockClick}
                  onNoteClick={handleOutlineJump}
                  onUserNoteClick={handleUserNoteClick}
                  onSaveUserNote={handleSaveUserNote}
                  onDeleteUserNote={handleDeleteUserNote}
                  documentId={docId}
                  darkMode={darkMode}
                  translationSurface={translationSurface}
                  onTranslationSurfaceChange={setTranslationSurface}
                />
              ) : (
                <>
                  {/* 虚拟消息列表 - 替代原有的 messages.map 渲染（需求 3.1） */}
                  <VirtualMessageList
                    messages={messages}
                    renderMessage={renderMessage}
                    streamingMessageId={streamingMessageId}
                    darkMode={darkMode}
                    className="flex-1 overflow-y-auto overflow-x-hidden p-6 pb-36 min-w-0"
                    itemClassName="pb-8"
                  />

                  {/* 预设问题：空对话时贴近输入框显示，形成「看到建议 → 提问」的动线 */}
                  {showPresetQuestions && (
                    <div className="px-6 pb-[130px]">
                      <div className={`mb-2 text-[11px] font-semibold ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>试试这样问</div>
                      <PresetQuestions onSelect={handlePresetSelect} disabled={isLoading || isChatInteractionLocked} />
                    </div>
                  )}
                </>
              )}
            </div>

            {/* 输入区域：仅在对话模式显示，避免遮挡速览/解析内容 */}
            {rightPanelMode === 'chat' && (
            <div className="p-6 pt-0 bg-transparent relative z-10">
              <div className={`absolute bottom-5 left-3 right-3 z-20 rounded-[28px] p-3 transition-[box-shadow,background-color] duration-200 ${
                darkMode
                  ? 'bg-[#24272d] shadow-[0_18px_46px_-22px_rgba(0,0,0,0.76),0_5px_14px_-9px_rgba(0,0,0,0.50)] focus-within:shadow-[0_22px_52px_-22px_rgba(0,0,0,0.82),0_6px_18px_-10px_rgba(0,0,0,0.58)]'
                  : 'bg-white shadow-[0_18px_44px_-20px_rgba(0,0,0,0.28),0_5px_15px_-9px_rgba(0,0,0,0.14)] focus-within:shadow-[0_22px_52px_-22px_rgba(0,0,0,0.34),0_7px_18px_-10px_rgba(0,0,0,0.18)]'
              }`}>
                {/* 截图预览 - 嵌入输入框顶部，避免被遮挡 */}
                <ScreenshotPreview
                  screenshots={screenshots}
                  onAction={handleParseAwareScreenshotAction}
                  onClose={handleScreenshotClose}
                />
                {/* 第一行：输入文本（参考版式：文本在上，工具在下） */}
                <textarea
                  ref={textareaRef}
                  disabled={isChatInteractionLocked}
                  onChange={(e) => {
                    e.target.style.height = '24px';
                    e.target.style.height = e.target.scrollHeight + 'px';
                    const newHasInput = !!e.target.value.trim();
                    if (newHasInput !== hasInput) setHasInput(newHasInput);
                  }}
                  onKeyDown={(e) => {
                    if (isChatInteractionLocked) return;
                    if (sendShortcut === 'Ctrl+Enter') {
                      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); sendMessage(); }
                    } else {
                      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
                    }
                  }}
                  placeholder={isChatInteractionLocked ? chatInteractionLockedNotice : 'Summarize, rephrase, convert...'}
                  className={`w-full bg-transparent outline-none px-2 pt-1 text-[14px] min-w-0 resize-none h-[24px] overflow-hidden leading-relaxed ${darkMode ? 'text-gray-100 placeholder:text-gray-500' : 'text-gray-800 placeholder:text-gray-400'} ${isChatInteractionLocked ? 'cursor-not-allowed opacity-60' : ''}`}
                  rows={1}
                  style={{ minHeight: '24px', maxHeight: '120px' }}
                />

                {/* 第二行：模型与工具在左，发送在右 */}
                <div className="flex items-center justify-between mt-2 px-1">
                  <div className="flex min-w-0 items-center gap-1.5">
                    <ModelQuickSwitch onThinkingChange={handleThinkingChange} />
                    <ChatContextIndicator
                      messages={messages}
                      contextCount={contextCount}
                      memoryEnabled={enableMemory}
                      lastUsage={lastCallInfo?.usage}
                      darkMode={darkMode}
                    />
                    <div className="relative shrink-0">
                      <button
                        type="button"
                        onClick={toggleCrossDocumentMenu}
                        disabled={!docId || isChatInteractionLocked}
                        aria-label="关联其他文档"
                        aria-expanded={crossDocumentMenuOpen}
                        title={crossDocumentIds.length > 0 ? `已关联 ${crossDocumentIds.length} 篇文档` : '关联其他文档'}
                        className={`relative flex h-7 min-w-7 items-center justify-center rounded-lg px-1.5 transition-colors ${
                          !docId || isChatInteractionLocked
                            ? 'cursor-not-allowed text-gray-300'
                            : crossDocumentIds.length > 0
                              ? darkMode
                                ? 'bg-[#FFA07A]/15 text-[#FFAD8A]'
                                : 'bg-[#FCE9E2] text-[#C95E3B]'
                              : darkMode
                                ? 'text-gray-400 hover:bg-white/10 hover:text-gray-200'
                                : 'text-gray-500 hover:bg-gray-100 hover:text-gray-800'
                        }`}
                      >
                        <Files size={15} />
                        {crossDocumentIds.length > 0 && (
                          <span className={`ml-1 text-[10px] font-semibold leading-none ${darkMode ? 'text-[#FFD3C5]' : 'text-[#B84D2C]'}`}>
                            {crossDocumentIds.length}
                          </span>
                        )}
                      </button>
                      <AnimatePresence>
                        {crossDocumentMenuOpen && (
                          <motion.div
                            initial={{ opacity: 0, y: 5, scale: 0.98 }}
                            animate={{ opacity: 1, y: 0, scale: 1 }}
                            exit={{ opacity: 0, y: 4, scale: 0.98 }}
                            transition={{ duration: 0.14, ease: [0.22, 1, 0.36, 1] }}
                            className={`absolute bottom-9 left-0 z-40 w-[274px] overflow-hidden border p-1.5 shadow-[0_16px_36px_-18px_rgba(30,28,24,0.38)] ${
                              darkMode ? 'border-white/10 bg-[#282b31]' : 'border-[#E9E5DF] bg-white'
                            } rounded-[14px]`}
                          >
                            <div className={`flex items-center justify-between px-2.5 py-1.5 text-[11px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                              <span>关联文档</span>
                              {crossDocumentIds.length > 0 && (
                                <button
                                  type="button"
                                  onClick={() => setCrossDocumentIds([])}
                                  className={`text-[10px] font-medium transition-colors ${darkMode ? 'text-gray-500 hover:text-gray-300' : 'text-gray-400 hover:text-gray-700'}`}
                                >
                                  清除
                                </button>
                              )}
                            </div>
                            <div className="max-h-[224px] space-y-0.5 overflow-y-auto px-0.5 pb-0.5">
                              {crossDocumentOptions.map((candidate) => {
                                const selected = crossDocumentIds.includes(candidate.doc_id);
                                const unavailable = candidate.parse_ready === false;
                                return (
                                  <button
                                    key={candidate.doc_id}
                                    type="button"
                                    disabled={unavailable}
                                    onClick={() => toggleCrossDocument(candidate.doc_id)}
                                    title={unavailable ? '该文档仍在解析，暂不可关联' : candidate.filename}
                                    className={`flex w-full items-center gap-2 rounded-[9px] px-2.5 py-2 text-left transition-colors ${
                                      unavailable
                                        ? 'cursor-not-allowed opacity-45'
                                        : selected
                                          ? darkMode
                                            ? 'bg-[#FFA07A]/12 text-gray-100'
                                            : 'bg-[#FDF0EA] text-gray-800'
                                          : darkMode
                                            ? 'text-gray-300 hover:bg-white/10'
                                            : 'text-gray-600 hover:bg-[#F7F6F3]'
                                    }`}
                                  >
                                    <span className={`flex h-4 w-4 shrink-0 items-center justify-center rounded-[5px] border ${
                                      selected
                                        ? 'border-[#DF6A45] bg-[#DF6A45] text-white'
                                        : darkMode ? 'border-white/20' : 'border-gray-300'
                                    }`}>
                                      {selected && <Check size={11} strokeWidth={3} />}
                                    </span>
                                    <span className="min-w-0 flex-1 truncate text-[12px] font-medium">{candidate.filename}</span>
                                    {candidate.total_pages > 0 && (
                                      <span className={`shrink-0 text-[10px] ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>{candidate.total_pages}页</span>
                                    )}
                                  </button>
                                );
                              })}
                              {!crossDocumentLoading && crossDocumentOptions.length === 0 && (
                                <div className={`px-2.5 py-4 text-center text-[11px] ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>暂无可关联文档</div>
                              )}
                              {crossDocumentLoading && (
                                <div className={`px-2.5 py-4 text-center text-[11px] ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>正在载入文档</div>
                              )}
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                    <div aria-hidden="true" className={`mx-1 h-4 w-px shrink-0 ${darkMode ? 'bg-white/10' : 'bg-gray-200'}`} />
                    <div className={`flex items-center gap-2 shrink-0 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                      <button onClick={openSettings} className={`transition-colors p-1 rounded-md ${darkMode ? 'hover:text-gray-200' : 'hover:text-gray-800'}`} title="设置中心" aria-label="设置中心">
                        <Settings size={15} />
                      </button>
                      <button
                        type="button"
                        onClick={openUploadHome}
                        aria-label="上传 PDF"
                        title="上传新 PDF"
                        className={`transition-colors p-1 rounded-md ${darkMode ? 'hover:text-gray-200' : 'hover:text-gray-800'}`}
                      >
                        <Paperclip size={15} />
                      </button>
                      <WebSearchButton />
                      {isVisionCapable && (
                        <button
                          onClick={() => setIsSelectingArea(true)}
                          disabled={!docId || isChatInteractionLocked}
                          className={`transition-colors p-1 rounded-md ${docId && !isChatInteractionLocked ? isSelectingArea ? 'text-[#B85F47] dark:text-[#FFA07A]' : darkMode ? 'hover:text-gray-200' : 'hover:text-gray-800' : 'text-gray-300 cursor-not-allowed'}`}
                          title={!docId ? '请先上传文档' : isChatInteractionLocked ? chatInteractionLockedNotice : isSelectingArea ? '框选模式已开启' : '区域截图'}
                        >
                          <Scan size={15} />
                        </button>
                      )}
                    </div>
                  </div>

                  <button
                    onClick={isLoading ? handleStop : sendMessage}
                    disabled={isChatInteractionLocked || (!isLoading && (!hasInput && screenshots.length === 0))}
                    aria-label={isLoading ? '停止生成' : '发送'}
                    className={`w-9 h-9 shrink-0 rounded-full transition-all flex items-center justify-center ${
                      !isChatInteractionLocked && (isLoading || hasInput || screenshots.length > 0)
                        ? 'bg-[#F0653A] text-white shadow-[0_6px_16px_-6px_rgba(240,101,58,0.55)] hover:bg-[#D9552B] active:scale-95'
                        : darkMode
                          ? 'border border-white/[0.08] bg-white/[0.06] text-gray-500 cursor-not-allowed'
                          : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                    }`}
                  >
                    <SendPauseIconSwap isPaused={isLoading} />
                  </button>
                </div>
              </div>
            </div>
            )}
          </motion.div>
        </div>
      </div>

      </div>{/* /统一应用外壳 */}

      {/* 设置模态框 */}
      <AnimatePresence initial={false} onExitComplete={handleSettingsModalExitComplete}>
        {showSettings && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            transition={{ duration: 0.1 }}
            className={`fixed inset-0 z-50 flex items-center justify-center bg-slate-950/25 p-4 ${showSettings ? 'pointer-events-auto' : 'pointer-events-none'}`}
            onClick={() => setShowSettings(false)}
          >
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0, transition: { duration: 0.16, ease: [0.22, 1, 0.36, 1] } }}
              exit={{ opacity: 0, y: 6, transition: { duration: 0.09, ease: [0.4, 0, 1, 1] } }}
              onClick={(e) => e.stopPropagation()}
              className={`settings-modal-surface settings-solid settings-shell w-[1040px] max-w-[96vw] h-[min(760px,94vh)] overflow-hidden flex flex-col border ${darkMode ? 'settings-shell-dark bg-[#1d2026] border-[#353941]' : 'bg-[#f6f7f9] border-white/80 relative'}`}
            >
              <div className="flex flex-shrink-0 items-center justify-between px-7 py-5">
                <div className="flex items-center gap-3">
                  <div className={`flex h-10 w-10 items-center justify-center rounded-[13px] border ${darkMode ? 'bg-[#292d35] border-[#3b4049]' : 'bg-white border-gray-200'}`}>
                    <Settings className="text-[#B85F47]" size={22} />
                  </div>
                  <div>
                    <h2 className={`text-[19px] font-bold ${darkMode ? 'text-gray-100' : 'text-gray-800'}`}>设置中心</h2>
                    <p className={`mt-0.5 text-[11px] font-medium ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                      阅读器、模型与文档处理配置
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button onClick={() => setShowSettings(false)} className={`p-2 rounded-full transition-colors z-10 ${darkMode ? 'hover:bg-white/10 text-gray-500 hover:text-gray-300' : 'hover:bg-black/5 text-gray-400 hover:text-gray-700'}`} title="关闭设置中心" aria-label="关闭设置中心">
                    <X className="w-5 h-5" />
                  </button>
                </div>
              </div>

              <div className={`settings-workspace flex min-h-0 flex-1 border-t ${darkMode ? 'border-[#353941]' : 'border-gray-200/80'}`}>
                <aside className={`settings-nav flex w-[218px] shrink-0 flex-col border-r px-3 py-4 ${darkMode ? 'border-[#353941]' : 'border-gray-200/80'}`}>
                  <div className={`mb-2 px-2 text-[10px] font-bold ${darkMode ? 'text-gray-600' : 'text-gray-400'}`}>设置分类</div>
                  <nav className="space-y-1.5" role="tablist" aria-label="设置分类">
                    {SETTINGS_SECTIONS.map(({ id, label, description, Icon }) => {
                      const isActive = settingsSection === id;
                      return (
                        <button
                          key={id}
                          type="button"
                          role="tab"
                          aria-selected={isActive}
                          data-active={isActive}
                          onClick={() => handleSettingsSectionChange(id)}
                          className="settings-nav-item relative isolate grid w-full grid-cols-[30px_minmax(0,1fr)] items-center gap-3 rounded-[18px] px-3 py-2.5 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/30"
                        >
                          {isActive && (
                            <motion.span
                              layoutId="settings-active-section-card"
                              aria-hidden="true"
                              initial={false}
                              className={`pointer-events-none absolute inset-0 z-0 rounded-[18px] will-change-transform ${
                                darkMode
                                  ? 'bg-white/[0.075] shadow-[0_8px_20px_-14px_rgba(0,0,0,0.72),inset_0_1px_0_rgba(255,255,255,0.04)]'
                                  : 'bg-white shadow-[0_8px_20px_-13px_rgba(31,41,55,0.26),0_2px_5px_rgba(31,41,55,0.045)]'
                              }`}
                              transition={{ type: 'tween', duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
                            />
                          )}
                          <span className="settings-nav-item-icon relative z-10 flex h-[30px] w-[30px] items-center justify-center">
                            <Icon size={17} strokeWidth={2.2} />
                          </span>
                          <span className="relative z-10 min-w-0">
                            <span className="block text-[12px] font-semibold leading-4">{label}</span>
                            <span className="mt-0.5 block truncate text-[10px] leading-4">{description}</span>
                          </span>
                        </button>
                      );
                    })}
                  </nav>
                  <div className="mt-auto px-2 pb-1 pt-5">
                    <div className={`text-[10px] font-medium ${darkMode ? 'text-gray-600' : 'text-gray-400'}`}>当前配置</div>
                    <div className={`mt-1 truncate text-[11px] font-semibold ${darkMode ? 'text-gray-400' : 'text-gray-600'}`} title={defaultModelOverview.assistant.modelName}>
                      {defaultModelOverview.assistant.modelName}
                    </div>
                  </div>
                </aside>

                <div className="flex min-w-0 flex-1 flex-col">
                  <div className={`settings-content-heading flex flex-shrink-0 items-center justify-between gap-4 border-b px-6 py-4 ${darkMode ? 'border-[#353941]' : 'border-gray-200/75'}`}>
                    <div className="flex min-w-0 items-center gap-3">
                      <div className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-[11px] ${darkMode ? 'bg-white/[0.06] text-gray-400' : 'bg-white text-[#B85F47]'}`}>
                        <ActiveSettingsSectionIcon size={17} strokeWidth={2.1} />
                      </div>
                      <div className="min-w-0">
                        <h3 className={`text-[15px] font-semibold ${darkMode ? 'text-gray-100' : 'text-gray-900'}`}>{activeSettingsSectionMeta.label}</h3>
                        <p className={`mt-0.5 text-[11px] ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>{activeSettingsSectionMeta.description}</p>
                      </div>
                    </div>
                    {settingsSection === 'retrieval' ? (
                      <span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold tabular-nums ${darkMode ? 'bg-white/[0.06] text-gray-400' : 'bg-white text-gray-500'}`}>
                        基础检索 {baseRetrievalEnabledCount}/2
                      </span>
                    ) : null}
                  </div>

                  <motion.div
                    key={settingsSection}
                    initial={{ opacity: 0, x: 8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
                    className="custom-scrollbar flex-1 space-y-[18px] overflow-y-auto px-6 pb-6 pt-5"
                  >
                
                {settingsSection === 'common' && (
                <section className="px-1" aria-labelledby="current-models-heading">
                  <div className="mb-2.5 flex items-end justify-between gap-4 px-1">
                    <div>
                      <h3 id="current-models-heading" className={`text-[13px] font-bold ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>当前工作配置</h3>
                      <p className={`mt-0.5 text-[11px] ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>对话、嵌入和重排各用各的模型</p>
                    </div>
                    <button
                      type="button"
                      onClick={handleOpenEmbeddingSettings}
                      onPointerEnter={preloadEmbeddingSettings}
                      onFocus={preloadEmbeddingSettings}
                      title="配置服务商和默认模型"
                      aria-label="打开模型服务管理"
                      className="model-service-cta"
                    >
                      <span className="model-service-cta__fill" aria-hidden="true" />
                      <span className="model-service-cta__icon" aria-hidden="true">
                        <ArrowRight size={15} strokeWidth={2.3} />
                      </span>
                      <span className="model-service-cta__label">模型服务</span>
                    </button>
                  </div>

                  <div className={`settings-card grid min-h-[180px] grid-cols-[1.08fr_1fr] overflow-hidden ${darkMode ? 'settings-card-dark bg-[#24272e] border-[#373b44]' : 'bg-white border-gray-200/90'}`}>
                    <div className="flex min-w-0 flex-col p-5">
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex items-center gap-3">
                          <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-[13px] ${darkMode ? 'bg-white/[0.07] text-gray-300' : 'bg-[#fff2ec] text-[#B85F47]'}`}>
                            <MessageSquare size={18} strokeWidth={2.1} />
                          </div>
                          <div>
                            <div className={`text-[13px] font-bold ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>对话模型</div>
                            <div className={`mt-0.5 text-[10px] font-medium ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>问答、总结与内容生成</div>
                          </div>
                        </div>
                        <span className="settings-config-status" data-state={defaultModelOverview.assistant.state}>{defaultModelOverview.assistant.statusLabel}</span>
                      </div>

                      <div className="mt-5 min-w-0">
                        <div className={`text-[10px] font-semibold ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>{defaultModelOverview.assistant.providerName}</div>
                        <div className={`mt-1 truncate text-[15px] font-semibold ${darkMode ? 'text-gray-100' : 'text-gray-900'}`} title={defaultModelOverview.assistant.modelName}>
                          {defaultModelOverview.assistant.modelName}
                        </div>
                      </div>

                      <div className="mt-auto flex flex-wrap gap-1.5 pt-4">
                        <span className={`rounded-full px-2 py-1 text-[10px] font-semibold ${darkMode ? 'bg-white/[0.06] text-gray-400' : 'bg-gray-100 text-gray-500'}`}>{streamOutput ? '流式输出' : '整段输出'}</span>
                        <span className={`rounded-full px-2 py-1 text-[10px] font-semibold ${darkMode ? 'bg-white/[0.06] text-gray-400' : 'bg-gray-100 text-gray-500'}`}>{enableMemory ? '记忆已开启' : '记忆已关闭'}</span>
                      </div>
                    </div>

                    <div className={`grid min-w-0 grid-rows-2 border-l ${darkMode ? 'border-[#373b44]' : 'border-gray-100'}`}>
                      {[
                        { Icon: Database, label: '嵌入模型', purpose: '建立语义索引', model: defaultModelOverview.embedding },
                        { Icon: ArrowUpDown, label: '重排模型', purpose: '筛选高相关证据', model: defaultModelOverview.rerank },
                      ].map(({ Icon, label, purpose, model: modelInfo }, index) => (
                        <div key={label} className={`flex min-w-0 items-center gap-3 px-4 py-3.5 ${index > 0 ? (darkMode ? 'border-t border-[#373b44]' : 'border-t border-gray-100') : ''}`}>
                          <div className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-[12px] ${darkMode ? 'bg-white/[0.06] text-gray-400' : 'bg-gray-100/90 text-gray-500'}`}>
                            <Icon size={16} strokeWidth={2.1} />
                          </div>
                          <div className="min-w-0 flex-1">
                            <div className="flex items-center justify-between gap-2">
                              <span className={`text-[12px] font-bold ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>{label}</span>
                              <span className="settings-config-status" data-state={modelInfo.state}>{modelInfo.statusLabel}</span>
                            </div>
                            <div className={`mt-1 truncate text-[12px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`} title={modelInfo.modelName}>{modelInfo.modelName}</div>
                            <div className={`mt-0.5 truncate text-[10px] ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>{modelInfo.providerName} · {purpose}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </section>
                )}

                {/* 智能阅读 — 从「全局设置 > 阅读」上移为一级分区：
                    大纲/总结/预翻译/速览是产品核心能力，不应藏在三级深度 */}
                {settingsSection === 'reading' && (
                <div className={`settings-card p-5 border space-y-3 mt-2 mx-1 ${darkMode ? 'settings-card-dark bg-[#24272e] border-[#373b44]' : 'bg-white border-gray-200/90'}`}>
                  <SettingsCheckRow
                    title="智能阅读"
                    description="关闭后打开文档不调用模型"
                    checked={aiAutoProcess}
                    onChange={setAiAutoProcess}
                    darkMode={darkMode}
                  />

                  <div className={`space-y-3 pt-1 ${aiAutoProcess ? '' : 'opacity-50 pointer-events-none'}`}>
                    <SettingsCheckRow
                      title="大纲与总结"
                      description="打开后生成左侧总结和大纲"
                      checked={autoOutlineSummary}
                      onChange={setAutoOutlineSummary}
                      darkMode={darkMode}
                    />
                    <SettingsCheckRow
                      title="预翻译全文"
                      description="提前译好，悬浮即可查看"
                      hint="较耗额度"
                      checked={enableHoverPretranslate}
                      onChange={setAutoPretranslate}
                      darkMode={darkMode}
                    />
                    <SettingsCheckRow
                      title="逐段要点"
                      description="正文每段多一句要点"
                      hint="每段多一次"
                      checked={blockSummary}
                      onChange={setBlockSummary}
                      darkMode={darkMode}
                    />

                    <div className="settings-inset p-3.5 rounded-[14px]">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <div className={`text-[12px] font-bold ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>预翻译并发</div>
                          <div className={`text-[11px] mt-0.5 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>限速模型用 3–6，高速模型可更高</div>
                        </div>
                        <span className="text-[12px] font-bold text-[#B85F47] tabular-nums">{pretranslateConcurrency}</span>
                      </div>
                      <SettingsRange
                        ariaLabel="预翻译并发"
                        min="1"
                        max="16"
                        step="1"
                        value={pretranslateConcurrency}
                        onChange={setPretranslateConcurrency}
                        className="mt-3"
                      />
                    </div>

                    <div className="settings-inset p-3.5 rounded-[14px]">
                      <div className={`text-[12px] font-bold ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>速览默认详细度</div>
                      <SettingsSegmentedControl
                        ariaLabel="速览默认详细度"
                        value={overviewDefaultDepth}
                        onChange={setOverviewDefaultDepth}
                        options={[
                          { value: 'brief', label: '简略' },
                          { value: 'standard', label: '标准' },
                          { value: 'detailed', label: '详细' },
                        ]}
                        className="mt-2 rounded-[12px]"
                        buttonClassName="py-1.5 text-[11px] font-bold text-center rounded-[9px]"
                        indicatorClassName="rounded-[9px]"
                      />
                      <p className={`mt-2 text-[11px] ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>只改速览深度，不会自动开始生成</p>
                    </div>

                    <div className="settings-inset p-3.5 rounded-[14px]">
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <div className={`text-[12px] font-bold ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>图表理解模型</div>
                          <p className={`mt-0.5 text-[11px] leading-snug ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                            解读图表、核验表格
                          </p>
                        </div>
                        <span className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-semibold ${visualPolicyReady ? 'bg-emerald-500/10 text-emerald-600' : 'bg-[#FBE9E2] text-[#B85F47]'}`}>
                          {visualStrategy === 'privacy'
                            ? (hasLocalVisualModel ? '仅本地' : '需本地模型')
                            : (visualPolicyReady ? '可用' : '需选择')}
                        </span>
                      </div>
                      <SettingsSegmentedControl
                        ariaLabel="视觉增强策略"
                        value={visualStrategy}
                        onChange={setVisualStrategy}
                        options={[
                          { value: 'privacy', label: '隐私优先' },
                          { value: 'balanced', label: '平衡' },
                          { value: 'quality', label: '质量优先' },
                        ]}
                        className="mt-2.5 rounded-[12px]"
                        buttonClassName="py-1.5 text-[11px] font-bold text-center rounded-[9px]"
                        indicatorClassName="rounded-[9px]"
                      />
                      <div className="mt-3">
                        <div className={`mb-1.5 text-[10px] font-semibold ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>强视觉模型</div>
                        <CustomSelect
                          value={visualModelKey}
                          onChange={setVisualModelKey}
                          options={visualModelOptions}
                          unavailableLabel={visualModelKey && visualModelKey !== 'follow_chat'
                            ? `已保存：${visualModelKey.replace(':', ' · ')}（当前不可用）`
                            : undefined}
                        />
                      </div>
                      <div className="mt-2.5">
                        <div className={`mb-1.5 flex items-center justify-between text-[10px] font-semibold ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                          <span>本地视觉模型</span>
                          {visualStrategy === 'privacy' && <span className="text-[#B85F47]">仅本地</span>}
                        </div>
                        <CustomSelect
                          value={localVisualModelKey}
                          onChange={setLocalVisualModelKey}
                          options={localVisualModelOptions}
                          unavailableLabel={localVisualModelKey && localVisualModelKey !== 'none'
                            ? `已保存：${localVisualModelKey.replace(':', ' · ')}（当前不可用）`
                            : undefined}
                        />
                      </div>
                      <div className={`mt-3 rounded-[11px] border px-3 py-2.5 ${darkMode ? 'border-white/10 bg-black/15' : 'border-[#eadfd9] bg-[#fffaf7]'}`}>
                        <div className={`text-[10px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>表格核验</div>
                        <div className={`mt-1 text-[11px] leading-snug ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                          表格截图走上方视觉模型，不走检索辅助模型
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                )}

                {settingsSection === 'interface' && (
                <section className="px-1" aria-labelledby="display-font-size-heading">
                  <div className="mb-2.5 px-1">
                    <h3 id="display-font-size-heading" className={`text-[13px] font-bold ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>显示与字号</h3>
                    <p className={`mt-0.5 text-[11px] ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>内容和界面分开调</p>
                  </div>
                  <div className={`settings-card divide-y overflow-hidden ${darkMode ? 'settings-card-dark divide-[#373b44] bg-[#24272e] border-[#373b44]' : 'divide-gray-100 bg-white border-gray-200/90'}`}>
                    <div className="grid grid-cols-[1fr_300px] items-center gap-5 px-5 py-[18px]">
                      <div className="min-w-0">
                        <div className={`text-[13px] font-semibold ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>内容字号</div>
                        <p className={`mt-1 text-[11px] leading-snug ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>回答、总结、大纲、翻译等正文</p>
                      </div>
                      <SettingsSegmentedControl
                        ariaLabel="内容字号"
                        value={messageFontSize <= 13 ? 13 : messageFontSize >= 18 ? 18 : messageFontSize >= 16 ? 16 : 14}
                        onChange={setMessageFontSize}
                        options={[
                          { value: 13, label: '小' },
                          { value: 14, label: '标准' },
                          { value: 16, label: '大' },
                          { value: 18, label: '特大' },
                        ]}
                        className="rounded-[12px]"
                        buttonClassName="py-1.5 text-[11px] font-bold text-center rounded-[9px]"
                        indicatorClassName="rounded-[9px]"
                      />
                    </div>
                    <div className="grid grid-cols-[1fr_300px] items-center gap-5 px-5 py-[18px]">
                      <div className="min-w-0">
                        <div className={`text-[13px] font-semibold ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>界面字号</div>
                        <p className={`mt-1 text-[11px] leading-snug ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>侧栏、按钮和设置</p>
                      </div>
                      <SettingsSegmentedControl
                        ariaLabel="界面字号"
                        value={globalScale < 0.95 ? 0.9 : globalScale > 1.05 ? 1.1 : 1}
                        onChange={setGlobalScale}
                        options={[
                          { value: 0.9, label: '紧凑' },
                          { value: 1, label: '标准' },
                          { value: 1.1, label: '放大' },
                        ]}
                        className="rounded-[12px]"
                        buttonClassName="py-1.5 text-[11px] font-bold text-center rounded-[9px]"
                        indicatorClassName="rounded-[9px]"
                      />
                    </div>
                  </div>
                </section>
                )}

                {settingsSection === 'retrieval' && (
                <>
                  <section className="px-1" aria-label="检索状态总览">
                    <div className={`settings-status-strip grid grid-cols-3 overflow-hidden rounded-[14px] border ${darkMode ? 'border-[#373b44]' : 'border-gray-200/90'}`}>
                      <div className="settings-status-cell">
                        <div className={`flex h-8 w-8 items-center justify-center rounded-[10px] ${darkMode ? 'bg-white/[0.055] text-gray-400' : 'bg-white text-[#B85F47]'}`}>
                          <ListFilter size={15} strokeWidth={2.1} />
                        </div>
                        <div className="min-w-0">
                          <div className={`text-[10px] font-medium ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>基础召回</div>
                          <div className={`mt-0.5 text-[12px] font-semibold tabular-nums ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>{baseRetrievalEnabledCount}/2 已启用</div>
                        </div>
                      </div>
                      <div className="settings-status-cell">
                        <div className={`flex h-8 w-8 items-center justify-center rounded-[10px] ${darkMode ? 'bg-white/[0.055] text-gray-400' : 'bg-white text-[#B85F47]'}`}>
                          <Brain size={15} strokeWidth={2.1} />
                        </div>
                        <div className="min-w-0">
                          <div className={`text-[10px] font-medium ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>知识图谱</div>
                          <div className={`mt-0.5 truncate text-[12px] font-semibold ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>{graphRagStatusLabel}</div>
                        </div>
                      </div>
                      <div className="settings-status-cell">
                        <div className={`flex h-8 w-8 items-center justify-center rounded-[10px] ${darkMode ? 'bg-white/[0.055] text-gray-400' : 'bg-white text-[#B85F47]'}`}>
                          <Sparkles size={15} strokeWidth={2.1} />
                        </div>
                        <div className="min-w-0">
                          <div className={`text-[10px] font-medium ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>Agent 模式</div>
                          <div className={`mt-0.5 truncate text-[12px] font-semibold ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>{agentModeLabel}</div>
                        </div>
                      </div>
                    </div>
                  </section>

                  <section className="px-1" aria-labelledby="retrieval-foundation-heading">
                    <div className="mb-2.5 px-1">
                      <h3 id="retrieval-foundation-heading" className={`text-[13px] font-bold ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>基础召回</h3>
                      <p className={`mt-0.5 text-[11px] ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>语义检索和关键词一起用</p>
                    </div>
                    <div className={`settings-card divide-y overflow-hidden ${darkMode ? 'settings-card-dark divide-[#373b44] bg-[#24272e] border-[#373b44]' : 'divide-gray-100 bg-white border-gray-200/90'}`}>
                      <SettingsFeatureRow
                        Icon={ListFilter}
                        title="向量检索"
                        description="按语义找相近段落"
                        checked={enableVectorSearch}
                        onChange={setEnableVectorSearch}
                        statusLabel={enableVectorSearch ? '工作中' : '已关闭'}
                        statusTone={enableVectorSearch ? 'ready' : 'muted'}
                        darkMode={darkMode}
                      />
                      <SettingsFeatureRow
                        Icon={Type}
                        title="jieba 中文分词"
                        description="提高中文关键词命中"
                        checked={enableJiebaBM25}
                        onChange={setEnableJiebaBM25}
                        statusLabel={enableJiebaBM25 ? '工作中' : '已关闭'}
                        statusTone={enableJiebaBM25 ? 'ready' : 'muted'}
                        darkMode={darkMode}
                      />
                    </div>
                  </section>

                  <section className="px-1" aria-labelledby="retrieval-structure-heading">
                    <div className="mb-2.5 px-1">
                      <h3 id="retrieval-structure-heading" className={`text-[13px] font-bold ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>结构增强</h3>
                      <p className={`mt-0.5 text-[11px] ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>复杂问题用图谱或多轮检索</p>
                    </div>
                    <div className={`settings-card overflow-hidden ${darkMode ? 'settings-card-dark bg-[#24272e] border-[#373b44]' : 'bg-white border-gray-200/90'}`}>
                      <SettingsFeatureRow
                        Icon={Brain}
                        title="GraphRAG 知识图谱"
                        description="从文档抽出实体和关系"
                        checked={enableGraphRAG}
                        onChange={setEnableGraphRAG}
                        statusLabel={graphRagStatusLabel}
                        statusTone={graphragStatus === 'built' ? 'ready' : graphragStatus === 'error' ? 'warning' : enableGraphRAG ? 'accent' : 'muted'}
                        darkMode={darkMode}
                      />

                      {enableGraphRAG && docId && (
                        <div className={`settings-subpanel mx-5 mb-4 px-4 py-3 ${darkMode ? 'bg-[#20242a]' : 'bg-[#faf8f6]'}`}>
                          <div className="flex items-center justify-between gap-4">
                            <div className="min-w-0 flex-1 text-[11px] leading-[1.55]">
                              {isMinerUFullRoutePending ? (
                                <span className={darkMode ? 'text-gray-400' : 'text-gray-600'}>{minerUParsePendingNotice}</span>
                              ) : graphragStatus === 'built' && graphragStats ? (
                                <span className={darkMode ? 'text-gray-300' : 'text-gray-700'}>
                                  <b className="font-semibold tabular-nums">{graphragStats.num_nodes ?? 0}</b> 实体 · <b className="font-semibold tabular-nums">{graphragStats.num_edges ?? 0}</b> 关系 · <b className="font-semibold tabular-nums">{graphragStats.num_chunks ?? 0}</b> 分块
                                </span>
                              ) : graphragStatus === 'building' ? (
                                <span className={`inline-flex items-center gap-1.5 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                                  <Loader2 size={12} className="shrink-0 animate-spin" />
                                  {graphRagStatusLabel}
                                </span>
                              ) : graphragStatus === 'error' ? (
                                <span className="text-red-500">{graphragError || '构建失败，请重试'}</span>
                              ) : (
                                <span className={darkMode ? 'text-gray-500' : 'text-gray-500'}>尚未构建，大约要几十秒到几分钟</span>
                              )}
                            </div>
                            <button
                              type="button"
                              onClick={handleBuildGraphRAG}
                              disabled={graphragStatus === 'building' || isMinerUFullRoutePending}
                              className={`shrink-0 rounded-[10px] px-3 py-1.5 text-[11px] font-semibold transition-all active:translate-y-px disabled:cursor-not-allowed disabled:opacity-45 ${darkMode ? 'bg-white/10 text-gray-200 hover:bg-white/15' : 'bg-white text-[#A65B45] shadow-sm ring-1 ring-[#ead7cf] hover:bg-[#fff7f3]'}`}
                            >
                              {isMinerUFullRoutePending ? '等待解析' : graphragStatus === 'built' ? '重新构建' : graphragStatus === 'building' ? '构建中' : '立即构建'}
                            </button>
                          </div>
                        </div>
                      )}

                      <div className={`border-t ${darkMode ? 'border-[#373b44]' : 'border-gray-100'}`}>
                        <SettingsFeatureRow
                          Icon={Sparkles}
                          title="检索代理 (Agentic RAG)"
                          description="复杂问题自动组合多种检索"
                          checked={enableAgentRetrieval}
                          onChange={setEnableAgentRetrieval}
                          statusLabel={agentModeLabel}
                          statusTone={enableAgentRetrieval ? 'accent' : 'muted'}
                          darkMode={darkMode}
                        />
                      </div>

                      {enableAgentRetrieval && (
                        <div className={`settings-subpanel mx-5 mb-4 px-4 py-3 ${darkMode ? 'bg-[#20242a]' : 'bg-[#faf8f6]'}`}>
                          <div className="grid grid-cols-3 gap-3">
                            {[
                              ['触发策略', forceAgentRetrieval ? '全部问题' : '按需判断'],
                              ['检索工具', '7 种'],
                              ['默认上限', '5 轮'],
                            ].map(([label, value]) => (
                              <div key={label}>
                                <div className={`text-[9px] font-medium ${darkMode ? 'text-gray-600' : 'text-gray-400'}`}>{label}</div>
                                <div className={`mt-0.5 text-[11px] font-semibold tabular-nums ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>{value}</div>
                              </div>
                            ))}
                          </div>
                          <div className={`mt-3 flex items-center justify-between border-t pt-3 ${darkMode ? 'border-white/[0.07]' : 'border-[#ebe4df]'}`}>
                            <div className="min-w-0 pr-4">
                              <div className={`text-[11px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>全部问题使用 Agent</div>
                              <div className={`mt-0.5 text-[10px] ${darkMode ? 'text-gray-600' : 'text-gray-400'}`}>日常阅读建议保持按需</div>
                            </div>
                            <SettingsSwitch checked={forceAgentRetrieval} onChange={setForceAgentRetrieval} label="全部问题使用 Agent" darkMode={darkMode} />
                          </div>
                        </div>
                      )}
                    </div>
                  </section>

                  <section className="px-1" aria-labelledby="retrieval-tuning-heading">
                    <div className={`settings-card overflow-hidden ${darkMode ? 'settings-card-dark bg-[#24272e] border-[#373b44]' : 'bg-white border-gray-200/90'}`}>
                      <button
                        type="button"
                        aria-expanded={showRetrievalTuning}
                        onClick={() => setShowRetrievalTuning(!showRetrievalTuning)}
                        className="settings-entry-row flex w-full items-center justify-between gap-4 px-5 py-4 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[#D97A5D]/25"
                      >
                        <div className="flex min-w-0 items-center gap-3.5">
                          <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-[12px] ${darkMode ? 'bg-white/[0.055] text-gray-400' : 'bg-gray-100 text-gray-500'}`}>
                            <SlidersHorizontal size={17} strokeWidth={2.1} />
                          </div>
                          <div className="min-w-0">
                            <div className="flex flex-wrap items-center gap-2">
                              <h3 id="retrieval-tuning-heading" className={`text-[13px] font-semibold ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>会话级检索调优</h3>
                              <span className="settings-feature-status" data-tone="muted">高级</span>
                            </div>
                            <p className={`mt-1 text-[11px] ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>只改当前会话</p>
                          </div>
                        </div>
                        <ChevronDown className={`h-4 w-4 shrink-0 text-gray-400 transition-transform duration-200 ${showRetrievalTuning ? 'rotate-180' : ''}`} />
                      </button>

                      <AnimatePresence initial={false}>
                        {showRetrievalTuning && (
                          <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
                            className="overflow-hidden"
                          >
                            <div className={`border-t px-5 pb-5 pt-3 ${darkMode ? 'border-[#373b44]' : 'border-gray-100'}`}>
                              <TriStateToggle
                                title="表格数值增强"
                                desc="比较表格数字时加强检索"
                                value={overrideNumericTable}
                                onChange={setOverrideNumericTable}
                                darkMode={darkMode}
                              />
                              <TriStateToggle
                                title="同义词扩展"
                                desc="换种说法也能搜到"
                                value={overrideBM25Synonyms}
                                onChange={setOverrideBM25Synonyms}
                                darkMode={darkMode}
                              />
                              <TriStateToggle
                                title="查询改写"
                                desc="补全指代和省略"
                                value={overrideLLMQueryRewrite}
                                onChange={setOverrideLLMQueryRewrite}
                                darkMode={darkMode}
                              />
                              <TriStateToggle
                                title="答案自审"
                                desc="检查回答是否对得上证据"
                                value={overrideAnswerCritic}
                                onChange={setOverrideAnswerCritic}
                                darkMode={darkMode}
                              />
                              <VisualVerificationMode
                                value={numericTableVisualVerification}
                                onChange={setNumericTableVisualVerification}
                                darkMode={darkMode}
                                visualModelSummary={visualModelSummary}
                              />

                              <div className={`settings-subpanel mt-3 px-4 py-3 ${darkMode ? 'bg-[#20242a]' : 'bg-[#faf8f6]'}`}>
                                <div className="flex items-center justify-between gap-3">
                                  <div>
                                    <div className={`text-[11px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>检索辅助模型</div>
                                    <div className={`mt-0.5 text-[10px] leading-snug ${darkMode ? 'text-gray-600' : 'text-gray-400'}`}>用于改写、命名和自审，不看图片</div>
                                  </div>
                                  <span className={`shrink-0 text-[10px] ${darkMode ? 'text-gray-600' : 'text-gray-400'}`}>可选</span>
                                </div>
                                <div className="mt-3">
                                  <CustomSelect
                                    value={cheapModelKey}
                                    onChange={handleCheapModelChange}
                                    options={cheapModelOptions}
                                    unavailableLabel={cheapModelUnavailableLabel}
                                  />
                                  <p className={`mt-1.5 text-[10px] leading-snug ${darkMode ? 'text-gray-600' : 'text-gray-400'}`}>
                                    留空则跟随后端。只列出当前服务商的模型。
                                  </p>
                                </div>
                              </div>
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  </section>
                </>
                )}

                {settingsSection === 'interface' && (
                <section className="px-1" aria-labelledby="reading-toolbar-heading">
                  <div className="mb-2.5 px-1">
                    <h3 id="reading-toolbar-heading" className={`text-[13px] font-bold ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>阅读工具栏</h3>
                    <p className={`mt-0.5 text-[11px] ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>截图、搜索和工具尺寸</p>
                  </div>
                  <div className={`settings-card overflow-hidden ${darkMode ? 'settings-card-dark bg-[#24272e] border-[#373b44]' : 'bg-white border-gray-200/90'}`}>
                    <SettingsFeatureRow
                      Icon={Scan}
                      title="区域截图"
                      description="框选页面区域，随问题一起发给模型"
                      checked={enableScreenshot}
                      onChange={setEnableScreenshot}
                      statusLabel={enableScreenshot ? '已启用' : '已关闭'}
                      statusTone={enableScreenshot ? 'ready' : 'muted'}
                      darkMode={darkMode}
                    />
                    <div className={`space-y-3.5 px-5 pt-4 pb-5 ${darkMode ? 'border-t border-[#373b44]' : 'border-t border-gray-100'}`}>
                      <div className="flex flex-col gap-1.5">
                        <label className={`text-[12px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>默认搜索引擎</label>
                        <CustomSelect
                          value={searchEngine}
                          onChange={setSearchEngine}
                          options={[
                            { value: 'google', label: 'Google' },
                            { value: 'bing', label: 'Bing' },
                            { value: 'baidu', label: '百度' },
                            { value: 'sogou', label: '搜狗' },
                            { value: 'custom', label: '自定义' }
                          ]}
                        />
                        {searchEngine === 'custom' && (
                          <div className="mt-1">
                            <input type="text" value={searchEngineUrl} onChange={(e) => setSearchEngineUrl(e.target.value)} className={`w-full p-2.5 rounded-[12px] border text-sm outline-none transition-colors ${darkMode ? 'bg-[#1d2026] border-[#353941] text-white focus:border-[#FFA07A]/50' : 'bg-white border-gray-200 focus:border-[#FFA07A]/50'}`} placeholder="例如：https://www.google.com/search?q={query}" />
                            <p className="text-[11px] text-gray-500 mt-1">使用 <code className="font-mono bg-black/5 px-1 rounded">{'<query>'}</code> 作为搜索词占位符</p>
                          </div>
                        )}
                      </div>
                      <div className="flex flex-col gap-1.5">
                        <label className={`text-[12px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>工具栏尺寸</label>
                        <CustomSelect
                          value={toolbarSize}
                          onChange={setToolbarSize}
                          options={[
                            { value: 'compact', label: '紧凑' },
                            { value: 'normal', label: '常规' },
                            { value: 'large', label: '大号' }
                          ]}
                        />
                      </div>
                    </div>
                  </div>
                </section>
                )}

                {settingsSection === 'interface' && (
                <section className="px-1" aria-labelledby="answer-presentation-heading">
                  <div className="mb-2.5 px-1">
                    <h3 id="answer-presentation-heading" className={`text-[13px] font-bold ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>回答呈现</h3>
                    <p className={`mt-0.5 text-[11px] ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>出字速度和渐显效果</p>
                  </div>
                  <div className={`settings-card overflow-hidden ${darkMode ? 'settings-card-dark bg-[#24272e] border-[#373b44]' : 'bg-white border-gray-200/90'}`}>
                    <SettingsFeatureRow
                      Icon={Sparkles}
                      title="模糊渐显"
                      description="新字先模糊再变清晰"
                      checked={enableBlurReveal}
                      onChange={setEnableBlurReveal}
                      statusLabel={enableBlurReveal ? '已启用' : '已关闭'}
                      statusTone={enableBlurReveal ? 'accent' : 'muted'}
                      darkMode={darkMode}
                    />
                    <div className={`space-y-3.5 px-5 pt-4 pb-5 ${darkMode ? 'border-t border-[#373b44]' : 'border-t border-gray-100'}`}>
                      <div className="flex flex-col gap-1.5">
                        <label className={`text-[12px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>流式输出速度</label>
                        <CustomSelect
                          value={streamSpeed}
                          onChange={setStreamSpeed}
                          options={[
                            { value: 'fast', label: '即时' },
                            { value: 'normal', label: '平滑（推荐）' },
                            { value: 'slow', label: '逐字' },
                            { value: 'off', label: '关闭流式' }
                          ]}
                        />
                        <p className="text-[11px] text-gray-500">流结束后会稍微加快收尾</p>
                      </div>
                      {enableBlurReveal && (
                        <div className="flex flex-col gap-1.5">
                          <label className={`text-[12px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>模糊强度</label>
                          <CustomSelect
                            value={blurIntensity}
                            onChange={setBlurIntensity}
                            options={[
                              { value: 'light', label: '轻度' },
                              { value: 'medium', label: '中度' },
                              { value: 'strong', label: '明显' }
                            ]}
                          />
                        </div>
                      )}
                    </div>
                  </div>
                </section>
                )}

                {settingsSection === 'storage' && (
                <div className={`settings-card p-5 border space-y-4 mx-1 ${darkMode ? 'settings-card-dark bg-[#24272e] border-[#373b44]' : 'bg-white border-gray-200/90'}`}>
                  <div>
                    <h3 className={`text-[13px] font-bold tracking-wider uppercase mb-3 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>存储信息</h3>
                    {storageInfo ? (
                      <div className="space-y-2">
                        <div className={`p-3 rounded-[14px] border transition-colors ${darkMode ? 'bg-[#1d2026] border-[#353941] hover:bg-[#20242b]' : 'bg-gray-50 border-gray-200 hover:bg-gray-100'}`}>
                          <div className="flex items-center justify-between mb-1.5">
                            <span className={`text-[12px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>PDF文件 ({storageInfo.pdf_count})</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <div className={`flex-1 text-[11px] px-2.5 py-1.5 rounded-[8px] overflow-x-auto whitespace-nowrap font-mono border ${darkMode ? 'bg-black/40 border-white/5 text-gray-400' : 'bg-white border-gray-100 text-gray-500'}`}>
                              {storageInfo.uploads_dir}
                            </div>
                            <button onClick={() => { navigator.clipboard.writeText(storageInfo.uploads_dir); alert('路径已复制到剪贴板！'); }} className="p-1.5 rounded-[8px] bg-[#FFA07A]/10 text-[#B85F47] hover:bg-[#FFA07A]/20 transition-colors shrink-0" title="复制路径">
                              <Copy size={14} />
                            </button>
                          </div>
                        </div>
                        <div className={`p-3 rounded-[14px] border transition-colors ${darkMode ? 'bg-[#1d2026] border-[#353941] hover:bg-[#20242b]' : 'bg-gray-50 border-gray-200 hover:bg-gray-100'}`}>
                          <div className="flex items-center justify-between mb-1.5">
                            <span className={`text-[12px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>对话历史 ({storageInfo.doc_count})</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <div className={`flex-1 text-[11px] px-2.5 py-1.5 rounded-[8px] overflow-x-auto whitespace-nowrap font-mono border ${darkMode ? 'bg-black/40 border-white/5 text-gray-400' : 'bg-white border-gray-100 text-gray-500'}`}>
                              {storageInfo.data_dir}
                            </div>
                            <button onClick={() => { navigator.clipboard.writeText(storageInfo.data_dir); alert('路径已复制到剪贴板！'); }} className="p-1.5 rounded-[8px] bg-[#FFA07A]/10 text-[#B85F47] hover:bg-[#FFA07A]/20 transition-colors shrink-0" title="复制路径">
                              <Copy size={14} />
                            </button>
                          </div>
                        </div>
                        <p className="text-[11px] text-gray-500 mt-2 px-1">
                          在 {storageInfo.platform === 'Windows' ? '文件资源管理器' : storageInfo.platform === 'Darwin' ? 'Finder' : '文件管理器'} 中打开以管理文件
                        </p>
                        <div className={`mt-3 rounded-[14px] border p-3 ${darkMode ? 'border-[#353941] bg-[#1d2026]' : 'border-gray-200 bg-white'}`}>
                          <div className="flex items-center justify-between gap-3">
                            <div className="min-w-0">
                              <div className={`text-[12px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                                当前文档 AI 缓存
                              </div>
                              <div className={`mt-0.5 text-[11px] leading-relaxed ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                                清理总结、大纲、速览和翻译缓存
                              </div>
                            </div>
                            <button
                              type="button"
                              onClick={handleClearDocumentAICache}
                              disabled={!docId || isMinerUFullRoutePending}
                              className={`inline-flex shrink-0 items-center gap-1.5 rounded-full px-3 py-1.5 text-[11px] font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
                                darkMode
                                  ? 'bg-white/10 text-gray-100 hover:bg-white/15'
                                  : 'bg-gray-900 text-white hover:bg-gray-800'
                              }`}
                            >
                              <Trash2 size={13} />
                              清理
                            </button>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="text-[12px] text-gray-500 py-2">加载中...</div>
                    )}
                  </div>
                </div>
                )}

                {settingsSection === 'retrieval' && (
                <div className={`settings-card p-5 border space-y-4 mt-4 mx-1 ${darkMode ? 'settings-card-dark bg-[#24272e] border-[#373b44]' : 'bg-white border-gray-200/90'}`}>
                  <h3 className={`text-[13px] font-bold tracking-wider uppercase mb-1 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    检索参数
                  </h3>
                  
                  <div className="space-y-3">
                    <div className="flex flex-col gap-1.5">
                      <label className={`text-[12px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>邻居上下文扩展</label>
                      <CustomSelect
                        value={numExpandContextChunk}
                        onChange={setNumExpandContextChunk}
                        options={[
                          { value: 0, label: '关闭' },
                          { value: 1, label: '±1 块' },
                          { value: 2, label: '±2 块' },
                          { value: 3, label: '±3 块' },
                        ]}
                      />
                      <p className="text-[11px] text-gray-500 mt-0.5">命中段前后各带上几块</p>
                    </div>
                  </div>

                  {lastCallInfo && (
                    <div className={`mt-4 p-3.5 rounded-[14px] border text-[12px] ${darkMode ? 'bg-[#1d2026] border-[#353941]' : 'bg-gray-50 border-gray-200'}`}>
                      <div className="flex justify-between items-center mb-1.5">
                        <span className="text-gray-500">调用来源</span>
                        <strong className={darkMode ? 'text-gray-300' : 'text-gray-700'}>{lastCallInfo.provider || '未知'}</strong>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-gray-500">模型</span>
                        <strong className={darkMode ? 'text-gray-300' : 'text-gray-700'}>{lastCallInfo.model || '未返回'}</strong>
                      </div>
                      {(() => {
                        const usage = getUsageTokenSummary(lastCallInfo.usage);
                        if (!usage) return null;
                        return (
                          <>
                          <div className="mt-1.5 flex justify-between items-center">
                            <span className="text-gray-500">Token</span>
                            <strong className={darkMode ? 'text-gray-300' : 'text-gray-700'}>
                              {Number.isFinite(usage.total) ? usage.total : '-'}
                              <span className="ml-1 font-normal text-gray-400">
                                ({Number.isFinite(usage.prompt) ? usage.prompt : '-'} / {Number.isFinite(usage.completion) ? usage.completion : '-'})
                              </span>
                              {usage.estimated && <span className="ml-1 font-normal text-gray-400">估</span>}
                            </strong>
                          </div>
                          {usage.cost && Number.isFinite(usage.cost.amount) && (
                            <div className="mt-1.5 flex justify-between items-center">
                              <span className="text-gray-500">费用</span>
                              <strong className={darkMode ? 'text-gray-300' : 'text-gray-700'}>
                                {usage.cost.currency} {usage.cost.amount}
                                {usage.cost.estimated && <span className="ml-1 font-normal text-gray-400">估</span>}
                              </strong>
                            </div>
                          )}
                          </>
                        );
                      })()}
                      {lastCallInfo.fallback && (
                        <div className="mt-2 pt-2 border-t border-gray-200/50 text-amber-600 font-medium flex items-center justify-center gap-1.5">
                          <span className="w-1.5 h-1.5 rounded-full bg-amber-500"></span>
                          已切换备用模型
                        </div>
                      )}
                    </div>
                  )}
                </div>
                )}

                {/* Other Settings Access：展示当前关键值，并进入对应的完整设置页 */}
                {settingsSection === 'common' && (
                <section className="px-1" aria-labelledby="more-settings-heading">
                  <div className="mb-2.5 px-1">
                    <h3 id="more-settings-heading" className={`text-[13px] font-bold ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>详细设置</h3>
                    <p className={`mt-0.5 text-[11px] ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>点击进入完整配置</p>
                  </div>
                  <div className="grid grid-cols-3 gap-3">
                    {[
                      {
                        Icon: Type,
                        label: '全局设置',
                        desc: '字体、记忆和联网',
                        meta: `${Math.round((globalScale || 1) * 100)}% 界面 · 记忆${enableMemory ? '开启' : '关闭'}`,
                        onClick: () => { setShowSettings(false); setShowGlobalSettings(true); },
                      },
                      {
                        Icon: SlidersHorizontal,
                        label: '对话设置',
                        desc: '发送、消息和公式',
                        meta: `${sendShortcut === 'Ctrl+Enter' ? 'Ctrl + Enter' : 'Enter'} 发送 · ${streamOutput ? '流式' : '整段'} · ${messageStyle === 'bubble' ? '气泡' : '简洁'}`,
                        onClick: () => { setShowSettings(false); setShowChatSettings(true); },
                      },
                      {
                        Icon: ScanText,
                        label: '解析设置',
                        desc: '上传路线和 OCR',
                        meta: selectedParseRoute === 'local' ? '本地全程 · 设备内处理' : 'MinerU 全程 · 结构化解析',
                        onClick: () => { setShowSettings(false); setShowOCRSettings(true); },
                      },
                    ].map(({ Icon, label, desc, meta, onClick }) => (
                      <button
                        key={label}
                        onClick={onClick}
                        className={`settings-entry-row settings-card settings-card-interactive flex min-h-[128px] w-full flex-col p-4 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/25 ${darkMode ? 'settings-card-dark bg-[#24272e] border-[#373b44]' : 'bg-white border-gray-200/90'}`}
                      >
                        <div className="flex w-full items-start justify-between">
                          <div className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-[12px] ${darkMode ? 'bg-white/[0.06] text-gray-400' : 'bg-gray-100/90 text-gray-500'}`}>
                            <Icon size={16} strokeWidth={2.1} />
                          </div>
                          <span className="settings-entry-arrow" aria-hidden="true">
                            <ChevronRight className="h-3.5 w-3.5" strokeWidth={2.2} />
                          </span>
                        </div>
                        <div className="mt-3 min-w-0">
                          <div className={`text-[13px] font-bold ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>{label}</div>
                          <div className={`mt-1 text-[11px] leading-relaxed ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>{desc}</div>
                        </div>
                        <div className={`mt-auto truncate pt-3 text-[10px] font-semibold ${darkMode ? 'text-[#e5a28d]' : 'text-[#A65B45]'}`} title={meta}>{meta}</div>
                      </button>
                    ))}
                  </div>
                </section>
                )}
                  </motion.div>
                </div>
              </div>

              <div className={`settings-chrome flex flex-shrink-0 items-center justify-between gap-3 border-t px-7 py-3.5 ${darkMode ? 'border-[#353941]' : 'border-gray-200/80'}`}>
                <div className={`flex items-center gap-1.5 text-[11px] font-medium ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                  <Check size={13} strokeWidth={2.5} />
                  <span>更改已保存</span>
                </div>
                <button onClick={() => setShowSettings(false)} className="accent-cta flex items-center gap-1.5 rounded-full px-5 py-2.5 text-[13px] font-semibold">
                  <Check size={15} strokeWidth={2.5} />
                  <span>完成</span>
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 懒加载设置面板（使用 useCallback 稳定的关闭回调） */}
      <Suspense fallback={null}>
        <EmbeddingSettings
          isOpen={showEmbeddingSettings}
          onClose={handleEmbeddingSettingsClose}
          onExitComplete={handleEmbeddingSettingsExitComplete}
        />
      </Suspense>
      <Suspense fallback={null}>
        <GlobalSettings isOpen={showGlobalSettings} onClose={handleGlobalSettingsClose} />
      </Suspense>
      <Suspense fallback={null}>
        <ChatSettings isOpen={showChatSettings} onClose={handleChatSettingsClose} />
      </Suspense>
      <Suspense fallback={null}>
        <OCRSettingsPanel isOpen={showOCRSettings} onClose={handleOCRSettingsClose} />
      </Suspense>

      {/* 反馈 Modal */}
      <AnimatePresence>
        {feedbackTarget && (
          <FeedbackModal
            onSubmit={handleFeedbackSubmit}
            onClose={() => setFeedbackTarget(null)}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

// 反馈 Modal 组件
const ISSUE_OPTIONS = [
  { value: 'wrong_answer', label: '答案错误' },
  { value: 'wrong_citation', label: '引文不对' },
  { value: 'irrelevant', label: '答非所问' },
  { value: 'offensive', label: '内容不当' },
];

const FeedbackModal = ({ onSubmit, onClose }) => {
  const [selectedIssues, setSelectedIssues] = useState([]);
  const [detail, setDetail] = useState('');

  const toggleIssue = (val) => {
    setSelectedIssues(prev =>
      prev.includes(val) ? prev.filter(v => v !== val) : [...prev, val]
    );
  };

  return (
    <motion.div
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.95, opacity: 0 }}
        className="bg-white rounded-2xl shadow-xl p-5 w-80 max-w-[90vw]"
        onClick={e => e.stopPropagation()}
      >
        <h3 className="text-sm font-semibold text-gray-800 mb-3">反馈问题</h3>
        <div className="flex flex-wrap gap-2 mb-3">
          {ISSUE_OPTIONS.map(opt => (
            <button
              key={opt.value}
              onClick={() => toggleIssue(opt.value)}
              className={`text-xs px-3 py-1.5 rounded-full border transition-colors ${
                selectedIssues.includes(opt.value)
                  ? 'border-orange-400 bg-orange-50 text-orange-600'
                  : 'border-gray-200 text-gray-500 hover:border-gray-300'
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
        <textarea
          value={detail}
          onChange={e => setDetail(e.target.value)}
          placeholder="补充说明（可选）"
          className="w-full text-xs border border-gray-200 rounded-lg p-2.5 resize-none h-16 focus:outline-none focus:ring-1 focus:ring-orange-300 mb-3"
        />
        <div className="flex justify-end gap-2">
          <button onClick={onClose} className="text-xs px-3 py-1.5 text-gray-500 hover:text-gray-700">取消</button>
          <button
            onClick={() => onSubmit(selectedIssues, detail)}
            disabled={selectedIssues.length === 0}
            className="text-xs px-4 py-1.5 rounded-lg bg-orange-500 text-white hover:bg-orange-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            提交
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
};

// 下拉挂到 body，避免被 settings-card 的 overflow-hidden / 滚动容器裁切
const SELECT_MENU_MAX_HEIGHT = 240;
const SELECT_MENU_GAP = 6;
const SELECT_VIEWPORT_PAD = 8;

const getSelectMenuStyle = (trigger, optionCount) => {
  const rect = trigger.getBoundingClientRect();
  const estimatedHeight = Math.min(SELECT_MENU_MAX_HEIGHT, optionCount * 42 + 16);
  const spaceBelow = window.innerHeight - rect.bottom - SELECT_VIEWPORT_PAD;
  const spaceAbove = rect.top - SELECT_VIEWPORT_PAD;
  const opensUpward = spaceBelow < Math.min(estimatedHeight, 160) && spaceAbove > spaceBelow;
  const available = (opensUpward ? spaceAbove : spaceBelow) - SELECT_MENU_GAP;
  const maxHeight = Math.max(96, Math.min(SELECT_MENU_MAX_HEIGHT, available));
  const left = Math.max(
    SELECT_VIEWPORT_PAD,
    Math.min(rect.left, window.innerWidth - rect.width - SELECT_VIEWPORT_PAD)
  );

  return {
    opensUpward,
    style: {
      position: 'fixed',
      left,
      width: rect.width,
      zIndex: 80,
      maxHeight,
      ...(opensUpward
        ? { bottom: window.innerHeight - rect.top + SELECT_MENU_GAP, top: 'auto' }
        : { top: rect.bottom + SELECT_MENU_GAP, bottom: 'auto' }),
    },
  };
};

const CustomSelect = ({ value, onChange, options, unavailableLabel }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [menuPlacement, setMenuPlacement] = useState(null);
  const containerRef = useRef(null);
  const menuRef = useRef(null);

  const selectedOption = options.find(opt => opt.value === value);
  const selectedLabel = selectedOption?.label || unavailableLabel || '请选择';
  const opensUpward = Boolean(menuPlacement?.opensUpward);
  const menuStyle = menuPlacement?.style;

  const updateMenuPosition = useCallback(() => {
    const trigger = containerRef.current;
    if (!trigger) return;
    setMenuPlacement(getSelectMenuStyle(trigger, options.length));
  }, [options.length]);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (containerRef.current?.contains(event.target)) return;
      if (menuRef.current?.contains(event.target)) return;
      setIsOpen(false);
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    if (!isOpen) return undefined;
    updateMenuPosition();

    const handleReposition = () => updateMenuPosition();
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') setIsOpen(false);
    };

    window.addEventListener('resize', handleReposition);
    document.addEventListener('scroll', handleReposition, true);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('resize', handleReposition);
      document.removeEventListener('scroll', handleReposition, true);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, updateMenuPosition]);

  const toggleMenu = () => {
    if (isOpen) {
      setIsOpen(false);
      return;
    }
    const trigger = containerRef.current;
    if (trigger) {
      setMenuPlacement(getSelectMenuStyle(trigger, options.length));
    }
    setIsOpen(true);
  };

  const menuNode = (
    <AnimatePresence>
      {isOpen && menuStyle && (
        <motion.div
          ref={menuRef}
          initial={{ opacity: 0, y: opensUpward ? 5 : -5, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: opensUpward ? 5 : -5, scale: 0.95 }}
          transition={{ duration: 0.15 }}
          style={menuStyle}
          className="p-1.5 bg-white/95 dark:bg-[#1a1d21]/95 backdrop-blur-xl border border-gray-100 dark:border-white/10 rounded-[16px] shadow-[0_8px_30px_rgb(0,0,0,0.12)] overflow-y-auto"
        >
          {options.map((option) => (
            <button
              key={option.value}
              type="button"
              onClick={() => {
                onChange(option.value);
                setIsOpen(false);
              }}
              className={`w-full text-left px-3 py-2.5 rounded-[10px] text-[13px] transition-colors flex items-center justify-between ${
                value === option.value
                  ? 'bg-[#FFA07A]/10 text-[#B85F47] font-semibold'
                  : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100/50 dark:hover:bg-white/5'
              }`}
            >
              {option.label}
              {value === option.value && <Check size={14} className="text-[#B85F47]" />}
            </button>
          ))}
        </motion.div>
      )}
    </AnimatePresence>
  );

  return (
    <div className="relative w-full" ref={containerRef}>
      <button
        type="button"
        onClick={toggleMenu}
        aria-expanded={isOpen}
        aria-haspopup="listbox"
        className={`w-full flex items-center justify-between p-2.5 rounded-[12px] bg-white/50 dark:bg-black/20 border text-sm transition-all outline-none ${
          isOpen
            ? 'border-[#FFA07A] shadow-[0_0_0_3px_rgba(255,160,122,0.14)]'
            : 'border-gray-200 dark:border-white/10 hover:border-[#FFA07A]/50'
        }`}
      >
        <span className="min-w-0 flex-1 truncate pr-3 text-left text-gray-700 dark:text-gray-300 font-medium" title={selectedLabel}>
          {selectedLabel}
        </span>
        <ChevronDown size={14} className={`text-gray-500 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
      </button>
      {typeof document !== 'undefined' ? createPortal(menuNode, document.body) : menuNode}
    </div>
  );
};

export default ChatPDF;
export { resolveTableVisualVerificationStatus } from './ChatMessageRow';
