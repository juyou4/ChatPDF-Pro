import React, { useState, useRef, useEffect, useMemo, useCallback, lazy, Suspense } from 'react';
import { Upload, Send, Settings, ChevronLeft, ChevronRight, ChevronDown, ZoomIn, ZoomOut, Copy, Bot, X, Crop, Image as ImageIcon, History, Moon, Sun, Plus, MessageSquare, Trash2, Menu, Type, Loader2, Server, Database, ListFilter, ArrowUpRight, SlidersHorizontal, Paperclip, ScanText, Scan, Brain, MessageCircle, ArrowUpDown, Globe, Check, Sparkles } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { supportsVision } from '../utils/visionDetectorUtils';
import ScreenshotPreview from './ScreenshotPreview';
import 'katex/dist/katex.min.css';
import 'highlight.js/styles/github.css';
import PDFViewer from './PDFViewer';
import StreamingMarkdown from './StreamingMarkdown';
import TextSelectionToolbar from './TextSelectionToolbar';
import { useProvider } from '../contexts/ProviderContext';
import { useModel } from '../contexts/ModelContext';
import { useDefaults } from '../contexts/DefaultsContext';
import { useCapabilities } from '../contexts/CapabilitiesContext';
const EmbeddingSettings = lazy(() => import('./EmbeddingSettings'));
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
import { usePDFState } from '../hooks/usePDFState';
import { useScreenshotState } from '../hooks/useScreenshotState';
import PresetQuestions from './PresetQuestions';
import ModelQuickSwitch from './ModelQuickSwitch';
import ThinkingBlock from './ThinkingBlock';
import EvidencePanel from './EvidencePanel';
import MindmapView from './MindmapView';
import VirtualMessageList from './VirtualMessageList';
import WebSearchButton from './WebSearchButton';
import DocumentOutline from './DocumentOutline';
import ReadingAnalysisPanel from './ReadingAnalysisPanel';
import ReadingSummaryPanel from './ReadingSummaryPanel';
import { shouldStreamAssistantContent } from '../utils/messageRenderUtils';

const WebSearchSourcesBadge = ({ sources }) => {
  const [expanded, setExpanded] = useState(false);
  if (!sources || sources.length === 0) return null;
  return (
    <div className="mt-3 border-t border-gray-100 pt-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-xs text-purple-600 hover:text-purple-800 transition-colors font-medium"
      >
        <Globe className="w-3.5 h-3.5" />
        <span>联网搜索来源 ({sources.length})</span>
        <ChevronDown className={`w-3 h-3 transition-transform ${expanded ? 'rotate-180' : ''}`} />
      </button>
      {expanded && (
        <div className="mt-2 space-y-1.5">
          {sources.map((src, i) => (
            <a
              key={i}
              href={src.url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-start gap-2 p-2 rounded-lg bg-purple-50/50 hover:bg-purple-50 transition-colors group"
            >
              <span className="text-[10px] font-bold text-purple-500 bg-purple-100 rounded px-1 py-0.5 mt-0.5 flex-shrink-0">{i + 1}</span>
              <div className="min-w-0">
                <div className="text-xs font-medium text-gray-800 truncate group-hover:text-purple-700">{src.title}</div>
                {src.snippet && <div className="text-[11px] text-gray-500 line-clamp-2 mt-0.5">{src.snippet}</div>}
              </div>
              <ArrowUpRight className="w-3 h-3 text-gray-400 group-hover:text-purple-500 flex-shrink-0 mt-0.5" />
            </a>
          ))}
        </div>
      )}
    </div>
  );
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
        {meta?.truncated && <span className="rounded-full bg-emerald-50 px-1.5 py-0.5 text-[10px] text-emerald-700">已截断</span>}
        <ChevronDown className={`w-3 h-3 transition-transform ${expanded ? 'rotate-180' : ''}`} />
      </button>
      {expanded && (
        <div className="mt-2 space-y-2">
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
};

export const resolveTableVisualVerificationStatus = (verification) => {
  const rawState = String(verification?.state || '').trim().toLowerCase();
  const state = ['queued', 'running', 'pending'].includes(rawState) ? 'pending' : rawState;
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

const SendIcon = () => (
  <svg className="glass-btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="m6.998 10.247l.435.76c.277.485.415.727.415.993s-.138.508-.415.992l-.435.761c-1.238 2.167-1.857 3.25-1.375 3.788c.483.537 1.627.037 3.913-.963l6.276-2.746c1.795-.785 2.693-1.178 2.693-1.832s-.898-1.047-2.693-1.832L9.536 7.422c-2.286-1-3.43-1.5-3.913-.963s.137 1.62 1.375 3.788Z" />
  </svg>
);

const PauseIcon = () => (
  <svg className="glass-btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="7" y="6" width="4" height="12" rx="2" />
    <rect x="13" y="6" width="4" height="12" rx="2" />
  </svg>
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

const UPLOAD_RING_CONFIGS = [
  { s: 298, w: 14, c: 'rgba(100, 50, 255, 0.5)',  br: '52% 48% 55% 45% / 48% 52% 48% 52%', dur: 4.2, del: -2.1, dir: 'normal',  mix: 'screen' },
  { s: 302, w: 22, c: 'rgba(50, 150, 255, 0.5)',  br: '45% 55% 48% 52% / 55% 45% 52% 48%', dur: 6.8, del: -4.3, dir: 'reverse', mix: 'screen' },
  { s: 295, w: 17, c: 'rgba(0, 200, 255, 0.4)',   br: '58% 42% 45% 55% / 42% 58% 48% 52%', dur: 3.5, del: -1.7, dir: 'normal',  mix: 'overlay' },
  { s: 304, w: 20, c: 'rgba(255, 100, 50, 0.5)',  br: '48% 52% 52% 48% / 58% 42% 55% 45%', dur: 7.3, del: -3.6, dir: 'reverse', mix: 'screen' },
  { s: 293, w: 13, c: 'rgba(255, 200, 50, 0.4)',  br: '55% 45% 48% 52% / 45% 55% 42% 58%', dur: 5.1, del: -0.8, dir: 'normal',  mix: 'screen' },
  { s: 301, w: 19, c: 'rgba(150, 50, 200, 0.5)',  br: '42% 58% 55% 45% / 52% 48% 58% 42%', dur: 4.7, del: -2.9, dir: 'reverse', mix: 'overlay' },
  { s: 297, w: 16, c: 'rgba(100, 50, 255, 0.4)',  br: '50% 50% 52% 48% / 55% 45% 50% 50%', dur: 6.2, del: -4.8, dir: 'normal',  mix: 'screen' },
  { s: 303, w: 23, c: 'rgba(50, 150, 255, 0.4)',  br: '46% 54% 50% 50% / 48% 52% 45% 55%', dur: 3.8, del: -1.2, dir: 'reverse', mix: 'screen' },
  { s: 299, w: 15, c: 'rgba(0, 200, 255, 0.5)',   br: '53% 47% 46% 54% / 50% 50% 53% 47%', dur: 7.6, del: -3.1, dir: 'normal',  mix: 'overlay' },
  { s: 305, w: 21, c: 'rgba(255, 100, 50, 0.4)',  br: '49% 51% 53% 47% / 46% 54% 49% 51%', dur: 4.9, del: -2.4, dir: 'reverse', mix: 'screen' },
  { s: 292, w: 18, c: 'rgba(255, 200, 50, 0.5)',  br: '57% 43% 49% 51% / 53% 47% 46% 54%', dur: 5.5, del: -0.5, dir: 'normal',  mix: 'screen' },
  { s: 300, w: 12, c: 'rgba(150, 50, 200, 0.4)',  br: '44% 56% 51% 49% / 57% 43% 52% 48%', dur: 6.5, del: -4.0, dir: 'reverse', mix: 'overlay' },
];

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
  ready: {
    label: 'Ready',
    title: '文档准备完成',
    desc: '正在进入阅读界面',
  },
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
const TRANSLATABLE_READING_BLOCK_TYPES = new Set(['heading', 'paragraph', 'caption']);
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

const ChatPDF = () => {
  // ========== Context Hooks ==========
  const { getProviderById } = useProvider();
  const { getModelById } = useModel();
  const { getDefaultModel } = useDefaults();
  const { hasLocalRerank } = useCapabilities();
  const globalSettings = useGlobalSettings();
  const { setReasoningEffort, reasoningEffort, streamOutput, setStreamOutput } = globalSettings;
  const {
    sendShortcut, confirmDeleteMessage, confirmRegenerateMessage, messageStyle, messageFontSize, codeCollapsible, codeWrappable, codeShowLineNumbers,
    overrideNumericTable, setOverrideNumericTable,
    overrideAnswerCritic, setOverrideAnswerCritic,
    overrideLLMQueryRewrite, setOverrideLLMQueryRewrite,
    overrideBM25Synonyms, setOverrideBM25Synonyms,
    numericTableVisualVerification, setNumericTableVisualVerification,
    cheapModel, setCheapModel,
    cheapModelProvider, setCheapModelProvider,
  } = useChatParams();
  const {
    aiAutoProcess,
    setAiAutoProcess,
    autoOutlineSummary,
    setAutoOutlineSummary,
    autoPretranslate: enableHoverPretranslate,
    setAutoPretranslate,
    pretranslateConcurrency,
    setPretranslateConcurrency,
    overviewDefaultDepth,
    setOverviewDefaultDepth,
  } = useReadingSettings();
  const shouldAutoPretranslate = aiAutoProcess && enableHoverPretranslate;

  // ========== 设置状态 - 使用防抖 localStorage 写入（需求 8.1） ==========
  const [apiKey, setApiKey] = useDebouncedLocalStorage('apiKey', '');
  const [apiProvider, setApiProvider] = useDebouncedLocalStorage('apiProvider', 'openai');
  const [model, setModel] = useDebouncedLocalStorage('model', 'gpt-4o');
  const [embeddingApiKey, setEmbeddingApiKey] = useDebouncedLocalStorage('embeddingApiKey', '');
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
  const [searchEngine, setSearchEngine] = useDebouncedLocalStorage('searchEngine', 'google');
  const [searchEngineUrl, setSearchEngineUrl] = useDebouncedLocalStorage('searchEngineUrl', 'https://www.google.com/search?q={query}');
  const [toolbarSize, setToolbarSize] = useDebouncedLocalStorage('toolbarSize', 'normal');
  const [toolbarScale, setToolbarScale] = useDebouncedLocalStorage('toolbarScale', 1);
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
  const [toolbarPosition, setToolbarPosition] = useState({ x: 0, y: 0 });
  const [sidebarMode, setSidebarMode] = useState('history');
  const [settingsSection, setSettingsSection] = useState('common');
  const [showRetrievalTuning, setShowRetrievalTuning] = useState(false);
  const [blockIndex, setBlockIndex] = useState(null);
  const [blockIndexLoading, setBlockIndexLoading] = useState(false);
  const [blockIndexError, setBlockIndexError] = useState('');
  const [blockIndexReloadKey, setBlockIndexReloadKey] = useState(0);
  const [readingOutline, setReadingOutline] = useState(null);
  const [readingOutlineLoading, setReadingOutlineLoading] = useState(false);
  const [readingOutlineError, setReadingOutlineError] = useState('');
  const [readingOutlineReloadKey, setReadingOutlineReloadKey] = useState(0);
  const [sectionOutline, setSectionOutline] = useState(null);
  const [sectionOutlineLoading, setSectionOutlineLoading] = useState(false);
  const [sectionOutlineError, setSectionOutlineError] = useState('');
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
  const [pretranslateProgress, setPretranslateProgress] = useState({ running: false, done: 0, total: 0 });
  const [failedTranslationBlockIds, setFailedTranslationBlockIds] = useState(new Set());
  const [pretranslateNotice, setPretranslateNotice] = useState('');
  const [showAiProcessingPanel, setShowAiProcessingPanel] = useState(false);
  const [deepParseStatus, setDeepParseStatus] = useState(null);
  const [deepParseNotice, setDeepParseNotice] = useState('');
  const [ragIndexStatus, setRagIndexStatus] = useState(null);
  const [ragIndexBusy, setRagIndexBusy] = useState(false);
  const [ragIndexNotice, setRagIndexNotice] = useState('');
  const [hoveredReadingBlockId, setHoveredReadingBlockId] = useState(null);
  const [pinnedReadingBlockId, setPinnedReadingBlockId] = useState(null);
  const pretranslateRunRef = useRef(0);
  const pretranslateStartedDocRef = useRef(null);
  const pretranslateAbortRef = useRef(null);
  const prevShouldAutoPretranslateRef = useRef(shouldAutoPretranslate);
  const readingOutlineRequestRef = useRef(0);
  const readingOutlineForceRef = useRef(false);
  const sectionOutlineForceRef = useRef(false);
  const streamConfigMigratedRef = useRef(false);

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
    showSidebar, setShowSidebar,
    isHeaderExpanded, setIsHeaderExpanded,
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

  useEffect(() => {
    if (overviewDefaultDepth && overviewDefaultDepth !== overviewDepth) {
      setOverviewDepth(overviewDefaultDepth);
    }
  }, [overviewDefaultDepth, overviewDepth, setOverviewDepth]);

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

  const getEmbeddingApiKey = useCallback(() => {
    const config = getEmbeddingConfig();
    if (config.isValid && config.provider?.apiKey) {
      return config.provider.apiKey;
    }
    return embeddingApiKey || apiKey;
  }, [getEmbeddingConfig, embeddingApiKey, apiKey]);

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
      return { providerId, modelId, apiKey: provider?.apiKey || '' };
    }
    return { providerId, modelId, apiKey: provider?.apiKey || apiKey };
  }, [getDefaultModel, getCurrentChatModel, getProviderById, apiKey]);

  const getChatRequestConfig = useCallback(() => {
    const chatCredentials = getChatCredentials?.();
    const chatProvider = chatCredentials?.providerId || 'openai';
    const chatModel = chatCredentials?.modelId || 'gpt-4o';
    const chatApiKey = chatCredentials?.apiKey || '';
    const chatProviderFull = getProviderById?.(chatProvider);
    const providerLower = String(chatProvider || '').toLowerCase();
    const canCallModel = Boolean(chatApiKey) || providerLower === 'local' || providerLower === 'ollama';
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

    return {
      headers,
      providerId: chatProvider,
      providerName: chatProviderFull?.name || chatProvider,
      canCallModel,
    };
  }, [getChatCredentials, getProviderById]);

  const getCurrentRerankModel = useCallback(() => {
    const rrk = getDefaultModel('rerankModel');
    if (rrk) {
      const [pid, mid] = rrk.split(':');
      return { providerId: pid, modelId: mid };
    }
    // 没有配置 rerank 模型时，仅在本地 rerank 可用时才 fallback 到本地
    if (hasLocalRerank) {
      return { providerId: 'local', modelId: 'BAAI/bge-reranker-base' };
    }
    return null;
  }, [getDefaultModel, hasLocalRerank]);

  const getRerankCredentials = useCallback(() => {
    const rerankModel = getCurrentRerankModel();
    if (!rerankModel) return null;
    const { providerId, modelId } = rerankModel;
    const provider = getProviderById(providerId);
    return { providerId, modelId, apiKey: provider?.apiKey || getEmbeddingApiKey() };
  }, [getCurrentRerankModel, getProviderById, getEmbeddingApiKey]);

  const getDefaultModelLabel = useCallback((key, fallback = '未选择') => {
    if (!key) return fallback;
    const [pid, mid] = key.split(':');
    const p = getProviderById(pid);
    const m = getModelById(mid, pid);
    return `${p?.name || pid} - ${m?.name || mid}`;
  }, [getProviderById, getModelById]);

  const currentChatModelObj = useMemo(() => {
    const chatKey = getDefaultModel('assistantModel');
    if (!chatKey || !chatKey.includes(':')) return null;
    const [pid, mid] = chatKey.split(':');
    return getModelById(mid, pid);
  }, [getDefaultModel, getModelById]);

  const isVisionCapable = useMemo(() => supportsVision(currentChatModelObj), [currentChatModelObj]);

  // ========== 文档状态 Hook（需求 1.1） ==========
  // useDocumentState 内部管理 docId/docInfo，需要其他 Hook 的 setter 函数
  // setter 函数通过 ref 桥接，避免 Hook 调用顺序问题
  const messageSettersRef = useRef({});
  const pdfSettersRef = useRef({});
  const screenshotSettersRef = useRef({});

  const documentState = useDocumentState({
    getEmbeddingConfig,
    getChatCredentials,
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
    isUploading, uploadProgress, uploadStatus,
    history, storageInfo,
    fileInputRef,
    handleFileUpload, startNewChat, loadSession, deleteSession,
    saveCurrentSession, fetchStorageInfo,
    overview, overviewLoading, overviewError, fetchOverview, clearOverviewCache, overviewFigureMode,
  } = documentState;

  // ========== PDF 状态 Hook（需求 1.1） ==========
  const pdfState = usePDFState({
    docId,
    docInfo,
    useRerank: useRerankSetting,
    rerankerModel,
    getRerankCredentials,
    embeddingApiKey: getEmbeddingApiKey(),
    apiKey,
  });
  const {
    currentPage, setCurrentPage,
    pdfScale, setPdfScale,
    selectedText, setSelectedText,
    showTextMenu, setShowTextMenu,
    menuPosition, setMenuPosition,
    searchQuery, setSearchQuery,
    searchResults,
    currentResultIndex,
    isSearching,
    searchHistory,
    activeHighlight, setActiveHighlight,
    pdfContainerRef,
    handleSearch, focusResult, handleCitationClick,
    formatSimilarity, renderHighlightedSnippet,
  } = pdfState;

  useEffect(() => {
    let cancelled = false;

    if (!docId) {
      setBlockIndex(null);
      setBlockIndexError('');
      setBlockIndexLoading(false);
      setReadingOutline(null);
      setReadingOutlineError('');
      setReadingOutlineLoading(false);
      setSectionOutline(null);
      setSectionOutlineError('');
      setSectionOutlineLoading(false);
      setActiveReadingNodeId(null);
      setVisitedReadingNodeIds(new Set());
      setActiveSectionNodeId(null);
      setVisitedSectionNodeIds(new Set());
      setBlockTranslations({});
      setBlockTranslateError('');
      setPretranslateNotice('');
      setBlockTranslateLoading(false);
      setTranslatingBlockIds(new Set());
      setBlockTranslationsLoaded(false);
      setPretranslateProgress({ running: false, done: 0, total: 0 });
      setFailedTranslationBlockIds(new Set());
      setDeepParseStatus(null);
      setDeepParseNotice('');
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
    setReadingOutlineLoading(false);
    setSectionOutline(null);
    setSectionOutlineError('');
    setSectionOutlineLoading(false);
    setActiveReadingNodeId(null);
    setVisitedReadingNodeIds(new Set());
    setActiveSectionNodeId(null);
    setVisitedSectionNodeIds(new Set());
    setBlockTranslations({});
    setBlockTranslateError('');
    setPretranslateNotice('');
    setBlockTranslateLoading(false);
    setTranslatingBlockIds(new Set());
    setBlockTranslationsLoaded(false);
    setPretranslateProgress({ running: false, done: 0, total: 0 });
    setFailedTranslationBlockIds(new Set());
    setDeepParseStatus(null);
    setDeepParseNotice('');
    setRagIndexStatus(null);
    setRagIndexNotice('');
    setRagIndexBusy(false);
    setHoveredReadingBlockId(null);
    setPinnedReadingBlockId(null);
    pretranslateRunRef.current += 1;
    pretranslateAbortRef.current?.abort();
    pretranslateAbortRef.current = null;
    pretranslateStartedDocRef.current = null;

    fetch(`${API_BASE_URL}/documents/${docId}/blocks?t=${Date.now()}`)
      .then(async (res) => {
        if (!res.ok) {
          const detail = await res.json().catch(() => ({}));
          throw new Error(detail?.detail || `HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data) => {
        if (!cancelled) setBlockIndex(data);
      })
      .catch((error) => {
        if (!cancelled) {
          console.warn('[ImmersiveReading] blocks 加载失败', error);
          setBlockIndexError('大纲加载失败');
        }
      })
      .finally(() => {
        if (!cancelled) setBlockIndexLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [docId, blockIndexReloadKey]);

  const refreshDeepParseStatus = useCallback(async () => {
    if (!docId) return null;
    const res = await fetch(`${API_BASE_URL}/documents/${docId}/deep-parse/status?t=${Date.now()}`);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data?.detail || `HTTP ${res.status}`);
    setDeepParseStatus(data);
    if (data?.rag_index) setRagIndexStatus(data.rag_index);
    return data;
  }, [docId]);

  const refreshRagIndexStatus = useCallback(async () => {
    if (!docId) return null;
    const res = await fetch(`${API_BASE_URL}/documents/${docId}/rag-index/status?t=${Date.now()}`);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data?.detail || `HTTP ${res.status}`);
    setRagIndexStatus(data);
    return data;
  }, [docId]);

  const refreshReadingBlocksAfterDeepParse = useCallback(() => {
    setBlockTranslations({});
    setBlockTranslationsLoaded(false);
    setFailedTranslationBlockIds(new Set());
    setTranslatingBlockIds(new Set());
    setBlockTranslateError('');
    setPretranslateNotice('');
    setPretranslateProgress({ running: false, done: 0, total: 0 });
    setReadingOutline(null);
    setSectionOutline(null);
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

  useEffect(() => {
    let cancelled = false;
    if (!docId) return () => {};
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
    return () => {
      cancelled = true;
    };
  }, [docId, refreshDeepParseStatus, refreshRagIndexStatus]);

  useEffect(() => {
    if (!docId || !['queued', 'running'].includes(deepParseStatus?.status)) {
      return () => {};
    }
    let cancelled = false;
    const timer = window.setInterval(async () => {
      try {
        const data = await refreshDeepParseStatus();
        if (cancelled || !data) return;
        if (data.status === 'ready' && data.active_mineru) {
          setDeepParseNotice(
            data.recommend_rag_index_rebuild
              ? 'MinerU 深度解析完成，阅读结构、大纲与速览图表均已刷新；建议重建问答索引以启用结构化表格证据'
              : 'MinerU 深度解析完成，阅读结构、大纲与速览图表均已刷新'
          );
          refreshReadingBlocksAfterDeepParse();
          window.clearInterval(timer);
        } else if (data.status === 'failed') {
          setDeepParseNotice(data.error || 'MinerU 深度解析失败');
          window.clearInterval(timer);
        } else if (data.status === 'cancelled') {
          setDeepParseNotice('MinerU 深度解析已取消');
          window.clearInterval(timer);
        }
      } catch (error) {
        if (!cancelled) {
          setDeepParseNotice(error.message || 'MinerU 深度解析状态同步失败');
        }
      }
    }, 2500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [deepParseStatus?.status, docId, refreshDeepParseStatus, refreshReadingBlocksAfterDeepParse]);

  const handleStartMinerUDeepParse = useCallback(async () => {
    if (!docId || ['queued', 'running'].includes(deepParseStatus?.status)) return;
    const activeMinerU = Boolean(deepParseStatus?.active_mineru);
    const parseTarget = deepParseStatus?.access_mode === 'direct'
      ? 'MinerU 官方 API'
      : '你配置的 MinerU Worker 服务';
    const confirmed = window.confirm(
      activeMinerU
        ? `重新运行 MinerU 深度解析会把当前 PDF 上传到${parseTarget}，并替换阅读块、大纲、翻译缓存和速览图表。是否继续？`
        : `MinerU 深度解析会把当前 PDF 上传到${parseTarget}，用于生成带坐标的结构化解析结果，速览图表也会随之升级。是否继续？`
    );
    if (!confirmed) return;

    setDeepParseNotice('');
    try {
      const res = await fetch(`${API_BASE_URL}/documents/${docId}/deep-parse`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider: 'mineru', force: activeMinerU }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.detail || `HTTP ${res.status}`);
      setDeepParseStatus(data);
      if (data.status === 'ready' && data.active_mineru) {
        setDeepParseNotice(
          data.recommend_rag_index_rebuild
            ? 'MinerU 深度解析已就绪，阅读结构、大纲与速览图表均已刷新；建议重建问答索引以启用结构化表格证据'
            : 'MinerU 深度解析已就绪，阅读结构、大纲与速览图表均已刷新'
        );
        refreshReadingBlocksAfterDeepParse();
      } else {
        setDeepParseNotice('MinerU 深度解析已开始，可继续阅读，完成后阅读结构与速览图表会自动刷新');
      }
    } catch (error) {
      setDeepParseNotice(error.message || 'MinerU 深度解析启动失败');
      setDeepParseStatus((prev) => ({ ...(prev || {}), status: 'failed', error: error.message || '启动失败' }));
    }
  }, [deepParseStatus?.access_mode, deepParseStatus?.active_mineru, deepParseStatus?.status, docId, refreshReadingBlocksAfterDeepParse]);

  const handleCancelMinerUDeepParse = useCallback(async () => {
    if (!docId || !['queued', 'running'].includes(deepParseStatus?.status)) return;
    try {
      const res = await fetch(`${API_BASE_URL}/documents/${docId}/deep-parse/cancel`, {
        method: 'POST',
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.detail || `HTTP ${res.status}`);
      setDeepParseStatus(data);
      setDeepParseNotice('MinerU 深度解析已取消');
    } catch (error) {
      setDeepParseNotice(error.message || 'MinerU 深度解析取消失败');
    }
  }, [deepParseStatus?.status, docId]);

  const handleRebuildMinerURagIndex = useCallback(async () => {
    if (!docId || ragIndexBusy) return;
    if (!deepParseStatus?.active_mineru) {
      setRagIndexNotice('请先完成 MinerU 深度解析');
      return;
    }
    const embedConfig = getEmbeddingConfig?.() || {};
    if (!embedConfig.isValid) {
      setRagIndexNotice('请先在模型设置里选择可用的 Embedding 模型');
      return;
    }
    const embeddingModel = embedConfig.compositeKey || getDefaultModel('embeddingModel') || 'local-minilm';
    const embeddingApiKeyValue = getEmbeddingApiKey?.() || '';
    const embeddingApiHost = embedConfig.provider?.apiHost || '';

    setRagIndexBusy(true);
    setRagIndexNotice('正在评估 MinerU 问答索引重建成本...');
    try {
      const estimateRes = await fetch(`${API_BASE_URL}/documents/${docId}/rag-index/rebuild`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ estimate_only: true }),
      });
      const estimateData = await estimateRes.json().catch(() => ({}));
      if (!estimateRes.ok) {
        const detail = estimateData?.detail;
        throw new Error(typeof detail === 'string' ? detail : detail?.message || `HTTP ${estimateRes.status}`);
      }
      if (!estimateData.can_rebuild) {
        const failures = (estimateData.quality_failures || []).join('、') || '质量门未通过';
        throw new Error(`MinerU 结果暂不能重建问答索引：${failures}`);
      }
      const estimate = estimateData.estimate || {};
      const confirmed = window.confirm(
        `将使用 MinerU 结构化结果重建问答索引。\n\n`
        + `预计重新嵌入约 ${estimate.estimated_embedding_tokens || 0} tokens，约 ${estimate.estimated_chunk_count || 0} 个分块，表格 ${estimate.structured_table_count || 0} 个，耗时约 1-3 分钟。\n`
        + `历史对话中的引用可能发生偏移；阅读侧翻译、大纲和速览不受影响。\n`
        + `重建期间旧问答索引会继续可用。是否继续？`
      );
      if (!confirmed) {
        setRagIndexNotice('已取消问答索引重建');
        return;
      }
      setRagIndexNotice('正在重建 MinerU 问答索引...');
      const rebuildRes = await fetch(`${API_BASE_URL}/documents/${docId}/rag-index/rebuild`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          embedding_model: embeddingModel,
          embedding_api_key: embeddingApiKeyValue,
          embedding_api_host: embeddingApiHost,
        }),
      });
      const rebuildData = await rebuildRes.json().catch(() => ({}));
      if (!rebuildRes.ok) {
        const detail = rebuildData?.detail;
        throw new Error(typeof detail === 'string' ? detail : detail?.message || `HTTP ${rebuildRes.status}`);
      }
      setRagIndexStatus(rebuildData.rag_index || null);
      setRagIndexNotice('MinerU 问答索引已重建，表格问答会优先使用结构化证据');
      refreshDeepParseStatus().catch(() => {});
    } catch (error) {
      setRagIndexNotice(error.message || 'MinerU 问答索引重建失败，已保留原索引');
    } finally {
      setRagIndexBusy(false);
    }
  }, [
    deepParseStatus?.active_mineru,
    docId,
    getDefaultModel,
    getEmbeddingApiKey,
    getEmbeddingConfig,
    ragIndexBusy,
    refreshDeepParseStatus,
  ]);

  const handleRollbackRagIndex = useCallback(async () => {
    if (!docId || ragIndexBusy) return;
    const confirmed = window.confirm('将回退到本地 PDF 解析生成的问答索引。阅读侧 MinerU 解析不受影响。是否继续？');
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
  }, [docId, ragIndexBusy, refreshDeepParseStatus]);

  useEffect(() => {
    let cancelled = false;
    if (!docId || !blockIndex) {
      setReadingOutline(null);
      setReadingOutlineError('');
      setReadingOutlineLoading(false);
      return () => {};
    }

    const requestId = readingOutlineRequestRef.current + 1;
    readingOutlineRequestRef.current = requestId;
    const { headers, canCallModel } = getChatRequestConfig();
    const shouldForce = canCallModel && readingOutlineForceRef.current;
    readingOutlineForceRef.current = false;
    const shouldGenerate = canCallModel && (shouldForce || (aiAutoProcess && autoOutlineSummary));
    const method = shouldGenerate ? 'POST' : 'GET';
    const url = shouldGenerate
      ? `${API_BASE_URL}/documents/${docId}/reading-outline`
      : `${API_BASE_URL}/documents/${docId}/reading-outline?t=${Date.now()}`;
    const requestOptions = shouldGenerate
      ? {
          method,
          headers,
          body: JSON.stringify({ force: shouldForce }),
        }
      : { method };

    setReadingOutlineLoading(true);
    setReadingOutlineError('');
    fetch(url, requestOptions)
      .then(async (res) => {
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data?.detail || `HTTP ${res.status}`);
        return data;
      })
      .then((data) => {
        if (!cancelled && readingOutlineRequestRef.current === requestId) {
          setReadingOutline(data);
        }
      })
      .catch((error) => {
        if (!cancelled && readingOutlineRequestRef.current === requestId) {
          console.warn('[ImmersiveReading] AI 大纲加载失败', error);
          setReadingOutline(buildClientReadingFallback(blockIndex));
          setReadingOutlineError('');
        }
      })
      .finally(() => {
        if (!cancelled && readingOutlineRequestRef.current === requestId) {
          setReadingOutlineLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [docId, blockIndex, getChatRequestConfig, readingOutlineReloadKey, aiAutoProcess, autoOutlineSummary]);

  const handleRegenerateReadingOutline = useCallback(() => {
    const { canCallModel, providerName } = getChatRequestConfig();
    if (!canCallModel) {
      setReadingOutlineError(`请先为 ${providerName} 配置 API Key 后再重新生成`);
      return;
    }
    readingOutlineForceRef.current = true;
    setReadingOutlineReloadKey((value) => value + 1);
  }, [getChatRequestConfig]);

  const handleRegenerateSectionOutline = useCallback(() => {
    const { canCallModel, providerName } = getChatRequestConfig();
    if (!canCallModel) {
      setSectionOutlineError(`请先为 ${providerName} 配置 API Key 后再重新生成`);
      return;
    }
    sectionOutlineForceRef.current = true;
    setSectionOutlineReloadKey((value) => value + 1);
  }, [getChatRequestConfig]);

  useEffect(() => {
    let cancelled = false;
    if (!docId || !blockIndex) {
      setSectionOutline(null);
      setSectionOutlineError('');
      setSectionOutlineLoading(false);
      return () => {};
    }

    const { headers, canCallModel } = getChatRequestConfig();
    const shouldForce = canCallModel && sectionOutlineForceRef.current;
    sectionOutlineForceRef.current = false;
    const shouldGenerate = canCallModel && (shouldForce || (aiAutoProcess && autoOutlineSummary));
    const method = shouldGenerate ? 'POST' : 'GET';
    const url = shouldGenerate
      ? `${API_BASE_URL}/documents/${docId}/section-outline`
      : `${API_BASE_URL}/documents/${docId}/section-outline?t=${Date.now()}`;

    setSectionOutlineLoading(true);
    setSectionOutlineError('');
    fetch(url, shouldGenerate ? {
      method,
      headers,
      body: JSON.stringify({ force: shouldForce }),
    } : { method })
      .then(async (res) => {
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data?.detail || `HTTP ${res.status}`);
        return data;
      })
      .then((data) => {
        if (!cancelled) setSectionOutline(data);
      })
      .catch((error) => {
        if (!cancelled) {
          console.warn('[ImmersiveReading] 章节大纲加载失败，回退启发式大纲', error);
          setSectionOutline(null);
          setSectionOutlineError('');
        }
      })
      .finally(() => {
        if (!cancelled) setSectionOutlineLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [docId, blockIndex, getChatRequestConfig, sectionOutlineReloadKey, aiAutoProcess, autoOutlineSummary]);

  useEffect(() => {
    let cancelled = false;
    if (!docId || !blockIndex) {
      setBlockTranslations({});
      setBlockTranslationsLoaded(false);
      setFailedTranslationBlockIds(new Set());
      return () => {};
    }
    setBlockTranslationsLoaded(false);
    setFailedTranslationBlockIds(new Set());

    fetch(`${API_BASE_URL}/documents/${docId}/blocks/translations?target_lang=zh&t=${Date.now()}`)
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (!cancelled) setBlockTranslations(filterValidBlockTranslations(data?.items || {}));
      })
      .catch((error) => {
        if (!cancelled) {
          console.warn('[ImmersiveReading] 翻译缓存加载失败', error);
        }
      })
      .finally(() => {
        if (!cancelled) setBlockTranslationsLoaded(true);
      });

    return () => {
      cancelled = true;
    };
  }, [docId, blockIndex]);

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

  const inlineBlockTranslations = useMemo(() => {
    const result = { ...blockTranslations };
    readingOutlineFlat.forEach((item) => {
      const ids = item.evidence?.block_ids || item.evidence_block_ids || [];
      const outlineSummary = (item.summary || item.title || '').trim();
      ids.forEach((blockId) => {
        if (!blockId || !outlineSummary) return;
        if (result[blockId]) {
          const currentSummary = String(result[blockId]?.summary || '').trim();
          if (!currentSummary) {
            result[blockId] = {
              ...result[blockId],
              summary: outlineSummary,
              summary_source: 'reading_outline',
            };
          }
          return;
        }
        if (item.summary) {
          result[blockId] = {
            block_id: blockId,
            target_lang: 'zh',
            translation: item.summary,
            summary: item.title,
            source: 'reading_outline',
          };
        }
      });
    });
    return result;
  }, [blockTranslations, readingOutlineFlat]);

  const activeReadingNode = activeReadingNodeId ? readingNodeById[activeReadingNodeId] : null;
  const focusedReadingBlockIds = useMemo(() => {
    const ids = activeReadingNode?.evidence?.block_ids || activeReadingNode?.evidence_block_ids || [];
    const primaryBlockId = activeReadingNode?.first_block || ids[0] || pinnedReadingBlockId;
    return primaryBlockId ? [primaryBlockId] : [];
  }, [activeReadingNode, pinnedReadingBlockId]);

  const activeReadingBlockId = hoveredReadingBlockId || focusedReadingBlockIds[0] || pinnedReadingBlockId;

  const allTranslatableReadingBlocks = useMemo(() => {
    return (blockIndex?.pages || []).flatMap((page) => {
      const pageNumber = Number(page.page) || 1;
      return (page.blocks || [])
        .filter(isTranslatableReadingBlock)
        .map((block) => ({ ...block, page: pageNumber }));
    });
  }, [blockIndex]);

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
    if (!pretranslateProgress.running || !docId || !blockIndex || allTranslatableReadingBlocks.length === 0) {
      return () => {};
    }

    let cancelled = false;
    const pollCachedTranslations = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/documents/${docId}/blocks/translations?target_lang=zh&t=${Date.now()}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (cancelled) return;

        const items = data?.items || {};
        setBlockTranslations((prev) => ({ ...prev, ...items }));
        const done = allTranslatableReadingBlocks.filter((block) => items[block.block_id]).length;
        setPretranslateProgress((prev) => (
          prev.running
            ? { ...prev, done: Math.max(prev.done || 0, Math.min(done, prev.total || allTranslatableReadingBlocks.length)) }
            : prev
        ));
      } catch (error) {
        if (!cancelled) {
          console.warn('[ImmersiveReading] 预翻译进度同步失败', error);
        }
      }
    };

    pollCachedTranslations();
    const timer = window.setInterval(pollCachedTranslations, 2500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [allTranslatableReadingBlocks, blockIndex, docId, pretranslateProgress.running]);

  const currentPageBlocks = useMemo(() => {
    const page = blockIndex?.pages?.find((item) => Number(item.page) === Number(currentPage));
    const blocks = page?.blocks || [];
    return blocks
      .filter((block) => {
        const type = block?.type || 'paragraph';
        const text = String(block?.text || '').trim();
        return text.length > 1 && ['heading', 'paragraph', 'caption', 'figure', 'table'].includes(type);
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

  const handleOutlineJump = useCallback((item) => {
    if (!item) return;
    const targetPage = item.evidence?.primary_page || item.page;
    if (targetPage) {
      setCurrentPage(Number(targetPage));
    }
    const firstBlock = item.first_block || item.evidence?.block_ids?.[0] || item.evidence_block_ids?.[0] || null;
    setActiveReadingNodeId(item.id || null);
    if (item.id) {
      setVisitedReadingNodeIds((prev) => {
        const next = new Set(prev);
        next.add(item.id);
        return next;
      });
    }
    setPinnedReadingBlockId(firstBlock);
    setHoveredReadingBlockId(null);
  }, [setCurrentPage]);

  const handleSectionOutlineJump = useCallback((item) => {
    if (!item) return;
    const firstBlock = resolveSectionOutlineAnchor(item);
    const resolvedBlock = firstBlock ? blockMap[firstBlock] : null;
    const targetPage = resolvedBlock?.page || item.evidence?.primary_page || item.page;
    if (targetPage) {
      setCurrentPage(Number(targetPage));
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
    const blockIds = selectedBlocks.map((block) => block.block_id);

    const chatCredentials = getChatCredentials?.();
    const chatProvider = chatCredentials?.providerId || 'openai';
    const chatModel = chatCredentials?.modelId || 'gpt-4o';
    const chatApiKey = chatCredentials?.apiKey || '';
    const chatProviderFull = getProviderById?.(chatProvider);

    if (getChatCredentials && !chatApiKey && chatProvider !== 'local' && chatProvider !== 'ollama') {
      setBlockTranslateError(`请先为 ${chatProviderFull?.name || chatProvider} 配置 API Key`);
      return null;
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
      const translatedItems = filterValidBlockTranslations(data?.items || {});
      const translatedIds = Object.keys(translatedItems);
      const invalidReturnedIds = Object.keys(data?.items || {}).filter((blockId) => !translatedItems[blockId]);
      setBlockTranslations((prev) => ({ ...prev, ...translatedItems }));
      const failedBlockIds = [
        ...(Array.isArray(data?.failed_block_ids) ? data.failed_block_ids : []),
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
      const normalizedData = { ...(data || {}), items: translatedItems, failed_block_ids: failedBlockIds };
      return options.returnRaw ? normalizedData : translatedItems;
    } catch (error) {
      if (error?.name === 'AbortError') {
        return null;
      }
      setBlockTranslateError(sanitizeTranslationError(error.message));
      return null;
    } finally {
      setTranslatingBlockIds((prev) => {
        const next = new Set(prev);
        blockIds.forEach((blockId) => next.delete(blockId));
        return next;
      });
      if (options.showPanelLoading) {
        setBlockTranslateLoading(false);
      }
    }
  }, [docId, getChatCredentials, getProviderById, pretranslateConcurrency]);

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
    if (!docId || allTranslatableReadingBlocks.length === 0) {
      setPretranslateProgress({ running: false, done: 0, total: 0 });
      setPretranslateNotice('当前文档还没有可缓存的文本块');
      return;
    }

    const { canCallModel, providerName } = getChatRequestConfig();
    if (!canCallModel) {
      setBlockTranslateError(`请先为 ${providerName} 配置 API Key，再开启悬浮预翻译`);
      setPretranslateNotice(`请先为 ${providerName} 配置 API Key`);
      return;
    }

    const translatedIds = new Set(Object.keys(blockTranslations || {}));
    const failedIds = new Set(failedTranslationBlockIds || []);
    const pendingBlocks = allTranslatableReadingBlocks.filter((block) => {
      if (retryFailed) return failedIds.has(block.block_id);
      return force || !translatedIds.has(block.block_id);
    });
    const total = allTranslatableReadingBlocks.length;
    const initialDone = force ? 0 : translatedReadingBlockCount;

    if (pendingBlocks.length === 0) {
      setBlockTranslateError('');
      setPretranslateNotice('全文翻译缓存已是最新');
      setPretranslateProgress({ running: false, done: translatedReadingBlockCount, total });
      return;
    }

    const runId = pretranslateRunRef.current + 1;
    pretranslateRunRef.current = runId;
    pretranslateAbortRef.current?.abort();
    const abortController = new AbortController();
    pretranslateAbortRef.current = abortController;
    setBlockTranslateError('');
    setPretranslateNotice(`正在缓存 ${pendingBlocks.length} 个段落译文`);
    setPretranslateProgress({ running: true, done: initialDone, total });

    const data = await translateReadingBlocks(pendingBlocks, {
      force,
      bulk: true,
      returnRaw: true,
      signal: abortController.signal,
    });

    if (pretranslateRunRef.current === runId) {
      pretranslateAbortRef.current = null;
      if (!data && abortController.signal.aborted) {
        pretranslateStartedDocRef.current = docId;
        setPretranslateProgress({ running: false, done: initialDone, total });
        setBlockTranslateError(`已取消预翻译，已保留 ${initialDone}/${total} 个缓存译文`);
        setPretranslateNotice(`已取消，保留 ${initialDone}/${total} 个缓存译文`);
        return;
      }
      const successCount = Object.keys(data?.items || {}).length;
      const failedCount = Array.isArray(data?.failed_block_ids) ? data.failed_block_ids.length : pendingBlocks.length;
      const done = Math.min(total, force ? successCount : initialDone + successCount);
      setPretranslateProgress({ running: false, done, total });
      if (!data || failedCount > 0) {
        pretranslateStartedDocRef.current = docId;
        setBlockTranslateError(`部分段落暂未翻译成功，已保留 ${done}/${total} 个缓存译文，可稍后继续补齐`);
        setPretranslateNotice(`部分完成：${done}/${total}`);
      } else {
        setPretranslateNotice(`缓存完成：${done}/${total}`);
      }
    }
  }, [
    allTranslatableReadingBlocks,
    blockTranslations,
    docId,
    failedTranslationBlockIds,
    getChatRequestConfig,
    translatedReadingBlockCount,
    translateReadingBlocks,
  ]);

  const cancelPretranslateReadingDocument = useCallback(() => {
    const wasRunning = pretranslateProgress.running;
    pretranslateRunRef.current += 1;
    pretranslateStartedDocRef.current = docId || null;
    pretranslateAbortRef.current?.abort();
    pretranslateAbortRef.current = null;
    setPretranslateProgress((prev) => (
      prev.running ? { ...prev, running: false } : prev
    ));
    setTranslatingBlockIds(new Set());
    if (wasRunning) {
      setBlockTranslateError('已取消预翻译，已完成的译文缓存会保留');
      setPretranslateNotice('已取消，已完成的译文缓存会保留');
    }
  }, [docId, pretranslateProgress.running]);

  const handleStartPretranslate = useCallback((options = {}) => {
    const retryFailed = options.retryFailed ?? failedReadingBlockCount > 0;
    pretranslateReadingDocument({ ...options, retryFailed });
  }, [failedReadingBlockCount, pretranslateReadingDocument]);

  useEffect(() => {
    const wasEnabled = prevShouldAutoPretranslateRef.current;
    prevShouldAutoPretranslateRef.current = shouldAutoPretranslate;
    if (wasEnabled && !shouldAutoPretranslate) {
      cancelPretranslateReadingDocument();
    }
  }, [cancelPretranslateReadingDocument, shouldAutoPretranslate]);

  useEffect(() => {
    if (!shouldAutoPretranslate || !docId || !blockIndex || !blockTranslationsLoaded) return;
    if (pretranslateStartedDocRef.current === docId) return;
    const { canCallModel } = getChatRequestConfig();
    if (!canCallModel) return;

    const hasPendingBlocks = allTranslatableReadingBlocks.some((block) => !blockTranslations[block.block_id]);
    if (!hasPendingBlocks) {
      pretranslateStartedDocRef.current = docId;
      setPretranslateProgress({
        running: false,
        done: allTranslatableReadingBlocks.length,
        total: allTranslatableReadingBlocks.length,
      });
      return;
    }

    pretranslateStartedDocRef.current = docId;
    pretranslateReadingDocument();
  }, [
    allTranslatableReadingBlocks,
    blockIndex,
    blockTranslations,
    blockTranslationsLoaded,
    docId,
    shouldAutoPretranslate,
    getChatRequestConfig,
    pretranslateReadingDocument,
  ]);

  // ========== 截图状态 Hook（需求 1.1） ==========
  // textareaRef 来自 useMessageState（后续初始化），通过代理 ref 桥接
  const textareaRefProxy = useRef(null);
  const screenshotState = useScreenshotState({
    pdfContainerRef,
    textareaRef: textareaRefProxy,
    isVisionCapable,
    setInputValue: (...args) => messageSettersRef.current.setInputValue?.(...args),
    sendMessage: (...args) => messageSettersRef.current.sendMessage?.(...args),
  });
  const {
    screenshots,
    isSelectingArea, setIsSelectingArea,
    handleAreaSelected, handleSelectionCancel,
    handleScreenshotAction, handleScreenshotClose,
  } = screenshotState;

  // ========== 消息状态 Hook（需求 1.2） ==========
  const messageState = useMessageState({
    docId,
    screenshots,
    selectedText,
    getChatCredentials,
    getCurrentChatModel,
    getProviderById,
    streamSpeed,
    enableVectorSearch,
    embeddingApiKey: getEmbeddingApiKey(),
    enableGraphRAG,
    enableAgentRetrieval,
    forceAgentRetrieval,
    enableJiebaBM25,
    numExpandContextChunk,
    enableBlurReveal,
    blurIntensity,
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
    messagesEndRef, textareaRef,
    sendMessage, handleStop,
    regenerateMessage, copyMessage, saveToMemory,
    setInputValue,
    // ref 直写模式：流式输出期间直接更新 DOM
    streamingContentRef,
    streamingThinkingRef,
  } = messageState;

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

  // 文档变更时保存会话
  useEffect(() => {
    if (docId && docInfo) saveCurrentSession(messages);
  }, [docId, docInfo, messages]);

  // ========== GraphRAG 构建 / 状态查询 ==========
  // 切换文档时先重置状态，然后查询是否已有图谱实例在后端内存中。
  // 注意：后端 `_graphrag_instances` 只在进程内存里存活，重启后会丢；磁盘上的
  // `data/graphrag/<doc_id>/` 目录存在但没有自动 reload 逻辑，所以这里只能拿
  // 到「本次进程已构建」的状态。
  useEffect(() => {
    if (!docId) {
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
  }, [docId]);

  const handleBuildGraphRAG = useCallback(async (opts = {}) => {
    if (!docId) { alert('请先上传文档'); return; }
    const { providerId: chatProvider, modelId: chatModel, apiKey: chatApiKey } = getChatCredentials?.() || {};
    if (!chatApiKey && chatProvider !== 'ollama' && chatProvider !== 'local') {
      alert('请先配置对话模型的 API Key');
      return;
    }
    if (!chatModel) {
      alert('请先选择对话模型');
      return;
    }

    setGraphragStatus('building');
    setGraphragError('');

    const chatProviderFull = getProviderById?.(chatProvider);
    const embedConfig = getEmbeddingConfig?.() || {};
    const embedModel = embedConfig.model || chatModel;
    const embedProvider = embedConfig.provider || chatProvider;
    const embedApiKey = embedConfig.apiKey || chatApiKey;
    const embedApiHost = embedConfig.apiHost || '';

    const body = {
      api_key: chatApiKey,
      model: chatModel,
      api_provider: chatProviderFull?.provider || chatProvider,
      api_host: chatProviderFull?.apiHost || '',
      embedding_model: embedModel,
      embedding_api_key: embedApiKey,
      embedding_api_host: embedApiHost,
      force_rebuild: opts.forceRebuild || false,
    };

    // 轮询 progress API
    let pollInterval = null;
    const startPolling = () => {
      pollInterval = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE_URL}/document/${docId}/graphrag/progress`);
          if (!res.ok) return;
          const data = await res.json();
          const prog = data.progress || {};
          setGraphragProgress(prog);
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
  }, [docId, getChatCredentials, getProviderById, getEmbeddingConfig, getEmbeddingApiKey]);

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
  const handleTextSelection = useCallback(() => {
    const selection = window.getSelection();
    const text = selection.toString().trim();
    if (text) {
      setSelectedText(text);
      setShowTextMenu(true);
      if (selection.rangeCount > 0) {
        const range = selection.getRangeAt(0);
        const rect = range.getBoundingClientRect();
        const nextPos = { x: rect.left + rect.width / 2, y: rect.top - 10 };
        setMenuPosition(nextPos);
        setToolbarPosition(nextPos);
      }
    }
  }, [setSelectedText, setShowTextMenu, setMenuPosition]);

  const handleCloseToolbar = useCallback(() => {
    setShowTextMenu(false);
    setSelectedText('');
  }, [setShowTextMenu, setSelectedText]);

  const handleToolbarPositionChange = useCallback((pos) => setToolbarPosition(pos), []);
  const handleToolbarScaleChange = useCallback((scale) => setToolbarScale(scale), [setToolbarScale]);

  // PDFViewer 的文本选择回调（useCallback 稳定引用，避免 PDFViewer 不必要重渲染）
  const handlePdfTextSelect = useCallback((text) => {
    if (text) {
      setSelectedText(text);
      setShowTextMenu(true);
      const selection = window.getSelection();
      if (selection.rangeCount > 0) {
        const range = selection.getRangeAt(0);
        const rect = range.getBoundingClientRect();
        const nextPos = { x: rect.left + rect.width / 2, y: rect.top - 10 };
        setMenuPosition(nextPos);
        setToolbarPosition(nextPos);
      }
    }
  }, [setSelectedText, setShowTextMenu, setMenuPosition]);

  // ModelQuickSwitch 的思考模式切换回调（useCallback 稳定引用）
  const handleThinkingChange = useCallback((enabled) => {
    setEnableThinking(enabled);
  }, [setEnableThinking]);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(selectedText).then(() => {
      alert('✅ 已复制到剪贴板');
    });
  }, [selectedText]);

  const handleHighlight = useCallback(() => {
    const highlights = JSON.parse(localStorage.getItem(`highlights_${docId}`) || '[]');
    const newHighlight = {
      text: selectedText, page: currentPage,
      timestamp: Date.now(), color: '#fef08a',
    };
    highlights.push(newHighlight);
    localStorage.setItem(`highlights_${docId}`, JSON.stringify(highlights));
    alert('✅ 已添加高亮标注');
  }, [docId, selectedText, currentPage]);

  const handleAddNote = useCallback(() => {
    const note = prompt('请输入您的笔记：', '');
    if (note) {
      const notes = JSON.parse(localStorage.getItem(`notes_${docId}`) || '[]');
      notes.push({
        text: selectedText, note, page: currentPage,
        timestamp: Date.now(),
      });
      localStorage.setItem(`notes_${docId}`, JSON.stringify(notes));
      alert('✅ 笔记已保存');
    }
  }, [docId, selectedText, currentPage]);

  const handleAIExplain = useCallback(() => {
    setInputValue(`请解释这段话：\n\n"${selectedText}"`);
    setShowTextMenu(false);
    setTimeout(() => sendMessage(), 100);
  }, [selectedText, setInputValue, sendMessage, setShowTextMenu]);

  const handleTranslate = useCallback(() => {
    setInputValue(`请将以下内容翻译成中文：\n\n"${selectedText}"`);
    setShowTextMenu(false);
    setTimeout(() => sendMessage(), 100);
  }, [selectedText, setInputValue, sendMessage, setShowTextMenu]);

  const handleWebSearch = useCallback(() => {
    const q = encodeURIComponent(selectedText);
    const searchTemplates = {
      google: `https://www.google.com/search?q=${q}`,
      bing: `https://www.bing.com/search?q=${q}`,
      baidu: `https://www.baidu.com/s?wd=${q}`,
      sogou: `https://www.sogou.com/web?query=${q}`,
      custom: searchEngineUrl.includes('{query}')
        ? searchEngineUrl.replace('{query}', q)
        : `${searchEngineUrl}?q=${q}`,
    };
    window.open(searchTemplates[searchEngine] || searchTemplates.google, '_blank');
  }, [selectedText, searchEngine, searchEngineUrl]);

  const handleShare = useCallback(() => {
    const shareText = `📄 来自《${docInfo?.filename || '文档'}》第 ${currentPage} 页：\n\n"${selectedText}"\n\n--- ChatPDF Pro ---`;
    navigator.clipboard.writeText(shareText).then(() => {
      alert('✅ 引用卡片已复制到剪贴板，可直接粘贴分享');
    });
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
    if (!window.confirm('只清理当前文档的 AI 辅助缓存，不会删除原始 PDF、向量索引或对话历史。确定继续吗？')) {
      return;
    }
    try {
      const res = await fetch(`${API_BASE_URL}/documents/${docId}/ai-cache`, { method: 'DELETE' });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({ detail: '清理缓存失败' }));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }
      setReadingOutline(null);
      setReadingOutlineError('');
      setReadingOutlineReloadKey((prev) => prev + 1);
      setSectionOutline(null);
      setSectionOutlineError('');
      setSectionOutlineReloadKey((prev) => prev + 1);
      setBlockTranslations({});
      setFailedTranslationBlockIds(new Set());
      setBlockTranslateError('');
      setBlockTranslationsLoaded(false);
      setPretranslateProgress({ running: false, done: 0, total: allTranslatableReadingBlocks.length });
      setPretranslateNotice(allTranslatableReadingBlocks.length > 0 ? '缓存已清理，可重新开始全文缓存' : '');
      pretranslateRunRef.current += 1;
      pretranslateStartedDocRef.current = null;
      clearOverviewCache?.(docId);
      setShowAiProcessingPanel(false);
      alert('当前文档 AI 缓存已清理');
    } catch (error) {
      alert(error.message || '清理缓存失败');
    }
  }, [allTranslatableReadingBlocks.length, clearOverviewCache, docId]);

  const handleRegenerateOverview = useCallback(() => {
    if (!docId) return;
    fetchOverview?.(overviewDepth, { force: true }).catch(() => {});
    setRightPanelMode('overview');
  }, [docId, fetchOverview, overviewDepth, setRightPanelMode]);

  // ========== 预设问题（useMemo 缓存计算结果） ==========
  const showPresetQuestions = useMemo(() => docId && messages.filter(
    msg => msg.type === 'user' || msg.type === 'assistant'
  ).length === 0, [docId, messages]);

  const handlePresetSelect = useCallback((query) => {
    setInputValue(query);
    requestAnimationFrame(() => sendMessage());
  }, [setInputValue, sendMessage]);

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
  const handleEmbeddingSettingsClose = useCallback(() => { setShowEmbeddingSettings(false); setSettingsSection('common'); setShowSettings(true); }, [setShowEmbeddingSettings, setShowSettings]);
  const handleGlobalSettingsClose = useCallback(() => { setShowGlobalSettings(false); setSettingsSection('common'); setShowSettings(true); }, [setShowGlobalSettings, setShowSettings]);
  const handleChatSettingsClose = useCallback(() => { setShowChatSettings(false); setSettingsSection('common'); setShowSettings(true); }, [setShowChatSettings, setShowSettings]);
  const handleOCRSettingsClose = useCallback(() => { setShowOCRSettings(false); setSettingsSection('common'); setShowSettings(true); }, [setShowOCRSettings, setShowSettings]);

  // ========== 根容器点击回调（useCallback 稳定引用） ==========
  const handleRootClick = useCallback((e) => {
    if (!showTextMenu) return;
    const selection = window.getSelection();
    const hasActiveSelection = selection && selection.toString().trim().length > 0;
    if (hasActiveSelection) return;
    if (!e.target.closest('.text-selection-toolbar-container')) {
      handleCloseToolbar();
    }
  }, [showTextMenu, handleCloseToolbar]);

  // ========== 虚拟消息列表渲染回调（useCallback 稳定引用） ==========
  const renderMessage = useCallback((msg, idx) => {
    const hasThinking = typeof msg.thinking === 'string' && msg.thinking.trim().length > 0;
    const hasAgentTrace = msg.type === 'assistant' && msg.agentTrace && msg.agentTrace.enabled;
    const isStreamingCurrentMessage = shouldStreamAssistantContent(msg, streamingMessageId);
    // 只要当前消息还在生成，且正文还没开始出现，就先展示思考/生成阶段块。
    // 这样即使 reasoningEffort 关闭，用户也不会只看到三个等待点。
    const shouldShowThinking = hasThinking || hasAgentTrace || (
      isStreamingCurrentMessage && (
        reasoningEffort !== 'off'
        || !msg.content
        || !msg.content.trim()
      )
    );
    const shouldStreamContent = isStreamingCurrentMessage;
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className={`flex flex-col ${msg.type === 'user' ? 'items-end' : 'items-start'}`}
        style={{ fontSize: `${messageFontSize}px` }}
      >
        <div className={`${msg.type === 'user'
          ? messageStyle === 'bubble'
            ? 'max-w-[85%] rounded-2xl px-4 py-3 message-bubble-user rounded-tr-sm text-sm'
            : 'max-w-[85%] rounded-2xl px-4 py-3 message-bubble-user rounded-tr-sm text-sm'
          : messageStyle === 'bubble'
            ? 'max-w-[90%] min-w-0 bg-gray-50 dark:bg-gray-800/50 rounded-2xl rounded-tl-sm px-4 py-3 text-gray-800 dark:text-gray-50 overflow-hidden shadow-sm'
            : 'w-full max-w-full min-w-0 bg-transparent shadow-none p-0 text-gray-800 dark:text-gray-50 overflow-hidden'
        }`}
          style={msg.type !== 'user' && messageStyle !== 'bubble' ? { contain: 'inline-size' } : undefined}
        >
          {msg.type === 'assistant' && (
            <div className="flex items-center gap-2 mb-2 select-none">
              <span className="px-3 py-1 bg-[#eeeffe] text-[#4f46e5] dark:bg-[#4f46e5]/20 dark:text-[#a5b4fc] rounded-[24px] text-[11px] font-bold uppercase tracking-wider">
                {msg.model || 'ASSISTANT'}
              </span>
            </div>
          )}
          {shouldShowThinking && (
            <ThinkingBlock
              content={msg.thinking}
              isStreaming={isStreamingCurrentMessage}
              darkMode={darkMode}
              thinkingMs={msg.thinkingMs || 0}
              streamingRef={isStreamingCurrentMessage ? streamingThinkingRef : undefined}
              agentTrace={msg.agentTrace || null}
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
          {msg.answerCritic && msg.answerCritic.has_hallucination && !msg.isStreaming && (
            <div className="mb-2 px-3 py-2 rounded-lg bg-orange-50 border border-orange-200 text-orange-700 text-xs flex items-start gap-1.5">
              <svg className="w-4 h-4 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" /></svg>
              <div className="flex-1">
                <div className="font-medium">答案自审检测到潜在幻觉</div>
                {msg.answerCritic.reason && (
                  <div className="mt-0.5 text-orange-600/80">{msg.answerCritic.reason}</div>
                )}
                {typeof msg.answerCritic.confidence === 'number' && (
                  <div className="mt-0.5 text-orange-600/70">自审置信度: {(msg.answerCritic.confidence * 100).toFixed(0)}%</div>
                )}
              </div>
            </div>
          )}
          <StreamingMarkdown
            content={msg.content}
            isStreaming={shouldStreamContent}
            enableBlurReveal={enableBlurReveal}
            blurIntensity={blurIntensity}
            citations={msg.citations || null}
            onCitationClick={(c) => { setActiveCitationRef(c?.ref ?? null); handleCitationClick(c); }}
            streamingRef={shouldStreamContent ? streamingContentRef : undefined}
            webSearchSources={msg.webSearchSources || null}
            suppressInitialDots={shouldShowThinking && isStreamingCurrentMessage && (!msg.content || !msg.content.trim())}
          />
          {/* 联网搜索来源 */}
          {msg.webSearchSources && msg.webSearchSources.length > 0 && !msg.isStreaming && (
            <WebSearchSourcesBadge sources={msg.webSearchSources} />
          )}
          {msg.type === 'assistant' && !msg.isStreaming && msg.memoryHits && msg.memoryHits.length > 0 && (
            <MemoryHitsBadge hits={msg.memoryHits} meta={msg.memoryMeta} />
          )}
        </div>
        {msg.type === 'assistant' && !msg.isStreaming && msg.visualVerification && (
          <TableVisualVerificationStatus verification={msg.visualVerification} />
        )}
        {/* 证据面板 */}
        {msg.type === 'assistant' && !msg.isStreaming && msg.citations && msg.citations.length > 0 && (
          <EvidencePanel
            citations={msg.citations}
            docId={docId}
            onCitationClick={(c) => { setActiveCitationRef(c?.ref ?? null); handleCitationClick(c); }}
            activeRef={activeCitationRef}
            onRefHover={setActiveCitationRef}
          />
        )}
        {/* 思维导图 */}
        {msg.type === 'assistant' && !msg.isStreaming && msg.mindmapMarkdown && (
          <MindmapView markdown={msg.mindmapMarkdown} />
        )}
        {/* 消息操作按钮 */}
        {msg.type === 'assistant' && !msg.isStreaming && (
          <div className="flex items-center gap-1 mt-1 ml-2">
            <button onClick={() => copyMessage(msg.content, msg.id || idx)} className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-500 hover:text-gray-700 transition-colors" title="复制">
              {copiedMessageId === (msg.id || idx) ? (
                <svg className="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>
              ) : (<Copy className="w-4 h-4" />)}
            </button>
            <button onClick={() => { if (!confirmRegenerateMessage || confirm('确定要重新生成这条回答吗？')) regenerateMessage(idx); }} className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-500 hover:text-gray-700 transition-colors" title="重新生成">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>
            </button>
            <button onClick={() => saveToMemory(idx, 'liked')} className={`p-1.5 rounded-lg hover:bg-gray-100 transition-colors ${likedMessages.has(idx) ? 'text-pink-500' : 'text-gray-500 hover:text-gray-700'}`} title="点赞并记忆">
              <svg className="w-4 h-4" fill={likedMessages.has(idx) ? 'currentColor' : 'none'} stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" /></svg>
            </button>
            <button onClick={() => setFeedbackTarget({ idx, msg })} className={`p-1.5 rounded-lg hover:bg-gray-100 transition-colors ${dislikedMessages.has(idx) ? 'text-orange-500' : 'text-gray-500 hover:text-gray-700'}`} title="点踩并反馈">
              <svg className="w-4 h-4" fill={dislikedMessages.has(idx) ? 'currentColor' : 'none'} stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018a2 2 0 01.485.06l3.76.94m-7 10v5a2 2 0 002 2h.096c.5 0 .905-.405.905-.904 0-.715.211-1.413.608-2.008L17 13V4m-7 10h2m5-10h2a2 2 0 012 2v6a2 2 0 01-2 2h-2.5" /></svg>
            </button>
            <button onClick={() => saveToMemory(idx, 'manual')} className={`p-1.5 rounded-lg hover:bg-gray-100 transition-colors ${rememberedMessages.has(idx) ? 'text-purple-500' : 'text-gray-500 hover:text-gray-700'}`} title="记住这个">
              <Brain className={`w-4 h-4 ${rememberedMessages.has(idx) ? 'fill-current' : ''}`} />
            </button>
            {msg.qaScore != null && (
              <span className={`ml-1 text-[10px] px-1.5 py-0.5 rounded-full font-medium ${msg.qaScore >= 0.7 ? 'bg-green-50 text-green-600' : msg.qaScore >= 0.4 ? 'bg-yellow-50 text-yellow-600' : 'bg-red-50 text-red-600'}`} title={`回答置信度: ${(msg.qaScore * 100).toFixed(0)}%`}>
                {(msg.qaScore * 100).toFixed(0)}%
              </span>
            )}
          </div>
        )}
        {/* 动态追问建议 */}
        {msg.type === 'assistant' && !msg.isStreaming && msg.followupQuestions && msg.followupQuestions.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-2 ml-2">
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
                className="text-xs px-2.5 py-1.5 rounded-full border border-blue-200 bg-blue-50 text-blue-600 hover:bg-blue-100 hover:border-blue-300 transition-colors cursor-pointer"
              >
                {q}
              </button>
            ))}
          </div>
        )}
      </motion.div>
    );
  }, [
    streamingMessageId, darkMode, enableBlurReveal, blurIntensity,
    streamingThinkingRef, streamingContentRef, copiedMessageId,
    likedMessages, rememberedMessages,
    handleCitationClick, copyMessage, regenerateMessage, saveToMemory,
    messageStyle, messageFontSize, confirmRegenerateMessage, reasoningEffort,
    activeCitationRef, setActiveCitationRef,
    dislikedMessages, setFeedbackTarget,
    docId,
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
    { id: 'analysis', label: '解析' },
    { id: 'chat', label: '对话' },
  ];
  const activeRightPanelTabIndex = Math.max(0, rightPanelTabs.findIndex((item) => item.id === rightPanelMode));
  const deepParseRunning = ['queued', 'running'].includes(deepParseStatus?.status);
  const aiProcessingRunning = readingOutlineLoading || sectionOutlineLoading || overviewLoading || pretranslateProgress.running || deepParseRunning;
  const aiProcessingStatusText = aiAutoProcess
    ? (autoOutlineSummary || enableHoverPretranslate ? '自动处理开启' : '按需生成')
    : '自动处理已关闭';
  const aiProcessingItems = useMemo(() => {
    const summaryStatus = readingOutlineLoading
      ? '生成中'
      : readingOutlineItems.length > 0
        ? (readingOutline?.source === 'fallback' ? '基础结果' : '已完成')
        : (aiAutoProcess && autoOutlineSummary ? '等待生成' : '自动已关');
    const outlineStatus = sectionOutlineLoading
      ? '生成中'
      : sectionOutlineItems.length > 0
        ? (sectionOutline?.source === 'fallback' || sectionOutline?.source === 'heuristic' ? '基础大纲' : '已完成')
        : (aiAutoProcess && autoOutlineSummary ? '等待生成' : '自动已关');
    const overviewStatus = overviewLoading
      ? '生成中'
      : overview
        ? '已完成'
        : '按需生成';
    const translationStatus = pretranslateProgress.running
      ? `${Math.min(pretranslateProgress.done, pretranslateProgress.total)}/${pretranslateProgress.total || allTranslatableReadingBlocks.length}`
      : failedReadingBlockCount > 0
        ? `失败 ${failedReadingBlockCount}`
        : translatedReadingBlockCount > 0
          ? `${translatedReadingBlockCount}/${allTranslatableReadingBlocks.length}`
          : (aiAutoProcess && enableHoverPretranslate ? '等待缓存' : '未开始');
    const deepParseStatusText = deepParseRunning
      ? (deepParseStatus?.poll_attempt && deepParseStatus?.poll_total
        ? `等待处理 ${deepParseStatus.poll_attempt}/${deepParseStatus.poll_total}`
        : deepParseStatus?.stage === 'requesting_upload'
          ? '申请上传中'
          : deepParseStatus?.stage === 'uploading'
            ? '上传中'
            : deepParseStatus?.stage === 'polling'
              ? '等待处理'
              : deepParseStatus?.stage === 'downloading'
                ? '下载结果中'
                : deepParseStatus?.stage === 'building_index'
                  ? '重建索引中'
                  : '解析中')
      : deepParseStatus?.status === 'ready' && deepParseStatus?.active_mineru
        ? '已完成'
      : deepParseStatus?.status === 'failed'
        ? '失败'
        : deepParseStatus?.status === 'cancelled'
          ? '已取消'
        : deepParseStatus?.configured === false
          ? '未配置'
          : '按需解析';

    const deepParseRecommended = Boolean(
      !deepParseStatus?.active_mineru && !deepParseRunning && deepParseStatus?.recommend_deep_parse
    );
    const ragIndexSource = ragIndexStatus?.index_source || (ragIndexStatus?.ready ? 'pdf_native' : '');
    const ragIndexIsMinerU = ragIndexSource === 'mineru';
    const ragIndexRecommended = Boolean(deepParseStatus?.recommend_rag_index_rebuild && !ragIndexBusy);
    const ragIndexStatusText = ragIndexBusy
      ? '处理中'
      : !ragIndexStatus?.ready
        ? '未就绪'
        : ragIndexIsMinerU
          ? 'MinerU'
          : ragIndexRecommended
            ? '建议重建'
            : '本地';
    const ragIndexDesc = !deepParseStatus?.active_mineru
      ? '先完成 MinerU 深度解析后，才能把问答索引升级为同源结构化结果'
      : ragIndexIsMinerU
        ? `问答检索已使用 MinerU 结构化结果${ragIndexStatus?.table_chunk_count ? `，${ragIndexStatus.table_chunk_count} 个结构化表格块` : ''}`
        : deepParseStatus?.recommend_rag_index_reason
          || '当前问答仍使用本地 PDF 解析索引，建议重建为 MinerU 结构化问答索引以改善表格和双栏问答';

    return [
      {
        id: 'deep_parse',
        title: 'MinerU 深度解析',
        recommended: deepParseRecommended,
        desc: deepParseStatus?.active_mineru
          ? `当前阅读结构、大纲与速览图表均来自 MinerU${deepParseStatus?.block_count ? `，${deepParseStatus.block_count} 个块` : ''}${deepParseStatus?.figure_count ? `，${deepParseStatus.figure_count} 个图表` : ''}`
          : deepParseRunning && deepParseStatus?.message
            ? deepParseStatus.message
            : deepParseStatus?.status === 'failed' && deepParseStatus?.error
              ? deepParseStatus.error
              : deepParseRecommended
                ? deepParseStatus.recommend_reason
                : `用 MinerU 重建带坐标的阅读块、大纲和速览图表，手动触发才会上传 PDF${deepParseStatus?.access_mode === 'direct' ? '到官方 API' : '到 Worker'}`,
        status: deepParseStatusText,
        busy: deepParseRunning,
        actionLabel: deepParseRunning
          ? '取消'
          : deepParseStatus?.active_mineru
            ? '重新解析'
            : '开始解析',
        onAction: deepParseRunning ? handleCancelMinerUDeepParse : handleStartMinerUDeepParse,
        disabled: !docId || deepParseStatus?.configured === false,
      },
      {
        id: 'rag_index',
        title: '问答索引',
        recommended: ragIndexRecommended,
        desc: ragIndexDesc,
        status: ragIndexStatusText,
        busy: ragIndexBusy,
        actionLabel: ragIndexBusy
          ? '处理中'
          : ragIndexIsMinerU
            ? '回退'
            : deepParseStatus?.active_mineru
              ? '重建'
              : '需解析',
        onAction: ragIndexIsMinerU ? handleRollbackRagIndex : handleRebuildMinerURagIndex,
        disabled: !docId || ragIndexBusy || (!deepParseStatus?.active_mineru && !ragIndexIsMinerU) || (ragIndexIsMinerU && !ragIndexStatus?.can_rollback),
      },
      {
        id: 'summary',
        title: 'AI 总结',
        desc: '左侧总结栏的结构化论文梳理',
        status: summaryStatus,
        busy: readingOutlineLoading,
        actionLabel: readingOutlineLoading ? '生成中' : (readingOutlineItems.length > 0 ? '重新生成' : '立即生成'),
        onAction: handleRegenerateReadingOutline,
        disabled: readingOutlineLoading || !docId,
      },
      {
        id: 'outline',
        title: '章节大纲',
        desc: '左侧大纲栏的原文章节树',
        status: outlineStatus,
        busy: sectionOutlineLoading,
        actionLabel: sectionOutlineLoading ? '生成中' : (sectionOutlineItems.length > 0 ? '重新生成' : '立即生成'),
        onAction: handleRegenerateSectionOutline,
        disabled: sectionOutlineLoading || !docId,
      },
      {
        id: 'overview',
        title: '速览',
        desc: `当前默认详细度：${overviewDepth === 'brief' ? '简略' : overviewDepth === 'detailed' ? '详细' : '标准'}`,
        status: overviewStatus,
        busy: overviewLoading,
        actionLabel: overviewLoading ? '生成中' : (overview ? '重新生成' : '生成速览'),
        onAction: handleRegenerateOverview,
        disabled: overviewLoading || !docId,
      },
      {
        id: 'translation',
        title: '悬浮翻译',
        desc: '预缓存段落翻译，悬浮时直接显示',
        status: translationStatus,
        busy: pretranslateProgress.running,
        actionLabel: pretranslateProgress.running
          ? '取消'
          : failedReadingBlockCount > 0
            ? '补齐失败'
            : translatedReadingBlockCount > 0
              ? '补齐全文'
              : '开始缓存',
        onAction: pretranslateProgress.running
          ? cancelPretranslateReadingDocument
          : handleStartPretranslate,
        disabled: !docId || allTranslatableReadingBlocks.length === 0,
      },
    ];
  }, [
    aiAutoProcess,
    allTranslatableReadingBlocks.length,
    autoOutlineSummary,
    cancelPretranslateReadingDocument,
    docId,
    deepParseRunning,
    deepParseStatus?.active_mineru,
    deepParseStatus?.access_mode,
    deepParseStatus?.recommend_deep_parse,
    deepParseStatus?.recommend_rag_index_rebuild,
    deepParseStatus?.recommend_rag_index_reason,
    deepParseStatus?.recommend_reason,
    deepParseStatus?.block_count,
    deepParseStatus?.figure_count,
    deepParseStatus?.configured,
    deepParseStatus?.stage,
    deepParseStatus?.status,
    enableHoverPretranslate,
    failedReadingBlockCount,
    handleRebuildMinerURagIndex,
    handleRollbackRagIndex,
    handleStartPretranslate,
    handleRegenerateOverview,
    handleRegenerateReadingOutline,
    handleRegenerateSectionOutline,
    handleCancelMinerUDeepParse,
    handleStartMinerUDeepParse,
    overview,
    overviewDepth,
    overviewLoading,
    pretranslateProgress.done,
    pretranslateProgress.running,
    pretranslateProgress.total,
    ragIndexBusy,
    ragIndexStatus?.can_rollback,
    ragIndexStatus?.index_source,
    ragIndexStatus?.ready,
    ragIndexStatus?.table_chunk_count,
    readingOutline?.source,
    readingOutlineItems.length,
    readingOutlineLoading,
    sectionOutline?.source,
    sectionOutlineItems.length,
    sectionOutlineLoading,
    translatedReadingBlockCount,
  ]);
  const hasAiRecommendation = aiProcessingItems.some((item) => item.recommended);

  const uploadStatusMeta = UPLOAD_STATUS_META[uploadStatus] || UPLOAD_STATUS_META.uploading;
  const uploadProgressLabel = Math.max(0, Math.min(100, Math.round(Number(uploadProgress) || 0)));
  const deepParsePanelNotice = deepParseNotice || (
    deepParseStatus?.status === 'ready' && deepParseStatus?.active_mineru
      ? 'MinerU 深度解析已完成，阅读结构、大纲与速览图表均已切换为 MinerU'
      : ''
  );

  // ========== 渲染 ==========
  return (
    <div
      className={`chatpdf-shell ${docId ? 'chatpdf-shell--document' : 'chatpdf-shell--welcome'} ${darkMode ? 'chatpdf-shell--dark' : ''} h-screen w-full flex overflow-hidden transition-colors duration-300 ${darkMode ? 'bg-[#0f1115] text-gray-200' : 'text-[var(--color-text-main)]'}`}
      onClick={handleRootClick}
    >
      {/* 划词工具栏 */}
      {showTextMenu && selectedText && (
        <div className="text-selection-toolbar-container">
          <TextSelectionToolbar
            selectedText={selectedText}
            position={toolbarPosition.x === 0 && toolbarPosition.y === 0 ? menuPosition : toolbarPosition}
            onPositionChange={handleToolbarPositionChange}
            scale={toolbarScale}
            onScaleChange={handleToolbarScaleChange}
            onClose={handleCloseToolbar}
            onCopy={handleCopy}
            onHighlight={handleHighlight}
            onAddNote={handleAddNote}
            onAIExplain={handleAIExplain}
            onTranslate={handleTranslate}
            onWebSearch={handleWebSearch}
            onShare={handleShare}
            size={toolbarSize}
          />
        </div>
      )}

      {/* 侧边栏（历史记录） */}
      <motion.div
        initial={false}
        animate={{ width: showSidebar ? 320 : 0, opacity: showSidebar ? 1 : 0 }}
        transition={{ duration: 0.2, ease: "easeInOut" }}
        style={{ pointerEvents: showSidebar ? 'auto' : 'none' }}
        className={`flex-shrink-0 m-6 mr-0 h-[calc(100vh-3rem)] flex flex-col z-20 overflow-hidden ${darkMode ? 'bg-[#1a1d21]/90 border-white/5 backdrop-blur-3xl backdrop-saturate-150 rounded-[40px]' : 'bg-white/80 backdrop-blur-2xl border border-white/70 rounded-[40px] shadow-2xl shadow-gray-300/40 shadow-[inset_0_1px_1px_rgba(255,255,255,0.8)]'}`}
      >
        <div className="w-[320px] mx-auto flex flex-col h-full items-stretch relative">
          <div className="px-8 py-8 flex items-center justify-between mb-2">
            <div className="flex items-center gap-3 font-bold text-2xl text-purple-600 tracking-tight pl-2">
              <Bot className="w-9 h-9" />
              <span>ChatPDF</span>
            </div>
            <div className="flex items-center gap-1">
              <button onClick={() => setDarkMode(!darkMode)} className={`p-2 rounded-full transition-colors ${darkMode ? 'hover:bg-white/10 text-gray-400 hover:text-yellow-400' : 'hover:bg-black/5'}`}>
                {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
              </button>
            </div>
          </div>

          <div className="px-5 mb-4 flex justify-center">
            <button
              onClick={() => { startNewChat(); fileInputRef.current?.click(); }}
              className="tanya-btn max-w-[260px]"
            >
              <Plus className="w-5 h-5 opacity-70" />
              <span>上传文件/新对话</span>
            </button>
            <input ref={fileInputRef} type="file" accept=".pdf" onChange={handleFileUpload} className="hidden" />
          </div>

          <div className="px-6 mb-4">
            <div className={`relative grid grid-cols-3 overflow-hidden rounded-[22px] p-1 border shadow-[inset_0_1px_0_rgba(255,255,255,0.85)] backdrop-blur-xl ${
              darkMode
                ? 'bg-white/[0.05] border-white/10'
                : 'bg-white/35 border-white/70 shadow-[inset_0_1px_1px_rgba(255,255,255,0.9),0_8px_24px_rgba(148,163,184,0.10)]'
            }`}>
              <motion.div
                className={`absolute top-1 bottom-1 left-1 rounded-[18px] ${
                  darkMode
                    ? 'bg-white/12 shadow-[0_8px_22px_rgba(0,0,0,0.22)]'
                    : 'bg-white/90 shadow-[0_10px_24px_rgba(139,124,200,0.14),0_2px_6px_rgba(31,41,55,0.06),inset_0_1px_0_rgba(255,255,255,0.95)]'
                }`}
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
                    className={`relative z-10 flex h-10 items-center justify-center gap-2 rounded-[18px] text-[13px] font-semibold transition-colors duration-200 ${
                      isActive
                        ? (darkMode ? 'text-white' : 'text-[#8b7cc8]')
                        : (darkMode ? 'text-gray-500 hover:text-gray-300' : 'text-gray-400 hover:text-gray-600')
                    }`}
                  >
                    <Icon className={`h-4 w-4 transition-all duration-200 ${isActive ? 'scale-105' : 'scale-100 opacity-80'}`} />
                    <span>{label}</span>
                  </motion.button>
                );
              })}
            </div>
          </div>

          {sidebarMode === 'history' ? (
            <div className="flex-1 overflow-y-auto px-8">
              <h2 className="text-[11px] font-bold text-gray-500 tracking-wider mb-4 pl-4">
                会话历史
              </h2>
              <ul className="space-y-2 relative">
                {history.map((item, idx) => {
                  const isActive = item.id === docId;
                  return (
                    <li key={idx}>
                      <div
                        onClick={() => loadSession(item)}
                        className={`w-full flex items-center justify-between px-5 py-4 rounded-3xl cursor-pointer group transition-all duration-300 ${
                          isActive
                            ? (darkMode ? 'bg-white/10 backdrop-blur-md border border-white/10 text-white font-bold shadow-[0_10px_25px_rgba(0,0,0,0.2)]' : 'bg-white/80 backdrop-blur-md border border-white/80 text-gray-900 font-bold shadow-[0_10px_25px_rgba(0,0,0,0.06)]')
                            : (darkMode ? 'text-gray-400 font-medium hover:bg-white/5' : 'text-gray-600 font-medium hover:bg-white/40')
                        }`}
                      >
                        <div className="flex items-center gap-4 overflow-hidden">
                          <MessageSquare
                            size={22}
                            className={`${isActive ? 'text-[#9333ea]' : 'text-gray-500'} transition-colors flex-shrink-0`}
                            strokeWidth={isActive ? 2.5 : 2}
                          />
                          <span className="text-[15px] truncate">{item.filename}</span>
                        </div>
                        <button
                          onClick={(e) => { e.stopPropagation(); if (!confirmDeleteMessage || confirm('确定要删除这条对话记录吗？')) deleteSession(item.id); }}
                          className={`opacity-0 group-hover:opacity-100 p-1.5 rounded-full hover:bg-red-50 hover:text-red-500 transition-all flex-shrink-0 ${isActive ? 'text-gray-400' : 'text-gray-400'}`}
                        >
                          <Trash2 size={18} strokeWidth={2} />
                        </button>
                      </div>
                    </li>
                  );
                })}
              </ul>
            </div>
          ) : sidebarMode === 'summary' ? (
            <div className="flex-1 min-h-0 px-5 overflow-hidden">
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
                darkMode={darkMode}
              />
            </div>
          ) : (
            <div className="flex-1 min-h-0 px-5 overflow-hidden">
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

          <div className="px-8 py-6">
            <button 
              onClick={openSettings}
              className={`w-full flex items-center gap-4 px-5 py-4 rounded-3xl transition-all duration-300 ${darkMode ? 'text-gray-400 font-medium hover:bg-white/5' : 'text-gray-600 font-medium hover:bg-white/40'}`}
            >
              <Settings size={22} className="text-gray-500" strokeWidth={2} />
              <span className="text-[15px]">设置中心</span>
            </button>
          </div>
        </div>
      </motion.div>

      {/* 主内容区域 */}
      <div className="flex-1 flex flex-col h-full relative transition-all duration-200 ease-in-out">
        {/* 侧边栏展开按钮 (未打开文档时显示) */}
        {!showSidebar && !docId && (
          <button
            onClick={() => setShowSidebar(true)}
            className={`absolute top-4 left-4 z-20 p-2 backdrop-blur-md shadow-sm rounded-full hover:scale-105 transition-all border ${darkMode ? 'bg-white/10 text-gray-300 border-white/10 hover:bg-white/20' : 'bg-white/80 text-gray-700 border-white/50 hover:bg-white'}`}
            title="显示侧边栏"
          >
            <Menu className="w-5 h-5" />
          </button>
        )}

        {/* 内容区域 */}
        <div className="flex-1 flex overflow-hidden px-8 pb-6 gap-4 pt-6">
          {/* 左侧：PDF 预览 */}
          {docId ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className={`soft-panel overflow-hidden flex flex-col relative flex-shrink-0 rounded-[var(--radius-panel)] min-w-0 ${darkMode ? 'bg-gray-800/50' : ''}`}
              style={{ width: `${pdfPanelWidth}%`, minWidth: '350px' }}
            >
              <div className="flex-1 overflow-hidden">
                {docInfo?.pdf_url ? (
                  <PDFViewer
                    ref={pdfContainerRef}
                    pdfUrl={docInfo.pdf_url}
                    page={currentPage}
                    onPageChange={setCurrentPage}
                    highlightInfo={activeHighlight}
                    isSelecting={isSelectingArea}
                    onAreaSelected={handleAreaSelected}
                    onSelectionCancel={handleSelectionCancel}
                    darkMode={darkMode}
                    onTextSelect={handlePdfTextSelect}
                    onToggleSidebar={() => setShowSidebar(prev => !prev)}
                    blockIndex={blockIndex}
                    activeBlockId={activeReadingBlockId}
                    focusedBlockIds={focusedReadingBlockIds}
                    visitedBlockIds={[]}
                    inlineTranslationBlockIds={[]}
                    onBlockHover={handleReadingBlockHover}
                    onBlockClick={handleReadingBlockClick}
                    blockTranslations={inlineBlockTranslations}
                    translatingBlockIds={[...translatingBlockIds]}
                  />
                ) : (docInfo?.pages || docInfo?.data?.pages) ? (
                  <>
                    <div className="h-14 border-b border-black/5 flex items-center justify-between px-6 bg-white/30 backdrop-blur-sm">
                      <div className="flex items-center gap-2">
                        <button onClick={() => setCurrentPage(Math.max(1, currentPage - 1))} className="p-1.5 hover:bg-black/5 rounded-lg"><ChevronLeft className="w-5 h-5" /></button>
                        <span className="text-sm font-medium w-16 text-center">{currentPage} / {docInfo?.total_pages || docInfo?.data?.total_pages || 1}</span>
                        <button onClick={() => setCurrentPage(Math.min(docInfo?.total_pages || docInfo?.data?.total_pages || 1, currentPage + 1))} className="p-1.5 hover:bg-black/5 rounded-lg"><ChevronRight className="w-5 h-5" /></button>
                      </div>
                      <div className="flex items-center gap-2">
                        <button onClick={() => setPdfScale(s => Math.max(0.5, s - 0.1))} className="p-1.5 hover:bg-black/5 rounded-lg"><ZoomOut className="w-5 h-5" /></button>
                        <span className="text-sm font-medium w-12 text-center">{Math.round(pdfScale * 100)}%</span>
                        <button onClick={() => setPdfScale(s => Math.min(2.0, s + 0.1))} className="p-1.5 hover:bg-black/5 rounded-lg"><ZoomIn className="w-5 h-5" /></button>
                      </div>
                    </div>
                    <div ref={pdfContainerRef} className="h-full overflow-auto bg-gray-50/50">
                      <div className="min-h-full flex items-start justify-center p-8" style={{ zoom: pdfScale }}>
                        <div className="bg-white shadow-2xl p-12 rounded-lg max-w-4xl w-full" onMouseUp={handleTextSelection}>
                          <pre className="whitespace-pre-wrap font-serif text-gray-800 leading-relaxed">
                            {(docInfo.pages || docInfo.data?.pages)?.[currentPage - 1]?.content || 'No content'}
                          </pre>
                        </div>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="flex items-center justify-center h-full text-gray-400">
                    <p>Loading PDF...</p>
                  </div>
                )}
              </div>
            </motion.div>
          ) : (
            /* 空状态 */
            <div className="flex-1 flex items-center justify-center relative overflow-hidden">
              <div className="text-center space-y-8 max-w-md relative z-10">
                <div className={`w-24 h-24 backdrop-blur-md rounded-[32px] flex items-center justify-center mx-auto shadow-sm border ${darkMode ? 'bg-white/10 border-white/10' : 'bg-white/50 border-white/60'}`}>
                  <Upload className={`w-10 h-10 ${darkMode ? 'text-purple-300' : 'text-purple-500/80'}`} />
                </div>
                <div className="space-y-2">
                  <h2 className={`text-3xl font-bold tracking-tight ${darkMode ? 'text-gray-100' : 'text-gray-800'}`}>Upload a PDF to Start</h2>
                  <p className={`text-lg ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Chat with your documents using AI.</p>
                </div>
              </div>
            </div>
          )}

          {/* 可拖拽分隔线 */}
          <div
            className="w-4 cursor-col-resize flex-shrink-0 relative group -ml-2 z-10 flex justify-center"
            onMouseDown={(e) => {
              e.preventDefault();
              const startX = e.clientX;
              const startWidth = pdfPanelWidth;
              const handleMouseMove = (e) => {
                const containerWidth = e.currentTarget?.parentElement?.offsetWidth || window.innerWidth;
                const deltaX = e.clientX - startX;
                const deltaPercent = (deltaX / containerWidth) * 100;
                const newWidth = Math.max(30, Math.min(70, startWidth + deltaPercent));
                setPdfPanelWidth(newWidth);
              };
              const handleMouseUp = () => {
                document.removeEventListener('mousemove', handleMouseMove);
                document.removeEventListener('mouseup', handleMouseUp);
              };
              document.addEventListener('mousemove', handleMouseMove);
              document.addEventListener('mouseup', handleMouseUp);
            }}
          >
            <div className="w-1 h-full rounded-full bg-transparent group-hover:bg-purple-500/50 transition-colors duration-200" />
          </div>

          {/* 右侧：聊天/速览/解析区域 */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className={`soft-panel relative flex flex-col overflow-hidden rounded-[var(--radius-panel)] min-w-0 ${darkMode ? 'bg-gray-800/50' : ''}`}
            style={{ width: `calc(${100 - pdfPanelWidth}% - 2rem)`, minWidth: '350px' }}
          >
            <div className="absolute right-5 top-5 z-30">
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation();
                  setShowAiProcessingPanel((prev) => !prev);
                }}
                className={`inline-flex items-center gap-2 rounded-full border px-3 py-2 text-[12px] font-semibold shadow-[0_10px_24px_rgba(148,163,184,0.14),inset_0_1px_0_rgba(255,255,255,0.9)] transition-all hover:-translate-y-0.5 ${
                  darkMode
                    ? 'border-white/10 bg-white/[0.06] text-gray-200 hover:bg-white/[0.09]'
                    : 'border-white/70 bg-white/75 text-gray-600 hover:text-[#8871e4]'
                }`}
              >
                <span className={`relative flex h-2 w-2 rounded-full ${
                  aiProcessingRunning
                    ? 'bg-[#8871e4]'
                    : hasAiRecommendation
                      ? 'bg-amber-400'
                      : aiAutoProcess ? 'bg-emerald-400' : 'bg-gray-300'
                }`}>
                  {aiProcessingRunning && <span className="absolute inset-0 animate-ping rounded-full bg-[#8871e4]/50" />}
                  {!aiProcessingRunning && hasAiRecommendation && <span className="absolute inset-0 animate-ping rounded-full bg-amber-400/60" />}
                </span>
                AI 处理
              </button>

              {showAiProcessingPanel && (
                <div
                  onClick={(event) => event.stopPropagation()}
                  className={`absolute right-0 top-12 w-[340px] rounded-[24px] border p-4 shadow-[0_24px_60px_rgba(15,23,42,0.16)] backdrop-blur-xl ${
                    darkMode
                      ? 'border-white/10 bg-[#1f2329]/95 text-gray-100'
                      : 'border-white/80 bg-white/95 text-gray-900'
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="text-[14px] font-bold">当前文档 AI 处理</div>
                      <div className={`mt-0.5 text-[11px] ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>{aiProcessingStatusText}</div>
                    </div>
                    <button
                      type="button"
                      onClick={() => setShowAiProcessingPanel(false)}
                      className={`rounded-full p-1.5 transition-colors ${darkMode ? 'text-gray-400 hover:bg-white/10 hover:text-gray-200' : 'text-gray-400 hover:bg-gray-100 hover:text-gray-600'}`}
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </div>

                  <div className="mt-4 space-y-2.5">
                    {aiProcessingItems.map((item) => (
                      <div
                        key={item.id}
                        className={`rounded-[18px] border p-3 ${
                          item.recommended
                            ? darkMode ? 'border-[#8871e4]/40 bg-[#8871e4]/[0.08]' : 'border-[#8871e4]/30 bg-[#8871e4]/[0.05]'
                            : darkMode ? 'border-white/10 bg-white/[0.04]' : 'border-gray-100 bg-gray-50/70'
                        }`}
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div className="min-w-0">
                            <div className="flex items-center gap-2">
                              <span className="text-[13px] font-bold">{item.title}</span>
                              {item.recommended && (
                                <span className="rounded-full bg-[#8871e4] px-2 py-0.5 text-[10px] font-semibold text-white">
                                  推荐
                                </span>
                              )}
                              <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                                item.busy
                                  ? 'bg-[#8871e4]/10 text-[#8871e4]'
                                  : darkMode ? 'bg-white/10 text-gray-300' : 'bg-white text-gray-500'
                              }`}>
                                {item.status}
                              </span>
                            </div>
                            <div className={`mt-1 text-[11px] leading-relaxed ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>{item.desc}</div>
                          </div>
                          <button
                            type="button"
                            onClick={(event) => {
                              event.stopPropagation();
                              item.onAction?.();
                            }}
                            disabled={item.disabled}
                            className={`shrink-0 rounded-full px-3 py-1.5 text-[11px] font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
                              item.busy
                                ? darkMode ? 'bg-white/10 text-gray-300' : 'bg-gray-200 text-gray-500'
                                : darkMode ? 'bg-white/10 text-gray-100 hover:bg-white/15' : 'bg-white text-gray-700 shadow-sm hover:text-[#8871e4]'
                            }`}
                          >
                            {item.actionLabel}
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="mt-3 flex items-center justify-between gap-3">
                    {!aiAutoProcess && (
                      <span className={`text-[11px] ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>自动关闭时不会主动消耗 token</span>
                    )}
                    <button
                      type="button"
                      onClick={handleClearDocumentAICache}
                      disabled={!docId}
                      className={`ml-auto inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-[11px] font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
                        darkMode ? 'text-gray-300 hover:bg-white/10' : 'text-gray-500 hover:bg-gray-100 hover:text-gray-700'
                      }`}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                      清理缓存
                    </button>
                  </div>
                  {pretranslateNotice && (
                    <div className={`mt-2 rounded-2xl px-3 py-2 text-[11px] ${
                      darkMode ? 'bg-white/[0.05] text-gray-300' : 'bg-[#f6f3ff] text-[#6f5cc2]'
                    }`}>
                      {pretranslateNotice}
                    </div>
                  )}
                  {deepParsePanelNotice && (
                    <div className={`mt-2 rounded-2xl px-3 py-2 text-[11px] ${
                      darkMode ? 'bg-white/[0.05] text-gray-300' : 'bg-[#f6f3ff] text-[#6f5cc2]'
                    }`}>
                      {deepParsePanelNotice}
                    </div>
                  )}
                  {ragIndexNotice && (
                    <div className={`mt-2 rounded-2xl px-3 py-2 text-[11px] ${
                      darkMode ? 'bg-white/[0.05] text-gray-300' : 'bg-[#f6f3ff] text-[#6f5cc2]'
                    }`}>
                      {ragIndexNotice}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* 顶部导航：速览 / 解析 / 对话 */}
              <div className="pt-6 pb-2 flex justify-center shrink-0">
                <div className={`relative grid w-[240px] grid-cols-3 overflow-hidden rounded-[22px] p-1 border shadow-[inset_0_1px_0_rgba(255,255,255,0.85)] backdrop-blur-xl ${
                  darkMode
                    ? 'bg-white/[0.05] border-white/10'
                    : 'bg-white/35 border-white/70 shadow-[inset_0_1px_1px_rgba(255,255,255,0.9),0_8px_24px_rgba(148,163,184,0.10)]'
                }`}>
                  <motion.div
                    className={`absolute top-1 bottom-1 left-1 rounded-[18px] z-0 ${
                      darkMode
                        ? 'bg-white/12 shadow-[0_8px_22px_rgba(0,0,0,0.22)]'
                        : 'bg-white/90 shadow-[0_10px_24px_rgba(139,124,200,0.14),0_2px_6px_rgba(31,41,55,0.06),inset_0_1px_0_rgba(255,255,255,0.95)]'
                    }`}
                    initial={false}
                    style={{ width: 'calc((100% - 0.5rem) / 3)' }}
                    animate={{ x: `${activeRightPanelTabIndex * 100}%` }}
                    transition={{ type: 'spring', stiffness: 430, damping: 34, mass: 0.72 }}
                  />

                  {rightPanelTabs.map(({ id, label }) => {
                    const isActive = rightPanelMode === id;
                    return (
                      <motion.button
                        key={id}
                        type="button"
                        onClick={() => setRightPanelMode(id)}
                        whileTap={{ scale: 0.97 }}
                        className={`relative z-10 h-9 rounded-[18px] text-[13px] font-semibold transition-colors duration-200 ${
                          isActive
                            ? (darkMode ? 'text-white' : 'text-[#8b7cc8]')
                            : (darkMode ? 'text-gray-500 hover:text-gray-300' : 'text-gray-400 hover:text-gray-600')
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
                    figureMode={overviewFigureMode}
                    onDepthChange={handleOverviewDepthChange}
                    onFetch={fetchOverview}
                  />
                </Suspense>
              ) : rightPanelMode === 'analysis' ? (
                <ReadingAnalysisPanel
                  blocks={currentPageBlocks}
                  translations={blockTranslations}
                  translatingBlockIds={[...translatingBlockIds]}
                  loading={blockTranslateLoading}
                  error={blockTranslateError}
                  notice={pretranslateNotice}
                  pretranslateProgress={pretranslateProgress}
                  currentPage={currentPage}
                  activeBlockId={activeReadingBlockId}
                  notes={currentPageReadingNotes}
                  activeNodeId={activeReadingNodeId}
                  visitedNodeIds={[...visitedReadingNodeIds]}
                  onTranslate={handleTranslateCurrentPage}
                  onRetranslateBlock={handleRetranslateReadingBlock}
                  onPretranslate={handleStartPretranslate}
                  onBlockHover={handleReadingBlockHover}
                  onBlockClick={handleReadingBlockClick}
                  onNoteClick={handleOutlineJump}
                  darkMode={darkMode}
                />
              ) : (
                <>
                  {/* 预设问题 */}
                  {showPresetQuestions && (
                    <div className="p-6 pb-0">
                      <PresetQuestions onSelect={handlePresetSelect} disabled={isLoading} />
                    </div>
                  )}

                  {/* 虚拟消息列表 - 替代原有的 messages.map 渲染（需求 3.1） */}
                  <VirtualMessageList
                    messages={messages}
                    renderMessage={renderMessage}
                    streamingMessageId={streamingMessageId}
                    className="flex-1 overflow-y-auto overflow-x-hidden p-6 pb-36 space-y-6 min-w-0"
                  />
                </>
              )}
            </div>

            {/* 输入区域：仅在对话模式显示，避免遮挡速览/解析内容 */}
            {rightPanelMode === 'chat' && (
            <div className="p-6 pt-0 bg-transparent relative z-10">
              <div className="absolute bottom-5 left-3 right-3 bg-[#f2f3f9] shadow-[0_12px_40px_rgba(0,0,0,0.12),inset_0_1px_0_rgba(255,255,255,0.8)] border border-white/80 rounded-[2rem] p-2.5 z-20">
                {/* 截图预览 - 嵌入输入框顶部，避免被遮挡 */}
                <ScreenshotPreview
                  screenshots={screenshots}
                  onAction={handleScreenshotAction}
                  onClose={handleScreenshotClose}
                />
                {/* 上半部分：模型选择、状态、工具图标 */}
                <div className="flex items-center justify-between mb-2.5 px-1">
                  <ModelQuickSwitch onThinkingChange={handleThinkingChange} />
                  
                  {/* 右侧工具图标 */}
                  <div className="flex items-center gap-2 text-gray-500 shrink-0">
                    <button onClick={openSettings} className="hover:text-gray-800 transition-colors p-1 rounded-md" title="设置中心" aria-label="设置中心">
                      <Settings size={15} />
                    </button>
                    <button onClick={() => fileInputRef.current?.click()} className="hover:text-gray-800 transition-colors p-1 rounded-md">
                      <Paperclip size={15} />
                    </button>
                    <WebSearchButton />
                    {isVisionCapable && (
                      <button
                        onClick={() => setIsSelectingArea(true)}
                        disabled={!docId}
                        className={`transition-colors p-1 rounded-md ${docId ? isSelectingArea ? 'text-purple-600' : 'hover:text-gray-800' : 'text-gray-300 cursor-not-allowed'}`}
                        title={!docId ? '请先上传文档' : isSelectingArea ? '框选模式已开启' : '区域截图'}
                      >
                        <Scan size={15} />
                      </button>
                    )}
                  </div>
                </div>

                {/* 下半部分：输入区 */}
                <div className="flex items-center bg-white rounded-full p-1.5 shadow-sm border border-black/5 focus-within:ring-2 focus-within:ring-purple-100 focus-within:border-purple-200 transition-all">
                  <textarea
                    ref={textareaRef}
                    onChange={(e) => {
                      e.target.style.height = '24px';
                      e.target.style.height = e.target.scrollHeight + 'px';
                      const newHasInput = !!e.target.value.trim();
                      if (newHasInput !== hasInput) setHasInput(newHasInput);
                    }}
                    onKeyDown={(e) => {
                      if (sendShortcut === 'Ctrl+Enter') {
                        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); sendMessage(); }
                      } else {
                        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
                      }
                    }}
                    placeholder="Summarize, rephrase, convert..."
                    className="flex-1 bg-transparent outline-none px-4 text-[14px] text-gray-800 min-w-0 resize-none h-[24px] overflow-hidden leading-relaxed py-0"
                    rows={1}
                    style={{ minHeight: '24px', maxHeight: '120px' }}
                  />
                  
                  {/* Send 文字和发送按钮，固定在右侧 */}
                  <div className="flex items-center gap-3 pr-1 shrink-0">
                    <span className="text-[#aba6d1] text-[13px] font-medium select-none pointer-events-none">Send</span>
                    <button
                      onClick={isLoading ? handleStop : sendMessage}
                      disabled={!isLoading && (!hasInput && screenshots.length === 0)}
                      className={`w-9 h-9 rounded-full transition-colors flex items-center justify-center shadow-sm ${
                        isLoading || hasInput || screenshots.length > 0 
                          ? 'bg-[#f0efff] text-[#7c3aed] hover:bg-[#7c3aed] hover:text-white' 
                          : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                      }`}
                    >
                      <AnimatePresence initial={false}>
                        {isLoading ? (
                          <motion.div key="pause" initial={{ rotate: -90, scale: 0.5, opacity: 0 }} animate={{ rotate: 0, scale: 1, opacity: 1 }} exit={{ rotate: 90, scale: 0.5, opacity: 0 }} transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }} className="absolute flex items-center justify-center">
                            <PauseIcon />
                          </motion.div>
                        ) : (
                          <motion.div key="send" initial={{ rotate: -90, scale: 0.5, opacity: 0 }} animate={{ rotate: 0, scale: 1, opacity: 1 }} exit={{ rotate: 90, scale: 0.5, opacity: 0 }} transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }} className="absolute flex items-center justify-center ml-0.5">
                            <SendIcon />
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </button>
                  </div>
                </div>
              </div>
            </div>
            )}
          </motion.div>
        </div>
      </div>

      {/* 上传进度模态框 */}
      <AnimatePresence>
        {isUploading && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/90">
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              transition={{ type: 'spring', stiffness: 350, damping: 25 }}
              className="flex flex-col items-center"
            >
              <div style={{ position: 'relative', width: 300, height: 300 }}>
                <div style={{ position: 'absolute', inset: 0, filter: 'blur(0.5px) contrast(1.2)' }}>
                  {UPLOAD_RING_CONFIGS.map((cfg, i) => (
                    <div key={i} style={{
                      position: 'absolute', top: '50%', left: '50%',
                      width: cfg.s, height: cfg.s, borderRadius: cfg.br,
                      border: `${cfg.w}px solid ${cfg.c}`, background: 'transparent',
                      mixBlendMode: cfg.mix, pointerEvents: 'none',
                      animation: `chatpdf-spin ${cfg.dur}s linear ${cfg.del}s infinite ${cfg.dir}`,
                    }} />
                  ))}
                </div>
                <div style={{
                  position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
                  alignItems: 'center', justifyContent: 'center', zIndex: 10, pointerEvents: 'none',
                }}>
                  <span style={{ color: 'rgba(255, 255, 255, 0.9)', fontSize: '2.5rem', fontWeight: 200, letterSpacing: '2px', textShadow: '0 0 15px rgba(255, 255, 255, 0.3)', fontVariantNumeric: 'tabular-nums' }}>
                    {uploadProgressLabel}%
                  </span>
                  <span style={{ color: 'rgba(255, 255, 255, 0.55)', fontSize: '0.7rem', letterSpacing: '4px', textTransform: 'uppercase', marginTop: '6px' }}>
                    {uploadStatusMeta.label}
                  </span>
                </div>
              </div>
              <motion.p
                key={uploadStatus}
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                style={{ color: 'rgba(255, 255, 255, 0.6)', fontSize: '0.9rem', fontWeight: 300, letterSpacing: '0.5px', marginTop: '8px' }}
              >
                {uploadStatusMeta.title}
              </motion.p>
              <motion.p
                key={`${uploadStatus}-desc`}
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                style={{ color: 'rgba(255, 255, 255, 0.38)', fontSize: '0.75rem', fontWeight: 300, letterSpacing: '0.2px', marginTop: '6px', maxWidth: 340, textAlign: 'center', lineHeight: 1.6 }}
              >
                {uploadStatusMeta.desc}
              </motion.p>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* 设置模态框 */}
      <AnimatePresence initial={false}>
        {showSettings && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            transition={{ duration: 0.12 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/25 p-4"
            onClick={() => setShowSettings(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.95, opacity: 0, y: 10 }}
              transition={{ type: 'spring', stiffness: 300, damping: 30, mass: 0.8 }}
              onClick={(e) => e.stopPropagation()}
              className={`settings-solid settings-shell w-[460px] max-w-full max-h-[90vh] overflow-hidden flex flex-col border ${darkMode ? 'settings-shell-dark bg-[#1d2026] border-[#353941]' : 'bg-[#f6f7f9] border-white/80 relative'}`}
            >
              <div className="p-6 pb-2 flex-shrink-0 flex items-center justify-between mt-1 px-7">
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-[12px] border ${darkMode ? 'bg-[#292d35] border-[#3b4049]' : 'bg-white border-gray-200'}`}>
                    <Settings className="text-[#7c4dff]" size={22} />
                  </div>
                  <h2 className={`text-xl font-bold tracking-tight ${darkMode ? 'text-gray-100' : 'text-gray-800'}`}>设置中心</h2>
                </div>
                <div className="flex items-center gap-2">
                  <button onClick={() => { setShowSettings(false); setShowEmbeddingSettings(true); }} className={`transition-all duration-300 hover:shadow-[0_8px_20px_rgba(42,36,66,0.3)] hover:-translate-y-0.5 text-white text-xs font-semibold px-3.5 py-2 rounded-full ${darkMode ? 'bg-[#3a3452] hover:bg-[#2a2442]' : 'bg-[#2a2442] hover:bg-[#1a1528]'}`}>
                    模型服务
                  </button>
                  <button onClick={() => setShowSettings(false)} className={`p-2 rounded-full transition-colors z-10 ${darkMode ? 'hover:bg-white/10 text-gray-500 hover:text-gray-300' : 'hover:bg-black/5 text-gray-400 hover:text-gray-700'}`} title="关闭设置中心" aria-label="关闭设置中心">
                    <X className="w-5 h-5" />
                  </button>
                </div>
              </div>

              <div className={`relative grid grid-cols-5 mx-6 mb-4 p-1 rounded-[14px] border flex-shrink-0 ${darkMode ? 'bg-[#272a31] border-[#383c45]' : 'bg-gray-100 border-gray-200'}`} role="tablist" aria-label="设置分类">
                <span
                  aria-hidden="true"
                  className={`absolute left-1 top-1 bottom-1 rounded-[10px] shadow-sm transition-transform duration-[320ms] ease-[cubic-bezier(0.4,0,0.2,1)] motion-reduce:transition-none ${darkMode ? 'bg-[#3a3f49]' : 'bg-white border border-gray-200'}`}
                  style={{
                    width: 'calc((100% - 0.5rem) / 5)',
                    transform: `translateX(${['common', 'reading', 'retrieval', 'interface', 'storage'].indexOf(settingsSection) * 100}%)`,
                  }}
                />
                {[
                  { id: 'common', label: '常用' },
                  { id: 'reading', label: '阅读' },
                  { id: 'retrieval', label: '检索' },
                  { id: 'interface', label: '界面' },
                  { id: 'storage', label: '存储' },
                ].map((section) => (
                  <button
                    key={section.id}
                    type="button"
                    role="tab"
                    aria-selected={settingsSection === section.id}
                    onClick={() => handleSettingsSectionChange(section.id)}
                    className={`relative z-10 min-w-0 rounded-[14px] px-2 py-2 text-[12px] font-semibold transition-colors ${
                      settingsSection === section.id
                        ? darkMode
                          ? 'text-white'
                          : 'text-[#7c4dff]'
                        : darkMode
                          ? 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
                          : 'text-gray-500 hover:text-gray-800 hover:bg-gray-200/70'
                    }`}
                  >
                    {section.label}
                  </button>
                ))}
              </div>

              <motion.div
                key={settingsSection}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.18, ease: 'easeOut' }}
                className="space-y-5 px-6 overflow-y-auto flex-1 pb-6 custom-scrollbar"
              >
                
                {settingsSection === 'common' && (
                <div className="space-y-3.5 px-1">
                  {/* Chat Model Card */}
                  <div className={`settings-card p-4 flex items-center space-x-4 border ${darkMode ? 'settings-card-dark bg-[#24272e] border-[#373b44]' : 'bg-white border-gray-200/90'}`}>
                    <div className="w-[46px] h-[46px] rounded-[14px] bg-[#ece9fb] flex items-center justify-center text-[#7c4dff] shrink-0 border border-[#ddd7f6]">
                      <MessageSquare size={22} />
                    </div>
                    <div className="flex flex-col min-w-0 flex-1">
                      <h3 className={`text-[13px] font-bold uppercase tracking-wider ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                        对话模型
                      </h3>
                      <p className={`text-[12px] mt-0.5 font-medium truncate ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} title={getDefaultModelLabel(getDefaultModel('assistantModel'))}>
                        {getDefaultModelLabel(getDefaultModel('assistantModel')) || '未设置'}
                      </p>
                    </div>
                  </div>

                  {/* Embedding Model Card */}
                  <div className={`settings-card p-4 flex items-center space-x-4 border ${darkMode ? 'settings-card-dark bg-[#24272e] border-[#373b44]' : 'bg-white border-gray-200/90'}`}>
                    <div className="w-[46px] h-[46px] rounded-[14px] bg-[#ece9fb] flex items-center justify-center text-[#7c4dff] shrink-0 border border-[#ddd7f6]">
                      <Database size={22} />
                    </div>
                    <div className="flex flex-col min-w-0 flex-1">
                      <h3 className={`text-[13px] font-bold uppercase tracking-wider ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                        嵌入模型
                      </h3>
                      <p className={`text-[12px] mt-0.5 font-medium truncate ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} title={getDefaultModelLabel(getDefaultModel('embeddingModel'))}>
                        {getDefaultModelLabel(getDefaultModel('embeddingModel')) || '未设置'}
                      </p>
                    </div>
                  </div>

                  {/* Rerank Model Card */}
                  <div className={`settings-card p-4 flex items-center space-x-4 border ${darkMode ? 'settings-card-dark bg-[#24272e] border-[#373b44]' : 'bg-white border-gray-200/90'}`}>
                    <div className="w-[46px] h-[46px] rounded-[14px] bg-[#ece9fb] flex items-center justify-center text-[#7c4dff] shrink-0 border border-[#ddd7f6]">
                      <ArrowUpDown size={22} />
                    </div>
                    <div className="flex flex-col min-w-0 flex-1">
                      <h3 className={`text-[13px] font-bold uppercase tracking-wider ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                        重排模型
                      </h3>
                      <p className={`text-[12px] mt-0.5 font-medium truncate ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} title={getDefaultModelLabel(getDefaultModel('rerankModel'))}>
                        {getDefaultModelLabel(getDefaultModel('rerankModel')) || '未设置'}
                      </p>
                    </div>
                  </div>
                </div>
                )}

                {/* 智能阅读 — 从「全局设置 > 阅读」上移为一级分区：
                    大纲/总结/预翻译/速览是产品核心能力，不应藏在三级深度 */}
                {settingsSection === 'reading' && (
                <div className={`settings-card p-5 border space-y-3 mt-2 mx-1 ${darkMode ? 'settings-card-dark bg-[#24272e] border-[#373b44]' : 'bg-white border-gray-200/90'}`}>
                  <label className="flex items-start space-x-3.5 group cursor-pointer p-1 rounded-2xl hover:bg-white/40 transition-colors">
                    <div className={`w-5 h-5 rounded-[6px] flex items-center justify-center shrink-0 mt-0.5 transition-transform group-hover:scale-105 ${aiAutoProcess ? 'bg-[#7c4dff] text-white shadow-[0_4px_12px_rgba(124,77,255,0.3)]' : 'border-2 border-gray-300 bg-transparent'}`}>
                      {aiAutoProcess && <Check size={13} strokeWidth={3.5} />}
                    </div>
                    <div className="flex flex-col flex-1">
                      <div className="flex items-center justify-between">
                        <h4 className={`text-[14px] font-semibold leading-snug ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>智能阅读</h4>
                        <input type="checkbox" checked={aiAutoProcess} onChange={e => setAiAutoProcess(e.target.checked)} className="hidden" />
                      </div>
                      <p className={`text-[12px] mt-0.5 leading-relaxed font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        控制文档打开后是否自动生成 AI 阅读辅助；关闭后打开文档零模型调用
                      </p>
                    </div>
                  </label>

                  <div className={`space-y-3 pt-1 ${aiAutoProcess ? '' : 'opacity-50 pointer-events-none'}`}>
                    <label className="flex items-start space-x-3.5 group cursor-pointer p-1 rounded-2xl hover:bg-white/40 transition-colors">
                      <div className={`w-5 h-5 rounded-[6px] flex items-center justify-center shrink-0 mt-0.5 transition-transform group-hover:scale-105 ${autoOutlineSummary ? 'bg-[#7c4dff] text-white shadow-[0_4px_12px_rgba(124,77,255,0.3)]' : 'border-2 border-gray-300 bg-transparent'}`}>
                        {autoOutlineSummary && <Check size={13} strokeWidth={3.5} />}
                      </div>
                      <div className="flex flex-col flex-1">
                        <div className="flex items-center justify-between">
                          <h4 className={`text-[14px] font-semibold leading-snug ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>自动生成大纲与总结</h4>
                          <input type="checkbox" checked={autoOutlineSummary} onChange={e => setAutoOutlineSummary(e.target.checked)} className="hidden" />
                        </div>
                        <p className={`text-[12px] mt-0.5 leading-relaxed font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                          打开文档后自动生成左侧 AI 总结和章节大纲；关闭后只读取已有缓存
                        </p>
                      </div>
                    </label>

                    <label className="flex items-start space-x-3.5 group cursor-pointer p-1 rounded-2xl hover:bg-white/40 transition-colors">
                      <div className={`w-5 h-5 rounded-[6px] flex items-center justify-center shrink-0 mt-0.5 transition-transform group-hover:scale-105 ${enableHoverPretranslate ? 'bg-[#7c4dff] text-white shadow-[0_4px_12px_rgba(124,77,255,0.3)]' : 'border-2 border-gray-300 bg-transparent'}`}>
                        {enableHoverPretranslate && <Check size={13} strokeWidth={3.5} />}
                      </div>
                      <div className="flex flex-col flex-1">
                        <div className="flex items-center justify-between">
                          <h4 className={`text-[14px] font-semibold leading-snug ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>自动预翻译全文</h4>
                          <input type="checkbox" checked={enableHoverPretranslate} onChange={e => setAutoPretranslate(e.target.checked)} className="hidden" />
                        </div>
                        <p className={`text-[12px] mt-0.5 leading-relaxed font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                          提前缓存正文、标题和图注翻译，悬浮时直接显示；会产生较多模型调用
                        </p>
                      </div>
                    </label>

                    <div className={`p-3.5 rounded-[14px] border ${darkMode ? 'bg-[#1d2026] border-[#353941]' : 'bg-gray-50 border-gray-200'}`}>
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <div className={`text-[12px] font-bold ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>预翻译并发</div>
                          <div className={`text-[11px] mt-0.5 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>限速模型建议 3-6，付费高速模型可用 8-16</div>
                        </div>
                        <span className="text-[12px] font-bold text-[#7c4dff] tabular-nums">{pretranslateConcurrency}</span>
                      </div>
                      <input
                        type="range"
                        min="1"
                        max="16"
                        step="1"
                        value={pretranslateConcurrency}
                        onChange={(e) => setPretranslateConcurrency(Number(e.target.value))}
                        className="mt-3 w-full accent-[#7c4dff]"
                      />
                    </div>

                    <div className={`p-3.5 rounded-[14px] border ${darkMode ? 'bg-[#1d2026] border-[#353941]' : 'bg-gray-50 border-gray-200'}`}>
                      <div className={`text-[12px] font-bold ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>速览默认详细度</div>
                      <div className={`mt-2 grid grid-cols-3 gap-1 p-1 rounded-[12px] border ${darkMode ? 'bg-[#282c34] border-[#3a3f49]' : 'bg-white border-gray-200'}`}>
                        {[
                          { id: 'brief', label: '简略' },
                          { id: 'standard', label: '标准' },
                          { id: 'detailed', label: '详细' },
                        ].map((item) => (
                          <button
                            key={item.id}
                            onClick={() => setOverviewDefaultDepth(item.id)}
                            className={`py-1.5 rounded-[10px] text-[11px] font-bold transition-all ${
                              overviewDefaultDepth === item.id
                                ? 'bg-[#7c4dff] text-white shadow-sm'
                                : darkMode ? 'text-gray-400 hover:text-gray-200' : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                            }`}
                          >
                            {item.label}
                          </button>
                        ))}
                      </div>
                      <p className={`mt-2 text-[11px] ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>速览仍然按需触发，不会因这个选项自动消耗 token</p>
                    </div>
                  </div>
                </div>
                )}

                {/* Features Section - Glass Inner Panel */}
                {(settingsSection === 'retrieval' || settingsSection === 'interface') && (
                <div className={`settings-card p-5 border space-y-3 mt-2 mx-1 ${darkMode ? 'settings-card-dark bg-[#24272e] border-[#373b44]' : 'bg-white border-gray-200/90'}`}>
                  
                  {settingsSection === 'retrieval' && (
                  <label className="flex items-start space-x-3.5 group cursor-pointer p-1 rounded-2xl hover:bg-white/40 transition-colors">
                    <div className={`w-5 h-5 rounded-[6px] flex items-center justify-center shrink-0 mt-0.5 transition-transform group-hover:scale-105 ${enableVectorSearch ? 'bg-[#7c4dff] text-white shadow-[0_4px_12px_rgba(124,77,255,0.3)]' : 'border-2 border-gray-300 bg-transparent'}`}>
                      {enableVectorSearch && <Check size={13} strokeWidth={3.5} />}
                    </div>
                    <div className="flex flex-col flex-1">
                      <div className="flex items-center justify-between">
                        <h4 className={`text-[14px] font-semibold leading-snug ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>向量检索</h4>
                        <input type="checkbox" checked={enableVectorSearch} onChange={e => setEnableVectorSearch(e.target.checked)} className="hidden" />
                      </div>
                      <p className={`text-[12px] mt-0.5 leading-relaxed font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        基于向量的语义相似度检索，提供更准确的匹配
                      </p>
                    </div>
                  </label>
                  )}

                  {settingsSection === 'interface' && (
                  <label className="flex items-start space-x-3.5 group cursor-pointer p-1 rounded-2xl hover:bg-white/40 transition-colors">
                    <div className={`w-5 h-5 rounded-[6px] flex items-center justify-center shrink-0 mt-0.5 transition-transform group-hover:scale-105 ${enableScreenshot ? 'bg-[#7c4dff] text-white shadow-[0_4px_12px_rgba(124,77,255,0.3)]' : 'border-2 border-gray-300 bg-transparent'}`}>
                      {enableScreenshot && <Check size={13} strokeWidth={3.5} />}
                    </div>
                    <div className="flex flex-col flex-1">
                      <div className="flex items-center justify-between">
                        <h4 className={`text-[14px] font-semibold leading-snug ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>截图分析</h4>
                        <input type="checkbox" checked={enableScreenshot} onChange={e => setEnableScreenshot(e.target.checked)} className="hidden" />
                      </div>
                      <p className={`text-[12px] mt-0.5 leading-relaxed font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        启用截图分析功能，理解视觉内容及图表
                      </p>
                    </div>
                  </label>
                  )}

                  {settingsSection === 'retrieval' && (
                  <label className="flex items-start space-x-3.5 group cursor-pointer p-1 rounded-2xl hover:bg-white/40 transition-colors">
                    <div className={`w-5 h-5 rounded-[6px] flex items-center justify-center shrink-0 mt-0.5 transition-transform group-hover:scale-105 ${enableGraphRAG ? 'bg-[#7c4dff] text-white shadow-[0_4px_12px_rgba(124,77,255,0.3)]' : 'border-2 border-gray-300 bg-transparent'}`}>
                      {enableGraphRAG && <Check size={13} strokeWidth={3.5} />}
                    </div>
                    <div className="flex flex-col flex-1">
                      <div className="flex items-center justify-between">
                        <h4 className={`text-[14px] font-semibold leading-snug ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>GraphRAG 知识图谱</h4>
                        <input type="checkbox" checked={enableGraphRAG} onChange={e => setEnableGraphRAG(e.target.checked)} className="hidden" />
                      </div>
                      <p className={`text-[12px] mt-0.5 leading-relaxed font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        实体关系提取 + 社区聚类增强检索，提供全局视角
                      </p>
                    </div>
                  </label>
                  )}

                  {/* GraphRAG 构建控制：仅在勾选 + 有活动文档时显示 */}
                  {settingsSection === 'retrieval' && enableGraphRAG && docId && (
                    <div className={`ml-[34px] -mt-1 p-3 rounded-[16px] border ${darkMode ? 'bg-purple-500/10 border-purple-400/20' : 'bg-purple-50/80 border-purple-200'}`}>
                      <div className="flex items-start justify-between gap-3">
                        <div className="text-[12px] leading-relaxed flex-1 min-w-0">
                          {graphragStatus === 'built' && graphragStats ? (
                            <span className={`inline-flex items-center gap-1.5 font-medium ${darkMode ? 'text-purple-300' : 'text-purple-700'}`}>
                              <Check size={12} strokeWidth={3} className="shrink-0" />
                              已构建：{graphragStats.num_nodes ?? 0} 实体 · {graphragStats.num_edges ?? 0} 关系 · {graphragStats.num_chunks ?? 0} 分块
                            </span>
                          ) : graphragStatus === 'building' ? (
                            <span className={`inline-flex items-center gap-1.5 ${darkMode ? 'text-purple-300' : 'text-purple-700'}`}>
                              <Loader2 size={12} className="animate-spin shrink-0" />
                              {graphragProgress?.stage ? (
                                <>
                                  {graphragProgress.stage === 'chunking' && '分块中'}
                                  {graphragProgress.stage === 'extracting' && '提取实体/关系中'}
                                  {graphragProgress.stage === 'clustering' && '社区聚类中'}
                                  {graphragProgress.stage === 'reporting' && '生成社区报告中'}
                                  {graphragProgress.stage === 'persisting' && '持久化中'}
                                  {graphragProgress.progress > 0 && ` (${graphragProgress.progress}%)`}
                                </>
                              ) : '正在构建知识图谱，请勿关闭页面...'}
                            </span>
                          ) : graphragStatus === 'error' ? (
                            <span className="text-red-500">构建失败</span>
                          ) : (
                            <span className={darkMode ? 'text-gray-400' : 'text-gray-600'}>此文档尚未构建知识图谱</span>
                          )}
                        </div>
                        <button
                          type="button"
                          onClick={(e) => { e.preventDefault(); e.stopPropagation(); handleBuildGraphRAG(); }}
                          disabled={graphragStatus === 'building'}
                          className={`shrink-0 px-3 py-1 rounded-full text-[11px] font-semibold transition-colors ${
                            graphragStatus === 'building'
                              ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                              : 'bg-[#7c4dff] text-white hover:bg-[#6a3ff0] shadow-[0_2px_8px_rgba(124,77,255,0.25)]'
                          }`}
                        >
                          {graphragStatus === 'built' ? '重新构建' : graphragStatus === 'building' ? '构建中' : '立即构建'}
                        </button>
                      </div>
                      {graphragStatus === 'error' && graphragError && (
                        <div className="mt-2 text-[11px] text-red-500 leading-relaxed break-words">
                          {graphragError}
                        </div>
                      )}
                      {graphragStatus === 'idle' && (
                        <p className={`mt-1.5 text-[11px] leading-relaxed ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                          首次构建会调用对话模型逐块提取，通常耗时 30 秒至数分钟。构建后知识图谱会自动注入聊天上下文。
                        </p>
                      )}
                    </div>
                  )}

                  {/* 检索代理：多轮规划 + 工具集 */}
                  {settingsSection === 'retrieval' && (
                  <label className="flex items-start space-x-3.5 group cursor-pointer p-1 rounded-2xl hover:bg-white/40 transition-colors">
                    <div className={`w-5 h-5 rounded-[6px] flex items-center justify-center shrink-0 mt-0.5 transition-transform group-hover:scale-105 ${enableAgentRetrieval ? 'bg-[#7c4dff] text-white shadow-[0_4px_12px_rgba(124,77,255,0.3)]' : 'border-2 border-gray-300 bg-transparent'}`}>
                      {enableAgentRetrieval && <Check size={13} strokeWidth={3.5} />}
                    </div>
                    <div className="flex flex-col flex-1">
                      <div className="flex items-center justify-between">
                        <h4 className={`text-[14px] font-semibold leading-snug ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>检索代理 (Agentic RAG)</h4>
                        <input type="checkbox" checked={enableAgentRetrieval} onChange={e => setEnableAgentRetrieval(e.target.checked)} className="hidden" />
                      </div>
                      <p className={`text-[12px] mt-0.5 leading-relaxed font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        多轮规划 + 7 种检索工具，仅对综述/比较/章节解析等高价值题型触发
                      </p>
                    </div>
                  </label>
                  )}

                  {settingsSection === 'retrieval' && enableAgentRetrieval && (
                    <div className={`ml-[34px] -mt-1 p-3 rounded-[16px] border ${darkMode ? 'bg-violet-500/10 border-violet-400/20' : 'bg-violet-50/80 border-violet-200'}`}>
                      <p className={`text-[11px] leading-relaxed ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                        启用后，对「综述全文 / 多角度比较 / 章节解读 / 文献元信息」等高价值问题会自动进入多轮代理：
                        每轮 LLM 规划 → 调用 vector / BM25 / GREP / 正则 / 布尔 / 意群 fetch / 文档地图 等工具采集证据，
                        最多 5 轮（可在后端 <code className="px-1 rounded bg-black/5">agent_max_rounds</code> 配置 1–10）。
                        每条回答下方会展示完整执行轨迹。
                      </p>
                    </div>
                  )}

                  {settingsSection === 'retrieval' && enableAgentRetrieval && (
                    <label className="ml-[34px] mt-1 flex items-center gap-2 text-[12px] text-gray-600 dark:text-gray-400 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={forceAgentRetrieval}
                        onChange={e => setForceAgentRetrieval(e.target.checked)}
                        className="w-3.5 h-3.5 rounded border-gray-300 text-violet-500 focus:ring-violet-300"
                      />
                      <span>强制启用 Agent</span>
                      <span title="勾选后所有问题都将走 Agent 路径，绕过 query_type / evidence_needs 白名单门控。仅供调试与高价值题型评估。"
                            className="text-gray-400 cursor-help">ⓘ</span>
                    </label>
                  )}

                  {settingsSection === 'retrieval' && (
                  <label className="flex items-start space-x-3.5 group cursor-pointer p-1 rounded-2xl hover:bg-white/40 transition-colors">
                    <div className={`w-5 h-5 rounded-[6px] flex items-center justify-center shrink-0 mt-0.5 transition-transform group-hover:scale-105 ${enableJiebaBM25 ? 'bg-[#7c4dff] text-white shadow-[0_4px_12px_rgba(124,77,255,0.3)]' : 'border-2 border-gray-300 bg-transparent'}`}>
                      {enableJiebaBM25 && <Check size={13} strokeWidth={3.5} />}
                    </div>
                    <div className="flex flex-col flex-1">
                      <div className="flex items-center justify-between">
                        <h4 className={`text-[14px] font-semibold leading-snug ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>jieba 中文分词</h4>
                        <input type="checkbox" checked={enableJiebaBM25} onChange={e => setEnableJiebaBM25(e.target.checked)} className="hidden" />
                      </div>
                      <p className={`text-[12px] mt-0.5 leading-relaxed font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        使用结巴分词提升 BM25 中文关键词匹配精度
                      </p>
                    </div>
                  </label>
                  )}

                  {/* 检索增强调优：从「全局设置 > 高级」上移到这里，
                      和上面的检索开关同域，不再分散到两处 tab */}
                  {settingsSection === 'retrieval' && (
                  <div className={`pt-2 mt-1 border-t ${darkMode ? 'border-white/10' : 'border-gray-100/70'}`}>
                    <button onClick={() => setShowRetrievalTuning(!showRetrievalTuning)} className="w-full flex items-center justify-between text-left p-1 rounded-2xl hover:bg-white/40 transition-colors">
                      <div className="flex items-center gap-3">
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${darkMode ? 'bg-violet-500/10 text-violet-300' : 'bg-violet-50 text-violet-500'}`}>
                          <Sparkles className="w-4 h-4" />
                        </div>
                        <div>
                          <h4 className={`text-[14px] font-semibold leading-snug ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>检索增强调优</h4>
                          <p className={`text-[12px] mt-0.5 leading-relaxed font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>对当前会话覆盖后端检索开关，不持久化到后端 config</p>
                        </div>
                      </div>
                      <div className={`transform transition-transform text-gray-400 ${showRetrievalTuning ? 'rotate-180' : ''}`}>▼</div>
                    </button>

                    {showRetrievalTuning && (
                      <div className={`mt-2 ml-[44px] pt-3 border-t space-y-1 ${darkMode ? 'border-white/10' : 'border-gray-100/50'}`}>
                        <TriStateToggle
                          title="numeric_table 专项增强"
                          desc="表格数值比较类查询（如「第二好的方法」「Table 7 DiffuLT」）的专项检索增强"
                          value={overrideNumericTable}
                          onChange={setOverrideNumericTable}
                        />
                        <TriStateToggle
                          title="BM25 同义词扩展"
                          desc="查询时自动扩展同义词，提升召回率"
                          value={overrideBM25Synonyms}
                          onChange={setOverrideBM25Synonyms}
                        />
                        <TriStateToggle
                          title="LLM 查询改写"
                          desc="多轮对话中用 LLM 消解指代（代词/省略），长查询自动跳过"
                          value={overrideLLMQueryRewrite}
                          onChange={setOverrideLLMQueryRewrite}
                        />
                        <TriStateToggle
                          title="答案自审"
                          desc="回答结束后用 cheap model 检测幻觉；会增加 1-3s 延迟"
                          value={overrideAnswerCritic}
                          onChange={setOverrideAnswerCritic}
                        />
                        <VisualVerificationMode
                          value={numericTableVisualVerification}
                          onChange={setNumericTableVisualVerification}
                        />

                        {/* Cheap Model 配置 */}
                        <div className={`p-3 rounded-[14px] border space-y-2 ${darkMode ? 'bg-[#1d2026] border-[#353941]' : 'bg-gray-50 border-gray-200'}`}>
                          <div className="flex items-center justify-between px-1">
                            <span className={`text-[12px] font-bold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>辅助模型（双模型策略）</span>
                            <span className="text-[10px] text-gray-500">为空则跟随后端默认</span>
                          </div>
                          <p className="text-[11px] text-gray-500 px-1">
                            用于非核心 LLM 任务（查询改写 / 追问建议 / 自动命名 / 答案自审）
                          </p>
                          <div className="grid grid-cols-2 gap-2">
                            <input
                              type="text"
                              value={cheapModelProvider || ''}
                              onChange={(e) => setCheapModelProvider(e.target.value)}
                              placeholder="provider (如 openai)"
                              className={`text-[12px] font-mono rounded-[12px] px-3 py-2 outline-none focus:ring-2 focus:ring-violet-500/20 shadow-sm border ${darkMode ? 'bg-black/20 border-white/10 text-gray-200' : 'bg-white border-gray-200'}`}
                            />
                            <input
                              type="text"
                              value={cheapModel || ''}
                              onChange={(e) => setCheapModel(e.target.value)}
                              placeholder="model (如 gpt-4o-mini)"
                              className={`text-[12px] font-mono rounded-[12px] px-3 py-2 outline-none focus:ring-2 focus:ring-violet-500/20 shadow-sm border ${darkMode ? 'bg-black/20 border-white/10 text-gray-200' : 'bg-white border-gray-200'}`}
                            />
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                  )}

                  {settingsSection === 'interface' && (
                  <label className="flex items-start space-x-3.5 group cursor-pointer p-1 rounded-2xl hover:bg-white/40 transition-colors">
                    <div className={`w-5 h-5 rounded-[6px] flex items-center justify-center shrink-0 mt-0.5 transition-transform group-hover:scale-105 ${enableBlurReveal ? 'bg-[#7c4dff] text-white shadow-[0_4px_12px_rgba(124,77,255,0.3)]' : 'border-2 border-gray-300 bg-transparent'}`}>
                      {enableBlurReveal && <Check size={13} strokeWidth={3.5} />}
                    </div>
                    <div className="flex flex-col flex-1">
                      <div className="flex items-center justify-between">
                        <h4 className={`text-[14px] font-semibold leading-snug ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>模糊渐显效果</h4>
                        <input type="checkbox" checked={enableBlurReveal} onChange={e => setEnableBlurReveal(e.target.checked)} className="hidden" />
                      </div>
                      <p className={`text-[12px] mt-0.5 leading-relaxed font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        流式输出时每个新字符从模糊到清晰的渐变效果
                      </p>
                    </div>
                  </label>
                  )}
                </div>
                )}

                {/* Toolbar and Storage Settings Area */}
                {(settingsSection === 'interface' || settingsSection === 'storage') && (
                <div className={`settings-card p-5 border space-y-4 mt-4 mx-1 ${darkMode ? 'settings-card-dark bg-[#24272e] border-[#373b44]' : 'bg-white border-gray-200/90'}`}>
                  {/* Toolbar Settings */}
                  {settingsSection === 'interface' && (
                  <div className="space-y-3">
                    <h3 className={`text-[13px] font-bold tracking-wider uppercase ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>工具栏配置</h3>
                    <div className="space-y-3">
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
                            <input type="text" value={searchEngineUrl} onChange={(e) => setSearchEngineUrl(e.target.value)} className={`w-full p-2.5 rounded-[12px] border text-sm outline-none transition-colors ${darkMode ? 'bg-[#1d2026] border-[#353941] text-white focus:border-[#7c4dff]/50' : 'bg-white border-gray-200 focus:border-[#7c4dff]/50'}`} placeholder="例如：https://www.google.com/search?q={query}" />
                            <p className="text-[11px] text-gray-500 mt-1">使用 <code className="font-mono bg-black/5 px-1 rounded">{'<query>'}</code> 作为搜索词占位符</p>
                          </div>
                        )}
                      </div>
                      <div className="flex flex-col gap-1.5 pt-1">
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
                  )}

                  {/* Storage Info */}
                  {settingsSection === 'storage' && (
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
                            <button onClick={() => { navigator.clipboard.writeText(storageInfo.uploads_dir); alert('路径已复制到剪贴板！'); }} className="p-1.5 rounded-[8px] bg-[#7c4dff]/10 text-[#7c4dff] hover:bg-[#7c4dff]/20 transition-colors shrink-0" title="复制路径">
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
                            <button onClick={() => { navigator.clipboard.writeText(storageInfo.data_dir); alert('路径已复制到剪贴板！'); }} className="p-1.5 rounded-[8px] bg-[#7c4dff]/10 text-[#7c4dff] hover:bg-[#7c4dff]/20 transition-colors shrink-0" title="复制路径">
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
                                清理总结、大纲、速览和悬浮翻译缓存
                              </div>
                            </div>
                            <button
                              type="button"
                              onClick={handleClearDocumentAICache}
                              disabled={!docId}
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
                  )}
                </div>
                )}

                {/* Advanced Configuration Section */}
                {(settingsSection === 'retrieval' || settingsSection === 'interface') && (
                <div className={`settings-card p-5 border space-y-4 mt-4 mx-1 ${darkMode ? 'settings-card-dark bg-[#24272e] border-[#373b44]' : 'bg-white border-gray-200/90'}`}>
                  <h3 className={`text-[13px] font-bold tracking-wider uppercase mb-1 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    {settingsSection === 'retrieval' ? '检索参数' : '交互效果'}
                  </h3>
                  
                  <div className="space-y-3">
                    {settingsSection === 'retrieval' && (
                    <div className="flex flex-col gap-1.5">
                      <label className={`text-[12px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>邻居上下文扩展</label>
                      <CustomSelect
                        value={numExpandContextChunk}
                        onChange={setNumExpandContextChunk}
                        options={[
                          { value: 0, label: '关闭' },
                          { value: 1, label: '±1 块（前后各 1 个）' },
                          { value: 2, label: '±2 块（前后各 2 个）' },
                          { value: 3, label: '±3 块（前后各 3 个）' },
                        ]}
                      />
                      <p className="text-[11px] text-gray-500 mt-0.5">命中 chunk 前后各扩展 N 个邻居块作为上下文</p>
                    </div>
                    )}

                    {settingsSection === 'interface' && (
                    <div className="flex flex-col gap-1.5 pt-2 border-t border-gray-200/50 mt-2">
                      <label className={`text-[12px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>流式输出速度</label>
                        <CustomSelect
                          value={streamSpeed}
                          onChange={setStreamSpeed}
                          options={[
                          { value: 'fast', label: '快速 (4字符/次, ~16ms)' },
                          { value: 'normal', label: '正常 (2字符/次, ~28ms)' },
                          { value: 'slow', label: '慢速 (1字符/次, ~48ms)' },
                          { value: 'off', label: '关闭流式（直接显示）' }
                        ]}
                      />
                      <p className="text-[11px] text-gray-500 mt-0.5">调整AI回复的打字机效果速度</p>
                    </div>
                    )}

                    {settingsSection === 'interface' && enableBlurReveal && (
                      <div className="flex flex-col gap-1.5 pt-2 border-t border-gray-200/50 mt-2">
                        <label className={`text-[12px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>模糊效果强度</label>
                        <CustomSelect
                          value={blurIntensity}
                          onChange={setBlurIntensity}
                          options={[
                            { value: 'light', label: '轻度 (3px blur, 0.2s)' },
                            { value: 'medium', label: '中度 (5px blur, 0.25s)' },
                            { value: 'strong', label: '强烈 (8px blur, 0.3s)' }
                          ]}
                        />
                      </div>
                    )}
                  </div>

                  {settingsSection === 'retrieval' && lastCallInfo && (
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

                {/* Other Settings Access */}
                {settingsSection === 'common' && (
                <div className="grid grid-cols-3 gap-3 px-1 mt-4">
                  <button onClick={() => { setShowSettings(false); setShowGlobalSettings(true); }} title="字体 · 记忆与联网 · 检索调优 · 配置导入导出" className={`settings-card settings-card-interactive flex flex-col items-center justify-center p-3 border ${darkMode ? 'settings-card-dark bg-[#292d35] border-[#3b4049] hover:bg-[#30343d]' : 'bg-white border-gray-200/90 hover:border-gray-300'}`}>
                    <Type className={`w-5 h-5 mb-1.5 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`} />
                    <span className={`text-[12px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>全局设置</span>
                  </button>
                  <button onClick={() => { setShowSettings(false); setShowChatSettings(true); }} className={`settings-card settings-card-interactive flex flex-col items-center justify-center p-3 border ${darkMode ? 'settings-card-dark bg-[#292d35] border-[#3b4049] hover:bg-[#30343d]' : 'bg-white border-gray-200/90 hover:border-gray-300'}`}>
                    <SlidersHorizontal className={`w-5 h-5 mb-1.5 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`} />
                    <span className={`text-[12px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>对话设置</span>
                  </button>
                  <button onClick={() => { setShowSettings(false); setShowOCRSettings(true); }} className={`settings-card settings-card-interactive flex flex-col items-center justify-center p-3 border ${darkMode ? 'settings-card-dark bg-[#292d35] border-[#3b4049] hover:bg-[#30343d]' : 'bg-white border-gray-200/90 hover:border-gray-300'}`}>
                    <ScanText className={`w-5 h-5 mb-1.5 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`} />
                    <span className={`text-[12px] font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>解析设置</span>
                  </button>
                </div>
                )}
              </motion.div>

              <div className="p-6 pt-2 pb-8 flex-shrink-0 relative">
                <button onClick={() => setShowSettings(false)} className="w-full bg-[#7c4dff] hover:bg-[#6836f5] transition-colors text-white text-[15px] font-semibold py-3.5 rounded-[14px] shadow-[0_8px_18px_-10px_rgba(124,77,255,0.55)] flex items-center justify-center gap-2">
                  <Check size={17} strokeWidth={2.5} />
                  <span>完成</span>
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 懒加载设置面板（使用 useCallback 稳定的关闭回调） */}
      <Suspense fallback={null}>
        <EmbeddingSettings isOpen={showEmbeddingSettings} onClose={handleEmbeddingSettingsClose} />
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

// 自定义下拉选择组件
const CustomSelect = ({ value, onChange, options }) => {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (containerRef.current && !containerRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const selectedOption = options.find(opt => opt.value === value);

  return (
    <div className="relative w-full" ref={containerRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between p-2.5 rounded-[12px] bg-white/50 dark:bg-black/20 border border-gray-200 dark:border-white/10 text-sm hover:border-[#7c4dff]/50 transition-all outline-none"
      >
        <span className="text-gray-700 dark:text-gray-300 font-medium">
          {selectedOption ? selectedOption.label : 'Select...'}
        </span>
        <ChevronDown size={14} className={`text-gray-500 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -5, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -5, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="absolute z-50 w-full mt-1.5 p-1.5 bg-white/90 dark:bg-[#1a1d21]/95 backdrop-blur-xl border border-gray-100 dark:border-white/10 rounded-[16px] shadow-[0_8px_30px_rgb(0,0,0,0.08)] max-h-60 overflow-y-auto"
          >
            {options.map((option) => (
              <button
                key={option.value}
                onClick={() => {
                  onChange(option.value);
                  setIsOpen(false);
                }}
                className={`w-full text-left px-3 py-2.5 rounded-[10px] text-[13px] transition-colors flex items-center justify-between ${
                  value === option.value
                    ? 'bg-[#7c4dff]/10 text-[#7c4dff] font-semibold'
                    : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100/50 dark:hover:bg-white/5'
                }`}
              >
                {option.label}
                {value === option.value && <Check size={14} className="text-[#7c4dff]" />}
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ChatPDF;





























