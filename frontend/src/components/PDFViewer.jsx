import React, { useState, useEffect, useRef, useCallback, forwardRef, useMemo } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import {
    BookOpen,
    Check,
    ChevronDown,
    ChevronLeft,
    ChevronRight,
    Columns2,
    FileText,
    Languages,
    Loader2,
    RotateCcw,
    RotateCw,
    ScrollText,
    Sidebar,
    X,
    ZoomIn,
    ZoomOut,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import BreatheLoader from './BreatheLoader';
import SelectionOverlay from './SelectionOverlay';
import StreamingMarkdown from './StreamingMarkdown';
import { useDebouncedValue } from '../hooks/useDebouncedValue';
import pdfPageCache from '../utils/pdfPageCache';
import {
    citationRectsToRendered,
    collectTextRangeClientRects,
    findBestCitationBlock,
    findCitationTextRange,
    isCitationGeometryCurrent,
    mergeClientRectsByLine,
    normalizeCitationBBox,
} from '../utils/citationHighlightUtils';
import {
    highlightColorToRgba,
    normalizeDocumentHighlightColor,
    normalizeDocumentHighlightRect,
    normalizeDocumentHighlightStyle,
} from '../utils/documentHighlightUtils';
import {
    PDF_READER_FLOW_MODES,
    PDF_READER_LAYOUTS,
    getPdfReaderDisplayPages,
    getPdfReaderNavigationTarget,
    getPdfReaderRotationTransform,
    mapPdfReaderDisplayPointToPage,
    mapPdfReaderDisplayRectToPage,
    mapPdfReaderPageRectToDisplay,
    normalizePdfReaderRotation,
    rotatePdfReader,
} from '../utils/pdfReaderLayoutUtils';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';
import pdfWorkerSrc from 'pdfjs-dist/build/pdf.worker.min.mjs?url';

// Configure worker - 直接指定版本以确保匹配
pdfjs.GlobalWorkerOptions.workerSrc = pdfWorkerSrc;

const PDFLoadingState = ({ darkMode }) => (
    <div
        className={`flex h-full flex-col items-center justify-center gap-4 text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}
        role="status"
        aria-live="polite"
    >
        <BreatheLoader className={darkMode ? 'text-[#FFA07A]' : 'text-[#D97A5D]'} />
        <span>加载 PDF 中...</span>
    </div>
);

const getPageBlockData = (blockIndex, pageNumber) => {
    if (!blockIndex?.pages || !pageNumber) return null;
    return blockIndex.pages.find((item) => Number(item.page) === Number(pageNumber)) || null;
};

const normalizeBlockBBox = (bbox) => {
    if (!Array.isArray(bbox) || bbox.length < 4) return null;
    const nums = bbox.slice(0, 4).map((value) => Number(value));
    if (nums.some((value) => !Number.isFinite(value))) return null;
    const [x0, y0, x1, y1] = nums;
    if (x1 <= x0 || y1 <= y0) return null;
    return [x0, y0, x1, y1];
};

const blockHitPriority = (block) => {
    if (block?.type === 'heading') return 4;
    if (block?.type === 'caption') return 3;
    if (block?.type === 'figure' || block?.type === 'table') return 2;
    return 1;
};

const isInteractiveBlock = (block) => block?.type !== 'artifact';

const HOVER_TRANSLATION_SWITCH_DELAY = 360;
const HOVER_TRANSLATION_MOVE_TOLERANCE = 10;
const TRANSLATION_DOCK_DEFAULT_WIDTH = 344;
const TRANSLATION_DOCK_MIN_WIDTH = 280;
const TRANSLATION_DOCK_MAX_WIDTH = 520;
const TRANSLATION_DOCK_GAP = 32;
const EMPTY_SAVED_HIGHLIGHTS = Object.freeze([]);
const BARE_MATH_ATOM = String.raw`(?:\\[A-Za-z]+(?:\{[^{}]*\})*|[A-Za-z][A-Za-z0-9]*(?:_\{[^{}]+\}|_[A-Za-z0-9+\-]+|\^\{[^{}]+\}|\^[A-Za-z0-9+\-]+)*(?:\([^，。；;\n]*?\))?|\d+(?:\.\d+)?|\([^，。；;\n]*?\)|\{[^{}，。；;\n]*\})`;
const BARE_MATH_OPERATOR = String.raw`(?:=|\\approx|\\leq|\\geq|\\neq|\\times|\\cdot|[+\-*/×·<>≤≥])`;
const BARE_MATH_EXPR_REGEX = new RegExp(`${BARE_MATH_ATOM}(?:\\s*${BARE_MATH_OPERATOR}\\s*${BARE_MATH_ATOM})+`, 'g');
const MARKDOWN_MATH_SEGMENT_REGEX = /(\$\$[\s\S]*?\$\$|\$[^$\n]+\$|```[\s\S]*?```|`[^`]*`)/g;

const normalizeHoverTranslationMath = (value) => {
    if (!value || typeof value !== 'string') return value;
    return value
        .split(MARKDOWN_MATH_SEGMENT_REGEX)
        .map((segment) => {
            if (
                !segment ||
                segment.startsWith('$') ||
                segment.startsWith('`') ||
                !/[=_^\\+\-*/×·<>≤≥]/.test(segment)
            ) {
                return segment;
            }
            return segment.replace(BARE_MATH_EXPR_REGEX, (match) => {
                const trimmed = match.trim();
                if (!trimmed || !/[_^\\=<>≤≥+\-*/×·]/.test(trimmed)) return match;
                const leading = match.match(/^\s*/)?.[0] || '';
                const trailing = match.match(/\s*$/)?.[0] || '';
                return `${leading}$${trimmed}$${trailing}`;
            });
        })
        .join('');
};

const PinIcon = ({ className = '' }) => (
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
        <path d="M12 17v5" />
        <path d="M9 10.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24V16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V7a1 1 0 0 1 1-1 2 2 0 0 0 0-4H8a2 2 0 0 0 0 4 1 1 0 0 1 1 1z" />
    </svg>
);

const DockIcon = ({ className = '' }) => (
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
        <rect x="3" y="4" width="18" height="16" rx="2" />
        <path d="M14 4v16" />
        <path d="M7 8h4" />
        <path d="M7 12h4" />
        <path d="M16.5 8h2" />
        <path d="M16.5 12h2" />
        <path d="M16.5 16h2" />
    </svg>
);

const clampNumber = (value, min, max) => Math.max(min, Math.min(max, value));

const scheduleIdleTask = (callback, timeout = 1000) => {
    if (typeof window === 'undefined') return null;
    if (typeof window.requestIdleCallback === 'function') {
        return { type: 'idle', id: window.requestIdleCallback(callback, { timeout }) };
    }
    return { type: 'timeout', id: window.setTimeout(callback, 0) };
};

const cancelIdleTask = (task) => {
    if (!task || typeof window === 'undefined') return;
    if (task.type === 'idle' && typeof window.cancelIdleCallback === 'function') {
        window.cancelIdleCallback(task.id);
        return;
    }
    window.clearTimeout(task.id);
};

const DEFERRED_PAGE_WIDTH = 612;
const DEFERRED_PAGE_HEIGHT = 792;

const DeferredPdfPage = ({
    pageNumber,
    scale,
    displayScale = scale,
    rotation,
    devicePixelRatio,
    darkMode,
    deferRender = false,
    scrollRootRef,
    onActivate,
    isActive = false,
    viewerRef,
    pageInputRef,
    cachedImage,
    onLoadSuccess,
    onRenderSuccess,
    children,
}) => {
    const holderRef = useRef(null);
    // 当前页永远立即渲染：它不该等 IntersectionObserver，也不该被 content-visibility 跳过。
    const [shouldRender, setShouldRender] = useState(!deferRender || isActive);
    const [pageSize, setPageSize] = useState({
        width: DEFERRED_PAGE_WIDTH,
        height: DEFERRED_PAGE_HEIGHT,
    });
    const loadedPageRef = useRef(null);
    const renderedPageRef = useRef(null);
    const [loadedVersion, setLoadedVersion] = useState(0);
    const [renderedVersion, setRenderedVersion] = useState(0);

    useEffect(() => {
        if (!deferRender || isActive) {
            setShouldRender(true);
            return undefined;
        }
        if (shouldRender || typeof IntersectionObserver === 'undefined') {
            setShouldRender(true);
            return undefined;
        }

        const observer = new IntersectionObserver((entries) => {
            if (!entries.some((entry) => entry.isIntersecting)) return;
            setShouldRender(true);
            observer.disconnect();
        }, {
            root: scrollRootRef?.current || null,
            rootMargin: '1200px 0px',
        });
        const element = holderRef.current;
        if (element) observer.observe(element);
        return () => observer.disconnect();
    }, [deferRender, isActive, scrollRootRef, shouldRender]);

    const handleLoadSuccess = useCallback((pdfPage) => {
        loadedPageRef.current = pdfPage;
        const viewport = pdfPage?.getViewport?.({ scale: 1 });
        const width = Number(viewport?.width);
        const height = Number(viewport?.height);
        if (Number.isFinite(width) && Number.isFinite(height) && width > 0 && height > 0) {
            setPageSize((current) => (
                current.width === width && current.height === height ? current : { width, height }
            ));
        }
        setLoadedVersion((version) => version + 1);
    }, []);

    const handleRenderSuccess = useCallback((pdfPage) => {
        renderedPageRef.current = pdfPage;
        setRenderedVersion((version) => version + 1);
    }, []);

    useEffect(() => {
        if (!isActive || !loadedPageRef.current) return;
        onLoadSuccess?.(loadedPageRef.current);
    }, [isActive, loadedVersion, onLoadSuccess]);

    useEffect(() => {
        if (!isActive || !renderedPageRef.current) return;
        onRenderSuccess?.(renderedPageRef.current);
    }, [isActive, onRenderSuccess, renderedVersion]);

    const renderedWidth = pageSize.width * scale;
    const renderedHeight = pageSize.height * scale;
    const rotationTransform = getPdfReaderRotationTransform({
        rotation,
        width: renderedWidth,
        height: renderedHeight,
    });
    const stageWidth = rotationTransform.stageWidth || renderedWidth;
    const stageHeight = rotationTransform.stageHeight || renderedHeight;
    const liveScaleRatio = scale > 0 ? displayScale / scale : 1;
    const activatePage = (event) => {
        event?.stopPropagation?.();
        onActivate?.(pageNumber);
    };

    return (
        <div
            ref={holderRef}
            data-pdf-page-number={pageNumber}
            data-pdf-passive-page={isActive ? undefined : 'true'}
            data-pdf-active-page={isActive ? 'true' : undefined}
            data-pdf-source-width={isActive ? renderedWidth : undefined}
            data-pdf-source-height={isActive ? renderedHeight : undefined}
            className={`pdf-page-frame relative shrink-0 overflow-hidden rounded-sm bg-white shadow-[0_2px_15px_rgba(0,0,0,0.06)] transition-shadow duration-200 ${
                onActivate ? 'cursor-pointer hover:shadow-[0_8px_24px_rgba(62,42,30,0.14)] focus:outline-none focus:ring-2 focus:ring-[#dc8a69]/45' : ''
            } ${darkMode ? 'pdf-page-frame--dark' : ''}`}
            style={{
                width: `${stageWidth}px`,
                height: `${stageHeight}px`,
                contentVisibility: deferRender && !isActive ? 'auto' : undefined,
                containIntrinsicSize: deferRender && !isActive ? `${stageHeight}px ${stageWidth}px` : undefined,
            }}
            role={onActivate ? 'button' : undefined}
            tabIndex={onActivate ? 0 : undefined}
            aria-label={onActivate ? `切换到第 ${pageNumber} 页` : undefined}
            onClick={onActivate ? activatePage : undefined}
            onKeyDown={onActivate ? (event) => {
                if (event.key !== 'Enter' && event.key !== ' ') return;
                event.preventDefault();
                activatePage();
            } : undefined}
        >
            {shouldRender ? (
                <div
                    ref={isActive ? viewerRef : undefined}
                    className={`pdf-page-stage relative ${darkMode ? 'pdf-page-stage--dark' : 'bg-white'}`}
                    style={{
                        width: renderedWidth,
                        minHeight: renderedHeight,
                        transform: rotationTransform.transform,
                        transformOrigin: 'top left',
                    }}
                >
                    <div
                        className={`pdf-page-render-layer ${darkMode ? 'pdf-page-render-layer--dimmed' : ''}`}
                        style={liveScaleRatio !== 1 ? {
                            transform: `scale(${liveScaleRatio})`,
                            transformOrigin: 'top left',
                        } : undefined}
                    >
                        {cachedImage && (
                            <img
                                src={cachedImage}
                                alt=""
                                className="pointer-events-none absolute left-0 top-0 z-0"
                            />
                        )}
                        <Page
                            inputRef={isActive ? pageInputRef : undefined}
                            pageNumber={pageNumber}
                            scale={scale}
                            devicePixelRatio={devicePixelRatio}
                            renderTextLayer={isActive}
                            renderAnnotationLayer={isActive}
                            onLoadSuccess={handleLoadSuccess}
                            onRenderSuccess={handleRenderSuccess}
                        />
                    </div>
                    {isActive ? children : null}
                </div>
            ) : (
                <div className={`flex h-full min-h-[240px] items-center justify-center text-xs ${darkMode ? 'text-gray-600' : 'text-gray-400'}`}>
                    加载第 {pageNumber} 页
                </div>
            )}
        </div>
    );
};

const PDFViewer = React.memo(forwardRef(({ pdfUrl, onTextSelect, highlightInfo = null, savedHighlights = EMPTY_SAVED_HIGHLIGHTS, onSavedHighlightClick, page = 1, onPageChange, isSelecting = false, onAreaSelected, onSelectionCancel, darkMode = false, onToggleSidebar, blockIndex = null, activeBlockId = null, focusedBlockIds = [], focusPulseToken = 0, navigationRequest = null, visitedBlockIds = [], inlineTranslationBlockIds = [], onBlockHover, onBlockClick, blockTranslations = {}, translatingBlockIds = [], hasDockedSelectionToolbar = false, selectionToolbar = null, translationSurface = 'panel', onTranslationSurfaceChange }, ref) => {
    const [numPages, setNumPages] = useState(null);
    const [pageNumber, setPageNumber] = useState(page || 1);
    const [scale, setScale] = useState(1.0);
    const [pageFlowMode, setPageFlowMode] = useState(PDF_READER_FLOW_MODES.PAGED);
    const [pageLayout, setPageLayout] = useState(PDF_READER_LAYOUTS.SINGLE);
    const [pageRotation, setPageRotation] = useState(0);
    const [isPageLayoutMenuOpen, setIsPageLayoutMenuOpen] = useState(false);
    const [pageBaseSize, setPageBaseSize] = useState({ width: 0, height: 0 });
    // 防抖缩放值：实际 PDF 渲染使用防抖后的值（150ms），避免频繁重渲染
    const debouncedScale = useDebouncedValue(scale, 150);
    const [selectedText, setSelectedText] = useState('');
    const [error, setError] = useState(null);
    const [hoveredBlockId, setHoveredBlockId] = useState(null);
    const [hoverTranslationBlockId, setHoverTranslationBlockId] = useState(null);
    const [isTranslationPositionPinned, setIsTranslationPositionPinned] = useState(false);
    // 吸附栏由父级统管：它和右侧阅读区的「页面翻译」卡片是同一件事的两个落点，
    // 同时开着只会重复占屏幕，所以状态提到 ChatPDF 里做互斥。
    const isTranslationDocked = translationSurface === 'dock';
    const translationSurfaceRef = useRef(translationSurface);
    const onTranslationSurfaceChangeRef = useRef(onTranslationSurfaceChange);
    useEffect(() => {
        translationSurfaceRef.current = translationSurface;
        onTranslationSurfaceChangeRef.current = onTranslationSurfaceChange;
    }, [onTranslationSurfaceChange, translationSurface]);
    // 身份恒定，等价于原来的 useState setter。
    // 这个文件里好几处 useCallback 没把它列进依赖，如果 shim 自己捕获 state，
    // 那些回调就会拿着冻住的旧值去取反，按钮变成点不动的死按钮。
    const setIsTranslationDocked = useCallback((next) => {
        const current = translationSurfaceRef.current === 'dock';
        const resolved = typeof next === 'function' ? next(current) : next;
        onTranslationSurfaceChangeRef.current?.(resolved ? 'dock' : 'panel');
    }, []);
    const [translationDockWidth, setTranslationDockWidth] = useState(TRANSLATION_DOCK_DEFAULT_WIDTH);
    const [floatingTranslationStyle, setFloatingTranslationStyle] = useState(null);
    const [hoverCorner, setHoverCorner] = useState('');
    const hoveredBlockIdRef = useRef(null);
    const hoverTranslationBlockIdRef = useRef(null);
    const popupHoveredRef = useRef(false);
    const hoverClearTimerRef = useRef(null);
    const hoverSwitchTimerRef = useRef(null);
    const hoverSwitchPointRef = useRef({ x: 0, y: 0 });
    const hoverSwitchTargetRef = useRef(null);
    const translationPanelDragRef = useRef({ dragging: false, start: { x: 0, y: 0 }, origin: null });
    const translationPanelResizeRef = useRef({ resizing: false, start: { x: 0, y: 0 }, origin: null });
    const translationDockResizeRef = useRef({ resizing: false, startX: 0, originWidth: TRANSLATION_DOCK_DEFAULT_WIDTH });
    const pdfDocumentRef = useRef(null);
    const pageRef = useRef(null);
    const pageLayoutMenuRef = useRef(null);
    const pdfScrollRef = useRef(null);
    const handledNavigationRevisionRef = useRef(null);
    const loadedPdfUrlRef = useRef('');
    const hasAutoFitRef = useRef(false);
    const backgroundDelayRef = useRef(null);
    const backgroundIdleTaskRef = useRef(null);
    const backgroundGenerationRef = useRef(0);
    const isDesktop = typeof window !== 'undefined' && window.chatpdfDesktop?.isDesktop === true;
    const [desktopApiBaseUrl, setDesktopApiBaseUrl] = useState('');

    useEffect(() => {
        if (typeof page === 'number' && page > 0 && page !== pageNumber) {
            setPageNumber(page);
        }
    }, [page, pageNumber]);

    useEffect(() => {
        if (!isPageLayoutMenuOpen) return undefined;
        const closeMenu = (event) => {
            if (event.key === 'Escape') {
                setIsPageLayoutMenuOpen(false);
                return;
            }
            if (event.type === 'pointerdown' && !pageLayoutMenuRef.current?.contains(event.target)) {
                setIsPageLayoutMenuOpen(false);
            }
        };
        window.addEventListener('keydown', closeMenu);
        window.addEventListener('pointerdown', closeMenu);
        return () => {
            window.removeEventListener('keydown', closeMenu);
            window.removeEventListener('pointerdown', closeMenu);
        };
    }, [isPageLayoutMenuOpen]);

    useEffect(() => {
        setIsTranslationPositionPinned(false);
        setFloatingTranslationStyle(null);
        hoveredBlockIdRef.current = null;
        hoverTranslationBlockIdRef.current = null;
        setHoveredBlockId(null);
        setHoverTranslationBlockId(null);
        popupHoveredRef.current = false;
        translationDockResizeRef.current = { resizing: false, startX: 0, originWidth: TRANSLATION_DOCK_DEFAULT_WIDTH };
    }, [pdfUrl, pageNumber]);

    // 退出吸附时复位悬浮态。开吸附时 toggleDockedTranslation 会把 popupHoveredRef 置 true，
    // 而吸附期间悬浮窗是被卸载的，onMouseLeave 永远不会补发。
    // 以前只有吸附栏自己的 X 按钮记得复位；现在右侧面板也能关，必须在这里统一收口，
    // 否则 updateHoveredBlock(null) 会被 popupHoveredRef 挡住，悬浮译文窗再也清不掉。
    useEffect(() => {
        if (isTranslationDocked) return;
        popupHoveredRef.current = false;
    }, [isTranslationDocked]);

    useEffect(() => {
        if (loadedPdfUrlRef.current === pdfUrl) return;
        setIsTranslationDocked(false);
        pdfDocumentRef.current = null;
        hasAutoFitRef.current = false;
        setNumPages(null);
        setError(null);
        setPageBaseSize({ width: 0, height: 0 });
        setPageRotation(0);
        setPageFlowMode(PDF_READER_FLOW_MODES.PAGED);
        setPageLayout(PDF_READER_LAYOUTS.SINGLE);
        setIsPageLayoutMenuOpen(false);
    }, [pdfUrl]);

    // 桌面模式下通过 preload IPC 获取后端地址；鉴权由主进程网络层处理。
    useEffect(() => {
        let cancelled = false;

        if (!isDesktop) return () => {};

        (async () => {
            try {
                const apiBaseUrl = await window.chatpdfDesktop.getApiBaseUrl();
                if (cancelled) return;
                setDesktopApiBaseUrl((apiBaseUrl || '').replace(/\/$/, ''));
            } catch (e) {
                console.warn('[PDFViewer] 获取桌面后端连接信息失败', e);
            }
        })();

        return () => {
            cancelled = true;
        };
    }, [isDesktop]);

    // 构建 PDF 完整 URL：桌面端使用后端地址，Web 端使用当前 origin
    const fullPdfUrl = useMemo(() => {
        if (!pdfUrl) return null;
        if (pdfUrl.startsWith('http://') || pdfUrl.startsWith('https://')) return pdfUrl;

        if (isDesktop) {
            if (!desktopApiBaseUrl) return null;
            return `${desktopApiBaseUrl}${pdfUrl.startsWith('/') ? '' : '/'}${pdfUrl}`;
        }

        const origin = typeof window !== 'undefined' ? window.location.origin : '';
        if (!origin) return pdfUrl;
        return `${origin}${pdfUrl.startsWith('/') ? '' : '/'}${pdfUrl}`;
    }, [pdfUrl, isDesktop, desktopApiBaseUrl]);
    const pdfCacheKey = fullPdfUrl || pdfUrl || '';
    const renderPixelRatio = useMemo(() => {
        if (typeof window === 'undefined') return 1;
        return Math.min(Math.max(window.devicePixelRatio || 1, 1), 2);
    }, []);
    const normalizedPageRotation = normalizePdfReaderRotation(pageRotation);
    const pageRenderedSize = useMemo(() => ({
        width: Math.max(0, Number(pageBaseSize.width) || 0) * debouncedScale,
        height: Math.max(0, Number(pageBaseSize.height) || 0) * debouncedScale,
    }), [debouncedScale, pageBaseSize.height, pageBaseSize.width]);
    const pageRotationTransform = useMemo(() => getPdfReaderRotationTransform({
        rotation: normalizedPageRotation,
        width: pageRenderedSize.width,
        height: pageRenderedSize.height,
    }), [normalizedPageRotation, pageRenderedSize.height, pageRenderedSize.width]);
    const displayPageNumbers = useMemo(() => getPdfReaderDisplayPages({
        totalPages: numPages,
        pageNumber,
        flowMode: pageFlowMode,
        layout: pageLayout,
    }), [numPages, pageFlowMode, pageLayout, pageNumber]);
    const isContinuousReading = pageFlowMode === PDF_READER_FLOW_MODES.CONTINUOUS;
    const pageIndicator = isContinuousReading
        ? String(pageNumber)
        : displayPageNumbers.join('–') || String(pageNumber);
    const previousPageTarget = useMemo(() => getPdfReaderNavigationTarget({
        totalPages: numPages,
        pageNumber,
        flowMode: pageFlowMode,
        layout: pageLayout,
        direction: -1,
    }), [numPages, pageFlowMode, pageLayout, pageNumber]);
    const nextPageTarget = useMemo(() => getPdfReaderNavigationTarget({
        totalPages: numPages,
        pageNumber,
        flowMode: pageFlowMode,
        layout: pageLayout,
        direction: 1,
    }), [numPages, pageFlowMode, pageLayout, pageNumber]);
    const activePageStageStyle = pageRotationTransform.stageWidth > 0 && pageRotationTransform.stageHeight > 0
        ? {
            width: pageRotationTransform.stageWidth,
            height: pageRotationTransform.stageHeight,
        }
        : undefined;
    const readerPageStackClassName = isContinuousReading
        ? (pageLayout === PDF_READER_LAYOUTS.SINGLE
            ? 'flex min-w-full flex-col items-center gap-6'
            : 'grid min-w-full w-max grid-cols-2 items-start justify-items-center gap-6')
        : (displayPageNumbers.length > 1
            ? 'flex min-h-full min-w-full items-start justify-center gap-6'
            : 'flex min-h-full min-w-full items-start justify-center');

    // Electron 主进程会为本机后端请求注入 token，渲染进程不持有凭据。
    const pdfFile = useMemo(() => {
        if (!fullPdfUrl) return null;
        return fullPdfUrl;
    }, [fullPdfUrl]);

    function onDocumentLoadSuccess(pdfDocument) {
        const { numPages } = pdfDocument;
        pdfDocumentRef.current = pdfDocument;
        loadedPdfUrlRef.current = pdfUrl;
        setNumPages(numPages);
        setError(null);
        setPageNumber(prev => {
            const safePage = Math.min(Math.max(prev, 1), numPages);
            if (onPageChange && safePage !== prev) {
                onPageChange(safePage);
            }
            return safePage;
        });
    }

    const updatePageBaseSize = useCallback((pdfPage) => {
        const viewport = pdfPage?.getViewport?.({ scale: 1 });
        const width = Number(viewport?.width);
        const height = Number(viewport?.height);
        if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return;
        setPageBaseSize((current) => (
            current.width === width && current.height === height ? current : { width, height }
        ));
    }, []);

    // 首次加载按容器宽度自适应缩放（fit-width）；只做一次，不覆盖用户手动缩放
    const handleFirstPageLoad = useCallback((page) => {
        updatePageBaseSize(page);
        if (hasAutoFitRef.current) return;
        const el = pdfScrollRef.current;
        if (!el) return;
        const naturalWidth = page.getViewport({ scale: 1 }).width;
        const available = el.clientWidth - 48;
        if (naturalWidth > 0 && available > 200) {
            hasAutoFitRef.current = true;
            const fit = Math.min(Math.max(available / naturalWidth, 0.9), 1.75);
            setScale(Math.round(fit * 20) / 20);
        }
    }, [updatePageBaseSize]);

    function onDocumentLoadError(error) {
        console.error('❌ PDF load error:', error);
        setError(error.message || 'Failed to load PDF');
    }

    const getCurrentPageMetrics = useCallback((pageElement = pageRef.current) => {
        if (!pageElement) return null;
        const bounds = pageElement.getBoundingClientRect();
        const inferredWidth = normalizedPageRotation % 180 === 0 ? bounds.width : bounds.height;
        const inferredHeight = normalizedPageRotation % 180 === 0 ? bounds.height : bounds.width;
        const width = pageRenderedSize.width || inferredWidth;
        const height = pageRenderedSize.height || inferredHeight;
        if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return null;
        return { bounds, width, height };
    }, [normalizedPageRotation, pageRenderedSize.height, pageRenderedSize.width]);

    const clientRectsToPageLocal = useCallback((clientRects, pageElement, padding = 1) => {
        const metrics = getCurrentPageMetrics(pageElement);
        if (!metrics) return [];
        const localRects = Array.from(clientRects || [])
            .filter((rect) => (
                rect.width > 1
                && rect.height > 1
                && rect.right > metrics.bounds.left
                && rect.left < metrics.bounds.right
                && rect.bottom > metrics.bounds.top
                && rect.top < metrics.bounds.bottom
            ))
            .map((rect) => mapPdfReaderDisplayRectToPage({
                left: rect.left - metrics.bounds.left,
                top: rect.top - metrics.bounds.top,
                width: rect.width,
                height: rect.height,
                pageWidth: metrics.width,
                pageHeight: metrics.height,
                rotation: normalizedPageRotation,
            }))
            .map((rect) => ({
                ...rect,
                right: rect.left + rect.width,
                bottom: rect.top + rect.height,
            }));
        return mergeClientRectsByLine(localRects, { left: 0, top: 0 }, padding);
    }, [getCurrentPageMetrics, normalizedPageRotation]);

    const handleTextSelection = () => {
        const selection = window.getSelection();
        const text = selection.toString().trim();
        if (text) {
            setSelectedText(text);
            if (onTextSelect) {
                let anchor = null;
                const pageElement = pageRef.current;
                if (selection.rangeCount > 0 && pageElement && debouncedScale > 0) {
                    const range = selection.getRangeAt(0);
                    const ancestor = range.commonAncestorContainer;
                    const ancestorElement = ancestor?.nodeType === 1 ? ancestor : ancestor?.parentElement;
                    if (ancestorElement && pageElement.contains(ancestorElement)) {
                        const metrics = getCurrentPageMetrics(pageElement);
                        const rects = clientRectsToPageLocal(range.getClientRects(), pageElement, 1).map((rect) => ({
                            left: rect.left / debouncedScale,
                            top: rect.top / debouncedScale,
                            width: rect.width / debouncedScale,
                            height: rect.height / debouncedScale,
                        }));
                        anchor = {
                            page: pageNumber,
                            rects,
                            coordinate_space: 'pdf_top_left_points',
                            page_size: metrics
                                ? [metrics.width / debouncedScale, metrics.height / debouncedScale]
                                : undefined,
                        };
                    }
                }
                onTextSelect(text, anchor);
            }
        }
    };

    const scrollToReaderPage = useCallback((targetPage, behavior = 'smooth') => {
        const target = pdfScrollRef.current?.querySelector?.(`[data-pdf-page-number="${targetPage}"]`);
        target?.scrollIntoView?.({ block: 'start', inline: 'center', behavior });
    }, []);

    const scrollToReaderTarget = useCallback((targetPage, blockId, behavior = 'smooth') => {
        const scroller = pdfScrollRef.current;
        const pageElement = scroller?.querySelector?.(`[data-pdf-page-number="${targetPage}"]`);
        if (!scroller || !pageElement) return false;

        const pageData = getPageBlockData(blockIndex, targetPage);
        const block = blockId
            ? pageData?.blocks?.find((item) => item?.block_id === blockId)
            : null;
        const bbox = normalizeBlockBBox(block?.bbox);
        if (!bbox) {
            pageElement.scrollIntoView?.({ block: 'start', inline: 'center', behavior });
            return true;
        }

        const sourceWidth = Math.max(1, Number(pageData?.width_pts) || Number(pageBaseSize.width) || 612);
        const sourceHeight = Math.max(1, Number(pageData?.height_pts) || Number(pageBaseSize.height) || 792);
        const renderedWidth = sourceWidth * scale;
        const renderedHeight = sourceHeight * scale;
        const displayRect = mapPdfReaderPageRectToDisplay({
            left: bbox[0] * scale,
            top: bbox[1] * scale,
            width: (bbox[2] - bbox[0]) * scale,
            height: (bbox[3] - bbox[1]) * scale,
            pageWidth: renderedWidth,
            pageHeight: renderedHeight,
            rotation: normalizedPageRotation,
        });
        const pageBounds = pageElement.getBoundingClientRect();
        const viewportBounds = scroller.getBoundingClientRect();
        const targetTop = scroller.scrollTop + pageBounds.top - viewportBounds.top + displayRect.top;
        const targetLeft = scroller.scrollLeft + pageBounds.left - viewportBounds.left
            + displayRect.left + displayRect.width / 2;
        const contextOffset = Math.min(140, Math.max(48, scroller.clientHeight * 0.16));
        const nextTop = clampNumber(
            targetTop - contextOffset,
            0,
            Math.max(0, scroller.scrollHeight - scroller.clientHeight)
        );
        const nextLeft = clampNumber(
            targetLeft - scroller.clientWidth / 2,
            0,
            Math.max(0, scroller.scrollWidth - scroller.clientWidth)
        );

        if (typeof scroller.scrollTo === 'function') {
            scroller.scrollTo({ top: nextTop, left: nextLeft, behavior });
        } else {
            scroller.scrollTop = nextTop;
            scroller.scrollLeft = nextLeft;
        }
        return true;
    }, [blockIndex, normalizedPageRotation, pageBaseSize.height, pageBaseSize.width, scale]);

    useEffect(() => {
        const revision = navigationRequest?.revision;
        if (revision == null || revision === handledNavigationRevisionRef.current) return;
        if (!isContinuousReading || typeof window === 'undefined') {
            handledNavigationRevisionRef.current = revision;
            return;
        }

        const requestedPage = Math.max(1, Number(navigationRequest.page) || 1);
        const totalPages = Math.max(1, Number(numPages) || requestedPage);
        const targetPage = Math.min(totalPages, requestedPage);
        const schedule = window.requestAnimationFrame || ((callback) => window.setTimeout(callback, 0));
        const cancelScheduled = window.cancelAnimationFrame || window.clearTimeout;
        let cancelled = false;
        let frameId = null;
        let attempts = 0;

        const navigate = () => {
            if (cancelled) return;
            if (scrollToReaderTarget(targetPage, navigationRequest.blockId, 'smooth')) {
                handledNavigationRevisionRef.current = revision;
                return;
            }
            attempts += 1;
            if (attempts < 12) {
                frameId = schedule(navigate);
            }
        };

        frameId = schedule(navigate);
        return () => {
            cancelled = true;
            if (frameId != null) cancelScheduled(frameId);
        };
    }, [isContinuousReading, navigationRequest, numPages, scrollToReaderTarget]);

    const updateReaderPage = useCallback((nextPage, { scroll = false, behavior = 'smooth' } = {}) => {
        const total = Math.max(1, Number(numPages) || 1);
        const safePage = Math.min(total, Math.max(1, Math.round(Number(nextPage) || 1)));
        if (safePage !== pageNumber) {
            setPageNumber(safePage);
            onPageChange?.(safePage);
        }
        if (scroll && typeof window !== 'undefined') {
            const schedule = window.requestAnimationFrame || ((callback) => window.setTimeout(callback, 0));
            schedule(() => scrollToReaderPage(safePage, behavior));
        }
    }, [numPages, onPageChange, pageNumber, scrollToReaderPage]);

    const changePage = useCallback((direction) => {
        const nextPage = getPdfReaderNavigationTarget({
            totalPages: numPages,
            pageNumber,
            flowMode: pageFlowMode,
            layout: pageLayout,
            direction,
        });
        if (nextPage === pageNumber) return;
        updateReaderPage(nextPage, { scroll: pageFlowMode === PDF_READER_FLOW_MODES.CONTINUOUS });
    }, [numPages, pageFlowMode, pageLayout, pageNumber, updateReaderPage]);

    const selectPageFlowMode = useCallback((nextMode) => {
        if (nextMode === pageFlowMode) return;
        setPageFlowMode(nextMode);
        if (nextMode === PDF_READER_FLOW_MODES.PAGED) {
            const firstVisiblePage = getPdfReaderDisplayPages({
                totalPages: numPages,
                pageNumber,
                flowMode: nextMode,
                layout: pageLayout,
            })[0];
            updateReaderPage(firstVisiblePage || pageNumber);
            return;
        }
        if (typeof window !== 'undefined') {
            const schedule = window.requestAnimationFrame || ((callback) => window.setTimeout(callback, 0));
            schedule(() => scrollToReaderPage(pageNumber, 'auto'));
        }
    }, [numPages, pageFlowMode, pageLayout, pageNumber, scrollToReaderPage, updateReaderPage]);

    const selectPageLayout = useCallback((nextLayout) => {
        if (nextLayout === pageLayout) return;
        setPageLayout(nextLayout);
        if (!isContinuousReading) {
            const firstVisiblePage = getPdfReaderDisplayPages({
                totalPages: numPages,
                pageNumber,
                flowMode: pageFlowMode,
                layout: nextLayout,
            })[0];
            updateReaderPage(firstVisiblePage || pageNumber);
            return;
        }
        if (typeof window !== 'undefined') {
            const schedule = window.requestAnimationFrame || ((callback) => window.setTimeout(callback, 0));
            schedule(() => scrollToReaderPage(pageNumber, 'auto'));
        }
    }, [isContinuousReading, numPages, pageFlowMode, pageLayout, pageNumber, scrollToReaderPage, updateReaderPage]);

    const rotateReader = useCallback((direction) => {
        setPageRotation((current) => rotatePdfReader(current, direction));
    }, []);

    const zoomIn = () => setScale(prev => Math.min(prev + 0.2, 3.0));
    const zoomOut = () => setScale(prev => Math.max(prev - 0.2, 0.5));

    const [highlightRect, setHighlightRect] = useState(null);
    const [highlightRects, setHighlightRects] = useState([]);
    const [savedHighlightRects, setSavedHighlightRects] = useState([]);
    const pageRenderEpoch = useMemo(() => ({
        documentKey: pdfCacheKey,
        pageNumber,
        scale: debouncedScale,
    }), [debouncedScale, pageNumber, pdfCacheKey]);
    const activePageRenderEpochRef = useRef(pageRenderEpoch);
    activePageRenderEpochRef.current = pageRenderEpoch;
    const [renderedPageEpoch, setRenderedPageEpoch] = useState(null);

    // ── PDF 页面 canvas 缓存：渲染完成后捕获 canvas 数据 ──
    // 缓存的图片 dataURL，用于在页面加载/重渲染期间显示占位图
    const [cachedImage, setCachedImage] = useState(() =>
        pdfPageCache.get(pageNumber, scale, pdfCacheKey) || null
    );

    // 页码或缩放变化时，立即尝试从缓存获取占位图
    useEffect(() => {
        const cached = pdfPageCache.get(pageNumber, debouncedScale, pdfCacheKey);
        setCachedImage(cached || null);
    }, [pdfCacheKey, pageNumber, debouncedScale]);

    const cancelBackgroundWork = useCallback(() => {
        backgroundGenerationRef.current += 1;
        if (backgroundDelayRef.current) {
            window.clearTimeout(backgroundDelayRef.current);
            backgroundDelayRef.current = null;
        }
        cancelIdleTask(backgroundIdleTaskRef.current);
        backgroundIdleTaskRef.current = null;
    }, []);

    useEffect(() => {
        cancelBackgroundWork();
        return cancelBackgroundWork;
    }, [pdfCacheKey, pageNumber, debouncedScale, cancelBackgroundWork]);

    // 首屏绘制完成后再异步缓存，并只预取下一页的解析数据，避免与当前页抢资源。
    const handlePageRenderSuccess = useCallback((renderedPage) => {
        const renderedPageNumber = Number(renderedPage?.pageNumber);
        if (
            activePageRenderEpochRef.current !== pageRenderEpoch
            || (Number.isFinite(renderedPageNumber) && renderedPageNumber !== pageRenderEpoch.pageNumber)
        ) {
            return;
        }
        updatePageBaseSize(renderedPage);
        setRenderedPageEpoch(pageRenderEpoch);
        cancelBackgroundWork();
        const generation = backgroundGenerationRef.current;
        const cachePage = pageNumber;
        const cacheScale = debouncedScale;
        const cacheDocumentKey = pdfCacheKey;
        const pdfDocument = pdfDocumentRef.current;

        backgroundDelayRef.current = window.setTimeout(() => {
            backgroundDelayRef.current = null;
            if (generation !== backgroundGenerationRef.current) return;

            backgroundIdleTaskRef.current = scheduleIdleTask(() => {
                backgroundIdleTaskRef.current = null;
                if (generation !== backgroundGenerationRef.current) return;

                const pageEl = pageRef.current;
                const canvas = pageEl?.querySelector('canvas');
                if (canvas?.toBlob && !pdfPageCache.has(cachePage, cacheScale, cacheDocumentKey)) {
                    canvas.toBlob((blob) => {
                        if (!blob || generation !== backgroundGenerationRef.current || typeof FileReader === 'undefined') return;
                        const reader = new FileReader();
                        reader.onload = () => {
                            if (generation !== backgroundGenerationRef.current || typeof reader.result !== 'string') return;
                            pdfPageCache.set(cachePage, cacheScale, reader.result, cacheDocumentKey);
                        };
                        reader.readAsDataURL(blob);
                    }, 'image/webp', 0.86);
                }

                const nextPage = cachePage < numPages ? cachePage + 1 : null;
                if (nextPage && typeof pdfDocument?.getPage === 'function') {
                    pdfDocument
                        .getPage(nextPage)
                        .then((pageProxy) => pageProxy.getOperatorList())
                        .catch(() => {});
                }
            }, 1200);
        }, 180);
    }, [cancelBackgroundWork, debouncedScale, numPages, pageNumber, pageRenderEpoch, pdfCacheKey, updatePageBaseSize]);

    const currentBlockPage = useMemo(
        () => getPageBlockData(blockIndex, pageNumber),
        [blockIndex, pageNumber]
    );

    const currentBlocks = useMemo(
        () => (currentBlockPage?.blocks || []).filter((block) => isInteractiveBlock(block) && normalizeBlockBBox(block.bbox)),
        [currentBlockPage]
    );

    const findBlockAtPoint = useCallback((clientX, clientY) => {
        if (isSelecting || currentBlocks.length === 0) return null;
        const metrics = getCurrentPageMetrics();
        if (!metrics) return null;
        const pageRect = metrics.bounds;
        if (
            clientX < pageRect.left ||
            clientX > pageRect.right ||
            clientY < pageRect.top ||
            clientY > pageRect.bottom
        ) {
            return null;
        }

        const point = mapPdfReaderDisplayPointToPage({
            x: clientX - pageRect.left,
            y: clientY - pageRect.top,
            width: metrics.width,
            height: metrics.height,
            rotation: normalizedPageRotation,
        });
        const x = point.x / debouncedScale;
        const y = point.y / debouncedScale;
        const matches = currentBlocks
            .map((block) => ({ block, bbox: normalizeBlockBBox(block.bbox) }))
            .filter(({ bbox }) => bbox && x >= bbox[0] && x <= bbox[2] && y >= bbox[1] && y <= bbox[3])
            .sort((a, b) => {
                const areaA = (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1]);
                const areaB = (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]);
                const priorityDiff = blockHitPriority(b.block) - blockHitPriority(a.block);
                return priorityDiff || areaA - areaB;
            });

        return matches[0]?.block || null;
    }, [currentBlocks, debouncedScale, getCurrentPageMetrics, isSelecting, normalizedPageRotation]);

    const updateHoveredBlock = useCallback((block, point = null) => {
        const nextId = block?.block_id || null;
        if (hoverClearTimerRef.current) {
            clearTimeout(hoverClearTimerRef.current);
            hoverClearTimerRef.current = null;
        }
        if (!nextId) {
            if (hoverSwitchTimerRef.current) {
                clearTimeout(hoverSwitchTimerRef.current);
                hoverSwitchTimerRef.current = null;
            }
            hoverSwitchTargetRef.current = null;
            if (isTranslationPositionPinned || isTranslationDocked) return;
            if (popupHoveredRef.current) return;
            if (hoveredBlockIdRef.current === nextId) return;
            hoverClearTimerRef.current = setTimeout(() => {
                if (popupHoveredRef.current) return;
                hoveredBlockIdRef.current = null;
                hoverTranslationBlockIdRef.current = null;
                setHoveredBlockId(null);
                setHoverTranslationBlockId(null);
                onBlockHover?.(null);
            }, 180);
            return;
        }
        if (hoveredBlockIdRef.current === nextId) return;
        hoveredBlockIdRef.current = nextId;
        setHoveredBlockId(nextId);
        onBlockHover?.(block || null);
        if (hoverSwitchTimerRef.current) {
            clearTimeout(hoverSwitchTimerRef.current);
        }
        hoverSwitchPointRef.current = point || { x: 0, y: 0 };
        hoverSwitchTargetRef.current = nextId;
        hoverSwitchTimerRef.current = setTimeout(() => {
            if (hoveredBlockIdRef.current !== nextId) return;
            if (hoverSwitchTargetRef.current !== nextId) return;
            hoverTranslationBlockIdRef.current = nextId;
            setHoverTranslationBlockId(nextId);
            hoverSwitchTimerRef.current = null;
            hoverSwitchTargetRef.current = null;
        }, HOVER_TRANSLATION_SWITCH_DELAY);
    }, [isTranslationDocked, isTranslationPositionPinned, onBlockHover]);

    const handleBlockMouseMove = useCallback((event) => {
        const point = { x: event.clientX, y: event.clientY };
        const targetBlock = findBlockAtPoint(point.x, point.y);
        if (hoverSwitchTimerRef.current && targetBlock?.block_id === hoverSwitchTargetRef.current) {
            const dx = point.x - hoverSwitchPointRef.current.x;
            const dy = point.y - hoverSwitchPointRef.current.y;
            if (Math.hypot(dx, dy) > HOVER_TRANSLATION_MOVE_TOLERANCE) {
                clearTimeout(hoverSwitchTimerRef.current);
                hoverSwitchTimerRef.current = null;
                hoverSwitchPointRef.current = point;
                hoverSwitchTimerRef.current = setTimeout(() => {
                    if (hoveredBlockIdRef.current !== targetBlock.block_id) return;
                    if (hoverSwitchTargetRef.current !== targetBlock.block_id) return;
                    hoverTranslationBlockIdRef.current = targetBlock.block_id;
                    setHoverTranslationBlockId(targetBlock.block_id);
                    hoverSwitchTimerRef.current = null;
                    hoverSwitchTargetRef.current = null;
                }, HOVER_TRANSLATION_SWITCH_DELAY);
            }
        }
        updateHoveredBlock(targetBlock, point);
    }, [findBlockAtPoint, updateHoveredBlock]);

    const handleBlockMouseLeave = useCallback(() => {
        updateHoveredBlock(null);
    }, [updateHoveredBlock]);

    const findSavedHighlightAtPoint = useCallback((clientX, clientY) => {
        const metrics = getCurrentPageMetrics();
        if (!metrics) return null;
        const pageRect = metrics.bounds;
        const point = mapPdfReaderDisplayPointToPage({
            x: clientX - pageRect.left,
            y: clientY - pageRect.top,
            width: metrics.width,
            height: metrics.height,
            rotation: normalizedPageRotation,
        });
        const localX = point.x;
        const localY = point.y;
        const hitSlop = 3;
        const hit = [...savedHighlightRects].reverse().find(({ rect }) => (
            localX >= rect.left - hitSlop
            && localX <= rect.left + rect.width + hitSlop
            && localY >= rect.top - hitSlop
            && localY <= rect.top + rect.height + hitSlop
        ));
        return hit?.annotation || null;
    }, [getCurrentPageMetrics, normalizedPageRotation, savedHighlightRects]);

    const handleBlockClick = useCallback((event) => {
        const selection = window.getSelection?.();
        const selectionText = selection?.toString().trim() || '';
        if (selectionText) return;
        const savedHighlight = findSavedHighlightAtPoint(event.clientX, event.clientY);
        if (savedHighlight) {
            event.stopPropagation();
            onSavedHighlightClick?.(savedHighlight);
            return;
        }
        const block = findBlockAtPoint(event.clientX, event.clientY);
        if (block) {
            onBlockClick?.(block);
        }
    }, [findBlockAtPoint, findSavedHighlightAtPoint, onBlockClick, onSavedHighlightClick]);

    useEffect(() => () => {
        if (hoverClearTimerRef.current) {
            clearTimeout(hoverClearTimerRef.current);
        }
        if (hoverSwitchTimerRef.current) {
            clearTimeout(hoverSwitchTimerRef.current);
        }
    }, []);

    // ── 自定义滚动条 ──
    const THUMB_SIZE = 48;
    const [vThumb, setVThumb] = useState({ top: 0, visible: false });
    const [hThumb, setHThumb] = useState({ left: 0, visible: false });
    const isDragging = useRef(false);
    const dragStart = useRef({});

    const updateThumbs = useCallback(() => {
        const el = pdfScrollRef.current;
        if (!el) return;
        const { scrollTop, scrollHeight, clientHeight, scrollLeft, scrollWidth, clientWidth } = el;
        setVThumb(scrollHeight > clientHeight
            ? { visible: true, top: 8 + (scrollTop / (scrollHeight - clientHeight)) * (clientHeight - THUMB_SIZE - 16) }
            : { visible: false, top: 0 });
        setHThumb(scrollWidth > clientWidth
            ? { visible: true, left: 8 + (scrollLeft / (scrollWidth - clientWidth)) * (clientWidth - THUMB_SIZE - 16) }
            : { visible: false, left: 0 });
    }, []);

    const handlePdfScroll = useCallback((event) => {
        updateThumbs();
        if (!isContinuousReading) return;

        const scroller = event.currentTarget || pdfScrollRef.current;
        if (!scroller) return;
        const viewport = scroller.getBoundingClientRect();
        const readingLine = viewport.top + Math.min(viewport.height * 0.32, 220);
        let closestPage = null;
        let closestDistance = Number.POSITIVE_INFINITY;

        scroller.querySelectorAll('[data-pdf-page-number]').forEach((element) => {
            const candidate = Number(element.dataset.pdfPageNumber);
            if (!Number.isFinite(candidate) || candidate < 1) return;
            const bounds = element.getBoundingClientRect();
            const distance = Math.abs(bounds.top + Math.min(bounds.height * 0.38, 180) - readingLine);
            if (distance < closestDistance) {
                closestDistance = distance;
                closestPage = candidate;
            }
        });

        if (closestPage && closestPage !== pageNumber) {
            setPageNumber(closestPage);
            onPageChange?.(closestPage);
        }
    }, [isContinuousReading, onPageChange, pageNumber, updateThumbs]);

    useEffect(() => {
        const el = pdfScrollRef.current;
        if (!el) return;
        const ro = new ResizeObserver(updateThumbs);
        ro.observe(el);
        const t = setTimeout(updateThumbs, 100);
        return () => { ro.disconnect(); clearTimeout(t); };
    }, [updateThumbs]);

    // 当缩放比例、页码或总页数变化时（PDF 重新渲染后），重新计算滚动条可见性
    useEffect(() => {
        const t = setTimeout(updateThumbs, 300);
        return () => clearTimeout(t);
    }, [scale, debouncedScale, pageNumber, numPages, pageFlowMode, pageLayout, updateThumbs]);

    const makeDragHandler = useCallback((axis) => (e) => {
        e.preventDefault();
        e.stopPropagation();
        isDragging.current = true;
        const el = pdfScrollRef.current;
        dragStart.current = {
            x: e.clientX, y: e.clientY,
            scrollLeft: el.scrollLeft, scrollTop: el.scrollTop,
        };
        document.body.style.userSelect = 'none';
        document.body.style.cursor = 'grabbing';
        const onMove = (e) => {
            const el = pdfScrollRef.current;
            if (!el) return;
            if (axis === 'v') {
                const dy = e.clientY - dragStart.current.y;
                const trackH = el.clientHeight - THUMB_SIZE - 16;
                el.scrollTop = dragStart.current.scrollTop + (dy / trackH) * (el.scrollHeight - el.clientHeight);
            } else {
                const dx = e.clientX - dragStart.current.x;
                const trackW = el.clientWidth - THUMB_SIZE - 16;
                el.scrollLeft = dragStart.current.scrollLeft + (dx / trackW) * (el.scrollWidth - el.clientWidth);
            }
        };
        const onUp = () => {
            isDragging.current = false;
            document.body.style.userSelect = '';
            document.body.style.cursor = '';
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
        };
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
    }, []);


    const scrollHighlightIntoView = useCallback((rects) => {
        const firstRect = Array.isArray(rects) ? rects[0] : null;
        const scroller = pdfScrollRef.current;
        const pageElement = pageRef.current;
        if (!firstRect || !scroller || !pageElement) return;

        const pageBounds = pageElement.getBoundingClientRect();
        const viewportBounds = scroller.getBoundingClientRect();
        const metrics = getCurrentPageMetrics(pageElement);
        const displayRect = metrics
            ? mapPdfReaderPageRectToDisplay({
                ...firstRect,
                pageWidth: metrics.width,
                pageHeight: metrics.height,
                rotation: normalizedPageRotation,
            })
            : firstRect;
        const targetTop = scroller.scrollTop + pageBounds.top - viewportBounds.top + displayRect.top;
        const targetLeft = scroller.scrollLeft + pageBounds.left - viewportBounds.left + displayRect.left;
        const targetBottom = targetTop + displayRect.height;
        const targetRight = targetLeft + displayRect.width;
        const margin = 48;
        const verticallyVisible = (
            targetTop >= scroller.scrollTop + margin
            && targetBottom <= scroller.scrollTop + scroller.clientHeight - margin
        );
        const horizontallyVisible = (
            targetLeft >= scroller.scrollLeft + margin
            && targetRight <= scroller.scrollLeft + scroller.clientWidth - margin
        );
        if (verticallyVisible && horizontallyVisible) return;

        const nextTop = clampNumber(
            targetTop - scroller.clientHeight * 0.35,
            0,
            Math.max(0, scroller.scrollHeight - scroller.clientHeight)
        );
        const nextLeft = clampNumber(
            targetLeft - scroller.clientWidth * 0.35,
            0,
            Math.max(0, scroller.scrollWidth - scroller.clientWidth)
        );
        if (typeof scroller.scrollTo === 'function') {
            scroller.scrollTo({ top: nextTop, left: nextLeft, behavior: 'smooth' });
        } else {
            scroller.scrollTop = nextTop;
            scroller.scrollLeft = nextLeft;
        }
    }, [getCurrentPageMetrics, normalizedPageRotation]);

    useEffect(() => {
        let isMounted = true;
        let retryTimer = null;
        let retryCount = 0;
        const MAX_RETRIES = 15; // 最多重试 15 次（约 1.5 秒）

        const citationAnchor = highlightInfo?.citationAnchor || {};
        const hasCitationAnchor = Boolean(
            citationAnchor.blockId
            || normalizeCitationBBox(citationAnchor.bbox)
            || (Array.isArray(citationAnchor.rects) && citationAnchor.rects.length > 0)
        );
        if (!highlightInfo || (!highlightInfo.text && !hasCitationAnchor)) {
            setHighlightRect(null);
            setHighlightRects([]);
            return;
        }

        // 使用 prop page 作为目标页码（而非内部 pageNumber 状态），避免竞态条件
        const targetPage = highlightInfo.page;
        if (targetPage !== pageNumber) {
            // 页面还没切换到位，等下一次 pageNumber 更新后再匹配
            setHighlightRect(null);
            setHighlightRects([]);
            return;
        }

        // pageNumber 更新时 pageRef 仍可能指向旧页。只在当前渲染代际完成后消费引用定位，
        // onRenderSuccess 会更新 renderedPageEpoch 并触发本 effect 重试。
        if (renderedPageEpoch !== pageRenderEpoch) {
            setHighlightRect(null);
            setHighlightRects([]);
            return;
        }

        const geometryCurrent = isCitationGeometryCurrent(highlightInfo, blockIndex);
        const anchorBBox = geometryCurrent
            ? normalizeCitationBBox(citationAnchor.bbox)
            : null;
        const blockMatch = findBestCitationBlock({
            blocks: currentBlocks,
            text: highlightInfo.text,
            startPhrase: highlightInfo.startPhrase,
            endPhrase: highlightInfo.endPhrase,
            blockId: citationAnchor.blockId,
            allowBlockId: geometryCurrent,
        });
        const spatialBBox = anchorBBox || normalizeCitationBBox(blockMatch?.block?.bbox);
        const initialPageElement = pageRef.current;
        const renderedPageSize = pageRenderedSize.width > 0 && pageRenderedSize.height > 0 && debouncedScale > 0
            ? {
                width: pageRenderedSize.width / debouncedScale,
                height: pageRenderedSize.height / debouncedScale,
            }
            : null;
        const renderOptions = {
            pageData: currentBlockPage,
            renderedPageSize,
            scale: debouncedScale,
        };
        const explicitRects = geometryCurrent
            ? citationRectsToRendered({
                rects: citationAnchor.rects,
                coordinateSpace: citationAnchor.coordinateSpace,
                pageSize: citationAnchor.pageSize,
                ...renderOptions,
            })
            : [];
        if (explicitRects.length > 0) {
            setHighlightRect(explicitRects[0]);
            setHighlightRects(explicitRects);
            scrollHighlightIntoView(explicitRects);
            return;
        }

        const spatialFallbackRects = spatialBBox
            ? citationRectsToRendered({
                bbox: spatialBBox,
                coordinateSpace: anchorBBox
                    ? citationAnchor.coordinateSpace
                    : 'pdf_top_left_points',
                pageSize: anchorBBox
                    ? citationAnchor.pageSize
                    : [currentBlockPage?.width_pts, currentBlockPage?.height_pts],
                ...renderOptions,
            })
            : [];
        setHighlightRect(spatialFallbackRects[0] || null);
        setHighlightRects(spatialFallbackRects);
        if (spatialFallbackRects.length > 0) {
            scrollHighlightIntoView(spatialFallbackRects);
        }
        if (!highlightInfo.text) return;

        const findHighlight = () => {
            if (!isMounted) return;

            const pageElement = pageRef.current;
            if (!pageElement) {
                if (retryCount < MAX_RETRIES) {
                    retryCount++;
                    retryTimer = setTimeout(findHighlight, 100);
                }
                return;
            }

            const textLayer = pageElement.querySelector('.react-pdf__Page__textContent');
            if (!textLayer || textLayer.children.length === 0) {
                // 文本层尚未渲染完成，重试
                if (retryCount < MAX_RETRIES) {
                    retryCount++;
                    retryTimer = setTimeout(findHighlight, 100);
                }
                return;
            }

            try {
                const allSpans = Array.from(textLayer.querySelectorAll('span'));
                const pageRect = pageElement.getBoundingClientRect();
                let spans = allSpans;

                if (spatialFallbackRects.length > 0) {
                    const constraint = spatialFallbackRects[0];
                    const metrics = getCurrentPageMetrics(pageElement);
                    const displayConstraint = metrics
                        ? mapPdfReaderPageRectToDisplay({
                            ...constraint,
                            pageWidth: metrics.width,
                            pageHeight: metrics.height,
                            rotation: normalizedPageRotation,
                        })
                        : constraint;
                    const left = pageRect.left + displayConstraint.left;
                    const top = pageRect.top + displayConstraint.top;
                    const right = left + displayConstraint.width;
                    const bottom = top + displayConstraint.height;
                    const constrainedSpans = allSpans.filter((span) => {
                        const rect = span.getBoundingClientRect();
                        const overlapX = Math.min(rect.right, right) - Math.max(rect.left, left);
                        const overlapY = Math.min(rect.bottom, bottom) - Math.max(rect.top, top);
                        return overlapX > 1 && overlapY > 1;
                    });
                    if (constrainedSpans.length > 0) spans = constrainedSpans;
                }

                const buildText = (items) => items.map((span) => span.textContent || '').join('');
                let fullText = buildText(spans);
                let matchedRange = findCitationTextRange({
                    fullText,
                    text: highlightInfo.text,
                    startPhrase: highlightInfo.startPhrase,
                    endPhrase: highlightInfo.endPhrase,
                });

                // 块内定位失败时才回退到全页，避免双栏和重复短语优先命中错误区域。
                if (!matchedRange && spans.length !== allSpans.length) {
                    spans = allSpans;
                    fullText = buildText(spans);
                    matchedRange = findCitationTextRange({
                        fullText,
                        text: highlightInfo.text,
                        startPhrase: highlightInfo.startPhrase,
                        endPhrase: highlightInfo.endPhrase,
                    });
                }
                if (!matchedRange) return;
                const { startIndex, endIndex } = matchedRange;

                const clientRects = collectTextRangeClientRects(
                    spans,
                    { startIndex, endIndex }
                );
                const resultRects = clientRectsToPageLocal(clientRects, pageElement, 3);
                if (resultRects.length > 0 && isMounted) {
                    setHighlightRect(resultRects[0]);
                    setHighlightRects(resultRects);
                    scrollHighlightIntoView(resultRects);
                }
            } catch (e) {
                console.error('Error calculating highlight:', e);
            }
        };

        // Debounce slightly to allow rendering to settle
        const initialTimer = setTimeout(findHighlight, 300);

        return () => {
            isMounted = false;
            clearTimeout(initialTimer);
            if (retryTimer) clearTimeout(retryTimer);
        };

    }, [blockIndex, clientRectsToPageLocal, currentBlockPage, currentBlocks, debouncedScale, getCurrentPageMetrics, highlightInfo, normalizedPageRotation, numPages, pageNumber, pageRenderEpoch, pageRenderedSize.height, pageRenderedSize.width, renderedPageEpoch, scrollHighlightIntoView]);

    useEffect(() => {
        if (renderedPageEpoch !== pageRenderEpoch) {
            setSavedHighlightRects([]);
            return;
        }

        const pageElement = pageRef.current;
        if (!pageElement) {
            setSavedHighlightRects([]);
            return;
        }

        const pageHighlights = (Array.isArray(savedHighlights) ? savedHighlights : [])
            .filter((item) => Number(item?.page) === Number(pageNumber));
        if (pageHighlights.length === 0) {
            setSavedHighlightRects([]);
            return;
        }

        let allSpans = null;
        let fullText = '';
        const resolveLegacyTextRects = (highlight) => {
            if (!highlight?.text) return [];
            if (!allSpans) {
                const textLayer = pageElement.querySelector('.react-pdf__Page__textContent');
                allSpans = textLayer ? Array.from(textLayer.querySelectorAll('span')) : [];
                fullText = allSpans.map((span) => span.textContent || '').join('');
            }
            if (allSpans.length === 0) return [];
            const matchedRange = findCitationTextRange({ fullText, text: highlight.text });
            if (!matchedRange) return [];
            return clientRectsToPageLocal(
                collectTextRangeClientRects(allSpans, matchedRange),
                pageElement,
                1
            );
        };

        const nextRects = [];
        pageHighlights.forEach((highlight) => {
            const storedRects = (Array.isArray(highlight?.rects) ? highlight.rects : [])
                .map(normalizeDocumentHighlightRect)
                .filter(Boolean);
            const renderedRects = storedRects.length > 0
                ? storedRects.map((rect) => ({
                    left: rect.left * debouncedScale,
                    top: rect.top * debouncedScale,
                    width: rect.width * debouncedScale,
                    height: rect.height * debouncedScale,
                }))
                : resolveLegacyTextRects(highlight);
            renderedRects.forEach((rect, rectIndex) => {
                const color = String(highlight.color || '#FFE066');
                nextRects.push({
                    // Include color in key so color updates remount the overlay.
                    key: `${highlight.id || highlight.text}-${color}-${rectIndex}`,
                    highlightId: String(highlight.id || ''),
                    color,
                    annotationStyle: normalizeDocumentHighlightStyle(highlight.style),
                    annotation: highlight,
                    rect,
                });
            });
        });
        setSavedHighlightRects(nextRects);
    }, [clientRectsToPageLocal, debouncedScale, normalizedPageRotation, pageNumber, pageRenderEpoch, renderedPageEpoch, savedHighlights]);

    const activeOverlayBlockId = hoveredBlockId || activeBlockId;
    const savedMarkerRects = useMemo(
        () => savedHighlightRects.filter(({ annotationStyle }) => annotationStyle === 'highlight'),
        [savedHighlightRects]
    );
    const savedUnderlineRects = useMemo(
        () => savedHighlightRects.filter(({ annotationStyle }) => annotationStyle === 'underline'),
        [savedHighlightRects]
    );
    const focusedBlockSet = useMemo(() => new Set(focusedBlockIds || []), [focusedBlockIds]);
    const visitedBlockSet = useMemo(() => new Set(visitedBlockIds || []), [visitedBlockIds]);
    const inlineTranslationSet = useMemo(() => new Set(inlineTranslationBlockIds || []), [inlineTranslationBlockIds]);
    const translatingSet = useMemo(() => new Set(translatingBlockIds || []), [translatingBlockIds]);
    const inlineTranslationBlock = useMemo(() => {
        if (!hoveredBlockId) return null;
        const block = currentBlocks.find((item) => item.block_id === hoveredBlockId) || null;
        if (!block || !['paragraph', 'caption', 'heading'].includes(block.type || 'paragraph')) return null;
        if (!inlineTranslationSet.has(hoveredBlockId)) return null;
        const translation = blockTranslations?.[hoveredBlockId]?.translation;
        return translation ? block : null;
    }, [blockTranslations, currentBlocks, hoveredBlockId, inlineTranslationSet]);
    const inlineTranslationBBox = normalizeBlockBBox(inlineTranslationBlock?.bbox);
    const displayedTranslationBlockId = hoverTranslationBlockId || (isTranslationDocked ? activeBlockId : null);
    const isTranslationPinned = isTranslationPositionPinned;
    const hoverTranslationBlock = useMemo(() => {
        if (!displayedTranslationBlockId) return null;
        const block = currentBlocks.find((item) => item.block_id === displayedTranslationBlockId) || null;
        if (!block || !['paragraph', 'caption', 'heading', 'figure', 'table'].includes(block.type || 'paragraph')) return null;
        return block;
    }, [currentBlocks, displayedTranslationBlockId]);
    const hoverTranslationBBox = normalizeBlockBBox(hoverTranslationBlock?.bbox);
    const hoverTranslationItem = hoverTranslationBlock?.block_id ? blockTranslations?.[hoverTranslationBlock.block_id] : null;
    const hoverTranslationSummaryContent = useMemo(
        () => normalizeHoverTranslationMath(hoverTranslationItem?.summary || ''),
        [hoverTranslationItem?.summary]
    );
    const hoverTranslationBodyContent = useMemo(
        () => normalizeHoverTranslationMath(hoverTranslationItem?.translation || ''),
        [hoverTranslationItem?.translation]
    );
    const hoverTranslationLoading = hoverTranslationBlock?.block_id ? translatingSet.has(hoverTranslationBlock.block_id) : false;
    const hoverTranslationTitle = hoverTranslationBlock?.type === 'heading'
        ? '标题翻译'
        : hoverTranslationBlock?.type === 'caption'
            ? '图注翻译'
            : hoverTranslationBlock?.type === 'figure' || hoverTranslationBlock?.type === 'table'
                ? '图表解析'
                : '段落翻译';

    const renderTranslationPanelContent = (bodyMaxHeight = 224) => {
        if (!hoverTranslationBlock) return null;
        if (hoverTranslationItem) {
            return (
                <div className="space-y-3">
                    {hoverTranslationItem.summary && (
                        <div className={`rounded-lg border px-3 py-2 ${darkMode ? 'border-white/10 bg-white/[0.04]' : 'border-slate-200 bg-slate-50/80'}`}>
                            <div className={`mb-1 text-[10px] font-semibold tracking-wide ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                                要点 / 讲解
                            </div>
                            <div className={`text-[12px] font-semibold leading-relaxed ${darkMode ? 'text-gray-200' : 'text-slate-700'}`}>
                                <StreamingMarkdown content={hoverTranslationSummaryContent} isStreaming={false} suppressInitialDots />
                            </div>
                        </div>
                    )}
                    <div
                        className={`overflow-y-auto pr-1 text-[14px] leading-relaxed ${darkMode ? 'text-gray-100' : 'text-gray-800'}`}
                        style={{ maxHeight: bodyMaxHeight }}
                    >
                        <StreamingMarkdown content={hoverTranslationBodyContent} isStreaming={false} suppressInitialDots />
                    </div>
                </div>
            );
        }
        if (hoverTranslationLoading) {
            return (
                <div className={`flex items-center gap-2 text-[13px] ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    正在翻译...
                </div>
            );
        }
        return (
            <>
                <div className={`text-[12px] leading-relaxed line-clamp-4 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    {hoverTranslationBlock.text}
                </div>
                <div className={`mt-2 text-[11px] ${darkMode ? 'text-gray-600' : 'text-gray-400'}`}>
                    暂无缓存译文。请先补齐全文缓存或翻译当前页；悬浮不会自动发起模型请求。
                </div>
            </>
        );
    };
    const computedHoverTranslationStyle = useMemo(() => {
        if (!hoverTranslationBBox || !currentBlockPage) return null;
        const pageElement = pageRef.current;
        const scrollElement = pdfScrollRef.current;
        const pageWidth = Number(currentBlockPage.width_pts || 612) * debouncedScale;
        const pageHeight = Number(currentBlockPage.height_pts || 792) * debouncedScale;
        const [x0, y0, x1, y1] = hoverTranslationBBox;
        const gap = 16;
        const pagePadding = 18;
        const minWidth = 260;
        const desiredWidth = Math.min(380, Math.max(300, pageWidth * 0.4));
        const estimatedHeight = 292;
        const x0Scaled = x0 * debouncedScale;
        const x1Scaled = x1 * debouncedScale;
        const y0Scaled = y0 * debouncedScale;
        const y1Scaled = y1 * debouncedScale;
        const pageRect = pageElement?.getBoundingClientRect?.();
        const scrollRect = scrollElement?.getBoundingClientRect?.();
        const leftGutter = pageRect && scrollRect ? Math.max(0, pageRect.left - scrollRect.left - gap) : 0;
        const rightGutter = pageRect && scrollRect ? Math.max(0, scrollRect.right - pageRect.right - gap) : 0;
        const blockCenterX = (x0Scaled + x1Scaled) / 2;
        const preferRight = blockCenterX < pageWidth / 2;
        const visibleTop = pageRect && scrollRect
            ? clampNumber(scrollRect.top - pageRect.top + pagePadding, pagePadding, Math.max(pagePadding, pageHeight - estimatedHeight - pagePadding))
            : pagePadding;
        const visibleBottom = pageRect && scrollRect
            ? clampNumber(scrollRect.bottom - pageRect.top - pagePadding, visibleTop + 160, pageHeight - pagePadding)
            : pageHeight - pagePadding;
        const clampTop = (value, height = estimatedHeight) => clampNumber(
            value,
            visibleTop,
            Math.max(visibleTop, visibleBottom - height)
        );
        const topNearBlock = clampTop(y0Scaled - 8);

        let popupWidth = desiredWidth;
        let left = null;

        const canUseRightGutter = rightGutter >= minWidth;
        const canUseLeftGutter = leftGutter >= minWidth;
        if (preferRight && canUseRightGutter) {
            popupWidth = Math.min(desiredWidth, rightGutter);
            left = pageWidth + gap;
        } else if (!preferRight && canUseLeftGutter) {
            popupWidth = Math.min(desiredWidth, leftGutter);
            left = -popupWidth - gap;
        } else if (canUseRightGutter || canUseLeftGutter) {
            const useRight = canUseRightGutter && (!canUseLeftGutter || rightGutter >= leftGutter);
            popupWidth = Math.min(desiredWidth, useRight ? rightGutter : leftGutter);
            left = useRight ? pageWidth + gap : -popupWidth - gap;
        }

        if (left !== null) {
            const bodyMaxHeight = clampNumber(visibleBottom - topNearBlock - 92, 132, 360);
            return { left, top: topNearBlock, width: popupWidth, bodyMaxHeight };
        }

        popupWidth = Math.min(desiredWidth, Math.max(minWidth, pageWidth - pagePadding * 2));
        const dockRight = blockCenterX < pageWidth / 2;
        left = dockRight ? pageWidth - popupWidth - pagePadding : pagePadding;

        const overlapsHorizontally = left < x1Scaled && left + popupWidth > x0Scaled;
        let fallbackTop = topNearBlock;
        if (overlapsHorizontally) {
            const belowTop = y1Scaled + gap;
            const aboveTop = y0Scaled - estimatedHeight - gap;
            const hasBelowSpace = belowTop + estimatedHeight <= visibleBottom;
            const hasAboveSpace = aboveTop >= visibleTop;
            if (hasBelowSpace) {
                fallbackTop = belowTop;
            } else if (hasAboveSpace) {
                fallbackTop = aboveTop;
            } else {
                const distanceToTop = Math.abs(y0Scaled - visibleTop);
                const distanceToBottom = Math.abs(visibleBottom - y1Scaled);
                fallbackTop = distanceToTop > distanceToBottom ? visibleTop : Math.max(visibleTop, visibleBottom - estimatedHeight);
            }
        }

        const safeTop = clampTop(fallbackTop);
        const bodyMaxHeight = clampNumber(visibleBottom - safeTop - 92, 132, 360);
        return { left, top: safeTop, width: popupWidth, bodyMaxHeight };
    }, [currentBlockPage, debouncedScale, hoverTranslationBBox, hoverTranslationItem?.summary]);
    const hoverTranslationStyle = isTranslationPinned
        ? (floatingTranslationStyle || computedHoverTranslationStyle)
        : computedHoverTranslationStyle;
    const hoverTranslationBodyMaxHeight = hoverTranslationStyle?.bodyMaxHeight || 224;

    const togglePinnedTranslation = useCallback((event) => {
        event.preventDefault();
        event.stopPropagation();
        if (!hoverTranslationBlock?.block_id) return;

        if (isTranslationPinned) {
            setIsTranslationPositionPinned(false);
            setFloatingTranslationStyle(null);
            popupHoveredRef.current = false;
            return;
        }

        setIsTranslationPositionPinned(true);
        setFloatingTranslationStyle(hoverTranslationStyle || computedHoverTranslationStyle);
        popupHoveredRef.current = true;
    }, [computedHoverTranslationStyle, hoverTranslationBlock, hoverTranslationStyle, isTranslationPinned]);

    const toggleDockedTranslation = useCallback((event) => {
        event?.preventDefault?.();
        event?.stopPropagation?.();
        if (!hoverTranslationBlock?.block_id && !activeBlockId) return;
        setIsTranslationPositionPinned(false);
        setFloatingTranslationStyle(null);
        setIsTranslationDocked((prev) => !prev);
        popupHoveredRef.current = true;
    }, [activeBlockId, hoverTranslationBlock]);

    const startTranslationPanelDrag = useCallback((event) => {
        if (!isTranslationPinned || !hoverTranslationStyle) return;
        event.preventDefault();
        event.stopPropagation();
        translationPanelDragRef.current = {
            dragging: true,
            start: { x: event.clientX, y: event.clientY },
            origin: { ...hoverTranslationStyle },
        };

        const handleMove = (moveEvent) => {
            const state = translationPanelDragRef.current;
            if (!state.dragging || !state.origin) return;
            const dx = moveEvent.clientX - state.start.x;
            const dy = moveEvent.clientY - state.start.y;
            setFloatingTranslationStyle({
                ...state.origin,
                left: state.origin.left + dx,
                top: state.origin.top + dy,
            });
        };

        const handleUp = () => {
            translationPanelDragRef.current = { dragging: false, start: { x: 0, y: 0 }, origin: null };
            window.removeEventListener('mousemove', handleMove);
            window.removeEventListener('mouseup', handleUp);
        };

        window.addEventListener('mousemove', handleMove);
        window.addEventListener('mouseup', handleUp);
    }, [hoverTranslationStyle, isTranslationPinned]);

    const startTranslationPanelResize = useCallback((event, corner) => {
        if (!isTranslationPinned || !hoverTranslationStyle) return;
        event.preventDefault();
        event.stopPropagation();
        const isTop = corner.includes('top');
        const isLeft = corner.includes('left');
        translationPanelResizeRef.current = {
            resizing: true,
            start: { x: event.clientX, y: event.clientY },
            origin: { ...hoverTranslationStyle },
        };

        const handleResize = (moveEvent) => {
            const state = translationPanelResizeRef.current;
            if (!state.resizing || !state.origin) return;
            const dx = moveEvent.clientX - state.start.x;
            const dy = moveEvent.clientY - state.start.y;
            const originHeight = state.origin.bodyMaxHeight || 224;
            let nextWidth = isLeft ? state.origin.width - dx : state.origin.width + dx;
            let nextHeight = isTop ? originHeight - dy : originHeight + dy;
            nextWidth = clampNumber(nextWidth, 240, 560);
            nextHeight = clampNumber(nextHeight, 120, 520);

            setFloatingTranslationStyle({
                ...state.origin,
                width: nextWidth,
                bodyMaxHeight: nextHeight,
                left: isLeft ? state.origin.left + (state.origin.width - nextWidth) : state.origin.left,
                top: isTop ? state.origin.top + (originHeight - nextHeight) : state.origin.top,
            });
        };

        const handleResizeEnd = () => {
            translationPanelResizeRef.current = { resizing: false, start: { x: 0, y: 0 }, origin: null };
            window.removeEventListener('mousemove', handleResize);
            window.removeEventListener('mouseup', handleResizeEnd);
        };

        window.addEventListener('mousemove', handleResize);
        window.addEventListener('mouseup', handleResizeEnd);
    }, [hoverTranslationStyle, isTranslationPinned]);

    const startTranslationDockResize = useCallback((event) => {
        if (!isTranslationDocked) return;
        event.preventDefault();
        event.stopPropagation();
        translationDockResizeRef.current = {
            resizing: true,
            startX: event.clientX,
            originWidth: translationDockWidth,
        };
        document.body.style.userSelect = 'none';
        document.body.style.cursor = 'col-resize';

        const handleResize = (moveEvent) => {
            const state = translationDockResizeRef.current;
            if (!state.resizing) return;
            const delta = state.startX - moveEvent.clientX;
            setTranslationDockWidth(clampNumber(
                state.originWidth + delta,
                TRANSLATION_DOCK_MIN_WIDTH,
                TRANSLATION_DOCK_MAX_WIDTH
            ));
        };

        const handleResizeEnd = () => {
            translationDockResizeRef.current = { resizing: false, startX: 0, originWidth: translationDockWidth };
            document.body.style.userSelect = '';
            document.body.style.cursor = '';
            window.removeEventListener('mousemove', handleResize);
            window.removeEventListener('mouseup', handleResizeEnd);
        };

        window.addEventListener('mousemove', handleResize);
        window.addEventListener('mouseup', handleResizeEnd);
    }, [isTranslationDocked, translationDockWidth]);

    const translationDockReservedWidth = isTranslationDocked
        ? translationDockWidth + TRANSLATION_DOCK_GAP
        : 0;
    const activeHighlightSource = String(highlightInfo?.source || 'search');
    const activeHighlightIsCitation = activeHighlightSource === 'citation';
    const activeHighlightIsNote = activeHighlightSource === 'note';

    return (
        <div data-pdf-reader-surface className={`relative h-full flex flex-col overflow-hidden ${darkMode ? 'bg-[#1a1d21]' : 'bg-[#f3f1ee]'}`}>
            {/* Toolbar Area */}
            {/* relative 是必须的：原来只写 z-10 挂在 static 元素上，z-index 直接失效，
                里面的页面转换下拉只能靠 DOM 顺序绘制，被后面的划词工具栏压住。
                这里显式抬到 z-30，高于划词工具栏(z-20)和吸附翻译栏(z-20)。 */}
            <div data-pdf-reader-toolbar className={`relative z-30 flex-shrink-0 border-b px-3 py-2.5 transition-colors duration-200 ${darkMode ? 'border-white/[0.08] bg-[#1a1d21] text-gray-200' : 'border-[#ded8d2]/80 bg-[#f7f5f2] text-gray-600'}`}>
                <div className="flex items-center justify-between px-1 py-1">
                    {/* Left Controls */}
                    <div className="flex items-center gap-1">
                        {onToggleSidebar && (
                            <button onClick={onToggleSidebar} className={`p-1.5 rounded-lg transition-colors ${darkMode ? 'hover:bg-white/10 text-gray-400' : 'hover:bg-gray-100 text-gray-500'}`} title="切换侧边栏">
                                <Sidebar size={18} strokeWidth={2} />
                            </button>
                        )}
                        <div className={`w-[1px] h-4 mx-1 ${darkMode ? 'bg-gray-700' : 'bg-gray-200'}`}></div>
                        <button className={`p-1.5 rounded-lg transition-colors ${darkMode ? 'hover:bg-white/10 text-gray-400' : 'hover:bg-gray-100 text-gray-500'}`} title="文档信息">
                            <FileText size={18} strokeWidth={2} />
                        </button>
                    </div>

                    <div className="flex items-center gap-2">
                        <button onClick={() => changePage(-1)} disabled={previousPageTarget === pageNumber} className={`p-1.5 rounded-lg disabled:opacity-50 transition-colors ${darkMode ? 'hover:bg-white/10 text-gray-400' : 'hover:bg-gray-100 text-gray-400'}`} title="上一页">
                            <ChevronLeft className="w-5 h-5" />
                        </button>
                        <div className={`flex items-center border rounded-md px-2 py-1 text-sm ${darkMode ? 'bg-black/20 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
                            <span className="text-center font-medium min-w-[1.5rem] tabular-nums">{pageIndicator}</span>
                        </div>
                        <span className="text-sm text-gray-400 font-medium">/ {numPages || '--'}</span>
                        <button onClick={() => changePage(1)} disabled={nextPageTarget === pageNumber} className={`p-1.5 rounded-lg disabled:opacity-50 transition-colors ${darkMode ? 'hover:bg-white/10 text-gray-400' : 'hover:bg-gray-100 text-gray-500'}`} title="下一页">
                            <ChevronRight className="w-5 h-5" />
                        </button>
                    </div>
                    <div className="flex items-center gap-1">
                        <div ref={pageLayoutMenuRef} className="relative">
                            <button
                                type="button"
                                onClick={() => setIsPageLayoutMenuOpen((open) => !open)}
                                className={`inline-flex h-8 items-center gap-1.5 rounded-xl px-2 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-[#dc8a69]/35 ${
                                    isPageLayoutMenuOpen
                                        ? (darkMode ? 'bg-white/12 text-white' : 'bg-[#f3ddd5] text-[#a4533d]')
                                        : (darkMode ? 'text-gray-400 hover:bg-white/10 hover:text-gray-100' : 'text-gray-500 hover:bg-[#f0ebe7] hover:text-[#9f5541]')
                                }`}
                                title="页面转换"
                                aria-label="页面转换"
                                aria-expanded={isPageLayoutMenuOpen}
                                aria-controls="pdf-reader-layout-menu"
                            >
                                <BookOpen className="h-4 w-4" />
                                <ChevronDown className={`h-3.5 w-3.5 transition-transform duration-200 ${isPageLayoutMenuOpen ? 'rotate-180' : ''}`} />
                            </button>
                            <AnimatePresence>
                                {isPageLayoutMenuOpen && (
                                    <motion.div
                                        id="pdf-reader-layout-menu"
                                        role="menu"
                                        initial={{ opacity: 0, y: -6, scale: 0.98 }}
                                        animate={{ opacity: 1, y: 0, scale: 1 }}
                                        exit={{ opacity: 0, y: -4, scale: 0.98 }}
                                        transition={{ duration: 0.16, ease: 'easeOut' }}
                                        className={`absolute right-0 top-[calc(100%+10px)] z-50 w-[196px] overflow-hidden rounded-2xl border p-2 shadow-[0_18px_46px_rgba(72,47,35,0.18)] ${
                                            darkMode
                                                ? 'border-white/10 bg-[#25292f] text-gray-100 shadow-black/35'
                                                : 'border-[#ebe4dd] bg-[#fffdfb] text-[#3c342f]'
                                        }`}
                                    >
                                        <div className={`px-2 pb-1 pt-1 text-[11px] font-semibold ${darkMode ? 'text-gray-500' : 'text-[#9b857a]'}`}>页面转换</div>
                                        <button
                                            type="button"
                                            role="menuitemradio"
                                            aria-checked={pageFlowMode === PDF_READER_FLOW_MODES.CONTINUOUS}
                                            onClick={() => selectPageFlowMode(PDF_READER_FLOW_MODES.CONTINUOUS)}
                                            className={`flex w-full items-center gap-2 rounded-xl px-2.5 py-2 text-left text-sm transition-colors ${
                                                pageFlowMode === PDF_READER_FLOW_MODES.CONTINUOUS
                                                    ? (darkMode ? 'bg-white/12 text-white' : 'bg-[#f1dfd8] text-[#9d503c]')
                                                    : (darkMode ? 'text-gray-300 hover:bg-white/8' : 'text-[#51473f] hover:bg-[#f6efeb]')
                                            }`}
                                        >
                                            <ScrollText className="h-4 w-4 shrink-0" />
                                            <span className="flex-1">连续</span>
                                            {pageFlowMode === PDF_READER_FLOW_MODES.CONTINUOUS && <Check className="h-4 w-4" />}
                                        </button>
                                        <button
                                            type="button"
                                            role="menuitemradio"
                                            aria-checked={pageFlowMode === PDF_READER_FLOW_MODES.PAGED}
                                            onClick={() => selectPageFlowMode(PDF_READER_FLOW_MODES.PAGED)}
                                            className={`flex w-full items-center gap-2 rounded-xl px-2.5 py-2 text-left text-sm transition-colors ${
                                                pageFlowMode === PDF_READER_FLOW_MODES.PAGED
                                                    ? (darkMode ? 'bg-white/12 text-white' : 'bg-[#f1dfd8] text-[#9d503c]')
                                                    : (darkMode ? 'text-gray-300 hover:bg-white/8' : 'text-[#51473f] hover:bg-[#f6efeb]')
                                            }`}
                                        >
                                            <FileText className="h-4 w-4 shrink-0" />
                                            <span className="flex-1">逐页</span>
                                            {pageFlowMode === PDF_READER_FLOW_MODES.PAGED && <Check className="h-4 w-4" />}
                                        </button>

                                        <div className={`mx-2 my-1.5 h-px ${darkMode ? 'bg-white/8' : 'bg-[#ebdfd9]'}`} />
                                        <div className={`px-2 pb-1 text-[11px] font-semibold ${darkMode ? 'text-gray-500' : 'text-[#9b857a]'}`}>旋转</div>
                                        <button
                                            type="button"
                                            role="menuitem"
                                            onClick={() => rotateReader(1)}
                                            className={`flex w-full items-center gap-2 rounded-xl px-2.5 py-2 text-left text-sm transition-colors ${darkMode ? 'text-gray-300 hover:bg-white/8' : 'text-[#51473f] hover:bg-[#f6efeb]'}`}
                                        >
                                            <RotateCw className="h-4 w-4 shrink-0" />
                                            <span>顺时针</span>
                                        </button>
                                        <button
                                            type="button"
                                            role="menuitem"
                                            onClick={() => rotateReader(-1)}
                                            className={`flex w-full items-center gap-2 rounded-xl px-2.5 py-2 text-left text-sm transition-colors ${darkMode ? 'text-gray-300 hover:bg-white/8' : 'text-[#51473f] hover:bg-[#f6efeb]'}`}
                                        >
                                            <RotateCcw className="h-4 w-4 shrink-0" />
                                            <span>逆时针</span>
                                        </button>

                                        <div className={`mx-2 my-1.5 h-px ${darkMode ? 'bg-white/8' : 'bg-[#ebdfd9]'}`} />
                                        <div className={`px-2 pb-1 text-[11px] font-semibold ${darkMode ? 'text-gray-500' : 'text-[#9b857a]'}`}>布局</div>
                                        {[
                                            { value: PDF_READER_LAYOUTS.SINGLE, label: '单页', Icon: FileText },
                                            { value: PDF_READER_LAYOUTS.DOUBLE, label: '双页', Icon: Columns2 },
                                            { value: PDF_READER_LAYOUTS.COVER, label: '封面', Icon: BookOpen },
                                        ].map(({ value, label, Icon }) => {
                                            const active = pageLayout === value;
                                            return (
                                                <button
                                                    key={value}
                                                    type="button"
                                                    role="menuitemradio"
                                                    aria-checked={active}
                                                    onClick={() => selectPageLayout(value)}
                                                    className={`flex w-full items-center gap-2 rounded-xl px-2.5 py-2 text-left text-sm transition-colors ${
                                                        active
                                                            ? (darkMode ? 'bg-white/12 text-white' : 'bg-[#f1dfd8] text-[#9d503c]')
                                                            : (darkMode ? 'text-gray-300 hover:bg-white/8' : 'text-[#51473f] hover:bg-[#f6efeb]')
                                                    }`}
                                                >
                                                    <Icon className="h-4 w-4 shrink-0" />
                                                    <span className="flex-1">{label}</span>
                                                    {active && <Check className="h-4 w-4" />}
                                                </button>
                                            );
                                        })}
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </div>
                        <div className={`flex items-center border rounded-lg p-0.5 ${darkMode ? 'bg-black/20 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
                            <button onClick={zoomOut} className={`p-1 rounded-md transition-colors ${darkMode ? 'hover:bg-gray-700 text-gray-400' : 'hover:bg-white text-gray-500'}`}>
                                <ZoomOut className="w-4 h-4" />
                            </button>
                            <span className="text-sm font-medium px-2 w-14 text-center">{Math.round(scale * 100)}%</span>
                            <button onClick={zoomIn} className={`p-1 rounded-md transition-colors ${darkMode ? 'hover:bg-gray-700 text-gray-400' : 'hover:bg-white text-gray-500'}`}>
                                <ZoomIn className="w-4 h-4" />
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* 划词工具栏就长在这里：正常流的下一条，跟着面板一起伸缩，
                既不会盖住别的组件，也不可能溢出到面板外面。 */}
            {selectionToolbar}

            <div className="relative flex-1 min-h-0">
            <div
                ref={pdfScrollRef}
                className={`absolute left-0 top-0 bottom-0 overflow-auto p-4 md:p-8 flex items-start justify-center pdf-scroll transition-[right,padding] duration-300 ${
                    isTranslationDocked ? 'md:pr-6' : ''
                } ${darkMode ? 'bg-[#0f1115]' : 'bg-[#f6f4f1]'}`}
                style={{
                    scrollbarWidth: 'none',
                    right: translationDockReservedWidth,
                }}
                onMouseUp={handleTextSelection}
                onMouseMove={handleBlockMouseMove}
                onMouseLeave={handleBlockMouseLeave}
                onClick={handleBlockClick}
                onScroll={handlePdfScroll}
            >
                {!pdfFile ? (
                    <PDFLoadingState darkMode={darkMode} />
                ) : error ? (
                    <div className="flex flex-col items-center justify-center h-full text-center p-8">
                        <div className="text-red-500 text-6xl mb-4">⚠️</div>
                        <div className="text-lg font-semibold text-gray-700 mb-2">PDF加载失败</div>
                        <div className="text-sm text-gray-500 mb-4">{error}</div>
                        <div className="text-xs text-gray-400 bg-gray-100 p-3 rounded-lg max-w-md">
                            <div className="font-mono break-all">URL: {fullPdfUrl}</div>
                        </div>
                    </div>
                ) : (
                    <Document
                        file={pdfFile}
                        onLoadSuccess={onDocumentLoadSuccess}
                        onLoadError={onDocumentLoadError}
                        loading={<PDFLoadingState darkMode={darkMode} />}
                    >
                        <div className={readerPageStackClassName}>
                        {displayPageNumbers.map((visiblePageNumber) => {
                            const isActivePage = visiblePageNumber === pageNumber;
                            const isCoverPage = isContinuousReading
                                && pageLayout === PDF_READER_LAYOUTS.COVER
                                && visiblePageNumber === 1;
                            // 当前页和其它页必须走同一棵组件树。以前当前页是一棵单独的内联树，
                            // 翻页时两页要在两棵结构不同的树之间互换，React 只能卸载重挂：
                            // canvas 被销毁重建、DeferredPdfPage 的 shouldRender 归位，
                            // 于是先闪一下「加载第 N 页」占位再重绘 —— 连续模式翻页的卡顿就来自这里。
                            return (
                            <div key={visiblePageNumber} className={isCoverPage ? 'col-span-2' : undefined}>
                                <DeferredPdfPage
                                    pageNumber={visiblePageNumber}
                                    scale={debouncedScale}
                                    displayScale={scale}
                                    rotation={normalizedPageRotation}
                                    devicePixelRatio={renderPixelRatio}
                                    darkMode={darkMode}
                                    deferRender={isContinuousReading}
                                    scrollRootRef={pdfScrollRef}
                                    onActivate={isActivePage ? undefined : updateReaderPage}
                                    isActive={isActivePage}
                                    viewerRef={ref}
                                    pageInputRef={pageRef}
                                    cachedImage={isActivePage ? cachedImage : null}
                                    onLoadSuccess={handleFirstPageLoad}
                                    onRenderSuccess={handlePageRenderSuccess}
                                >
                            {isActivePage && (<>
                            {/* 框选遮罩层，覆盖在 PDF 页面上方 */}
                            <SelectionOverlay
                                active={isSelecting}
                                onCapture={onAreaSelected}
                                onCancel={onSelectionCancel}
                                rotation={normalizedPageRotation}
                                pageWidth={pageRenderedSize.width}
                                pageHeight={pageRenderedSize.height}
                            />
                            {currentBlocks.length > 0 && currentBlocks.map((block) => {
                                const bbox = normalizeBlockBBox(block.bbox);
                                if (!bbox) return null;
                                const isFocused = focusedBlockSet.has(block.block_id);
                                const isVisited = !isFocused && visitedBlockSet.has(block.block_id);
                                const isFocusedHover = isFocused && block.block_id === hoveredBlockId;
                                const isActive = !isVisited
                                    && block.block_id === activeOverlayBlockId
                                    && (!isFocused || isFocusedHover);
                                const showFocusPulse = isFocused && focusPulseToken > 0;
                                const tone = isFocused
                                    ? 'border-transparent bg-transparent'
                                    : isVisited
                                        ? 'border-slate-300 bg-slate-200/8 shadow-[0_0_0_1px_rgba(15,23,42,0.08)]'
                                        : isActive
                                            ? 'border-slate-400 bg-slate-200/10 shadow-[0_0_0_1px_rgba(15,23,42,0.10)]'
                                            : 'border-transparent bg-transparent';
                                return (
                                    <div
                                        key={block.block_id}
                                        className={`absolute rounded-md border pointer-events-none z-[8] transition-all duration-150 ${tone}`}
                                        style={{
                                            left: bbox[0] * debouncedScale,
                                            top: bbox[1] * debouncedScale,
                                            width: (bbox[2] - bbox[0]) * debouncedScale,
                                            height: (bbox[3] - bbox[1]) * debouncedScale,
                                            opacity: showFocusPulse || isVisited || isActive ? 1 : 0,
                                        }}
                                    >
                                        {showFocusPulse && (
                                            <span
                                                key={`${block.block_id}-${focusPulseToken}`}
                                                className={`pdf-reading-jump-pulse ${darkMode ? 'pdf-reading-jump-pulse--dark' : ''}`}
                                                data-reading-jump-pulse={focusPulseToken}
                                                aria-hidden="true"
                                            />
                                        )}
                                    </div>
                                );
                            })}
                            {inlineTranslationBlock && inlineTranslationBBox && (
                                <div
                                    className={`absolute z-[12] pointer-events-none overflow-hidden rounded-[2px] ${
                                        darkMode ? 'bg-[#f5f5f5] text-gray-950' : 'bg-white/95 text-gray-950'
                                    }`}
                                    style={{
                                        left: inlineTranslationBBox[0] * debouncedScale,
                                        top: inlineTranslationBBox[1] * debouncedScale,
                                        width: (inlineTranslationBBox[2] - inlineTranslationBBox[0]) * debouncedScale,
                                        height: (inlineTranslationBBox[3] - inlineTranslationBBox[1]) * debouncedScale,
                                    }}
                                >
                                    <div
                                        className="h-full w-full overflow-hidden px-1 py-0.5 font-serif"
                                        style={{
                                            fontSize: Math.max(9, Math.min(14, 10.5 * debouncedScale)),
                                            lineHeight: 1.38,
                                        }}
                                    >
                                        {blockTranslations[inlineTranslationBlock.block_id]?.translation}
                                    </div>
                                </div>
                            )}
                            {hoverTranslationBlock && hoverTranslationStyle && !isTranslationDocked && (
                                <div
                                    data-translation-popup="true"
                                    className={`absolute z-[18] rounded-xl border shadow-xl backdrop-blur-md transition-opacity ${
                                        darkMode
                                            ? 'border-white/10 bg-[#1f2329]/95 text-gray-100 shadow-black/30'
                                            : 'border-gray-200/80 bg-white/95 text-gray-900 shadow-gray-300/40'
                                    }`}
                                    style={{
                                        left: hoverTranslationStyle.left,
                                        top: hoverTranslationStyle.top,
                                        width: hoverTranslationStyle.width,
                                        transform: normalizedPageRotation
                                            ? `rotate(${-normalizedPageRotation}deg)`
                                            : undefined,
                                        transformOrigin: 'top left',
                                    }}
                                    onMouseEnter={() => {
                                        popupHoveredRef.current = true;
                                        if (hoverClearTimerRef.current) {
                                            clearTimeout(hoverClearTimerRef.current);
                                            hoverClearTimerRef.current = null;
                                        }
                                    }}
                                    onMouseLeave={() => {
                                        popupHoveredRef.current = false;
                                        if (isTranslationPinned) return;
                                        updateHoveredBlock(null);
                                    }}
                                    onClick={(event) => event.stopPropagation()}
                                >
                                    <div
                                        className={`px-4 py-3 border-b select-none ${darkMode ? 'border-white/10' : 'border-gray-100'} ${isTranslationPinned ? 'cursor-move' : ''}`}
                                        onMouseDown={startTranslationPanelDrag}
                                    >
                                        <div className="flex items-center gap-2">
                                            <div className="-ml-1 flex shrink-0 items-center gap-0.5">
                                                <button
                                                    type="button"
                                                    onMouseDown={(event) => {
                                                        event.preventDefault();
                                                        event.stopPropagation();
                                                    }}
                                                    onClick={toggleDockedTranslation}
                                                    className={`inline-flex h-7 w-7 items-center justify-center rounded-lg transition-colors ${
                                                        darkMode ? 'text-gray-500 hover:bg-white/10 hover:text-gray-200' : 'text-gray-400 hover:bg-gray-100 hover:text-gray-700'
                                                    }`}
                                                    title="吸附到 PDF 右侧"
                                                >
                                                    <DockIcon className="h-4 w-4" />
                                                </button>
                                                <button
                                                    type="button"
                                                    onMouseDown={(event) => {
                                                        event.preventDefault();
                                                        event.stopPropagation();
                                                    }}
                                                    onClick={togglePinnedTranslation}
                                                    className={`inline-flex h-7 w-7 items-center justify-center rounded-lg transition-colors ${
                                                        isTranslationPinned
                                                            ? (darkMode ? 'bg-white/15 text-gray-100' : 'bg-gray-900 text-white')
                                                            : (darkMode ? 'text-gray-500 hover:bg-white/10 hover:text-gray-200' : 'text-gray-400 hover:bg-gray-100 hover:text-gray-700')
                                                    }`}
                                                    title={isTranslationPinned ? '取消固定位置' : '固定窗口位置'}
                                                >
                                                    <PinIcon className="h-4 w-4" />
                                                </button>
                                            </div>
                                            <div className="min-w-0">
                                                <div className="text-[12px] font-bold truncate">
                                                    {hoverTranslationTitle}
                                                </div>
                                                <div className={`mt-0.5 text-[10px] font-medium ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                                                    {hoverTranslationBlock.block_id}
                                                </div>
                                            </div>
                                            <div className="ml-auto shrink-0">
                                                {hoverTranslationLoading && <Loader2 className="w-4 h-4 animate-spin text-purple-500" />}
                                                {!hoverTranslationLoading && <Languages className={`w-4 h-4 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`} />}
                                            </div>
                                        </div>
                                    </div>
                                    <div className="px-4 py-3">
                                        {renderTranslationPanelContent(hoverTranslationBodyMaxHeight)}
                                    </div>
                                    {isTranslationPinned && ['top-left', 'top-right', 'bottom-left', 'bottom-right'].map((corner) => {
                                        const isTop = corner.includes('top');
                                        const isLeft = corner.includes('left');
                                        const cursor = corner === 'top-left' || corner === 'bottom-right'
                                            ? 'nwse-resize'
                                            : 'nesw-resize';
                                        return (
                                            <div
                                                key={corner}
                                                className="absolute h-7 w-7"
                                                style={{
                                                    top: isTop ? -4 : 'auto',
                                                    bottom: isTop ? 'auto' : -4,
                                                    left: isLeft ? -4 : 'auto',
                                                    right: isLeft ? 'auto' : -4,
                                                    cursor,
                                                }}
                                                onMouseEnter={() => setHoverCorner(corner)}
                                                onMouseLeave={() => setHoverCorner('')}
                                                onMouseDown={(event) => startTranslationPanelResize(event, corner)}
                                                aria-label={`resize-translation-${corner}`}
                                            >
                                                {hoverCorner === corner && (
                                                    <div
                                                        className="absolute pointer-events-none"
                                                        style={{
                                                            width: 9,
                                                            height: 9,
                                                            borderRight: isLeft ? '0' : `2px solid ${darkMode ? '#9ca3af' : '#64748b'}`,
                                                            borderBottom: isTop ? '0' : `2px solid ${darkMode ? '#9ca3af' : '#64748b'}`,
                                                            borderLeft: isLeft ? `2px solid ${darkMode ? '#9ca3af' : '#64748b'}` : '0',
                                                            borderTop: isTop ? `2px solid ${darkMode ? '#9ca3af' : '#64748b'}` : '0',
                                                            right: isLeft ? 'auto' : 4,
                                                            left: isLeft ? 4 : 'auto',
                                                            bottom: isTop ? 'auto' : 4,
                                                            top: isTop ? 4 : 'auto',
                                                        }}
                                                    />
                                                )}
                                            </div>
                                        );
                                    })}
                                </div>
                            )}
                            {/* Marker layer: highlights use translucent ink; underlines stay crisp at the text baseline. */}
                            {savedMarkerRects.length > 0 && (
                                <div
                                    className="absolute inset-0 z-[9] pointer-events-none"
                                    aria-hidden="true"
                                    style={{
                                        mixBlendMode: 'multiply',
                                    }}
                                >
                                    <div
                                        className="absolute inset-0"
                                        style={{
                                            // Flatten children first, then fade the whole stamp once.
                                            opacity: darkMode ? 0.42 : 0.52,
                                        }}
                                    >
                                        {savedMarkerRects.map(({ key, highlightId, rect, color }) => {
                                            const normalized = normalizeDocumentHighlightColor(color);
                                            return (
                                                <div
                                                    key={key}
                                                    data-saved-highlight-id={highlightId}
                                                    data-saved-annotation-style="highlight"
                                                    data-highlight-color={normalized}
                                                    className="absolute rounded-[2px]"
                                                    style={{
                                                        left: rect.left,
                                                        top: rect.top,
                                                        width: rect.width,
                                                        height: rect.height,
                                                        background: normalized,
                                                    }}
                                                />
                                            );
                                        })}
                                    </div>
                                </div>
                            )}
                            {savedUnderlineRects.length > 0 && (
                                <div
                                    className="absolute inset-0 z-[9] pointer-events-none"
                                    aria-hidden="true"
                                    style={{ opacity: darkMode ? 0.88 : 0.9 }}
                                >
                                    {savedUnderlineRects.map(({ key, highlightId, rect, color }) => {
                                        const normalized = normalizeDocumentHighlightColor(color);
                                        const underlineHeight = Math.max(2, Math.min(4, Math.round(rect.height * 0.16)));
                                        return (
                                            <div
                                                key={key}
                                                data-saved-highlight-id={highlightId}
                                                data-saved-annotation-style="underline"
                                                data-highlight-color={normalized}
                                                className="absolute rounded-full"
                                                style={{
                                                    left: rect.left,
                                                    top: rect.top + rect.height - underlineHeight,
                                                    width: rect.width,
                                                    height: underlineHeight,
                                                    background: normalized,
                                                }}
                                            />
                                        );
                                    })}
                                </div>
                            )}
                            {/* 多矩形高亮，避免跨越空白区域的巨大单一框 */}
                            <AnimatePresence>
                                {highlightRects.length > 0 && highlightRects.map((rect, idx) => (
                                    <motion.div
                                        key={`highlight-${idx}`}
                                        initial={activeHighlightIsNote ? false : { opacity: 0, scale: 0.9 }}
                                        animate={activeHighlightIsNote ? {
                                            top: rect.top,
                                            left: rect.left,
                                            width: rect.width,
                                            height: rect.height
                                        } : {
                                            opacity: 1,
                                            scale: 1,
                                            top: rect.top,
                                            left: rect.left,
                                            width: rect.width,
                                            height: rect.height
                                        }}
                                        exit={activeHighlightIsNote ? { opacity: 0 } : { opacity: 0, scale: 0.9 }}
                                        transition={activeHighlightIsNote ? {
                                            duration: 0.14,
                                            ease: 'easeOut'
                                        } : {
                                            type: "spring",
                                            stiffness: 300,
                                            damping: 30,
                                            mass: 1
                                        }}
                                        data-note-jump-highlight={activeHighlightIsNote ? 'true' : undefined}
                                        className={activeHighlightIsNote
                                            ? `pdf-note-jump-highlight absolute pointer-events-none z-10 ${darkMode ? 'pdf-note-jump-highlight--dark' : ''}`
                                            : `absolute border-2 rounded-lg pointer-events-none z-10 ${
                                                activeHighlightIsCitation
                                                    ? 'border-amber-500 bg-amber-500/20'
                                                    : 'border-purple-500 bg-purple-500/20'
                                            }`
                                        }
                                        style={activeHighlightIsNote ? undefined : {
                                            boxShadow: activeHighlightIsCitation
                                                ? '0 0 0 2px rgba(245, 158, 11, 0.15), 0 4px 12px -1px rgba(245, 158, 11, 0.2)'
                                                : '0 0 0 2px rgba(237, 140, 104, 0.1), 0 4px 6px -1px rgba(237, 140, 104, 0.1)'
                                        }}
                                    >
                                        {/* 只在第一个矩形上显示标签 */}
                                        {idx === 0 && !activeHighlightIsNote && (
                                            <div className={`absolute -top-3 -right-3 text-white text-[10px] font-bold px-1.5 py-0.5 rounded-full shadow-sm ${
                                                activeHighlightIsCitation
                                                    ? 'bg-amber-500'
                                                    : 'bg-purple-500'
                                            }`}>
                                                {activeHighlightIsCitation ? '📎 引用' : '匹配'}
                                            </div>
                                        )}
                                    </motion.div>
                                ))}
                            </AnimatePresence>
                            </>)}
                                </DeferredPdfPage>
                            </div>
                            );
                        })}
                        </div>
                    </Document>
                )}
            </div>

            <AnimatePresence>
                {isTranslationDocked && (
                    <motion.aside
                        key="translation-dock"
                        initial={{ opacity: 0, x: 24, scale: 0.98 }}
                        animate={{ opacity: 1, x: 0, scale: 1 }}
                        exit={{ opacity: 0, x: 18, scale: 0.98 }}
                        transition={{ type: 'spring', stiffness: 360, damping: 34, mass: 0.8 }}
                        data-translation-dock="true"
                        className={`absolute bottom-4 right-4 top-4 z-20 flex flex-col overflow-hidden rounded-2xl border shadow-[0_20px_50px_rgba(15,23,42,0.16)] backdrop-blur-xl ${
                            darkMode
                                ? 'border-white/10 bg-[#1f2329]/95 text-gray-100 shadow-black/30'
                                : 'border-white/85 bg-white/95 text-gray-900 shadow-gray-300/40'
                        }`}
                        style={{ width: translationDockWidth }}
                        onMouseMove={(event) => event.stopPropagation()}
                        onMouseUp={(event) => event.stopPropagation()}
                        onMouseLeave={(event) => event.stopPropagation()}
                        onClick={(event) => event.stopPropagation()}
                    >
                        <div
                            className={`absolute -left-2 top-5 bottom-5 z-30 flex w-4 cursor-col-resize items-center justify-center rounded-full transition-colors ${
                                darkMode ? 'hover:bg-white/10' : 'hover:bg-gray-200/70'
                            }`}
                            onMouseDown={startTranslationDockResize}
                            title="拖拽调整吸附栏宽度"
                        >
                            <div className={`h-12 w-1 rounded-full ${darkMode ? 'bg-white/20' : 'bg-gray-300/80'}`} />
                        </div>
                        <div className={`flex items-center justify-between gap-3 border-b px-4 py-3 ${darkMode ? 'border-white/10' : 'border-gray-100'}`}>
                            <div className="flex min-w-0 items-center gap-3">
                                <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-xl ${
                                    darkMode ? 'bg-white/10 text-gray-100' : 'bg-[#fcf2ee] text-[#ce8e76]'
                                }`}>
                                    <DockIcon className="h-4 w-4" />
                                </div>
                                <div className="min-w-0">
                                    <div className="truncate text-[13px] font-bold">{hoverTranslationBlock ? hoverTranslationTitle : '吸附翻译栏'}</div>
                                    <div className={`mt-0.5 truncate text-[10px] font-medium ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                                        {hoverTranslationBlock?.block_id || '悬浮到 PDF 段落后显示译文'}
                                    </div>
                                </div>
                            </div>
                            <button
                                type="button"
                                onClick={(event) => {
                                    event.preventDefault();
                                    event.stopPropagation();
                                    setIsTranslationDocked(false);
                                    popupHoveredRef.current = false;
                                }}
                                className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-xl transition-colors ${
                                    darkMode ? 'text-gray-400 hover:bg-white/10 hover:text-gray-100' : 'text-gray-400 hover:bg-gray-100 hover:text-gray-700'
                                }`}
                                title="关闭吸附栏"
                            >
                                <X className="h-4 w-4" />
                            </button>
                        </div>
                        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
                            {hoverTranslationBlock ? (
                                renderTranslationPanelContent(9999)
                            ) : (
                                <div className={`flex h-full flex-col items-center justify-center rounded-xl border border-dashed px-5 text-center ${
                                    darkMode ? 'border-white/10 text-gray-500' : 'border-gray-200 text-gray-400'
                                }`}>
                                    <Languages className="mb-3 h-6 w-6 opacity-70" />
                                    <div className="text-[13px] font-semibold">悬浮到段落查看翻译</div>
                                    <div className="mt-1 text-[11px] leading-relaxed">吸附栏位置固定，内容会跟随当前段落更新。</div>
                                </div>
                            )}
                        </div>
                    </motion.aside>
                )}
            </AnimatePresence>

            {/* 竖向滚动条 */}
            {vThumb.visible && (
                <div
                    className="absolute top-0 bottom-0 w-1.5 pointer-events-none z-10 transition-[right] duration-300"
                    style={{ right: translationDockReservedWidth + 6 }}
                >
                    <div
                        className={`absolute w-full rounded-full pointer-events-auto cursor-grab active:cursor-grabbing transition-colors duration-200 ${
                            darkMode ? 'bg-white/30 hover:bg-white/55' : 'bg-black/25 hover:bg-black/45'
                        }`}
                        style={{ top: vThumb.top, height: THUMB_SIZE }}
                        onMouseDown={makeDragHandler('v')}
                    />
                </div>
            )}
            {/* 横向滚动条 */}
            {hThumb.visible && (
                <div
                    className="absolute left-0 bottom-1.5 h-1.5 pointer-events-none z-10 transition-[right] duration-300"
                    style={{ right: translationDockReservedWidth }}
                >
                    <div
                        className={`absolute h-full rounded-full pointer-events-auto cursor-grab active:cursor-grabbing transition-colors duration-200 ${
                            darkMode ? 'bg-white/30 hover:bg-white/55' : 'bg-black/25 hover:bg-black/45'
                        }`}
                        style={{ left: hThumb.left, width: THUMB_SIZE }}
                        onMouseDown={makeDragHandler('h')}
                    />
                </div>
            )}
            </div>
        </div>
    );
}));

// 设置 displayName 便于 React DevTools 调试
PDFViewer.displayName = 'PDFViewer';

export default PDFViewer;
