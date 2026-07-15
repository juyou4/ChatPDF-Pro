import React, { useState, useEffect, useRef, useCallback, forwardRef, useMemo } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut, Sidebar, FileText, Languages, Loader2, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import SelectionOverlay from './SelectionOverlay';
import StreamingMarkdown from './StreamingMarkdown';
import { useDebouncedValue } from '../hooks/useDebouncedValue';
import pdfPageCache from '../utils/pdfPageCache';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';
import pdfWorkerSrc from 'pdfjs-dist/build/pdf.worker.min.mjs?url';

// Configure worker - 直接指定版本以确保匹配
pdfjs.GlobalWorkerOptions.workerSrc = pdfWorkerSrc;

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

const PDFViewer = React.memo(forwardRef(({ pdfUrl, onTextSelect, highlightInfo = null, page = 1, onPageChange, isSelecting = false, onAreaSelected, onSelectionCancel, darkMode = false, onToggleSidebar, blockIndex = null, activeBlockId = null, focusedBlockIds = [], visitedBlockIds = [], inlineTranslationBlockIds = [], onBlockHover, onBlockClick, blockTranslations = {}, translatingBlockIds = [] }, ref) => {
    const [numPages, setNumPages] = useState(null);
    const [pageNumber, setPageNumber] = useState(page || 1);
    const [scale, setScale] = useState(1.0);
    // 防抖缩放值：实际 PDF 渲染使用防抖后的值（150ms），避免频繁重渲染
    const debouncedScale = useDebouncedValue(scale, 150);
    const [selectedText, setSelectedText] = useState('');
    const [error, setError] = useState(null);
    const [hoveredBlockId, setHoveredBlockId] = useState(null);
    const [hoverTranslationBlockId, setHoverTranslationBlockId] = useState(null);
    const [isTranslationPositionPinned, setIsTranslationPositionPinned] = useState(false);
    const [isTranslationDocked, setIsTranslationDocked] = useState(false);
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
    const hasAutoFitRef = useRef(false);
    const backgroundDelayRef = useRef(null);
    const backgroundIdleTaskRef = useRef(null);
    const backgroundGenerationRef = useRef(0);
    const isDesktop = typeof window !== 'undefined' && window.chatpdfDesktop?.isDesktop === true;
    const [desktopApiBaseUrl, setDesktopApiBaseUrl] = useState('');
    const [desktopBackendToken, setDesktopBackendToken] = useState('');

    useEffect(() => {
        if (typeof page === 'number' && page > 0 && page !== pageNumber) {
            setPageNumber(page);
        }
    }, [page, pageNumber]);

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

    useEffect(() => {
        setIsTranslationDocked(false);
        pdfDocumentRef.current = null;
        hasAutoFitRef.current = false;
        setNumPages(null);
        setError(null);
    }, [pdfUrl]);

    // 桌面模式下通过 preload IPC 获取后端地址与鉴权 token
    useEffect(() => {
        let cancelled = false;

        if (!isDesktop) return () => {};

        (async () => {
            try {
                const [apiBaseUrl, backendToken] = await Promise.all([
                    window.chatpdfDesktop.getApiBaseUrl(),
                    window.chatpdfDesktop.getBackendToken(),
                ]);
                if (cancelled) return;
                setDesktopApiBaseUrl((apiBaseUrl || '').replace(/\/$/, ''));
                setDesktopBackendToken(backendToken || '');
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
        return Math.min(Math.max(window.devicePixelRatio || 1, 1), 1.5);
    }, []);

    // react-pdf 支持通过 file 对象传递 httpHeaders，桌面端必须携带 token 访问 /uploads
    const pdfFile = useMemo(() => {
        if (!fullPdfUrl) return null;

        if (isDesktop) {
            if (!desktopBackendToken) return null;
            return {
                url: fullPdfUrl,
                httpHeaders: {
                    'X-ChatPDF-Token': desktopBackendToken,
                },
            };
        }

        return fullPdfUrl;
    }, [fullPdfUrl, isDesktop, desktopBackendToken]);

    function onDocumentLoadSuccess(pdfDocument) {
        const { numPages } = pdfDocument;
        pdfDocumentRef.current = pdfDocument;
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

    // 首次加载按容器宽度自适应缩放（fit-width）；只做一次，不覆盖用户手动缩放
    const handleFirstPageLoad = useCallback((page) => {
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
    }, []);

    function onDocumentLoadError(error) {
        console.error('❌ PDF load error:', error);
        setError(error.message || 'Failed to load PDF');
    }

    const handleTextSelection = () => {
        const selection = window.getSelection();
        const text = selection.toString().trim();
        if (text) {
            setSelectedText(text);
            if (onTextSelect) {
                onTextSelect(text);
            }
        }
    };

    const changePage = (offset) => {
        setPageNumber(prevPageNumber => {
            const nextPage = Math.max(1, Math.min(prevPageNumber + offset, numPages || prevPageNumber || 1));
            if (onPageChange) {
                onPageChange(nextPage);
            }
            return nextPage;
        });
    };

    const zoomIn = () => setScale(prev => Math.min(prev + 0.2, 3.0));
    const zoomOut = () => setScale(prev => Math.max(prev - 0.2, 0.5));

    const [highlightRect, setHighlightRect] = useState(null);
    const [highlightRects, setHighlightRects] = useState([]);
    const pageRef = useRef(null);

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
    const handlePageRenderSuccess = useCallback(() => {
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
    }, [cancelBackgroundWork, debouncedScale, numPages, pageNumber, pdfCacheKey]);

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
        const pageElement = pageRef.current;
        if (!pageElement) return null;

        const pageRect = pageElement.getBoundingClientRect();
        if (
            clientX < pageRect.left ||
            clientX > pageRect.right ||
            clientY < pageRect.top ||
            clientY > pageRect.bottom
        ) {
            return null;
        }

        const x = (clientX - pageRect.left) / debouncedScale;
        const y = (clientY - pageRect.top) / debouncedScale;
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
    }, [currentBlocks, debouncedScale, isSelecting]);

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

    const handleBlockClick = useCallback((event) => {
        const selection = window.getSelection?.();
        const selectionText = selection?.toString().trim() || '';
        if (selectionText) return;
        const block = findBlockAtPoint(event.clientX, event.clientY);
        if (block) {
            onBlockClick?.(block);
        }
    }, [findBlockAtPoint, onBlockClick]);

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
    const pdfScrollRef = useRef(null);
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
    }, [scale, debouncedScale, pageNumber, numPages, updateThumbs]);

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

    useEffect(() => {
        let isMounted = true;
        let retryTimer = null;
        let retryCount = 0;
        const MAX_RETRIES = 15; // 最多重试 15 次（约 1.5 秒）

        if (!highlightInfo || !highlightInfo.text) {
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
                const spans = Array.from(textLayer.querySelectorAll('span'));
                let fullText = '';

                // 构建完整文本
                spans.forEach(span => {
                    fullText += span.textContent;
                });

                if (!fullText) {
                    return;
                }

                // 去除空白后的标准化字符串用于比较
                const normalize = (value) => String(value || '').replace(/\s+/g, '').toLowerCase();
                const searchStr = normalize(highlightInfo.text);
                const startPhrase = normalize(highlightInfo.startPhrase);
                const endPhrase = normalize(highlightInfo.endPhrase);
                const pageStr = normalize(fullText);

                const collectMatches = (needle, limit = 8) => {
                    if (!needle) return [];
                    const result = [];
                    let from = 0;
                    while (from < pageStr.length && result.length < limit) {
                        const idx = pageStr.indexOf(needle, from);
                        if (idx === -1) break;
                        result.push(idx);
                        from = idx + Math.max(1, Math.floor(needle.length / 2));
                    }
                    return result;
                };

                const resolveAnchorSpan = () => {
                    const startMatches = collectMatches(startPhrase);
                    const endMatches = collectMatches(endPhrase);
                    const targetLen = Math.min(Math.max(searchStr.length || startPhrase.length || endPhrase.length, 40), 140);

                    let best = null;
                    for (const s of startMatches.length ? startMatches : [-1]) {
                        for (const e of endMatches.length ? endMatches : [-1]) {
                            let start = s;
                            let end = e;
                            if (start === -1 && end === -1) continue;
                            if (start === -1) {
                                end = end + endPhrase.length;
                                start = Math.max(0, end - targetLen);
                            } else if (end === -1) {
                                end = Math.min(pageStr.length, start + targetLen);
                            } else {
                                end = end + endPhrase.length;
                                if (end <= start) continue;
                                if (end - start > 220) continue;
                            }
                            const spanLen = end - start;
                            const score = (startPhrase ? 20 : 0) + (endPhrase ? 20 : 0) - spanLen * 0.08;
                            if (!best || score > best.score || (score === best.score && spanLen < best.spanLen)) {
                                best = { start, end, score, spanLen };
                            }
                        }
                    }
                    return best ? { startIndex: best.start, endIndex: best.end } : null;
                };

                // 策略 0: 优先使用 start/end phrase 锚点，避免大范围误框
                const anchored = (startPhrase || endPhrase) ? resolveAnchorSpan() : null;
                let startIndex = anchored ? anchored.startIndex : pageStr.indexOf(searchStr);
                let endIndex = -1;

                if (anchored) {
                    endIndex = anchored.endIndex;
                }

                if (!anchored && startIndex !== -1) {
                    endIndex = startIndex + searchStr.length;
                } else if (!anchored) {
                    const candidateMaxLen = Math.min(120, searchStr.length);
                    const candidateTexts = [];
                    if (candidateMaxLen >= 24) {
                        const midStart = Math.max(0, Math.floor((searchStr.length - candidateMaxLen) / 2));
                        candidateTexts.push(searchStr.substring(midStart, midStart + candidateMaxLen));
                        candidateTexts.push(searchStr.substring(0, candidateMaxLen));
                        candidateTexts.push(searchStr.substring(searchStr.length - candidateMaxLen));
                    }
                    for (const candidate of candidateTexts) {
                        const idx = pageStr.indexOf(candidate);
                        if (idx !== -1) {
                            startIndex = idx;
                            endIndex = idx + candidate.length;
                            break;
                        }
                    }
                }

                if (!anchored && endIndex === -1) {
                    // 策略 2: 多锚点匹配（灵活大小）
                    const anchorSize = Math.min(12, Math.floor(searchStr.length * 0.15));
                    if (anchorSize < 4) {
                        return;
                    }
                    const startAnchor = searchStr.substring(0, anchorSize);
                    const endAnchor = searchStr.substring(searchStr.length - anchorSize);

                    const startAnchorIndex = pageStr.indexOf(startAnchor);

                    if (startAnchorIndex !== -1) {
                        // 尝试找到结尾锚点
                        const endAnchorIndex = pageStr.indexOf(endAnchor, startAnchorIndex + anchorSize);

                        if (endAnchorIndex !== -1 && endAnchorIndex > startAnchorIndex) {
                            // 两个锚点都找到了
                            startIndex = startAnchorIndex;
                            endIndex = endAnchorIndex + endAnchor.length;
                        } else {
                            // 尝试中间锚点作为后备
                            const midPoint = Math.floor(searchStr.length / 2);
                            const midAnchor = searchStr.substring(midPoint, midPoint + anchorSize);
                            const midAnchorIndex = pageStr.indexOf(midAnchor, startAnchorIndex);

                            if (midAnchorIndex !== -1) {
                                startIndex = startAnchorIndex;
                                endIndex = Math.min(startIndex + Math.floor(searchStr.length * 1.3), pageStr.length);
                            } else {
                                // 最后手段：从起始锚点逐字符匹配
                                startIndex = startAnchorIndex;
                                let matchLen = anchorSize;
                                while (matchLen < searchStr.length && startIndex + matchLen < pageStr.length) {
                                    if (pageStr[startIndex + matchLen] === searchStr[matchLen]) {
                                        matchLen++;
                                    } else {
                                        break;
                                    }
                                }
                                endIndex = startIndex + matchLen;
                            }
                        }
                    } else {
                        // 策略 3: 滑动窗口子串匹配 — 取搜索文本中间一段尝试匹配
                        const windowSize = Math.min(20, Math.floor(searchStr.length * 0.3));
                        if (windowSize >= 6) {
                            const midStart = Math.floor((searchStr.length - windowSize) / 2);
                            const midSlice = searchStr.substring(midStart, midStart + windowSize);
                            const midSliceIndex = pageStr.indexOf(midSlice);
                            if (midSliceIndex !== -1) {
                                // 从中间片段向两侧扩展
                                startIndex = Math.max(0, midSliceIndex - midStart);
                                endIndex = Math.min(startIndex + searchStr.length, pageStr.length);
                            }
                        }
                    }
                }

                if (startIndex === -1 || endIndex === -1) return;

                // 将字符串索引映射到 DOM 节点
                let startNode = null;
                let startOffset = 0;
                let endNode = null;
                let endOffset = 0;

                let currentCharCount = 0;
                let foundStart = false;
                let foundEnd = false;

                for (const span of spans) {
                    const text = span.textContent;
                    const cleanText = text.replace(/\s+/g, '');
                    const spanLength = cleanText.length;

                    if (!foundStart) {
                        if (currentCharCount + spanLength > startIndex) {
                            foundStart = true;
                            // Find exact offset in this span
                            let localCount = 0;
                            for (let i = 0; i < text.length; i++) {
                                if (!/\s/.test(text[i])) {
                                    if (currentCharCount + localCount === startIndex) {
                                        startNode = span.firstChild;
                                        startOffset = i;
                                        break;
                                    }
                                    localCount++;
                                }
                            }
                        }
                    }

                    if (foundStart && !foundEnd) {
                        if (currentCharCount + spanLength >= endIndex) {
                            foundEnd = true;
                            // Find exact end offset
                            let localCount = 0;
                            for (let i = 0; i < text.length; i++) {
                                if (!/\s/.test(text[i])) {
                                    localCount++;
                                    if (currentCharCount + localCount === endIndex) {
                                        endNode = span.firstChild;
                                        endOffset = i + 1;
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    currentCharCount += spanLength;
                    if (foundEnd) break;
                }

                if (startNode && endNode) {
                    const range = document.createRange();
                    range.setStart(startNode, startOffset);
                    range.setEnd(endNode, endOffset);
                    const rects = Array.from(range.getClientRects());

                    if (rects.length > 0) {
                        const pageRect = pageElement.getBoundingClientRect();
                        const padding = 4;

                        // 过滤掉零尺寸的矩形
                        const validRects = rects.filter(r => r.width > 1 && r.height > 1);
                        if (validRects.length === 0) return;

                        // 按行分组：将垂直位置接近的矩形归为同一行
                        const lineGroups = [];
                        for (const rect of validRects) {
                            let added = false;
                            for (const group of lineGroups) {
                                // 如果矩形的垂直中心与组内矩形接近（差距小于行高的一半），归为同一行
                                const groupMidY = (group[0].top + group[0].bottom) / 2;
                                const rectMidY = (rect.top + rect.bottom) / 2;
                                const lineHeight = group[0].bottom - group[0].top;
                                if (Math.abs(rectMidY - groupMidY) < lineHeight * 0.6) {
                                    group.push(rect);
                                    added = true;
                                    break;
                                }
                            }
                            if (!added) {
                                lineGroups.push([rect]);
                            }
                        }

                        // 按垂直位置排序行组
                        lineGroups.sort((a, b) => a[0].top - b[0].top);

                        // 将连续的行组合并为紧凑的高亮块（行间距超过 1.5 倍行高则分割）
                        const highlightBlocks = [];
                        let currentBlock = [lineGroups[0]];

                        for (let i = 1; i < lineGroups.length; i++) {
                            const prevGroup = currentBlock[currentBlock.length - 1];
                            const currGroup = lineGroups[i];
                            const prevBottom = Math.max(...prevGroup.map(r => r.bottom));
                            const currTop = Math.min(...currGroup.map(r => r.top));
                            const avgLineHeight = prevGroup[0].bottom - prevGroup[0].top;
                            const gap = currTop - prevBottom;

                            if (gap > avgLineHeight * 1.5) {
                                // 间距过大，开始新的高亮块
                                highlightBlocks.push(currentBlock);
                                currentBlock = [currGroup];
                            } else {
                                currentBlock.push(currGroup);
                            }
                        }
                        highlightBlocks.push(currentBlock);

                        // 为每个高亮块计算边界矩形
                        const resultRects = highlightBlocks.map(block => {
                            const allRects = block.flat();
                            return {
                                top: Math.min(...allRects.map(r => r.top)) - pageRect.top - padding,
                                left: Math.min(...allRects.map(r => r.left)) - pageRect.left - padding,
                                width: (Math.max(...allRects.map(r => r.right)) - Math.min(...allRects.map(r => r.left))) + padding * 2,
                                height: (Math.max(...allRects.map(r => r.bottom)) - Math.min(...allRects.map(r => r.top))) + padding * 2
                            };
                        });

                        if (isMounted) {
                            // 兼容旧的单矩形模式（取第一个块）
                            setHighlightRect(resultRects[0] || null);
                            setHighlightRects(resultRects);
                        }
                    }
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

    }, [highlightInfo, pageNumber, scale, numPages]);

    const activeOverlayBlockId = hoveredBlockId || activeBlockId;
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

    return (
        <div className={`relative h-full flex flex-col overflow-hidden ${darkMode ? 'bg-[#1a1d21]' : 'bg-[#f3f1ee]'}`}>
            {/* Toolbar Area */}
            <div className={`z-10 flex-shrink-0 border-b px-3 py-2.5 transition-colors duration-200 ${darkMode ? 'border-white/[0.08] bg-[#1a1d21] text-gray-200' : 'border-[#ded8d2]/80 bg-[#f7f5f2] text-gray-600'}`}>
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
                        <button onClick={() => changePage(-1)} disabled={pageNumber <= 1} className={`p-1.5 rounded-lg disabled:opacity-50 transition-colors ${darkMode ? 'hover:bg-white/10 text-gray-400' : 'hover:bg-gray-100 text-gray-400'}`}>
                            <ChevronLeft className="w-5 h-5" />
                        </button>
                        <div className={`flex items-center border rounded-md px-2 py-1 text-sm ${darkMode ? 'bg-black/20 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
                            <span className="text-center font-medium min-w-[1.5rem]">{pageNumber}</span>
                        </div>
                        <span className="text-sm text-gray-400 font-medium">/ {numPages || '--'}</span>
                        <button onClick={() => changePage(1)} disabled={pageNumber >= (numPages || 1)} className={`p-1.5 rounded-lg disabled:opacity-50 transition-colors ${darkMode ? 'hover:bg-white/10 text-gray-400' : 'hover:bg-gray-100 text-gray-500'}`}>
                            <ChevronRight className="w-5 h-5" />
                        </button>
                    </div>
                    <div className="flex items-center gap-1">
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
            <div className="relative flex-1 min-h-0">
            <div
                ref={pdfScrollRef}
                className={`absolute left-0 top-0 bottom-0 overflow-auto p-4 md:p-8 flex items-start justify-center pdf-scroll transition-[right] duration-300 ${
                    isTranslationDocked ? 'md:pr-6' : ''
                } ${darkMode ? 'bg-[#0f1115]' : 'bg-[#f6f4f1]'}`}
                style={{ scrollbarWidth: 'none', right: translationDockReservedWidth }}
                onMouseUp={handleTextSelection}
                onMouseMove={handleBlockMouseMove}
                onMouseLeave={handleBlockMouseLeave}
                onClick={handleBlockClick}
                onScroll={updateThumbs}
            >
                {!pdfFile ? (
                    <div className="flex items-center justify-center h-full">
                        <div className="text-center">
                            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500 mb-4"></div>
                            <div className="text-gray-500">加载PDF中...</div>
                        </div>
                    </div>
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
                        loading={
                            <div className="flex items-center justify-center h-full">
                                <div className="text-center">
                                    <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500 mb-4"></div>
                                    <div className="text-gray-500">加载PDF中...</div>
                                </div>
                            </div>
                        }
                    >
                        <div ref={ref} className={`relative shadow-[0_2px_15px_rgba(0,0,0,0.06)] bg-white rounded-sm ${darkMode ? 'shadow-none !bg-transparent' : ''}`} style={{ filter: darkMode ? 'grayscale(1) invert(1)' : 'none' }}>
                            {/* 缩放过渡期间使用 CSS transform 即时缩放缓存画面，避免白屏 */}
                            <div style={scale !== debouncedScale ? {
                                transform: `scale(${scale / debouncedScale})`,
                                transformOrigin: 'top left',
                            } : undefined}>
                            {/* 缓存占位图：在页面加载/重渲染期间显示已缓存的 canvas 快照 */}
                            {cachedImage && (
                                <img
                                    src={cachedImage}
                                    alt=""
                                    style={{
                                        position: 'absolute',
                                        top: 0,
                                        left: 0,
                                        zIndex: 0,
                                        pointerEvents: 'none',
                                    }}
                                />
                            )}
                            <Page
                                inputRef={pageRef}
                                pageNumber={pageNumber}
                                scale={debouncedScale}
                                devicePixelRatio={renderPixelRatio}
                                renderTextLayer={true}
                                renderAnnotationLayer={true}
                                onLoadSuccess={handleFirstPageLoad}
                                onRenderSuccess={handlePageRenderSuccess}
                            />
                            </div>
                            {/* 框选遮罩层，覆盖在 PDF 页面上方 */}
                            <SelectionOverlay
                                active={isSelecting}
                                onCapture={onAreaSelected}
                                onCancel={onSelectionCancel}
                            />
                            {currentBlocks.length > 0 && currentBlocks.map((block) => {
                                const bbox = normalizeBlockBBox(block.bbox);
                                if (!bbox) return null;
                                const isFocused = focusedBlockSet.has(block.block_id);
                                const isVisited = !isFocused && visitedBlockSet.has(block.block_id);
                                const isActive = !isFocused && !isVisited && block.block_id === activeOverlayBlockId;
                                const tone = isFocused
                                    ? 'border-slate-500 bg-slate-300/10 shadow-[0_0_0_2px_rgba(15,23,42,0.14)]'
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
                                            opacity: isFocused || isVisited || isActive ? 1 : 0,
                                        }}
                                    >
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
                            {/* 多矩形高亮，避免跨越空白区域的巨大单一框 */}
                            <AnimatePresence>
                                {highlightRects.length > 0 && highlightRects.map((rect, idx) => (
                                    <motion.div
                                        key={`highlight-${idx}`}
                                        initial={{ opacity: 0, scale: 0.9 }}
                                        animate={{
                                            opacity: 1,
                                            scale: 1,
                                            top: rect.top,
                                            left: rect.left,
                                            width: rect.width,
                                            height: rect.height
                                        }}
                                        exit={{ opacity: 0, scale: 0.9 }}
                                        transition={{
                                            type: "spring",
                                            stiffness: 300,
                                            damping: 30,
                                            mass: 1
                                        }}
                                        className={`absolute border-2 rounded-lg pointer-events-none z-10 ${
                                            highlightInfo?.source === 'citation'
                                                ? 'border-amber-500 bg-amber-500/20'
                                                : 'border-purple-500 bg-purple-500/20'
                                        }`}
                                        style={{
                                            boxShadow: highlightInfo?.source === 'citation'
                                                ? '0 0 0 2px rgba(245, 158, 11, 0.15), 0 4px 12px -1px rgba(245, 158, 11, 0.2)'
                                                : '0 0 0 2px rgba(237, 140, 104, 0.1), 0 4px 6px -1px rgba(237, 140, 104, 0.1)'
                                        }}
                                    >
                                        {/* 只在第一个矩形上显示标签 */}
                                        {idx === 0 && (
                                            <div className={`absolute -top-3 -right-3 text-white text-[10px] font-bold px-1.5 py-0.5 rounded-full shadow-sm ${
                                                highlightInfo?.source === 'citation' ? 'bg-amber-500' : 'bg-purple-500'
                                            }`}>
                                                {highlightInfo?.source === 'citation' ? '📎 引用' : '匹配'}
                                            </div>
                                        )}
                                    </motion.div>
                                ))}
                            </AnimatePresence>
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
