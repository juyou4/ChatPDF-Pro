import React, { useState, useEffect, useRef, useCallback, forwardRef, useMemo } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import SelectionOverlay from './SelectionOverlay';
import { useDebouncedValue } from '../hooks/useDebouncedValue';
import pdfPageCache from '../utils/pdfPageCache';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';
import pdfWorkerSrc from 'pdfjs-dist/build/pdf.worker.min.mjs?url';

// Configure worker - 直接指定版本以确保匹配
pdfjs.GlobalWorkerOptions.workerSrc = pdfWorkerSrc;

const PDFViewer = React.memo(forwardRef(({ pdfUrl, onTextSelect, highlightInfo = null, page = 1, onPageChange, isSelecting = false, onAreaSelected, onSelectionCancel, darkMode = false }, ref) => {
    const [numPages, setNumPages] = useState(null);
    const [pageNumber, setPageNumber] = useState(page || 1);
    const [scale, setScale] = useState(1.0);
    // 防抖缩放值：实际 PDF 渲染使用防抖后的值（150ms），避免频繁重渲染
    const debouncedScale = useDebouncedValue(scale, 150);
    const [selectedText, setSelectedText] = useState('');
    const [error, setError] = useState(null);

    useEffect(() => {
        if (typeof page === 'number' && page > 0 && page !== pageNumber) {
            setPageNumber(page);
        }
    }, [page, pageNumber]);

    // 确保 PDF URL 是完整路径，且 pdfUrl 有效时才构建
    const fullPdfUrl = pdfUrl
        ? (pdfUrl.startsWith('http') ? pdfUrl : `${window.location.origin}${pdfUrl}`)
        : null;

    console.log('📄 PDFViewer - Loading PDF:', fullPdfUrl);

    function onDocumentLoadSuccess({ numPages }) {
        console.log('✅ PDF loaded successfully, pages:', numPages);
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
        pdfPageCache.get(pageNumber, scale) || null
    );

    // 页码或缩放变化时，立即尝试从缓存获取占位图
    useEffect(() => {
        const cached = pdfPageCache.get(pageNumber, debouncedScale);
        setCachedImage(cached || null);
    }, [pageNumber, debouncedScale]);

    // 页面渲染成功后，捕获 canvas 数据存入缓存
    const handlePageRenderSuccess = useCallback(() => {
        try {
            const pageEl = pageRef.current;
            if (!pageEl) return;
            const canvas = pageEl.querySelector('canvas');
            if (!canvas) return;
            const dataURL = canvas.toDataURL('image/png');
            pdfPageCache.set(pageNumber, debouncedScale, dataURL);
            // 更新当前缓存图片（下次切换回来时可用）
            setCachedImage(dataURL);
        } catch (e) {
            // canvas 捕获失败时静默忽略，不影响正常渲染
            console.warn('⚠️ PDF 页面缓存捕获失败:', e);
        }
    }, [pageNumber, debouncedScale]);

    // ── 相邻页面预渲染：计算需要预渲染的前后页码 ──
    const pagesToPrerender = useMemo(() => {
        if (!numPages) return [];
        const pages = [];
        if (pageNumber > 1) pages.push(pageNumber - 1);
        if (pageNumber < numPages) pages.push(pageNumber + 1);
        return pages;
    }, [pageNumber, numPages]);

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
                    console.log('⚠️ 高亮匹配：页面文本为空');
                    return;
                }

                // 去除空白后的标准化字符串用于比较
                const searchStr = String(highlightInfo.text).replace(/\s+/g, '').toLowerCase();
                const pageStr = fullText.replace(/\s+/g, '').toLowerCase();

                console.log(`🔍 高亮匹配：搜索文本长度=${searchStr.length}, 页面文本长度=${pageStr.length}`);

                // 策略 1: 完全匹配
                let startIndex = pageStr.indexOf(searchStr);
                let endIndex = -1;

                if (startIndex !== -1) {
                    endIndex = startIndex + searchStr.length;
                    console.log('✅ 高亮匹配：完全匹配成功');
                } else {
                    // 策略 2: 多锚点匹配（灵活大小）
                    const anchorSize = Math.min(12, Math.floor(searchStr.length * 0.15));
                    if (anchorSize < 4) {
                        console.log('⚠️ 高亮匹配：搜索文本太短，无法使用锚点匹配');
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
                            console.log('✅ 高亮匹配：双锚点匹配成功');
                        } else {
                            // 尝试中间锚点作为后备
                            const midPoint = Math.floor(searchStr.length / 2);
                            const midAnchor = searchStr.substring(midPoint, midPoint + anchorSize);
                            const midAnchorIndex = pageStr.indexOf(midAnchor, startAnchorIndex);

                            if (midAnchorIndex !== -1) {
                                startIndex = startAnchorIndex;
                                endIndex = Math.min(startIndex + Math.floor(searchStr.length * 1.3), pageStr.length);
                                console.log('✅ 高亮匹配：中间锚点匹配成功');
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
                                console.log(`✅ 高亮匹配：逐字符匹配 ${matchLen} 个字符`);
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
                                console.log('✅ 高亮匹配：中间子串滑动窗口匹配成功');
                            } else {
                                console.log('⚠️ 高亮匹配：所有策略均未匹配到文本');
                            }
                        } else {
                            console.log('⚠️ 高亮匹配：所有策略均未匹配到文本');
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

    return (
        <div className={`relative h-full flex flex-col rounded-2xl overflow-hidden ${darkMode ? 'bg-[#1a1d21]' : 'bg-[var(--color-bg-base)]'}`}>
            <div className={`flex items-center justify-between p-4 border-b transition-colors duration-200 ${darkMode ? 'bg-[#1a1d21] border-white/10 text-gray-200' : 'bg-white border-gray-200'}`}>
                <div className="flex items-center gap-2">
                    <button onClick={() => changePage(-1)} disabled={pageNumber <= 1} className={`p-2 rounded-lg disabled:opacity-50 transition-colors ${darkMode ? 'hover:bg-white/10' : 'hover:bg-gray-100'}`}>
                        <ChevronLeft className="w-5 h-5" />
                    </button>
                    <span className="text-sm font-medium px-3">{pageNumber} / {numPages || '--'}</span>
                    <button onClick={() => changePage(1)} disabled={pageNumber >= (numPages || 1)} className={`p-2 rounded-lg disabled:opacity-50 transition-colors ${darkMode ? 'hover:bg-white/10' : 'hover:bg-gray-100'}`}>
                        <ChevronRight className="w-5 h-5" />
                    </button>
                </div>
                <div className="flex items-center gap-2">
                    <button onClick={zoomOut} className={`p-2 rounded-lg transition-colors ${darkMode ? 'hover:bg-white/10' : 'hover:bg-gray-100'}`}>
                        <ZoomOut className="w-5 h-5" />
                    </button>
                    <span className="text-sm font-medium px-2">{Math.round(scale * 100)}%</span>
                    <button onClick={zoomIn} className={`p-2 rounded-lg transition-colors ${darkMode ? 'hover:bg-white/10' : 'hover:bg-gray-100'}`}>
                        <ZoomIn className="w-5 h-5" />
                    </button>
                </div>
            </div>
            <div className="relative flex-1 min-h-0">
            <div
                ref={pdfScrollRef}
                className={`absolute inset-0 overflow-auto p-6 flex items-start justify-center pdf-scroll ${darkMode ? 'bg-[#0f1115]' : 'bg-[var(--color-bg-base)]'}`}
                style={{ scrollbarWidth: 'none' }}
                onMouseUp={handleTextSelection}
                onScroll={updateThumbs}
            >
                {!fullPdfUrl ? (
                    <div className="flex items-center justify-center h-full">
                        <div className="text-center">
                            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500 mb-4"></div>
                            <div className="text-gray-500">文档加载中...</div>
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
                        file={fullPdfUrl}
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
                        <div ref={ref} className="relative" style={{ filter: darkMode ? 'grayscale(1) invert(1)' : 'none' }}>
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
                                renderTextLayer={true}
                                renderAnnotationLayer={true}
                                onRenderSuccess={handlePageRenderSuccess}
                            />
                            </div>
                            {/* 框选遮罩层，覆盖在 PDF 页面上方 */}
                            <SelectionOverlay
                                active={isSelecting}
                                onCapture={onAreaSelected}
                                onCancel={onSelectionCancel}
                            />
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
                                                : '0 0 0 2px rgba(136, 113, 228, 0.1), 0 4px 6px -1px rgba(136, 113, 228, 0.1)'
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
                            {/* 相邻页面预渲染：隐藏渲染前后页面，预热 canvas 缓存 */}
                            {pagesToPrerender.map(p => (
                                <div
                                    key={`prerender-${p}`}
                                    style={{
                                        position: 'absolute',
                                        visibility: 'hidden',
                                        pointerEvents: 'none',
                                        top: 0,
                                        left: 0,
                                    }}
                                    aria-hidden="true"
                                >
                                    <Page
                                        pageNumber={p}
                                        scale={debouncedScale}
                                        renderTextLayer={false}
                                        renderAnnotationLayer={false}
                                    />
                                </div>
                            ))}
                        </div>
                    </Document>
                )}
            </div>

            {/* 竖向滚动条 */}
            {vThumb.visible && (
                <div className="absolute right-1.5 top-0 bottom-0 w-1.5 pointer-events-none z-10">
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
                <div className="absolute left-0 right-0 bottom-1.5 h-1.5 pointer-events-none z-10">
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
