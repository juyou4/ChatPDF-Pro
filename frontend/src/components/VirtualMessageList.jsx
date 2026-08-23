import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import { ArrowDown } from 'lucide-react';

// 默认估算高度（未缓存消息的默认高度）
const DEFAULT_ESTIMATED_HEIGHT = 120;
// 默认缓冲区大小（上下各缓冲的消息数）
const DEFAULT_BUFFER_SIZE = 5;
// 只有用户明显离开最新内容时才显示回到底部按钮，避免在底部附近反复闪现。
const AUTO_SCROLL_BOTTOM_THRESHOLD = 50;
const SCROLL_TO_LATEST_VISIBLE_THRESHOLD = 96;

export function getDistanceToBottom(scrollTop, clientHeight, scrollHeight) {
  return Math.max(0, scrollHeight - scrollTop - clientHeight);
}

export function shouldShowScrollToLatest({
  scrollTop,
  clientHeight,
  scrollHeight,
  hasMessages,
}) {
  return Boolean(hasMessages)
    && getDistanceToBottom(scrollTop, clientHeight, scrollHeight) > SCROLL_TO_LATEST_VISIBLE_THRESHOLD;
}

export function scrollMetricsEqual(prev, next) {
  return prev?.scrollTop === next?.scrollTop
    && prev?.containerHeight === next?.containerHeight;
}

// 旧会话中的 system/assistant 消息可能没有 id。虚拟列表不能直接用
// undefined 作为 React key 或高度缓存键，否则多条历史消息会共享同一个
// DOM 身份，加载会话后可能把正在流式的节点复用到错误消息上。
export function getMessageListKey(message, index) {
  const candidates = [
    message?.id,
    message?.messageId,
    message?.message_id,
  ];
  const explicitId = candidates.find((value) => (
    value !== null
    && value !== undefined
    && String(value).trim().length > 0
  ));
  if (explicitId !== undefined) return explicitId;

  const timestamp = message?.createdAt || message?.created_at || message?.timestamp;
  const type = String(message?.type || message?.role || 'message').trim() || 'message';
  return timestamp
    ? `legacy-${type}-${String(timestamp)}-${index}`
    : `legacy-${type}-${index}`;
}

export const STREAMING_LIVE_WINDOW = 12;

/**
 * 流式时钉住最后一条，但不要把中间历史全部挂上。
 * 跟在底部时：窗口上限为 lastIdx-N。
 * 在上面翻历史时：保留视口，中间用占位，另外单独挂上正在生成的那条。
 */
export function pinStreamingRange(
  rawRange,
  messages,
  streamingMessageId,
  liveWindow = STREAMING_LIVE_WINDOW,
) {
  const start = Number(rawRange?.start) || 0;
  const end = Number(rawRange?.end) || 0;
  const fallback = { start, end, pinLast: false };
  if (!streamingMessageId || !Array.isArray(messages) || messages.length === 0) {
    return fallback;
  }
  const lastIdx = messages.length - 1;
  const streamMsg = messages[lastIdx];
  if (!streamMsg || String(streamMsg.id) !== String(streamingMessageId)) {
    return fallback;
  }
  const includesStream = start <= lastIdx && end > lastIdx;
  if (includesStream) {
    return {
      start: Math.max(start, lastIdx - liveWindow),
      end: messages.length,
      pinLast: false,
    };
  }
  return {
    start,
    end,
    pinLast: true,
  };
}

/**
 * 计算当前可视范围内的消息索引
 * @param {number} scrollTop - 滚动容器的 scrollTop
 * @param {number} containerHeight - 滚动容器的可视高度
 * @param {Array} messages - 消息数组
 * @param {Map} heightCache - 消息高度缓存 Map<messageId, height>
 * @param {number} bufferSize - 上下缓冲区大小
 * @param {number} estimatedHeight - 未缓存消息的估算高度
 * @returns {{ start: number, end: number }} 可视范围索引（含缓冲区）
 */
export function calculateVisibleRange(
  scrollTop,
  containerHeight,
  messages,
  heightCache,
  bufferSize = DEFAULT_BUFFER_SIZE,
  estimatedHeight = DEFAULT_ESTIMATED_HEIGHT
) {
  if (!messages || messages.length === 0) {
    return { start: 0, end: 0 };
  }

  let accumulatedHeight = 0;
  let visibleStart = -1;
  let visibleEnd = messages.length;

  // 遍历消息，找到可视区域的起始和结束索引
  for (let i = 0; i < messages.length; i++) {
    const messageKey = getMessageListKey(messages[i], i);
    const msgHeight = heightCache.get(messageKey) ?? estimatedHeight;
    accumulatedHeight += msgHeight;
    // 累积高度超过 scrollTop 时，找到可视区域起始位置
    if (accumulatedHeight > scrollTop && visibleStart === -1) {
      visibleStart = i;
    }
    // 累积高度超过 scrollTop + containerHeight 时，找到可视区域结束位置
    if (accumulatedHeight >= scrollTop + containerHeight) {
      visibleEnd = i + 1;
      break;
    }
  }

  if (visibleStart === -1) {
    visibleStart = 0;
  }

  // 应用缓冲区，扩展渲染范围
  const start = Math.max(0, visibleStart - bufferSize);
  const end = Math.min(messages.length, visibleEnd + bufferSize);

  return { start, end };
}

/**
 * 计算不可见区域的 padding 占位
 * @param {Array} messages - 消息数组
 * @param {{ start: number, end: number }} visibleRange - 可视范围索引
 * @param {Map} heightCache - 消息高度缓存
 * @param {number} estimatedHeight - 未缓存消息的估算高度
 * @returns {{ paddingTop: number, paddingBottom: number }}
 */
export function calculatePadding(
  messages,
  visibleRange,
  heightCache,
  estimatedHeight = DEFAULT_ESTIMATED_HEIGHT
) {
  if (!messages || messages.length === 0) {
    return { paddingTop: 0, paddingBottom: 0 };
  }

  // 计算顶部不可见区域的总高度
  let paddingTop = 0;
  for (let i = 0; i < visibleRange.start; i++) {
    const messageKey = getMessageListKey(messages[i], i);
    paddingTop += heightCache.get(messageKey) ?? estimatedHeight;
  }

  // 计算底部不可见区域的总高度
  let paddingBottom = 0;
  for (let i = visibleRange.end; i < messages.length; i++) {
    const messageKey = getMessageListKey(messages[i], i);
    paddingBottom += heightCache.get(messageKey) ?? estimatedHeight;
  }

  return { paddingTop, paddingBottom };
}

/**
 * 虚拟消息列表组件
 * 仅渲染可视区域及缓冲区内的消息，减少 DOM 节点数量
 *
 * @param {Object} props
 * @param {Array} props.messages - 消息数组，每条消息需有 id 字段
 * @param {Function} props.renderMessage - 渲染单条消息的函数 (message, index) => ReactNode
 * @param {string|null} props.streamingMessageId - 当前正在流式输出的消息 ID
 * @param {number} props.bufferSize - 缓冲区大小，默认 5
 * @param {number} props.estimatedHeight - 未缓存消息的估算高度，默认 120px
 * @param {string} props.className - 外层容器的额外 CSS 类名
 * @param {string} props.itemClassName - 每条消息外层包裹的类名。
 * @param {boolean} props.darkMode - 是否使用深色界面样式
 *   消息之间的间距必须写在这里而且只能用 padding：
 *   1) 在 className 上写 space-y-* 是无效的 —— 那只作用于滚动容器的直接子元素，
 *      而消息全都包在内层的 paddingTop/paddingBottom 占位 div 里；
 *   2) 高度是用 ResizeObserver 的 borderBoxSize 量的，它含 padding 不含 margin，
 *      用 margin 会让每条消息少算一截，虚拟滚动的占位高度会持续偏移。
 */
const VirtualMessageList = React.memo(function VirtualMessageList({
  messages = [],
  renderMessage,
  streamingMessageId = null,
  bufferSize = DEFAULT_BUFFER_SIZE,
  estimatedHeight = DEFAULT_ESTIMATED_HEIGHT,
  className = '',
  itemClassName = '',
  darkMode = false,
}) {
  // 滚动容器 ref
  const scrollContainerRef = useRef(null);
  // 消息高度缓存：Map<messageId, pixelHeight>
  const heightCacheRef = useRef(new Map());
  // 消息 DOM 元素 ref 映射：Map<messageId, HTMLElement>
  const itemRefsMap = useRef(new Map());
  // ResizeObserver 实例
  const resizeObserverRef = useRef(null);
  // 是否应自动滚动到底部
  const shouldAutoScrollRef = useRef(true);
  // 上一次消息数量，用于检测新消息
  const prevMessageCountRef = useRef(messages.length);

  // 滚动状态
  const [scrollState, setScrollState] = useState({
    scrollTop: 0,
    containerHeight: 0,
  });
  const [showScrollToLatest, setShowScrollToLatest] = useState(false);

  const streamingMessageIdRef = useRef(streamingMessageId);
  streamingMessageIdRef.current = streamingMessageId;

  const syncScrollState = useCallback((container = scrollContainerRef.current) => {
    if (!container) return;

    const { scrollTop, clientHeight, scrollHeight } = container;
    const distanceToBottom = getDistanceToBottom(scrollTop, clientHeight, scrollHeight);
    shouldAutoScrollRef.current = distanceToBottom < AUTO_SCROLL_BOTTOM_THRESHOLD;
    const nextShowScrollToLatest = shouldShowScrollToLatest({
      scrollTop,
      clientHeight,
      scrollHeight,
      hasMessages: messages.length > 0,
    });
    const nextScrollState = {
      scrollTop,
      containerHeight: clientHeight,
    };
    setShowScrollToLatest((current) => (
      current === nextShowScrollToLatest ? current : nextShowScrollToLatest
    ));
    setScrollState((current) => (
      scrollMetricsEqual(current, nextScrollState) ? current : nextScrollState
    ));
  }, [messages.length]);

  // 计算可视范围
  const rawVisibleRange = useMemo(
    () => calculateVisibleRange(
      scrollState.scrollTop,
      scrollState.containerHeight,
      messages,
      heightCacheRef.current,
      bufferSize,
      estimatedHeight
    ),
    [scrollState.scrollTop, scrollState.containerHeight, messages, bufferSize, estimatedHeight]
  );

  // 流式输出时钉住正在生成的消息，但截断中间历史，避免整列挂载。
  const streamWindow = useMemo(
    () => pinStreamingRange(
      rawVisibleRange,
      messages,
      streamingMessageId,
      Math.max(STREAMING_LIVE_WINDOW, bufferSize * 2),
    ),
    [rawVisibleRange, streamingMessageId, messages, bufferSize],
  );
  const visibleRange = streamWindow;

  const { paddingTop, paddingBottom } = useMemo(
    () => calculatePadding(messages, visibleRange, heightCacheRef.current, estimatedHeight),
    [messages, visibleRange, estimatedHeight]
  );
  const paddingGap = useMemo(() => {
    if (!streamWindow.pinLast) return 0;
    const lastIdx = Math.max(0, messages.length - 1);
    let gap = 0;
    for (let i = visibleRange.end; i < lastIdx; i++) {
      const messageKey = getMessageListKey(messages[i], i);
      gap += heightCacheRef.current.get(messageKey) ?? estimatedHeight;
    }
    return gap;
  }, [estimatedHeight, messages, streamWindow.pinLast, visibleRange.end]);

  const visibleMessages = useMemo(
    () => messages.slice(visibleRange.start, visibleRange.end),
    [messages, visibleRange.start, visibleRange.end]
  );
  const pinnedStreamMessage = streamWindow.pinLast ? messages[messages.length - 1] : null;
  const pinnedStreamIndex = streamWindow.pinLast ? messages.length - 1 : -1;

  // 初始化 ResizeObserver，监测消息元素高度变化
  useEffect(() => {
    try {
      resizeObserverRef.current = new ResizeObserver((entries) => {
        let hasChanges = false;
        for (const entry of entries) {
          const messageId = entry.target.dataset?.messageId;
          if (messageId == null) continue;

          const newHeight = entry.borderBoxSize?.[0]?.blockSize ?? entry.contentRect.height;
          const cachedHeight = heightCacheRef.current.get(messageId);

          // 高度变化超过 1px 才更新缓存（避免浮点数抖动）
          if (cachedHeight === undefined || Math.abs(cachedHeight - newHeight) > 1) {
            heightCacheRef.current.set(messageId, newHeight);
            hasChanges = true;
          }
        }

        // 跟在底部流式长高时只记账，不要每一行都重算占位，否则整列会跟着抖。
        if (hasChanges && !(streamingMessageIdRef.current && shouldAutoScrollRef.current)) {
          syncScrollState();
        }
      });
    } catch {
      // ResizeObserver 不可用时降级为固定高度估算
      resizeObserverRef.current = null;
    }

    return () => {
      resizeObserverRef.current?.disconnect();
    };
  }, [syncScrollState]);

  // 为可视消息元素注册/注销 ResizeObserver 观察
  const setItemRef = useCallback((messageId, element) => {
    const observer = resizeObserverRef.current;
    const prevElement = itemRefsMap.current.get(messageId);

    // 如果元素没变，不做任何操作
    if (prevElement === element) return;

    // 注销旧元素的观察
    if (prevElement && observer) {
      observer.unobserve(prevElement);
    }

    if (element) {
      // 注册新元素
      itemRefsMap.current.set(messageId, element);
      element.dataset.messageId = String(messageId);
      if (observer) {
        observer.observe(element);
      }
      // 立即测量并缓存高度
      const height = element.getBoundingClientRect().height;
      if (height > 0) {
        heightCacheRef.current.set(messageId, height);
      }
    } else {
      // 元素卸载，移除引用（保留高度缓存用于占位计算）
      itemRefsMap.current.delete(messageId);
    }
  }, []);

  // 滚动事件处理
  const handleScroll = useCallback(() => {
    syncScrollState();
  }, [syncScrollState]);

  // 初始化容器高度测量
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;

    syncScrollState(container);

    // 监听容器自身尺寸变化（如窗口缩放）
    let containerObserver;
    try {
      containerObserver = new ResizeObserver(() => {
        syncScrollState(container);
      });
      containerObserver.observe(container);
    } catch {
      // 降级：不监听容器尺寸变化
    }

    return () => {
      containerObserver?.disconnect();
    };
  }, [syncScrollState]);

  const handleScrollToLatest = useCallback(() => {
    const container = scrollContainerRef.current;
    if (!container) return;

    // 用户主动回到最新内容后，后续流式内容可以继续自然跟随。
    shouldAutoScrollRef.current = true;
    setShowScrollToLatest(false);
    if (typeof container.scrollTo === 'function') {
      container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
    } else {
      container.scrollTop = container.scrollHeight;
    }
  }, []);

  // 新消息到达时自动滚动到底部
  useEffect(() => {
    const currentCount = messages.length;
    const prevCount = prevMessageCountRef.current;
    prevMessageCountRef.current = currentCount;

    // 新消息到达且之前在底部附近，自动滚动
    if (currentCount > prevCount && shouldAutoScrollRef.current) {
      requestAnimationFrame(() => {
        const container = scrollContainerRef.current;
        if (container) {
          container.scrollTop = container.scrollHeight;
        }
      });
    }
  }, [messages.length]);

  // 流式输出期间持续滚动到底部
  useEffect(() => {
    if (!streamingMessageId || !shouldAutoScrollRef.current) return;

    const scrollToBottom = () => {
      const container = scrollContainerRef.current;
      if (container) {
        container.scrollTop = container.scrollHeight;
      }
    };

    // 使用 MutationObserver 监听流式内容变化，自动滚动
    const container = scrollContainerRef.current;
    if (!container) return;

    const mutationObserver = new MutationObserver(() => {
      if (shouldAutoScrollRef.current) {
        scrollToBottom();
      }
    });

    mutationObserver.observe(container, {
      childList: true,
      subtree: true,
      characterData: true,
    });

    return () => {
      mutationObserver.disconnect();
    };
  }, [streamingMessageId]);

  const wasStreamingRef = useRef(Boolean(streamingMessageId));
  useEffect(() => {
    const wasStreaming = wasStreamingRef.current;
    const isStreaming = Boolean(streamingMessageId);
    wasStreamingRef.current = isStreaming;
    if (!wasStreaming || isStreaming) return undefined;
    const frame = window.requestAnimationFrame(() => syncScrollState());
    return () => window.cancelAnimationFrame(frame);
  }, [streamingMessageId, syncScrollState]);

  return (
    <div
      className="relative flex-1 min-h-0"
    >
      <div
        ref={scrollContainerRef}
        data-testid="virtual-message-list"
        onScroll={handleScroll}
        className={`${className} h-full`}
        style={{ overflow: 'auto', scrollbarGutter: 'stable' }}
      >
        <div style={{ paddingTop, paddingBottom: pinnedStreamMessage ? 0 : paddingBottom }}>
          {visibleMessages.map((msg, idx) => {
            const originalIndex = visibleRange.start + idx;
            const messageId = getMessageListKey(msg, originalIndex);
            return (
              <div
                key={messageId}
                ref={(el) => setItemRef(messageId, el)}
                data-message-id={messageId}
                className={itemClassName}
              >
                {renderMessage(msg, originalIndex)}
              </div>
            );
          })}
          {pinnedStreamMessage && (
            <>
              {paddingGap > 0 ? (
                <div aria-hidden="true" style={{ height: paddingGap }} />
              ) : null}
              <div
                key={getMessageListKey(pinnedStreamMessage, pinnedStreamIndex)}
                ref={(el) => setItemRef(getMessageListKey(pinnedStreamMessage, pinnedStreamIndex), el)}
                data-message-id={getMessageListKey(pinnedStreamMessage, pinnedStreamIndex)}
                className={itemClassName}
              >
                {renderMessage(pinnedStreamMessage, pinnedStreamIndex)}
              </div>
            </>
          )}
        </div>
      </div>

      {messages.length > 0 && (
        <button
          type="button"
          data-testid="scroll-to-latest"
          aria-label="回到最新内容"
          aria-hidden={!showScrollToLatest}
          tabIndex={showScrollToLatest ? 0 : -1}
          title="回到最新内容"
          onClick={handleScrollToLatest}
          className={`absolute bottom-28 left-1/2 z-20 flex h-10 w-10 -translate-x-1/2 items-center justify-center rounded-full border backdrop-blur-md transition-[opacity,transform,background-color,box-shadow] duration-200 ease-out motion-reduce:transition-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/45 active:scale-95 ${
            showScrollToLatest
              ? 'translate-y-0 opacity-100 pointer-events-auto'
              : 'translate-y-2 opacity-0 pointer-events-none'
          } ${
            darkMode
              ? 'border-white/[0.11] bg-[#2a2e35]/95 text-gray-200 shadow-[0_10px_24px_-12px_rgba(0,0,0,0.72)] hover:bg-[#353a43] hover:text-white hover:shadow-[0_14px_28px_-12px_rgba(0,0,0,0.82)]'
              : 'border-[#e6e1dc]/90 bg-white/95 text-[#68635e] shadow-[0_10px_24px_-12px_rgba(72,63,54,0.36)] hover:-translate-y-0.5 hover:bg-[#fffefd] hover:text-[#B85F47] hover:shadow-[0_14px_28px_-12px_rgba(72,63,54,0.42)]'
          }`}
        >
          <ArrowDown size={18} strokeWidth={2.25} aria-hidden="true" />
        </button>
      )}
    </div>
  );
});

export default VirtualMessageList;
