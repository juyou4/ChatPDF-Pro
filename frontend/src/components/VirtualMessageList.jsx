import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react';

// 默认估算高度（未缓存消息的默认高度）
const DEFAULT_ESTIMATED_HEIGHT = 120;
// 默认缓冲区大小（上下各缓冲的消息数）
const DEFAULT_BUFFER_SIZE = 5;

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
    const msgHeight = heightCache.get(messages[i].id) ?? estimatedHeight;
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
    paddingTop += heightCache.get(messages[i].id) ?? estimatedHeight;
  }

  // 计算底部不可见区域的总高度
  let paddingBottom = 0;
  for (let i = visibleRange.end; i < messages.length; i++) {
    paddingBottom += heightCache.get(messages[i].id) ?? estimatedHeight;
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
 */
const VirtualMessageList = React.memo(function VirtualMessageList({
  messages = [],
  renderMessage,
  streamingMessageId = null,
  bufferSize = DEFAULT_BUFFER_SIZE,
  estimatedHeight = DEFAULT_ESTIMATED_HEIGHT,
  className = '',
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

  // 计算可视范围
  const visibleRange = useMemo(
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

  // 计算 padding 占位
  const { paddingTop, paddingBottom } = useMemo(
    () => calculatePadding(messages, visibleRange, heightCacheRef.current, estimatedHeight),
    [messages, visibleRange, estimatedHeight]
  );

  // 提取可视消息切片
  const visibleMessages = useMemo(
    () => messages.slice(visibleRange.start, visibleRange.end),
    [messages, visibleRange.start, visibleRange.end]
  );

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

        // 高度变化时触发重新计算可视范围
        if (hasChanges) {
          const container = scrollContainerRef.current;
          if (container) {
            setScrollState({
              scrollTop: container.scrollTop,
              containerHeight: container.clientHeight,
            });
          }
        }
      });
    } catch {
      // ResizeObserver 不可用时降级为固定高度估算
      resizeObserverRef.current = null;
    }

    return () => {
      resizeObserverRef.current?.disconnect();
    };
  }, []);

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
    const container = scrollContainerRef.current;
    if (!container) return;

    const { scrollTop, clientHeight, scrollHeight } = container;

    // 判断是否在底部附近（距底部 50px 以内视为在底部）
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 50;
    shouldAutoScrollRef.current = isNearBottom;

    setScrollState({
      scrollTop,
      containerHeight: clientHeight,
    });
  }, []);

  // 初始化容器高度测量
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;

    setScrollState({
      scrollTop: container.scrollTop,
      containerHeight: container.clientHeight,
    });

    // 监听容器自身尺寸变化（如窗口缩放）
    let containerObserver;
    try {
      containerObserver = new ResizeObserver(() => {
        setScrollState({
          scrollTop: container.scrollTop,
          containerHeight: container.clientHeight,
        });
      });
      containerObserver.observe(container);
    } catch {
      // 降级：不监听容器尺寸变化
    }

    return () => {
      containerObserver?.disconnect();
    };
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

  return (
    <div
      ref={scrollContainerRef}
      onScroll={handleScroll}
      className={className}
      style={{ overflow: 'auto' }}
    >
      <div style={{ paddingTop, paddingBottom }}>
        {visibleMessages.map((msg, idx) => {
          const originalIndex = visibleRange.start + idx;
          const messageId = msg.id;
          return (
            <div
              key={messageId}
              ref={(el) => setItemRef(messageId, el)}
              data-message-id={messageId}
            >
              {renderMessage(msg, originalIndex)}
            </div>
          );
        })}
      </div>
    </div>
  );
});

export default VirtualMessageList;
