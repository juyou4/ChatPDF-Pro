import { describe, it, expect } from 'vitest';
import { calculateVisibleRange, calculatePadding } from '../VirtualMessageList.jsx';

// 辅助函数：生成指定数量的消息数组
function makeMessages(count) {
  return Array.from({ length: count }, (_, i) => ({ id: `msg-${i}`, content: `消息 ${i}` }));
}

describe('calculateVisibleRange', () => {
  it('空消息列表返回 { start: 0, end: 0 }', () => {
    const result = calculateVisibleRange(0, 500, [], new Map(), 5, 120);
    expect(result).toEqual({ start: 0, end: 0 });
  });

  it('null 消息列表返回 { start: 0, end: 0 }', () => {
    const result = calculateVisibleRange(0, 500, null, new Map(), 5, 120);
    expect(result).toEqual({ start: 0, end: 0 });
  });

  it('消息数量少于可视区域时，全部渲染', () => {
    const messages = makeMessages(3);
    // 容器高度 500px，每条消息估算 120px，3 条共 360px < 500px
    const result = calculateVisibleRange(0, 500, messages, new Map(), 5, 120);
    expect(result.start).toBe(0);
    expect(result.end).toBe(3);
  });

  it('scrollTop=0 时从第一条消息开始渲染', () => {
    const messages = makeMessages(50);
    const result = calculateVisibleRange(0, 500, messages, new Map(), 5, 120);
    expect(result.start).toBe(0);
    // 可视区域约 500/120 ≈ 4.17 条，加上缓冲区 5 条
    expect(result.end).toBeLessThanOrEqual(messages.length);
    expect(result.end).toBeGreaterThan(0);
  });

  it('滚动到中间位置时，start 和 end 正确反映可视范围', () => {
    const messages = makeMessages(100);
    // scrollTop = 6000px，每条 120px，第 50 条开始可见
    const result = calculateVisibleRange(6000, 500, messages, new Map(), 5, 120);
    // 可视起始约 index 50，减去缓冲区 5 = 45
    expect(result.start).toBeGreaterThanOrEqual(40);
    expect(result.start).toBeLessThanOrEqual(50);
    // 可视结束约 index 55，加上缓冲区 5 = 60
    expect(result.end).toBeGreaterThan(result.start);
    expect(result.end).toBeLessThanOrEqual(100);
  });

  it('使用高度缓存时计算更精确', () => {
    const messages = makeMessages(20);
    const heightCache = new Map();
    // 前 10 条高度 200px，后 10 条高度 50px
    messages.forEach((msg, i) => {
      heightCache.set(msg.id, i < 10 ? 200 : 50);
    });
    // scrollTop = 0, containerHeight = 500
    // 前 10 条总高 2000px，500px 可视约 2.5 条
    const result = calculateVisibleRange(0, 500, messages, heightCache, 2, 120);
    expect(result.start).toBe(0);
    // 可视结束约 index 3，加缓冲区 2 = 5
    expect(result.end).toBeLessThanOrEqual(10);
  });

  it('缓冲区大小为 0 时不扩展范围', () => {
    const messages = makeMessages(50);
    const result = calculateVisibleRange(0, 500, messages, new Map(), 0, 120);
    expect(result.start).toBe(0);
    // 无缓冲区，仅渲染可视区域内的消息
    expect(result.end).toBeLessThanOrEqual(10);
  });

  it('滚动到底部时 end 不超过消息总数', () => {
    const messages = makeMessages(20);
    // scrollTop 远超总高度
    const result = calculateVisibleRange(99999, 500, messages, new Map(), 5, 120);
    expect(result.end).toBe(20);
  });

  it('单条消息时正确处理', () => {
    const messages = makeMessages(1);
    const result = calculateVisibleRange(0, 500, messages, new Map(), 5, 120);
    expect(result).toEqual({ start: 0, end: 1 });
  });
});

describe('calculatePadding', () => {
  it('空消息列表返回 { paddingTop: 0, paddingBottom: 0 }', () => {
    const result = calculatePadding([], { start: 0, end: 0 }, new Map(), 120);
    expect(result).toEqual({ paddingTop: 0, paddingBottom: 0 });
  });

  it('null 消息列表返回 { paddingTop: 0, paddingBottom: 0 }', () => {
    const result = calculatePadding(null, { start: 0, end: 0 }, new Map(), 120);
    expect(result).toEqual({ paddingTop: 0, paddingBottom: 0 });
  });

  it('全部消息可见时 padding 均为 0', () => {
    const messages = makeMessages(5);
    const result = calculatePadding(messages, { start: 0, end: 5 }, new Map(), 120);
    expect(result).toEqual({ paddingTop: 0, paddingBottom: 0 });
  });

  it('顶部有不可见消息时 paddingTop 正确', () => {
    const messages = makeMessages(20);
    // 可视范围从 index 5 开始
    const result = calculatePadding(messages, { start: 5, end: 15 }, new Map(), 120);
    // 前 5 条不可见，每条 120px
    expect(result.paddingTop).toBe(5 * 120);
    // 后 5 条不可见
    expect(result.paddingBottom).toBe(5 * 120);
  });

  it('使用高度缓存计算 padding', () => {
    const messages = makeMessages(10);
    const heightCache = new Map();
    // 前 3 条高度分别为 100, 200, 150
    heightCache.set('msg-0', 100);
    heightCache.set('msg-1', 200);
    heightCache.set('msg-2', 150);
    // 后 2 条高度分别为 80, 90
    heightCache.set('msg-8', 80);
    heightCache.set('msg-9', 90);

    const result = calculatePadding(messages, { start: 3, end: 8 }, heightCache, 120);
    expect(result.paddingTop).toBe(100 + 200 + 150);
    expect(result.paddingBottom).toBe(80 + 90);
  });

  it('混合缓存和估算高度计算 padding', () => {
    const messages = makeMessages(10);
    const heightCache = new Map();
    // 仅缓存 msg-0 的高度
    heightCache.set('msg-0', 200);

    const result = calculatePadding(messages, { start: 3, end: 8 }, heightCache, 120);
    // msg-0: 200 (缓存), msg-1: 120 (估算), msg-2: 120 (估算)
    expect(result.paddingTop).toBe(200 + 120 + 120);
    // msg-8: 120 (估算), msg-9: 120 (估算)
    expect(result.paddingBottom).toBe(120 + 120);
  });

  it('start=0 时 paddingTop 为 0', () => {
    const messages = makeMessages(10);
    const result = calculatePadding(messages, { start: 0, end: 5 }, new Map(), 120);
    expect(result.paddingTop).toBe(0);
    expect(result.paddingBottom).toBe(5 * 120);
  });

  it('end=messages.length 时 paddingBottom 为 0', () => {
    const messages = makeMessages(10);
    const result = calculatePadding(messages, { start: 5, end: 10 }, new Map(), 120);
    expect(result.paddingTop).toBe(5 * 120);
    expect(result.paddingBottom).toBe(0);
  });
});
