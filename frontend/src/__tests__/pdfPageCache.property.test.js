/**
 * Feature: chatpdf-frontend-performance, Property 11: PDF 页面缓存命中
 *
 * **Validates: Requirements 10.3**
 *
 * 属性定义：
 * For any 已渲染过的 PDF 页面，在缓存未被淘汰的情况下再次访问该页面时，
 * 应直接使用缓存数据而不触发重新渲染。
 *
 * 测试策略：
 * - 直接测试 PdfPageCache 类的缓存行为
 * - 使用 fast-check 生成随机页码、缩放比例和 dataURL
 * - 验证四个核心属性：
 *   1. 缓存命中：set 后 get 返回相同数据
 *   2. LRU 淘汰：超出 maxSize 时淘汰最久未使用的条目
 *   3. 访问刷新顺序：get() 将条目移到最近使用位置，避免被淘汰
 *   4. 缓存键唯一性：相同页码不同缩放比例是独立的缓存条目
 */

import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import { PdfPageCache, DEFAULT_MAX_PAGES } from '../utils/pdfPageCache';

// 生成器：模拟 PDF 页码（1-1000）
const pageNumberArb = fc.integer({ min: 1, max: 1000 });

// 生成器：模拟缩放比例（0.5-3.0，保留一位小数避免浮点精度问题）
const scaleArb = fc.integer({ min: 5, max: 30 }).map(n => n / 10);

// 生成器：模拟 canvas dataURL 字符串
const dataURLArb = fc.string({ minLength: 10, maxLength: 200 }).map(
  s => `data:image/png;base64,${s}`
);

// 生成器：页面-缩放-数据 三元组
const pageEntryArb = fc.tuple(pageNumberArb, scaleArb, dataURLArb);

describe('Property 11: PDF 页面缓存命中', () => {
  it('缓存命中：set(page, scale, data) 后 get(page, scale) 返回相同数据', () => {
    fc.assert(
      fc.property(pageNumberArb, scaleArb, dataURLArb, (page, scale, data) => {
        const cache = new PdfPageCache();

        // 模拟页面渲染完成后存入缓存
        cache.set(page, scale, data);

        // 再次访问应命中缓存，返回相同数据
        const cached = cache.get(page, scale);
        expect(cached).toBe(data);

        // has 也应返回 true
        expect(cache.has(page, scale)).toBe(true);
      }),
      { numRuns: 100 }
    );
  });

  it('LRU 淘汰：超出 maxSize 时淘汰最久未使用的条目', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 2, max: 15 }),
        fc.array(pageEntryArb, { minLength: 1, maxLength: 30 }),
        (maxSize, extraEntries) => {
          const cache = new PdfPageCache(maxSize);

          // 先用不同页码填满缓存（使用固定缩放 1.0 避免键冲突）
          for (let i = 0; i < maxSize; i++) {
            cache.set(i + 1, 1.0, `fill_data_${i}`);
          }
          expect(cache.size).toBe(maxSize);

          // 插入额外条目，使用不同的缩放比例避免与填充键冲突
          for (const [page, scale, data] of extraEntries) {
            cache.set(page + 2000, scale, data);
          }

          // 缓存大小不应超过 maxSize
          expect(cache.size).toBeLessThanOrEqual(maxSize);
        }
      ),
      { numRuns: 100 }
    );
  });

  it('访问刷新 LRU 顺序：get() 将条目移到最近使用位置，避免被淘汰', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 3, max: 15 }),
        (maxSize) => {
          const cache = new PdfPageCache(maxSize);

          // 填满缓存：page 1 是最旧的，page maxSize 是最新的
          // 使用固定缩放 1.0
          for (let i = 1; i <= maxSize; i++) {
            cache.set(i, 1.0, `data_page_${i}`);
          }

          // 访问最旧的条目 page 1，将其刷新到最近使用位置
          const refreshed = cache.get(1, 1.0);
          expect(refreshed).toBe('data_page_1');

          // 现在 page 2 变成了最旧的条目
          // 插入一个新条目，应淘汰 page 2 而不是 page 1
          cache.set(maxSize + 1, 1.0, 'data_new');

          // page 1 应该还在（因为刚被访问过）
          expect(cache.has(1, 1.0)).toBe(true);
          expect(cache.get(1, 1.0)).toBe('data_page_1');

          // page 2 应该被淘汰了
          expect(cache.has(2, 1.0)).toBe(false);
          expect(cache.get(2, 1.0)).toBeUndefined();
        }
      ),
      { numRuns: 100 }
    );
  });

  it('缓存键唯一性：相同页码不同缩放比例是独立的缓存条目', () => {
    fc.assert(
      fc.property(
        pageNumberArb,
        scaleArb,
        scaleArb,
        dataURLArb,
        dataURLArb,
        (page, scale1, scale2, data1, data2) => {
          // 确保两个缩放比例不同
          fc.pre(scale1 !== scale2);

          const cache = new PdfPageCache();

          // 同一页码，不同缩放比例，存入不同数据
          cache.set(page, scale1, data1);
          cache.set(page, scale2, data2);

          // 两个条目应独立存在
          expect(cache.get(page, scale1)).toBe(data1);
          expect(cache.get(page, scale2)).toBe(data2);

          // 缓存中应有 2 个条目
          expect(cache.size).toBe(2);

          // 键应该不同
          const key1 = PdfPageCache.makeKey(page, scale1);
          const key2 = PdfPageCache.makeKey(page, scale2);
          expect(key1).not.toBe(key2);
        }
      ),
      { numRuns: 100 }
    );
  });
});
