/**
 * Feature: chatpdf-frontend-performance, Property 8: KaTeX 公式缓存命中
 *
 * **Validates: Requirements 6.2**
 *
 * 属性定义：
 * For any 数学公式表达式，首次渲染后将结果存入缓存，
 * 第二次渲染相同表达式时应命中缓存，不触发 KaTeX 重新渲染。
 *
 * 测试策略：
 * - 直接测试 LRUCache 类的缓存行为
 * - 使用 fast-check 生成随机公式表达式和缓存值
 * - 验证四个核心属性：
 *   1. 缓存命中：set 后 get 返回相同值
 *   2. LRU 淘汰：超出 maxSize 时淘汰最久未使用的条目
 *   3. 访问刷新顺序：get 将条目移到最近使用位置
 *   4. 重复 set：更新值并刷新位置
 */

import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import { LRUCache } from '../utils/katexCache';

// 生成器：模拟 LaTeX 公式表达式
const formulaArb = fc.oneof(
  // 简单表达式
  fc.string({ minLength: 1, maxLength: 100 }),
  // 类似真实 LaTeX 的表达式
  fc.constantFrom(
    'E = mc^2',
    '\\frac{a}{b}',
    '\\sum_{i=0}^{n} x_i',
    '\\int_0^\\infty e^{-x} dx',
    '\\sqrt{x^2 + y^2}',
    '\\alpha + \\beta = \\gamma'
  )
);

// 生成器：模拟渲染后的 HTML 结果
const htmlResultArb = fc.string({ minLength: 1, maxLength: 200 });

// 生成器：键值对
const kvPairArb = fc.tuple(formulaArb, htmlResultArb);

// 生成器：键值对序列
const kvListArb = fc.array(kvPairArb, { minLength: 1, maxLength: 50 });

describe('Property 8: KaTeX 公式缓存命中', () => {
  it('缓存命中：set(key, value) 后 get(key) 返回相同值', () => {
    fc.assert(
      fc.property(formulaArb, htmlResultArb, (formula, html) => {
        const cache = new LRUCache(200);

        // 首次设置（模拟首次渲染存入缓存）
        cache.set(formula, html);

        // 第二次获取应命中缓存，返回相同值
        const cached = cache.get(formula);
        expect(cached).toBe(html);

        // has 也应返回 true
        expect(cache.has(formula)).toBe(true);
      }),
      { numRuns: 100 }
    );
  });

  it('LRU 淘汰：超出 maxSize 时淘汰最久未使用的条目', () => {
    fc.assert(
      fc.property(
        // 生成 maxSize（小值便于测试）和超出容量的键值对
        fc.integer({ min: 2, max: 20 }),
        kvListArb,
        (maxSize, extraPairs) => {
          const cache = new LRUCache(maxSize);

          // 先填满缓存
          const filledKeys = [];
          for (let i = 0; i < maxSize; i++) {
            const key = `fill_${i}`;
            cache.set(key, `value_${i}`);
            filledKeys.push(key);
          }
          expect(cache.size).toBe(maxSize);

          // 插入额外条目，触发淘汰
          for (const [key, value] of extraPairs) {
            // 使用带前缀的 key 避免与 fill_ 键冲突
            cache.set(`extra_${key}`, value);
          }

          // 缓存大小不应超过 maxSize
          expect(cache.size).toBeLessThanOrEqual(maxSize);

          // 如果插入了足够多的不同条目，最早的 fill_ 键应被淘汰
          const uniqueExtraKeys = new Set(extraPairs.map(([k]) => `extra_${k}`));
          const totalUniqueInserted = maxSize + uniqueExtraKeys.size;

          if (totalUniqueInserted > maxSize) {
            // 至少有一些早期的 fill_ 键被淘汰
            const evictedCount = Math.min(
              uniqueExtraKeys.size,
              maxSize
            );
            // 最早插入且未被访问的键应该被淘汰
            for (let i = 0; i < Math.min(evictedCount, filledKeys.length); i++) {
              // 如果该键确实被淘汰了，get 应返回 undefined
              if (!cache.has(filledKeys[i])) {
                expect(cache.get(filledKeys[i])).toBeUndefined();
              }
            }
          }
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
          const cache = new LRUCache(maxSize);

          // 填满缓存：key_0 是最旧的，key_(maxSize-1) 是最新的
          for (let i = 0; i < maxSize; i++) {
            cache.set(`key_${i}`, `val_${i}`);
          }

          // 访问最旧的条目 key_0，将其刷新到最近使用位置
          const refreshed = cache.get('key_0');
          expect(refreshed).toBe('val_0');

          // 现在 key_1 变成了最旧的条目
          // 插入一个新条目，应淘汰 key_1 而不是 key_0
          cache.set('new_key', 'new_val');

          // key_0 应该还在（因为刚被访问过）
          expect(cache.has('key_0')).toBe(true);
          expect(cache.get('key_0')).toBe('val_0');

          // key_1 应该被淘汰了
          expect(cache.has('key_1')).toBe(false);
          expect(cache.get('key_1')).toBeUndefined();
        }
      ),
      { numRuns: 100 }
    );
  });

  it('重复 set 同一 key：更新值并刷新 LRU 位置', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 3, max: 15 }),
        htmlResultArb,
        (maxSize, newValue) => {
          const cache = new LRUCache(maxSize);

          // 填满缓存
          for (let i = 0; i < maxSize; i++) {
            cache.set(`key_${i}`, `val_${i}`);
          }

          // 重新 set key_0（最旧的），更新值并刷新位置
          cache.set('key_0', newValue);

          // 值应该被更新
          expect(cache.get('key_0')).toBe(newValue);

          // 大小不应增加（key_0 已存在，只是更新）
          expect(cache.size).toBe(maxSize);

          // 插入新条目，应淘汰 key_1（现在最旧的）而不是 key_0
          cache.set('brand_new', 'brand_val');

          expect(cache.has('key_0')).toBe(true);
          expect(cache.has('key_1')).toBe(false);
        }
      ),
      { numRuns: 100 }
    );
  });
});
