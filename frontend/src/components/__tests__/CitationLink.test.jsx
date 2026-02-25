// @vitest-environment jsdom
/**
 * CitationLink 组件属性测试
 *
 * Feature: chatpdf-citation-fix
 * Property 2: CitationLink 点击传递正确 citation
 *
 * Validates: Requirements 2.1
 */
import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, fireEvent } from '@testing-library/react';
import * as fc from 'fast-check';
import CitationLink from '../CitationLink';

/**
 * 生成随机 citation 对象的 arbitrary
 * 模拟后端返回的引文数据结构
 */
const citationArb = fc.record({
  ref: fc.integer({ min: 1, max: 99 }),
  group_id: fc.string({ minLength: 1, maxLength: 20 }),
  page_range: fc.tuple(
    fc.integer({ min: 1, max: 500 }),
    fc.integer({ min: 1, max: 500 })
  ).map(([a, b]) => [Math.min(a, b), Math.max(a, b)]),
  highlight_text: fc.string({ minLength: 0, maxLength: 200 }),
});

describe('Feature: chatpdf-citation-fix, Property 2: CitationLink 点击传递正确 citation', () => {
  /**
   * 属性测试：对于任意 citation 对象，点击 CitationLink 时
   * onClick 回调应接收到与传入的 citation 完全相同的对象
   *
   * Validates: Requirements 2.1
   */
  it('点击 CitationLink 时 onClick 回调应接收到传入的 citation 对象', () => {
    fc.assert(
      fc.property(citationArb, (citation) => {
        const handleClick = vi.fn();

        const { container } = render(
          <CitationLink
            refNumber={citation.ref}
            citation={citation}
            onClick={handleClick}
          />
        );

        // 找到按钮并模拟点击
        const button = container.querySelector('button');
        expect(button).not.toBeNull();
        fireEvent.click(button);

        // 验证回调被调用且参数为传入的 citation 对象
        expect(handleClick).toHaveBeenCalledTimes(1);
        expect(handleClick).toHaveBeenCalledWith(citation);

        // 验证传递的是同一个引用（完全相同的对象）
        expect(handleClick.mock.calls[0][0]).toBe(citation);
      }),
      { numRuns: 100 }
    );
  });

  /**
   * 属性测试：当 citation 为 null 时，点击不应触发 onClick 回调
   */
  it('当 citation 为 null 时点击不应触发 onClick', () => {
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 99 }), (refNumber) => {
        const handleClick = vi.fn();

        const { container } = render(
          <CitationLink
            refNumber={refNumber}
            citation={null}
            onClick={handleClick}
          />
        );

        // 没有 citation 时渲染为 span 而非 button
        const span = container.querySelector('span');
        expect(span).not.toBeNull();
        fireEvent.click(span);

        // 回调不应被调用
        expect(handleClick).not.toHaveBeenCalled();
      }),
      { numRuns: 100 }
    );
  });
});
