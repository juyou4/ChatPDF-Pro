// @vitest-environment jsdom
import { describe, it, expect, vi } from 'vitest';
import * as fc from 'fast-check';
import { streamingMarkdownAreEqual } from '../StreamingMarkdown.jsx';

/**
 * 生成随机 citation 对象的 arbitrary
 */
const citationArb = fc.record({
  ref: fc.integer({ min: 1, max: 99 }),
  group_id: fc.string({ minLength: 1, maxLength: 20 }),
  page_range: fc.tuple(
    fc.integer({ min: 1, max: 500 }),
    fc.integer({ min: 1, max: 500 })
  ),
  highlight_text: fc.string({ minLength: 0, maxLength: 100 }),
});

/**
 * 生成随机 citations 数组的 arbitrary
 */
const citationsArb = fc.array(citationArb, { minLength: 1, maxLength: 10 });

/**
 * 生成基础 props 对象的 arbitrary（不含 citations）
 */
const basePropsArb = fc.record({
  content: fc.string({ minLength: 0, maxLength: 200 }),
  isStreaming: fc.boolean(),
  streamingRef: fc.oneof(fc.constant(null), fc.constant({ current: null })),
});

describe('Feature: chatpdf-citation-fix, Property 1: React.memo 比较函数检测 citations 变化', () => {
  /**
   * 属性测试：当 citations 不同而其他 props 相同时，比较函数应返回 false
   * **Validates: Requirements 1.1, 1.2**
   */
  it('当 citations 从 null 变为有效数组时，比较函数应返回 false（需要重渲染）', () => {
    fc.assert(
      fc.property(basePropsArb, citationsArb, (baseProps, citations) => {
        const prevProps = { ...baseProps, citations: null };
        const nextProps = { ...baseProps, citations };

        // citations 不同，应返回 false
        const result = streamingMarkdownAreEqual(prevProps, nextProps);
        expect(result).toBe(false);
      }),
      { numRuns: 100 }
    );
  });

  it('当 citations 从有效数组变为 null 时，比较函数应返回 false（需要重渲染）', () => {
    fc.assert(
      fc.property(basePropsArb, citationsArb, (baseProps, citations) => {
        const prevProps = { ...baseProps, citations };
        const nextProps = { ...baseProps, citations: null };

        const result = streamingMarkdownAreEqual(prevProps, nextProps);
        expect(result).toBe(false);
      }),
      { numRuns: 100 }
    );
  });

  it('当 citations 引用相同（===）且其他 props 相同时，比较函数应返回 true（无需重渲染）', () => {
    fc.assert(
      fc.property(basePropsArb, fc.oneof(fc.constant(null), citationsArb), (baseProps, citations) => {
        const prevProps = { ...baseProps, citations };
        // 使用同一引用
        const nextProps = { ...baseProps, citations };

        const result = streamingMarkdownAreEqual(prevProps, nextProps);
        expect(result).toBe(true);
      }),
      { numRuns: 100 }
    );
  });

  it('当 citations 内容相同但引用不同时，比较函数应返回 false（浅比较）', () => {
    fc.assert(
      fc.property(basePropsArb, citationsArb, (baseProps, citations) => {
        const prevProps = { ...baseProps, citations };
        // 创建新数组引用（内容相同但 !== 原数组）
        const nextProps = { ...baseProps, citations: [...citations] };

        const result = streamingMarkdownAreEqual(prevProps, nextProps);
        // 浅比较下不同引用应返回 false
        expect(result).toBe(false);
      }),
      { numRuns: 100 }
    );
  });

  it('当任意其他 prop 变化时，无论 citations 是否相同，比较函数应返回 false', () => {
    fc.assert(
      fc.property(
        basePropsArb,
        basePropsArb,
        fc.oneof(fc.constant(null), citationsArb),
        (baseProps1, baseProps2, citations) => {
          // 确保至少有一个基础 prop 不同
          const hasDiff =
            baseProps1.content !== baseProps2.content ||
            baseProps1.isStreaming !== baseProps2.isStreaming ||
            (baseProps1.streamingRef != null) !== (baseProps2.streamingRef != null);

          // 仅在确实有差异时测试
          fc.pre(hasDiff);

          const prevProps = { ...baseProps1, citations };
          const nextProps = { ...baseProps2, citations };

          const result = streamingMarkdownAreEqual(prevProps, nextProps);
          expect(result).toBe(false);
        }
      ),
      { numRuns: 100 }
    );
  });
});

import { processCitationRefs } from '../StreamingMarkdown.jsx';

/**
 * 生成有效引用编号的 arbitrary（1-99）
 */
const refNumArb = fc.integer({ min: 1, max: 99 });

/**
 * 生成 citation 对象的 arbitrary（用于 processCitationRefs 测试）
 */
const citationRefArb = (ref) =>
  fc.record({
    ref: fc.constant(ref),
    group_id: fc.string({ minLength: 1, maxLength: 10 }),
    page_range: fc.tuple(fc.integer({ min: 1, max: 500 }), fc.integer({ min: 1, max: 500 })),
    highlight_text: fc.string({ minLength: 0, maxLength: 50 }),
  });

/**
 * 生成不含方括号和感叹号的安全文本片段，避免干扰引文正则匹配
 */
const safeTextArb = fc
  .string({ minLength: 0, maxLength: 30 })
  .map((s) => s.replace(/[\[\]!\0()]/g, ''));

describe('Feature: chatpdf-citation-fix, Property 4: processCitationRefs 正确替换有效引用并保留无效引用', () => {
  /**
   * 属性测试：有效引用编号应被替换为 <cite> 标签
   * **Validates: Requirements 3.1, 3.3, 3.4, 3.5**
   */
  it('有效引用编号 [N] 应被替换为 <cite data-ref="N">[N]</cite>', () => {
    fc.assert(
      fc.property(
        refNumArb,
        safeTextArb,
        safeTextArb,
        (ref, prefix, suffix) => {
          const citations = [{ ref, group_id: 'g1', page_range: [1, 2], highlight_text: '' }];
          const input = `${prefix}[${ref}]${suffix}`;
          const result = processCitationRefs(input, citations);

          // 有效引用应被替换为 cite 标签
          expect(result).toContain(`<cite data-ref="${ref}">[${ref}]</cite>`);
        }
      ),
      { numRuns: 100 }
    );
  });

  it('不在有效集合中的引用编号应保留原始文本', () => {
    fc.assert(
      fc.property(
        refNumArb,
        refNumArb,
        safeTextArb,
        (validRef, invalidRef, prefix) => {
          // 确保 invalidRef 不在有效集合中
          fc.pre(validRef !== invalidRef);

          const citations = [{ ref: validRef, group_id: 'g1', page_range: [1, 2], highlight_text: '' }];
          const input = `${prefix}[${invalidRef}]`;
          const result = processCitationRefs(input, citations);

          // 无效引用应保持原样
          expect(result).toContain(`[${invalidRef}]`);
          // 不应被包裹在 cite 标签中
          expect(result).not.toContain(`<cite data-ref="${invalidRef}">`);
        }
      ),
      { numRuns: 100 }
    );
  });

  it('连续引文 [A][B] 中每个标记应被独立处理', () => {
    fc.assert(
      fc.property(
        refNumArb,
        refNumArb,
        (refA, refB) => {
          // 确保两个引用编号不同
          fc.pre(refA !== refB);

          const citations = [
            { ref: refA, group_id: 'g1', page_range: [1, 2], highlight_text: '' },
            { ref: refB, group_id: 'g2', page_range: [3, 4], highlight_text: '' },
          ];
          const input = `文本[${refA}][${refB}]结尾`;
          const result = processCitationRefs(input, citations);

          // 两个引用都应被独立替换
          expect(result).toContain(`<cite data-ref="${refA}">[${refA}]</cite>`);
          expect(result).toContain(`<cite data-ref="${refB}">[${refB}]</cite>`);
          // 两个 cite 标签应紧邻
          expect(result).toContain(
            `<cite data-ref="${refA}">[${refA}]</cite><cite data-ref="${refB}">[${refB}]</cite>`
          );
        }
      ),
      { numRuns: 100 }
    );
  });

  it('Markdown 链接 [N](url) 不应被替换', () => {
    fc.assert(
      fc.property(refNumArb, (ref) => {
        const citations = [{ ref, group_id: 'g1', page_range: [1, 2], highlight_text: '' }];
        const input = `查看[${ref}](https://example.com)链接`;
        const result = processCitationRefs(input, citations);

        // Markdown 链接不应被替换为 cite 标签
        expect(result).not.toContain(`<cite data-ref="${ref}">`);
        // 原始链接语法应保持完整
        expect(result).toContain(`[${ref}](https://example.com)`);
      }),
      { numRuns: 100 }
    );
  });

  it('Markdown 图片 ![N](url) 不应被替换', () => {
    fc.assert(
      fc.property(refNumArb, (ref) => {
        const citations = [{ ref, group_id: 'g1', page_range: [1, 2], highlight_text: '' }];
        const input = `图片![${ref}](https://example.com/img.png)描述`;
        const result = processCitationRefs(input, citations);

        // 图片语法不应被替换为 cite 标签
        expect(result).not.toContain(`<cite data-ref="${ref}">`);
        // 原始图片语法应保持完整
        expect(result).toContain(`![${ref}](https://example.com/img.png)`);
      }),
      { numRuns: 100 }
    );
  });

  it('混合场景：有效引用、无效引用、链接和图片共存时各自行为正确', () => {
    fc.assert(
      fc.property(
        refNumArb,
        refNumArb,
        refNumArb,
        (validRef, invalidRef, linkRef) => {
          // 确保三个编号互不相同
          fc.pre(validRef !== invalidRef && validRef !== linkRef && invalidRef !== linkRef);

          const citations = [{ ref: validRef, group_id: 'g1', page_range: [1, 2], highlight_text: '' }];
          const input = `正文[${validRef}]参考[${invalidRef}]链接[${linkRef}](url)图片![${validRef}](img)`;
          const result = processCitationRefs(input, citations);

          // 有效引用被替换
          expect(result).toContain(`<cite data-ref="${validRef}">[${validRef}]</cite>`);
          // 无效引用保持原样
          expect(result).toContain(`[${invalidRef}]`);
          expect(result).not.toContain(`<cite data-ref="${invalidRef}">`);
          // Markdown 链接保持原样
          expect(result).toContain(`[${linkRef}](url)`);
          expect(result).not.toContain(`<cite data-ref="${linkRef}">`);
          // 图片语法保持原样
          expect(result).toContain(`![${validRef}](img)`);
        }
      ),
      { numRuns: 100 }
    );
  });

  it('citations 为 null 或空数组时返回原始文本', () => {
    fc.assert(
      fc.property(
        fc.string({ minLength: 1, maxLength: 100 }),
        (text) => {
          // null citations
          expect(processCitationRefs(text, null)).toBe(text);
          // 空数组
          expect(processCitationRefs(text, [])).toBe(text);
        }
      ),
      { numRuns: 100 }
    );
  });
});


// ============================================================
// 集成测试：验证完整引文渲染链路
// ============================================================
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import StreamingMarkdown from '../StreamingMarkdown.jsx';

// mock katex 模块，避免 jsdom 环境下 KaTeX 渲染报错
vi.mock('katex', () => ({
  default: {
    renderToString: (expr) => `<span class="katex">${expr}</span>`,
  },
  renderToString: (expr) => `<span class="katex">${expr}</span>`,
}));

// mock CSS 导入
vi.mock('katex/dist/katex.min.css', () => ({}));
vi.mock('highlight.js/styles/github.css', () => ({}));

describe('集成测试：完整引文渲染链路', () => {
  /**
   * 验证 StreamingMarkdown 接收 citations 后渲染出 CitationLink 组件
   * _Requirements: 4.1, 4.2, 4.3_
   */

  const mockCitations = [
    { ref: 1, group_id: 'g1', page_range: [3, 5], highlight_text: '测试文本' },
    { ref: 2, group_id: 'g2', page_range: [10, 12], highlight_text: '另一段文本' },
  ];

  it('内容包含 [1] 标记且 citations 包含 ref=1 时，应渲染出 CitationLink 按钮', async () => {
    const content = '这是一段包含引用的文本[1]。';

    render(
      <StreamingMarkdown
        content={content}
        isStreaming={false}
        citations={mockCitations}
      />
    );

    // CitationLink 渲染为 button，文本内容为引用编号
    await waitFor(() => {
      const button = screen.getByRole('button', { name: '1' });
      expect(button).toBeInTheDocument();
    });
  });

  it('多个引文标记 [1] 和 [2] 应各自渲染为独立的 CitationLink 按钮', async () => {
    const content = '第一个引用[1]，第二个引用[2]。';

    render(
      <StreamingMarkdown
        content={content}
        isStreaming={false}
        citations={mockCitations}
      />
    );

    await waitFor(() => {
      const btn1 = screen.getByRole('button', { name: '1' });
      const btn2 = screen.getByRole('button', { name: '2' });
      expect(btn1).toBeInTheDocument();
      expect(btn2).toBeInTheDocument();
    });
  });

  it('连续引文 [1][2] 应渲染为两个独立的 CitationLink 按钮', async () => {
    const content = '连续引用[1][2]结尾。';

    render(
      <StreamingMarkdown
        content={content}
        isStreaming={false}
        citations={mockCitations}
      />
    );

    await waitFor(() => {
      const btn1 = screen.getByRole('button', { name: '1' });
      const btn2 = screen.getByRole('button', { name: '2' });
      expect(btn1).toBeInTheDocument();
      expect(btn2).toBeInTheDocument();
    });
  });

  it('点击 CitationLink 按钮应触发 onCitationClick 回调并传递正确的 citation 对象', async () => {
    const content = '点击测试[1]。';
    const handleClick = vi.fn();

    render(
      <StreamingMarkdown
        content={content}
        isStreaming={false}
        citations={mockCitations}
        onCitationClick={handleClick}
      />
    );

    await waitFor(() => {
      expect(screen.getByRole('button', { name: '1' })).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole('button', { name: '1' }));

    // 验证回调被调用，且传递了正确的 citation 对象（包含 page_range）
    expect(handleClick).toHaveBeenCalledTimes(1);
    expect(handleClick).toHaveBeenCalledWith(
      expect.objectContaining({
        ref: 1,
        page_range: [3, 5],
      })
    );
  });

  it('citations 为 null 时，[N] 标记应保持为纯文本，不渲染 CitationLink', async () => {
    const content = '无引文数据[1]。';

    render(
      <StreamingMarkdown
        content={content}
        isStreaming={false}
        citations={null}
      />
    );

    // 等待渲染完成
    await waitFor(() => {
      expect(screen.getByText(/无引文数据/)).toBeInTheDocument();
    });

    // 不应有 CitationLink 按钮
    const buttons = screen.queryAllByRole('button');
    expect(buttons.length).toBe(0);
  });

  it('<cite> 标签在 ReactMarkdown 管线（rehypeRaw + rehypeKatex + rehypeHighlight）中保持完整', async () => {
    // 内容同时包含引文和普通 Markdown 语法
    const content = '**加粗文本**和引用[1]以及`代码片段`。';

    const { container } = render(
      <StreamingMarkdown
        content={content}
        isStreaming={false}
        citations={mockCitations}
      />
    );

    await waitFor(() => {
      expect(screen.getByRole('button', { name: '1' })).toBeInTheDocument();
    });

    // 验证加粗文本也正常渲染（管线未被破坏）
    const strong = container.querySelector('strong');
    expect(strong).toBeInTheDocument();
    expect(strong.textContent).toBe('加粗文本');

    // 验证代码片段也正常渲染
    const code = container.querySelector('code');
    expect(code).toBeInTheDocument();
    expect(code.textContent).toBe('代码片段');
  });

  it('不在 citations 有效集合中的引用编号不应渲染为 CitationLink', async () => {
    const content = '有效引用[1]和无效引用[99]。';
    const citations = [
      { ref: 1, group_id: 'g1', page_range: [3, 5], highlight_text: '测试' },
    ];

    render(
      <StreamingMarkdown
        content={content}
        isStreaming={false}
        citations={citations}
      />
    );

    await waitFor(() => {
      expect(screen.getByRole('button', { name: '1' })).toBeInTheDocument();
    });

    // [99] 不在有效集合中，不应渲染为按钮
    const buttons = screen.queryAllByRole('button');
    expect(buttons.length).toBe(1);
    // [99] 应保持为纯文本
    expect(screen.getByText(/\[99\]/)).toBeInTheDocument();
  });
});
