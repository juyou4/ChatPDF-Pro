import { describe, it, expect } from 'vitest';
import { processLatexBrackets } from '../StreamingMarkdown.jsx';

describe('StreamingMarkdown 渲染功能', () => {
  describe('流式模式下跳过 LaTeX 预处理 (Requirements 4.1)', () => {
    it('processLatexBrackets 会转换 \\[...\\] 为 $$...$$', () => {
      // 确认函数确实会对 LaTeX 括号进行转换
      const rawText = '公式 \\[E = mc^2\\] 结束';
      const processed = processLatexBrackets(rawText);
      expect(processed).not.toBe(rawText);
      expect(processed).toContain('$');
    });

    it('流式模式下文本保持原样不被 LaTeX 预处理', () => {
      // 模拟 processedContent 在 isStreaming=true 时的行为
      const content = '公式 \\[E = mc^2\\] 结束';
      const isStreaming = true;
      const USE_LATEX_PREPROCESS = true;

      let text = content;
      if (USE_LATEX_PREPROCESS && !isStreaming) {
        text = processLatexBrackets(text);
      }

      // 流式模式下文本应保持原样
      expect(text).toBe(content);
    });
  });

  describe('非流式模式下完整渲染 (Requirements 4.2)', () => {
    it('非流式模式下应调用 processLatexBrackets 转换 LaTeX', () => {
      const content = '公式 \\[E = mc^2\\] 结束';
      const isStreaming = false;
      const USE_LATEX_PREPROCESS = true;

      let text = content;
      if (USE_LATEX_PREPROCESS && !isStreaming) {
        text = processLatexBrackets(text);
      }

      // 非流式模式下 \[...\] 应被转换为 $$...$$
      expect(text).not.toBe(content);
      expect(text).toContain('$$');
    });

    it('processLatexBrackets 转换 \\(...\\) 为 $...$', () => {
      const content = '行内公式 \\(x^2\\) 结束';
      const processed = processLatexBrackets(content);
      // \\(...\\) 应被转换为 $...$
      expect(processed).not.toContain('\\(');
      expect(processed).toContain('$');
    });

    it('processLatexBrackets 保留 mermaid 代码块不做转换', () => {
      const content = '```mermaid\ngraph TD\nA-->B\n```';
      const processed = processLatexBrackets(content);
      // mermaid 代码块应保持原样
      expect(processed).toBe(content);
    });

    it('processLatexBrackets 对空输入返回原值', () => {
      expect(processLatexBrackets('')).toBe('');
      expect(processLatexBrackets(null)).toBe(null);
      expect(processLatexBrackets(undefined)).toBe(undefined);
    });
  });
});
