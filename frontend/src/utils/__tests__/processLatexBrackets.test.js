import { describe, it, expect } from 'vitest';
import { processLatexBrackets } from '../processLatexBrackets.js';

describe('processLatexBrackets', () => {
  it('应将 \\( ... \\) 转换为 $...$，并规范双反斜杠命令', () => {
    const input = '尺寸为\\(N \\\\times H \\\\times W\\)';
    const output = processLatexBrackets(input);
    expect(output).toBe('尺寸为$N \\times H \\times W$');
  });

  it('应规范已有 $...$ 中过度转义的命令', () => {
    const input = '维度 $N \\\\times H \\\\times W$';
    const output = processLatexBrackets(input);
    expect(output).toBe('维度 $N \\times H \\times W$');
  });

  it('应自动包裹无分隔符的命令型片段', () => {
    const input = '符号 \\boldsymbol{\\otimes} 用于融合';
    const output = processLatexBrackets(input);
    expect(output).toContain('$\\boldsymbol{\\otimes}$');
  });

  it('应自动包裹无分隔符的运算命令片段', () => {
    const input = '尺寸 N \\times H \\times W \\times 3';
    const output = processLatexBrackets(input);
    expect(output).toContain('$\\times$');
    expect(output).toBe('尺寸 N $\\times$ H $\\times$ W $\\times$ 3');
  });

  it('不应把后续中文说明吞进同一个数学片段', () => {
    const input = '输入是多相机原始图像 N \\times H \\times W \\times 3 (N是相机数量)';
    const output = processLatexBrackets(input);
    expect(output).toContain('$\\times$');
    expect(output).toContain('(N是相机数量)');
    expect(output).not.toContain('$N \\times H \\times W \\times 3 (N$');
  });

  it('不应改写代码块中的 LaTeX 片段', () => {
    const input = '```\\n\\\\boldsymbol{\\\\otimes}\\n```';
    const output = processLatexBrackets(input);
    expect(output).toBe(input);
  });
});
