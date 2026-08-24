/** @vitest-environment jsdom */
import { render } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import StreamingMarkdown from '../StreamingMarkdown';

vi.mock('../../contexts/ChatParamsContext', () => ({
  useChatParams: () => ({
    codeCollapsible: false,
    codeWrappable: true,
    codeShowLineNumbers: false,
    mathEngine: 'KaTeX',
    mathEnableSingleDollar: true,
  }),
}));

describe('StreamingMarkdown math without blur reveal', () => {
  it('renders inline $x_{adv}$ and a raw \\min objective', () => {
    const content = [
      '对抗补丁记为 $x_{adv}$，再送入检测器。',
      '',
      '\\min_{x_{adv}} L_{adv} = E_{\\theta}(L_{det}(x_{o}))',
      '',
      '实线表示梯度回传。',
    ].join('\n');

    const { container } = render(
      <StreamingMarkdown
        content={content}
        isStreaming={false}
        enableBlurReveal={false}
      />,
    );

    expect(container.querySelectorAll('.katex').length).toBeGreaterThan(0);
    expect(container.textContent).not.toContain('$x_{adv}$');
    expect(container.textContent).not.toMatch(/\\min_\{x_\{adv\}\}/);
  });

  it('still renders math after citation HTML is injected', () => {
    const { container } = render(
      <StreamingMarkdown
        content={'该方法生成 $x_{adv}$。[1]'}
        isStreaming={false}
        enableBlurReveal={false}
        citations={[{ ref: 1, display_ref: 1, text: 'sec 3' }]}
      />,
    );

    expect(container.querySelector('.katex')).toBeTruthy();
    expect(container.textContent).not.toContain('$x_{adv}$');
  });
});
