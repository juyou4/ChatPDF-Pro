// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import DocumentParseStatusBar, { PARSE_NOTICE_SEEN_STORAGE_KEY } from '../DocumentParseStatusBar.jsx';

const manifest = (overrides = {}) => ({
  route: 'mineru',
  requested_route: 'mineru',
  resolved_route: 'mineru',
  generation: 'parse-ui-1',
  source_hash: 'source-ui-1',
  status: 'pending',
  stage: 'awaiting_rag_index',
  metadata: { full_route: true },
  ...overrides,
});

describe('DocumentParseStatusBar terminal and paused states', () => {
  beforeEach(() => {
    localStorage.removeItem(PARSE_NOTICE_SEEN_STORAGE_KEY);
  });

  it('shows awaiting publication without a progress bar or spinner', () => {
    const { container } = render(
      <DocumentParseStatusBar
        documentId="doc-ui"
        manifest={manifest()}
        parseReady={false}
        deepParseStatus={{ status: 'ready', stage: 'awaiting_rag_index', active: false }}
        ragIndexStatus={{ status: 'missing', ready: false }}
        onOpenProcessing={vi.fn()}
      />,
    );

    expect(screen.getByText('等待问答索引发布')).toBeTruthy();
    expect(container.querySelector('[role="progressbar"]')).toBeNull();
    expect(container.querySelector('.animate-spin')).toBeNull();
  });

  it('shows a recoverable index failure with a direct retry action', () => {
    const onRetryIndex = vi.fn();
    const { container } = render(
      <DocumentParseStatusBar
        documentId="doc-ui"
        manifest={manifest({ metadata: {
          full_route: true,
          rag_index_failure: { preserve_parse: true },
        } })}
        parseReady={false}
        deepParseStatus={{ status: 'ready', stage: 'awaiting_rag_index', active: false }}
        ragIndexStatus={{
          status: 'failed',
          preserve_parse: true,
          error: 'Embedding 余额不足，MinerU 结果已保留',
        }}
        onRetryIndex={onRetryIndex}
      />,
    );

    expect(screen.getByText('问答索引发布失败')).toBeTruthy();
    expect(screen.getByRole('button', { name: '重试发布' })).toBeTruthy();
    expect(container.querySelector('[role="progressbar"]')).toBeNull();
  });

  it('offers restart after cancellation instead of leaving an active indicator', () => {
    const onRetry = vi.fn();
    const { container } = render(
      <DocumentParseStatusBar
        documentId="doc-ui"
        manifest={manifest({ status: 'cancelled', stage: 'cancelled' })}
        parseReady={false}
        deepParseStatus={{ status: 'cancelled', stage: 'cancelled', active: false }}
        ragIndexStatus={{ status: 'missing' }}
        onRetry={onRetry}
      />,
    );

    expect(screen.getByText('解析已取消')).toBeTruthy();
    expect(screen.getByRole('button', { name: '开始解析' })).toBeTruthy();
    expect(container.querySelector('.animate-spin')).toBeNull();
  });
});
