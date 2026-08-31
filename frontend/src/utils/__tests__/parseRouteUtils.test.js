import { describe, expect, it } from 'vitest';
import { resolveDocumentParseState, shouldPollMinerUStatus } from '../parseRouteUtils.js';

const mineruManifest = (overrides = {}) => ({
  route: 'mineru',
  requested_route: 'mineru',
  resolved_route: 'mineru',
  generation: 'parse-1',
  source_hash: 'source-1',
  status: 'running',
  stage: 'mineru_parsing',
  metadata: { full_route: true },
  ...overrides,
});

describe('document parse status presentation', () => {
  it('does not treat an idle MinerU probe as a failure for a local document', () => {
    const state = resolveDocumentParseState({
      manifest: {
        ...mineruManifest(),
        route: 'local',
        requested_route: 'local',
        resolved_route: 'local',
        status: 'ready',
        stage: 'local_ready',
        metadata: { full_route: false },
      },
      parseReady: true,
      deepParseStatus: { status: 'idle', stage: 'not_started' },
    });

    expect(state.state).toBe('ready');
  });

  it('prefers a terminal sync failure over a stale running manifest', () => {
    const state = resolveDocumentParseState({
      manifest: mineruManifest({ stage: 'building_rag_index' }),
      parseReady: false,
      deepParseStatus: {
        status: 'failed',
        stage: 'status_sync_failed',
        error_code: 'status_sync_failed',
        error: '解析状态连续同步失败，已暂停等待',
      },
    });

    expect(state.state).toBe('failed');
    expect(state.statusLabel).toBe('状态同步失败');
    expect(state.detail).toContain('已暂停等待');
  });

  it('shows awaiting publication as a paused action state', () => {
    const state = resolveDocumentParseState({
      manifest: mineruManifest({ status: 'pending', stage: 'awaiting_rag_index' }),
      parseReady: false,
      deepParseStatus: { status: 'ready', stage: 'awaiting_rag_index', active: false },
      ragIndexStatus: { status: 'missing', ready: false },
    });

    expect(state.state).toBe('awaiting_publish');
    expect(shouldPollMinerUStatus({
      status: 'ready',
      primaryMinerURoute: true,
      routePending: true,
      awaitingPublish: true,
    })).toBe(false);
  });

  it('fails closed on an unknown active MinerU status', () => {
    const state = resolveDocumentParseState({
      manifest: mineruManifest(),
      parseReady: false,
      deepParseStatus: { status: 'mystery_state', stage: 'polling' },
    });

    expect(state.state).toBe('failed');
    expect(state.statusLabel).toBe('状态同步失败');
    expect(state.detail).toContain('无法识别');
  });

  it('distinguishes a pending task that never started', () => {
    const state = resolveDocumentParseState({
      manifest: mineruManifest({ status: 'pending', stage: 'selected' }),
      parseReady: false,
      deepParseStatus: { status: 'pending', stage: 'selected', active: false },
    });

    expect(state.state).toBe('not_started');
    expect(state.statusLabel).toBe('尚未开始解析');
    expect(shouldPollMinerUStatus({
      status: 'pending',
      primaryMinerURoute: true,
      routePending: true,
      notStarted: true,
    })).toBe(false);
  });
});
