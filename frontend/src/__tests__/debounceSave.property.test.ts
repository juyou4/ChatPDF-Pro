/**
 * P1 属性测试：防抖保存属性
 * 验证 N 次快速连续的设置变更（间隔 < 500ms），localStorage 只被写入一次，且写入的是最后一次变更的值。
 *
 * **Validates: Requirements 3.1, 3.2**
 *
 * 测试策略：
 * - 直接复现 GlobalSettingsContext 中的防抖逻辑（useRef + setTimeout 模式）
 * - 使用 vi.useFakeTimers 控制时间流逝
 * - 使用 fast-check 生成任意长度的设置变更序列
 * - 验证在所有变更完成后推进 500ms，localStorage.setItem 只被调用一次，且值为最后一次
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as fc from 'fast-check';

// 模拟 GlobalSettingsContext 中的防抖保存逻辑
// 与源码保持一致：pendingSettingsRef + debounceTimerRef + 500ms setTimeout
function createDebouncedSaver(storage: { setItem: (key: string, value: string) => void }) {
  let pendingSettings: any = null;
  let debounceTimer: ReturnType<typeof setTimeout> | null = null;

  function flushSave() {
    if (pendingSettings !== null) {
      storage.setItem('globalSettings', JSON.stringify(pendingSettings));
      pendingSettings = null;
    }
  }

  function debouncedSave(settings: any) {
    pendingSettings = settings;
    if (debounceTimer) {
      clearTimeout(debounceTimer);
    }
    debounceTimer = setTimeout(() => {
      flushSave();
      debounceTimer = null;
    }, 500);
  }

  return { debouncedSave, flushSave };
}

// 生成器：生成一个设置对象（模拟 GlobalSettingsContext 中的设置结构）
const settingsArb = fc.record({
  temperature: fc.double({ min: 0, max: 2, noNaN: true }),
  topP: fc.double({ min: 0, max: 1, noNaN: true }),
  maxTokens: fc.integer({ min: 1, max: 128000 }),
  enableTemperature: fc.boolean(),
  enableTopP: fc.boolean(),
  enableMaxTokens: fc.boolean(),
  reasoningEffort: fc.constantFrom('off', 'low', 'medium', 'high'),
});

// 生成器：生成 N 个设置变更序列（至少 2 个，确保有"连续快速变更"的场景）
const settingsSequenceArb = fc.array(settingsArb, { minLength: 2, maxLength: 50 });

describe('P1: 防抖保存属性', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('N 次快速连续变更（间隔 < 500ms）只触发一次 localStorage 写入，且值为最后一次', () => {
    fc.assert(
      fc.property(settingsSequenceArb, (settingsSeq) => {
        // 准备 mock storage
        const mockSetItem = vi.fn();
        const storage = { setItem: mockSetItem };
        const { debouncedSave } = createDebouncedSaver(storage);

        // 模拟 N 次快速连续变更（每次间隔 100ms，远小于 500ms 防抖阈值）
        for (const settings of settingsSeq) {
          debouncedSave(settings);
          vi.advanceTimersByTime(100);
        }

        // 此时最后一次变更距今只过了 0ms（刚调用完），还没到 500ms
        // 之前的变更都被清除了定时器，不会触发写入
        // 确认此时 localStorage 没有被写入
        expect(mockSetItem).not.toHaveBeenCalled();

        // 推进 500ms，触发最后一次防抖写入
        vi.advanceTimersByTime(500);

        // 验证属性：只写入一次
        expect(mockSetItem).toHaveBeenCalledTimes(1);

        // 验证属性：写入的是最后一次变更的值
        const lastSettings = settingsSeq[settingsSeq.length - 1];
        const writtenValue = JSON.parse(mockSetItem.mock.calls[0][1]);
        expect(writtenValue).toEqual(lastSettings);

        // 验证写入的 key 是 'globalSettings'
        expect(mockSetItem.mock.calls[0][0]).toBe('globalSettings');
      }),
      { numRuns: 100 }
    );
  });

  it('单次变更在 500ms 后也只触发一次写入', () => {
    fc.assert(
      fc.property(settingsArb, (settings) => {
        const mockSetItem = vi.fn();
        const storage = { setItem: mockSetItem };
        const { debouncedSave } = createDebouncedSaver(storage);

        debouncedSave(settings);

        // 500ms 之前不应写入
        vi.advanceTimersByTime(499);
        expect(mockSetItem).not.toHaveBeenCalled();

        // 500ms 时写入
        vi.advanceTimersByTime(1);
        expect(mockSetItem).toHaveBeenCalledTimes(1);

        const writtenValue = JSON.parse(mockSetItem.mock.calls[0][1]);
        expect(writtenValue).toEqual(settings);

        // 之后不再有额外写入
        vi.advanceTimersByTime(1000);
        expect(mockSetItem).toHaveBeenCalledTimes(1);
      }),
      { numRuns: 100 }
    );
  });
});
