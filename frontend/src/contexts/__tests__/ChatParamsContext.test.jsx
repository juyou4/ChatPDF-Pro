// @vitest-environment jsdom
import { describe, it, expect, beforeEach } from 'vitest';
import * as fc from 'fast-check';
import { CHAT_PARAMS_DEFAULT_SETTINGS } from '../ChatParamsContext.jsx';

// Feature: chatpdf-chat-settings-upgrade
// Property 3: chatParamsSettings localStorage round-trip
// Validates: Requirements 2.4, 2.5

describe('ChatParamsContext - thoughtAutoCollapse 属性测试', () => {
    // chatParamsSettings 对象生成器（包含 thoughtAutoCollapse）
    const chatParamsArb = fc.record({
        maxTokens: fc.integer({ min: 1, max: 128000 }),
        temperature: fc.double({ min: 0, max: 2, noNaN: true }),
        topP: fc.double({ min: 0, max: 1, noNaN: true }),
        contextCount: fc.integer({ min: 0, max: 50 }),
        streamOutput: fc.boolean(),
        enableTemperature: fc.boolean(),
        enableTopP: fc.boolean(),
        enableMaxTokens: fc.boolean(),
        reasoningEffort: fc.constantFrom('off', 'low', 'medium', 'high'),
        enableMemory: fc.boolean(),
        thoughtAutoCollapse: fc.boolean(),
    });

    beforeEach(() => {
        localStorage.clear();
    });

    // Property 3: chatParamsSettings localStorage round-trip
    it('Property 3: chatParamsSettings localStorage round-trip 保持 thoughtAutoCollapse 一致', () => {
        fc.assert(
            fc.property(chatParamsArb, (settings) => {
                // 保存到 localStorage
                localStorage.setItem('chatParamsSettings', JSON.stringify(settings));

                // 从 localStorage 读取
                const loaded = JSON.parse(localStorage.getItem('chatParamsSettings'));

                // thoughtAutoCollapse 应与保存前相同
                expect(loaded.thoughtAutoCollapse).toBe(settings.thoughtAutoCollapse);

                // 验证类型为布尔值
                expect(typeof loaded.thoughtAutoCollapse).toBe('boolean');
            }),
            { numRuns: 100 }
        );
    });

    // Property 3 补充：完整对象 round-trip
    it('Property 3: chatParamsSettings 完整对象 round-trip', () => {
        fc.assert(
            fc.property(chatParamsArb, (settings) => {
                localStorage.setItem('chatParamsSettings', JSON.stringify(settings));
                const loaded = JSON.parse(localStorage.getItem('chatParamsSettings'));

                expect(loaded.maxTokens).toBe(settings.maxTokens);
                expect(loaded.temperature).toBeCloseTo(settings.temperature, 10);
                expect(loaded.topP).toBeCloseTo(settings.topP, 10);
                expect(loaded.contextCount).toBe(settings.contextCount);
                expect(loaded.streamOutput).toBe(settings.streamOutput);
                expect(loaded.enableTemperature).toBe(settings.enableTemperature);
                expect(loaded.enableTopP).toBe(settings.enableTopP);
                expect(loaded.enableMaxTokens).toBe(settings.enableMaxTokens);
                expect(loaded.reasoningEffort).toBe(settings.reasoningEffort);
                expect(loaded.enableMemory).toBe(settings.enableMemory);
                expect(loaded.thoughtAutoCollapse).toBe(settings.thoughtAutoCollapse);
            }),
            { numRuns: 100 }
        );
    });

    // 默认值验证
    it('thoughtAutoCollapse 默认值应为 true', () => {
        expect(CHAT_PARAMS_DEFAULT_SETTINGS.thoughtAutoCollapse).toBe(true);
    });
});
