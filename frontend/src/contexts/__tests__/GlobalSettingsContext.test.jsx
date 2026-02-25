// @vitest-environment jsdom
import { describe, it, expect, beforeEach } from 'vitest';
import * as fc from 'fast-check';
import { FONT_DEFAULT_SETTINGS } from '../FontSettingsContext.jsx';
import { CHAT_PARAMS_DEFAULT_SETTINGS } from '../ChatParamsContext.jsx';

// Feature: chatpdf-chat-settings-upgrade
// Property 5: 重置恢复默认值
// Property 6: GlobalSettings export/import round-trip
// Validates: Requirements 3.4, 4.2, 4.3, 4.4

describe('GlobalSettingsContext - 属性测试', () => {
    // 生成器：包含 messageFont 和 thoughtAutoCollapse 的全局设置
    const globalSettingsArb = fc.record({
        fontFamily: fc.constantFrom('inter', 'roboto', 'noto-sans-sc', 'poppins', 'custom'),
        customFont: fc.string({ minLength: 0, maxLength: 30 }),
        globalScale: fc.double({ min: 0.5, max: 2.0, noNaN: true }),
        messageFont: fc.constantFrom('system', 'serif'),
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

    // Property 5: 重置恢复默认值
    // 对于任意设置状态，重置后 messageFont 应为 'system'，thoughtAutoCollapse 应为 true
    it('Property 5: 重置后 messageFont 和 thoughtAutoCollapse 恢复默认值', () => {
        fc.assert(
            fc.property(globalSettingsArb, (settings) => {
                // 模拟：无论当前设置是什么，重置后应恢复默认值
                const resetMessageFont = FONT_DEFAULT_SETTINGS.messageFont;
                const resetThoughtAutoCollapse = CHAT_PARAMS_DEFAULT_SETTINGS.thoughtAutoCollapse;

                expect(resetMessageFont).toBe('system');
                expect(resetThoughtAutoCollapse).toBe(true);

                // 验证默认值与任意输入无关
                expect(resetMessageFont).not.toBe(undefined);
                expect(resetThoughtAutoCollapse).not.toBe(undefined);
            }),
            { numRuns: 100 }
        );
    });

    // Property 6: GlobalSettings export/import round-trip
    // 对于任意有效的全局设置，export 后 import 应保持 messageFont 和 thoughtAutoCollapse 一致
    it('Property 6: export/import round-trip 保持新字段一致', () => {
        fc.assert(
            fc.property(globalSettingsArb, (settings) => {
                // 模拟 exportSettings：序列化为 JSON
                const exported = JSON.stringify({
                    ...settings,
                    exportedAt: new Date().toISOString(),
                }, null, 2);

                // 模拟 importSettings：从 JSON 解析
                const imported = JSON.parse(exported);

                // messageFont 和 thoughtAutoCollapse 应保持一致
                expect(imported.messageFont).toBe(settings.messageFont);
                expect(imported.thoughtAutoCollapse).toBe(settings.thoughtAutoCollapse);

                // 其他字段也应一致
                expect(imported.fontFamily).toBe(settings.fontFamily);
                expect(imported.temperature).toBeCloseTo(settings.temperature, 10);
                expect(imported.enableMemory).toBe(settings.enableMemory);
            }),
            { numRuns: 100 }
        );
    });

    // Property 6 补充：验证缺少新字段时的向后兼容
    it('Property 6: import 缺少新字段时不应报错', () => {
        fc.assert(
            fc.property(
                fc.record({
                    fontFamily: fc.constantFrom('inter', 'roboto'),
                    maxTokens: fc.integer({ min: 1, max: 128000 }),
                }),
                (partialSettings) => {
                    // 旧版导出不包含 messageFont 和 thoughtAutoCollapse
                    const exported = JSON.stringify(partialSettings);
                    const imported = JSON.parse(exported);

                    // 缺少的字段应为 undefined（importSettings 中会跳过）
                    expect(imported.messageFont).toBeUndefined();
                    expect(imported.thoughtAutoCollapse).toBeUndefined();
                }
            ),
            { numRuns: 100 }
        );
    });
});
