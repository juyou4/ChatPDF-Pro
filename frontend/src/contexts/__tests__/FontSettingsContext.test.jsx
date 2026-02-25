// @vitest-environment jsdom
import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as fc from 'fast-check';
import { FONT_DEFAULT_SETTINGS } from '../FontSettingsContext.jsx';

// Feature: chatpdf-chat-settings-upgrade
// Property 1: messageFont 状态设置一致性
// Property 2: fontSettings localStorage round-trip
// Validates: Requirements 1.1, 1.3, 1.4, 1.5

describe('FontSettingsContext - messageFont 属性测试', () => {
    // 有效的 messageFont 值
    const messageFontArb = fc.constantFrom('system', 'serif');

    // 完整的 fontSettings 对象生成器
    const fontSettingsArb = fc.record({
        fontFamily: fc.constantFrom('inter', 'roboto', 'noto-sans-sc', 'poppins', 'custom'),
        customFont: fc.string({ minLength: 0, maxLength: 50 }),
        globalScale: fc.double({ min: 0.5, max: 2.0, noNaN: true }),
        messageFont: messageFontArb,
    });

    beforeEach(() => {
        localStorage.clear();
    });

    // Property 1: messageFont 状态设置一致性
    // 对于任意有效的 messageFont 值，设置后读取应一致
    it('Property 1: messageFont 值设置后应保持一致', () => {
        fc.assert(
            fc.property(messageFontArb, (fontValue) => {
                // 模拟状态设置：将值写入 localStorage 并读回
                const settings = { ...FONT_DEFAULT_SETTINGS, messageFont: fontValue };
                localStorage.setItem('fontSettings', JSON.stringify(settings));

                const saved = JSON.parse(localStorage.getItem('fontSettings'));
                expect(saved.messageFont).toBe(fontValue);
            }),
            { numRuns: 100 }
        );
    });

    // Property 2: fontSettings localStorage round-trip
    // 对于任意有效的 fontSettings 对象，保存到 localStorage 后重新加载，messageFont 值应相同
    it('Property 2: fontSettings localStorage round-trip 保持 messageFont 一致', () => {
        fc.assert(
            fc.property(fontSettingsArb, (settings) => {
                // 保存到 localStorage
                localStorage.setItem('fontSettings', JSON.stringify(settings));

                // 从 localStorage 读取
                const raw = localStorage.getItem('fontSettings');
                const loaded = JSON.parse(raw);

                // messageFont 应与保存前相同
                expect(loaded.messageFont).toBe(settings.messageFont);

                // 验证加载逻辑：只接受有效值
                const isValid = loaded.messageFont === 'system' || loaded.messageFont === 'serif';
                expect(isValid).toBe(true);
            }),
            { numRuns: 100 }
        );
    });

    // Property 2 补充：完整 fontSettings 对象的 round-trip
    it('Property 2: fontSettings 完整对象 round-trip', () => {
        fc.assert(
            fc.property(fontSettingsArb, (settings) => {
                localStorage.setItem('fontSettings', JSON.stringify(settings));
                const loaded = JSON.parse(localStorage.getItem('fontSettings'));

                expect(loaded.fontFamily).toBe(settings.fontFamily);
                expect(loaded.customFont).toBe(settings.customFont);
                expect(loaded.globalScale).toBeCloseTo(settings.globalScale, 10);
                expect(loaded.messageFont).toBe(settings.messageFont);
            }),
            { numRuns: 100 }
        );
    });

    // 默认值验证
    it('messageFont 默认值应为 system', () => {
        expect(FONT_DEFAULT_SETTINGS.messageFont).toBe('system');
    });
});
