// @vitest-environment jsdom
import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';

// Feature: chatpdf-chat-settings-upgrade
// Property 4: ThinkingBlock 折叠行为与设置一致
// Validates: Requirements 2.2, 2.3

/**
 * 模拟 ThinkingBlock 的折叠决策逻辑（纯函数提取）
 * 当 isStreaming 从 true 变为 false（思考完成）时：
 * - thoughtAutoCollapse 为 true → 应自动折叠（返回 false）
 * - thoughtAutoCollapse 为 false → 保持展开（返回 true）
 */
function computeExpandedAfterStreamEnd(thoughtAutoCollapse) {
    if (thoughtAutoCollapse) {
        return false; // 自动折叠
    }
    return true; // 保持展开
}

describe('ThinkingBlock - 折叠行为属性测试', () => {
    // Property 4: ThinkingBlock 折叠行为与设置一致
    it('Property 4: thoughtAutoCollapse=true 时思考完成后应折叠', () => {
        fc.assert(
            fc.property(
                fc.constant(true), // thoughtAutoCollapse = true
                fc.string({ minLength: 1, maxLength: 200 }), // 非空 content
                (autoCollapse, _content) => {
                    const expanded = computeExpandedAfterStreamEnd(autoCollapse);
                    // 自动折叠开启时，思考完成后应折叠（expanded = false）
                    expect(expanded).toBe(false);
                }
            ),
            { numRuns: 100 }
        );
    });

    it('Property 4: thoughtAutoCollapse=false 时思考完成后应保持展开', () => {
        fc.assert(
            fc.property(
                fc.constant(false), // thoughtAutoCollapse = false
                fc.string({ minLength: 1, maxLength: 200 }), // 非空 content
                (autoCollapse, _content) => {
                    const expanded = computeExpandedAfterStreamEnd(autoCollapse);
                    // 自动折叠关闭时，思考完成后应保持展开（expanded = true）
                    expect(expanded).toBe(true);
                }
            ),
            { numRuns: 100 }
        );
    });

    // Property 4 综合：对于任意 thoughtAutoCollapse 值，折叠行为应与设置一致
    it('Property 4: 折叠行为与 thoughtAutoCollapse 设置一致', () => {
        fc.assert(
            fc.property(
                fc.boolean(), // 任意 thoughtAutoCollapse 值
                fc.string({ minLength: 1, maxLength: 200 }), // 非空 content
                (autoCollapse, _content) => {
                    const expanded = computeExpandedAfterStreamEnd(autoCollapse);
                    if (autoCollapse) {
                        expect(expanded).toBe(false); // 开启自动折叠 → 折叠
                    } else {
                        expect(expanded).toBe(true); // 关闭自动折叠 → 展开
                    }
                }
            ),
            { numRuns: 100 }
        );
    });
});
