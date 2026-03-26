// @vitest-environment jsdom
import { describe, it, expect } from 'vitest';
import { normalizeAssistantKey } from '../DefaultsContext.tsx';

describe('DefaultsContext - assistant model normalization', () => {
    it('should upgrade deprecated model ids to official ids', () => {
        expect(normalizeAssistantKey('gpt-5.4')).toBe('gpt-5.2');
        expect(normalizeAssistantKey('gpt-5.4-pro')).toBe('gpt-5.2');
        expect(normalizeAssistantKey('gpt-5.4-mini')).toBe('gpt-5-mini');
        expect(normalizeAssistantKey('gpt-5.4-nano')).toBe('gpt-5-nano');

        expect(normalizeAssistantKey('claude-opus-4-6')).toBe('claude-opus-4-1-20250805');
        expect(normalizeAssistantKey('claude-sonnet-4-6')).toBe('claude-sonnet-4-20250514');
        expect(normalizeAssistantKey('claude-opus-4-5')).toBe('claude-opus-4-20250514');
        expect(normalizeAssistantKey('claude-sonnet-4-5')).toBe('claude-sonnet-4-20250514');
        expect(normalizeAssistantKey('claude-haiku-4-5')).toBe('claude-3-5-haiku-20241022');
        expect(normalizeAssistantKey('claude-haiku-3-5')).toBe('claude-3-5-haiku-20241022');

        expect(normalizeAssistantKey('gemini-3.1-pro')).toBe('gemini-3-pro-preview');
        expect(normalizeAssistantKey('gemini-3.1-pro-preview')).toBe('gemini-3-pro-preview');
        expect(normalizeAssistantKey('gemini-3-flash')).toBe('gemini-3-flash-preview');

        expect(normalizeAssistantKey('kimi-k2.5')).toBe('kimi-thinking-preview');
        expect(normalizeAssistantKey('kimi-k2')).toBe('kimi-k2-0905-preview');

        expect(normalizeAssistantKey('grok-4.20')).toBe('grok-4.20-beta-latest-non-reasoning');
        expect(normalizeAssistantKey('grok-4-1-fast')).toBe('grok-4-1-fast-reasoning');

        expect(normalizeAssistantKey('MiniMax-Text-01')).toBe('MiniMax-M2.5');
        expect(normalizeAssistantKey('abab6.5s-chat')).toBe('MiniMax-M2.1');

        expect(normalizeAssistantKey('glm-4-air')).toBe('glm-4-air-250414');

        expect(normalizeAssistantKey('doubao-seed-2-0-pro')).toBe('doubao-seed-2-0-pro-260215');
        expect(normalizeAssistantKey('doubao-seed-2.0-pro')).toBe('doubao-seed-2-0-pro-260215');
        expect(normalizeAssistantKey('doubao:doubao-seed-2-0-code-preview-260215')).toBe('doubao:doubao-seed-code-preview-latest');
        expect(normalizeAssistantKey('deepseek-ai/DeepSeek-V3')).toBe('deepseek-ai/DeepSeek-V3.2');
        expect(normalizeAssistantKey('Qwen/Qwen3-235B-A22B')).toBe('Qwen/Qwen3-32B');
        expect(normalizeAssistantKey('deepseek:deepseek-chat')).toBe('deepseek:deepseek-chat');
    });
});
