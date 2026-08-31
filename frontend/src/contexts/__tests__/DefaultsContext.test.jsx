// @vitest-environment jsdom
import { describe, it, expect } from 'vitest';
import { normalizeAssistantKey } from '../DefaultsContext.tsx';

describe('DefaultsContext - assistant model normalization', () => {
    it('should upgrade deprecated model ids to official ids', () => {
        expect(normalizeAssistantKey('gpt-5.2')).toBe('gpt-5.6-terra');
        expect(normalizeAssistantKey('gpt-5.1')).toBe('gpt-5.6-terra');
        expect(normalizeAssistantKey('gpt-5-mini')).toBe('gpt-5.6-luna');
        expect(normalizeAssistantKey('gpt-5-nano')).toBe('gpt-5.6-luna');

        expect(normalizeAssistantKey('claude-opus-4-6')).toBe('claude-opus-5');
        expect(normalizeAssistantKey('claude-sonnet-4-6')).toBe('claude-sonnet-5');
        expect(normalizeAssistantKey('claude-opus-4-5')).toBe('claude-opus-5');
        expect(normalizeAssistantKey('claude-sonnet-4-5')).toBe('claude-sonnet-5');
        expect(normalizeAssistantKey('claude-haiku-4-5')).toBe('claude-haiku-4-5-20251001');
        expect(normalizeAssistantKey('claude-haiku-3-5')).toBe('claude-haiku-4-5-20251001');

        expect(normalizeAssistantKey('gemini-3-pro-preview')).toBe('gemini-3.1-pro-preview');
        expect(normalizeAssistantKey('gemini-3.1-pro')).toBe('gemini-3.1-pro-preview');
        expect(normalizeAssistantKey('gemini-3-flash')).toBe('gemini-3.7-flash');
        expect(normalizeAssistantKey('gemini-3-flash-preview')).toBe('gemini-3.7-flash');

        expect(normalizeAssistantKey('kimi-latest')).toBe('kimi-k3');
        expect(normalizeAssistantKey('kimi-thinking-preview')).toBe('kimi-k3');
        expect(normalizeAssistantKey('kimi-k2.5')).toBe('kimi-k2.6');
        expect(normalizeAssistantKey('kimi-k2')).toBe('kimi-k3');

        expect(normalizeAssistantKey('grok-4.20')).toBe('grok-4.6');
        expect(normalizeAssistantKey('grok-4-1-fast')).toBe('grok-4.6');

        expect(normalizeAssistantKey('MiniMax-Text-01')).toBe('MiniMax-M3');
        expect(normalizeAssistantKey('abab6.5s-chat')).toBe('MiniMax-M3');

        expect(normalizeAssistantKey('glm-4-air')).toBe('glm-5.3');

        expect(normalizeAssistantKey('Doubao-Seed-1.6-lite')).toBe('doubao-seed-evolving');
        expect(normalizeAssistantKey('doubao-seed-2-0-pro')).toBe('doubao-seed-evolving');
        expect(normalizeAssistantKey('doubao-seed-2.0-pro')).toBe('doubao-seed-evolving');
        expect(normalizeAssistantKey('doubao:doubao-seed-2-0-code-preview-260215')).toBe('doubao:doubao-seed-evolving');
        expect(normalizeAssistantKey('deepseek-ai/DeepSeek-V3')).toBe('deepseek-ai/DeepSeek-V4-Flash');
        expect(normalizeAssistantKey('Qwen/Qwen3-235B-A22B')).toBe('Qwen/Qwen3-32B');
        expect(normalizeAssistantKey('deepseek:deepseek-chat')).toBe('deepseek:deepseek-v4-flash');
        expect(normalizeAssistantKey('deepseek:deepseek-reasoner')).toBe('deepseek:deepseek-v4-flash');
    });
});
