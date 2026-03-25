import { describe, it, expect } from 'vitest';
import { shouldStreamAssistantContent } from '../messageRenderUtils.js';

describe('shouldStreamAssistantContent', () => {
  it('仅对当前正在流式的消息返回 true', () => {
    expect(shouldStreamAssistantContent({ id: 101, isStreaming: true }, 101)).toBe(true);
  });

  it('非当前消息或非流式消息返回 false', () => {
    expect(shouldStreamAssistantContent({ id: 101, isStreaming: true }, 202)).toBe(false);
    expect(shouldStreamAssistantContent({ id: 101, isStreaming: false }, 101)).toBe(false);
    expect(shouldStreamAssistantContent(null, 101)).toBe(false);
  });
});
