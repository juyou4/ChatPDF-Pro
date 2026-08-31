import { describe, expect, it } from 'vitest'

import {
  inferReasoningProfile,
  normalizeReasoningProfile,
  requiresPreservedReasoning,
} from '../reasoningEffortService'


const profile = (providerId, modelId, tags = ['reasoning']) => inferReasoningProfile({
  providerId,
  modelId,
  model: { tags, metadata: {} },
  provider: { apiConfig: {} },
})


describe('reasoning provider capability fallback', () => {
  it('uses Qwen budget controls instead of OpenAI reasoning_effort', () => {
    expect(profile('aliyun', 'qwen3.8-max')).toMatchObject({
      mode: 'qwen_budget',
      options: ['off', 'low', 'medium', 'high', 'max'],
      default: 'high',
      off_control: 'enable_thinking_false',
    })
  })

  it('does not offer off for Claude models that reject disabled thinking', () => {
    expect(profile('anthropic', 'claude-fable-5')).toMatchObject({
      options: ['low', 'medium', 'high', 'xhigh', 'max'],
      always_enabled: true,
      can_disable: false,
    })
  })

  it('uses reasoning_effort none for Grok 4.6 off', () => {
    expect(profile('grok', 'grok-4.6')).toMatchObject({
      options: ['off', 'low', 'medium', 'high', 'xhigh'],
      off_control: 'reasoning_effort_none',
    })
  })

  it('keeps MiniMax M2 thinking enabled', () => {
    expect(profile('minimax', 'MiniMax-M2.7')).toMatchObject({
      mode: 'fixed',
      options: ['medium'],
      always_enabled: true,
      split_reasoning_output: true,
    })
  })

  it('uses model-specific Gemini 3 thinking levels', () => {
    expect(profile('gemini', 'gemini-3.7-flash')).toMatchObject({
      options: ['off', 'minimal', 'low', 'medium', 'high'],
      default: 'high',
      off_control: 'gemini_budget_zero',
    })
    expect(profile('gemini', 'gemini-3.1-pro-preview')).toMatchObject({
      options: ['low', 'high'],
      default: 'high',
    })
  })

  it('removes a fake off option without a native disable control', () => {
    expect(normalizeReasoningProfile({
      mode: 'openai_effort',
      options: ['off', 'low', 'high'],
    }).options).toEqual(['low', 'high'])
  })

  it('preserves required interleaved reasoning families only', () => {
    expect(requiresPreservedReasoning({ modelId: 'MiniMax-M3' })).toBe(true)
    expect(requiresPreservedReasoning({ modelId: 'kimi-k2.7-code' })).toBe(true)
    expect(requiresPreservedReasoning({ modelId: 'gpt-4.1' })).toBe(false)
  })
})
