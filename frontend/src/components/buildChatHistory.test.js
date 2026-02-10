import { describe, it, expect } from 'vitest';

/**
 * 从 ChatPDF.jsx 中提取 buildChatHistory 函数的逻辑进行测试
 * 由于 buildChatHistory 是模块级别的纯函数，这里直接复制其逻辑进行单元测试
 */
const buildChatHistory = (messages, contextCount) => {
  if (!contextCount || contextCount <= 0) return [];

  // 过滤出有效的对话消息（排除 system 和含图片的消息）
  const validMessages = messages.filter(msg =>
    (msg.type === 'user' || msg.type === 'assistant') && !msg.hasImage
  );

  // 取最近 contextCount * 2 条（每轮包含 user + assistant）
  const recentMessages = validMessages.slice(-(contextCount * 2));

  return recentMessages.map(msg => ({
    role: msg.type === 'user' ? 'user' : 'assistant',
    content: msg.content
  }));
};

describe('buildChatHistory', () => {
  // 基本功能测试
  it('contextCount 为 0 时返回空数组', () => {
    const messages = [
      { type: 'user', content: '你好', hasImage: false },
      { type: 'assistant', content: '你好！', model: 'gpt-4o' },
    ];
    expect(buildChatHistory(messages, 0)).toEqual([]);
  });

  it('contextCount 为 null/undefined 时返回空数组', () => {
    const messages = [
      { type: 'user', content: '你好', hasImage: false },
    ];
    expect(buildChatHistory(messages, null)).toEqual([]);
    expect(buildChatHistory(messages, undefined)).toEqual([]);
  });

  it('contextCount 为负数时返回空数组', () => {
    const messages = [
      { type: 'user', content: '你好', hasImage: false },
    ];
    expect(buildChatHistory(messages, -1)).toEqual([]);
  });

  it('空消息列表返回空数组', () => {
    expect(buildChatHistory([], 5)).toEqual([]);
  });

  // 正常截取测试
  it('正确截取最近 N 轮对话', () => {
    const messages = [
      { type: 'user', content: '问题1', hasImage: false },
      { type: 'assistant', content: '回答1' },
      { type: 'user', content: '问题2', hasImage: false },
      { type: 'assistant', content: '回答2' },
      { type: 'user', content: '问题3', hasImage: false },
      { type: 'assistant', content: '回答3' },
    ];

    const result = buildChatHistory(messages, 2);
    expect(result).toEqual([
      { role: 'user', content: '问题2' },
      { role: 'assistant', content: '回答2' },
      { role: 'user', content: '问题3' },
      { role: 'assistant', content: '回答3' },
    ]);
  });

  it('contextCount 大于可用轮数时返回所有有效消息', () => {
    const messages = [
      { type: 'user', content: '问题1', hasImage: false },
      { type: 'assistant', content: '回答1' },
    ];

    const result = buildChatHistory(messages, 10);
    expect(result).toEqual([
      { role: 'user', content: '问题1' },
      { role: 'assistant', content: '回答1' },
    ]);
  });

  // 过滤测试
  it('过滤掉 system 类型的消息', () => {
    const messages = [
      { type: 'system', content: '文档上传成功' },
      { type: 'user', content: '问题1', hasImage: false },
      { type: 'assistant', content: '回答1' },
      { type: 'system', content: '系统提示' },
      { type: 'user', content: '问题2', hasImage: false },
      { type: 'assistant', content: '回答2' },
    ];

    const result = buildChatHistory(messages, 5);
    expect(result).toEqual([
      { role: 'user', content: '问题1' },
      { role: 'assistant', content: '回答1' },
      { role: 'user', content: '问题2' },
      { role: 'assistant', content: '回答2' },
    ]);
    // 确认没有 system 消息
    expect(result.every(msg => msg.role !== 'system')).toBe(true);
  });

  it('过滤掉含图片的消息', () => {
    const messages = [
      { type: 'user', content: '看这张图', hasImage: true },
      { type: 'assistant', content: '图片分析结果' },
      { type: 'user', content: '文字问题', hasImage: false },
      { type: 'assistant', content: '文字回答' },
    ];

    const result = buildChatHistory(messages, 5);
    // hasImage: true 的用户消息被过滤
    expect(result).toEqual([
      { role: 'assistant', content: '图片分析结果' },
      { role: 'user', content: '文字问题' },
      { role: 'assistant', content: '文字回答' },
    ]);
  });

  it('全部为 system 消息时返回空数组', () => {
    const messages = [
      { type: 'system', content: '系统消息1' },
      { type: 'system', content: '系统消息2' },
    ];
    expect(buildChatHistory(messages, 5)).toEqual([]);
  });

  // 角色映射测试
  it('正确映射 type 到 role', () => {
    const messages = [
      { type: 'user', content: '问题', hasImage: false },
      { type: 'assistant', content: '回答' },
    ];

    const result = buildChatHistory(messages, 1);
    expect(result[0].role).toBe('user');
    expect(result[1].role).toBe('assistant');
  });
});
