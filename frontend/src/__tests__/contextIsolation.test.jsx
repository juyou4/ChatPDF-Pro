// @vitest-environment jsdom
/**
 * Feature: chatpdf-frontend-performance, Property 2: Context 细粒度订阅隔离
 *
 * **Validates: Requirements 2.1, 2.2, 2.3**
 *
 * 验证 Context 拆分后的渲染隔离性：
 * - 字体设置变更不触发对话参数消费者重渲染
 * - 对话参数变更不触发字体设置消费者重渲染
 * - useGlobalSettings 消费者在任意设置变更时都会重渲染（因为它订阅了所有值）
 */

import React, { useRef, useEffect } from 'react';
import { describe, it, expect, beforeEach } from 'vitest';
import { render, act } from '@testing-library/react';
import { GlobalSettingsProvider } from '../contexts/GlobalSettingsContext';
import { useFontSettings } from '../contexts/FontSettingsContext';
import { useChatParams } from '../contexts/ChatParamsContext';
import { useGlobalSettings } from '../contexts/GlobalSettingsContext';

// ========== 渲染计数器测试组件 ==========

/**
 * 字体设置消费者 —— 仅订阅 useFontSettings
 * 通过 renderCount ref 追踪渲染次数
 */
function FontConsumer({ onRender }) {
  const renderCount = useRef(0);
  const { fontFamily, globalScale } = useFontSettings();

  renderCount.current += 1;

  // 每次渲染时通知外部当前渲染次数
  useEffect(() => {
    if (onRender) onRender(renderCount.current);
  });

  return (
    <div data-testid="font-consumer">
      <span data-testid="font-render-count">{renderCount.current}</span>
      <span data-testid="font-family">{fontFamily}</span>
      <span data-testid="global-scale">{globalScale}</span>
    </div>
  );
}

/**
 * 对话参数消费者 —— 仅订阅 useChatParams
 */
function ChatParamsConsumer({ onRender }) {
  const renderCount = useRef(0);
  const { temperature, maxTokens } = useChatParams();

  renderCount.current += 1;

  useEffect(() => {
    if (onRender) onRender(renderCount.current);
  });

  return (
    <div data-testid="chatparams-consumer">
      <span data-testid="chatparams-render-count">{renderCount.current}</span>
      <span data-testid="temperature">{temperature}</span>
      <span data-testid="max-tokens">{maxTokens}</span>
    </div>
  );
}

/**
 * 全局设置消费者 —— 订阅 useGlobalSettings（聚合层，订阅所有值）
 */
function GlobalConsumer({ onRender }) {
  const renderCount = useRef(0);
  const { fontFamily, temperature } = useGlobalSettings();

  renderCount.current += 1;

  useEffect(() => {
    if (onRender) onRender(renderCount.current);
  });

  return (
    <div data-testid="global-consumer">
      <span data-testid="global-render-count">{renderCount.current}</span>
      <span data-testid="global-font-family">{fontFamily}</span>
      <span data-testid="global-temperature">{temperature}</span>
    </div>
  );
}

/**
 * 触发器组件 —— 提供按钮来修改字体设置和对话参数
 * 同时使用两个细粒度 Hook 以获取 setter 方法
 */
function SettingsTrigger() {
  const { setFontFamily, setGlobalScale } = useFontSettings();
  const { setTemperature, setMaxTokens } = useChatParams();

  return (
    <div data-testid="triggers">
      <button
        data-testid="change-font"
        onClick={() => setFontFamily('roboto')}
      >
        切换字体
      </button>
      <button
        data-testid="change-scale"
        onClick={() => setGlobalScale(1.5)}
      >
        修改缩放
      </button>
      <button
        data-testid="change-temperature"
        onClick={() => setTemperature(0.9)}
      >
        修改温度
      </button>
      <button
        data-testid="change-max-tokens"
        onClick={() => setMaxTokens(4096)}
      >
        修改最大 Token
      </button>
    </div>
  );
}

// ========== 测试用例 ==========

describe('Property 2: Context 细粒度订阅隔离', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('字体设置变更不触发对话参数消费者重渲染', () => {
    let fontRenderCount = 0;
    let chatParamsRenderCount = 0;

    const { getByTestId } = render(
      <GlobalSettingsProvider>
        <FontConsumer onRender={(count) => { fontRenderCount = count; }} />
        <ChatParamsConsumer onRender={(count) => { chatParamsRenderCount = count; }} />
        <SettingsTrigger />
      </GlobalSettingsProvider>
    );

    // 初始渲染后，两个消费者各渲染 1 次
    const initialFontCount = fontRenderCount;
    const initialChatParamsCount = chatParamsRenderCount;

    // 修改字体设置
    act(() => {
      getByTestId('change-font').click();
    });

    // 字体消费者应该重渲染（渲染次数增加）
    expect(fontRenderCount).toBeGreaterThan(initialFontCount);
    // 对话参数消费者不应重渲染（渲染次数不变）
    expect(chatParamsRenderCount).toBe(initialChatParamsCount);

    // 验证字体值确实更新了
    expect(getByTestId('font-family').textContent).toBe('roboto');
  });

  it('缩放设置变更不触发对话参数消费者重渲染', () => {
    let fontRenderCount = 0;
    let chatParamsRenderCount = 0;

    const { getByTestId } = render(
      <GlobalSettingsProvider>
        <FontConsumer onRender={(count) => { fontRenderCount = count; }} />
        <ChatParamsConsumer onRender={(count) => { chatParamsRenderCount = count; }} />
        <SettingsTrigger />
      </GlobalSettingsProvider>
    );

    const initialChatParamsCount = chatParamsRenderCount;

    // 修改缩放
    act(() => {
      getByTestId('change-scale').click();
    });

    // 字体消费者应该重渲染（globalScale 属于字体设置）
    expect(fontRenderCount).toBeGreaterThan(1);
    // 对话参数消费者不应重渲染
    expect(chatParamsRenderCount).toBe(initialChatParamsCount);

    // 验证缩放值确实更新了
    expect(getByTestId('global-scale').textContent).toBe('1.5');
  });

  it('对话参数变更不触发字体设置消费者重渲染', () => {
    let fontRenderCount = 0;
    let chatParamsRenderCount = 0;

    const { getByTestId } = render(
      <GlobalSettingsProvider>
        <FontConsumer onRender={(count) => { fontRenderCount = count; }} />
        <ChatParamsConsumer onRender={(count) => { chatParamsRenderCount = count; }} />
        <SettingsTrigger />
      </GlobalSettingsProvider>
    );

    const initialFontCount = fontRenderCount;

    // 修改温度参数
    act(() => {
      getByTestId('change-temperature').click();
    });

    // 对话参数消费者应该重渲染
    expect(chatParamsRenderCount).toBeGreaterThan(1);
    // 字体消费者不应重渲染
    expect(fontRenderCount).toBe(initialFontCount);

    // 验证温度值确实更新了
    expect(getByTestId('temperature').textContent).toBe('0.9');
  });

  it('maxTokens 变更不触发字体设置消费者重渲染', () => {
    let fontRenderCount = 0;
    let chatParamsRenderCount = 0;

    const { getByTestId } = render(
      <GlobalSettingsProvider>
        <FontConsumer onRender={(count) => { fontRenderCount = count; }} />
        <ChatParamsConsumer onRender={(count) => { chatParamsRenderCount = count; }} />
        <SettingsTrigger />
      </GlobalSettingsProvider>
    );

    const initialFontCount = fontRenderCount;

    // 修改 maxTokens
    act(() => {
      getByTestId('change-max-tokens').click();
    });

    // 对话参数消费者应该重渲染
    expect(chatParamsRenderCount).toBeGreaterThan(1);
    // 字体消费者不应重渲染
    expect(fontRenderCount).toBe(initialFontCount);

    // 验证 maxTokens 值确实更新了
    expect(getByTestId('max-tokens').textContent).toBe('4096');
  });

  it('useGlobalSettings 消费者在字体设置变更时重渲染', () => {
    let globalRenderCount = 0;

    const { getByTestId } = render(
      <GlobalSettingsProvider>
        <GlobalConsumer onRender={(count) => { globalRenderCount = count; }} />
        <SettingsTrigger />
      </GlobalSettingsProvider>
    );

    const initialGlobalCount = globalRenderCount;

    // 修改字体设置
    act(() => {
      getByTestId('change-font').click();
    });

    // 全局消费者应该重渲染（它订阅了所有值）
    expect(globalRenderCount).toBeGreaterThan(initialGlobalCount);
    expect(getByTestId('global-font-family').textContent).toBe('roboto');
  });

  it('useGlobalSettings 消费者在对话参数变更时重渲染', () => {
    let globalRenderCount = 0;

    const { getByTestId } = render(
      <GlobalSettingsProvider>
        <GlobalConsumer onRender={(count) => { globalRenderCount = count; }} />
        <SettingsTrigger />
      </GlobalSettingsProvider>
    );

    const initialGlobalCount = globalRenderCount;

    // 修改温度参数
    act(() => {
      getByTestId('change-temperature').click();
    });

    // 全局消费者应该重渲染
    expect(globalRenderCount).toBeGreaterThan(initialGlobalCount);
    expect(getByTestId('global-temperature').textContent).toBe('0.9');
  });

  it('连续多次不同域的变更保持隔离性', () => {
    let fontRenderCount = 0;
    let chatParamsRenderCount = 0;

    const { getByTestId } = render(
      <GlobalSettingsProvider>
        <FontConsumer onRender={(count) => { fontRenderCount = count; }} />
        <ChatParamsConsumer onRender={(count) => { chatParamsRenderCount = count; }} />
        <SettingsTrigger />
      </GlobalSettingsProvider>
    );

    const initialFontCount = fontRenderCount;
    const initialChatParamsCount = chatParamsRenderCount;

    // 先修改字体
    act(() => {
      getByTestId('change-font').click();
    });

    const afterFontChangeFontCount = fontRenderCount;
    const afterFontChangeChatParamsCount = chatParamsRenderCount;

    // 字体消费者渲染次数增加，对话参数消费者不变
    expect(afterFontChangeFontCount).toBeGreaterThan(initialFontCount);
    expect(afterFontChangeChatParamsCount).toBe(initialChatParamsCount);

    // 再修改对话参数
    act(() => {
      getByTestId('change-temperature').click();
    });

    // 对话参数消费者渲染次数增加，字体消费者不变
    expect(chatParamsRenderCount).toBeGreaterThan(afterFontChangeChatParamsCount);
    expect(fontRenderCount).toBe(afterFontChangeFontCount);
  });
});
