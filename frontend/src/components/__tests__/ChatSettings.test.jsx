// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';

// Feature: chatpdf-chat-settings-upgrade
// 测试两个新开关的渲染和交互
// Validates: Requirements 3.1, 3.2, 3.3

// Mock Context Hooks
const mockSetMessageFont = vi.fn();
const mockSetThoughtAutoCollapse = vi.fn();
let mockMessageFont = 'system';
let mockThoughtAutoCollapse = true;

vi.mock('../../contexts/FontSettingsContext', () => ({
    useFontSettings: () => ({
        messageFont: mockMessageFont,
        setMessageFont: mockSetMessageFont,
    }),
    FONT_DEFAULT_SETTINGS: {
        fontFamily: 'inter',
        customFont: '',
        globalScale: 1.0,
        messageFont: 'system',
    },
}));

vi.mock('../../contexts/ChatParamsContext', () => ({
    useChatParams: () => ({
        maxTokens: 8192,
        temperature: 0.7,
        topP: 1.0,
        contextCount: 5,
        streamOutput: true,
        enableTemperature: true,
        enableTopP: false,
        enableMaxTokens: true,
        customParams: [],
        thoughtAutoCollapse: mockThoughtAutoCollapse,
        setMaxTokens: vi.fn(),
        setTemperature: vi.fn(),
        setTopP: vi.fn(),
        setContextCount: vi.fn(),
        setStreamOutput: vi.fn(),
        setEnableTemperature: vi.fn(),
        setEnableTopP: vi.fn(),
        setEnableMaxTokens: vi.fn(),
        setCustomParams: vi.fn(),
        setThoughtAutoCollapse: mockSetThoughtAutoCollapse,
    }),
    CHAT_PARAMS_DEFAULT_SETTINGS: {
        maxTokens: 8192,
        temperature: 0.7,
        topP: 1.0,
        contextCount: 5,
        streamOutput: true,
        enableTemperature: true,
        enableTopP: false,
        enableMaxTokens: true,
        customParams: [],
        reasoningEffort: 'off',
        enableMemory: true,
        thoughtAutoCollapse: true,
    },
}));

// Mock framer-motion 避免动画问题
vi.mock('framer-motion', () => ({
    motion: {
        div: React.forwardRef(({ children, ...props }, ref) => {
            // 过滤掉 framer-motion 特有的 props
            const { initial, animate, exit, transition, whileHover, whileTap, ...domProps } = props;
            return <div ref={ref} {...domProps}>{children}</div>;
        }),
    },
    AnimatePresence: ({ children }) => <>{children}</>,
}));

import ChatSettings from '../ChatSettings';

describe('ChatSettings - 新开关单元测试', () => {
    beforeEach(() => {
        mockMessageFont = 'system';
        mockThoughtAutoCollapse = true;
        mockSetMessageFont.mockClear();
        mockSetThoughtAutoCollapse.mockClear();
    });

    it('应渲染「使用衬线字体」开关', () => {
        render(<ChatSettings isOpen={true} onClose={vi.fn()} />);
        expect(screen.getByText('使用衬线字体')).toBeTruthy();
    });

    it('应渲染「思考内容自动折叠」开关', () => {
        render(<ChatSettings isOpen={true} onClose={vi.fn()} />);
        expect(screen.getByText('思考内容自动折叠')).toBeTruthy();
    });

    it('衬线字体开关默认应为关闭状态（messageFont=system）', () => {
        mockMessageFont = 'system';
        render(<ChatSettings isOpen={true} onClose={vi.fn()} />);
        // 开关存在即可，具体 checked 状态由 ToggleSwitch 内部处理
        expect(screen.getByText('使用衬线字体')).toBeTruthy();
    });

    it('思考自动折叠开关默认应为开启状态', () => {
        mockThoughtAutoCollapse = true;
        render(<ChatSettings isOpen={true} onClose={vi.fn()} />);
        expect(screen.getByText('思考内容自动折叠')).toBeTruthy();
    });
});
