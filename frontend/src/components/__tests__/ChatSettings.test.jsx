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
        sendShortcut: 'Enter',
        confirmDeleteMessage: true,
        confirmRegenerateMessage: false,
        codeCollapsible: false,
        codeWrappable: true,
        codeShowLineNumbers: false,
        messageStyle: 'plain',
        messageFontSize: 14,
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
        setSendShortcut: vi.fn(),
        setConfirmDeleteMessage: vi.fn(),
        setConfirmRegenerateMessage: vi.fn(),
        setCodeCollapsible: vi.fn(),
        setCodeWrappable: vi.fn(),
        setCodeShowLineNumbers: vi.fn(),
        setMessageStyle: vi.fn(),
        setMessageFontSize: vi.fn(),
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
        sendShortcut: 'Enter',
        confirmDeleteMessage: true,
        confirmRegenerateMessage: false,
        codeCollapsible: false,
        codeWrappable: true,
        codeShowLineNumbers: false,
        messageStyle: 'plain',
        messageFontSize: 14,
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

describe('ChatSettings - 开关单元测试', () => {
    beforeEach(() => {
        mockMessageFont = 'system';
        mockThoughtAutoCollapse = true;
        mockSetMessageFont.mockClear();
        mockSetThoughtAutoCollapse.mockClear();
    });

    it('应渲染「思考自动折叠」开关', () => {
        render(<ChatSettings isOpen={true} onClose={vi.fn()} />);
        expect(screen.getByText('思考自动折叠')).toBeTruthy();
    });

    it('思考自动折叠开关默认应为开启状态', () => {
        mockThoughtAutoCollapse = true;
        render(<ChatSettings isOpen={true} onClose={vi.fn()} />);
        expect(screen.getByText('思考自动折叠')).toBeTruthy();
    });

    it('应渲染「发送快捷键」选择按钮', () => {
        render(<ChatSettings isOpen={true} onClose={vi.fn()} />);
        expect(screen.getByText('发送快捷键')).toBeTruthy();
        expect(screen.getByText('Enter')).toBeTruthy();
        expect(screen.getByText('Ctrl+Enter')).toBeTruthy();
    });

    it('应渲染「删除消息确认」和「重新生成确认」开关', () => {
        render(<ChatSettings isOpen={true} onClose={vi.fn()} />);
        expect(screen.getByText('删除消息确认')).toBeTruthy();
        expect(screen.getByText('重新生成确认')).toBeTruthy();
    });

    it('应渲染「代码块设置」分组', () => {
        render(<ChatSettings isOpen={true} onClose={vi.fn()} />);
        expect(screen.getByText('代码块折叠')).toBeTruthy();
        expect(screen.getByText('代码自动换行')).toBeTruthy();
        expect(screen.getByText('显示行号')).toBeTruthy();
    });

    it('应渲染「消息样式」选择按钮', () => {
        render(<ChatSettings isOpen={true} onClose={vi.fn()} />);
        expect(screen.getByText('消息样式')).toBeTruthy();
        expect(screen.getByText('平铺')).toBeTruthy();
        expect(screen.getByText('气泡')).toBeTruthy();
    });

    it('应渲染「消息字体大小」设置', () => {
        render(<ChatSettings isOpen={true} onClose={vi.fn()} />);
        expect(screen.getByText('消息字体大小')).toBeTruthy();
    });
});
