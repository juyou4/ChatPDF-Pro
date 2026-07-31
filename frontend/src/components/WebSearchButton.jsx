import React from 'react';
import { Globe } from 'lucide-react';
import { useWebSearch } from '../contexts/WebSearchContext';

/**
 * 联网搜索按钮组件
 * 
 * 单击切换联网搜索开关；从关闭打开时进入强制搜索，避免用户点亮按钮后
 * 被自动策略静默跳过。自动策略仍可在设置中心单独选择。
 * 搜索引擎设置已移至"全局设置"面板
 */
const WebSearchButton = () => {
    const { enableWebSearch, webSearchMode, toggleWebSearch, getCurrentProvider } = useWebSearch();

    const currentProvider = getCurrentProvider();

    return (
        <button
            onClick={toggleWebSearch}
            className={`transition-colors flex items-center justify-center shrink-0 ${
                enableWebSearch
                    ? 'text-purple-600 dark:text-[#FFA07A]'
                    : 'text-gray-500 hover:text-gray-800 dark:text-gray-400 dark:hover:text-gray-200'
            }`}
            title={enableWebSearch
                ? `联网搜索已开启（${webSearchMode === 'force' ? '本轮必搜' : '自动判断'}，${currentProvider.name}）`
                : '联网搜索'
            }
            aria-label={enableWebSearch
                ? `联网搜索已开启，${webSearchMode === 'force' ? '本轮必搜' : '自动判断'}`
                : '开启联网搜索'
            }
            aria-pressed={enableWebSearch}
        >
            <Globe size={15} />
        </button>
    );
};

export default WebSearchButton;
