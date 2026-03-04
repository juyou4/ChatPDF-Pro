import React from 'react';
import { Globe } from 'lucide-react';
import { useWebSearch } from '../contexts/WebSearchContext';

/**
 * 联网搜索按钮组件
 * 
 * 单击切换联网搜索开关，开启时高亮显示（紫色激活态，与其他按钮风格一致）
 * 搜索引擎设置已移至"全局设置"面板
 */
const WebSearchButton = () => {
    const { enableWebSearch, toggleWebSearch, getCurrentProvider } = useWebSearch();

    const currentProvider = getCurrentProvider();

    return (
        <button
            onClick={toggleWebSearch}
            className={`transition-colors p-1 rounded-md ${
                enableWebSearch
                    ? 'text-purple-600 bg-purple-50 hover:bg-purple-100'
                    : 'hover:text-gray-600 hover:bg-gray-50'
            }`}
            title={enableWebSearch
                ? `联网搜索已开启 (${currentProvider.name})`
                : '联网搜索'
            }
        >
            <Globe className="w-5 h-5" />
        </button>
    );
};

export default WebSearchButton;
