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
            className={`transition-colors flex items-center justify-center shrink-0 ${
                enableWebSearch
                    ? 'text-purple-600'
                    : 'text-gray-500 hover:text-gray-800'
            }`}
            title={enableWebSearch
                ? `联网搜索已开启 (${currentProvider.name})`
                : '联网搜索'
            }
        >
            <Globe size={15} />
        </button>
    );
};

export default WebSearchButton;
