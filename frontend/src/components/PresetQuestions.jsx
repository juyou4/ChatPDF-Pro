import React, { useState, useEffect } from 'react';

/**
 * 预设问题栏组件
 *
 * 从后端 /api/presets 获取预设问题列表，渲染为可点击的按钮行。
 * 用户点击按钮时，调用 onSelect(question.query) 将问题文本填入输入框并发送。
 *
 * @param {Object} props
 * @param {function} props.onSelect - 点击预设问题时的回调，接收 query 字符串
 * @param {boolean} props.disabled - 是否禁用所有按钮（如消息发送中）
 */

// API 基础 URL — 空字符串，由 Vite 代理转发到后端
const API_BASE_URL = '';

function PresetQuestions({ onSelect, disabled = false }) {
  // 预设问题列表
  const [presets, setPresets] = useState([]);
  // 加载状态
  const [loading, setLoading] = useState(true);
  // 错误状态
  const [error, setError] = useState(null);

  // 组件挂载时从后端获取预设问题列表
  useEffect(() => {
    let cancelled = false;

    const fetchPresets = async () => {
      try {
        setLoading(true);
        setError(null);

        const response = await fetch(`${API_BASE_URL}/api/presets`);
        if (!response.ok) {
          throw new Error(`获取预设问题失败: ${response.status}`);
        }

        const data = await response.json();
        if (!cancelled) {
          setPresets(data.presets || []);
        }
      } catch (err) {
        console.error('获取预设问题失败:', err);
        if (!cancelled) {
          setError(err.message);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    fetchPresets();

    // 清理函数，防止组件卸载后更新状态
    return () => {
      cancelled = true;
    };
  }, []);

  // 处理按钮点击
  const handleClick = (query) => {
    if (!disabled && onSelect) {
      onSelect(query);
    }
  };

  // 加载中状态：显示占位骨架
  if (loading) {
    return (
      <div className="flex flex-wrap gap-2 px-3 py-2">
        {[1, 2, 3, 4, 5].map((i) => (
          <div
            key={i}
            className="h-8 w-20 rounded-full bg-gray-200 dark:bg-gray-700 animate-pulse"
          />
        ))}
      </div>
    );
  }

  // 错误状态：静默处理，不显示错误信息以免干扰用户
  if (error || presets.length === 0) {
    return null;
  }

  // 正常渲染预设问题按钮
  return (
    <div className="flex flex-wrap gap-2 px-3 py-2">
      {presets.map((preset) => (
        <button
          key={preset.id}
          onClick={() => handleClick(preset.query)}
          disabled={disabled}
          className={`
            inline-flex items-center px-3 py-1.5 rounded-full text-sm
            border border-gray-300 dark:border-gray-600
            transition-all duration-200 ease-in-out
            ${disabled
              ? 'opacity-50 cursor-not-allowed bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-500'
              : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-purple-50 dark:hover:bg-purple-900/30 hover:border-purple-400 dark:hover:border-purple-500 hover:text-purple-600 dark:hover:text-purple-400 cursor-pointer active:scale-95'
            }
          `}
          title={preset.query}
        >
          {preset.label}
        </button>
      ))}
    </div>
  );
}

export default PresetQuestions;
