import { useState, useCallback, useEffect } from 'react';
import {
  SCREENSHOT_ACTIONS,
  clampSelectionToPage,
  captureArea,
} from '../utils/screenshotUtils';

/**
 * 截图相关状态管理 Hook
 * 管理截图列表、区域选择状态，以及截图的增删和快捷操作逻辑
 *
 * 从 ChatPDF 主组件中提取截图功能域的状态，
 * 使截图状态变更不会触发其他功能域（消息、PDF、设置等）的重渲染。
 *
 * @param {Object} options - 配置项
 * @param {React.RefObject} options.pdfContainerRef - PDF 页面容器的 ref 引用
 * @param {React.RefObject} options.textareaRef - 输入框的 ref 引用（用于截图后聚焦）
 * @param {boolean} options.isVisionCapable - 当前模型是否支持视觉能力
 * @param {(value: string) => void} options.setInputValue - 设置输入框内容的函数
 * @param {() => void} options.sendMessage - 发送消息的函数
 */
export function useScreenshotState({
  pdfContainerRef,
  textareaRef,
  isVisionCapable,
  setInputValue,
  sendMessage,
} = {}) {
  // ========== 截图状态 ==========
  const [screenshots, setScreenshots] = useState([]);
  const [isSelectingArea, setIsSelectingArea] = useState(false);

  // ========== 当模型不支持视觉时清空截图 ==========
  useEffect(() => {
    if (screenshots.length > 0 && !isVisionCapable) {
      setScreenshots([]);
    }
  }, [isVisionCapable, screenshots.length]);

  // ========== 区域选择完成回调 ==========

  /**
   * 处理区域选择完成事件
   * 将选区裁剪到页面范围内，生成截图并添加到列表（最多 9 张）
   */
  const handleAreaSelected = useCallback(async (rect) => {
    const container = pdfContainerRef?.current;
    if (!container) {
      setIsSelectingArea(false);
      return;
    }
    const cr = container.getBoundingClientRect();
    const clamped = clampSelectionToPage(rect, cr.width, cr.height);
    try {
      const res = await captureArea(pdfContainerRef, clamped);
      if (res) {
        setScreenshots(prev => {
          if (prev.length >= 9) {
            alert('最多只能截图 9 张');
            return prev;
          }
          return [
            ...prev,
            {
              id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
              dataUrl: res,
            },
          ];
        });
      } else {
        alert('截图生成失败');
      }
    } catch (e) {
      alert('截图生成失败');
    } finally {
      setIsSelectingArea(false);
    }
  }, [pdfContainerRef]);

  // ========== 取消区域选择 ==========

  /**
   * 取消当前区域选择操作
   */
  const handleSelectionCancel = useCallback(() => {
    setIsSelectingArea(false);
  }, []);

  // ========== 截图快捷操作 ==========

  /**
   * 执行截图快捷操作（提问、解释、表格提取、公式识别、OCR、翻译、复制）
   * @param {string} key - 操作键名（对应 SCREENSHOT_ACTIONS）
   * @param {string|null} sid - 指定截图 ID，为 null 时操作最后一张截图
   */
  const handleScreenshotAction = useCallback(async (key, sid = null) => {
    const action = SCREENSHOT_ACTIONS[key];
    if (!action) return;

    const target = sid
      ? screenshots.find(s => s.id === sid)
      : screenshots[screenshots.length - 1];
    if (!target) return;

    // 复制截图到剪贴板
    if (key === 'copy') {
      try {
        const res = await fetch(target.dataUrl);
        const blob = await res.blob();
        await navigator.clipboard.write([
          new ClipboardItem({ 'image/png': blob }),
        ]);
      } catch (e) {
        alert('复制失败');
      }
      return;
    }

    // 提问：聚焦输入框
    if (key === 'ask') {
      setTimeout(() => textareaRef?.current?.focus(), 100);
      return;
    }

    // 自动发送预设提示词
    if (action.autoSend && action.prompt) {
      setInputValue?.(action.prompt);
      requestAnimationFrame(() => sendMessage?.());
    }
  }, [screenshots, textareaRef, setInputValue, sendMessage]);

  // ========== 关闭/删除截图 ==========

  /**
   * 关闭截图：传入 id 删除单张，不传则清空全部
   * @param {string|null} id - 要删除的截图 ID，为 null 时清空全部
   */
  const handleScreenshotClose = useCallback((id = null) => {
    if (id) {
      setScreenshots(prev => prev.filter(s => s.id !== id));
    } else {
      setScreenshots([]);
    }
  }, []);

  // ========== 清空截图（用于新建对话等场景） ==========

  /**
   * 清空所有截图
   */
  const clearScreenshots = useCallback(() => {
    setScreenshots([]);
  }, []);

  return {
    // 状态
    screenshots,
    setScreenshots,
    isSelectingArea,
    setIsSelectingArea,

    // 操作方法
    handleAreaSelected,
    handleSelectionCancel,
    handleScreenshotAction,
    handleScreenshotClose,
    clearScreenshots,
  };
}
