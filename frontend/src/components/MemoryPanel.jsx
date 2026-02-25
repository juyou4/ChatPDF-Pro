import React, { useState, useEffect, useCallback } from 'react';
import { X, Brain, Trash2, Edit3, Save, Clock, Tag, AlertTriangle, Loader2, ChevronDown, ChevronUp } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// API 基础路径（使用 Vite 代理）
const API_BASE_URL = '';

// 来源类型标签映射
const SOURCE_TYPE_LABELS = {
  auto_qa: '自动摘要',
  manual: '手动记忆',
  liked: '点赞记忆',
  keyword: '关键词',
};

// 来源类型对应的颜色样式
const SOURCE_TYPE_COLORS = {
  auto_qa: 'bg-purple-100 text-purple-700',
  manual: 'bg-green-100 text-green-700',
  liked: 'bg-pink-100 text-pink-700',
  keyword: 'bg-purple-100 text-purple-700',
};

// 截取内容摘要（最多 50 字符）
const truncateContent = (content, maxLen = 50) => {
  if (!content) return '';
  return content.length > maxLen ? content.slice(0, maxLen) + '...' : content;
};

// 格式化时间显示
const formatTime = (isoStr) => {
  if (!isoStr) return '';
  try {
    const date = new Date(isoStr);
    return date.toLocaleString('zh-CN', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return isoStr;
  }
};

const MemoryPanel = ({ isOpen, onClose }) => {
  // 记忆条目列表
  const [entries, setEntries] = useState([]);
  // 加载状态
  const [loading, setLoading] = useState(false);
  // 当前展开的条目 ID
  const [expandedId, setExpandedId] = useState(null);
  // 当前编辑的条目 ID
  const [editingId, setEditingId] = useState(null);
  // 编辑内容
  const [editContent, setEditContent] = useState('');
  // 清空确认对话框
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  // 操作中的条目 ID（用于显示 loading）
  const [operatingId, setOperatingId] = useState(null);

  // 获取记忆列表
  const fetchEntries = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/memory/profile`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      // entries 来自用户画像
      setEntries(data.entries || []);
    } catch (err) {
      console.error('获取记忆列表失败:', err);
      setEntries([]);
    } finally {
      setLoading(false);
    }
  }, []);

  // 面板打开时加载数据
  useEffect(() => {
    if (isOpen) {
      fetchEntries();
      // 重置状态
      setExpandedId(null);
      setEditingId(null);
      setShowClearConfirm(false);
    }
  }, [isOpen, fetchEntries]);

  // 编辑记忆条目
  const handleEdit = (entry) => {
    setEditingId(entry.id);
    setEditContent(entry.content);
    setExpandedId(entry.id);
  };

  // 保存编辑
  const handleSave = async (entryId) => {
    setOperatingId(entryId);
    try {
      const res = await fetch(`${API_BASE_URL}/api/memory/entries/${entryId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: editContent }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      // 更新本地状态
      setEntries((prev) =>
        prev.map((e) => (e.id === entryId ? { ...e, content: editContent } : e))
      );
      setEditingId(null);
    } catch (err) {
      console.error('编辑记忆失败:', err);
    } finally {
      setOperatingId(null);
    }
  };

  // 删除单条记忆
  const handleDelete = async (entryId) => {
    if (!confirm('确定要删除这条记忆吗？')) return;
    setOperatingId(entryId);
    try {
      const res = await fetch(`${API_BASE_URL}/api/memory/entries/${entryId}`, {
        method: 'DELETE',
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setEntries((prev) => prev.filter((e) => e.id !== entryId));
      if (expandedId === entryId) setExpandedId(null);
      if (editingId === entryId) setEditingId(null);
    } catch (err) {
      console.error('删除记忆失败:', err);
    } finally {
      setOperatingId(null);
    }
  };

  // 清空所有记忆
  const handleClearAll = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/memory/all`, {
        method: 'DELETE',
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setEntries([]);
      setExpandedId(null);
      setEditingId(null);
    } catch (err) {
      console.error('清空记忆失败:', err);
    } finally {
      setShowClearConfirm(false);
    }
  };

  // 切换展开/收起
  const toggleExpand = (entryId) => {
    if (editingId === entryId) return; // 编辑中不允许收起
    setExpandedId((prev) => (prev === entryId ? null : entryId));
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          transition={{ type: 'spring', damping: 20 }}
          className="soft-panel rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-auto"
          onClick={(e) => e.stopPropagation()}
        >
          {/* 头部 */}
          <div className="sticky top-0 bg-white/90 backdrop-blur-md border-b border-gray-100 p-6 flex items-center justify-between z-10">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-600 rounded-xl flex items-center justify-center">
                <Brain className="w-5 h-5 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold">记忆管理</h2>
                <p className="text-sm text-gray-500">
                  共 {entries.length} 条记忆
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-black/5 rounded-full transition-colors"
            >
              <X className="w-6 h-6" />
            </button>
          </div>

          <div className="p-6 space-y-4">
            {/* 加载状态 */}
            {loading && (
              <div className="flex items-center justify-center py-12 text-gray-400">
                <Loader2 className="w-6 h-6 animate-spin mr-2" />
                <span>加载中...</span>
              </div>
            )}

            {/* 空状态 */}
            {!loading && entries.length === 0 && (
              <div className="text-center py-12 text-gray-400">
                <Brain className="w-12 h-12 mx-auto mb-3 opacity-30" />
                <p>暂无记忆条目</p>
                <p className="text-sm mt-1">对话中点击"记住这个"或点赞即可保存记忆</p>
              </div>
            )}

            {/* 记忆列表 */}
            {!loading && entries.length > 0 && (
              <div className="space-y-2">
                {entries.map((entry) => {
                  const isExpanded = expandedId === entry.id;
                  const isEditing = editingId === entry.id;
                  const isOperating = operatingId === entry.id;

                  return (
                    <div
                      key={entry.id}
                      className="soft-card rounded-xl p-4 transition-all hover:shadow-md"
                    >
                      {/* 条目头部：摘要 + 标签 + 时间 */}
                      <div
                        className="flex items-start gap-3 cursor-pointer"
                        onClick={() => toggleExpand(entry.id)}
                      >
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1 flex-wrap">
                            {/* 来源类型标签 */}
                            <span
                              className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${SOURCE_TYPE_COLORS[entry.source_type] || 'bg-gray-100 text-gray-600'}`}
                            >
                              <Tag className="w-3 h-3" />
                              {SOURCE_TYPE_LABELS[entry.source_type] || entry.source_type}
                            </span>
                            {/* 创建时间 */}
                            <span className="inline-flex items-center gap-1 text-xs text-gray-400">
                              <Clock className="w-3 h-3" />
                              {formatTime(entry.created_at)}
                            </span>
                          </div>
                          {/* 内容摘要 */}
                          {!isExpanded && (
                            <p className="text-sm text-gray-700 truncate">
                              {truncateContent(entry.content)}
                            </p>
                          )}
                        </div>
                        {/* 展开/收起图标 */}
                        <div className="text-gray-400 mt-1">
                          {isExpanded ? (
                            <ChevronUp className="w-4 h-4" />
                          ) : (
                            <ChevronDown className="w-4 h-4" />
                          )}
                        </div>
                      </div>

                      {/* 展开区域：完整内容 + 操作按钮 */}
                      {isExpanded && (
                        <div className="mt-3 pt-3 border-t border-gray-100">
                          {isEditing ? (
                            /* 编辑模式 */
                            <div className="space-y-3">
                              <textarea
                                value={editContent}
                                onChange={(e) => setEditContent(e.target.value)}
                                className="w-full px-3 py-2 soft-input rounded-lg outline-none text-sm resize-none min-h-[100px]"
                                autoFocus
                              />
                              <div className="flex gap-2 justify-end">
                                <button
                                  onClick={() => setEditingId(null)}
                                  className="px-3 py-1.5 text-sm text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                                >
                                  取消
                                </button>
                                <button
                                  onClick={() => handleSave(entry.id)}
                                  disabled={isOperating}
                                  className="inline-flex items-center gap-1 px-3 py-1.5 text-sm bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50"
                                >
                                  {isOperating ? (
                                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                  ) : (
                                    <Save className="w-3.5 h-3.5" />
                                  )}
                                  保存
                                </button>
                              </div>
                            </div>
                          ) : (
                            /* 查看模式 */
                            <div>
                              <p className="text-sm text-gray-700 whitespace-pre-wrap break-words">
                                {entry.content}
                              </p>
                              <div className="flex gap-2 justify-end mt-3">
                                <button
                                  onClick={() => handleEdit(entry)}
                                  disabled={isOperating}
                                  className="inline-flex items-center gap-1 px-3 py-1.5 text-sm text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                                >
                                  <Edit3 className="w-3.5 h-3.5" />
                                  编辑
                                </button>
                                <button
                                  onClick={() => handleDelete(entry.id)}
                                  disabled={isOperating}
                                  className="inline-flex items-center gap-1 px-3 py-1.5 text-sm text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                                >
                                  {isOperating ? (
                                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                  ) : (
                                    <Trash2 className="w-3.5 h-3.5" />
                                  )}
                                  删除
                                </button>
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}

            {/* 清空所有记忆按钮 */}
            {!loading && entries.length > 0 && (
              <div className="pt-4 border-t border-gray-200">
                <button
                  onClick={() => setShowClearConfirm(true)}
                  className="w-full soft-card flex items-center justify-center gap-2 px-4 py-3 text-red-600 hover:bg-red-50/50 rounded-xl transition-all"
                >
                  <Trash2 className="w-4 h-4" />
                  <span className="font-medium">清空所有记忆</span>
                </button>
              </div>
            )}
          </div>
        </motion.div>

        {/* 清空确认对话框 */}
        {showClearConfirm && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-[60] flex items-center justify-center p-4"
            onClick={() => setShowClearConfirm(false)}
          >
            <div
              className="soft-panel rounded-2xl p-6 max-w-sm w-full"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-red-100 rounded-full flex items-center justify-center">
                  <AlertTriangle className="w-5 h-5 text-red-600" />
                </div>
                <h3 className="text-lg font-bold">确认清空</h3>
              </div>
              <p className="text-sm text-gray-600 mb-6">
                此操作将删除所有记忆条目，且无法恢复。确定要继续吗？
              </p>
              <div className="flex gap-3">
                <button
                  onClick={handleClearAll}
                  className="flex-1 py-3 bg-red-600 hover:bg-red-700 text-white rounded-xl transition-colors font-medium"
                >
                  确认清空
                </button>
                <button
                  onClick={() => setShowClearConfirm(false)}
                  className="flex-1 py-3 bg-gray-200 hover:bg-gray-300 rounded-xl transition-colors font-medium"
                >
                  取消
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </motion.div>
    </AnimatePresence>
  );
};

export default MemoryPanel;
