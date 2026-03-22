import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  BookOpen, AlertCircle, Bot, X, Trash2, Edit2, Check, RefreshCw, MessageSquare, Tag, AlignLeft, Brain, Cpu, Database, Hash, GitCommit, GitPullRequest, Search, FileText, Globe, Clock, Copy, Plus, Activity, AlertTriangle, Layers, Type, Sparkles, Image as ImageIcon, MessageCircle, ExternalLink, Download, FileUp, FolderOpen, Box, Hash as HashIcon, Archive, Folder, RotateCcw, ChevronDown, ChevronUp, GitBranch, Network, Loader2, Save, Edit3
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE_URL = '';

const TAB_CONFIGS = [
  { id: 'profile', label: '全局画像', kind: 'profile', icon: Brain },
  { id: 'doc_fact', label: '文档事实', kind: 'doc_fact', icon: Database },
  { id: 'consolidated', label: '压缩事实', kind: 'consolidated', icon: GitBranch },
  { id: 'graph', label: '图谱摘要', kind: 'graph', icon: Network },
];

const SOURCE_TYPE_LABELS = {
  auto_qa: '自动摘要',
  manual: '手动记忆',
  liked: '点赞记忆',
  keyword: '关键词',
  llm_distilled: '提炼事实',
  compressed: '压缩记忆',
};

const MEMORY_KIND_LABELS = {
  working: '工作记忆',
  profile: '画像',
  doc_fact: '文档事实',
  episodic: '对话摘要',
  consolidated: '压缩事实',
  graph: '图谱',
};

const STATUS_LABELS = {
  active: '生效中',
  archived_raw: '原始归档',
};

const truncateContent = (content, maxLen = 64) => {
  if (!content) return '';
  return content.length > maxLen ? `${content.slice(0, maxLen)}...` : content;
};

const formatTime = (isoStr) => {
  if (!isoStr) return '';
  try {
    return new Date(isoStr).toLocaleString('zh-CN', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return isoStr;
  }
};

const KindBadge = ({ label, tone = 'purple' }) => {
  const palette = {
    purple: 'bg-purple-100 text-purple-700',
    emerald: 'bg-emerald-100 text-emerald-700',
    amber: 'bg-amber-100 text-amber-700',
    slate: 'bg-slate-100 text-slate-700',
    rose: 'bg-rose-100 text-rose-700',
  };
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium ${palette[tone] || palette.slate}`}>
      {label}
    </span>
  );
};

const MemoryPanel = ({ isOpen, onClose }) => {
  const [allEntries, setAllEntries] = useState([]);
  const [focusAreas, setFocusAreas] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('profile');
  const [selectedDocId, setSelectedDocId] = useState('');
  const [expandedId, setExpandedId] = useState(null);
  const [editingId, setEditingId] = useState(null);
  const [editContent, setEditContent] = useState('');
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [operatingId, setOperatingId] = useState(null);
  const [traceById, setTraceById] = useState({});
  const [traceLoadingId, setTraceLoadingId] = useState(null);
  const [graphData, setGraphData] = useState(null);
  const [graphLoading, setGraphLoading] = useState(false);
  const [statusData, setStatusData] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');
  const [rebuilding, setRebuilding] = useState(false);

  const fetchAllData = useCallback(async () => {
    setLoading(true);
    try {
      const [profileRes, entriesRes, statusRes] = await Promise.all([
        fetch(`${API_BASE_URL}/api/memory/profile`),
        fetch(`${API_BASE_URL}/api/memory/entries`),
        fetch(`${API_BASE_URL}/api/memory/status`),
      ]);
      if (!profileRes.ok || !entriesRes.ok || !statusRes.ok) {
        throw new Error(`HTTP ${profileRes.status}/${entriesRes.status}/${statusRes.status}`);
      }
      const profileData = await profileRes.json();
      const entriesData = await entriesRes.json();
      const status = await statusRes.json();
      const nextEntries = entriesData.entries || [];
      setFocusAreas(profileData.focus_areas || []);
      setAllEntries(nextEntries);
      setStatusData(status);

      const docIds = [...new Set(nextEntries.map((entry) => entry.doc_id).filter(Boolean))];
      setSelectedDocId((prev) => (prev && docIds.includes(prev) ? prev : (docIds[0] || '')));
    } catch (err) {
      console.error('获取记忆数据失败:', err);
      setFocusAreas([]);
      setAllEntries([]);
      setStatusData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const docOptions = useMemo(
    () => [...new Set(allEntries.map((entry) => entry.doc_id).filter(Boolean))],
    [allEntries]
  );

  const filteredEntries = useMemo(() => {
    const matchesTab = (entry) => {
      if (activeTab === 'profile') return entry.memory_scope === 'profile';
      if (activeTab === 'doc_fact') return entry.memory_kind === 'doc_fact';
      if (activeTab === 'consolidated') return entry.memory_kind === 'consolidated';
      return false;
    };

    return allEntries.filter((entry) => {
      if (!matchesTab(entry)) return false;
      if (!selectedDocId || activeTab === 'profile') return true;
      return entry.doc_id === selectedDocId;
    });
  }, [activeTab, allEntries, selectedDocId]);

  const tabCounts = useMemo(() => ({
    profile: allEntries.filter((entry) => entry.memory_scope === 'profile').length,
    doc_fact: allEntries.filter((entry) => entry.memory_kind === 'doc_fact').length,
    consolidated: allEntries.filter((entry) => entry.memory_kind === 'consolidated').length,
    graph: docOptions.length,
  }), [allEntries, docOptions.length]);

  const fetchTrace = useCallback(async (entryId) => {
    if (traceById[entryId]) return;
    setTraceLoadingId(entryId);
    try {
      const res = await fetch(`${API_BASE_URL}/api/memory/entries/${entryId}/trace`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setTraceById((prev) => ({ ...prev, [entryId]: data }));
    } catch (err) {
      console.error('获取记忆来源链失败:', err);
    } finally {
      setTraceLoadingId(null);
    }
  }, [traceById]);

  const fetchGraph = useCallback(async (docId) => {
    if (!docId) {
      setGraphData(null);
      return;
    }
    setGraphLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/memory/graph/${docId}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setGraphData(await res.json());
    } catch (err) {
      console.error('获取图谱摘要失败:', err);
      setGraphData(null);
    } finally {
      setGraphLoading(false);
    }
  }, []);

  const handleRebuildFromEvents = useCallback(async () => {
    setRebuilding(true);
    setStatusMessage('');
    try {
      const res = await fetch(`${API_BASE_URL}/api/memory/rebuild-from-events`, {
        method: 'POST',
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setStatusMessage(`已从事件恢复：画像 ${data.profile_entries || 0} 条，文档 ${data.session_count || 0} 个，索引 ${data.indexed_entries || 0} 条。`);
      await fetchAllData();
      if (activeTab === 'graph' && selectedDocId) {
        fetchGraph(selectedDocId);
      }
    } catch (err) {
      console.error('从事件恢复记忆失败:', err);
      setStatusMessage('从事件恢复失败，请检查后端日志。');
    } finally {
      setRebuilding(false);
    }
  }, [activeTab, fetchAllData, fetchGraph, selectedDocId]);

  useEffect(() => {
    if (!isOpen) return;
    fetchAllData();
    setExpandedId(null);
    setEditingId(null);
    setTraceById({});
    setGraphData(null);
    setShowClearConfirm(false);
    setStatusMessage('');
  }, [isOpen, fetchAllData]);

  useEffect(() => {
    if (!isOpen || activeTab !== 'graph') return;
    fetchGraph(selectedDocId);
  }, [isOpen, activeTab, selectedDocId, fetchGraph]);

  const handleEdit = (entry) => {
    setEditingId(entry.id);
    setEditContent(entry.content);
    setExpandedId(entry.id);
  };

  const handleSave = async (entryId) => {
    setOperatingId(entryId);
    try {
      const res = await fetch(`${API_BASE_URL}/api/memory/entries/${entryId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: editContent }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const updated = await res.json();
      setAllEntries((prev) => prev.map((entry) => (
        entry.id === entryId ? { ...entry, ...updated } : entry
      )));
      setTraceById((prev) => {
        if (!prev[entryId]) return prev;
        return {
          ...prev,
          [entryId]: {
            ...prev[entryId],
            entry: { ...prev[entryId].entry, ...updated },
          },
        };
      });
      setEditingId(null);
    } catch (err) {
      console.error('编辑记忆失败:', err);
    } finally {
      setOperatingId(null);
    }
  };

  const handleDelete = async (entryId) => {
    if (!confirm('确定要删除这条记忆吗？')) return;
    setOperatingId(entryId);
    try {
      const res = await fetch(`${API_BASE_URL}/api/memory/entries/${entryId}`, {
        method: 'DELETE',
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setAllEntries((prev) => prev.filter((entry) => entry.id !== entryId));
      setTraceById((prev) => {
        const next = { ...prev };
        delete next[entryId];
        return next;
      });
      if (expandedId === entryId) setExpandedId(null);
      if (editingId === entryId) setEditingId(null);
    } catch (err) {
      console.error('删除记忆失败:', err);
    } finally {
      setOperatingId(null);
    }
  };

  const handleClearAll = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/memory/all`, {
        method: 'DELETE',
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setAllEntries([]);
      setFocusAreas([]);
      setExpandedId(null);
      setEditingId(null);
      setTraceById({});
      setGraphData(null);
      setStatusMessage('');
      await fetchAllData();
    } catch (err) {
      console.error('清空记忆失败:', err);
    } finally {
      setShowClearConfirm(false);
    }
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 backdrop-blur-sm"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.92, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.92, opacity: 0 }}
          transition={{ type: 'spring', damping: 22 }}
          className="soft-panel max-h-[90vh] w-full max-w-4xl overflow-auto rounded-2xl shadow-2xl"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="sticky top-0 z-10 flex items-center justify-between border-b border-gray-100 bg-white/90 p-6 backdrop-blur-md">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-purple-500 to-pink-600">
                <Brain className="h-5 w-5 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold">记忆管理</h2>
                <p className="text-sm text-gray-500">当前共 {allEntries.length} 条记忆</p>
              </div>
            </div>
            <button onClick={onClose} className="rounded-full p-2 transition-colors hover:bg-black/5">
              <X className="h-6 w-6" />
            </button>
          </div>

          <div className="space-y-4 p-6">
            <div className="flex flex-wrap gap-2">
              {TAB_CONFIGS.map((tab) => {
                const Icon = tab.icon;
                const active = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition-colors ${
                      active ? 'bg-purple-600 text-white shadow-sm' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    {tab.label}
                    <span className={`rounded-full px-1.5 py-0.5 text-[10px] ${active ? 'bg-white/15 text-white' : 'bg-white text-gray-500'}`}>
                      {tabCounts[tab.id]}
                    </span>
                  </button>
                );
              })}
            </div>

            <div className="rounded-2xl border border-gray-100 bg-white/80 p-4 shadow-sm">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <div className="text-sm font-semibold text-gray-800">存储状态</div>
                  <div className="mt-1 text-xs text-gray-500">
                    当前为事件快照优先读取，旧 JSON 仅作兼容兜底。
                  </div>
                </div>
                <div className="flex flex-wrap gap-2">
                  <button
                    onClick={fetchAllData}
                    disabled={loading || rebuilding}
                    className="inline-flex items-center gap-2 rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm text-gray-700 transition-colors hover:bg-gray-50 disabled:opacity-50"
                  >
                    {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RotateCcw className="h-4 w-4" />}
                    刷新状态
                  </button>
                  <button
                    onClick={handleRebuildFromEvents}
                    disabled={rebuilding || !statusData?.event_log_files}
                    className="inline-flex items-center gap-2 rounded-lg bg-emerald-600 px-3 py-2 text-sm text-white transition-colors hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {rebuilding ? <Loader2 className="h-4 w-4 animate-spin" /> : <GitBranch className="h-4 w-4" />}
                    从事件恢复
                  </button>
                </div>
              </div>

              <div className="mt-4 grid gap-3 md:grid-cols-4">
                <div className="rounded-xl border border-gray-100 bg-gray-50/70 p-3">
                  <div className="text-xs uppercase tracking-wider text-gray-400">索引同步</div>
                  <div className={`mt-1 text-sm font-semibold ${statusData?.rebuild_required ? 'text-rose-600' : statusData?.dirty ? 'text-amber-600' : 'text-emerald-700'}`}>
                    {statusData?.rebuild_required ? '需要安全重建' : statusData?.dirty ? '待落盘' : '已同步'}
                  </div>
                  <div className="mt-1 text-xs text-gray-500">
                    {statusData?.rebuild_required
                      ? `原因：${statusData?.rebuild_reason || '索引参数变化'}`
                      : statusData?.pending_sync ? '仍有防抖中的写入' : '当前无挂起同步'}
                  </div>
                </div>
                <div className="rounded-xl border border-gray-100 bg-gray-50/70 p-3">
                  <div className="text-xs uppercase tracking-wider text-gray-400">快照文件</div>
                  <div className="mt-1 text-sm font-semibold text-gray-800">
                    {statusData?.profile_snapshot_exists ? '画像已就绪' : '画像未生成'}
                  </div>
                  <div className="mt-1 text-xs text-gray-500">文档快照 {statusData?.session_snapshot_count ?? 0} 个</div>
                </div>
                <div className="rounded-xl border border-gray-100 bg-gray-50/70 p-3">
                  <div className="text-xs uppercase tracking-wider text-gray-400">事件日志</div>
                  <div className="mt-1 text-sm font-semibold text-gray-800">{statusData?.event_log_files ?? 0} 个文件</div>
                  <div className="mt-1 text-xs text-gray-500">{statusData?.last_event_at ? `最后事件 ${formatTime(statusData.last_event_at)}` : '暂无事件日志'}</div>
                </div>
                <div className="rounded-xl border border-gray-100 bg-gray-50/70 p-3">
                  <div className="text-xs uppercase tracking-wider text-gray-400">索引落盘</div>
                  <div className="mt-1 text-sm font-semibold text-gray-800">{statusData?.last_sync_at ? formatTime(statusData.last_sync_at) : '尚未落盘'}</div>
                  <div className="mt-1 text-xs text-gray-500">
                    重建 {statusData?.last_reindex_at ? formatTime(statusData.last_reindex_at) : '暂无'}
                    {statusData?.last_reindex_reason ? ` / ${statusData.last_reindex_reason}` : ''}
                  </div>
                </div>
              </div>

              {statusMessage && (
                <div className="mt-3 rounded-xl bg-emerald-50 px-3 py-2 text-sm text-emerald-700">
                  {statusMessage}
                </div>
              )}
            </div>

            <div className="flex flex-wrap items-center gap-3 rounded-xl border border-gray-100 bg-gray-50/70 p-3">
              <div className="text-sm text-gray-600">
                关注领域：
                {focusAreas.length > 0 ? (
                  <span className="ml-2 inline-flex flex-wrap gap-1.5">
                    {focusAreas.slice(0, 6).map((area) => <KindBadge key={area} label={area} tone="emerald" />)}
                  </span>
                ) : (
                  <span className="ml-2 text-gray-400">暂无</span>
                )}
              </div>
              {activeTab !== 'profile' && docOptions.length > 0 && (
                <div className="ml-auto flex items-center gap-2">
                  <span className="text-sm text-gray-500">文档</span>
                  <select
                    value={selectedDocId}
                    onChange={(e) => setSelectedDocId(e.target.value)}
                    className="rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-sm outline-none"
                  >
                    <option value="">全部文档</option>
                    {docOptions.map((docId) => (
                      <option key={docId} value={docId}>{docId}</option>
                    ))}
                  </select>
                </div>
              )}
            </div>

            {loading && (
              <div className="flex items-center justify-center py-12 text-gray-400">
                <Loader2 className="mr-2 h-6 w-6 animate-spin" />
                <span>加载中...</span>
              </div>
            )}

            {!loading && activeTab === 'graph' && (
              <div className="space-y-4">
                {graphLoading && (
                  <div className="flex items-center justify-center py-10 text-gray-400">
                    <Loader2 className="mr-2 h-6 w-6 animate-spin" />
                    <span>图谱摘要生成中...</span>
                  </div>
                )}
                {!graphLoading && !selectedDocId && (
                  <div className="rounded-xl border border-dashed border-gray-200 p-8 text-center text-sm text-gray-400">
                    暂无可用于图谱摘要的文档记忆
                  </div>
                )}
                {!graphLoading && selectedDocId && graphData && (
                  <>
                    <div className="grid gap-3 md:grid-cols-3">
                      <div className="rounded-xl border border-gray-100 bg-white/70 p-4">
                        <div className="text-xs uppercase tracking-wider text-gray-400">文档</div>
                        <div className="mt-1 text-sm font-semibold text-gray-800">{selectedDocId}</div>
                      </div>
                      <div className="rounded-xl border border-gray-100 bg-white/70 p-4">
                        <div className="text-xs uppercase tracking-wider text-gray-400">节点数</div>
                        <div className="mt-1 text-sm font-semibold text-gray-800">{graphData.node_count}</div>
                      </div>
                      <div className="rounded-xl border border-gray-100 bg-white/70 p-4">
                        <div className="text-xs uppercase tracking-wider text-gray-400">边数</div>
                        <div className="mt-1 text-sm font-semibold text-gray-800">{graphData.edge_count}</div>
                      </div>
                    </div>
                    <div className="rounded-xl border border-gray-100 bg-white/70 p-4">
                      <div className="mb-3 text-sm font-semibold text-gray-800">节点预览</div>
                      {graphData.nodes && graphData.nodes.length > 0 ? (
                        <div className="flex flex-wrap gap-2">
                          {graphData.nodes.map((node) => (
                            <KindBadge key={node.id} label={`${node.type}: ${node.label}`} tone={node.type === 'figure' ? 'amber' : node.type === 'table' ? 'rose' : 'purple'} />
                          ))}
                        </div>
                      ) : (
                        <div className="text-sm text-gray-400">暂无节点</div>
                      )}
                    </div>
                  </>
                )}
              </div>
            )}

            {!loading && activeTab !== 'graph' && filteredEntries.length === 0 && (
              <div className="rounded-xl border border-dashed border-gray-200 py-12 text-center text-gray-400">
                <Brain className="mx-auto mb-3 h-12 w-12 opacity-30" />
                <p>当前视图暂无记忆</p>
              </div>
            )}

            {!loading && activeTab !== 'graph' && filteredEntries.length > 0 && (
              <div className="space-y-3">
                {filteredEntries.map((entry) => {
                  const isExpanded = expandedId === entry.id;
                  const isEditing = editingId === entry.id;
                  const isOperating = operatingId === entry.id;
                  const trace = traceById[entry.id];
                  return (
                    <div key={entry.id} className="rounded-xl border border-gray-100 bg-white/70 p-4 shadow-sm transition-all hover:shadow-md">
                      <div className="flex cursor-pointer items-start gap-3" onClick={() => setExpandedId((prev) => (prev === entry.id ? null : entry.id))}>
                        <div className="min-w-0 flex-1">
                          <div className="mb-1 flex flex-wrap items-center gap-2">
                            <KindBadge label={MEMORY_KIND_LABELS[entry.memory_kind] || entry.memory_kind || '记忆'} tone={entry.memory_kind === 'consolidated' ? 'amber' : entry.memory_scope === 'profile' ? 'emerald' : 'purple'} />
                            <KindBadge label={SOURCE_TYPE_LABELS[entry.source_type] || entry.source_type} tone="slate" />
                            <KindBadge label={STATUS_LABELS[entry.status] || entry.status || 'active'} tone={entry.status === 'archived_raw' ? 'rose' : 'emerald'} />
                            {entry.doc_id && <KindBadge label={entry.doc_id} tone="slate" />}
                            <span className="inline-flex items-center gap-1 text-xs text-gray-400">
                              <Clock className="h-3 w-3" />
                              {formatTime(entry.created_at)}
                            </span>
                          </div>
                          <div className="text-sm font-semibold text-gray-800">{entry.title || '记忆条目'}</div>
                          {!isExpanded && (
                            <p className="mt-1 text-sm text-gray-600">{truncateContent(entry.summary || entry.content)}</p>
                          )}
                        </div>
                        <div className="mt-1 text-gray-400">
                          {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                        </div>
                      </div>

                      {isExpanded && (
                        <div className="mt-3 border-t border-gray-100 pt-3">
                          {isEditing ? (
                            <div className="space-y-3">
                              <textarea
                                value={editContent}
                                onChange={(e) => setEditContent(e.target.value)}
                                className="min-h-[120px] w-full resize-none rounded-lg border border-gray-200 px-3 py-2 text-sm outline-none"
                                autoFocus
                              />
                              <div className="flex justify-end gap-2">
                                <button onClick={() => setEditingId(null)} className="rounded-lg px-3 py-1.5 text-sm text-gray-600 transition-colors hover:bg-gray-100">取消</button>
                                <button
                                  onClick={() => handleSave(entry.id)}
                                  disabled={isOperating}
                                  className="inline-flex items-center gap-1 rounded-lg bg-purple-600 px-3 py-1.5 text-sm text-white transition-colors hover:bg-purple-700 disabled:opacity-50"
                                >
                                  {isOperating ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
                                  保存
                                </button>
                              </div>
                            </div>
                          ) : (
                            <>
                              <p className="whitespace-pre-wrap break-words text-sm text-gray-700">{entry.content}</p>
                              <div className="mt-3 rounded-lg bg-gray-50 p-3 text-xs text-gray-600">
                                <div>摘要：{entry.summary || '暂无'}</div>
                                <div className="mt-1">作用域：{entry.memory_scope === 'profile' ? '全局画像' : '当前文档'}</div>
                                {entry.derived_from && entry.derived_from.length > 0 && (
                                  <div className="mt-1">来源条目：{entry.derived_from.length} 条</div>
                                )}
                              </div>
                              <div className="mt-3 flex flex-wrap justify-end gap-2">
                                <button
                                  onClick={() => fetchTrace(entry.id)}
                                  disabled={traceLoadingId === entry.id}
                                  className="inline-flex items-center gap-1 rounded-lg px-3 py-1.5 text-sm text-emerald-700 transition-colors hover:bg-emerald-50"
                                >
                                  {traceLoadingId === entry.id ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <GitBranch className="h-3.5 w-3.5" />}
                                  来源链
                                </button>
                                <button onClick={() => handleEdit(entry)} disabled={isOperating} className="inline-flex items-center gap-1 rounded-lg px-3 py-1.5 text-sm text-gray-600 transition-colors hover:bg-gray-100">
                                  <Edit3 className="h-3.5 w-3.5" />
                                  编辑
                                </button>
                                <button onClick={() => handleDelete(entry.id)} disabled={isOperating} className="inline-flex items-center gap-1 rounded-lg px-3 py-1.5 text-sm text-red-600 transition-colors hover:bg-red-50">
                                  {isOperating ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                                  删除
                                </button>
                              </div>
                            </>
                          )}

                          {trace && (
                            <div className="mt-4 rounded-xl border border-emerald-100 bg-emerald-50/60 p-4">
                              <div className="mb-2 text-sm font-semibold text-emerald-800">来源链</div>
                              <div className="space-y-2 text-xs text-gray-700">
                                <div>Trace: {trace.trace && Object.keys(trace.trace).length > 0 ? JSON.stringify(trace.trace, null, 2) : '暂无'}</div>
                                <div>上游来源：{trace.parents?.length || 0} 条</div>
                                {trace.parents?.length > 0 && (
                                  <div className="space-y-1">
                                    {trace.parents.map((parent) => (
                                      <div key={parent.id} className="rounded-lg bg-white/80 px-3 py-2">
                                        <div className="font-medium text-gray-800">{parent.title || '来源记忆'}</div>
                                        <div className="mt-1 text-gray-600">{truncateContent(parent.summary || parent.content, 120)}</div>
                                      </div>
                                    ))}
                                  </div>
                                )}
                                <div>下游派生：{trace.children?.length || 0} 条</div>
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

            {!loading && allEntries.length > 0 && activeTab !== 'graph' && (
              <div className="border-t border-gray-200 pt-4">
                <button
                  onClick={() => setShowClearConfirm(true)}
                  className="soft-card flex w-full items-center justify-center gap-2 rounded-xl px-4 py-3 text-red-600 transition-all hover:bg-red-50/50"
                >
                  <Trash2 className="h-4 w-4" />
                  <span className="font-medium">清空所有记忆</span>
                </button>
              </div>
            )}
          </div>
        </motion.div>

        {showClearConfirm && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50 p-4 backdrop-blur-sm"
            onClick={() => setShowClearConfirm(false)}
          >
            <div className="soft-panel w-full max-w-sm rounded-2xl p-6" onClick={(e) => e.stopPropagation()}>
              <div className="mb-4 flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-red-100">
                  <AlertTriangle className="h-5 w-5 text-red-600" />
                </div>
                <h3 className="text-lg font-bold">确认清空</h3>
              </div>
              <p className="mb-6 text-sm text-gray-600">此操作将删除所有记忆条目，且无法恢复。确定要继续吗？</p>
              <div className="flex gap-3">
                <button onClick={handleClearAll} className="flex-1 rounded-xl bg-red-600 py-3 font-medium text-white transition-colors hover:bg-red-700">确认清空</button>
                <button onClick={() => setShowClearConfirm(false)} className="flex-1 rounded-xl bg-gray-200 py-3 font-medium transition-colors hover:bg-gray-300">取消</button>
              </div>
            </div>
          </motion.div>
        )}
      </motion.div>
    </AnimatePresence>
  );
};

export default MemoryPanel;
