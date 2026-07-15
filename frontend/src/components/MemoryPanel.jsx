import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  BookOpen, AlertCircle, Bot, X, Trash2, Edit2, Check, RefreshCw, MessageSquare, Tag, AlignLeft, Brain, Cpu, Database, Hash, GitCommit, GitPullRequest, Search, FileText, Globe, Clock, Copy, Plus, Activity, AlertTriangle, Layers, Type, Sparkles, Image as ImageIcon, MessageCircle, ExternalLink, Download, FileUp, FolderOpen, Box, Archive, Folder, RotateCcw, ChevronDown, ChevronUp, GitBranch, Network, Loader2, Save, Edit3, Fingerprint, Camera, History
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE_URL = '';

const TAB_CONFIGS = [
  { id: 'profile', label: '全局画像', kind: 'profile', icon: Brain },
  { id: 'doc_fact', label: '文档事实', kind: 'doc_fact', icon: Database },
  { id: 'consolidated', label: '压缩事实', kind: 'consolidated', icon: PackageIcon },
  { id: 'graph', label: '图谱摘要', kind: 'graph', icon: Network },
];

function PackageIcon(props) {
  return <Box {...props} />;
}

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
      month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
    });
  } catch {
    return isoStr;
  }
};

const KindBadge = ({ label, tone = 'purple' }) => {
  const palette = {
    purple: 'bg-purple-100 text-purple-700',
    emerald: 'bg-emerald-100 text-emerald-700',
    blue: 'bg-blue-100 text-blue-700',
    amber: 'bg-amber-100 text-amber-700',
    slate: 'bg-gray-100 text-gray-700',
    rose: 'bg-rose-100 text-rose-700',
  };
  return (
    <span className={`inline-flex items-center gap-1 rounded-[12px] px-2.5 py-0.5 text-[11px] font-bold tracking-wide uppercase ${palette[tone] || palette.slate}`}>
      {label}
    </span>
  );
};

// 按照 Figure 2 重构大卡片 Dashboard
const DashboardCard = ({ icon: Icon, tone, label, title, subtitle, labelTone }) => {
  const palette = {
    blue: 'bg-[#fff2ed] text-[#f16b3a]',
    emerald: 'bg-emerald-50 text-emerald-500',
    amber: 'bg-amber-50 text-amber-500',
    slate: 'bg-gray-50 text-gray-400',
    purple: 'bg-[#fff1eb] text-[#fc8256]',
  };
  return (
    <div className="bg-white rounded-[28px] p-5 shadow-[0_8px_30px_-12px_rgba(0,0,0,0.06)] border border-gray-100/50 flex flex-col min-h-[140px] transform transition-all hover:-translate-y-1">
      <div className="flex items-start justify-between mb-auto">
        <div className={`w-11 h-11 rounded-[18px] flex items-center justify-center ${palette[tone] || palette.blue}`}>
          <Icon className="w-[22px] h-[22px]" strokeWidth={2.5} />
        </div>
        <div className={`px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider rounded-full ${palette[labelTone] || palette.slate}`}>
          {label}
        </div>
      </div>
      <div className="mt-5">
        <h4 className="text-[16px] font-bold text-gray-900 tracking-tight">{title}</h4>
        <p className="text-[12px] text-gray-500 mt-1 font-medium">{subtitle}</p>
      </div>
    </div>
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
      console.error('获取链失败:', err);
    } finally {
      setTraceLoadingId(null);
    }
  }, [traceById]);

  const fetchGraph = useCallback(async (docId) => {
    if (!docId) { setGraphData(null); return; }
    setGraphLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/memory/graph/${docId}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setGraphData(await res.json());
    } catch (err) {
      console.error('获取摘要失败:', err);
      setGraphData(null);
    } finally {
      setGraphLoading(false);
    }
  }, []);

  const handleRebuildFromEvents = useCallback(async () => {
    setRebuilding(true); setStatusMessage('');
    try {
      const res = await fetch(`${API_BASE_URL}/api/memory/rebuild-from-events`, { method: 'POST' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setStatusMessage(`画像 ${data.profile_entries||0} 条, 文档 ${data.session_count||0} 个, 索引 ${data.indexed_entries||0} 条`);
      await fetchAllData();
      if (activeTab === 'graph' && selectedDocId) fetchGraph(selectedDocId);
    } catch (err) {
      console.error('从事件恢复失败:', err);
      setStatusMessage('从事件恢复失败，请检查后端日志。');
    } finally {
      setRebuilding(false);
    }
  }, [activeTab, fetchAllData, fetchGraph, selectedDocId]);

  useEffect(() => {
    if (!isOpen) return;
    fetchAllData();
    setExpandedId(null); setEditingId(null); setTraceById({}); setGraphData(null);
    setShowClearConfirm(false); setStatusMessage('');
  }, [isOpen, fetchAllData]);

  useEffect(() => {
    if (!isOpen || activeTab !== 'graph') return;
    fetchGraph(selectedDocId);
  }, [isOpen, activeTab, selectedDocId, fetchGraph]);

  const handleEdit = (entry) => { setEditingId(entry.id); setEditContent(entry.content); setExpandedId(entry.id); };

  const handleSave = async (entryId) => {
    setOperatingId(entryId);
    try {
      const res = await fetch(`${API_BASE_URL}/api/memory/entries/${entryId}`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ content: editContent }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const updated = await res.json();
      setAllEntries((prev) => prev.map((entry) => (entry.id === entryId ? { ...entry, ...updated } : entry)));
      setTraceById((prev) => {
        if (!prev[entryId]) return prev;
        return { ...prev, [entryId]: { ...prev[entryId], entry: { ...prev[entryId].entry, ...updated } } };
      });
      setEditingId(null);
    } catch (err) {
      console.error('编辑失败:', err);
    } finally { setOperatingId(null); }
  };

  const handleDelete = async (entryId) => {
    if (!confirm('确定要删除这条记忆吗？')) return;
    setOperatingId(entryId);
    try {
      const res = await fetch(`${API_BASE_URL}/api/memory/entries/${entryId}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setAllEntries((prev) => prev.filter((entry) => entry.id !== entryId));
      setTraceById((prev) => { const next = { ...prev }; delete next[entryId]; return next; });
      if (expandedId === entryId) setExpandedId(null);
      if (editingId === entryId) setEditingId(null);
    } catch (err) { console.error('删除失败:', err); } finally { setOperatingId(null); }
  };

  const handleClearAll = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/memory/all`, { method: 'DELETE' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setAllEntries([]); setFocusAreas([]); setExpandedId(null); setEditingId(null); setTraceById({}); setGraphData(null);
      setStatusMessage(''); await fetchAllData();
    } catch (err) { console.error('清空失败:', err); } finally { setShowClearConfirm(false); }
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
        className="fixed inset-0 z-[100] flex items-center justify-center bg-black/30 backdrop-blur-md p-4 transition-all opacity-100 font-sans"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.94, opacity: 0, y: 10 }} animate={{ scale: 1, opacity: 1, y: 0 }} exit={{ scale: 0.94, opacity: 0, y: 8 }}
          transition={{ type: 'spring', stiffness: 350, damping: 30 }}
          className="w-full max-w-[1000px] max-h-[92vh] bg-[#fbfbfc] shadow-[0_24px_60px_-15px_rgba(0,0,0,0.15)] rounded-[40px] overflow-hidden flex flex-col relative"
          onClick={(e) => e.stopPropagation()}
        >
          {/* ========== Figure 2 Top Header / Tabs ========== */}
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-8 pb-4 z-10 sticky top-0 bg-[#fbfbfc]/95 backdrop-blur-xl">
            {/* 顶层关闭按钮 (绝对定位或统一安排，这里用极简叉号) */}
            <button onClick={onClose} className="absolute top-6 right-6 p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full transition-colors z-[110]">
              <X className="w-5 h-5" />
            </button>

            {/* Left Pill Group */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-1.5 p-1.5 bg-gray-100/60 rounded-[32px] border border-gray-200/50 w-full md:w-auto">
              {TAB_CONFIGS.map((tab) => {
                const active = activeTab === tab.id;
                return (
                  <button
                    key={tab.id} onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center justify-center gap-2 px-4 py-2 text-[12px] font-bold rounded-[24px] transition-all relative ${
                      active ? 'bg-white text-[#f16b3a] shadow-[0_2px_12px_-4px_rgba(241, 107, 58,0.2)]' : 'text-gray-500 hover:text-gray-700 hover:bg-white/50'
                    }`}
                  >
                    <span>{tab.label}</span>
                    <span className={`text-[9px] w-[18px] h-[18px] flex items-center justify-center font-bold rounded-full transition-colors ${
                      active ? 'bg-[#fff2ed] text-[#f16b3a]' : 'bg-transparent text-gray-400'
                    }`}>
                      {tabCounts[tab.id]}
                    </span>
                  </button>
                );
              })}
            </div>

            {/* Right Buttons */}
            <div className="flex items-center gap-3 w-full md:w-auto mt-2 md:mt-0 pr-8 md:pr-4">
               <button onClick={fetchAllData} disabled={loading} className="flex-1 md:flex-none flex items-center justify-center gap-2 px-5 py-2.5 rounded-[24px] bg-white hover:bg-gray-50 border border-gray-200/80 text-gray-700 font-bold text-[13px] shadow-sm transition-all">
                 <RotateCcw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} strokeWidth={2.5}/> 刷新状态
               </button>
               <button onClick={handleRebuildFromEvents} disabled={rebuilding} className="flex-1 md:flex-none flex items-center justify-center gap-2 px-5 py-2.5 rounded-[24px] bg-[#f07345] hover:bg-[#d55a2d] text-white font-bold text-[13px] shadow-[0_8px_16px_-6px_rgba(241, 107, 58,0.4)] transition-all">
                 {rebuilding ? <Loader2 className="w-3.5 h-3.5 animate-spin"/> : <RefreshCw className="w-3.5 h-3.5" strokeWidth={2.5}/>} 事件恢复
               </button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto px-8 pb-8 pt-4 custom-scrollbar">
            {/* ========== 4 Dashboard Cards ========== */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10 mt-2">
               <DashboardCard 
                  icon={RefreshCw} tone="blue" labelTone={statusData?.rebuild_required ? 'rose' : statusData?.dirty ? 'amber' : 'emerald'}
                  label={statusData?.rebuild_required ? '待重建' : statusData?.dirty ? '待落盘' : '已同步'}
                  title="同步状态"
                  subtitle={statusData?.rebuild_required ? `原因: ${statusData?.rebuild_reason}` : statusData?.pending_sync ? '防抖写入中' : '全网同步实时'}
               />
               <DashboardCard 
                  icon={Fingerprint} tone="amber" labelTone={statusData?.profile_snapshot_exists ? 'emerald' : 'slate'}
                  label={statusData?.profile_snapshot_exists ? '已生成' : '未生成'}
                  title="内存快照"
                  subtitle={`文档关联 ${statusData?.session_snapshot_count ?? 0} 个`}
               />
               <DashboardCard 
                  icon={History} tone="slate" labelTone="slate"
                  label={`${statusData?.event_log_files ?? 0} 日志`}
                  title="事件追踪"
                  subtitle={statusData?.last_event_at ? formatTime(statusData.last_event_at) : '暂无事件日志上报'}
               />
               <DashboardCard 
                  icon={Database} tone="purple" labelTone={statusData?.last_sync_at ? 'emerald' : 'slate'}
                  label={statusData?.last_sync_at ? '持久化' : '待处理'}
                  title="数据库持久"
                  subtitle={statusData?.last_sync_at ? formatTime(statusData.last_sync_at) : '核心库未建立'}
               />
            </div>

            {/* ========== Filter & Context Selectors ========== */}
            <div className="flex flex-col md:flex-row items-center justify-between mb-8 pb-4">
              {/* Focus Area list */}
              <div className="flex items-center gap-2 w-full max-w-lg mb-4 md:mb-0">
                  <div className="flex flex-wrap items-center gap-1.5 bg-gray-100/50 px-2 py-1.5 rounded-full border border-gray-100 flex-1 min-h-[44px]">
                     {focusAreas.length > 0 ? (
                        focusAreas.slice(0, 5).map((area) => (
                           <span key={area} className="bg-emerald-50 text-emerald-600 px-3 py-1 rounded-full text-[11px] font-bold border border-emerald-100 flex items-center gap-1">
                             <Tag className="w-3 h-3"/> {area}
                           </span>
                        ))
                     ) : (
                        <span className="text-gray-400 text-[12px] font-bold px-3 py-1 uppercase tracking-wider">系统级全局画像 / 多领域互补</span>
                     )}
                  </div>
              </div>
              
              {/* Doc Selection */}
              {activeTab !== 'profile' && docOptions.length > 0 && (
                <div className="flex items-center bg-gray-100/50 px-3 py-2 rounded-full border border-gray-100">
                  <Archive className="w-4 h-4 text-gray-400 mr-2" />
                  <select
                    value={selectedDocId} onChange={(e) => setSelectedDocId(e.target.value)}
                    className="bg-transparent border-none outline-none text-[13px] font-bold text-gray-700 cursor-pointer min-w-[120px]"
                  >
                    <option value="">📁 全局视图文档检索</option>
                    {docOptions.map((docId) => (
                      <option key={docId} value={docId}>{docId}</option>
                    ))}
                  </select>
                </div>
              )}
            </div>

            {/* ========== Lists & Empty State ========== */}
            {loading && (
              <div className="flex flex-col items-center justify-center py-20 text-gray-400">
                <Loader2 className="w-8 h-8 animate-spin mb-4 text-[#ed8c68]" />
                <span className="text-[13px] font-bold tracking-wide">读取记忆核心...</span>
              </div>
            )}

            {!loading && activeTab === 'graph' && (
              <div className="space-y-4">
                {graphLoading && (
                  <div className="flex flex-col items-center justify-center py-12 text-gray-400">
                    <Loader2 className="h-6 w-6 animate-spin mb-3 text-[#f16b3a]" />
                    <span className="text-[13px] font-bold">图谱矩阵生成中...</span>
                  </div>
                )}
                {!graphLoading && !selectedDocId && (
                  <div className="flex flex-col items-center justify-center py-16 opacity-70">
                    <div className="w-20 h-20 rounded-full bg-blue-50 flex items-center justify-center text-blue-200 mb-4">
                       <Network className="w-8 h-8" strokeWidth={2} />
                    </div>
                    <p className="text-sm text-gray-400 font-bold">关联网络尚未选择对应实体节点源</p>
                  </div>
                )}
                {!graphLoading && selectedDocId && graphData && (
                  <motion.div initial={{opacity:0, y:10}} animate={{opacity:1, y:0}} className="space-y-4">
                    <div className="grid gap-4 md:grid-cols-3">
                      <div className="bg-white rounded-[24px] p-5 shadow-sm border border-gray-100/80">
                        <div className="text-[11px] font-bold uppercase tracking-wider text-gray-400 mb-1">检索索引来源</div>
                        <div className="text-[15px] font-bold text-gray-800 truncate">{selectedDocId}</div>
                      </div>
                      <div className="bg-white rounded-[24px] p-5 shadow-sm border border-gray-100/80">
                        <div className="text-[11px] font-bold uppercase tracking-wider text-gray-400 mb-1">聚簇网络节点</div>
                        <div className="text-[15px] font-bold text-gray-800">{graphData.node_count} 实体</div>
                      </div>
                      <div className="bg-white rounded-[24px] p-5 shadow-sm border border-gray-100/80">
                        <div className="text-[11px] font-bold uppercase tracking-wider text-gray-400 mb-1">动态拓扑边数</div>
                        <div className="text-[15px] font-bold text-gray-800">{graphData.edge_count} 关联</div>
                      </div>
                    </div>
                    <div className="bg-white rounded-[28px] p-6 shadow-[0_8px_30px_-12px_rgba(0,0,0,0.06)] border border-gray-100/50">
                      <div className="mb-4 text-[13px] font-bold text-gray-800 uppercase tracking-widest border-b border-gray-50 pb-2">图谱神经元网络预览</div>
                      {graphData.nodes && graphData.nodes.length > 0 ? (
                        <div className="flex flex-wrap gap-2.5">
                          {graphData.nodes.map((node) => (
                            <KindBadge key={node.id} label={`${node.type}: ${node.label}`} tone={node.type === 'figure' ? 'amber' : node.type === 'table' ? 'rose' : 'blue'} />
                          ))}
                        </div>
                      ) : (
                        <div className="text-[13px] text-gray-400 font-bold">神经元尚未固化内容</div>
                      )}
                    </div>
                  </motion.div>
                )}
              </div>
            )}

            {!loading && activeTab !== 'graph' && filteredEntries.length === 0 && (
              <div className="flex flex-col items-center justify-center py-20 opacity-90 mt-4">
                 <div className="relative flex items-center justify-center mb-8">
                    <div className="absolute w-[180px] h-[180px] rounded-full border border-gray-100 bg-gray-50/30 shadow-[inset_0_0_20px_rgba(0,0,0,0.02)]" />
                    <div className="absolute w-[130px] h-[130px] rounded-full border border-gray-100/80 bg-gray-50/50 shadow-[inset_0_0_15px_rgba(0,0,0,0.01)]" />
                    <div className="absolute w-[80px] h-[80px] rounded-full border border-gray-100 bg-[#fff1eb]" />
                    <div className="relative z-10 text-[#fc8256]">
                        <Brain className="w-10 h-10" strokeWidth={2} />
                    </div>
                 </div>
                 <h3 className="text-[20px] font-bold text-gray-900 mb-3 tracking-tight">空检索库 /暂无记忆条目</h3>
                 <p className="text-[14px] text-gray-500 max-w-sm text-center leading-relaxed">
                    当前视图下的深度记忆库是空的。同步您的文档、解析全新文献、或开始多轮深度思考来逐渐丰满智能体的知识框架。
                 </p>
              </div>
            )}

            {!loading && activeTab !== 'graph' && filteredEntries.length > 0 && (
              <div className="space-y-4">
                {filteredEntries.map((entry) => {
                  const isExpanded = expandedId === entry.id;
                  const isEditing = editingId === entry.id;
                  const isOperating = operatingId === entry.id;
                  const trace = traceById[entry.id];
                  return (
                    <motion.div layout key={entry.id} className="bg-white rounded-[24px] p-5 shadow-sm border border-gray-100/80 transition-all hover:shadow-[0_8px_30px_-12px_rgba(0,0,0,0.08)]">
                      <div className="flex cursor-pointer items-start gap-4" onClick={() => setExpandedId((prev) => (prev === entry.id ? null : entry.id))}>
                        <div className="min-w-0 flex-1">
                          <div className="mb-2.5 flex flex-wrap items-center gap-2">
                            <KindBadge label={MEMORY_KIND_LABELS[entry.memory_kind] || entry.memory_kind || '记忆'} tone={entry.memory_kind === 'consolidated' ? 'amber' : entry.memory_scope === 'profile' ? 'emerald' : 'blue'} />
                            <KindBadge label={SOURCE_TYPE_LABELS[entry.source_type] || entry.source_type} tone="slate" />
                            <KindBadge label={STATUS_LABELS[entry.status] || entry.status || '生效'} tone={entry.status === 'archived_raw' ? 'rose' : 'emerald'} />
                            {entry.doc_id && <KindBadge label={entry.doc_id} tone="slate" />}
                            <span className="inline-flex items-center gap-1 text-[11px] font-bold text-gray-400 ml-1">
                              <History className="h-3 w-3" />
                              {formatTime(entry.created_at)}
                            </span>
                          </div>
                          <div className="text-[15px] font-bold text-gray-900 tracking-tight leading-snug break-words">
                            {entry.title || '无标题记忆矩阵'}
                          </div>
                          {!isExpanded && (
                            <p className="mt-1.5 text-[13px] text-gray-500 font-medium">
                               {truncateContent(entry.summary || entry.content, 90)}
                            </p>
                          )}
                        </div>
                        <div className="mt-1 text-gray-300 bg-gray-50 rounded-full p-1 border border-gray-100">
                          {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                        </div>
                      </div>

                      <AnimatePresence>
                        {isExpanded && (
                          <motion.div initial={{height: 0, opacity:0}} animate={{height:'auto', opacity:1}} exit={{height:0, opacity:0}} className="overflow-hidden">
                            <div className="mt-4 pt-4 border-t border-gray-100 border-dashed">
                              {isEditing ? (
                                <div className="space-y-3">
                                  <textarea
                                    value={editContent} onChange={(e) => setEditContent(e.target.value)}
                                    className="min-h-[140px] w-full resize-none rounded-xl border border-gray-200 px-4 py-3 text-[13px] outline-none font-medium focus:ring-2 focus:ring-[#ed8c68]/20 focus:border-[#ed8c68]/50 transition-all bg-gray-50/50"
                                    autoFocus
                                  />
                                  <div className="flex justify-end gap-2">
                                    <button onClick={() => setEditingId(null)} className="rounded-[16px] px-4 py-2 text-[12px] font-bold text-gray-600 transition-colors hover:bg-gray-100">取消</button>
                                    <button
                                      onClick={() => handleSave(entry.id)} disabled={isOperating}
                                      className="inline-flex items-center gap-1.5 rounded-[16px] bg-[#f16b3a] px-5 py-2 text-[12px] font-bold text-white transition-all hover:bg-[#d55a2d] disabled:opacity-50 shadow-sm"
                                    >
                                      {isOperating ? <Loader2 className="h-3 w-3 animate-spin" /> : <Save className="h-3 w-3" />} 应用修改
                                    </button>
                                  </div>
                                </div>
                              ) : (
                                <>
                                  <p className="whitespace-pre-wrap break-words text-[14px] text-gray-700 leading-relaxed font-medium bg-gray-50/50 p-4 rounded-[16px]">
                                    {entry.content}
                                  </p>
                                  <div className="mt-3 flex flex-wrap items-center gap-4 px-2 text-[12px] font-bold text-gray-400">
                                    <div className="flex items-center gap-1"><AlignLeft className="w-3.5 h-3.5"/> 摘要: {entry.summary || '原始信息体'}</div>
                                    <div className="flex items-center gap-1"><Cpu className="w-3.5 h-3.5"/> 域宽: {entry.memory_scope === 'profile' ? '全域映射' : '文档限定'}</div>
                                    {entry.derived_from?.length > 0 && <div className="flex items-center gap-1"><Network className="w-3.5 h-3.5"/> 汇聚节点: {entry.derived_from.length}</div>}
                                  </div>
                                  <div className="mt-5 flex flex-wrap justify-end gap-2">
                                    <button
                                      onClick={() => fetchTrace(entry.id)} disabled={traceLoadingId === entry.id}
                                      className="inline-flex items-center gap-1.5 rounded-[16px] px-4 py-2 text-[12px] font-bold text-blue-600 bg-blue-50 transition-all hover:bg-blue-100"
                                    >
                                      {traceLoadingId === entry.id ? <Loader2 className="h-3 w-3 animate-spin" /> : <GitBranch className="h-3 w-3" />}
                                      生成追溯谱图
                                    </button>
                                    <button onClick={() => handleEdit(entry)} disabled={isOperating} className="inline-flex items-center gap-1.5 rounded-[16px] px-4 py-2 text-[12px] font-bold text-gray-600 border border-gray-200 transition-colors hover:bg-gray-50">
                                      <Edit3 className="h-3 w-3" /> 修改内容
                                    </button>
                                    <button onClick={() => handleDelete(entry.id)} disabled={isOperating} className="inline-flex items-center gap-1.5 rounded-[16px] px-4 py-2 text-[12px] font-bold text-rose-600 bg-rose-50 transition-colors hover:bg-rose-100">
                                      {isOperating ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />} 释放删除
                                    </button>
                                  </div>
                                </>
                              )}

                              {trace && (
                                <motion.div initial={{opacity:0}} animate={{opacity:1}} className="mt-4 rounded-[20px] border border-blue-100 bg-[#fff2ed]/50 p-5">
                                  <div className="mb-3 text-[13px] font-bold text-blue-800 tracking-wide">🔗 节点联结追溯</div>
                                  <div className="space-y-3 text-[12px] text-gray-700 font-medium font-mono">
                                    <div className="p-3 bg-white/60 rounded-xl whitespace-pre-wrap overflow-x-auto border border-blue-50">
                                       Trace: {trace.trace && Object.keys(trace.trace).length > 0 ? JSON.stringify(trace.trace, null, 2) : '孤立节点'}
                                    </div>
                                    <div className="flex items-center gap-2"><Network className="w-3.5 h-3.5 text-blue-400"/> 上游追溯：{trace.parents?.length || 0} 祖先节点</div>
                                    {trace.parents?.length > 0 && (
                                      <div className="space-y-2 pl-4 border-l-2 border-blue-200">
                                        {trace.parents.map((parent) => (
                                          <div key={parent.id} className="rounded-xl bg-white border border-blue-50 px-4 py-3 shadow-sm">
                                            <div className="font-bold text-gray-800 tracking-tight">{parent.title || '来源簇'}</div>
                                            <div className="mt-1 text-gray-500">{truncateContent(parent.summary || parent.content, 90)}</div>
                                          </div>
                                        ))}
                                      </div>
                                    )}
                                    <div className="flex items-center gap-2"><Layers className="w-3.5 h-3.5 text-blue-400"/> 衍射投射：{trace.children?.length || 0} 子节点</div>
                                  </div>
                                </motion.div>
                              )}
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </motion.div>
                  );
                })}
              </div>
            )}
          </div>
          
          {/* Footer Bar to match figure 2 */}
          <div className="flex items-center justify-between px-8 py-4 bg-white border-t border-gray-100/60 z-10 w-full relative shrink-0">
             <div className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]"></div>
                <span className="text-[10px] uppercase font-bold text-gray-500 tracking-widest">系统活动: 智能记忆枢纽 V2.4</span>
             </div>
             <div className="text-[10px] uppercase font-bold text-gray-400 tracking-widest">
                关注领域: {(focusAreas.length > 0) ? `${focusAreas.length} ACTIVATED` : 'NONE'}
             </div>
          </div>
        </motion.div>

        {showClearConfirm && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="fixed inset-0 z-[120] flex items-center justify-center bg-black/40 backdrop-blur-sm p-4"
            onClick={() => setShowClearConfirm(false)}
          >
            <div className="w-full max-w-[400px] bg-white rounded-[28px] p-8 shadow-2xl" onClick={(e) => e.stopPropagation()}>
              <div className="mb-5 flex flex-col items-center gap-3">
                <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-rose-50 border border-rose-100">
                  <AlertTriangle className="h-6 w-6 text-rose-500" strokeWidth={2.5} />
                </div>
                <h3 className="text-[18px] font-bold text-gray-900 tracking-tight mt-1">系统全盘清除确认</h3>
              </div>
              <p className="mb-8 text-[13px] text-gray-500 text-center font-medium leading-relaxed">这一严重操作将永久抹除并截断长时效、全范围维度的结构事实及文档矩阵索引，这不可恢复。是否执意释放缓存？</p>
              <div className="flex gap-3">
                <button onClick={() => setShowClearConfirm(false)} className="flex-[1] rounded-2xl bg-gray-50 border border-gray-100 py-3 font-bold text-[13px] text-gray-600 transition-colors hover:bg-gray-100">中止撤回</button>
                <button onClick={handleClearAll} className="flex-[1] rounded-2xl bg-rose-500 shadow-[0_8px_20px_-6px_rgba(244,63,94,0.5)] py-3 font-bold text-[13px] text-white transition-all hover:bg-rose-600">强制脱核清算</button>
              </div>
            </div>
          </motion.div>
        )}
      </motion.div>
    </AnimatePresence>
  );
};

export default MemoryPanel;
