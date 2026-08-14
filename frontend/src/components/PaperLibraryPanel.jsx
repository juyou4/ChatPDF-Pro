import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  X, Plus, RefreshCw, Trash2, Loader2, Library, ThumbsUp, ThumbsDown,
  ExternalLink, Sparkles, AlertCircle, Power, Inbox, FileDown,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE_URL = '';

// 后端把订阅发现严格限定为「显式触发」：聊天与上传路径都不会联网。
// 面板必须保持同样的克制，只有用户点按钮才发起 refresh。
const NOVELTY_LABELS = {
  new_work: { label: '新工作', tone: 'emerald' },
  new_version: { label: '新版本', tone: 'amber' },
};

const TONE_CLASSES = {
  emerald: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  amber: 'bg-amber-50 text-amber-700 border-amber-200',
  slate: 'bg-slate-50 text-slate-600 border-slate-200',
  blue: 'bg-blue-50 text-blue-700 border-blue-200',
};

const request = async (path, options = {}) => {
  const response = await fetch(`${API_BASE_URL}/paper-library${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!response.ok) {
    let detail = `请求失败（${response.status}）`;
    try {
      const body = await response.json();
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } catch {
      // 保留状态码兜底文案
    }
    throw new Error(detail);
  }
  return response.json();
};

const Badge = ({ tone = 'slate', children }) => (
  <span className={`px-2 py-0.5 rounded-lg border text-[11px] font-semibold ${TONE_CLASSES[tone] || TONE_CLASSES.slate}`}>
    {children}
  </span>
);

const PaperLibraryPanel = ({ isOpen, onClose }) => {
  const [subscriptions, setSubscriptions] = useState([]);
  const [feed, setFeed] = useState([]);
  const [activeSubscription, setActiveSubscription] = useState('');
  const [loading, setLoading] = useState(false);
  const [busyAction, setBusyAction] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [showCreate, setShowCreate] = useState(false);
  const [draftName, setDraftName] = useState('');
  const [draftQuery, setDraftQuery] = useState('');
  // 反馈是本地即时反映的：后端只在下一次入库时应用权重，界面先给出确认感。
  const [feedbackGiven, setFeedbackGiven] = useState({});

  const loadAll = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const [subscriptionPayload, feedPayload] = await Promise.all([
        request('/subscriptions'),
        request(`/feed?limit=50${activeSubscription ? `&subscription_id=${encodeURIComponent(activeSubscription)}` : ''}`),
      ]);
      setSubscriptions(subscriptionPayload.subscriptions || []);
      setFeed(feedPayload.items || []);
    } catch (exc) {
      setError(exc.message || '加载论文库失败');
    } finally {
      setLoading(false);
    }
  }, [activeSubscription]);

  useEffect(() => {
    if (isOpen) loadAll();
  }, [isOpen, loadAll]);

  const runAction = useCallback(async (action, fn, successText) => {
    setBusyAction(action);
    setError('');
    setNotice('');
    try {
      const result = await fn();
      if (successText) setNotice(typeof successText === 'function' ? successText(result) : successText);
      await loadAll();
    } catch (exc) {
      setError(exc.message || '操作失败');
    } finally {
      setBusyAction('');
    }
  }, [loadAll]);

  const handleCreate = useCallback(async () => {
    const name = draftName.trim();
    const query = draftQuery.trim();
    if (!name || !query) {
      setError('订阅名称和检索式都不能为空');
      return;
    }
    await runAction('create', () => request('/subscriptions', {
      method: 'POST',
      body: JSON.stringify({ name, query, keywords: [] }),
    }), '订阅已创建');
    setDraftName('');
    setDraftQuery('');
    setShowCreate(false);
  }, [draftName, draftQuery, runAction]);

  const handleFeedback = useCallback(async (item, relevance) => {
    const key = `${item.subscription_id}:${item.paper_id}`;
    setFeedbackGiven((prev) => ({ ...prev, [key]: relevance }));
    try {
      await request('/feedback', {
        method: 'POST',
        body: JSON.stringify({
          subscription_id: item.subscription_id,
          paper_id: item.paper_id,
          relevance,
          novelty: relevance === 'relevant' ? 'new' : 'known',
        }),
      });
    } catch (exc) {
      setFeedbackGiven((prev) => {
        const next = { ...prev };
        delete next[key];
        return next;
      });
      setError(exc.message || '反馈提交失败');
    }
  }, []);

  const subscriptionNames = useMemo(() => {
    const map = {};
    subscriptions.forEach((item) => { map[item.subscription_id] = item.name; });
    return map;
  }, [subscriptions]);

  const informedCount = useMemo(
    () => feed.filter((item) => item.feedback_informed).length,
    [feed],
  );

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-[70] bg-black/40 backdrop-blur-sm flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.97, y: 12 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.97, y: 12 }}
          transition={{ duration: 0.18 }}
          className="bg-white w-full max-w-4xl max-h-[86vh] rounded-3xl shadow-2xl flex flex-col overflow-hidden"
          onClick={(event) => event.stopPropagation()}
        >
          <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200/80">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-full bg-indigo-50 text-indigo-500 flex items-center justify-center">
                <Library className="w-4.5 h-4.5" />
              </div>
              <div>
                <h2 className="text-[15px] font-bold text-gray-900">论文订阅库</h2>
                <p className="text-[12px] text-gray-500">按订阅追踪新论文；反馈会调整后续相关性排序</p>
              </div>
            </div>
            <button onClick={onClose} className="p-2 rounded-xl hover:bg-gray-100 transition-colors">
              <X className="w-4 h-4 text-gray-500" />
            </button>
          </div>

          <div className="flex items-center gap-2 px-6 py-3 border-b border-gray-100 flex-wrap">
            <button
              onClick={() => setShowCreate((prev) => !prev)}
              className="inline-flex items-center gap-1.5 text-[12px] font-bold text-indigo-600 bg-indigo-50 hover:bg-indigo-100 px-3 py-1.5 rounded-xl transition-colors"
            >
              <Plus className="w-3.5 h-3.5" /> 新建订阅
            </button>
            <button
              onClick={() => runAction('scan', () => request('/process-new', { method: 'POST' }),
                (result) => `已扫描本地文档，新增 ${result.processed_count} 条`)}
              disabled={Boolean(busyAction)}
              className="inline-flex items-center gap-1.5 text-[12px] font-bold text-gray-700 bg-gray-100 hover:bg-gray-200 px-3 py-1.5 rounded-xl transition-colors disabled:opacity-50"
            >
              {busyAction === 'scan' ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <FileDown className="w-3.5 h-3.5" />}
              扫描已上传文档
            </button>
            <button
              onClick={() => runAction('refresh', () => request('/refresh', { method: 'POST' }),
                (result) => `联网发现完成，新增 ${result.processed_count} 条`)}
              disabled={Boolean(busyAction)}
              className="inline-flex items-center gap-1.5 text-[12px] font-bold text-emerald-700 bg-emerald-50 hover:bg-emerald-100 px-3 py-1.5 rounded-xl transition-colors disabled:opacity-50"
            >
              {busyAction === 'refresh' ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />}
              联网发现新论文
            </button>
            <div className="ml-auto flex items-center gap-2">
              {informedCount > 0 && (
                <span className="inline-flex items-center gap-1 text-[11px] text-indigo-600 font-semibold">
                  <Sparkles className="w-3.5 h-3.5" />
                  {informedCount} 条已按反馈调整
                </span>
              )}
            </div>
          </div>

          {showCreate && (
            <div className="px-6 py-4 bg-gray-50/80 border-b border-gray-100 space-y-2">
              <input
                value={draftName}
                onChange={(event) => setDraftName(event.target.value)}
                placeholder="订阅名称，例如：检索增强生成"
                className="w-full px-3 py-2 text-[13px] rounded-xl border border-gray-200 focus:outline-none focus:ring-2 focus:ring-indigo-200"
              />
              <input
                value={draftQuery}
                onChange={(event) => setDraftQuery(event.target.value)}
                placeholder="检索式，例如：retrieval augmented generation evaluation"
                className="w-full px-3 py-2 text-[13px] rounded-xl border border-gray-200 focus:outline-none focus:ring-2 focus:ring-indigo-200"
              />
              <div className="flex items-center gap-2">
                <button
                  onClick={handleCreate}
                  disabled={busyAction === 'create'}
                  className="text-[12px] font-bold text-white bg-indigo-600 hover:bg-indigo-700 px-3 py-1.5 rounded-xl transition-colors disabled:opacity-50"
                >
                  创建
                </button>
                <button
                  onClick={() => setShowCreate(false)}
                  className="text-[12px] font-bold text-gray-600 hover:bg-gray-100 px-3 py-1.5 rounded-xl transition-colors"
                >
                  取消
                </button>
                <span className="text-[11px] text-gray-400">检索式里的词会被切成关键词，用于相关性打分</span>
              </div>
            </div>
          )}

          {(error || notice) && (
            <div className={`px-6 py-2 text-[12px] flex items-center gap-2 ${error ? 'text-rose-600 bg-rose-50' : 'text-emerald-700 bg-emerald-50'}`}>
              <AlertCircle className="w-3.5 h-3.5 shrink-0" />
              {error || notice}
            </div>
          )}

          <div className="flex-1 overflow-y-auto">
            {subscriptions.length > 0 && (
              <div className="px-6 py-3 flex items-center gap-2 flex-wrap border-b border-gray-100">
                <button
                  onClick={() => setActiveSubscription('')}
                  className={`text-[12px] px-2.5 py-1 rounded-lg font-semibold transition-colors ${!activeSubscription ? 'bg-gray-900 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
                >
                  全部
                </button>
                {subscriptions.map((item) => (
                  <div key={item.subscription_id} className="inline-flex items-center gap-1">
                    <button
                      onClick={() => setActiveSubscription(item.subscription_id)}
                      className={`text-[12px] px-2.5 py-1 rounded-lg font-semibold transition-colors ${activeSubscription === item.subscription_id ? 'bg-gray-900 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
                    >
                      {item.name}
                      {!item.enabled && <span className="ml-1 text-[10px] opacity-70">已暂停</span>}
                    </button>
                    <button
                      title={item.enabled ? '暂停订阅' : '启用订阅'}
                      onClick={() => runAction(`toggle-${item.subscription_id}`, () => request(`/subscriptions/${item.subscription_id}`, {
                        method: 'PATCH',
                        body: JSON.stringify({ enabled: !item.enabled }),
                      }))}
                      className="p-1 rounded-lg hover:bg-gray-100 text-gray-400 hover:text-gray-600 transition-colors"
                    >
                      <Power className="w-3 h-3" />
                    </button>
                    <button
                      title="删除订阅"
                      onClick={() => runAction(`delete-${item.subscription_id}`, () => request(`/subscriptions/${item.subscription_id}`, {
                        method: 'DELETE',
                      }), '订阅已删除')}
                      className="p-1 rounded-lg hover:bg-rose-50 text-gray-400 hover:text-rose-500 transition-colors"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                ))}
              </div>
            )}

            {loading ? (
              <div className="py-16 flex items-center justify-center text-gray-400 text-[13px] gap-2">
                <Loader2 className="w-4 h-4 animate-spin" /> 正在加载
              </div>
            ) : feed.length === 0 ? (
              <div className="py-16 flex flex-col items-center justify-center text-center px-8">
                <Inbox className="w-8 h-8 text-gray-300 mb-3" />
                <p className="text-[13px] text-gray-500 font-semibold">
                  {subscriptions.length === 0 ? '还没有订阅' : '这个订阅还没有匹配到论文'}
                </p>
                <p className="text-[12px] text-gray-400 mt-1 max-w-md">
                  {subscriptions.length === 0
                    ? '先建一个订阅，再用「扫描已上传文档」把本地论文纳入，或用「联网发现新论文」按检索式抓取。'
                    : '试试放宽检索式，或点「联网发现新论文」抓取最新结果。'}
                </p>
              </div>
            ) : (
              <div className="divide-y divide-gray-100">
                {feed.map((item) => {
                  const key = `${item.subscription_id}:${item.paper_id}`;
                  const given = feedbackGiven[key];
                  const novelty = NOVELTY_LABELS[item.novelty];
                  return (
                    <div key={item.feed_id} className="px-6 py-4 hover:bg-gray-50/70 transition-colors">
                      <div className="flex items-start justify-between gap-4">
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-2 flex-wrap mb-1">
                            {novelty && <Badge tone={novelty.tone}>{novelty.label}</Badge>}
                            <Badge tone="blue">相关度 {(Number(item.relevance_score) * 100).toFixed(0)}%</Badge>
                            {item.feedback_informed && <Badge tone="emerald">已按反馈调整</Badge>}
                            {subscriptionNames[item.subscription_id] && (
                              <span className="text-[11px] text-gray-400">{subscriptionNames[item.subscription_id]}</span>
                            )}
                          </div>
                          <p className="text-[13px] font-semibold text-gray-900 leading-snug">{item.title}</p>
                          <p className="text-[12px] text-gray-500 mt-1 truncate">
                            {(item.authors || []).slice(0, 4).join('、')}
                            {item.year ? ` · ${item.year}` : ''}
                            {item.discovery_provider ? ` · 来源 ${item.discovery_provider}` : ''}
                          </p>
                          {(item.matched_keywords || []).length > 0 && (
                            <p className="text-[11px] text-gray-400 mt-1">
                              命中：{(item.matched_keywords || []).join('、')}
                            </p>
                          )}
                        </div>
                        <div className="flex items-center gap-1 shrink-0">
                          {item.external_url && (
                            <a
                              href={item.external_url}
                              target="_blank"
                              rel="noreferrer"
                              title="打开原文"
                              className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-400 hover:text-gray-700 transition-colors"
                            >
                              <ExternalLink className="w-3.5 h-3.5" />
                            </a>
                          )}
                          <button
                            title="相关"
                            onClick={() => handleFeedback(item, 'relevant')}
                            className={`p-1.5 rounded-lg transition-colors ${given === 'relevant' ? 'bg-emerald-100 text-emerald-600' : 'hover:bg-emerald-50 text-gray-400 hover:text-emerald-600'}`}
                          >
                            <ThumbsUp className="w-3.5 h-3.5" />
                          </button>
                          <button
                            title="不相关"
                            onClick={() => handleFeedback(item, 'not_relevant')}
                            className={`p-1.5 rounded-lg transition-colors ${given === 'not_relevant' ? 'bg-rose-100 text-rose-600' : 'hover:bg-rose-50 text-gray-400 hover:text-rose-500'}`}
                          >
                            <ThumbsDown className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <div className="px-6 py-3 border-t border-gray-100 text-[11px] text-gray-400">
            论文库与问答检索完全隔离：这里的订阅、反馈不参与回答的事实判断，也不会保存论文正文。
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default PaperLibraryPanel;
