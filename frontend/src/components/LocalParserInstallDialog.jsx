import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { AlertTriangle, Check, Download, Loader2, MonitorDown, X } from 'lucide-react';

const API_BASE_URL = '';

const getApiErrorMessage = async (response, fallback) => {
  try {
    const body = await response.json();
    return body?.detail || body?.message || fallback;
  } catch {
    return fallback;
  }
};

export default function LocalParserInstallDialog({
  open,
  darkMode = false,
  onClose,
  onReady,
}) {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState('');

  const refreshStatus = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/runtime/addons/local-parser/status`);
      if (!response.ok) throw new Error(await getApiErrorMessage(response, '无法读取本地解析组件状态'));
      const nextStatus = await response.json();
      setStatus(nextStatus);
      setError('');
      return nextStatus;
    } catch (requestError) {
      setError(requestError.message || '无法读取本地解析组件状态');
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (open) void refreshStatus();
  }, [open, refreshStatus]);

  const jobStatus = status?.installation?.status;
  const isInstalling = ['queued', 'installing', 'verifying'].includes(jobStatus) || starting;

  useEffect(() => {
    if (!open || !isInstalling) return undefined;
    const timer = window.setInterval(() => { void refreshStatus(); }, 1200);
    return () => window.clearInterval(timer);
  }, [isInstalling, open, refreshStatus]);

  const install = useCallback(async () => {
    setStarting(true);
    setError('');
    try {
      const response = await fetch(`${API_BASE_URL}/api/runtime/addons/local-parser/install`, {
        method: 'POST',
      });
      if (!response.ok) throw new Error(await getApiErrorMessage(response, '无法开始安装本地解析组件'));
      setStatus(await response.json());
    } catch (requestError) {
      setError(requestError.message || '无法开始安装本地解析组件');
    } finally {
      setStarting(false);
    }
  }, []);

  const missingComponents = useMemo(
    () => (status?.python_components || []).filter((component) => !component.available),
    [status]
  );
  const missingTools = useMemo(
    () => (status?.system_tools || []).filter((tool) => !tool.available),
    [status]
  );
  const ready = Boolean(status?.ready);
  const installerSupported = status?.installer?.supported !== false;
  const progressText = status?.installation?.message
    || (status?.installation?.stage === 'downloading_model' ? '正在下载版面识别模型' : '正在准备安装');

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-[140] flex items-center justify-center bg-black/35 p-5 backdrop-blur-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onMouseDown={(event) => {
            if (event.target === event.currentTarget && !isInstalling) onClose?.();
          }}
        >
          <motion.section
            role="dialog"
            aria-modal="true"
            aria-labelledby="local-parser-install-title"
            initial={{ opacity: 0, y: 14, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 8, scale: 0.985 }}
            transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
            className={`w-full max-w-[460px] overflow-hidden rounded-[22px] border shadow-[0_24px_70px_rgba(16,18,22,0.24)] ${
              darkMode ? 'border-white/[0.1] bg-[#202227] text-gray-100' : 'border-white bg-white text-gray-900'
            }`}
          >
            <div className="flex items-start gap-3 px-5 pb-3 pt-5">
              <span className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-[14px] ${
                ready
                  ? (darkMode ? 'bg-emerald-400/15 text-emerald-300' : 'bg-emerald-50 text-emerald-600')
                  : (darkMode ? 'bg-[#FFA07A]/15 text-[#FFA07A]' : 'bg-[#FFF0EA] text-[#C96649]')
              }`}>
                {ready ? <Check className="h-5 w-5" strokeWidth={2.5} /> : <MonitorDown className="h-5 w-5" />}
              </span>
              <div className="min-w-0 flex-1">
                <h2 id="local-parser-install-title" className="text-[15px] font-bold">本地解析组件</h2>
                <p className={`mt-1 text-[12px] leading-5 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  {ready ? '本地路线已就绪' : '本地路线需要先准备版面与 OCR 运行时'}
                </p>
              </div>
              <button
                type="button"
                aria-label="关闭本地解析组件窗口"
                disabled={isInstalling}
                onClick={onClose}
                className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
                  darkMode ? 'text-gray-500 hover:bg-white/[0.08] hover:text-gray-200' : 'text-gray-400 hover:bg-gray-100 hover:text-gray-700'
                }`}
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className={`mx-5 rounded-[16px] border px-3.5 py-3 ${
              darkMode ? 'border-white/[0.08] bg-white/[0.035]' : 'border-[#EEE9E5] bg-[#FCFBFA]'
            }`}>
              {loading && !status ? (
                <div className={`flex items-center gap-2 text-[12px] ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  正在检查组件状态
                </div>
              ) : ready ? (
                <div className={`flex items-center gap-2 text-[12px] font-medium ${darkMode ? 'text-emerald-300' : 'text-emerald-700'}`}>
                  <Check className="h-4 w-4" strokeWidth={2.5} />
                  版面识别模型与本地运行时可用
                </div>
              ) : isInstalling ? (
                <div className={`flex items-center gap-2 text-[12px] font-medium ${darkMode ? 'text-[#FFA07A]' : 'text-[#B85F47]'}`}>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  {progressText}
                </div>
              ) : (
                <div className="space-y-1.5 text-[12px]">
                  {missingComponents.slice(0, 4).map((component) => (
                    <div key={component.id} className={`flex items-center gap-2 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                      <span className={`h-1.5 w-1.5 rounded-full ${darkMode ? 'bg-[#FFA07A]' : 'bg-[#D97A5D]'}`} />
                      {component.label}
                    </div>
                  ))}
                  {status?.model && !status.model.available && (
                    <div className={`flex items-center gap-2 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                      <span className={`h-1.5 w-1.5 rounded-full ${darkMode ? 'bg-[#FFA07A]' : 'bg-[#D97A5D]'}`} />
                      DocLayout-YOLO 权重
                    </div>
                  )}
                </div>
              )}
            </div>

            {(error || status?.installation?.error || !installerSupported) && (
              <div className={`mx-5 mt-3 flex gap-2 rounded-[14px] px-3 py-2.5 text-[12px] leading-5 ${
                darkMode ? 'bg-red-400/10 text-red-300' : 'bg-red-50 text-red-700'
              }`}>
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                <span>{error || status?.installation?.error || status?.installer?.message}</span>
              </div>
            )}

            {ready && missingTools.length > 0 && (
              <div className={`mx-5 mt-3 text-[11px] leading-5 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                可选系统工具未就绪：{missingTools.map((tool) => tool.label).join('、')}
              </div>
            )}

            <div className="flex items-center justify-end gap-2 px-5 py-5">
              {!ready && (
                <button
                  type="button"
                  disabled={!installerSupported || isInstalling || loading}
                  onClick={() => { void install(); }}
                  className="inline-flex min-h-9 items-center gap-2 rounded-full bg-[#D66E50] px-4 text-[12px] font-bold text-white transition-colors hover:bg-[#BF5E43] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/35 disabled:cursor-not-allowed disabled:opacity-45"
                >
                  {isInstalling ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />}
                  {isInstalling ? '准备中' : '安装组件'}
                </button>
              )}
              {ready && (
                <button
                  type="button"
                  onClick={onReady}
                  className="inline-flex min-h-9 items-center gap-2 rounded-full bg-[#D66E50] px-4 text-[12px] font-bold text-white transition-colors hover:bg-[#BF5E43] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/35"
                >
                  <Check className="h-3.5 w-3.5" strokeWidth={2.5} />
                  使用本地解析
                </button>
              )}
            </div>
          </motion.section>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
