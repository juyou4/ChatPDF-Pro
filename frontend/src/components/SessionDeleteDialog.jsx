import { useEffect, useRef } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { MessageSquare, Trash2, X } from 'lucide-react';

const getSessionName = (session) => String(session?.filename || '').trim() || '未命名文档';

/**
 * 删除会话只影响浏览器内保存的聊天记录；把这层说明放在确认处，避免用户误以为会删除原文。
 */
export default function SessionDeleteDialog({
  session,
  darkMode = false,
  onConfirm,
  onClose,
}) {
  const open = Boolean(session?.id);
  const cancelButtonRef = useRef(null);
  const reduceMotion = useReducedMotion();

  useEffect(() => {
    if (!open) return undefined;

    const focusTimer = window.requestAnimationFrame(() => {
      cancelButtonRef.current?.focus();
    });
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onClose?.();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.cancelAnimationFrame(focusTimer);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [onClose, open]);

  const motionTransition = reduceMotion
    ? { duration: 0 }
    : { type: 'spring', stiffness: 460, damping: 34, mass: 0.72 };

  return (
    <AnimatePresence initial={false}>
      {open && (
        <motion.div
          className="fixed inset-0 z-[150] flex items-center justify-center bg-[#17191d]/40 p-5 backdrop-blur-[3px]"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={reduceMotion ? { duration: 0 } : { duration: 0.16 }}
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) onClose?.();
          }}
        >
          <motion.section
            role="dialog"
            aria-modal="true"
            aria-labelledby="session-delete-dialog-title"
            aria-describedby="session-delete-dialog-description"
            initial={reduceMotion ? false : { opacity: 0, y: 16, scale: 0.975 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={reduceMotion ? { opacity: 0 } : { opacity: 0, y: 8, scale: 0.985 }}
            transition={motionTransition}
            className={`w-full max-w-[420px] overflow-hidden rounded-[24px] border shadow-[0_24px_70px_rgba(20,22,27,0.26),0_2px_8px_rgba(20,22,27,0.08)] ${
              darkMode
                ? 'border-white/[0.1] bg-[#23252a] text-gray-100'
                : 'border-white/90 bg-[#fffefd] text-[#24272d]'
            }`}
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div className="flex items-start gap-3 px-5 pb-3 pt-5">
              <span className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-[14px] ${
                darkMode ? 'bg-[#ff9a7a]/15 text-[#ffad91]' : 'bg-[#fff0eb] text-[#bd6048]'
              }`}>
                <Trash2 className="h-[19px] w-[19px]" strokeWidth={2.25} />
              </span>
              <div className="min-w-0 flex-1 pt-0.5">
                <h2 id="session-delete-dialog-title" className="text-[16px] font-bold leading-5">
                  删除会话记录？
                </h2>
                <p
                  id="session-delete-dialog-description"
                  className={`mt-1.5 text-[12px] leading-5 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}
                >
                  对话会从侧边栏移除，原始 PDF、解析结果和阅读内容不会受影响。
                </p>
              </div>
              <button
                type="button"
                onClick={onClose}
                className={`-mr-1 -mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/35 ${
                  darkMode
                    ? 'text-gray-500 hover:bg-white/[0.08] hover:text-gray-200'
                    : 'text-gray-400 hover:bg-[#f5f2ef] hover:text-gray-700'
                }`}
                aria-label="取消删除会话"
                title="关闭"
              >
                <X className="h-4 w-4" strokeWidth={2.25} />
              </button>
            </div>

            <div className={`mx-5 flex items-center gap-2.5 rounded-[16px] px-3.5 py-3 ${
              darkMode ? 'bg-white/[0.045] text-gray-300' : 'bg-[#f8f6f4] text-[#52565d]'
            }`}>
              <span className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-[10px] ${
                darkMode ? 'bg-white/[0.07] text-[#ffad91]' : 'bg-white text-[#bd6048] shadow-[0_1px_2px_rgba(34,36,40,0.08)]'
              }`}>
                <MessageSquare className="h-3.5 w-3.5" strokeWidth={2.25} />
              </span>
              <span className="min-w-0 truncate text-[12px] font-semibold">{getSessionName(session)}</span>
            </div>

            <div className="flex items-center justify-end gap-2 px-5 pb-5 pt-5">
              <button
                ref={cancelButtonRef}
                type="button"
                onClick={onClose}
                className={`inline-flex h-9 items-center justify-center rounded-full px-4 text-[12px] font-semibold transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/35 active:scale-[0.98] ${
                  darkMode
                    ? 'bg-white/[0.08] text-gray-200 hover:bg-white/[0.13]'
                    : 'bg-[#f2efec] text-[#4b5058] hover:bg-[#eae5e1]'
                }`}
              >
                取消
              </button>
              <button
                type="button"
                onClick={onConfirm}
                className="inline-flex h-9 items-center justify-center gap-1.5 rounded-full bg-[#bd6048] px-4 text-[12px] font-bold text-white shadow-[0_5px_13px_rgba(160,76,55,0.23)] transition-all duration-200 hover:bg-[#a9513d] hover:shadow-[0_7px_16px_rgba(160,76,55,0.28)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/40 active:scale-[0.98]"
              >
                <Trash2 className="h-3.5 w-3.5" strokeWidth={2.4} />
                删除会话
              </button>
            </div>
          </motion.section>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
