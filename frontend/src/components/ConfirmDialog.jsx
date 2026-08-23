import { useCallback, useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { RefreshCw, ScanText, Trash2, X } from 'lucide-react';

const TONE_PRESENTATION = {
  caution: {
    Icon: ScanText,
    confirmIcon: RefreshCw,
  },
  danger: {
    Icon: Trash2,
    confirmIcon: Trash2,
  },
  neutral: {
    Icon: ScanText,
    confirmIcon: RefreshCw,
  },
};

/**
 * 应用内确认弹层。用来替换 window.confirm，避免浏览器自带对话框压在产品界面上。
 */
export default function ConfirmDialog({
  open = false,
  title,
  description,
  impacts = [],
  confirmLabel = '确定',
  cancelLabel = '取消',
  tone = 'caution',
  darkMode = false,
  onConfirm,
  onCancel,
}) {
  const cancelButtonRef = useRef(null);
  const reduceMotion = useReducedMotion();
  const presentation = TONE_PRESENTATION[tone] || TONE_PRESENTATION.caution;
  const TitleIcon = presentation.Icon;
  const ConfirmIcon = presentation.confirmIcon;
  const descriptionLines = Array.isArray(description)
    ? description.filter(Boolean)
    : String(description || '').split('\n').map((line) => line.trim()).filter(Boolean);

  useEffect(() => {
    if (!open) return undefined;
    const focusTimer = window.requestAnimationFrame(() => {
      cancelButtonRef.current?.focus();
    });
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onCancel?.();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.cancelAnimationFrame(focusTimer);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [onCancel, open]);

  const motionTransition = reduceMotion
    ? { duration: 0 }
    : { type: 'spring', stiffness: 460, damping: 34, mass: 0.72 };

  return (
    <AnimatePresence initial={false}>
      {open && (
        <motion.div
          className="fixed inset-0 z-[160] flex items-center justify-center bg-[#17191d]/40 p-5 backdrop-blur-[3px]"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={reduceMotion ? { duration: 0 } : { duration: 0.16 }}
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) onCancel?.();
          }}
        >
          <motion.section
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="app-confirm-dialog-title"
            aria-describedby="app-confirm-dialog-description"
            initial={reduceMotion ? false : { opacity: 0, y: 16, scale: 0.975 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={reduceMotion ? { opacity: 0 } : { opacity: 0, y: 8, scale: 0.985 }}
            transition={motionTransition}
            className={`w-full max-w-[440px] overflow-hidden rounded-[24px] border shadow-[0_24px_70px_rgba(20,22,27,0.26),0_2px_8px_rgba(20,22,27,0.08)] ${
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
                <TitleIcon className="h-[19px] w-[19px]" strokeWidth={2.25} />
              </span>
              <div className="min-w-0 flex-1 pt-0.5">
                <h2 id="app-confirm-dialog-title" className="text-[16px] font-bold leading-5">
                  {title}
                </h2>
                <div
                  id="app-confirm-dialog-description"
                  className={`mt-1.5 space-y-1.5 text-[12px] leading-5 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}
                >
                  {descriptionLines.map((line) => (
                    <p key={line}>{line}</p>
                  ))}
                </div>
              </div>
              <button
                type="button"
                onClick={onCancel}
                className={`-mr-1 -mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/35 ${
                  darkMode
                    ? 'text-gray-500 hover:bg-white/[0.08] hover:text-gray-200'
                    : 'text-gray-400 hover:bg-[#f5f2ef] hover:text-gray-700'
                }`}
                aria-label={cancelLabel}
                title={cancelLabel}
              >
                <X className="h-4 w-4" strokeWidth={2.25} />
              </button>
            </div>

            {impacts.length > 0 && (
              <div className="flex flex-wrap gap-1.5 px-5 pb-1">
                {impacts.map((item) => (
                  <span
                    key={item}
                    className={`rounded-full px-2.5 py-1 text-[11px] font-semibold ${
                      darkMode
                        ? 'bg-white/[0.06] text-gray-300'
                        : 'bg-[#f8f6f4] text-[#5d5852]'
                    }`}
                  >
                    {item}
                  </span>
                ))}
              </div>
            )}

            <div className="flex items-center justify-end gap-2 px-5 pb-5 pt-5">
              <button
                ref={cancelButtonRef}
                type="button"
                onClick={onCancel}
                className={`inline-flex h-9 items-center justify-center rounded-full px-4 text-[12px] font-semibold transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/35 active:scale-[0.98] ${
                  darkMode
                    ? 'bg-white/[0.08] text-gray-200 hover:bg-white/[0.13]'
                    : 'bg-[#f2efec] text-[#4b5058] hover:bg-[#eae5e1]'
                }`}
              >
                {cancelLabel}
              </button>
              <button
                type="button"
                onClick={onConfirm}
                className={`inline-flex h-9 items-center justify-center gap-1.5 rounded-full px-4 text-[12px] font-bold text-white shadow-[0_5px_13px_rgba(160,76,55,0.23)] transition-all duration-200 hover:shadow-[0_7px_16px_rgba(160,76,55,0.28)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D97A5D]/40 active:scale-[0.98] ${
                  tone === 'danger' ? 'bg-[#bd6048] hover:bg-[#a9513d]' : 'bg-[#D97A5D] hover:bg-[#c66b50]'
                }`}
              >
                <ConfirmIcon className="h-3.5 w-3.5" strokeWidth={2.4} />
                {confirmLabel}
              </button>
            </div>
          </motion.section>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

export function useConfirmDialog() {
  const [request, setRequest] = useState(null);
  const resolverRef = useRef(null);

  const settle = useCallback((value) => {
    resolverRef.current?.(value);
    resolverRef.current = null;
    setRequest(null);
  }, []);

  const confirm = useCallback((options) => (
    new Promise((resolve) => {
      resolverRef.current?.(false);
      resolverRef.current = resolve;
      setRequest(options || {});
    })
  ), []);

  useEffect(() => () => {
    resolverRef.current?.(false);
    resolverRef.current = null;
  }, []);

  return {
    confirm,
    confirmDialogProps: {
      open: Boolean(request),
      title: request?.title || '确认操作',
      description: request?.description || '',
      impacts: request?.impacts || [],
      confirmLabel: request?.confirmLabel || '确定',
      cancelLabel: request?.cancelLabel || '取消',
      tone: request?.tone || 'caution',
      onConfirm: () => settle(true),
      onCancel: () => settle(false),
    },
  };
}
