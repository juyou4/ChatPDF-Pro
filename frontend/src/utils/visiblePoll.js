/**
 * 仅在页面可见时轮询。切到后台标签页就停表，回来再打一枪。
 * callback 返回 false 时停止（例如 MinerU 已到终态）。
 */
export function startVisiblePoll(callback, intervalMs, options = {}) {
  const immediate = options.immediate !== false;
  let cancelled = false;
  let timer = null;
  let running = false;

  const clearTimer = () => {
    if (timer != null) {
      window.clearTimeout(timer);
      timer = null;
    }
  };

  const isHidden = () => typeof document !== 'undefined' && document.hidden;

  const schedule = () => {
    if (cancelled || isHidden()) return;
    clearTimer();
    timer = window.setTimeout(() => {
      void tick();
    }, intervalMs);
  };

  const tick = async () => {
    if (cancelled || isHidden() || running) return;
    running = true;
    let shouldContinue = true;
    try {
      shouldContinue = await callback();
    } catch {
      shouldContinue = true;
    } finally {
      running = false;
    }
    if (shouldContinue !== false) schedule();
  };

  const onVisibility = () => {
    if (cancelled) return;
    if (isHidden()) {
      clearTimer();
      return;
    }
    void tick();
  };

  if (typeof document !== 'undefined') {
    document.addEventListener('visibilitychange', onVisibility);
  }
  if (immediate) void tick();
  else schedule();

  return () => {
    cancelled = true;
    clearTimer();
    if (typeof document !== 'undefined') {
      document.removeEventListener('visibilitychange', onVisibility);
    }
  };
}

export function mergeRecordIfChanged(prev, patch) {
  if (!patch || typeof patch !== 'object') return prev;
  let changed = false;
  const next = { ...prev };
  Object.entries(patch).forEach(([key, value]) => {
    if (prev?.[key] !== value) {
      next[key] = value;
      changed = true;
    }
  });
  return changed ? next : prev;
}
