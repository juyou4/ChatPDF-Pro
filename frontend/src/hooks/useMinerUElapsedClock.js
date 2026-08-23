import { useEffect, useState } from 'react';

const elapsedAnchors = new Map();

export const resetMinerUElapsedClockAnchors = () => {
  elapsedAnchors.clear();
};

/**
 * 把 MinerU 秒表留在卡片内部。不要把 Date.now 抬到 ChatPDF，否则整列消息每秒重绘。
 */
export function useMinerUElapsedClock(active, resetKey, seedMs) {
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    if (!active || !resetKey) {
      if (resetKey && !active) elapsedAnchors.delete(resetKey);
      setElapsedSeconds(0);
      return undefined;
    }
    if (!elapsedAnchors.has(resetKey)) {
      const seed = Number(seedMs);
      elapsedAnchors.set(
        resetKey,
        Number.isFinite(seed) && seed > 0 && seed <= Date.now() + 2000
          ? seed
          : Date.now(),
      );
    }
    const tick = () => {
      const startedAt = elapsedAnchors.get(resetKey) || Date.now();
      setElapsedSeconds(Math.max(0, Math.floor((Date.now() - startedAt) / 1000)));
    };
    tick();
    const timer = window.setInterval(tick, 1000);
    return () => window.clearInterval(timer);
  }, [active, resetKey, seedMs]);

  return elapsedSeconds;
}
