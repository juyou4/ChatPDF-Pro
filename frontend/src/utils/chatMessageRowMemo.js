import { resolveDocumentUploadNotice } from '../components/DocumentUploadNotice';
import { canonicalizeMinerUProgressStage } from './mineruProgressUtils';

export function isDocumentUploadNoticeMessage(msg) {
  return msg?.type === 'system' && Boolean(resolveDocumentUploadNotice(msg));
}

export function resolveTableVisualVerificationState(verification) {
  const rawState = String(verification?.state || '').trim().toLowerCase();
  return ['queued', 'running', 'pending'].includes(rawState) ? 'pending' : rawState;
}

function sameParseGeneration(previous, next) {
  const prevGen = String(previous || '');
  const nextGen = String(next || '');
  if (prevGen && !nextGen) return true;
  return prevGen === nextGen;
}

/**
 * 旧行在父组件因解析轮询 / 点赞 / 流式外壳重绘时跳过 commit。
 * 上传通知才关心 liveParseStatus；冲突卡片才关心 embedding 标签。
 */
function liveProgressSignature(progress) {
  if (!progress || typeof progress !== 'object') return '';
  // 预估百分比和秒表都在卡片内部按耗时走。轮询带来的 percent 变化不能换卡片身份，
  // 否则虚拟列表重挂载，进度条会从满格收到当前值，看起来又在抖。
  const percentKey = progress.estimated === false
    ? [progress.percent, progress.remotePercent, progress.label]
    : [];
  return [
    ...percentKey,
    canonicalizeMinerUProgressStage(progress.stage),
    progress.stageLabel,
  ].map((value) => String(value ?? '')).join('|');
}

/**
 * 轮询会每几秒换一个新的 liveParseStatus 对象，但秒表已在卡片内部走。
 * 只在阶段/终态真正变化时交出新引用，避免上传卡反复进场动画。
 */
export function stabilizeLiveParseStatus(previous, next) {
  if (previous === next) return previous ?? next ?? null;
  if (!previous || !next) return next ?? null;
  if (
    previous.status === next.status
    && previous.title === next.title
    && previous.description === next.description
    && sameParseGeneration(previous.parseGeneration, next.parseGeneration)
    && liveProgressSignature(previous.progress) === liveProgressSignature(next.progress)
  ) {
    return previous;
  }
  return next;
}

export function chatMessageRowPropsAreEqual(prev, next) {
  if (prev.msg !== next.msg || prev.idx !== next.idx || prev.isLatest !== next.isLatest) return false;
  if (prev.isStreaming !== next.isStreaming) return false;
  if (prev.copied !== next.copied || prev.liked !== next.liked) return false;
  if (prev.remembered !== next.remembered || prev.disliked !== next.disliked) return false;
  if (prev.conflictRecoveryStatus !== next.conflictRecoveryStatus) return false;
  if (prev.runtime !== next.runtime) return false;
  if (isDocumentUploadNoticeMessage(next.msg) && prev.liveParseStatus !== next.liveParseStatus) {
    return false;
  }
  if (next.conflictRecoveryStatus !== 'idle') {
    if (prev.currentEmbeddingLabel !== next.currentEmbeddingLabel) return false;
    if (prev.indexedEmbeddingLabel !== next.indexedEmbeddingLabel) return false;
    if (prev.ragIndexBusy !== next.ragIndexBusy) return false;
  }
  return true;
}
