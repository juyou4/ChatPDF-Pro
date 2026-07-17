export const PRETRANSLATE_BATCH_SIZE = 48;

const normalizeIdentityPart = (value) => String(value ?? '').trim();

export function blockIndexMatchesParseContext({ blockIndex, docId, parseManifest } = {}) {
  if (!blockIndex || typeof blockIndex !== 'object') return false;

  const expectedDocId = normalizeIdentityPart(docId);
  const indexDocId = normalizeIdentityPart(blockIndex.doc_id);
  if (expectedDocId && indexDocId !== expectedDocId) return false;

  const manifest = parseManifest && typeof parseManifest === 'object' ? parseManifest : null;
  const isLegacyManifest = !manifest || Boolean(manifest?.metadata?.legacy_inferred);
  if (isLegacyManifest) return true;

  const expectedRoute = normalizeIdentityPart(
    manifest.resolved_route || manifest.requested_route || manifest.route,
  ).toLowerCase();
  const expectedGeneration = normalizeIdentityPart(manifest.generation);
  const expectedSourceHash = normalizeIdentityPart(manifest.source_hash);
  if (!expectedRoute || !expectedGeneration || !expectedSourceHash) return false;

  return normalizeIdentityPart(blockIndex.parser_route).toLowerCase() === expectedRoute
    && normalizeIdentityPart(blockIndex.parse_generation) === expectedGeneration
    && normalizeIdentityPart(blockIndex.document_source_hash) === expectedSourceHash;
}

export function buildPretranslateAutoIdentity({
  docId,
  parseIdentity,
  providerId,
  modelId,
} = {}) {
  const normalizedDocId = normalizeIdentityPart(docId);
  if (!normalizedDocId) return '';
  return [normalizedDocId, parseIdentity, providerId, modelId]
    .map((value) => encodeURIComponent(normalizeIdentityPart(value) || 'legacy'))
    .join('|');
}

export function selectPendingPretranslateBlocks({
  blocks,
  translations,
  failedBlockIds,
  force = false,
  retryFailed = false,
} = {}) {
  const translatedIds = new Set(Object.keys(translations || {}));
  const failedIds = failedBlockIds instanceof Set
    ? failedBlockIds
    : new Set(Array.isArray(failedBlockIds) ? failedBlockIds.map(String) : []);

  return (Array.isArray(blocks) ? blocks : []).filter((block) => {
    const blockId = normalizeIdentityPart(block?.block_id);
    if (!blockId) return false;
    if (force) return true;
    if (retryFailed) return failedIds.has(blockId);
    return !translatedIds.has(blockId);
  });
}

export function shouldForcePretranslateRequest({ force = false, retryFailed = false } = {}) {
  return Boolean(force || retryFailed);
}

export function chunkPretranslateBlocks(blocks, batchSize = PRETRANSLATE_BATCH_SIZE) {
  const numericSize = Number(batchSize);
  const safeBatchSize = Number.isFinite(numericSize) && numericSize > 0
    ? Math.max(1, Math.floor(numericSize))
    : PRETRANSLATE_BATCH_SIZE;
  const seen = new Set();
  const uniqueBlocks = [];

  (Array.isArray(blocks) ? blocks : []).forEach((block) => {
    const blockId = String(block?.block_id || '').trim();
    if (!blockId || seen.has(blockId)) return;
    seen.add(blockId);
    uniqueBlocks.push(block);
  });

  const batches = [];
  for (let index = 0; index < uniqueBlocks.length; index += safeBatchSize) {
    batches.push(uniqueBlocks.slice(index, index + safeBatchSize));
  }
  return batches;
}

export async function executePretranslateBatches({
  blocks,
  translateBatch,
  batchSize = PRETRANSLATE_BATCH_SIZE,
  isCurrent = () => true,
  onBatchStart,
  onBatchComplete,
}) {
  if (typeof translateBatch !== 'function') {
    throw new TypeError('translateBatch 必须是函数');
  }

  const batches = chunkPretranslateBlocks(blocks, batchSize);
  const successfulBlockIds = new Set();
  const failedBlockIds = new Set();

  const markRemainingFailed = (batchIndex) => {
    batches.slice(batchIndex + 1).flat().forEach((block) => {
      const blockId = String(block?.block_id || '').trim();
      if (blockId && !successfulBlockIds.has(blockId)) failedBlockIds.add(blockId);
    });
  };

  const buildResult = (extra = {}) => ({
    batchCount: batches.length,
    successfulBlockIds: [...successfulBlockIds],
    failedBlockIds: [...failedBlockIds],
    ...extra,
  });

  for (let batchIndex = 0; batchIndex < batches.length; batchIndex += 1) {
    if (!isCurrent()) return buildResult({ stale: true });

    const batch = batches[batchIndex];
    const batchMeta = {
      batchIndex,
      batchNumber: batchIndex + 1,
      batchCount: batches.length,
      batchSize: batch.length,
    };
    await onBatchStart?.(batchMeta);

    const data = await translateBatch(batch, batchMeta);
    if (!isCurrent() || data?.stale) return buildResult({ stale: true });
    if (!data) return buildResult({ aborted: true });

    const returnedItems = data?.items && typeof data.items === 'object' ? data.items : {};
    const batchSuccessfulIds = new Set(Object.keys(returnedItems));
    batchSuccessfulIds.forEach((blockId) => {
      successfulBlockIds.add(blockId);
      failedBlockIds.delete(blockId);
    });

    const reportedFailedIds = new Set(
      Array.isArray(data?.failed_block_ids) ? data.failed_block_ids.map(String) : [],
    );
    batch.forEach((block) => {
      const blockId = String(block?.block_id || '').trim();
      if (!blockId || batchSuccessfulIds.has(blockId)) return;
      reportedFailedIds.add(blockId);
    });
    reportedFailedIds.forEach((blockId) => {
      if (!successfulBlockIds.has(blockId)) failedBlockIds.add(blockId);
    });

    await onBatchComplete?.({
      ...batchMeta,
      successfulCount: successfulBlockIds.size,
      failedCount: failedBlockIds.size,
    });

    const error = String(data?.error || '').trim();
    if (error) {
      markRemainingFailed(batchIndex);
      return buildResult({ error });
    }

    const batchBlockIds = batch
      .map((block) => String(block?.block_id || '').trim())
      .filter(Boolean);
    const wholeBatchFailed = batchBlockIds.length > 0
      && batchSuccessfulIds.size === 0
      && batchBlockIds.every((blockId) => failedBlockIds.has(blockId));
    if (wholeBatchFailed) {
      markRemainingFailed(batchIndex);
      return buildResult({ circuitBroken: true });
    }
  }

  return buildResult();
}
