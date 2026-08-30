/**
 * 关联文档选择器的候选合并规则。
 *
 * 候选有两个来源：后端 /documents/recall（按标题匹配度和新近度排序）和本地
 * 会话历史（后端不可用时的兜底）。合并时有两条不能违背的约束：
 *
 * 1. 搜索时本地历史必须自己过滤——后端只对自己那份结果做了匹配，
 *    直接混入会让搜索结果里混进一堆不相关的最近文档。
 * 2. 已勾选的文档必须始终可见，否则用户搜索之后无法取消勾选。
 */

const CROSS_DOCUMENT_BROWSE_LIMIT = 8;
const CROSS_DOCUMENT_SEARCH_LIMIT = 20;

export function buildCrossDocumentOptions({
  backendCandidates = [],
  historyCandidates = [],
  selectedIds = [],
  query = '',
  primaryDocId = '',
} = {}) {
  const needle = String(query || '').trim().toLowerCase();
  const primary = String(primaryDocId || '');
  const selected = new Set((selectedIds || []).map((value) => String(value || '')));
  const backendIds = new Set(
    (backendCandidates || []).map((item) => String(item?.doc_id || '').trim()).filter(Boolean)
  );

  const byId = new Map();
  [...(backendCandidates || []), ...(historyCandidates || [])].forEach((candidate) => {
    const candidateId = String(candidate?.doc_id || '').trim();
    if (!candidateId || candidateId === primary) return;
    if (byId.has(candidateId)) return;
    const matchesQuery = !needle
      || backendIds.has(candidateId)
      || String(candidate?.filename || '').toLowerCase().includes(needle);
    if (!matchesQuery && !selected.has(candidateId)) return;
    byId.set(candidateId, { ...candidate, doc_id: candidateId });
  });

  const options = [...byId.values()];
  options.sort((left, right) => (
    (selected.has(left.doc_id) ? 0 : 1) - (selected.has(right.doc_id) ? 0 : 1)
  ));
  return options.slice(0, needle ? CROSS_DOCUMENT_SEARCH_LIMIT : CROSS_DOCUMENT_BROWSE_LIMIT);
}

export { CROSS_DOCUMENT_BROWSE_LIMIT, CROSS_DOCUMENT_SEARCH_LIMIT };
