/**
 * 答案自审（answer_critic）提示的展示逻辑。
 *
 * 抽成纯函数便于单测：这里的去重与占位符清洗对应过两次线上可见的错误提示，
 * 混在 ChatPDF.jsx 里无法在不挂载整棵组件树的情况下验证。
 */

// 结构化引文协议占位符（CITATION【n】/START_PHRASE 等）属于模型输出的内部格式，
// 不应出现在自审提示里。后端已改为只审展示态答案，这里是渲染前的兜底清洗。
const CRITIC_PROTOCOL_ARTIFACT_RE = /CITATION\s*[[【]\d{1,3}[\]】]|CITATION\s+LIST|FINAL\s+ANSWER|(?:START|END)_PHRASE\s*[:：]/gi;

export const sanitizeCriticText = (value) => {
  const text = typeof value === 'string' ? value : '';
  if (!text) return '';
  return text
    .replace(CRITIC_PROTOCOL_ARTIFACT_RE, '')
    // 清洗后可能剩下空的引号对（如「有 N 处…：「」」）
    .replace(/[：:]?\s*「\s*」/g, '')
    .replace(/\s+/g, ' ')
    .trim();
};

// issue 分类 → 展示样式。未知类型回落到 other，避免后端新增枚举时前端崩掉。
export const CRITIC_ISSUE_TYPE_META = {
  hallucination: { label: '无据论断', className: 'text-[#5c564f]' },
  unsupported_number: { label: '数值无依据', className: 'text-[#5c564f]' },
  missing_citation: { label: '缺少引用', className: 'text-[#5c564f]' },
  wrong_citation: { label: '引用不匹配', className: 'text-[#5c564f]' },
  overreach: { label: '结论过强', className: 'text-[#5c564f]' },
  other: { label: '其他问题', className: 'text-[#5c564f]' },
};

export const getCriticIssueTypeMeta = (issueType) => (
  CRITIC_ISSUE_TYPE_META[issueType] || CRITIC_ISSUE_TYPE_META.other
);

/**
 * reason 与 issues 曾因后端把 issues[0] 复制进 reason 而重复渲染，这里统一去重。
 *
 * 优先读结构化的 issue_details（带 issue_type 与 claim_span 锚点），旧消息只有
 * issues 字符串数组时退化为无类型条目。
 */
export const buildCriticDetailLines = (critic) => {
  if (!critic || typeof critic !== 'object') return [];

  const details = Array.isArray(critic.issue_details) ? critic.issue_details : [];
  const entries = details.length > 0
    ? details.map((item) => ({
      text: item && typeof item === 'object' ? item.text : item,
      issueType: item && typeof item === 'object' ? item.issue_type : undefined,
      claimSpan: item && typeof item === 'object' ? item.claim_span : '',
      evidenceRefs: item && typeof item === 'object' && Array.isArray(item.evidence_refs)
        ? item.evidence_refs
        : [],
    }))
    : (Array.isArray(critic.issues) ? critic.issues : []).map((text) => ({ text }));

  const candidates = [{ text: critic.reason }, ...entries];
  const seen = new Set();
  const lines = [];
  candidates.forEach((candidate) => {
    const text = sanitizeCriticText(candidate?.text);
    if (!text || seen.has(text)) return;
    seen.add(text);
    lines.push({
      text,
      issueType: candidate?.issueType,
      claimSpan: sanitizeCriticText(candidate?.claimSpan),
      evidenceRefs: Array.isArray(candidate?.evidenceRefs) ? candidate.evidenceRefs : [],
    });
  });
  return lines;
};

/**
 * 引用覆盖不足与「LLM 判定幻觉」是两类信号，后端用 citation_risk 单独表达。
 * 历史消息没有该字段，退回原先的绝对条数判断。
 */
export const hasCitationRisk = (critic) => {
  if (!critic || typeof critic !== 'object') return false;
  if (typeof critic.citation_risk === 'boolean') return critic.citation_risk;
  return Number(critic?.citation_coverage?.uncited_factual_count || 0) > 0;
};

export const hasOverreachRisk = (critic) => {
  if (!critic || typeof critic !== 'object') return false;
  if (typeof critic.overreach_risk === 'boolean') return critic.overreach_risk;
  const details = Array.isArray(critic.issue_details) ? critic.issue_details : [];
  return details.some((item) => item && item.issue_type === 'overreach');
};

export const formatEvidenceRefs = (refs) => {
  if (!Array.isArray(refs) || refs.length === 0) return '';
  const markers = [];
  refs.forEach((value) => {
    const ref = Number(value);
    if (!Number.isInteger(ref) || ref <= 0) return;
    const marker = `[${ref}]`;
    if (!markers.includes(marker)) markers.push(marker);
  });
  return markers.join('');
};

/**
 * 全文总结的覆盖度来自后端 reading_outline 的解析身份绑定账本。
 * 前端只做展示，不从回答文本或引用数量反推，以免把局部回答误标成“全文”。
 */
export const getFullDocumentSummaryCoverage = (certainty) => {
  if (!certainty || typeof certainty !== 'object') return null;
  const raw = certainty.full_document_summary || certainty.fullDocumentSummary;
  if (!raw || typeof raw !== 'object') return null;

  const toCount = (value) => {
    const count = Number(value);
    return Number.isFinite(count) && count >= 0 ? Math.floor(count) : 0;
  };
  const bodyExpected = toCount(raw.body_expected ?? raw.bodyExpected);
  const bodySummarized = toCount(raw.body_summarized ?? raw.bodySummarized);
  const appendixExpected = toCount(raw.appendix_expected ?? raw.appendixExpected);
  const appendixSummarized = toCount(raw.appendix_summarized ?? raw.appendixSummarized);
  const expected = bodyExpected + appendixExpected;
  const summarized = Math.min(expected, bodySummarized + appendixSummarized);
  if (expected <= 0) return null;

  const complete = Boolean(raw.complete)
    || (bodySummarized >= bodyExpected && appendixSummarized >= appendixExpected);
  const parts = [`正文 ${bodySummarized}/${bodyExpected}`];
  if (appendixExpected > 0) parts.push(`附录 ${appendixSummarized}/${appendixExpected}`);
  const presentationMode = String(raw.presentation_mode ?? raw.presentationMode ?? '').trim().toLowerCase();
  const visibleSectionCount = toCount(raw.visible_section_count ?? raw.visibleSectionCount);
  const structuralSectionCount = toCount(raw.structural_section_count ?? raw.structuralSectionCount);
  const semanticQualityStatus = String(raw.semantic_quality_status ?? raw.semanticQualityStatus ?? '').trim().toLowerCase();
  const semanticQuality = raw.semantic_quality ?? raw.semanticQuality ?? {};
  const derivedLandmarks = semanticQuality?.derived_landmark_result_coverage
    ?? semanticQuality?.derivedLandmarkResultCoverage
    ?? {};
  const landmarkExpected = toCount(
    semanticQuality?.landmark_expected_claim_count
    ?? semanticQuality?.landmarkExpectedClaimCount
    ?? derivedLandmarks?.expected_claim_count
    ?? derivedLandmarks?.expectedClaimCount,
  );
  const landmarkCovered = toCount(
    semanticQuality?.landmark_covered_claim_count
    ?? semanticQuality?.landmarkCoveredClaimCount
    ?? derivedLandmarks?.covered_claim_count
    ?? derivedLandmarks?.coveredClaimCount,
  );
  const missingSlots = Array.isArray(semanticQuality?.missing_slots ?? semanticQuality?.missingSlots)
    ? (semanticQuality.missing_slots ?? semanticQuality.missingSlots).filter(Boolean)
    : [];
  const restatingThemes = Array.isArray(semanticQuality?.themes_restating_sections ?? semanticQuality?.themesRestatingSections)
    ? (semanticQuality.themes_restating_sections ?? semanticQuality.themesRestatingSections).filter(Boolean)
    : [];
  const themesWithoutEvidence = Array.isArray(semanticQuality?.themes_without_evidence ?? semanticQuality?.themesWithoutEvidence)
    ? (semanticQuality.themes_without_evidence ?? semanticQuality.themesWithoutEvidence).filter(Boolean)
    : [];
  const projectionHint = presentationMode === 'thematic'
    ? '已按主题综合，不逐章复述。'
    : (visibleSectionCount > 0 ? `已展开 ${visibleSectionCount} 节章节导览。` : '当前为章节导览。');
  // 主题退化和主题缺证据是两类不同的缺口，笼统写成「有待补充」会让用户无从下手。
  const qualityDetails = [
    missingSlots.length ? `缺${missingSlots.join('、')}` : '',
    landmarkExpected ? `关键结论证据 ${landmarkCovered}/${landmarkExpected}` : '',
    restatingThemes.length ? `${restatingThemes.length} 个主题仅复述章节` : '',
    themesWithoutEvidence.length ? `${themesWithoutEvidence.length} 个主题缺证据` : '',
  ].filter(Boolean);
  const qualityHint = semanticQualityStatus === 'needs_review'
    ? `主题重点仍有待补充${qualityDetails.length ? `（${qualityDetails.join('；')}）` : ''}，建议核对关键结论。`
    : '';
  return {
    complete,
    text: `全文覆盖 ${summarized}/${expected}`,
    title: `${parts.join('，')}。${complete ? '当前解析版本的章节均已纳入。' : '部分章节尚未形成可验证提要。'}${projectionHint}${structuralSectionCount && structuralSectionCount !== summarized ? `结构节点 ${structuralSectionCount} 个。` : ''}${qualityHint}`,
  };
};

// 分档与同一条消息上的 qaScore 徽章保持一致的三档语义，避免 40% 和 90% 看起来
// 毫无区别。
export const CRITIC_CONFIDENCE_TIERS = [
  { min: 0.7, className: 'text-emerald-700', label: '较高' },
  { min: 0.4, className: 'text-amber-700', label: '中等' },
  { min: -Infinity, className: 'text-red-700', label: '偏低' },
];

export const getCriticConfidenceTier = (confidence) => (
  CRITIC_CONFIDENCE_TIERS.find((tier) => confidence >= tier.min)
  || CRITIC_CONFIDENCE_TIERS[CRITIC_CONFIDENCE_TIERS.length - 1]
);

/**
 * critic 超时或解析失败时分数完全来自本地规则，标成「置信度」会误导用户。
 */
export const shouldShowCriticConfidence = (critic) => (
  Boolean(critic)
  && typeof critic.confidence === 'number'
  && critic.critic_source !== 'rules_only'
);

/** 收集元素下所有文本节点，并记录每个节点在拼接串中的起始偏移。 */
const collectTextNodes = (root) => {
  const walker = root.ownerDocument.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const nodes = [];
  let total = 0;
  let node = walker.nextNode();
  while (node) {
    const length = node.nodeValue ? node.nodeValue.length : 0;
    if (length > 0) {
      nodes.push({ node, start: total, end: total + length });
      total += length;
    }
    node = walker.nextNode();
  }
  return { nodes, text: nodes.map((entry) => entry.node.nodeValue).join('') };
};

const findNodeAt = (nodes, offset) => nodes.find(
  (entry) => offset >= entry.start && offset < entry.end,
);

/**
 * 在已渲染的回答里定位并选中一段文字，用于「定位原句」。
 *
 * 用 Range + Selection 而不是注入 <mark>：回答由 React 渲染，直接改 DOM 会
 * 在下次 reconciliation 时被覆盖甚至报错，而原生选区不碰节点结构。
 * 目标文字可能被 <cite> 角标切成多个文本节点，因此按拼接串定位再映射回节点。
 */
export const locateTextInElement = (root, needle) => {
  const target = typeof needle === 'string' ? needle.trim() : '';
  if (!root || !target) return false;

  const { nodes, text } = collectTextNodes(root);
  if (!nodes.length) return false;

  let index = text.indexOf(target);
  if (index < 0) {
    // 渲染时空白可能被折叠，退一步用归一化空白后的形态再找一次。
    const collapsed = target.replace(/\s+/g, ' ');
    index = text.replace(/\s+/g, ' ').indexOf(collapsed);
    if (index < 0) return false;
  }

  const startEntry = findNodeAt(nodes, index);
  const endEntry = findNodeAt(nodes, Math.max(index, index + target.length - 1));
  if (!startEntry || !endEntry) return false;

  const range = root.ownerDocument.createRange();
  range.setStart(startEntry.node, index - startEntry.start);
  range.setEnd(endEntry.node, Math.min(
    endEntry.node.nodeValue.length,
    index + target.length - endEntry.start,
  ));

  const selection = root.ownerDocument.defaultView.getSelection();
  if (selection) {
    selection.removeAllRanges();
    selection.addRange(range);
  }

  const anchor = startEntry.node.parentElement;
  if (anchor && typeof anchor.scrollIntoView === 'function') {
    anchor.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
  return true;
};
