import { describe, it, expect } from 'vitest';
import { partitionEvidenceCitations, resolveEvidenceRef } from '../EvidencePanel.jsx';

describe('resolveEvidenceRef', () => {
  it('优先使用 display_ref', () => {
    expect(resolveEvidenceRef({ ref: 5, display_ref: 1 })).toBe(1);
  });

  it('display_ref 不可用时回退到 ref', () => {
    expect(resolveEvidenceRef({ ref: 7 })).toBe(7);
  });
});

describe('partitionEvidenceCitations', () => {
  it('应按展示编号排序，并将有高亮证据的项目排在前面', () => {
    const citations = [
      { ref: 5, display_ref: 2, source_ref: 5, group_id: 'g5', highlight_text: '第二条' },
      { ref: 9, display_ref: 3, source_ref: 9, group_id: 'g9' },
      { ref: 1, display_ref: 1, source_ref: 1, group_id: 'g1', highlight_text: '第一条' },
    ];

    const { cited, uncited } = partitionEvidenceCitations(citations);

    expect(cited.map((c) => c.display_ref)).toEqual([1, 2]);
    expect(uncited.map((c) => c.display_ref)).toEqual([3]);
  });
});
