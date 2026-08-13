import { describe, it, expect } from 'vitest';
import { getFullDocumentSummaryCoverage } from '../answerCriticUtils.js';

// 后端 full_document_summary 覆盖胶囊的展示合同。
// 这里锁定三件事：主题式/章节式两种投影的提示语、healthy 时不出现
// 「建议核对」告警、needs_review 时把具体缺口写进悬停说明。
const baseCoverage = (overrides = {}) => ({
  body_expected: 7,
  body_summarized: 7,
  appendix_expected: 0,
  appendix_summarized: 0,
  complete: true,
  presentation_mode: 'thematic',
  visible_section_count: 0,
  structural_section_count: 7,
  semantic_quality_status: 'healthy',
  semantic_quality: {},
  ...overrides,
});

describe('getFullDocumentSummaryCoverage', () => {
  it('没有全文总结元数据时返回 null', () => {
    expect(getFullDocumentSummaryCoverage(null)).toBeNull();
    expect(getFullDocumentSummaryCoverage({})).toBeNull();
    expect(getFullDocumentSummaryCoverage({ full_document_summary: { body_expected: 0 } })).toBeNull();
  });

  it('健康的主题式总结不出现核对告警', () => {
    const result = getFullDocumentSummaryCoverage({ full_document_summary: baseCoverage() });

    expect(result.complete).toBe(true);
    expect(result.text).toBe('全文覆盖 7/7');
    expect(result.title).toContain('已按主题综合，不逐章复述。');
    expect(result.title).not.toContain('建议核对');
  });

  it('章节展开模式提示已展开的章节数', () => {
    const result = getFullDocumentSummaryCoverage({
      full_document_summary: baseCoverage({
        presentation_mode: 'section_detail',
        visible_section_count: 18,
      }),
    });

    expect(result.title).toContain('已展开 18 节章节导览。');
  });

  it('needs_review 时把具体缺口写进悬停说明', () => {
    const result = getFullDocumentSummaryCoverage({
      full_document_summary: baseCoverage({
        semantic_quality_status: 'needs_review',
        semantic_quality: {
          missing_slots: ['data_or_setup'],
          landmark_expected_claim_count: 4,
          landmark_covered_claim_count: 2,
          themes_restating_sections: ['theme_innovation'],
          themes_without_evidence: ['theme_conclusion'],
        },
      }),
    });

    expect(result.title).toContain('缺data_or_setup');
    expect(result.title).toContain('关键结论证据 2/4');
    expect(result.title).toContain('1 个主题仅复述章节');
    expect(result.title).toContain('1 个主题缺证据');
    expect(result.title).toContain('建议核对关键结论');
  });

  it('部分覆盖时明确标记不完整', () => {
    const result = getFullDocumentSummaryCoverage({
      full_document_summary: baseCoverage({
        body_summarized: 5,
        complete: false,
      }),
    });

    expect(result.complete).toBe(false);
    expect(result.text).toBe('全文覆盖 5/7');
    expect(result.title).toContain('部分章节尚未形成可验证提要。');
  });
});
