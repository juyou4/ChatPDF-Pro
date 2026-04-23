"""引用相关性优化 - 单元测试

Feature: chatpdf-citation-relevance

覆盖融合逻辑的降级行为、边界场景等。
**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 4.2**
"""
import sys
import os

# 将 backend 目录添加到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from routes.chat_routes import (
    _build_citation_context_text,
    _build_fast_overview_context,
    _build_fused_context,
    _build_response_context_segments,
    _build_selected_text_citation,
    _build_selected_text_fallback_citations,
    _extract_inline_citation_refs,
    _align_citations_with_answer,
    _prepare_answer_and_citations_for_display,
    _extract_streaming_final_answer,
    _should_use_fast_overview_context,
    START_ANSWER,
    START_CITATION,
)


class TestBuildFusedContext:
    """_build_fused_context 单元测试"""

    def test_empty_retrieval_context_degradation(self):
        """降级行为：retrieval_context 为空时，融合上下文仅包含 selected_text
        **Validates: Requirements 1.3**
        """
        result = _build_fused_context("选中的文本内容", "", {"page_start": 3, "page_end": 3})

        assert "选中的文本内容" in result
        # 不应包含"相关文档片段"标记
        assert "相关文档片段" not in result

    def test_enable_vector_search_false_only_selected_text(self):
        """enable_vector_search=false 时仅用 selected_text
        模拟该场景：不传入 retrieval_context
        **Validates: Requirements 1.4**
        """
        result = _build_fused_context("用户框选的段落", "", None)

        assert "用户框选的段落" in result
        assert "相关文档片段" not in result

    def test_cross_page_range_label(self):
        """跨页 page_info 显示页码范围
        **Validates: Requirements 1.1, 1.2**
        """
        result = _build_fused_context(
            "跨页文本", "检索结果", {"page_start": 5, "page_end": 8}
        )

        # 应包含跨页页码标注
        assert "（页码: 5-8）" in result
        assert "跨页文本" in result
        assert "检索结果" in result

    def test_same_page_label(self):
        """同页 page_info 显示单页页码
        **Validates: Requirements 1.1**
        """
        result = _build_fused_context(
            "单页文本", "检索结果", {"page_start": 3, "page_end": 3}
        )

        # 同页时应显示单页页码格式
        assert "（页码: 3）" in result
        assert "3-3" not in result

    def test_none_page_info(self):
        """page_info 为 None 时不显示页码标注"""
        result = _build_fused_context("文本内容", "检索结果", None)

        assert "页码" not in result
        assert "文本内容" in result
        assert "检索结果" in result

    def test_empty_page_info(self):
        """page_info 为空字典时不显示页码标注（因为 page_start=0, page_end=0 相等）"""
        result = _build_fused_context("文本内容", "检索结果", {})

        assert "文本内容" in result
        assert "检索结果" in result

    def test_selected_text_before_retrieval_context(self):
        """selected_text 在 retrieval_context 之前
        **Validates: Requirements 1.2**
        """
        result = _build_fused_context(
            "框选内容AAA", "检索内容BBB", {"page_start": 1, "page_end": 1}
        )

        assert result.index("框选内容AAA") < result.index("检索内容BBB")


class TestBuildSelectedTextCitation:
    """_build_selected_text_citation 单元测试"""

    def test_none_page_info_defaults_to_page_1(self):
        """page_info 为 None 时默认页码为 1
        **Validates: Requirements 4.2**
        """
        citation = _build_selected_text_citation("一些文本", None)

        assert citation["page_range"] == [1, 1]
        assert citation["ref"] == 1
        assert citation["group_id"] == "selected-text"
        assert citation["highlight_text"] == "一些文本"

    def test_same_page_start_and_end(self):
        """page_start 和 page_end 相同
        **Validates: Requirements 4.2**
        """
        citation = _build_selected_text_citation(
            "单页文本", {"page_start": 5, "page_end": 5}
        )

        assert citation["page_range"] == [5, 5]

    def test_different_page_start_and_end(self):
        """page_start 和 page_end 不同（跨页）
        **Validates: Requirements 4.2**
        """
        citation = _build_selected_text_citation(
            "跨页文本", {"page_start": 3, "page_end": 7}
        )

        assert citation["page_range"] == [3, 7]

    def test_highlight_text_truncation(self):
        """selected_text 超过 200 字符时 highlight_text 被截断
        **Validates: Requirements 4.2**
        """
        long_text = "这是一段很长的文本内容" * 30  # 远超 200 字符
        citation = _build_selected_text_citation(
            long_text, {"page_start": 1, "page_end": 2}
        )

        assert len(citation["highlight_text"]) <= 200
        # highlight_text 应是 selected_text 前 200 字符的 strip 结果
        assert citation["highlight_text"] == long_text[:200].strip()

    def test_citation_structure_completeness(self):
        """citation 包含所有必需键
        **Validates: Requirements 4.2**
        """
        citation = _build_selected_text_citation(
            "测试文本", {"page_start": 10, "page_end": 12}
        )

        required_keys = {"ref", "group_id", "page_range", "highlight_text"}
        assert required_keys.issubset(set(citation.keys()))

    def test_highlight_text_stripped(self):
        """highlight_text 应去除首尾空白"""
        citation = _build_selected_text_citation(
            "  带空格的文本  ", {"page_start": 1, "page_end": 1}
        )

        assert citation["highlight_text"] == "带空格的文本"

    def test_empty_page_info_dict(self):
        """空字典 page_info 使用默认值"""
        citation = _build_selected_text_citation("文本", {})

        # page_start 默认 1，page_end 默认等于 page_start
        assert citation["page_range"] == [1, 1]


class TestSelectedTextFallbackCitation:
    """selected_text 兜底引用策略测试"""

    def test_short_selected_text_should_not_generate_fallback_citation(self):
        """短 selected_text 不应生成兜底 citation（避免出现无关单一引用）"""
        citations = _build_selected_text_fallback_citations(
            "短标题",
            {"page_start": 1, "page_end": 1},
        )
        assert citations == []

    def test_long_selected_text_should_generate_fallback_citation(self):
        """较长 selected_text 可生成 1 条兜底 citation"""
        citations = _build_selected_text_fallback_citations(
            "这是一个足够长的框选文本片段，用于测试兜底引用生成逻辑是否生效。",
            {"page_start": 3, "page_end": 3},
        )
        assert len(citations) == 1
        assert citations[0]["ref"] == 1
        assert citations[0]["group_id"] == "selected-text"
        assert citations[0]["page_range"] == [3, 3]

    def test_build_fused_context_with_selected_ref(self):
        """selected_ref 传入时，框选文本标题应显式带引用编号"""
        fused = _build_fused_context(
            selected_text="框选内容",
            retrieval_context="",
            selected_page_info={"page_start": 2, "page_end": 2},
            selected_ref=1,
        )

        assert "[1]用户选中的文本（页码: 2）" in fused
        assert "框选内容" in fused


class TestFastOverviewContext:
    def test_should_use_fast_overview_context_only_for_overview_queries(self):
        assert _should_use_fast_overview_context(
            "overview",
            enable_vector_search=True,
            selected_text=None,
        ) is True
        assert _should_use_fast_overview_context(
            "specific",
            enable_vector_search=True,
            selected_text=None,
        ) is False
        assert _should_use_fast_overview_context(
            "overview",
            enable_vector_search=False,
            selected_text=None,
        ) is False
        assert _should_use_fast_overview_context(
            "overview",
            enable_vector_search=True,
            selected_text="框选文本",
        ) is False

    def test_build_fast_overview_context_samples_front_middle_and_back_pages(self):
        pages = [
            {"text": f"第{i}页内容 " + ("A" * 400)}
            for i in range(1, 13)
        ]

        context = _build_fast_overview_context(pages, "全文内容")

        assert "[第1页]" in context
        assert "[第2页]" in context
        assert "[第11页]" in context
        assert "[第12页]" in context
        assert any(tag in context for tag in ("[第3页]", "[第5页]", "[第7页]", "[第8页]", "[第10页]"))


class TestCitationAlignment:
    """正文引文与来源列表对齐测试"""

    def test_extract_inline_refs_supports_half_and_full_width(self):
        answer = "结论A[1]，结论B【2】，补充[1]。"
        refs = _extract_inline_citation_refs(answer)
        assert refs == [1, 2]

    def test_align_citations_keeps_only_referenced_items(self):
        answer = "根据文档可知[2][1]。"
        citations = [
            {"ref": 1, "group_id": "group-1", "page_range": [3, 3], "highlight_text": "A"},
            {"ref": 2, "group_id": "group-2", "page_range": [7, 7], "highlight_text": "B"},
            {"ref": 3, "group_id": "group-3", "page_range": [9, 9], "highlight_text": "C"},
        ]

        aligned = _align_citations_with_answer(answer, citations)
        assert [c["ref"] for c in aligned] == [2, 1]
        assert all(c["ref"] != 3 for c in aligned)

    def test_align_citations_keeps_original_when_no_inline_ref(self):
        answer = "这是一个没有编号引用的回答。"
        citations = [{"ref": 1, "group_id": "group-1", "page_range": [1, 1], "highlight_text": "A"}]
        aligned = _align_citations_with_answer(answer, citations)
        assert len(aligned) == 1
        assert aligned[0]["ref"] == 1

    def test_align_citations_fallback_when_inline_refs_unmapped(self):
        answer = "结论见[99]。"
        citations = [
            {"ref": 1, "group_id": "group-1", "page_range": [1, 1], "highlight_text": "A"},
            {"ref": 2, "group_id": "group-2", "page_range": [2, 2], "highlight_text": "B"},
        ]
        aligned = _align_citations_with_answer(answer, citations)
        assert aligned == []


class TestCitationDisplayPreparation:

    def test_prepare_answer_repairs_and_remaps_display_refs(self):
        answer = "结论见[ID: 5]，补充见ref 2。"
        citations = [
            {"ref": 2, "group_id": "group-2", "page_range": [2, 2], "highlight_text": "B"},
            {"ref": 5, "group_id": "group-5", "page_range": [5, 5], "highlight_text": "E"},
        ]

        rewritten, projected = _prepare_answer_and_citations_for_display(answer, citations)

        assert rewritten == "结论见[1]，补充见[2]。"
        assert [c["ref"] for c in projected] == [1, 2]
        assert [c["source_ref"] for c in projected] == [5, 2]
        assert [c["display_ref"] for c in projected] == [1, 2]

    def test_prepare_answer_injects_inline_refs_when_model_omits_them(self):
        answer = (
            "固定大小分块是一种传统方法，它按预设大小切分文档而不考虑语义内容。\n\n"
            "语义分块通过理解文本含义来切分文档，能保持上下文完整性。"
        )
        citations = [
            {"ref": 5, "group_id": "group-5", "page_range": [1, 1], "highlight_text": "固定大小分块 传统方法 预设大小 切分文档"},
            {"ref": 8, "group_id": "group-8", "page_range": [2, 2], "highlight_text": "语义分块 理解文本含义 上下文完整性"},
        ]

        rewritten, projected = _prepare_answer_and_citations_for_display(answer, citations)

        assert rewritten.count("[1]") == 1
        assert rewritten.count("[2]") == 1
        assert [c["ref"] for c in projected] == [1, 2]
        assert [c["source_ref"] for c in projected] == [5, 8]

    def test_prepare_answer_reassigns_single_repeated_ref_by_paragraph(self):
        answer = "语义引导用于保持类别一致。[5]\n\n3D 渲染用于生成可打印伪装。[5]"
        citations = [
            {"ref": 5, "group_id": "group-5", "page_range": [3, 3], "highlight_text": "语义 引导 类别 一致"},
            {"ref": 9, "group_id": "group-9", "page_range": [7, 7], "highlight_text": "3D 渲染 可打印 伪装"},
        ]

        rewritten, projected = _prepare_answer_and_citations_for_display(answer, citations)

        assert "[1]" in rewritten
        assert "[2]" in rewritten
        assert [c["source_ref"] for c in projected] == [5, 9]

    def test_prepare_answer_prunes_weak_citations_for_numeric_table(self):
        answer = "作者来自东京大学并主导了实验实现[1]。"
        citations = [
            {
                "ref": 1,
                "group_id": "table-8",
                "page_range": [9, 9],
                "highlight_text": "DiffuLT ResNet-50 All 56.4 Many 63.3 Med. 55.6 Few 39.4",
            }
        ]

        guard = {}
        rewritten, projected = _prepare_answer_and_citations_for_display(
            answer,
            citations,
            evidence_need=["numeric_table"],
            answer_guard=guard,
        )

        assert rewritten == "作者来自东京大学并主导了实验实现。"
        assert projected == []
        assert guard["strict_mode"] is True
        assert guard["checked_sentence_count"] == 1
        assert guard["unsupported_sentence_count"] == 1
        assert guard["removed_ref_count"] == 1

    def test_prepare_answer_keeps_supported_citations_for_numeric_table(self):
        answer = "DiffuLT 在 ResNet-50 上的 All 为 56.4，Many 为 63.3[1]。"
        citations = [
            {
                "ref": 1,
                "group_id": "table-8",
                "page_range": [9, 9],
                "highlight_text": "DiffuLT ResNet-50 All 56.4 Many 63.3 Med. 55.6 Few 39.4",
            }
        ]

        guard = {}
        rewritten, projected = _prepare_answer_and_citations_for_display(
            answer,
            citations,
            evidence_need=["numeric_table"],
            answer_guard=guard,
        )

        assert rewritten == "DiffuLT 在 ResNet-50 上的 All 为 56.4，Many 为 63.3[1]。"
        assert len(projected) == 1
        assert projected[0]["source_ref"] == 1
        assert guard["strict_mode"] is True
        assert guard["checked_sentence_count"] == 1
        assert guard["unsupported_sentence_count"] == 0
        assert guard["removed_ref_count"] == 0

    def test_prepare_answer_keeps_numeric_table_citation_with_evidence_units(self):
        answer = "CBDM(τ=1) 的 FID 是 5.86，准确率是 46.6[1]。"
        citations = [
            {
                "ref": 1,
                "group_id": "table-1",
                "page_range": [4, 4],
                "highlight_text": "Table 1 main results",
                "source_text": "[Structured Table Bundle] Table 1 main results",
                "display_text": "[Structured Table Bundle] Table 1 main results",
                "evidence_units": [
                    {
                        "evidence_unit_type": "table_row",
                        "table_caption": "Table 1: generation results",
                        "table_header": "Method | FID | Acc",
                        "content": "CBDM(τ=1) | 5.86 | 46.6",
                        "cell_evidence_units": [
                            {"content": "CBDM(τ=1)"},
                            {"content": "5.86"},
                            {"content": "46.6"},
                        ],
                    }
                ],
            }
        ]

        guard = {}
        rewritten, projected = _prepare_answer_and_citations_for_display(
            answer,
            citations,
            evidence_need=["numeric_table"],
            answer_guard=guard,
        )

        assert rewritten == "CBDM(τ=1) 的 FID 是 5.86，准确率是 46.6[1]。"
        assert len(projected) == 1
        assert projected[0]["source_ref"] == 1
        assert guard["unsupported_sentence_count"] == 0
        assert guard["removed_ref_count"] == 0

    def test_prepare_answer_keeps_numeric_table_same_bundle_exact_rows_for_context(self):
        answer = (
            "On ResNet-50 All, DiffuLT is 9.1 points above cRT, "
            "1.5 above RIDE(3 experts), and 2.3 above ADRW[1][2]."
        )
        citations = [
            {
                "ref": 1,
                "group_id": "table-8",
                "table_id": "Table 8",
                "page_range": [9, 9],
                "highlight_text": "DiffuLT 56.4",
                "display_text": "DiffuLT 56.4",
                "chunk_type": "table_row",
                "numeric_table_exact_context_row_text": "DiffuLT 56.4 63.3 55.6 39.4",
            },
            {
                "ref": 2,
                "group_id": "table-8",
                "table_id": "Table 8",
                "page_range": [9, 9],
                "highlight_text": "cRT 47.3",
                "display_text": "cRT 47.3",
                "chunk_type": "table_row",
                "numeric_table_exact_context_row_text": "cRT 47.3 58.8 44.0 26.1",
            },
            {
                "ref": 3,
                "group_id": "table-8",
                "table_id": "Table 8",
                "page_range": [9, 9],
                "highlight_text": "RIDE(3 experts) 54.9",
                "display_text": "RIDE(3 experts) 54.9",
                "chunk_type": "table_row",
                "numeric_table_exact_context_row_text": "RIDE(3 experts) 54.9 66.2 51.7 34.9",
            },
            {
                "ref": 4,
                "group_id": "table-8",
                "table_id": "Table 8",
                "page_range": [9, 9],
                "highlight_text": "ADRW 54.1",
                "display_text": "ADRW 54.1",
                "chunk_type": "table_row",
                "numeric_table_exact_context_row_text": "ADRW 54.1 62.9 52.6 37.1",
            },
        ]

        rewritten, projected = _prepare_answer_and_citations_for_display(
            answer,
            citations,
            evidence_need=["numeric_table"],
            answer_guard={},
        )

        assert "2.3 above ADRW" in rewritten
        assert "[1]" in rewritten
        assert sorted(item["source_ref"] for item in projected) == [1, 2, 3, 4]

    def test_prepare_answer_strict_gate_skips_broad_table_summary_for_second_best_query(self):
        answer = "Table 8 中 second-best 的方法是 ADRW[1]。"
        citations = [
            {
                "ref": 1,
                "group_id": "table-8",
                "table_id": "Table 8",
                "page_range": [9, 9],
                "chunk_type": "table_row",
                "table_caption": "Table 8: long-tail recognition",
                "table_header": "Method | All | Many | Med. | Few",
                "display_text": "ADRW | ResNet-50 | 54.1 | 62.9 | 52.6 | 37.1",
                "highlight_text": "ADRW | ResNet-50 | 54.1 | 62.9 | 52.6 | 37.1",
                "numeric_table_exact_context_row_text": "ADRW | ResNet-50 | 54.1 | 62.9 | 52.6 | 37.1",
            },
            {
                "ref": 2,
                "group_id": "table-8",
                "table_id": "Table 8",
                "page_range": [9, 9],
                "chunk_type": "table",
                "table_caption": "Table 8: long-tail recognition",
                "table_header": "Method | All | Many | Med. | Few",
                "source_text": (
                    "Table 8: long-tail recognition\n"
                    "Method | All | Many | Med. | Few\n"
                    "DiffuLT | ResNet-50 | 56.4 | 63.3 | 55.6 | 39.4\n"
                    "cRT | ResNet-50 | 47.3 | 58.8 | 44.0 | 26.1\n"
                    "RIDE(3 experts) | ResNet-50 | 54.9 | 66.2 | 51.7 | 34.9\n"
                    "ADRW | ResNet-50 | 54.1 | 62.9 | 52.6 | 37.1\n"
                    "OCR junk row | ??? | ??? | ??? | ???"
                ),
                "display_text": "Table 8 full summary",
                "highlight_text": "Table 8 full summary",
            },
        ]

        rewritten, projected = _prepare_answer_and_citations_for_display(
            answer,
            citations,
            evidence_need=["numeric_table"],
            answer_guard={},
            query="Table 8 中 second-best 的方法是什么？它在 Few 上的数值是多少？",
        )

        assert "ADRW" in rewritten
        assert [item["source_ref"] for item in projected] == [1]

    def test_prepare_answer_cost_query_supplements_cost_anchor_citation(self):
        answer = "文中未明确记载额外开销和训练时间[1]。"
        citations = [
            {
                "ref": 1,
                "group_id": "title-page",
                "page_range": [1, 1],
                "source_text": "DiffuLT: Long-Tail Recognition with Diffusion Models.",
                "display_text": "DiffuLT title page",
                "highlight_text": "DiffuLT title page",
            },
            {
                "ref": 2,
                "group_id": "appendix-b",
                "page_range": [12, 12],
                "chunk_type": "text",
                "source_text": (
                    "Our method adds no extra overhead. "
                    "Training time is about 24 hours on CIFAR100-LT and approximately six days on ImageNet-LT."
                ),
                "display_text": "no extra overhead | 24 hours | six days",
                "highlight_text": "no extra overhead | 24 hours | six days",
            },
        ]

        rewritten, projected = _prepare_answer_and_citations_for_display(
            answer,
            citations,
            evidence_need=["numeric_table"],
            answer_guard={},
            query="这篇论文的额外开销、训练时间和持续时间分别是多少？",
        )

        assert "额外开销" in rewritten
        assert 2 in [item["source_ref"] for item in projected]

    def test_prepare_answer_rewrites_reference_trap_sentence_to_conservative_text(self):
        answer = "第一作者是东京大学教授并提出了这套方法[1]。"
        citations = [
            {
                "ref": 1,
                "group_id": "references",
                "page_range": [15, 15],
                "highlight_text": "Zhou et al. Diffusion models for image generation.",
            }
        ]

        guard = {}
        rewritten, projected = _prepare_answer_and_citations_for_display(
            answer,
            citations,
            evidence_need=["reference_trap"],
            answer_guard=guard,
        )

        assert rewritten == "根据当前检索证据，无法确认该信息，文档未明确说明。"
        assert projected == []
        assert guard["unsupported_sentence_count"] == 1
        assert guard["removed_ref_count"] == 1
        assert guard["rewritten_sentence_count"] == 1

    def test_prepare_answer_rewrites_reference_meta_sentence_to_conservative_text(self):
        answer = "通讯作者来自清华大学并在附录提供了邮箱地址[1]。"
        citations = [
            {
                "ref": 1,
                "group_id": "author-bio",
                "page_range": [1, 1],
                "highlight_text": "This paper introduces a diffusion-based long-tail learner.",
            }
        ]

        guard = {}
        rewritten, projected = _prepare_answer_and_citations_for_display(
            answer,
            citations,
            evidence_need=["reference_meta"],
            answer_guard=guard,
        )

        assert rewritten == "根据当前检索证据，文档未明确说明该引用元信息。"
        assert projected == []
        assert guard["unsupported_sentence_count"] == 1
        assert guard["removed_ref_count"] == 1
        assert guard["rewritten_sentence_count"] == 1


class TestNumericTableCitationContextAssembly:

    @pytest.mark.parametrize(
        "caption, header, focused_row, stale_row",
        [
            (
                "Table 1: generation results",
                "Method | FID | Acc",
                "CBDM(τ=1) | 5.86 | 46.6",
                "CBDM(τ=1) | 44.8 | 36.3",
            ),
            (
                "Table 8: long-tail recognition",
                "Method | All | Many | Med. | Few",
                "DiffuLT | ResNet-50 | 56.4 | 63.3 | 55.6 | 39.4",
                "DiffuLT | ResNet-50 | 44.8 | 36.3 | 55.6 | 39.4",
            ),
            (
                "Cost-anchor summary",
                "Method | Extra overhead | Duration",
                "No extra overhead | 24 hours | 6 days",
                "Shared header only | 48 hours | 12 days",
            ),
        ],
    )
    def test_build_citation_context_text_prefers_focused_numeric_table_context(
        self,
        caption,
        header,
        focused_row,
        stale_row,
    ):
        citation = {
            "ref": 1,
            "group_id": "table-test",
            "page_range": [4, 4],
            "table_id": caption,
            "chunk_type": "table_row",
            "table_caption": caption,
            "table_header": header,
            "context_segment_text": f"{caption}\n{header}\n{focused_row}",
            "source_text": f"{caption}\n{header}\n{focused_row}",
            "display_text": focused_row,
            "highlight_text": focused_row,
            "numeric_table_exact_context_row_text": stale_row,
            "table_row_boundary_text": stale_row,
            "table_row_raw_text": stale_row,
        }

        text = _build_citation_context_text(citation)

        assert caption in text
        assert header in text
        assert focused_row in text
        assert stale_row not in text

    def test_build_citation_context_text_drops_broad_raw_table_payload_when_exact_row_exists(self):
        raw_table = (
            "Table 8: long-tail recognition\n"
            "Method | All | Many | Med. | Few\n"
            "DiffuLT | ResNet-50 | 56.4 | 63.3 | 55.6 | 39.4\n"
            "cRT | ResNet-50 | 47.3 | 58.8 | 44.0 | 26.1\n"
            "RIDE(3 experts) | ResNet-50 | 54.9 | 66.2 | 51.7 | 34.9\n"
            "ADRW | ResNet-50 | 54.1 | 62.9 | 52.6 | 37.1\n"
            "OCR junk row | ??? | ??? | ??? | ???"
        )
        citation = {
            "ref": 1,
            "group_id": "table-8",
            "page_range": [9, 9],
            "table_id": "Table 8",
            "chunk_type": "table_row",
            "table_caption": "Table 8: long-tail recognition",
            "table_header": "Method | All | Many | Med. | Few",
            "context_segment_text": raw_table,
            "source_text": raw_table,
            "display_text": "ADRW | ResNet-50 | 54.1 | 62.9 | 52.6 | 37.1",
            "highlight_text": "ADRW | ResNet-50 | 54.1 | 62.9 | 52.6 | 37.1",
            "numeric_table_exact_context_row_text": "ADRW | ResNet-50 | 54.1 | 62.9 | 52.6 | 37.1",
            "table_row_boundary_text": "ADRW | ResNet-50 | 54.1 | 62.9 | 52.6 | 37.1",
            "table_row_raw_text": "ADRW | ResNet-50 | 54.1 | 62.9 | 52.6 | 37.1",
        }

        text = _build_citation_context_text(citation)

        assert "ADRW | ResNet-50 | 54.1 | 62.9 | 52.6 | 37.1" in text
        assert "OCR junk row" not in text

    def test_build_response_context_segments_numeric_table_prefers_focused_text_over_raw_segments(self):
        citation = {
            "ref": 1,
            "group_id": "table-8",
            "page_range": [9, 9],
            "table_id": "Table 8",
            "chunk_type": "table_row",
            "table_caption": "Table 8: long-tail recognition",
            "table_header": "Method | All | Many | Med. | Few",
            "context_segment_text": "Table 8: long-tail recognition\nMethod | All | Many | Med. | Few\nDiffuLT | ResNet-50 | 56.4 | 63.3 | 55.6 | 39.4",
            "source_text": "Table 8: long-tail recognition\nMethod | All | Many | Med. | Few\nDiffuLT | ResNet-50 | 56.4 | 63.3 | 55.6 | 39.4",
            "display_text": "DiffuLT | ResNet-50 | 56.4 | 63.3 | 55.6 | 39.4",
            "highlight_text": "DiffuLT | ResNet-50 | 56.4 | 63.3 | 55.6 | 39.4",
            "numeric_table_exact_context_row_text": "DiffuLT | ResNet-50 | 44.8 | 36.3 | 55.6 | 39.4",
            "table_row_boundary_text": "DiffuLT | ResNet-50 | 44.8 | 36.3 | 55.6 | 39.4",
            "table_row_raw_text": "DiffuLT | ResNet-50 | 44.8 | 36.3 | 55.6 | 39.4",
        }

        segments = _build_response_context_segments(
            {
                "evidence_need": ["numeric_table"],
                "citations": [citation],
                "_context_segments": [
                    {
                        "ref": 1,
                        "text": "DiffuLT | ResNet-50 | 44.8 | 36.3 | 55.6 | 39.4",
                        "page_range": [9, 9],
                        "group_id": "table-8",
                    }
                ],
            }
        )

        assert len(segments) == 1
        assert "56.4" in segments[0]["text"]
        assert "63.3" in segments[0]["text"]
        assert "44.8" not in segments[0]["text"]
        assert "36.3" not in segments[0]["text"]


class TestStreamingFinalAnswerExtraction:

    def test_extract_streaming_final_answer_strips_partial_citation_marker(self):
        full_output = f"{START_ANSWER}\n第一段回答。\n{START_CITATION[:4]}"

        answer = _extract_streaming_final_answer(full_output)

        assert answer == "第一段回答。"

    def test_extract_streaming_final_answer_without_final_answer_marker(self):
        full_output = "第一段回答。\n第二段回答。\nCITATION LIST\nCITATION【1】"

        answer = _extract_streaming_final_answer(full_output)

        assert answer == "第一段回答。\n第二段回答。"

    def test_extract_streaming_final_answer_handles_cross_chunk_partial_marker(self):
        full_output = f"{START_ANSWER}\n第一段回答[5]。\n\nCITATIO"

        answer = _extract_streaming_final_answer(full_output)

        assert answer == "第一段回答[5]。"
