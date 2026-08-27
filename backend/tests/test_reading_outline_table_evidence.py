"""精读表格证据：正文引用决定归属，长表格按整行裁剪。"""
from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.reading_outline_service import (  # noqa: E402
    MAX_SECTION_BLOCK_TEXT,
    MAX_SECTION_TABLE_BUNDLES,
    _bind_table_bundles_to_sections,
    _build_reading_section_skeleton,
    _flatten_skeleton_nodes,
    _prepare_section_payloads,
    _prompt_block_text,
    _section_coverage_ledger,
    _table_mention_labels,
)


def _bundle(*, table_id: str, caption: str, page: int, rows: int = 3) -> dict[str, Any]:
    return {
        "bundle_id": f"bundle:{table_id}",
        "table_id": table_id,
        "table_caption": caption,
        "pages": [page],
        "evidence_units": [
            {
                "evidence_unit_id": f"{table_id}::row::{index}",
                "evidence_unit_type": "table_row",
                "is_header_row": index == 0,
                "page": page,
                "row_text": (
                    "System | Acc" if index == 0 else f"Baseline{index} | {70 + index}.5"
                ),
            }
            for index in range(rows + 1)
        ],
    }


def _block_index(sections: list[dict[str, Any]]) -> dict[str, Any]:
    """按 [{id, title, page, blocks:[(type, text)]}] 拼一份最小解析索引。"""
    blocks: list[dict[str, Any]] = []
    outline: list[dict[str, Any]] = []
    for section in sections:
        section_id = str(section["id"])
        page = int(section["page"])
        heading_id = f"h_{section_id}"
        blocks.append({
            "block_id": heading_id,
            "type": "heading",
            "text": section["title"],
            "section_id": section_id,
            "page": page,
        })
        outline.append({
            "section_id": section_id,
            "title": section["title"],
            "level": 1,
            "page": page,
            "first_block": heading_id,
            "source": "pdf_outline",
        })
        for index, (block_type, text) in enumerate(section.get("blocks") or []):
            blocks.append({
                "block_id": f"b_{section_id}_{index}",
                "type": block_type,
                "text": text,
                "section_id": section_id,
                "page": page,
            })
    pages: dict[int, list[dict[str, Any]]] = {}
    for block in blocks:
        pages.setdefault(int(block["page"]), []).append(block)
    return {
        "source": "pdf",
        "outline": outline,
        "pages": [
            {"page": page, "blocks": page_blocks}
            for page, page_blocks in sorted(pages.items())
        ],
    }


_BODY = (
    "本节给出完整的方法细节与实验设置说明，篇幅足够长以便被当作一个可用的章节证据块使用。"
)


def _floating_table_index() -> dict[str, Any]:
    """Table 1 的 caption 浮到方法章，真正引用它的是结果章。"""
    return _block_index([
        {"id": "s1", "title": "1. Introduction", "page": 1, "blocks": [("paragraph", _BODY)]},
        {
            "id": "s2",
            "title": "2. Method",
            "page": 2,
            "blocks": [
                ("paragraph", _BODY),
                ("caption", "Table 1: Main results on the GLUE benchmark."),
            ],
        },
        {
            "id": "s3",
            "title": "3. Results",
            "page": 3,
            "blocks": [
                ("paragraph", f"As shown in Table 1, our method wins. {_BODY}"),
            ],
        },
    ])


def _bind(block_index: dict[str, Any], bundles: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    nodes = _flatten_skeleton_nodes(_build_reading_section_skeleton(block_index))
    return _bind_table_bundles_to_sections(
        nodes=nodes,
        block_index=block_index,
        structured_table_bundles=bundles,
    )


def test_citing_section_receives_floating_table():
    """排版把表格浮到方法章，引用它的结果章仍然必须拿到这张表。"""
    block_index = _floating_table_index()
    bundles = [_bundle(table_id="t1", caption="Table 1: Main results on the GLUE benchmark.", page=2)]

    binding = _bind(block_index, bundles)

    assert [bundle["table_id"] for bundle in binding.get("s3") or []] == ["t1"]


def test_caption_anchor_section_keeps_table_as_fallback():
    """引用通道是新增而非替换：caption 所在章节仍然保留这张表。"""
    binding = _bind(
        _floating_table_index(),
        [_bundle(table_id="t1", caption="Table 1: Main results on the GLUE benchmark.", page=2)],
    )

    assert [bundle["table_id"] for bundle in binding.get("s2") or []] == ["t1"]


def test_uncited_table_stays_with_its_caption_section():
    """没有任何章节引用时，仍然按 caption 位置归属。"""
    block_index = _block_index([
        {
            "id": "s1",
            "title": "1. Experiments",
            "page": 1,
            "blocks": [
                ("paragraph", _BODY),
                ("caption", "Table 1: Dataset statistics."),
            ],
        },
        {"id": "s2", "title": "2. Conclusion", "page": 2, "blocks": [("paragraph", _BODY)]},
    ])

    binding = _bind(block_index, [_bundle(table_id="t1", caption="Table 1: Dataset statistics.", page=1)])

    assert [bundle["table_id"] for bundle in binding.get("s1") or []] == ["t1"]
    assert "s2" not in binding


def test_cited_table_outranks_anchored_tables_under_budget():
    """章节表格预算有限时，正文引用的那张表不能被同页浮动的表挤掉。"""
    anchored = [
        ("caption", f"Table {index}: Auxiliary statistics {index}.")
        for index in range(2, 2 + MAX_SECTION_TABLE_BUNDLES)
    ]
    block_index = _block_index([
        {
            "id": "s1",
            "title": "1. Setup",
            "page": 1,
            "blocks": [
                ("paragraph", f"Our headline number appears in Table 1. {_BODY}"),
                *anchored,
            ],
        },
    ])
    bundles = [
        _bundle(table_id="t1", caption="Table 1: Main results.", page=9),
        *(
            _bundle(table_id=f"t{index}", caption=f"Table {index}: Auxiliary statistics {index}.", page=1)
            for index in range(2, 2 + MAX_SECTION_TABLE_BUNDLES)
        ),
    ]

    binding = _bind(block_index, bundles)
    delivered = [bundle["table_id"] for bundle in (binding.get("s1") or [])[:MAX_SECTION_TABLE_BUNDLES]]

    assert delivered[0] == "t1"


def test_section_payload_carries_cited_table_rows():
    """端到端：结果章的 payload 里能看到被引用表格的行级证据。"""
    block_index = _floating_table_index()
    skeleton = _build_reading_section_skeleton(block_index)
    bundles = [_bundle(table_id="t1", caption="Table 1: Main results on the GLUE benchmark.", page=2)]

    payloads = _prepare_section_payloads(skeleton, block_index, bundles)
    evidence = payloads["s3"]["table_evidence"]

    assert [item["table_id"] for item in evidence] == ["t1"]
    assert evidence[0]["row_count"] == 3
    assert payloads["s3"]["allowed_table_evidence_unit_ids"]


def test_table_mention_labels_cover_enumerations_and_ranges():
    assert _table_mention_labels("As shown in Table 1, ...") == {"1"}
    assert _table_mention_labels("Tables 1 and 3 report ...") == {"1", "3"}
    assert _table_mention_labels("See Tables 2-4 for details.") == {"2", "3", "4"}
    assert _table_mention_labels("详见表 5 与表 6。") == {"5", "6"}
    assert _table_mention_labels("Table A.1 lists hyperparameters.") == {"a.1"}


def test_table_mention_labels_ignore_english_words():
    """'Table of contents' 不能被读成表号 O，否则会把无关章节拉进来。"""
    assert _table_mention_labels("Table of contents follows.") == set()
    assert _table_mention_labels("The table shows results.") == set()


def _wide_table_markdown(rows: int) -> str:
    header = "| System | MNLI | QQP | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE | Avg |"
    separator = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    body = [
        f"| Baseline-{index:02d} | 80.{index} | 71.{index} | 90.{index} | 93.{index} | "
        f"52.{index} | 85.{index} | 87.{index} | 67.{index} | 79.{index} |"
        for index in range(rows - 1)
    ]
    body.append(
        "| OursMethod (ours) | 84.6 | 71.6 | 90.4 | 93.1 | 51.1 | 83.7 | 87.3 | 70.0 | 79.4 |"
    )
    return "\n".join([header, separator, *body])


def test_long_table_block_keeps_whole_rows_including_the_last():
    """结果表的末行通常就是本文方法，按字符硬切会只留下基线。"""
    text = _wide_table_markdown(rows=14)
    assert len(text) > MAX_SECTION_BLOCK_TEXT
    block = {"block_id": "p6_b0", "type": "table", "page": 6, "text": text}

    rendered = _prompt_block_text(block)
    lines = [line for line in rendered.splitlines() if line.strip()]

    assert len(rendered) <= MAX_SECTION_BLOCK_TEXT
    assert lines[0] == text.splitlines()[0]
    assert lines[1] == text.splitlines()[1]
    assert "OursMethod (ours)" in rendered
    assert all(
        line.startswith("|") and line.endswith("|")
        for line in lines
    )
    assert "未展示" in lines[-1]


def test_short_table_block_is_untouched():
    text = _wide_table_markdown(rows=2)
    block = {"block_id": "p1_b0", "type": "table", "page": 1, "text": text}

    assert _prompt_block_text(block) == text


def test_prose_block_still_truncates_by_characters():
    text = "结论。" * 600
    block = {"block_id": "p1_b1", "type": "paragraph", "page": 1, "text": text}

    rendered = _prompt_block_text(block)

    assert rendered.endswith("...")
    assert len(rendered) <= MAX_SECTION_BLOCK_TEXT + 3


def test_coverage_ledger_counts_abridged_table_chars():
    """账本必须记模型真正看到的字符数，而不是固定上限。"""
    text = _wide_table_markdown(rows=14)
    block = {"block_id": "p6_b0", "type": "table", "page": 6, "text": text}

    ledger = _section_coverage_ledger([block], [block])

    assert ledger["truncated_block_ids"] == ["p6_b0"]
    assert ledger["selected_char_count"] == len(_prompt_block_text(block))
    assert ledger["selected_source_char_count"] == len(text.strip())
    assert ledger["input_coverage_complete"] is False
