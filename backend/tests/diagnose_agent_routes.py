"""agent 路由稳定性诊断脚本

对代表性问题输出完整链路：
- query_type
- evidence_need
- agent_gate
- agent_mode

默认输出到 Problems/openspec/c_route_diagnosis.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from routes import chat_routes
from services.query_analyzer import get_retrieval_strategy


CASES = [
    {
        "label": "overview",
        "question": "用三句话概括 DiffuLT 论文解决了什么问题、用了什么方法、取得了什么结果。",
        "expected_use_agent": True,
        "expected_signal": "query_type=overview",
    },
    {
        "label": "section_explanation",
        "question": "请详细解释论文 Method 章节的完整流程，包括每个子模块的输入输出和作用。",
        "expected_use_agent": True,
        "expected_signal": "evidence_need=section_explanation",
    },
    {
        "label": "reference_meta",
        "question": "这篇论文的第一作者和通讯作者分别是谁？他们来自哪个机构？",
        "expected_use_agent": True,
        "expected_signal": "evidence_need=reference_meta",
    },
    {
        "label": "numeric_table",
        "question": "表 8 中 ImageNet-LT 的 ResNet-50 结果里，DiffuLT 的 All、Many、Med.、Few 分别是多少？",
        "expected_use_agent": False,
        "expected_signal": "numeric_table should stay off agent",
    },
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_output() -> Path:
    return _project_root() / "Problems" / "openspec" / "c_route_diagnosis.md"


def _diagnose_case(case: dict) -> dict:
    strategy = get_retrieval_strategy(case["question"])
    gate = chat_routes._build_agent_retrieval_gate(
        enable_agent_retrieval=True,
        selected_text=None,
        query_type=strategy["query_type"],
        evidence_need=strategy["evidence_need"],
    )
    annotated = chat_routes._annotate_agent_gate(
        gate,
        use_agent=bool(gate.get("enabled")),
        agent_mode=bool(gate.get("enabled")),
        search_query_passthrough=bool(gate.get("enabled")),
    )
    return {
        "label": case["label"],
        "question": case["question"],
        "query_type": strategy["query_type"],
        "evidence_need": strategy["evidence_need"],
        "top_k": strategy["top_k"],
        "agent_gate": annotated,
        "expected_use_agent": case["expected_use_agent"],
        "expected_signal": case["expected_signal"],
        "stable": annotated["use_agent"] == case["expected_use_agent"] and annotated["consistency_ok"] is True,
    }


def _render_markdown(results: list[dict]) -> str:
    stable = all(item["stable"] for item in results)
    decision = "转向 B" if stable else "继续收紧 C"
    lines = [
        "# C 阶段路由稳定性诊断",
        "",
        "## 诊断范围",
        "",
        "- `overview`",
        "- `section_explanation`",
        "- `reference_meta`",
        "- `numeric_table`",
        "",
        "## 结论",
        "",
        f"- 路由稳定性：`{'稳定' if stable else '不稳定'}`",
        f"- 建议动作：`{decision}`",
        "",
        "理由：",
        "",
        "- 当前诊断只验证 `query_type -> evidence_need -> agent_gate -> agent_mode` 链路是否符合既定策略。",
        "- 若四类代表性问题都符合预期，则无需继续扩 `C` 阶段覆盖范围，优先转向 `B` 阶段处理专用证据通道。",
        "- 若存在误触发或漏触发，则应继续收紧/修复 `C` 阶段路由。",
        "",
        "## 链路结果",
        "",
    ]

    for item in results:
        gate = item["agent_gate"]
        lines.extend(
            [
                f"### {item['label']}",
                "",
                f"- question: `{item['question']}`",
                f"- query_type: `{item['query_type']}`",
                f"- evidence_need: `{json.dumps(item['evidence_need'], ensure_ascii=False)}`",
                f"- agent_gate: `{json.dumps(gate, ensure_ascii=False)}`",
                f"- agent_mode: `{gate.get('agent_mode')}`",
                f"- expected_use_agent: `{item['expected_use_agent']}`",
                f"- expected_signal: `{item['expected_signal']}`",
                f"- stable: `{item['stable']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## 决策建议",
            "",
            f"- 最终建议：`{decision}`",
            "- 如果继续推进：优先进入 `B` 阶段，处理表格 / caption / 数值比较专用证据通道。",
            "- 如果需要保守：仅在后续出现实际误触发证据时，再回头扩展或收紧 `C` 阶段 agent 路由。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="诊断 agent 路由稳定性")
    parser.add_argument("--output", default=str(_default_output()), help="Markdown 输出路径")
    args = parser.parse_args()

    results = [_diagnose_case(case) for case in CASES]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_markdown(results), encoding="utf-8")
    print(output_path)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
