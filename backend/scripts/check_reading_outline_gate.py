"""Check persisted reading-outline fixtures against deterministic gates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# The script is intentionally runnable from either the repository root or the
# backend directory, just like the other release-gate helpers.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.reading_outline_quality_gate import evaluate_reading_outline_quality


DEFAULT_THRESHOLDS = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "reading_outline_gate.v1.json"
)


def evaluate_gate(result: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one outline or a named ``outlines`` artifact collection."""

    entries: list[tuple[str, dict[str, Any]]] = []
    if isinstance(result.get("outlines"), list):
        for index, entry in enumerate(result["outlines"], start=1):
            if isinstance(entry, dict):
                outline = entry.get("outline") if isinstance(entry.get("outline"), dict) else entry
                entries.append((str(entry.get("id") or index), outline))
    else:
        entries.append((str(result.get("doc_id") or "outline"), result))

    reports = [
        {"id": identifier, **evaluate_reading_outline_quality(outline, thresholds)}
        for identifier, outline in entries
    ]
    failures = [report for report in reports if report["status"] != "pass"]
    return {
        "status": "pass" if not failures else "fail",
        "outline_count": len(reports),
        "failures": failures,
        "reports": reports,
        "gate_version": str(thresholds.get("version") or "reading-outline-gate-v1"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 ChatPDF reading-outline 质量门禁")
    parser.add_argument("--results", required=True, help="单个 reading outline 或 outlines 集合 JSON")
    parser.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS), help="门槛 JSON")
    args = parser.parse_args()

    result: dict[str, Any] = json.loads(Path(args.results).read_text(encoding="utf-8"))
    thresholds: dict[str, Any] = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    report = evaluate_gate(result, thresholds)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
