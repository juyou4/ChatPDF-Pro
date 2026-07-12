"""Validate the checked-in PDF/VLM benchmark corpus before an external run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(Path(__file__).parents[1] / "benchmarks" / "pdf_vlm_manifest.json"))
    parser.add_argument("--require-vlm", action="store_true", help="Fail when no VLM credentials are configured.")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for case in manifest.get("cases", []):
        path = (manifest_path.parent / case["path"]).resolve()
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"{case.get('id')}: missing {path}")
            continue
        print(f"READY {case.get('id')} {path} tags={','.join(case.get('tags', []))}")
    if args.require_vlm:
        import os
        if not (os.getenv("CHATPDF_VLM_API_KEY") or os.getenv("OPENAI_API_KEY")):
            failures.append("VLM credentials are required but not configured")
    if failures:
        print("FAILED")
        print("\n".join(failures))
        return 1
    print("MANIFEST_VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
