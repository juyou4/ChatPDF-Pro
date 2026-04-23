import json
d = json.load(open("tests/ragas_results.json", "r", encoding="utf-8"))
ps = d["ragas_scores"].get("_per_sample", [])
for p in ps:
    f = p.get("faithfulness", 0) or 0
    ar = p.get("answer_relevancy", 0) or 0
    cp = p.get("llm_context_precision_without_reference", 0) or 0
    print(f"{p['index']:2d} F={f:.2f} AR={ar:.2f} CP={cp:.2f} {p['question']}")
