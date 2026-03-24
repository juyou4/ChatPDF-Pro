"""Check semantic groups and context assembly for InstaGen."""
import sys, os, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DOC_ID = "ebe975c3cc451af57211efcb89c90fd6"
groups_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "semantic_groups", f"{DOC_ID}.json"
)

with open(groups_path, "r", encoding="utf-8") as f:
    data = json.load(f)

gs = data if isinstance(data, list) else data.get("groups", [])
print(f"groups: {len(gs)}")

# Check raw JSON keys
g0 = gs[0] if gs else {}
print(f"g0 keys: {list(g0.keys())}")
print(f"g0 summary_status: {g0.get('summary_status', 'MISSING')!r}")

# Load as SemanticGroup objects
from services.semantic_group_service import SemanticGroup
groups = [SemanticGroup.from_dict(g) for g in gs]
print(f"\nLoaded {len(groups)} SemanticGroup objects")
print(f"g0.summary_status = {groups[0].summary_status!r}")

# Simulate select_mixed for "请总结本文的主要内容"
from services.granularity_selector import GranularitySelector
selector = GranularitySelector()
# Use first 10 as ranked (overview max_groups=10)
ranked = groups[:10]
mixed = selector.select_mixed(query="请总结本文的主要内容", ranked_groups=ranked)
print(f"\nselect_mixed results ({len(mixed)} items):")
for item in mixed:
    g = item["group"]
    gran = item["granularity"]
    text_attr = {"full": "full_text", "digest": "digest", "summary": "summary"}[gran]
    text_len = len(getattr(g, text_attr, ""))
    print(f"  {g.group_id}: {gran} -> {text_len}c")

# Simulate fit_within_budget
from services.token_budget import TokenBudgetManager
from services.rag_config import RAGConfig
cfg = RAGConfig()
print(f"\nToken budget: max={cfg.max_token_budget}, reserve={cfg.reserve_for_answer}")
bm = TokenBudgetManager(max_tokens=cfg.max_token_budget, reserve_for_answer=cfg.reserve_for_answer)
fitted = bm.fit_within_budget(mixed)
print(f"After budget fit: {len(fitted)} items")
total_tokens = sum(item.get("tokens", 0) for item in fitted)
print(f"Total tokens used: {total_tokens}/{bm.available_tokens}")
for item in fitted:
    g = item["group"]
    print(f"  {g.group_id}: {item['granularity']} tokens={item.get('tokens',0)}")
