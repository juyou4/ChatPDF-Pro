"""测试分层记忆注入器。"""
import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.context_injector import ContextInjector


class TestContextInjector:
    def test_prepare_memories_respects_layer_order_and_budget(self):
        injector = ContextInjector(token_budget=320)
        memories = [
            {
                "id": "episodic-1",
                "memory_kind": "episodic",
                "content": "E" * 120,
                "importance": 1.0,
                "score": 1.0,
            },
            {
                "id": "profile-1",
                "memory_kind": "profile",
                "content": "P" * 120,
                "importance": 0.8,
                "score": 0.8,
            },
            {
                "id": "working-1",
                "memory_kind": "working",
                "content": "W" * 120,
                "importance": 0.5,
                "score": 0.5,
            },
            {
                "id": "doc-1",
                "memory_kind": "doc_fact",
                "content": "D" * 160,
                "importance": 0.9,
                "score": 0.9,
            },
        ]

        selected = injector.prepare_memories(memories)

        selected_ids = [item["id"] for item in selected]
        assert selected_ids[:3] == ["working-1", "profile-1", "doc-1"]
        assert "episodic-1" not in selected_ids

    def test_inject_formats_memory_blocks_by_kind(self):
        injector = ContextInjector(token_budget=400)
        prompt = injector.inject(
            "system",
            [
                {"memory_kind": "working", "title": "最近对话", "content": "Q: 最近一轮\nA: 回答"},
                {"memory_kind": "graph", "title": "论文图谱摘要", "content": "图谱节点：figure=Figure 2"},
            ],
        )

        assert "## [工作记忆]" in prompt
        assert "## [论文图谱]" in prompt
