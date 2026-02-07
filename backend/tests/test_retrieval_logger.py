"""
检索日志记录器单元测试

测试 RetrievalLogger 类的功能，验证：
- RetrievalTrace 数据类创建（默认值和自定义值）
- log_trace 正常记录（验证日志输出包含关键信息）
- log_trace 降级场景（验证 WARNING 级别日志）
- to_retrieval_meta 正常场景（验证返回字段完整性）
- to_retrieval_meta 降级场景（验证 fallback 字段）
- to_retrieval_meta 粒度去重（验证 granularities 列表去重且保持顺序）
- to_retrieval_meta 空 citations（验证空列表处理）
- to_retrieval_meta 含 citations（验证引文映射正确传递）
"""

import logging

import pytest

from services.retrieval_logger import RetrievalLogger, RetrievalTrace


# ============================================================
# 辅助函数
# ============================================================


def _make_trace(
    query: str = "测试查询",
    query_type: str = "analytical",
    query_confidence: float = 0.85,
    chunk_hits: int = 5,
    group_hits: int = 3,
    rrf_top_k: list = None,
    token_budget: int = 8000,
    token_reserved: int = 1500,
    token_used: int = 4500,
    granularity_assignments: list = None,
    fallback_type: str = None,
    fallback_detail: str = None,
    citations: list = None,
) -> RetrievalTrace:
    """创建测试用的 RetrievalTrace 对象"""
    return RetrievalTrace(
        query=query,
        query_type=query_type,
        query_confidence=query_confidence,
        chunk_hits=chunk_hits,
        group_hits=group_hits,
        rrf_top_k=rrf_top_k
        if rrf_top_k is not None
        else [
            {"group_id": "group-0", "rank": 1, "source": "chunk"},
            {"group_id": "group-1", "rank": 2, "source": "group"},
        ],
        token_budget=token_budget,
        token_reserved=token_reserved,
        token_used=token_used,
        granularity_assignments=granularity_assignments
        if granularity_assignments is not None
        else [
            {"group_id": "group-0", "granularity": "full"},
            {"group_id": "group-1", "granularity": "digest"},
        ],
        fallback_type=fallback_type,
        fallback_detail=fallback_detail,
        citations=citations if citations is not None else [],
    )


# ============================================================
# 测试 RetrievalTrace 数据类
# ============================================================


class TestRetrievalTrace:
    """测试 RetrievalTrace 数据类的创建"""

    def test_默认值创建(self):
        """仅提供必填字段时，可选字段应使用默认值"""
        trace = RetrievalTrace(query="你好", query_type="overview")

        assert trace.query == "你好"
        assert trace.query_type == "overview"
        assert trace.query_confidence == 0.0
        assert trace.chunk_hits == 0
        assert trace.group_hits == 0
        assert trace.rrf_top_k == []
        assert trace.token_budget == 0
        assert trace.token_reserved == 0
        assert trace.token_used == 0
        assert trace.granularity_assignments == []
        assert trace.fallback_type is None
        assert trace.fallback_detail is None
        assert trace.citations == []

    def test_自定义值创建(self):
        """提供所有字段时，各字段值应正确设置"""
        rrf = [{"group_id": "group-0", "rank": 1, "source": "chunk"}]
        assignments = [{"group_id": "group-0", "granularity": "full"}]
        cites = [{"ref": 1, "group_id": "group-0", "page_range": [1, 3]}]

        trace = RetrievalTrace(
            query="关键公式有哪些",
            query_type="extraction",
            query_confidence=0.92,
            chunk_hits=10,
            group_hits=5,
            rrf_top_k=rrf,
            token_budget=8000,
            token_reserved=1500,
            token_used=6000,
            granularity_assignments=assignments,
            fallback_type="llm_failed",
            fallback_detail="LLM API 超时",
            citations=cites,
        )

        assert trace.query == "关键公式有哪些"
        assert trace.query_type == "extraction"
        assert trace.query_confidence == 0.92
        assert trace.chunk_hits == 10
        assert trace.group_hits == 5
        assert trace.rrf_top_k == rrf
        assert trace.token_budget == 8000
        assert trace.token_reserved == 1500
        assert trace.token_used == 6000
        assert trace.granularity_assignments == assignments
        assert trace.fallback_type == "llm_failed"
        assert trace.fallback_detail == "LLM API 超时"
        assert trace.citations == cites


# ============================================================
# 测试 log_trace 方法
# ============================================================


class TestLogTrace:
    """测试 RetrievalLogger.log_trace 方法的日志记录"""

    def test_正常记录包含关键信息(self, caplog):
        """正常检索时，日志应包含查询类型、命中数和 Token 使用信息"""
        logger_inst = RetrievalLogger()
        trace = _make_trace()

        with caplog.at_level(logging.INFO, logger="services.retrieval_logger"):
            logger_inst.log_trace(trace)

        log_text = caplog.text
        # 验证查询类型和置信度
        assert "analytical" in log_text
        assert "0.85" in log_text
        # 验证命中数
        assert "chunk=5" in log_text
        assert "group=3" in log_text
        # 验证 Token 信息
        assert "8000" in log_text
        assert "1500" in log_text
        assert "4500" in log_text

    def test_正常记录包含RRF信息(self, caplog):
        """日志应包含 RRF topK 的意群标识和排名"""
        logger_inst = RetrievalLogger()
        trace = _make_trace(
            rrf_top_k=[
                {"group_id": "group-0", "rank": 1, "source": "chunk"},
                {"group_id": "group-2", "rank": 2, "source": "group"},
            ]
        )

        with caplog.at_level(logging.INFO, logger="services.retrieval_logger"):
            logger_inst.log_trace(trace)

        log_text = caplog.text
        assert "group-0" in log_text
        assert "group-2" in log_text
        assert "rank=1" in log_text
        assert "rank=2" in log_text

    def test_正常记录包含粒度分配(self, caplog):
        """日志应包含各意群的粒度分配信息"""
        logger_inst = RetrievalLogger()
        trace = _make_trace(
            granularity_assignments=[
                {"group_id": "group-0", "granularity": "full"},
                {"group_id": "group-1", "granularity": "digest"},
                {"group_id": "group-2", "granularity": "summary"},
            ]
        )

        with caplog.at_level(logging.INFO, logger="services.retrieval_logger"):
            logger_inst.log_trace(trace)

        log_text = caplog.text
        assert "group-0=full" in log_text
        assert "group-1=digest" in log_text
        assert "group-2=summary" in log_text

    def test_降级场景使用WARNING级别(self, caplog):
        """发生降级时，日志应使用 WARNING 级别记录降级信息"""
        logger_inst = RetrievalLogger()
        trace = _make_trace(
            fallback_type="llm_failed",
            fallback_detail="LLM API 调用超时",
        )

        with caplog.at_level(logging.WARNING, logger="services.retrieval_logger"):
            logger_inst.log_trace(trace)

        # 验证 WARNING 级别日志存在
        warning_records = [
            r for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert len(warning_records) >= 1

        warning_text = " ".join(r.message for r in warning_records)
        assert "llm_failed" in warning_text
        assert "LLM API 调用超时" in warning_text

    def test_无降级时不产生WARNING日志(self, caplog):
        """未发生降级时，不应产生 WARNING 级别日志"""
        logger_inst = RetrievalLogger()
        trace = _make_trace(fallback_type=None, fallback_detail=None)

        with caplog.at_level(logging.WARNING, logger="services.retrieval_logger"):
            logger_inst.log_trace(trace)

        warning_records = [
            r for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert len(warning_records) == 0


# ============================================================
# 测试 to_retrieval_meta 方法
# ============================================================


class TestToRetrievalMeta:
    """测试 RetrievalLogger.to_retrieval_meta 方法"""

    def test_正常场景字段完整性(self):
        """正常检索时，返回的 meta 应包含所有必要字段"""
        logger_inst = RetrievalLogger()
        trace = _make_trace(
            query_type="analytical",
            token_used=4500,
            granularity_assignments=[
                {"group_id": "group-0", "granularity": "full"},
                {"group_id": "group-1", "granularity": "digest"},
            ],
        )

        meta = logger_inst.to_retrieval_meta(trace)

        # 验证所有必要字段存在
        assert "query_type" in meta
        assert "granularities" in meta
        assert "token_used" in meta
        assert "fallback" in meta
        assert "citations" in meta

        # 验证字段值
        assert meta["query_type"] == "analytical"
        assert meta["token_used"] == 4500
        assert meta["granularities"] == ["full", "digest"]
        assert meta["fallback"] is None

    def test_降级场景包含fallback字段(self):
        """发生降级时，fallback 字段应包含 type 和 detail"""
        logger_inst = RetrievalLogger()
        trace = _make_trace(
            fallback_type="index_missing",
            fallback_detail="意群向量索引不存在，回退到分块检索",
        )

        meta = logger_inst.to_retrieval_meta(trace)

        assert meta["fallback"] is not None
        assert meta["fallback"]["type"] == "index_missing"
        assert meta["fallback"]["detail"] == "意群向量索引不存在，回退到分块检索"

    def test_降级场景_groups_disabled(self):
        """意群功能禁用时，fallback 应正确标记"""
        logger_inst = RetrievalLogger()
        trace = _make_trace(
            fallback_type="groups_disabled",
            fallback_detail="意群功能已禁用",
        )

        meta = logger_inst.to_retrieval_meta(trace)

        assert meta["fallback"]["type"] == "groups_disabled"
        assert meta["fallback"]["detail"] == "意群功能已禁用"

    def test_粒度去重且保持顺序(self):
        """granularities 列表应去重且保持首次出现的顺序"""
        logger_inst = RetrievalLogger()
        trace = _make_trace(
            granularity_assignments=[
                {"group_id": "group-0", "granularity": "full"},
                {"group_id": "group-1", "granularity": "digest"},
                {"group_id": "group-2", "granularity": "full"},
                {"group_id": "group-3", "granularity": "summary"},
                {"group_id": "group-4", "granularity": "digest"},
            ]
        )

        meta = logger_inst.to_retrieval_meta(trace)

        # 去重后应为 ["full", "digest", "summary"]，保持首次出现顺序
        assert meta["granularities"] == ["full", "digest", "summary"]

    def test_空citations处理(self):
        """citations 为空列表时，返回的 citations 也应为空列表"""
        logger_inst = RetrievalLogger()
        trace = _make_trace(citations=[])

        meta = logger_inst.to_retrieval_meta(trace)

        assert meta["citations"] == []

    def test_含citations正确传递(self):
        """citations 包含引文映射时，应原样传递到 meta 中"""
        logger_inst = RetrievalLogger()
        citations_data = [
            {"ref": 1, "group_id": "group-0", "page_range": [1, 3]},
            {"ref": 2, "group_id": "group-1", "page_range": [4, 6]},
            {"ref": 3, "group_id": "group-2", "page_range": [7, 7]},
        ]
        trace = _make_trace(citations=citations_data)

        meta = logger_inst.to_retrieval_meta(trace)

        assert meta["citations"] == citations_data
        assert len(meta["citations"]) == 3
        # 验证各引文映射的字段
        assert meta["citations"][0]["ref"] == 1
        assert meta["citations"][0]["group_id"] == "group-0"
        assert meta["citations"][0]["page_range"] == [1, 3]
        assert meta["citations"][2]["ref"] == 3
        assert meta["citations"][2]["page_range"] == [7, 7]

    def test_空粒度分配返回空列表(self):
        """granularity_assignments 为空时，granularities 应为空列表"""
        logger_inst = RetrievalLogger()
        trace = _make_trace(granularity_assignments=[])

        meta = logger_inst.to_retrieval_meta(trace)

        assert meta["granularities"] == []

    def test_无降级时fallback为None(self):
        """未发生降级时，fallback 字段应为 None"""
        logger_inst = RetrievalLogger()
        trace = _make_trace(fallback_type=None, fallback_detail=None)

        meta = logger_inst.to_retrieval_meta(trace)

        assert meta["fallback"] is None
