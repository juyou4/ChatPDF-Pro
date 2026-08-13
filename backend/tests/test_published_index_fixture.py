"""守住 tests/vector_index_fixtures.py 里的「已发布索引」工厂。

这个工厂被 rebuild / upload / agentic 等多处回归当作前提使用。如果它产出的
产物其实过不了准入闸门，那些测试会以很难排查的方式失败，所以这里直接拿
生产侧的检查函数验它，而不是相信它的返回值。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import routes.document_routes as document_routes
from tests.vector_index_fixtures import (
    default_semantic_identity,
    make_fake_create_index,
    write_published_vector_index,
)

_IDENTITY = {
    "parse_generation": "gen-guard-1",
    "document_source_hash": "hash-guard-1",
    "block_index_hash": "block-guard-1",
    "content_source": "mineru",
    "evidence_schema_version": 1,
}


def test_factory_output_passes_vector_index_inspection(tmp_path):
    write_published_vector_index(
        tmp_path,
        "doc-guard",
        chunks=["第一段内容", "第二段内容", "第三段内容"],
        index_meta=dict(_IDENTITY),
    )

    inspection = document_routes._inspect_vector_index_artifacts(
        "doc-guard",
        tmp_path,
        expected_source="mineru",
        expected_parse_generation=_IDENTITY["parse_generation"],
        expected_document_source_hash=_IDENTITY["document_source_hash"],
        expected_block_index_hash=_IDENTITY["block_index_hash"],
        expected_content_source=_IDENTITY["content_source"],
        expected_evidence_schema_version=_IDENTITY["evidence_schema_version"],
    )

    assert inspection.get("errors") == [], inspection.get("errors")


def test_factory_output_passes_temp_index_quality_gate(tmp_path):
    write_published_vector_index(
        tmp_path,
        "doc-guard",
        chunks=["第一段内容", "第二段内容"],
        index_meta=dict(_IDENTITY),
    )

    ok, failures = document_routes._validate_temp_vector_index(
        "doc-guard",
        tmp_path,
        expected_parse_generation=_IDENTITY["parse_generation"],
        expected_document_source_hash=_IDENTITY["document_source_hash"],
        expected_block_index_hash=_IDENTITY["block_index_hash"],
        expected_content_source=_IDENTITY["content_source"],
        expected_evidence_schema_version=_IDENTITY["evidence_schema_version"],
    )

    assert ok, failures


def test_fake_create_index_persists_caller_identity(tmp_path):
    """替身必须复刻真实 create_index 的行为：把调用方的 index_meta 原样持久化。

    质量门随后按这份身份验证临时索引，替身丢字段的话所有 rebuild 测试都会
    死在 temp_*_mismatch 上——这正是本轮清理前的实际故障模式。
    """
    fake = make_fake_create_index()
    fake(
        "doc-guard",
        "全文内容",
        str(tmp_path),
        "local-minilm",
        index_source="mineru",
        index_meta=dict(_IDENTITY),
    )

    ok, failures = document_routes._validate_temp_vector_index(
        "doc-guard",
        tmp_path,
        expected_parse_generation=_IDENTITY["parse_generation"],
        expected_document_source_hash=_IDENTITY["document_source_hash"],
        expected_block_index_hash=_IDENTITY["block_index_hash"],
        expected_content_source=_IDENTITY["content_source"],
        expected_evidence_schema_version=_IDENTITY["evidence_schema_version"],
    )

    assert ok, failures


def test_default_semantic_identity_is_complete(tmp_path):
    """embedding/build 语义身份缺任意一项都会判 embedding_build_identity_incomplete。"""
    from services.embedding_service import (
        _semantic_generation_identity_complete,
    )

    assert _semantic_generation_identity_complete(default_semantic_identity("doc-guard"))
