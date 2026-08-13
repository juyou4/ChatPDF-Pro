"""身份完整的向量索引测试夹具。

MinerU 重建的质量门（``document_routes._inspect_vector_index_artifacts`` /
``_validate_temp_vector_index`` / ``_prepare_semantic_group_rebuild``）要求索引
对携带完整的解析身份（``index_meta`` 里的 parse_generation、
document_source_hash、block_index_hash、content_source、
evidence_schema_version）与 embedding/build 语义身份（模型、provider、
vector_build_id、identity version、维度），且 FAISS 索引必须真实可读、向量数
与分块一一对应。占位字节写出的假索引在这些门禁下全部被拒。

本模块一次性产出满足全部检查的索引对，供 rebuild / upload / agentic 等
回归共用；重依赖（faiss / numpy / embedding_service 常量）在函数内延迟导入，
避免拖慢不需要它们的轻量测试文件。
"""
from __future__ import annotations

import pickle
from pathlib import Path


def default_semantic_identity(doc_id: str, *, dimension: int = 8) -> dict:
    """与 ``write_published_vector_index`` 默认值同源的 embedding/build 身份。

    语义组备份的恢复会用当前向量索引的语义身份做期望校验；意群夹具引用
    同一份身份即可与向量夹具互认。
    """
    from services.embedding_service import EMBEDDING_IDENTITY_VERSION

    return {
        "parse_generation": f"test-gen-{doc_id}",
        "document_source_hash": f"test-hash-{doc_id}",
        "vector_build_id": f"test-build-{doc_id}",
        "embedding_identity_version": EMBEDDING_IDENTITY_VERSION,
        "embedding_model": "local-minilm",
        "embedding_provider": "local",
        "embedding_api_host": "",
        "vector_dimension": dimension,
    }


def write_published_vector_index(
    index_dir: Path | str,
    doc_id: str,
    *,
    chunks: list[str],
    index_source: str = "mineru",
    index_meta: dict | None = None,
    chunk_metadata: list[dict] | None = None,
    embedding_model: str = "local-minilm",
    embedding_provider: str = "local",
    embedding_api_host: str = "",
    dimension: int = 8,
    index_version: int | None = None,
) -> dict:
    """写出一对通过当前全部索引门禁的 ``.index`` / ``.pkl`` 文件。

    返回写入的 pkl payload，便于测试断言或在其上做针对性破坏。
    """
    import faiss
    import numpy as np

    from services.embedding_service import EMBEDDING_IDENTITY_VERSION, RAG_INDEX_VERSION

    directory = Path(index_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(20260813)
    vectors = rng.randn(max(1, len(chunks)), dimension).astype("float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors[: len(chunks)] if chunks else vectors[:0])
    faiss.write_index(index, str(directory / f"{doc_id}.index"))

    metadata = list(chunk_metadata or [])
    metadata = (metadata + [{} for _ in chunks])[: len(chunks)]
    meta = dict(index_meta or {})
    # embedding/build 语义身份的解析部分从 index_meta 读取；没有它们，
    # 索引会被判 embedding_build_identity_incomplete。调用方显式传入时不覆盖。
    meta.setdefault("parse_generation", f"test-gen-{doc_id}")
    meta.setdefault("document_source_hash", f"test-hash-{doc_id}")
    payload = {
        "index_version": RAG_INDEX_VERSION if index_version is None else index_version,
        "index_source": index_source,
        "chunks": list(chunks),
        "chunk_metadata": metadata,
        "chunk_pages": [1 for _ in chunks],
        "chunk_types": ["text" for _ in chunks],
        "embedding_model": embedding_model,
        "embedding_provider": embedding_provider,
        "embedding_api_host": embedding_api_host,
        "embedding_identity_version": EMBEDDING_IDENTITY_VERSION,
        "vector_build_id": f"test-build-{doc_id}",
        "vector_count": len(chunks),
        "vector_dimension": dimension,
        "index_meta": meta,
    }
    # 顶层镜像字段与真实 create_index 的持久化保持一致，_read_vector_index_meta
    # 会直接读它们展示。
    for key in ("source_hash", "rebuilt_at", "previous_index_source", "normalizer_version"):
        if key in meta:
            payload[key] = meta[key]
    with open(directory / f"{doc_id}.pkl", "wb") as f:
        pickle.dump(payload, f)
    return payload


def make_fake_create_index(captured: dict | None = None, *, extra_chunks: list[str] | None = None,
                           extra_chunk_metadata: list[dict] | None = None):
    """返回签名兼容 ``document_routes.create_index`` 的可信替身。

    真实实现会把调用方传入的 ``index_meta``（含全部解析身份）连同真实向量
    一起持久化；质量门随后按这份身份验证临时索引。替身必须复刻这一行为，
    否则任何 rebuild 测试都会死在 ``temp_*_mismatch`` 上。
    """

    def fake_create_index(
        doc_id,
        full_text,
        vector_store_dir,
        embedding_model,
        _api_key=None,
        _api_host=None,
        *,
        pages=None,
        structured_table_bundles=None,
        summary_api_key=None,
        index_source="mineru",
        index_meta=None,
        build_semantic_groups=True,
        **_kwargs,
    ):
        if captured is not None:
            captured["full_text"] = full_text
            captured["pages"] = pages
            captured["structured_table_bundles"] = structured_table_bundles
            captured["build_semantic_groups"] = build_semantic_groups
        chunks = [full_text, *(extra_chunks or [])]
        metadata = [{}, *(extra_chunk_metadata or [])]
        write_published_vector_index(
            Path(vector_store_dir),
            doc_id,
            chunks=chunks,
            chunk_metadata=metadata,
            index_source=index_source,
            index_meta=index_meta or {},
            embedding_model=embedding_model,
        )

    return fake_create_index
