import os
import pickle
import re
from typing import List, Optional, Tuple

import faiss
import numpy as np
from fastapi import HTTPException
from sentence_transformers import SentenceTransformer

from models.model_detector import is_embedding_model, is_rerank_model, get_model_provider
from models.model_registry import EMBEDDING_MODELS
from services.rerank_service import rerank_service

# Lazy-loaded caches
local_embedding_models = {}


def preprocess_text(text: str) -> str:
    """
    Lightweight preprocessing before chunking:
    - 去掉常见版权/噪声行（如 IEEE 授权提示）
    - 合并多余空行
    - 修复连字符断行
    """
    if not text:
        return ""

    lines = []
    noisy_patterns = [
        "Authorized licensed use limited to",
        "All rights reserved",
        "IEEE",
    ]

    for line in text.splitlines():
        lstrip = line.strip()
        if any(pat.lower() in lstrip.lower() for pat in noisy_patterns):
            continue
        lines.append(line)

    cleaned = "\n".join(lines)
    # 修复连字符断行：word-\nword -> wordword
    cleaned = re.sub(r"(\w)-\n(\w)", r"\1\2", cleaned)
    # 统一空白
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def normalize_embedding_model_id(embedding_model_id: Optional[str]) -> Optional[str]:
    """Normalize embedding model id to a configured key (supports provider:model or plain id)"""
    if not embedding_model_id:
        return None

    if embedding_model_id in EMBEDDING_MODELS:
        return embedding_model_id

    if ":" in embedding_model_id:
        provider_part, model_part = embedding_model_id.split(":", 1)
        if model_part in EMBEDDING_MODELS:
            return model_part
        combined_key = f"{provider_part}:{model_part}"
        if combined_key in EMBEDDING_MODELS:
            return combined_key

    return None


def get_embedding_function(embedding_model_id: str, api_key: str = None, base_url: str = None):
    """Get embedding function for the specified model"""
    normalized_id = normalize_embedding_model_id(embedding_model_id)
    if normalized_id:
        embedding_model_id = normalized_id
    else:
        print(f"Warning: embedding model '{embedding_model_id}' not in configuration, attempting inference")

    if not is_embedding_model(embedding_model_id):
        if is_rerank_model(embedding_model_id):
            raise ValueError(f"Model {embedding_model_id} is a rerank model, not an embedding model")
        print(f"Warning: {embedding_model_id} doesn't match embedding model patterns, attempting to use anyway")

    config = EMBEDDING_MODELS.get(embedding_model_id)

    if config:
        provider = config["provider"]
        model_name = config.get("model_name", embedding_model_id)
        api_base = base_url or config.get("base_url")
    else:
        provider = get_model_provider(embedding_model_id)
        model_name = embedding_model_id
        api_base = base_url or "https://api.openai.com/v1"
        if not api_base.endswith('/embeddings') and not api_base.endswith('/v1'):
            api_base = api_base.rstrip('/') + '/v1'

    if provider == "local":
        if model_name not in local_embedding_models:
            print(f"Loading local embedding model: {model_name}")
            local_embedding_models[model_name] = SentenceTransformer(model_name)
        model = local_embedding_models[model_name]
        return lambda texts: model.encode(texts)

    if not api_key:
        raise ValueError(f"API key required for {embedding_model_id}")

    from openai import OpenAI

    api_base = api_base or "https://api.openai.com/v1"
    if not api_base.endswith('/v1') and not api_base.endswith('/v1/'):
        api_base = api_base.rstrip('/') + '/v1'

    client = OpenAI(api_key=api_key, base_url=api_base)

    def embed_texts(texts):
        response = client.embeddings.create(
            model=embedding_model_id,
            input=texts
        )
        return np.array([item.embedding for item in response.data])

    return embed_texts


def get_chunk_params(embedding_model_id: str, base_chunk_size: int = 1000, base_overlap: int = 200) -> tuple[int, int]:
    """Return (chunk_size, chunk_overlap) with model-aware clamping."""
    cfg = EMBEDDING_MODELS.get(embedding_model_id, {})
    max_ctx = cfg.get("max_tokens")

    chunk_size = base_chunk_size
    if max_ctx:
        # 保守取 20% 的上下文长度，并限制在 300-1500 区间
        chunk_size = min(chunk_size, int(max_ctx * 0.2))
        chunk_size = max(300, min(chunk_size, 1500))

    # 重叠 10-20%，至少 100，且必须小于 chunk_size
    chunk_overlap = max(base_overlap, int(chunk_size * 0.2))
    chunk_overlap = min(chunk_overlap, int(chunk_size * 0.4))  # 避免重叠过大
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(50, int(chunk_size * 0.15))

    return chunk_size, chunk_overlap


def _distance_to_similarity(distance: float) -> float:
    try:
        safe_distance = max(distance, 0.0)
        return float(1.0 / (1.0 + safe_distance))
    except Exception:
        return 0.0


def _extract_snippet_and_highlights(text: str, query: str, window: int = 100) -> Tuple[str, List[dict]]:
    if not text:
        return "", []

    normalized_text = " ".join(text.split())
    lower_text = normalized_text.lower()
    terms = [t for t in re.split(r"[\s,;，。；、]+", query.lower()) if t]

    matches = []
    for term in terms:
        start = lower_text.find(term)
        while start != -1:
            end = start + len(term)
            matches.append((start, end, normalized_text[start:end]))
            start = lower_text.find(term, end)
    matches.sort(key=lambda x: x[0])

    if matches:
        snippet_start = max(0, matches[0][0] - window)
        snippet_end = min(len(normalized_text), matches[0][1] + window)
    else:
        snippet_start = 0
        snippet_end = min(len(normalized_text), window * 2)

    snippet = normalized_text[snippet_start:snippet_end]
    highlights = []
    for start, end, _ in matches:
        if end <= snippet_start or start >= snippet_end:
            continue
        local_start = max(0, start - snippet_start)
        local_end = min(snippet_end - snippet_start, end - snippet_start)
        highlights.append({
            "start": int(local_start),
            "end": int(local_end),
            "text": normalized_text[start:end]
        })

    return snippet, highlights


def _find_page_for_chunk(chunk_text: str, pages: List[dict]) -> int:
    if not pages:
        return 1

    for page in pages:
        content = page.get("content", "")
        if chunk_text[:80] in content:
            return page.get("page", 1)
        if chunk_text[:60].lower() in content.lower():
            return page.get("page", 1)
    return pages[0].get("page", 1)


def _apply_rerank(
    query: str,
    candidates: List[dict],
    reranker_model: Optional[str] = None,
    rerank_provider: Optional[str] = None,
    rerank_api_key: Optional[str] = None,
    rerank_endpoint: Optional[str] = None
) -> List[dict]:
    model_name = reranker_model or "BAAI/bge-reranker-base"
    return rerank_service.rerank(
        query,
        candidates,
        model_name=model_name,
        provider=rerank_provider or "local",
        api_key=rerank_api_key,
        endpoint=rerank_endpoint
    )


def build_vector_index(
    doc_id: str,
    text: str,
    vector_store_dir: str,
    embedding_model_id: str = "local-minilm",
    api_key: str = None,
    api_host: str = None
):
    try:
        print(f"Building vector index for {doc_id}...")
        if embedding_model_id not in EMBEDDING_MODELS:
            if ":" in embedding_model_id:
                _, model_part = embedding_model_id.split(":", 1)
                if model_part in EMBEDDING_MODELS:
                    embedding_model_id = model_part
                else:
                    raise ValueError(f"Embedding model '{embedding_model_id}' 未配置或不受支持，请检查模型选择")
            else:
                raise ValueError(f"Embedding model '{embedding_model_id}' 未配置或不受支持，请检查模型选择")

        # 分块策略：按模型最大上下文自适应，默认 1000 / 200（约 20% 重叠），限制在 300-1500
        chunk_size, chunk_overlap = get_chunk_params(embedding_model_id, base_chunk_size=1000, base_overlap=200)
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        preprocessed_text = preprocess_text(text)
        chunks = text_splitter.split_text(preprocessed_text)
        print(f"Split into {len(chunks)} chunks.")

        if not chunks:
            return

        embed_fn = get_embedding_function(embedding_model_id, api_key, api_host)
        embeddings = embed_fn(chunks)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))

        os.makedirs(vector_store_dir, exist_ok=True)
        index_path = os.path.join(vector_store_dir, f"{doc_id}.index")
        chunks_path = os.path.join(vector_store_dir, f"{doc_id}.pkl")

        faiss.write_index(index, index_path)
        with open(chunks_path, "wb") as f:
            pickle.dump({"chunks": chunks, "embedding_model": embedding_model_id}, f)

        print(f"Vector index saved to {index_path}")

    except Exception as e:
        print(f"Error building vector index for {doc_id}: {e}")
        raise


def search_document_chunks(
    doc_id: str,
    query: str,
    vector_store_dir: str,
    pages: List[dict],
    api_key: str = None,
    top_k: int = 5,
    candidate_k: int = 20,
    use_rerank: bool = False,
    reranker_model: Optional[str] = None,
    rerank_provider: Optional[str] = None,
    rerank_api_key: Optional[str] = None,
    rerank_endpoint: Optional[str] = None
) -> List[dict]:
    index_path = os.path.join(vector_store_dir, f"{doc_id}.index")
    chunks_path = os.path.join(vector_store_dir, f"{doc_id}.pkl")

    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        raise HTTPException(status_code=404, detail="向量索引未找到,请重新上传PDF")

    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        chunks = data["chunks"]
        embedding_model_id = data.get("embedding_model", "local-minilm")
    else:
        chunks = data
        embedding_model_id = "local-minilm"

    embed_fn = get_embedding_function(embedding_model_id, api_key)
    query_vector = embed_fn([query])

    search_k = max(candidate_k, top_k)
    D, I = index.search(np.array(query_vector).astype('float32'), search_k)

    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < len(chunks):
            chunk_text = chunks[idx]
            page_num = _find_page_for_chunk(chunk_text, pages)
            similarity = _distance_to_similarity(float(dist))
            snippet, highlights = _extract_snippet_and_highlights(chunk_text, query)

            results.append({
                "chunk": chunk_text,
                "page": page_num,
                "score": float(dist),
                "similarity": similarity,
                "similarity_percent": round(similarity * 100, 2),
                "snippet": snippet,
                "highlights": highlights,
                "reranked": False
            })

    if use_rerank:
        results = _apply_rerank(
            query,
            results,
            reranker_model,
            rerank_provider,
            rerank_api_key,
            rerank_endpoint
        )
    else:
        results = sorted(results, key=lambda x: x.get("similarity", 0), reverse=True)

    return results[:top_k]


def get_relevant_context(
    doc_id: str,
    query: str,
    vector_store_dir: str,
    pages: List[dict],
    api_key: str = None,
    top_k: int = 5,
    use_rerank: bool = False,
    reranker_model: Optional[str] = None,
    candidate_k: int = 20,
    rerank_provider: Optional[str] = None,
    rerank_api_key: Optional[str] = None,
    rerank_endpoint: Optional[str] = None
) -> str:
    results = search_document_chunks(
        doc_id,
        query,
        vector_store_dir=vector_store_dir,
        pages=pages,
        api_key=api_key,
        top_k=top_k,
        candidate_k=candidate_k,
        use_rerank=use_rerank,
        reranker_model=reranker_model,
        rerank_provider=rerank_provider,
        rerank_api_key=rerank_api_key,
        rerank_endpoint=rerank_endpoint
    )
    relevant_chunks = [item["chunk"] for item in results]
    return "\n\n...\n\n".join(relevant_chunks)
