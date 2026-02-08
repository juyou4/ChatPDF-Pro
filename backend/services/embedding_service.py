import asyncio
import logging
import os
import pickle
import re
from typing import List, Optional, Tuple

import faiss
import numpy as np
from fastapi import HTTPException
from sentence_transformers import SentenceTransformer

from models.api_key_selector import select_api_key
from models.model_detector import is_embedding_model, is_rerank_model, get_model_provider
from models.model_id_resolver import resolve_model_id, get_available_model_ids
from models.model_registry import EMBEDDING_MODELS
from services.rerank_service import rerank_service

logger = logging.getLogger(__name__)

# Lazy-loaded caches
local_embedding_models = {}


def preprocess_text(text: str) -> str:
    """
    Lightweight preprocessing before chunking:
    - 去掉常见版权/噪声行（如 IEEE 授权提示）
    - 合并多余空行
    - 修复连字符断行
    - 过滤图表乱码（NULL字符）
    """
    if not text:
        return ""

    lines = []
    noisy_patterns = [
        "Authorized licensed use limited to",
        "All rights reserved",
    ]

    for line in text.splitlines():
        lstrip = line.strip()
        if any(pat.lower() in lstrip.lower() for pat in noisy_patterns):
            continue
        
        # 只过滤包含大量 NULL 字符的行
        null_count = line.count('\u0000') + line.count('\x00')
        if len(line) > 5 and null_count / len(line) > 0.3:
            continue
        
        # 移除 NULL 字符
        cleaned_line = line.replace('\u0000', '').replace('\x00', '')
        if cleaned_line.strip():
            lines.append(cleaned_line)

    cleaned = "\n".join(lines)
    # 修复连字符断行：word-\nword -> wordword
    cleaned = re.sub(r"(\w)-\n(\w)", r"\1\2", cleaned)
    # 统一空白
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def normalize_embedding_model_id(embedding_model_id: Optional[str]) -> Optional[str]:
    """归一化 embedding 模型 ID，返回 Model_Registry 中的键名

    使用 Model_ID_Resolver 统一解析前端传入的模型 ID，
    支持 composite key（provider:modelId）和 plain key 两种格式。

    Args:
        embedding_model_id: 前端传入的模型 ID

    Returns:
        Model_Registry 中的键名，解析失败时返回 None 并记录警告日志（包含可用模型列表）
    """
    if not embedding_model_id:
        return None

    # 使用 Model_ID_Resolver 统一解析
    registry_key, config = resolve_model_id(embedding_model_id)
    if registry_key is not None:
        return registry_key

    # 解析失败，记录警告日志并返回 None
    available_models = get_available_model_ids()
    logger.warning(
        f"无法解析模型 ID '{embedding_model_id}'，"
        f"可用模型列表: {available_models}"
    )
    return None


def get_embedding_function(embedding_model_id: str, api_key: str = None, base_url: str = None):
    """获取指定模型的 embedding 函数

    优先使用 Model_ID_Resolver 解析模型 ID 并获取完整配置；
    如果 Resolver 无法解析（未注册模型），则回退到 model_detector 推断 provider 和 base_url，
    输出警告日志并尝试继续。

    Args:
        embedding_model_id: 模型 ID，支持 composite key（provider:modelId）或 plain key
        api_key: API 密钥（非本地模型必需）
        base_url: 自定义 API 基础 URL（可选，优先于注册表中的 base_url）

    Returns:
        embedding 函数，接受文本列表并返回向量数组

    Raises:
        ValueError: 当模型是 rerank 模型而非 embedding 模型时
        ValueError: 当非本地模型缺少 API Key 时
    """
    # 使用 Model_ID_Resolver 统一解析模型 ID
    registry_key, config = resolve_model_id(embedding_model_id)

    if registry_key is not None:
        # Resolver 解析成功，使用注册表中的配置
        embedding_model_id = registry_key
        provider = config["provider"]
        model_name = config.get("model_name", embedding_model_id)
        api_base = base_url or config.get("base_url")
    else:
        # Resolver 解析失败，尝试从 composite key 中提取 provider 信息
        logger.warning(
            f"模型 '{embedding_model_id}' 未在注册表中找到，"
            f"尝试从 composite key 推断 provider 和 base_url"
        )
        config = None

        if ":" in embedding_model_id:
            # composite key 格式：provider:modelId
            provider_part, model_part = embedding_model_id.split(":", 1)
            embedding_model_id = model_part  # 实际调用 API 时用 modelId 部分
            model_name = model_part

            # 根据 provider 推断 base_url
            from models.model_id_resolver import PROVIDER_ALIAS_MAP, PROVIDER_BASE_URL_HINTS
            provider_aliases = PROVIDER_ALIAS_MAP.get(provider_part, [provider_part])
            provider = provider_aliases[0] if provider_aliases else "openai"

            # 使用 provider 对应的默认 base_url
            PROVIDER_DEFAULT_BASE_URLS = {
                "silicon": "https://api.siliconflow.cn/v1",
                "aliyun": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "moonshot": "https://api.moonshot.cn/v1",
                "deepseek": "https://api.deepseek.com/v1",
                "zhipu": "https://open.bigmodel.cn/api/paas/v4",
                "minimax": "https://api.minimax.chat/v1",
                "openai": "https://api.openai.com/v1",
            }
            api_base = base_url or PROVIDER_DEFAULT_BASE_URLS.get(provider_part, "https://api.openai.com/v1")
            logger.info(
                f"从 composite key 推断: provider={provider_part}, "
                f"model={model_part}, base_url={api_base}"
            )
        else:
            # plain key，使用 model_detector 推断
            provider = get_model_provider(embedding_model_id)
            model_name = embedding_model_id
            api_base = base_url or "https://api.openai.com/v1"
            if not api_base.endswith('/embeddings') and not api_base.endswith('/v1'):
                api_base = api_base.rstrip('/') + '/v1'

    # 验证模型类型
    if not is_embedding_model(embedding_model_id):
        if is_rerank_model(embedding_model_id):
            raise ValueError(f"模型 {embedding_model_id} 是 rerank 模型，不是 embedding 模型")
        logger.warning(
            f"模型 '{embedding_model_id}' 不匹配 embedding 模型模式，尝试继续使用"
        )

    # 本地模型：使用 SentenceTransformer
    if provider == "local":
        if model_name not in local_embedding_models:
            logger.info(f"加载本地 embedding 模型: {model_name}")
            local_embedding_models[model_name] = SentenceTransformer(model_name)
        model = local_embedding_models[model_name]
        return lambda texts: model.encode(texts)

    # 远程模型：从 Key 池中随机选择一个有效 Key
    actual_key = select_api_key(api_key) if api_key else None
    if not actual_key:
        raise ValueError(f"模型 '{embedding_model_id}' 需要 API Key")

    from openai import OpenAI

    api_base = api_base or "https://api.openai.com/v1"
    if not api_base.endswith('/v1') and not api_base.endswith('/v1/'):
        api_base = api_base.rstrip('/') + '/v1'

    client = OpenAI(api_key=actual_key, base_url=api_base)

    def embed_texts(texts):
        response = client.embeddings.create(
            model=embedding_model_id,
            input=texts
        )
        return np.array([item.embedding for item in response.data])

    return embed_texts


def get_chunk_params(embedding_model_id: str, base_chunk_size: int = 1200, base_overlap: int = 200) -> tuple[int, int]:
    """Return (chunk_size, chunk_overlap) with model-aware clamping."""
    cfg = EMBEDDING_MODELS.get(embedding_model_id, {})
    max_ctx = cfg.get("max_tokens")

    chunk_size = base_chunk_size
    if max_ctx:
        # 使用更大的比例（50%而不是40%），并提高上限到2500
        chunk_size = min(chunk_size, int(max_ctx * 0.5))
        chunk_size = max(1000, min(chunk_size, 2500))  # 提高下限到1000，上限到2500
    else:
        # 如果没有max_tokens配置，使用默认的1200
        chunk_size = base_chunk_size

    # 重叠 15-25%
    chunk_overlap = max(base_overlap, int(chunk_size * 0.15))
    chunk_overlap = min(chunk_overlap, int(chunk_size * 0.25))
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(100, int(chunk_size * 0.15))

    return chunk_size, chunk_overlap


def _distance_to_similarity(distance: float) -> float:
    try:
        safe_distance = max(distance, 0.0)
        return float(1.0 / (1.0 + safe_distance))
    except Exception:
        return 0.0


def _extract_snippet_and_highlights(text: str, query: str, window: int = 100) -> Tuple[str, List[dict]]:
    """从文本中提取包含查询关键词的片段和高亮位置

    匹配策略（按优先级）：
    1. 完整短语匹配：尝试匹配整个查询字符串
    2. 单词级匹配：将查询拆分为单词逐个匹配
    """
    if not text:
        return "", []

    normalized_text = " ".join(text.split())
    lower_text = normalized_text.lower()
    query_lower = query.lower().strip()

    matches = []

    # 策略 1：完整短语匹配
    phrase_start = lower_text.find(query_lower)
    while phrase_start != -1:
        phrase_end = phrase_start + len(query_lower)
        matches.append((phrase_start, phrase_end, normalized_text[phrase_start:phrase_end]))
        phrase_start = lower_text.find(query_lower, phrase_end)

    # 策略 2：如果完整短语未匹配，回退到单词级匹配
    if not matches:
        terms = [t for t in re.split(r"[\s,;，。；、]+", query_lower) if t]
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
    """对候选结果应用重排序

    注意：此函数为同步调用，在 async 上下文中应通过
    asyncio.to_thread() 调用以避免阻塞事件循环。
    """
    model_name = reranker_model or "BAAI/bge-reranker-base"
    provider = (rerank_provider or "local").lower()
    logger.info(f"[Rerank] 开始重排序: provider={provider}, model={model_name}, 候选数={len(candidates)}")

    try:
        result = rerank_service.rerank(
            query,
            candidates,
            model_name=model_name,
            provider=provider,
            api_key=rerank_api_key,
            endpoint=rerank_endpoint
        )
        logger.info(f"[Rerank] 重排序完成，返回 {len(result)} 条结果")
        return result
    except Exception as e:
        logger.error(f"[Rerank] 重排序失败: {e}", exc_info=True)
        # 回退到相似度排序，不静默吞掉错误
        return sorted(candidates, key=lambda x: x.get("similarity", 0), reverse=True)


def build_vector_index(
    doc_id: str,
    text: str,
    vector_store_dir: str,
    embedding_model_id: str = "local-minilm",
    api_key: str = None,
    api_host: str = None,
    pages: List[dict] = None
):
    try:
        print(f"Building vector index for {doc_id}...")
        # 使用 Model_ID_Resolver 统一解析模型 ID
        registry_key, config = resolve_model_id(embedding_model_id)
        if registry_key is not None:
            embedding_model_id = registry_key
        else:
            available_models = get_available_model_ids()
            raise ValueError(
                f"Embedding 模型 '{embedding_model_id}' 未配置或不受支持，"
                f"可用模型列表: {available_models}"
            )

        # 分块策略：按模型最大上下文自适应，默认 1200 / 200（约 15-20% 重叠），限制在 1000-2500
        chunk_size, chunk_overlap = get_chunk_params(embedding_model_id, base_chunk_size=1200, base_overlap=200)
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

        # ---- 语义意群生成与意群级别向量索引构建 ----
        _build_semantic_group_index(
            doc_id=doc_id,
            chunks=chunks,
            pages=pages,
            embed_fn=embed_fn,
            api_key=api_key,
        )

    except Exception as e:
        print(f"Error building vector index for {doc_id}: {e}")
        raise


def _build_semantic_group_index(
    doc_id: str,
    chunks: List[str],
    pages: List[dict],
    embed_fn,
    api_key: str = None,
):
    """在分块索引构建完成后，生成语义意群并构建意群级别向量索引

    流程：
    1. 检查 RAGConfig.enable_semantic_groups 是否启用
    2. 从 pages 数据推导每个分块对应的页码（chunk_pages）
    3. 调用 SemanticGroupService.generate_groups 生成意群
    4. 为意群的 digest 文本构建 FAISS 向量索引
    5. 保存意群数据（JSON）和意群向量索引（FAISS + pkl）

    Args:
        doc_id: 文档唯一标识
        chunks: 文本分块列表
        pages: 文档页面数据列表（每个元素包含 page 和 content 字段），可为 None
        embed_fn: 嵌入函数
        api_key: LLM API 密钥（用于意群摘要生成）
    """
    from services.rag_config import RAGConfig
    from services.semantic_group_service import SemanticGroupService

    config = RAGConfig()

    # 检查是否启用语义意群功能
    if not config.enable_semantic_groups:
        logger.info(f"[{doc_id}] 语义意群功能已禁用，跳过意群生成")
        return

    try:
        logger.info(f"[{doc_id}] 开始生成语义意群...")

        # 从 pages 数据推导每个分块对应的页码
        chunk_pages = _derive_chunk_pages(chunks, pages)

        # 创建 SemanticGroupService 实例
        group_service = SemanticGroupService(api_key=api_key or "")

        # 调用 generate_groups 生成语义意群（异步方法需要在同步上下文中运行）
        groups = _run_async(group_service.generate_groups(
            chunks=chunks,
            chunk_pages=chunk_pages,
            target_chars=config.target_group_chars,
            min_chars=config.min_group_chars,
            max_chars=config.max_group_chars,
        ))

        if not groups:
            logger.warning(f"[{doc_id}] 语义意群生成结果为空，跳过意群索引构建")
            return

        logger.info(f"[{doc_id}] 生成了 {len(groups)} 个语义意群")

        # 确定意群数据存储目录
        # vector_store_dir 的父目录是 data/，意群存储在 data/semantic_groups/
        data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(os.path.dirname(data_dir), "data")
        groups_store_dir = os.path.join(data_dir, "semantic_groups")
        os.makedirs(groups_store_dir, exist_ok=True)

        # 保存意群数据为 JSON
        group_service.save_groups(doc_id, groups, groups_store_dir)
        logger.info(f"[{doc_id}] 意群数据已保存到 {groups_store_dir}")

        # 为意群的 digest 文本构建 FAISS 向量索引
        digest_texts = [g.digest for g in groups]
        group_ids = [g.group_id for g in groups]

        if digest_texts:
            group_embeddings = embed_fn(digest_texts)
            dimension = group_embeddings.shape[1]
            group_index = faiss.IndexFlatL2(dimension)
            group_index.add(np.array(group_embeddings).astype('float32'))

            # 保存意群 FAISS 索引
            group_index_path = os.path.join(groups_store_dir, f"{doc_id}_groups.index")
            faiss.write_index(group_index, group_index_path)

            # 保存意群元数据（digest 文本列表和 group_id 映射）
            group_meta_path = os.path.join(groups_store_dir, f"{doc_id}_groups.pkl")
            with open(group_meta_path, "wb") as f:
                pickle.dump({
                    "digest_texts": digest_texts,
                    "group_ids": group_ids,
                }, f)

            logger.info(
                f"[{doc_id}] 意群向量索引已保存: "
                f"index={group_index_path}, meta={group_meta_path}, "
                f"共 {len(groups)} 个意群"
            )

    except Exception as e:
        # 意群生成失败不影响主流程，记录警告并继续
        logger.warning(f"[{doc_id}] 语义意群生成失败，继续使用分块级别索引: {e}")
        print(f"Warning: Semantic group generation failed for {doc_id}: {e}")


def _derive_chunk_pages(chunks: List[str], pages: List[dict]) -> List[int]:
    """从 pages 数据推导每个分块对应的页码

    使用 _find_page_for_chunk 函数将每个分块映射到对应的页码。
    如果 pages 数据不可用，则所有分块默认分配到第 1 页。

    Args:
        chunks: 文本分块列表
        pages: 文档页面数据列表，可为 None

    Returns:
        每个分块对应的页码列表
    """
    if not pages:
        # 没有页面数据时，所有分块默认分配到第 1 页
        return [1] * len(chunks)

    return [_find_page_for_chunk(chunk, pages) for chunk in chunks]


def _run_async(coro):
    """在同步上下文中运行异步协程

    如果当前已有事件循环在运行，则使用 nest_asyncio 或创建新线程；
    否则直接使用 asyncio.run()。

    Args:
        coro: 异步协程对象

    Returns:
        协程的返回值
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # 当前已有事件循环在运行（如在 FastAPI 请求处理中）
        # 使用新线程运行异步任务
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


def _load_group_index(doc_id: str) -> Optional[dict]:
    """加载意群级别 FAISS 索引和元数据

    从 data/semantic_groups/ 目录加载意群的 FAISS 索引文件和 pkl 元数据文件。
    如果文件不存在或加载失败，返回 None。

    Args:
        doc_id: 文档唯一标识

    Returns:
        包含 index、digest_texts、group_ids 的字典，加载失败时返回 None
    """
    # 确定意群数据存储目录
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(os.path.dirname(data_dir), "data")
    groups_store_dir = os.path.join(data_dir, "semantic_groups")

    group_index_path = os.path.join(groups_store_dir, f"{doc_id}_groups.index")
    group_meta_path = os.path.join(groups_store_dir, f"{doc_id}_groups.pkl")

    if not os.path.exists(group_index_path) or not os.path.exists(group_meta_path):
        logger.info(f"[{doc_id}] 意群级别索引不存在，回退到仅分块级别检索")
        return None

    try:
        group_index = faiss.read_index(group_index_path)
        with open(group_meta_path, "rb") as f:
            group_meta = pickle.load(f)

        digest_texts = group_meta.get("digest_texts", [])
        group_ids = group_meta.get("group_ids", [])

        if not digest_texts or not group_ids:
            logger.warning(f"[{doc_id}] 意群元数据为空，回退到仅分块级别检索")
            return None

        logger.info(f"[{doc_id}] 已加载意群级别索引，共 {len(group_ids)} 个意群")
        return {
            "index": group_index,
            "digest_texts": digest_texts,
            "group_ids": group_ids,
        }
    except Exception as e:
        logger.warning(f"[{doc_id}] 加载意群级别索引失败，回退到仅分块级别检索: {e}")
        return None


def _load_group_data(doc_id: str) -> Optional[dict]:
    """加载意群 JSON 数据，获取每个意群包含的 chunk_indices 映射

    用于在 RRF 融合后进行同组 chunk 去重。

    Args:
        doc_id: 文档唯一标识

    Returns:
        group_id -> chunk_indices 的映射字典，加载失败时返回 None
    """
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(os.path.dirname(data_dir), "data")
    groups_json_path = os.path.join(data_dir, "semantic_groups", f"{doc_id}.json")

    if not os.path.exists(groups_json_path):
        return None

    try:
        import json
        with open(groups_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        groups = data.get("groups", [])
        # 构建 group_id -> chunk_indices 映射
        group_chunk_map = {}
        for g in groups:
            group_chunk_map[g["group_id"]] = g.get("chunk_indices", [])

        return group_chunk_map
    except Exception as e:
        logger.warning(f"[{doc_id}] 加载意群 JSON 数据失败: {e}")
        return None


def _search_group_index(
    group_index_data: dict,
    query_vector: np.ndarray,
    search_k: int,
) -> List[dict]:
    """在意群级别 FAISS 索引中搜索

    Args:
        group_index_data: _load_group_index 返回的字典
        query_vector: 查询向量
        search_k: 搜索返回的最大结果数

    Returns:
        意群级别搜索结果列表，每个元素包含 group_id、distance 等信息
    """
    group_index = group_index_data["index"]
    group_ids = group_index_data["group_ids"]

    # 限制搜索数量不超过索引中的向量数
    actual_k = min(search_k, group_index.ntotal)
    if actual_k <= 0:
        return []

    D, I = group_index.search(np.array(query_vector).astype('float32'), actual_k)

    results = []
    for dist, idx in zip(D[0], I[0]):
        if 0 <= idx < len(group_ids):
            results.append({
                "group_id": group_ids[idx],
                "distance": float(dist),
                "group_rank": len(results),  # 在意群搜索中的排名
            })

    return results


def _rrf_merge_chunk_and_group(
    chunk_results: List[dict],
    group_results: List[dict],
    group_chunk_map: Optional[dict],
    chunks: List[str],
    pages: List[dict],
    query: str,
    top_k: int = 10,
    k: int = 60,
) -> List[dict]:
    """使用 RRF 算法融合分块级别和意群级别检索结果

    RRF 公式: score = sum(1 / (k + rank_i)) 对每个排名列表

    融合策略：
    1. 分块级别结果直接参与 RRF 排名
    2. 意群级别结果展开为其包含的所有 chunk，每个 chunk 继承意群的排名
    3. 同一 chunk 在两路结果中的 RRF 分数累加
    4. 同组 chunk 去重：属于同一意群的多个 chunk 只保留 RRF 分数最高的

    Args:
        chunk_results: 分块级别检索结果列表
        group_results: 意群级别检索结果列表
        group_chunk_map: group_id -> chunk_indices 映射，可为 None
        chunks: 所有文本分块列表
        pages: 文档页面数据
        query: 用户查询文本
        top_k: 返回结果数量
        k: RRF 常数（默认 60）

    Returns:
        融合后的结果列表，按 RRF 分数降序排列
    """
    # 步骤 1：计算分块级别的 RRF 分数
    # chunk_text -> rrf_score
    rrf_scores = {}
    # chunk_text -> 原始结果数据
    chunk_data = {}
    # chunk_text -> 所属 group_id（用于去重）
    chunk_group_map = {}

    for rank, item in enumerate(chunk_results):
        chunk_text = item.get("chunk", "")
        if not chunk_text:
            continue
        rrf_score = 1.0 / (k + rank + 1)
        rrf_scores[chunk_text] = rrf_scores.get(chunk_text, 0.0) + rrf_score
        if chunk_text not in chunk_data:
            chunk_data[chunk_text] = item.copy()

    # 步骤 2：将意群级别结果展开为 chunk 级别，计算 RRF 分数
    if group_results and group_chunk_map:
        for rank, group_item in enumerate(group_results):
            group_id = group_item["group_id"]
            chunk_indices = group_chunk_map.get(group_id, [])
            group_rrf_score = 1.0 / (k + rank + 1)

            for chunk_idx in chunk_indices:
                if 0 <= chunk_idx < len(chunks):
                    chunk_text = chunks[chunk_idx]
                    # 累加意群级别的 RRF 分数
                    rrf_scores[chunk_text] = rrf_scores.get(chunk_text, 0.0) + group_rrf_score

                    # 记录 chunk 所属的 group_id
                    if chunk_text not in chunk_group_map:
                        chunk_group_map[chunk_text] = group_id

                    # 如果该 chunk 还没有结果数据，创建一个
                    if chunk_text not in chunk_data:
                        page_num = _find_page_for_chunk(chunk_text, pages)
                        snippet, highlights = _extract_snippet_and_highlights(chunk_text, query)
                        chunk_data[chunk_text] = {
                            "chunk": chunk_text,
                            "page": page_num,
                            "score": 0.0,
                            "similarity": 0.5,
                            "similarity_percent": 50.0,
                            "snippet": snippet,
                            "highlights": highlights,
                            "reranked": False,
                        }

    # 步骤 3：同组 chunk 去重 —— 属于同一意群的多个 chunk 只保留 RRF 分数最高的
    if chunk_group_map:
        # 构建反向映射：chunk_index -> group_id（基于 group_chunk_map）
        chunk_idx_to_group = {}
        if group_chunk_map:
            for gid, indices in group_chunk_map.items():
                for idx in indices:
                    if 0 <= idx < len(chunks):
                        chunk_idx_to_group[chunks[idx]] = gid

        # 按 group_id 分组，每组只保留 RRF 分数最高的 chunk
        # group_id -> (best_chunk_text, best_rrf_score)
        group_best = {}
        chunks_to_remove = set()

        for chunk_text, rrf_score in rrf_scores.items():
            gid = chunk_idx_to_group.get(chunk_text)
            if gid is None:
                # 不属于任何意群的 chunk，保留
                continue

            if gid not in group_best:
                group_best[gid] = (chunk_text, rrf_score)
            else:
                existing_text, existing_score = group_best[gid]
                if rrf_score > existing_score:
                    # 新的 chunk 分数更高，移除旧的
                    chunks_to_remove.add(existing_text)
                    group_best[gid] = (chunk_text, rrf_score)
                else:
                    # 旧的分数更高，移除新的
                    chunks_to_remove.add(chunk_text)

        # 移除被去重的 chunk
        for ct in chunks_to_remove:
            rrf_scores.pop(ct, None)

    # 步骤 4：按 RRF 分数排序并返回 top_k 结果
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for chunk_text, rrf_score in sorted_chunks[:top_k]:
        item = chunk_data.get(chunk_text, {})
        if not item:
            continue
        item = item.copy()
        item["rrf_score"] = rrf_score
        item["hybrid"] = True
        results.append(item)

    return results


def _is_table_fragment(text: str) -> bool:
    """检测文本是否为表格/数据碎片

    表格碎片特征：
    - 大量孤立的数字（被空格分隔的短数字序列）
    - 缺少完整句子（没有句号结尾的长句）
    - 高比例的数字 token vs 文字 token
    """
    if not text or len(text) < 20:
        return False

    # 按空格拆分为 token
    tokens = text.split()
    if len(tokens) < 3:
        return False

    # 统计数字 token（纯数字或小数）和文字 token
    num_tokens = 0
    for t in tokens:
        cleaned = t.strip('(),%↑↓·-')
        if not cleaned:
            continue
        # 纯数字、小数、百分比、带单位的数字（如 2.0m, 1.5m）
        if re.match(r'^-?\d+\.?\d*[a-zA-Z]?$', cleaned):
            num_tokens += 1

    num_ratio = num_tokens / len(tokens) if tokens else 0

    # 检查是否有完整句子（至少一个 10+ 字符的句子以句号结尾）
    sentences = re.split(r'[.!?。！？]', text)
    has_real_sentence = any(len(s.strip()) > 30 for s in sentences)

    # 数字 token 占比 > 35% 且没有完整句子 → 表格碎片
    if num_ratio > 0.35 and not has_real_sentence:
        return True

    # 数字 token 占比 > 50% → 几乎肯定是表格
    if num_ratio > 0.5:
        return True

    return False


def _phrase_boost(results: List[dict], query: str, boost_factor: float = 1.5) -> List[dict]:
    """对包含完整查询短语的 chunk 进行相似度加权提升

    向量检索是语义匹配，可能把只包含部分关键词的碎片排在前面。
    此函数检查每个 chunk 是否包含完整的查询短语（忽略大小写），
    如果包含则提升其 similarity 和 similarity_percent。

    同时对"表格碎片"进行降权。

    Args:
        results: 搜索结果列表
        query: 用户查询文本
        boost_factor: 提升倍数（默认 1.5）

    Returns:
        重新排序后的结果列表
    """
    if not results or not query or len(query.strip()) < 2:
        return results

    query_lower = query.lower().strip()
    # 将查询拆分为单词，用于计算覆盖率
    query_terms = [t for t in re.split(r"[\s,;，。；、]+", query_lower) if len(t) > 1]

    for item in results:
        chunk_text = item.get("chunk", "")
        chunk_lower = chunk_text.lower()
        if not chunk_lower:
            continue

        # 检测表格碎片：降权到 0.5x
        if _is_table_fragment(chunk_text):
            item["similarity"] = item.get("similarity", 0) * 0.5
            item["similarity_percent"] = round(item.get("similarity_percent", 0) * 0.5, 2)
            item["table_fragment"] = True
            continue

        # 完整短语匹配：最大提升
        if query_lower in chunk_lower:
            item["similarity"] = min(item.get("similarity", 0) * boost_factor, 1.0)
            item["similarity_percent"] = min(round(item.get("similarity_percent", 0) * boost_factor, 2), 99.99)
            item["phrase_match"] = True
            continue

        # 部分词覆盖率加权：覆盖越多提升越大
        if query_terms:
            matched = sum(1 for t in query_terms if t in chunk_lower)
            coverage = matched / len(query_terms)
            if coverage >= 0.8:
                factor = 1.0 + (boost_factor - 1.0) * coverage * 0.5
                item["similarity"] = min(item.get("similarity", 0) * factor, 1.0)
                item["similarity_percent"] = min(round(item.get("similarity_percent", 0) * factor, 2), 99.99)

    # 按调整后的 similarity 重新排序
    return sorted(results, key=lambda x: x.get("similarity", 0), reverse=True)


def search_document_chunks(
    doc_id: str,
    query: str,
    vector_store_dir: str,
    pages: List[dict],
    api_key: str = None,
    top_k: int = 10,
    candidate_k: int = 20,
    use_rerank: bool = False,
    reranker_model: Optional[str] = None,
    rerank_provider: Optional[str] = None,
    rerank_api_key: Optional[str] = None,
    rerank_endpoint: Optional[str] = None,
    use_hybrid: bool = True
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

    vector_results = []
    vector_chunk_set = set()  # 记录向量搜索已返回的 chunk
    for dist, idx in zip(D[0], I[0]):
        if idx < len(chunks):
            chunk_text = chunks[idx]
            page_num = _find_page_for_chunk(chunk_text, pages)
            similarity = _distance_to_similarity(float(dist))
            snippet, highlights = _extract_snippet_and_highlights(chunk_text, query)

            vector_results.append({
                "chunk": chunk_text,
                "page": page_num,
                "score": float(dist),
                "similarity": similarity,
                "similarity_percent": round(similarity * 100, 2),
                "snippet": snippet,
                "highlights": highlights,
                "reranked": False
            })
            vector_chunk_set.add(chunk_text)

    # --- 精确短语注入 ---
    # 如果查询包含多个词（短语），扫描所有 chunk 找到包含完整短语的，
    # 注入到结果中（如果向量搜索没返回它们）
    query_lower = query.lower().strip()
    if len(query_lower) > 3 and " " in query_lower:
        phrase_injected = 0
        for chunk_text in chunks:
            if chunk_text in vector_chunk_set:
                continue
            if query_lower in chunk_text.lower():
                page_num = _find_page_for_chunk(chunk_text, pages)
                snippet, highlights = _extract_snippet_and_highlights(chunk_text, query)
                vector_results.append({
                    "chunk": chunk_text,
                    "page": page_num,
                    "score": 0.0,
                    "similarity": 0.95,  # 精确短语匹配给高分
                    "similarity_percent": 95.0,
                    "snippet": snippet,
                    "highlights": highlights,
                    "reranked": False,
                    "phrase_match": True,
                })
                vector_chunk_set.add(chunk_text)
                phrase_injected += 1
                if phrase_injected >= 3:  # 最多注入 3 个
                    break
        if phrase_injected > 0:
            logger.info(f"[精确短语注入] 查询 '{query}' 注入了 {phrase_injected} 个包含完整短语的 chunk")

    # --- 短语匹配加权 + 表格碎片降权 ---
    vector_results = _phrase_boost(vector_results, query)

    # --- BM25混合检索 ---
    if use_hybrid and not use_rerank:
        try:
            from services.bm25_service import bm25_search
            from services.hybrid_search import hybrid_search_merge

            bm25_results = bm25_search(doc_id, query, chunks, top_k=search_k)
            # 为BM25结果补充page信息
            for item in bm25_results:
                item['page'] = _find_page_for_chunk(item['chunk'], pages)

            results = hybrid_search_merge(vector_results, bm25_results, top_k=top_k)
            # 补充snippet/highlights（BM25结果可能缺少）
            for item in results:
                if 'snippet' not in item or not item.get('snippet'):
                    snippet, highlights = _extract_snippet_and_highlights(item['chunk'], query)
                    item['snippet'] = snippet
                    item['highlights'] = highlights
                if 'similarity' not in item:
                    item['similarity'] = 0.5
                    item['similarity_percent'] = 50.0

            print(f"[Hybrid] 向量: {len(vector_results)}条, BM25: {len(bm25_results)}条, 融合后: {len(results)}条")

            # --- 意群级别检索 + RRF 融合（在 BM25 混合检索之后） ---
            results = _merge_with_group_search(
                doc_id=doc_id,
                chunk_results=results,
                query_vector=query_vector,
                chunks=chunks,
                pages=pages,
                query=query,
                top_k=top_k,
            )

            return results
        except Exception as e:
            print(f"[Hybrid] BM25混合检索失败，回退到纯向量检索: {e}")
            # 回退到纯向量检索

    # --- 纯向量检索（fallback或rerank模式） ---
    if use_rerank:
        results = _apply_rerank(
            query,
            vector_results,
            reranker_model,
            rerank_provider,
            rerank_api_key,
            rerank_endpoint
        )
        # rerank 后也做短语加权和表格碎片降权
        results = _phrase_boost(results, query, boost_factor=1.2)
    else:
        results = sorted(vector_results, key=lambda x: x.get("similarity", 0), reverse=True)

    results = results[:top_k]

    # --- 意群级别检索 + RRF 融合（在纯向量/rerank 检索之后） ---
    results = _merge_with_group_search(
        doc_id=doc_id,
        chunk_results=results,
        query_vector=query_vector,
        chunks=chunks,
        pages=pages,
        query=query,
        top_k=top_k,
    )

    return results


def _merge_with_group_search(
    doc_id: str,
    chunk_results: List[dict],
    query_vector: np.ndarray,
    chunks: List[str],
    pages: List[dict],
    query: str,
    top_k: int = 10,
) -> List[dict]:
    """尝试加载意群级别索引并与分块结果进行 RRF 融合

    如果意群索引不存在或加载失败，直接返回原始分块结果（需求 6.3 降级回退）。

    Args:
        doc_id: 文档唯一标识
        chunk_results: 分块级别检索结果
        query_vector: 查询向量
        chunks: 所有文本分块列表
        pages: 文档页面数据
        query: 用户查询文本
        top_k: 返回结果数量

    Returns:
        融合后的结果列表，或原始分块结果（降级时）
    """
    from services.rag_config import RAGConfig

    config = RAGConfig()

    # 检查是否启用语义意群功能
    if not config.enable_semantic_groups:
        logger.info(f"[{doc_id}] 语义意群功能已禁用，使用分块级别检索结果")
        return chunk_results

    try:
        # 加载意群级别索引
        group_index_data = _load_group_index(doc_id)
        if group_index_data is None:
            # 意群索引不存在，回退到仅分块级别检索（需求 6.3）
            return chunk_results

        # 在意群级别索引中搜索
        group_results = _search_group_index(
            group_index_data=group_index_data,
            query_vector=query_vector,
            search_k=top_k,
        )

        if not group_results:
            logger.info(f"[{doc_id}] 意群级别检索无结果，使用分块级别检索结果")
            return chunk_results

        # 加载意群 JSON 数据获取 chunk_indices 映射
        group_chunk_map = _load_group_data(doc_id)

        # 使用 RRF 融合分块和意群两路结果
        merged_results = _rrf_merge_chunk_and_group(
            chunk_results=chunk_results,
            group_results=group_results,
            group_chunk_map=group_chunk_map,
            chunks=chunks,
            pages=pages,
            query=query,
            top_k=top_k,
            k=60,  # 标准 RRF 常数
        )

        logger.info(
            f"[{doc_id}] RRF 融合完成: "
            f"分块结果={len(chunk_results)}条, "
            f"意群结果={len(group_results)}条, "
            f"融合后={len(merged_results)}条"
        )

        return merged_results

    except Exception as e:
        # 意群检索失败不影响主流程，回退到分块级别检索
        logger.warning(f"[{doc_id}] 意群级别检索失败，回退到分块级别检索: {e}")
        return chunk_results


def _get_semantic_groups_dir() -> str:
    """获取语义意群数据存储目录路径

    Returns:
        语义意群数据目录的绝对路径
    """
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(os.path.dirname(data_dir), "data")
    return os.path.join(data_dir, "semantic_groups")


def get_relevant_context(
    doc_id: str,
    query: str,
    vector_store_dir: str,
    pages: List[dict],
    api_key: str = None,
    top_k: int = 10,  # 增加到10
    use_rerank: bool = False,
    reranker_model: Optional[str] = None,
    candidate_k: int = 20,
    rerank_provider: Optional[str] = None,
    rerank_api_key: Optional[str] = None,
    rerank_endpoint: Optional[str] = None
) -> Tuple[str, dict]:
    """获取与查询相关的上下文文本和检索元数据

    集成 GranularitySelector、TokenBudgetManager、ContextBuilder 和 RetrievalLogger，
    实现混合粒度检索策略。当语义意群可用时，使用智能粒度选择和 Token 预算管理；
    否则回退到原有的简单拼接逻辑。

    Args:
        doc_id: 文档唯一标识
        query: 用户查询文本
        vector_store_dir: 向量索引存储目录
        pages: 文档页面数据列表
        api_key: API 密钥
        top_k: 返回结果数量
        use_rerank: 是否使用重排序
        reranker_model: 重排序模型
        candidate_k: 候选结果数量
        rerank_provider: 重排序提供商
        rerank_api_key: 重排序 API 密钥
        rerank_endpoint: 重排序端点

    Returns:
        (context_string, retrieval_meta) 元组
        - context_string: 格式化的上下文字符串
        - retrieval_meta: 检索元数据字典，包含 query_type、granularities、
          token_used、fallback、citations 等信息
    """
    from services.rag_config import RAGConfig
    from services.semantic_group_service import SemanticGroupService
    from services.granularity_selector import GranularitySelector
    from services.token_budget import TokenBudgetManager
    from services.context_builder import ContextBuilder
    from services.retrieval_logger import RetrievalLogger, RetrievalTrace

    # 获取搜索结果
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

    config = RAGConfig()

    # 尝试使用语义意群增强检索
    if config.enable_semantic_groups:
        try:
            context_str, retrieval_meta = _build_context_with_groups(
                doc_id=doc_id,
                query=query,
                results=results,
                config=config,
            )
            if context_str is not None:
                return context_str, retrieval_meta
        except Exception as e:
            # 意群增强失败，回退到简单拼接
            logger.warning(f"[{doc_id}] 意群增强检索失败，回退到简单拼接: {e}")

    # 回退逻辑：使用原有的简单拼接
    relevant_chunks = [item["chunk"] for item in results]
    context_string = "\n\n...\n\n".join(relevant_chunks)

    # 构建回退情况下的 retrieval_meta
    fallback_type = "groups_disabled" if not config.enable_semantic_groups else "index_missing"
    retrieval_logger = RetrievalLogger()
    trace = RetrievalTrace(
        query=query,
        query_type="unknown",
        query_confidence=0.0,
        chunk_hits=len(results),
        group_hits=0,
        token_budget=config.max_token_budget,
        token_reserved=config.reserve_for_answer,
        token_used=0,
        fallback_type=fallback_type,
        fallback_detail=f"回退到简单拼接逻辑，原因: {fallback_type}",
    )
    retrieval_logger.log_trace(trace)
    retrieval_meta = retrieval_logger.to_retrieval_meta(trace)

    return context_string, retrieval_meta


def _build_context_with_groups(
    doc_id: str,
    query: str,
    results: List[dict],
    config,
) -> Tuple[Optional[str], dict]:
    """使用语义意群构建增强上下文

    流程：
    1. 加载语义意群数据
    2. 使用 GranularitySelector.select_mixed 分配混合粒度
    3. 使用 TokenBudgetManager.fit_within_budget 调整 Token 预算
    4. 使用 ContextBuilder.build_context 构建格式化上下文
    5. 使用 RetrievalLogger 记录检索追踪

    Args:
        doc_id: 文档唯一标识
        query: 用户查询文本
        results: search_document_chunks 返回的搜索结果
        config: RAGConfig 配置对象

    Returns:
        (context_string, retrieval_meta) 元组，如果意群不可用返回 (None, {})
    """
    from services.semantic_group_service import SemanticGroupService
    from services.granularity_selector import GranularitySelector
    from services.token_budget import TokenBudgetManager
    from services.context_builder import ContextBuilder
    from services.retrieval_logger import RetrievalLogger, RetrievalTrace

    # 步骤 1：加载语义意群数据
    groups_store_dir = _get_semantic_groups_dir()
    group_service = SemanticGroupService()
    groups = group_service.load_groups(doc_id, groups_store_dir)

    if not groups:
        logger.info(f"[{doc_id}] 语义意群数据不可用，回退到简单拼接")
        return None, {}

    logger.info(f"[{doc_id}] 已加载 {len(groups)} 个语义意群，开始构建增强上下文")

    # 步骤 2：根据搜索结果对意群进行排序
    # 将搜索结果中的 chunk 映射回对应的意群，按 RRF/相关性排序
    ranked_groups = _rank_groups_by_results(groups, results)

    if not ranked_groups:
        logger.info(f"[{doc_id}] 无法将搜索结果映射到意群，回退到简单拼接")
        return None, {}

    # 步骤 3：使用 GranularitySelector 分配混合粒度
    selector = GranularitySelector()

    # 先获取查询类型对应的最大意群数限制
    selection_info = selector.select(query=query, groups=groups, max_tokens=config.max_token_budget)
    max_groups = selection_info.max_groups

    # 截断排序后的意群列表，避免引入过多低相关性意群
    ranked_groups_limited = ranked_groups[:max_groups]

    mixed_selections = selector.select_mixed(
        query=query,
        ranked_groups=ranked_groups_limited,
        max_tokens=config.max_token_budget,
    )

    # 步骤 4：使用 TokenBudgetManager 调整 Token 预算
    budget_manager = TokenBudgetManager(
        max_tokens=config.max_token_budget,
        reserve_for_answer=config.reserve_for_answer,
    )
    fitted_selections = budget_manager.fit_within_budget(mixed_selections)

    # 步骤 5：使用 ContextBuilder 构建格式化上下文
    context_builder = ContextBuilder()
    context_string, citations = context_builder.build_context(fitted_selections)

    # 步骤 6：计算实际使用的 Token 数
    token_used = sum(item.get("tokens", 0) for item in fitted_selections)

    # 步骤 7：使用 RetrievalLogger 记录检索追踪
    # 查询类型已在步骤 3 中获取（selection_info）

    retrieval_logger = RetrievalLogger()
    trace = RetrievalTrace(
        query=query,
        query_type=selection_info.query_type,
        query_confidence=1.0,
        chunk_hits=len(results),
        group_hits=len(ranked_groups),
        rrf_top_k=[
            {"group_id": g.group_id, "rank": i, "source": "rrf"}
            for i, g in enumerate(ranked_groups[:10])
        ],
        token_budget=config.max_token_budget,
        token_reserved=config.reserve_for_answer,
        token_used=token_used,
        granularity_assignments=[
            {"group_id": item["group"].group_id, "granularity": item["granularity"]}
            for item in fitted_selections
        ],
        fallback_type=None,
        fallback_detail=None,
        citations=citations,
    )
    retrieval_logger.log_trace(trace)
    retrieval_meta = retrieval_logger.to_retrieval_meta(trace)

    logger.info(
        f"[{doc_id}] 增强上下文构建完成: "
        f"意群数={len(fitted_selections)}, "
        f"Token 使用={token_used}/{budget_manager.available_tokens}, "
        f"查询类型={selection_info.query_type}"
    )

    return context_string, retrieval_meta


def _rank_groups_by_results(
    groups: list,
    results: List[dict],
) -> list:
    """根据搜索结果对语义意群进行排序

    将搜索结果中的 chunk 文本映射回对应的语义意群，
    按 chunk 在搜索结果中的排名对意群进行排序（去重，保留最高排名）。
    过滤掉相关性分数过低的结果，避免引入不相关的意群。

    Args:
        groups: 语义意群列表
        results: search_document_chunks 返回的搜索结果

    Returns:
        按相关性排序的语义意群列表（最相关的在前）
    """
    if not groups or not results:
        return []

    # 构建 chunk 文本到意群的映射
    # 每个意群的 full_text 是其所有 chunk 的拼接，
    # 需要检查搜索结果中的 chunk 是否属于某个意群
    group_scores = {}  # group_id -> 最佳排名（越小越好）
    group_similarity = {}  # group_id -> 最佳相似度分数

    for rank, result in enumerate(results):
        chunk_text = result.get("chunk", "")
        if not chunk_text:
            continue

        # 获取该 chunk 的相似度分数
        similarity = result.get("similarity", 0.0)

        # 查找该 chunk 属于哪个意群
        for group in groups:
            # 检查 chunk 文本是否是意群全文的子串
            if chunk_text in group.full_text:
                if group.group_id not in group_scores:
                    group_scores[group.group_id] = rank
                    group_similarity[group.group_id] = similarity
                else:
                    # 保留最高排名（最小的 rank 值）
                    if rank < group_scores[group.group_id]:
                        group_scores[group.group_id] = rank
                    # 保留最高相似度
                    group_similarity[group.group_id] = max(
                        group_similarity[group.group_id], similarity
                    )
                break  # 一个 chunk 只属于一个意群

    # 过滤掉相关性过低的意群
    # 策略：如果最佳意群的相似度 > 0.5，则过滤掉相似度低于最佳值 40% 的意群
    if group_similarity:
        best_similarity = max(group_similarity.values())
        if best_similarity > 0.5:
            threshold = best_similarity * 0.4
            filtered_ids = {
                gid for gid, sim in group_similarity.items()
                if sim >= threshold
            }
            removed = set(group_scores.keys()) - filtered_ids
            if removed:
                logger.info(
                    f"相关性过滤：移除 {len(removed)} 个低相关意群 "
                    f"(阈值={threshold:.3f}, 最佳={best_similarity:.3f})"
                )
            group_scores = {gid: r for gid, r in group_scores.items() if gid in filtered_ids}

    # 按排名排序意群
    sorted_group_ids = sorted(group_scores.keys(), key=lambda gid: group_scores[gid])

    # 构建 group_id -> group 对象的映射
    group_map = {g.group_id: g for g in groups}

    ranked_groups = [group_map[gid] for gid in sorted_group_ids if gid in group_map]

    return ranked_groups
