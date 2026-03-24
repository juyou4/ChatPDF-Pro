# 问题：LLM 回答内容过短（检索上下文不足）

## 现象

上传 PDF 后提问（如"请总结本文的主要内容"），LLM 回答只有几句话，远不够详尽。
同时发现引用格式为 `para-X-seg-Y`（段落兜底），而非 `group-X`（意群路径）。

---

## 根因分析

问题由三层原因叠加导致：

### 1. 页面文本键名不一致（content vs text）

`extract_text_from_pdf` 将页面内容存为 `content` 键，但下游代码（检索、分块、上下文组装）统一读 `text` 键。导致 `pages[i].text` 为空，全文内容丢失。

**文件**: `backend/routes/document_routes.py`

### 2. 超大段落不拆分导致分块异常

`_merge_segments_into_chunks` 对超过 `chunk_size` 的普通段落直接跳过拆分，导致 14 页 PDF 只产生 4 个巨型 chunk（第一个 33796 字符）。

**文件**: `backend/services/embedding_service.py`

### 3. 意群摘要生成失败 + 粒度未升级

LLM API Key 失效时，`SemanticGroupService` 的摘要生成全部 failed，回退到 `full_text[:80]` 截断。但 `GranularitySelector` 未检测 `summary_status=failed`，仍分配 `summary` 粒度，导致每个意群只提供 80 字符上下文。

**文件**: `backend/services/granularity_selector.py`, `backend/services/semantic_group_service.py`

### 4. 向量检索失败时回退上下文过少

当 embedding API Key 失效导致向量检索完全失败时，系统回退到 `full_text[:8000]`，且 `_build_numbered_context_and_citations` 仅选取 top 8 个 ~240 字符窗口作为上下文（约 1920 字符），LLM 可见内容极少。

**文件**: `backend/routes/chat_routes.py`

---

## 修复方案

### 修复 1：页面键名规范化

在文档加载和上传时统一规范化 page 字典，确保同时有 `text` 和 `content` 键：

```python
# backend/routes/document_routes.py
def _normalize_page_keys(data: dict):
    for page in data.get("data", {}).get("pages", []):
        if "text" not in page and "content" in page:
            page["text"] = page["content"]
        elif "content" not in page and "text" in page:
            page["content"] = page["text"]
```

加载文档和上传文档时均调用此函数。

### 修复 2：超大段落自动拆分

在 `_merge_segments_into_chunks` 中，对超过 `chunk_size` 的非保护段落使用 `RecursiveCharacterTextSplitter` 拆分：

```python
# backend/services/embedding_service.py -> _merge_segments_into_chunks
if seg_len > chunk_size:
    _commit_chunk()
    _oversized_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len,
    )
    for sub in _oversized_splitter.split_text(seg_text):
        chunks.append((sub, active_heading))
    continue
```

### 修复 3：摘要失败时粒度自动升级

在 `GranularitySelector.select_mixed` 和 `select_dynamic` 中，检测 `summary_status=failed` 时自动升级粒度：

```python
# backend/services/granularity_selector.py
# summary→digest, digest→full
if getattr(group, "summary_status", "ok") == "failed":
    if granularity == "summary":
        granularity = "digest"
    elif granularity == "digest":
        granularity = "full"
```

### 修复 4：回退路径上下文扩容

1. `full_text` 截取量从 8000 → 30000 字符
2. `_build_numbered_context_and_citations` 输出**全部段落**作为 context（而非仅 top 8 窗口），citations 仍只追踪最佳匹配窗口用于高亮

```python
# backend/routes/chat_routes.py -> _build_numbered_context_and_citations
# 将所有段落加入 context，但仅对 selected 生成 citation
all_formatted = [para for para in paragraphs]
formatted_context = "\n\n".join(all_formatted)
```

---

## 效果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 页面文本 | `text` 为空 | `text`/`content` 均有值 |
| 分块数量 | 4 个（最大 33796c） | 55 个（正常大小） |
| 意群粒度（failed时） | summary=80c | digest=1000c 或 full |
| 回退上下文量 | ~1920c（8 窗口） | 全部段落（最多 30000c） |
| 检索相关度 | 回退兜底 | 97%（向量检索正常时） |

---

## 涉及文件

- `backend/routes/document_routes.py` — `_normalize_page_keys` 函数
- `backend/services/embedding_service.py` — `_merge_segments_into_chunks` 超大段落拆分
- `backend/services/granularity_selector.py` — `select_mixed` / `select_dynamic` 摘要失败升级
- `backend/routes/chat_routes.py` — `_build_numbered_context_and_citations` 全段落输出 + `full_text[:30000]`
