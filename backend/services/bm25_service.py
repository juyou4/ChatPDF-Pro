"""
BM25检索服务 - 轻量级关键词检索，作为向量检索的补充

设计哲学（参考paper-burner-x）：
- BM25作为基础检索，始终可用，不依赖外部API
- 向量检索作为增强，提供语义理解
- 两者结合使用，互补优势

实现说明：
- 纯Python实现，无需额外依赖（不需要rank-bm25/jieba）
- 中文使用字符级unigram+bigram分词（效果接近jieba，零依赖）
- 英文使用空格分词+小写化
- 支持内存缓存，避免重复构建索引
"""
import math
import logging
import re
import threading
from collections import Counter
from typing import Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)

BM25_TOKENIZER_VERSION = 3

try:
    import jieba
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False

def _should_use_jieba() -> bool:
    """判断当前请求是否使用 jieba 分词。"""
    if not _HAS_JIEBA:
        return False
    try:
        from services.rag_config import should_use_jieba_bm25
        return should_use_jieba_bm25()
    except Exception:
        return _HAS_JIEBA


def _tokenizer_signature(use_jieba: bool) -> str:
    mode = "jieba" if use_jieba else "char_ngram"
    return f"v{BM25_TOKENIZER_VERSION}:{mode}"


def get_bm25_tokenizer_signature() -> str:
    return _tokenizer_signature(_should_use_jieba())


def _merge_token_channels(*channels: List[str]) -> List[str]:
    """合并不同分词通道，去除特征重叠但保留文本中的真实词频。"""
    normalized_channels = [
        [str(token).strip() for token in channel if str(token).strip()]
        for channel in channels
    ]
    target_counts: Counter[str] = Counter()
    for channel in normalized_channels:
        channel_counts = Counter(channel)
        for token, count in channel_counts.items():
            target_counts[token] = max(target_counts[token], count)

    merged: List[str] = []
    emitted: Counter[str] = Counter()
    for channel in normalized_channels:
        for token in channel:
            if emitted[token] >= target_counts[token]:
                continue
            merged.append(token)
            emitted[token] += 1
    return merged


def _tokenize(text: str, *, use_jieba: Optional[bool] = None) -> List[str]:
    """
    混合分词：中文用 jieba 分词（可选）或字符级 n-gram，英文用空格分词

    当 jieba 可用且配置启用时：
    - 中文：jieba 分词结果 + bigram（兼顾词语和短语匹配）
    - 英文：整词 + 小写化
    - 数字：保留（用于匹配公式编号、年份等）

    当 jieba 不可用时回退到字符级 unigram+bigram+trigram（零依赖）。
    """
    if not text:
        return []

    tokens = []
    text = text.lower()

    # 分离中文和英文/数字片段
    segments = re.findall(r'[\u4e00-\u9fff]+|[a-z0-9]+', text)

    if use_jieba is None:
        use_jieba = _should_use_jieba()

    for seg in segments:
        if re.match(r'[\u4e00-\u9fff]', seg):
            if use_jieba:
                words = [word for word in jieba.cut(seg) if word.strip()]
                word_tokens = list(words)
                # 相邻词组合保留词组级精度。
                for i in range(len(words) - 1):
                    word_tokens.append(words[i] + words[i + 1])
                # jieba 会把“编程语言”之类的复合词保留为一个词；查询“编程”
                # 因而无法命中。补充字符级 n-gram，保留词级精度同时恢复子词召回。
                character_tokens = []
                for i in range(len(seg) - 1):
                    character_tokens.append(seg[i:i + 2])
                for i in range(len(seg) - 2):
                    character_tokens.append(seg[i:i + 3])
                tokens.extend(_merge_token_channels(word_tokens, character_tokens))
            else:
                # 回退：unigram + bigram + trigram
                for ch in seg:
                    tokens.append(ch)
                for i in range(len(seg) - 1):
                    tokens.append(seg[i:i+2])
                for i in range(len(seg) - 2):
                    tokens.append(seg[i:i+3])
        else:
            # 英文/数字：整词
            if len(seg) > 1:
                tokens.append(seg)

    return tokens


class BM25Index:
    """
    BM25Okapi实现
    
    参数：
    - k1: 词频饱和参数，默认1.5
    - b: 文档长度归一化参数，默认0.75
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        *,
        use_jieba: Optional[bool] = None,
    ):
        self.k1 = k1
        self.b = b
        self.use_jieba = _should_use_jieba() if use_jieba is None else bool(use_jieba)
        self.tokenizer_version = BM25_TOKENIZER_VERSION
        self.tokenizer_signature = _tokenizer_signature(self.use_jieba)
        self.doc_count = 0
        self.avg_dl = 0.0
        self.doc_lengths: List[int] = []
        self.doc_freqs: Dict[str, int] = {}  # term -> 出现该term的文档数
        self.term_freqs: List[Dict[str, int]] = []  # 每个文档的term频率
        self.idf: Dict[str, float] = {}
        self.chunks: List[str] = []
        # 倒排索引: term -> 包含该 term 的文档索引列表
        self.inverted_index: Dict[str, List[int]] = {}

    def build(self, chunks: List[str]):
        """构建BM25索引"""
        self.tokenizer_version = BM25_TOKENIZER_VERSION
        self.tokenizer_signature = _tokenizer_signature(self.use_jieba)
        # 调用方通常会复用 pages/chunks 列表；必须保存快照，否则原地修改
        # 会让缓存内容校验失效，并造成倒排索引与返回文本错位。
        self.chunks = list(chunks)
        self.doc_count = len(self.chunks)
        self.doc_lengths = []
        self.doc_freqs = {}
        self.term_freqs = []

        for chunk in self.chunks:
            tokens = _tokenize(chunk, use_jieba=self.use_jieba)
            self.doc_lengths.append(len(tokens))

            # 统计该文档的term频率
            tf: Dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.term_freqs.append(tf)

            # 统计文档频率（每个term在多少文档中出现）
            seen = set(tf.keys())
            for token in seen:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.avg_dl = sum(self.doc_lengths) / max(self.doc_count, 1)

        # 构建倒排索引: term -> 包含该 term 的文档索引列表
        self.inverted_index = {}
        for i, tf in enumerate(self.term_freqs):
            for term in tf:
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append(i)

        # 预计算IDF
        self.idf = {}
        for term, df in self.doc_freqs.items():
            # BM25 IDF公式
            self.idf[term] = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str) -> List[float]:
        """计算查询与所有文档的BM25分数（使用倒排索引加速）"""
        query_tokens = _tokenize(query, use_jieba=self.use_jieba)
        scores = [0.0] * self.doc_count

        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]

            # 通过倒排索引只访问包含该 term 的文档，避免 O(D) 全量遍历
            for i in self.inverted_index.get(token, ()):
                tf = self.term_freqs[i][token]
                dl = self.doc_lengths[i]
                # BM25 scoring
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avg_dl, 1))
                scores[i] += idf_val * numerator / denominator

        return scores

    def score_with_expansion(self, query: str) -> List[float]:
        """带同义词扩展和细粒度分词的 BM25 评分

        扩展策略（参考 ragflow query.py）：
        1. 原始 token: 权重 1.0
        2. 同义词 token: 权重 0.4（由 SynonymDict.synonym_weight 控制）
        3. 细粒度子词: 权重 0.3（长中文词拆分为 bigram）
        """
        from services.synonym_service import get_synonym_dict, _fine_grained_tokenize

        query_tokens = _tokenize(query, use_jieba=self.use_jieba)
        syn_dict = get_synonym_dict()

        # 扩展：原始 token + 同义词（带权重）
        weighted_tokens = syn_dict.expand_tokens(query_tokens)

        # 细粒度分词补充：对长中文词生成 bigram 子词
        fine_grained_additions: List[Tuple[str, float]] = []
        seen = {t for t, _ in weighted_tokens}
        for token in query_tokens:
            for sub in _fine_grained_tokenize(token):
                if sub not in seen:
                    fine_grained_additions.append((sub, 0.3))
                    seen.add(sub)
        weighted_tokens.extend(fine_grained_additions)

        scores = [0.0] * self.doc_count

        for token, weight in weighted_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]

            for i in self.inverted_index.get(token, ()):
                tf = self.term_freqs[i][token]
                dl = self.doc_lengths[i]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avg_dl, 1))
                scores[i] += weight * idf_val * numerator / denominator

        return scores

    def search(self, query: str, top_k: int = 10, expand_synonyms: bool = True) -> List[dict]:
        """
        BM25检索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            expand_synonyms: 是否启用同义词扩展（默认 True）

        Returns:
            排序后的结果列表，每项包含 chunk, score, index
        """
        if not self.chunks:
            return []

        if expand_synonyms and _should_expand_synonyms():
            scores = self.score_with_expansion(query)
        else:
            scores = self.score(query)

        # 获取top_k结果
        indexed_scores = [(i, s) for i, s in enumerate(scores) if s > 0]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, sc in indexed_scores[:top_k]:
            results.append({
                'chunk': self.chunks[idx],
                'score': sc,
                'index': idx
            })

        return results


# ============================================================
# 同义词扩展开关
# ============================================================
def _should_expand_synonyms() -> bool:
    """判断是否启用 BM25 同义词扩展。

    委托给 ``services.rag_config.should_expand_bm25_synonyms`` 以支持 per-request
    ContextVar 覆盖。全局 settings 作为兜底。
    """
    try:
        from services.rag_config import should_expand_bm25_synonyms
        return should_expand_bm25_synonyms()
    except Exception:
        try:
            from config import settings
            return bool(getattr(settings, "bm25_expand_synonyms", True))
        except Exception:
            return True


# ============================================================
# 全局BM25索引缓存（按 doc_id + tokenizer signature）
# ============================================================
_bm25_cache: Dict[Tuple[str, str], BM25Index] = {}
_bm25_cache_lock = threading.RLock()


def get_or_build_bm25(doc_id: str, chunks: List[str]) -> BM25Index:
    """获取或构建BM25索引（带缓存）"""
    use_jieba = _should_use_jieba()
    signature = _tokenizer_signature(use_jieba)
    cache_key = (doc_id, signature)
    chunks_snapshot = list(chunks)

    with _bm25_cache_lock:
        cached = _bm25_cache.get(cache_key)
        if (
            cached is not None
            and cached.chunks == chunks_snapshot
            and getattr(cached, "tokenizer_signature", "") == signature
        ):
            return cached

    # 构建使用实例冻结的分词模式，不受其他并发请求影响。
    candidate = BM25Index(use_jieba=use_jieba)
    candidate.build(chunks_snapshot)
    with _bm25_cache_lock:
        cached = _bm25_cache.get(cache_key)
        if (
            cached is not None
            and cached.chunks == chunks_snapshot
            and getattr(cached, "tokenizer_signature", "") == signature
        ):
            return cached
        _bm25_cache[cache_key] = candidate
        return candidate


def clear_bm25_cache(doc_id: Optional[str] = None):
    """清除BM25缓存"""
    with _bm25_cache_lock:
        if doc_id:
            for cache_key in [key for key in _bm25_cache if key[0] == doc_id]:
                _bm25_cache.pop(cache_key, None)
        else:
            _bm25_cache.clear()


def bm25_search(doc_id: str, query: str, chunks: List[str], top_k: int = 10) -> List[dict]:
    """
    便捷函数：BM25检索
    
    自动构建/复用索引，返回top_k结果
    """
    idx = get_or_build_bm25(doc_id, chunks)
    return idx.search(query, top_k)
