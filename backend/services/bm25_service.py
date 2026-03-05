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
import re
from typing import Dict, List, Optional, Tuple


try:
    import jieba
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False

# 运行时开关：由 config.settings.bm25_use_jieba 控制
_use_jieba: Optional[bool] = None


def _should_use_jieba() -> bool:
    """判断是否使用 jieba 分词（每次从 settings 读取，支持运行时切换）"""
    if not _HAS_JIEBA:
        return False
    try:
        from config import settings
        return settings.bm25_use_jieba
    except Exception:
        return _HAS_JIEBA


def _tokenize(text: str) -> List[str]:
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

    use_jieba = _should_use_jieba()

    for seg in segments:
        if re.match(r'[\u4e00-\u9fff]', seg):
            if use_jieba:
                # jieba 分词 + bigram 补充
                words = list(jieba.cut(seg))
                tokens.extend(w for w in words if w.strip())
                # 补充 bigram 提升短语匹配
                for i in range(len(words) - 1):
                    tokens.append(words[i] + words[i + 1])
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

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
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
        self.chunks = chunks
        self.doc_count = len(chunks)
        self.doc_lengths = []
        self.doc_freqs = {}
        self.term_freqs = []

        for chunk in chunks:
            tokens = _tokenize(chunk)
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
        query_tokens = _tokenize(query)
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

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        """
        BM25检索
        
        Returns:
            排序后的结果列表，每项包含 chunk, score, index
        """
        if not self.chunks:
            return []

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
# 全局BM25索引缓存（按doc_id）
# ============================================================
_bm25_cache: Dict[str, BM25Index] = {}


def get_or_build_bm25(doc_id: str, chunks: List[str]) -> BM25Index:
    """获取或构建BM25索引（带缓存）"""
    if doc_id in _bm25_cache:
        cached = _bm25_cache[doc_id]
        # 简单校验：chunk数量一致就复用
        if len(cached.chunks) == len(chunks):
            return cached

    idx = BM25Index()
    idx.build(chunks)
    _bm25_cache[doc_id] = idx
    return idx


def clear_bm25_cache(doc_id: Optional[str] = None):
    """清除BM25缓存"""
    if doc_id:
        _bm25_cache.pop(doc_id, None)
    else:
        _bm25_cache.clear()


def bm25_search(doc_id: str, query: str, chunks: List[str], top_k: int = 10) -> List[dict]:
    """
    便捷函数：BM25检索
    
    自动构建/复用索引，返回top_k结果
    """
    idx = get_or_build_bm25(doc_id, chunks)
    return idx.search(query, top_k)
