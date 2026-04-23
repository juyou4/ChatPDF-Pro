"""
同义词扩展服务 — BM25 查询时自动扩展同义词，提升召回率

设计参考：ragflow query.py 的同义词扩展策略
- 查询 token 自动扩展同义词，同义词权重衰减（默认 0.4）
- 内置常用中英文学术/技术同义词词典
- 支持用户自定义词典文件扩展
- 细粒度分词补充：长词自动拆分为子词（类似 ragflow fine_grained_tokenize）
"""
import logging
import os
import json
import re
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ============================================================
# 内置同义词词典（中英文常用学术/技术术语）
# 格式: 每组为互相同义的词列表
# ============================================================
_BUILTIN_SYNONYM_GROUPS: List[List[str]] = [
    # --- 学术论文常用 ---
    ["方法", "方式", "手段", "途径", "策略", "方案"],
    ["结果", "成果", "结论", "发现", "研究结果"],
    ["分析", "解析", "研究", "探讨", "考察"],
    ["提出", "提议", "建议", "引入", "给出"],
    ["实验", "试验", "实测", "测试"],
    ["性能", "表现", "效果", "效能"],
    ["优化", "改进", "改善", "提升", "增强"],
    ["模型", "模式", "架构", "框架"],
    ["算法", "算式", "计算方法"],
    ["数据", "数据集", "样本", "语料"],
    ["训练", "学习", "微调", "拟合"],
    ["评估", "评价", "评测", "度量", "衡量"],
    ["准确率", "精度", "精确度", "accuracy"],
    ["召回率", "查全率", "recall"],
    ["特征", "特性", "属性", "feature"],
    ["参数", "超参数", "配置", "parameter"],
    ["损失", "loss", "代价函数", "目标函数"],
    ["网络", "神经网络", "network"],
    ["层", "layer", "网络层"],
    ["输入", "input", "输入数据"],
    ["输出", "output", "输出结果"],
    ["问题", "难题", "挑战", "困难"],
    ["影响", "作用", "效应", "impact"],
    ["比较", "对比", "比较分析", "comparison"],
    ["图", "图片", "图像", "figure", "图表"],
    ["表", "表格", "table"],
    ["公式", "方程", "等式", "formula", "equation"],
    ["定义", "概念", "含义", "definition"],
    ["证明", "推导", "论证", "proof"],
    ["假设", "前提", "assumption", "hypothesis"],
    ["应用", "运用", "使用", "application"],
    ["缺点", "不足", "局限", "缺陷", "limitation"],
    ["优点", "优势", "长处", "advantage"],
    ["摘要", "概要", "总结", "summary", "abstract"],
    ["引言", "介绍", "背景", "introduction"],
    ["相关工作", "文献综述", "related work"],
    ["结论", "总结", "conclusion"],
    ["参考文献", "引用", "reference"],
    # --- 技术常用 ---
    ["接口", "api", "端点", "endpoint"],
    ["数据库", "database", "存储", "storage"],
    ["服务器", "server", "后端", "backend"],
    ["客户端", "client", "前端", "frontend"],
    ["配置", "设置", "config", "setting"],
    ["错误", "异常", "bug", "error", "故障"],
    ["日志", "log", "记录"],
    ["部署", "发布", "deploy", "release"],
    ["版本", "version", "迭代"],
    # --- 英文同义词 ---
    ["method", "approach", "technique", "strategy"],
    ["result", "outcome", "finding"],
    ["analysis", "study", "investigation", "examination"],
    ["propose", "introduce", "present", "suggest"],
    ["experiment", "evaluation", "assessment"],
    ["performance", "effectiveness", "efficiency"],
    ["improve", "enhance", "optimize", "boost"],
    ["model", "architecture", "framework", "system"],
    ["dataset", "data", "corpus", "benchmark"],
    ["training", "learning", "fine-tuning"],
    ["accuracy", "precision", "correctness"],
    ["feature", "attribute", "property", "characteristic"],
    ["problem", "issue", "challenge", "difficulty"],
    ["advantage", "benefit", "strength"],
    ["disadvantage", "limitation", "weakness", "drawback"],
    ["compare", "contrast", "comparison"],
]


class SynonymDict:
    """同义词词典

    支持双向查找：给定一个词，返回它的所有同义词。
    同义词权重衰减系数默认 0.4（同义词贡献的 BM25 分数为原词的 40%）。
    """

    def __init__(self, synonym_weight: float = 0.4):
        self.synonym_weight = synonym_weight
        # word -> set of synonyms (不含自身)
        self._dict: Dict[str, Set[str]] = {}
        self._loaded = False

    def _add_group(self, words: List[str]):
        """添加一组同义词"""
        normalized = [w.lower().strip() for w in words if w.strip()]
        unique = list(dict.fromkeys(normalized))  # 去重保序
        for w in unique:
            if w not in self._dict:
                self._dict[w] = set()
            self._dict[w].update(s for s in unique if s != w)

    def load(self, custom_dict_path: Optional[str] = None):
        """加载同义词词典（内置 + 可选自定义文件）"""
        if self._loaded:
            return
        # 1. 加载内置词典
        for group in _BUILTIN_SYNONYM_GROUPS:
            self._add_group(group)

        # 2. 加载自定义词典文件（JSON 格式: [["词1","词2"], ["词3","词4"]]）
        if custom_dict_path and os.path.exists(custom_dict_path):
            try:
                with open(custom_dict_path, "r", encoding="utf-8") as f:
                    custom_groups = json.load(f)
                if isinstance(custom_groups, list):
                    for group in custom_groups:
                        if isinstance(group, list):
                            self._add_group(group)
                    logger.info(f"已加载自定义同义词词典: {custom_dict_path}, {len(custom_groups)} 组")
            except Exception as e:
                logger.warning(f"加载自定义同义词词典失败: {custom_dict_path}, {e}")

        self._loaded = True
        logger.info(f"同义词词典已加载: {len(self._dict)} 个词条")

    def lookup(self, word: str) -> List[str]:
        """查找同义词（返回不含自身的同义词列表）"""
        if not self._loaded:
            self.load()
        return list(self._dict.get(word.lower().strip(), set()))

    def expand_tokens(self, tokens: List[str]) -> List[Tuple[str, float]]:
        """扩展 token 列表，返回 (token, weight) 对

        原始 token 权重为 1.0，同义词权重为 self.synonym_weight。
        去重：如果同义词已在原始 token 中，不重复添加。

        Args:
            tokens: 原始分词结果

        Returns:
            [(token, weight), ...] 列表，包含原始词和同义词
        """
        if not self._loaded:
            self.load()

        result: List[Tuple[str, float]] = []
        seen: Set[str] = set()

        # 原始 token: weight = 1.0
        for t in tokens:
            t_lower = t.lower()
            if t_lower not in seen:
                result.append((t_lower, 1.0))
                seen.add(t_lower)

        # 同义词扩展: weight = synonym_weight
        for t in tokens:
            t_lower = t.lower()
            syns = self._dict.get(t_lower, set())
            for s in syns:
                if s not in seen:
                    result.append((s, self.synonym_weight))
                    seen.add(s)

        return result


def _fine_grained_tokenize(word: str) -> List[str]:
    """细粒度分词：将长中文词拆分为更小的子词

    参考 ragflow 的 fine_grained_tokenize，对 3+ 字符的中文词
    额外生成 bigram 子词，提升部分匹配的召回率。

    Args:
        word: 中文词语

    Returns:
        子词列表（不含原词本身）
    """
    # 只处理纯中文且长度 >= 3 的词
    if len(word) < 3 or not re.match(r'^[\u4e00-\u9fff]+$', word):
        return []

    sub_tokens = []
    for i in range(len(word) - 1):
        sub_tokens.append(word[i:i + 2])
    return sub_tokens


# ============================================================
# 模块级单例
# ============================================================
_synonym_dict: Optional[SynonymDict] = None


def get_synonym_dict() -> SynonymDict:
    """获取全局同义词词典单例"""
    global _synonym_dict
    if _synonym_dict is None:
        _synonym_dict = SynonymDict()
        # 尝试加载自定义词典
        try:
            from config import settings
            custom_path = getattr(settings, "bm25_synonym_dict_path", None)
        except Exception:
            custom_path = None
        _synonym_dict.load(custom_path)
    return _synonym_dict
