"""Few-shot 查询改写服务

参考 kotaemon 的 FewshotRewriteQuestionPipeline：
1. 维护 (input, output) 改写示例的向量索引
2. 改写时检索 top-k 最相似示例注入 LLM prompt
3. LLM 参考示例进行更精准的查询改写

示例来源：
- 内置种子示例（口语→规范映射）
- 用户反馈积累（future）
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 内置种子示例：口语化/模糊查询 → 规范化检索查询
SEED_EXAMPLES = [
    {"input": "这个东西怎么用", "output": "使用方法和操作步骤"},
    {"input": "它和那个有啥区别", "output": "两者的区别和差异对比"},
    {"input": "上面说的那个方法好不好", "output": "该方法的优缺点评价"},
    {"input": "能不能详细说说", "output": "详细说明和解释"},
    {"input": "这篇论文讲了啥", "output": "论文的主要内容和核心贡献"},
    {"input": "实验结果咋样", "output": "实验结果和性能指标"},
    {"input": "有没有什么限制", "output": "局限性和约束条件"},
    {"input": "后面还能怎么做", "output": "未来研究方向和改进建议"},
    {"input": "跟别人的比怎么样", "output": "与其他方法的对比分析"},
    {"input": "为什么要这么做", "output": "动机和原因分析"},
    {"input": "数据集用的啥", "output": "使用的数据集和实验设置"},
    {"input": "效果好吗", "output": "性能表现和效果评估"},
]

FEWSHOT_REWRITE_PROMPT = """你是查询改写专家。请将用户的口语化/模糊查询改写为更精确、更适合文档检索的规范化查询。

以下是一些改写示例供参考：
{examples}

请改写以下查询，直接输出改写后的查询，不要加任何前缀或解释。
如果查询已经足够清晰规范，直接返回原查询。

用户查询：{query}
"""


class FewshotRewriteService:
    """Few-shot 查询改写服务"""

    def __init__(self, data_dir: Optional[str] = None):
        self._examples: list[dict] = list(SEED_EXAMPLES)
        self._data_dir = Path(data_dir) if data_dir else None
        self._custom_examples_file: Optional[Path] = None

        if self._data_dir:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            self._custom_examples_file = self._data_dir / "fewshot_examples.json"
            self._load_custom_examples()

    def _load_custom_examples(self):
        """加载用户自定义示例"""
        if self._custom_examples_file and self._custom_examples_file.exists():
            try:
                with open(self._custom_examples_file, "r", encoding="utf-8") as f:
                    custom = json.load(f)
                if isinstance(custom, list):
                    self._examples.extend(custom)
                    logger.info(f"[FewshotRewrite] 加载 {len(custom)} 条自定义示例")
            except Exception as e:
                logger.warning(f"[FewshotRewrite] 加载自定义示例失败: {e}")

    def add_example(self, input_query: str, output_query: str):
        """添加新的改写示例"""
        self._examples.append({"input": input_query, "output": output_query})
        if self._custom_examples_file:
            try:
                existing = []
                if self._custom_examples_file.exists():
                    with open(self._custom_examples_file, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                existing.append({"input": input_query, "output": output_query})
                with open(self._custom_examples_file, "w", encoding="utf-8") as f:
                    json.dump(existing, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"[FewshotRewrite] 保存示例失败: {e}")

    def _find_similar_examples(self, query: str, top_k: int = 3) -> list[dict]:
        """简单字符串相似度匹配最相关的示例

        使用关键词重叠度作为相似度指标（轻量级，无需向量索引）。
        """
        query_chars = set(query)
        scored = []
        for ex in self._examples:
            input_chars = set(ex["input"])
            overlap = len(query_chars & input_chars)
            total = len(query_chars | input_chars) or 1
            score = overlap / total
            scored.append((score, ex))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:top_k]]

    async def rewrite(
        self,
        query: str,
        api_key: str,
        model: str,
        provider: str,
        endpoint: str = "",
    ) -> str:
        """使用 few-shot 示例改写查询

        Args:
            query: 用户原始查询
            api_key: LLM API 密钥
            model: LLM 模型
            provider: LLM 提供商
            endpoint: API 端点

        Returns:
            改写后的查询，失败返回原查询
        """
        if not api_key or len(query) < 4:
            return query

        try:
            from services.chat_service import call_ai_api

            similar = self._find_similar_examples(query, top_k=3)
            examples_text = "\n".join(
                f"- 输入: {ex['input']} → 输出: {ex['output']}"
                for ex in similar
            )

            prompt = FEWSHOT_REWRITE_PROMPT.format(
                examples=examples_text,
                query=query,
            )

            response = await call_ai_api(
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
                model=model,
                provider=provider,
                endpoint=endpoint,
                max_tokens=100,
                temperature=0.3,
            )

            content = ""
            if isinstance(response, dict):
                if response.get("error"):
                    return query
                choices = response.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")

            rewritten = content.strip().strip('"\'')
            if rewritten and len(rewritten) < len(query) * 3:
                logger.info(f"[FewshotRewrite] '{query}' → '{rewritten}'")
                return rewritten

            return query

        except Exception as e:
            logger.warning(f"[FewshotRewrite] 改写失败: {e}")
            return query
