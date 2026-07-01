"""
查询改写服务 - 使用本地规则将口语化查询转换为检索友好形式

不依赖 LLM 调用，使用正则表达式和模式匹配实现改写。
支持中英文混合查询。
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class QueryRewriter:
    """查询改写器 - 使用本地规则将口语化查询转换为检索友好形式"""

    # 口语化模式 → 规范化替换的映射表
    # 每个元素为 (正则模式, 替换字符串) 的元组
    # 注意：顺序很重要，更具体的模式应放在前面
    COLLOQUIAL_PATTERNS: list[tuple[re.Pattern, str]] = [
        # "啥意思" 相关模式
        (re.compile(r'(.+?)啥意思'), r'\1的含义和解释'),
        (re.compile(r'啥意思'), r'的含义和解释'),

        # "讲了啥" / "说了啥" 相关模式
        (re.compile(r'(.+?)讲了啥'), r'\1的主要内容'),
        (re.compile(r'(.+?)说了啥'), r'\1的主要内容'),
        (re.compile(r'讲了啥'), r'的主要内容'),
        (re.compile(r'说了啥'), r'的主要内容'),

        # "啥是 X" / "X 是啥" 定义类模式
        (re.compile(r'啥是\s*(.+)'), r'\1的定义'),
        (re.compile(r'(.+?)是啥'), r'\1的定义'),

        # "为啥" 原因类模式
        (re.compile(r'为啥要?\s*(.+)'), r'\1的原因和目的'),
        (re.compile(r'为啥'), r'的原因'),

        # "咋用" / "怎么用" 使用方法类模式
        (re.compile(r'咋用\s*(.+)'), r'\1的使用方法'),
        (re.compile(r'(.+?)咋用'), r'\1的使用方法'),
        (re.compile(r'怎么用\s*(.+)'), r'\1的使用方法'),
        (re.compile(r'(.+?)怎么用'), r'\1的使用方法'),

        # "咋 X" → "如何 X" 通用模式
        (re.compile(r'咋(.+)'), r'如何\1'),

        # "啥 X" → "什么 X" 通用模式
        (re.compile(r'啥(.+)'), r'什么\1'),
    ]

    # 指示代词列表
    PRONOUN_PATTERNS: list[str] = [
        '这个', '那个', '这块', '那块', '这部分', '那部分',
        '这段', '那段', '这里', '那里', '它',
    ]

    GENERIC_METRIC_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
        (re.compile(r'\baccuracy\b|分类准确率|准确率', re.IGNORECASE), 'Acc'),
        (re.compile(r'\bprecision\b|精确率', re.IGNORECASE), 'Precision'),
        (re.compile(r'\brecall\b|召回率', re.IGNORECASE), 'Recall'),
        (re.compile(r'\bf1(?:[-_ ]?score)?\b', re.IGNORECASE), 'F1'),
        (re.compile(r'\bauc\b|\bauroc\b', re.IGNORECASE), 'AUC'),
        (re.compile(r'\b(?:mAP|mean average precision)\b', re.IGNORECASE), 'mAP'),
        (re.compile(r'\b(?:BLEU|ROUGE|METEOR|WER|CER)\b', re.IGNORECASE), 'TextMetric'),
        (re.compile(r'\b(?:latency|runtime|throughput|flops|params?)\b|推理时间|耗时|吞吐|参数量|开销', re.IGNORECASE), 'CostMetric'),
        (re.compile(r'\b(?:overall|total|macro|micro|weighted)\b|总体|整体|宏平均|微平均', re.IGNORECASE), 'Aggregate'),
    )
    NUMERIC_COMPARISON_PATTERNS: tuple[tuple[re.Pattern, tuple[str, ...]], ...] = (
        (
            re.compile(r'第二好的方法|第二好|次优|次佳|second[- ]best|nearest competitor|runner[- ]up', re.IGNORECASE),
            ('second-best', 'nearest competitor'),
        ),
        (
            re.compile(r'提升|高多少|差多少|百分点|improv|gain|gap|delta', re.IGNORECASE),
            ('improvement', 'percentage-point', 'delta'),
        ),
    )
    _GENERIC_NUMERIC_STOPWORDS: set[str] = {
        'table', 'tables', 'overall', 'accuracy', 'results', 'result', 'dataset', 'datasets',
        'baseline', 'method', 'methods', 'metric', 'metrics',
        'flops', 'flop', 'runtime', 'latency', 'overhead', 'cost',
        'inference', 'training', 'time', 'appendix', 'limitation',
        'limitations',
    }

    def rewrite(
        self,
        query: str,
        selected_text: Optional[str] = None,
        evidence_need: Optional[list[str]] = None,
    ) -> str:
        """改写查询

        处理流程：
        1. 如果提供了 selected_text，先尝试解析指示代词
        2. 然后应用口语化模式替换
        3. 如果无需改写则返回原始查询

        Args:
            query: 原始用户查询
            selected_text: 用户选中的文本上下文（可选）

        Returns:
            改写后的查询文本，如果无需改写则返回原始查询
        """
        if not query or not query.strip():
            return query

        try:
            rewritten = query

            # 第一步：如果有选中文本，尝试解析指示代词
            if selected_text and selected_text.strip():
                rewritten = self._resolve_pronouns(rewritten, selected_text)

            # 第二步：应用口语化模式替换
            rewritten = self._replace_colloquial(rewritten)

            # 第三步（新增）：如果有 selected_text 且查询未被代词解析改变，
            # 追加 selected_text 关键内容以增强检索
            if selected_text and selected_text.strip() and rewritten == query:
                rewritten = self._augment_with_selected_text(rewritten, selected_text)

            # 数值表格题使用更明确的检索模板，避免退化成通用摘要检索。
            rewritten = self._apply_evidence_need_templates(
                rewritten,
                evidence_need=evidence_need,
            )

            return rewritten
        except Exception as e:
            # 改写失败时返回原始查询，记录警告日志
            logger.warning(f"查询改写失败，返回原始查询: {e}")
            return query

    def _replace_colloquial(self, query: str) -> str:
        """替换口语化表达

        遍历 COLLOQUIAL_PATTERNS，对查询应用正则替换。
        如果没有匹配到任何模式，返回原始查询。

        Args:
            query: 待处理的查询文本

        Returns:
            替换后的查询文本
        """
        result = query
        for pattern, replacement in self.COLLOQUIAL_PATTERNS:
            new_result = pattern.sub(replacement, result)
            if new_result != result:
                # 匹配到模式，使用替换结果
                result = new_result
                # 只应用第一个匹配的模式，避免多次替换导致语义混乱
                break
        return result

    def _resolve_pronouns(self, query: str, selected_text: str) -> str:
        """解析指示代词，用选中文本的关键内容替换

        检查查询中是否包含指示代词，如果包含则用选中文本的
        关键内容片段替换。

        Args:
            query: 包含指示代词的查询
            selected_text: 用户选中的文本上下文

        Returns:
            替换指示代词后的查询
        """
        if not selected_text or not selected_text.strip():
            return query

        # 提取选中文本的关键内容
        key_content = self._extract_key_content(selected_text)
        if not key_content:
            return query

        result = query
        for pronoun in self.PRONOUN_PATTERNS:
            if pronoun in result:
                # 用关键内容替换指示代词
                result = result.replace(pronoun, key_content, 1)

        return result

    def _augment_with_selected_text(
        self, query: str, selected_text: str, max_augment_chars: int = 80
    ) -> str:
        """将 selected_text 关键内容追加到查询中增强检索语义

        仅在查询未被指示代词解析改变时调用。
        提取 selected_text 的关键内容片段，追加到查询末尾。

        Args:
            query: 原始查询
            selected_text: 用户选中的文本
            max_augment_chars: 追加内容的最大字符数，默认 80

        Returns:
            追加关键内容后的查询，如果无法提取关键内容则返回原始查询
        """
        key_content = self._extract_key_content(selected_text, max_chars=max_augment_chars)
        if not key_content:
            return query
        return f"{query} {key_content}"

    def _apply_evidence_need_templates(
        self,
        query: str,
        evidence_need: Optional[list[str]] = None,
    ) -> str:
        from services.query_analyzer import analyze_evidence_need

        needs = set(evidence_need or analyze_evidence_need(query))
        if 'numeric_table' not in needs:
            return query
        return self._rewrite_numeric_table_query(query)

    def _rewrite_numeric_table_query(self, query: str) -> str:
        if not query:
            return query

        normalized = query.lower()
        guard_tokens = ('表格标题', '基线方法', 'comparison object', 'metric value', '列名别名', 'row anchor')
        if any(token.lower() in normalized for token in guard_tokens):
            return query

        hints = self.extract_numeric_table_hints(query)
        comparison_aliases = {value.lower() for value in hints.get("comparison", []) if value}
        if (
            {"second-best", "nearest competitor"} & comparison_aliases
            and len(hints.get("methods", [])) <= 1
        ) or (
            "second-best" in comparison_aliases
            and len(hints.get("columns", [])) >= 3
        ):
            return query

        if self._is_numeric_table_cost_query(query):
            hint_text = self.build_numeric_table_cost_hint_text(query)
            return f"{query} {hint_text}".strip()

        hint_text = self.build_numeric_table_hint_text(query)
        return f"{query} {hint_text}".strip()

    def build_numeric_table_hint_text(self, query: str) -> str:
        hints = self.extract_numeric_table_hints(query)
        parts = [
            (
                "表格标题 数据集 基线方法 比较对象 指标 数值 结果行列 列名别名 行名锚点 "
                "table caption dataset baseline metric value comparison row anchor column alias"
            )
        ]
        if hints["table_labels"]:
            parts.append(f"表号 {' '.join(hints['table_labels'])}")
        if hints["datasets"]:
            parts.append(f"数据集 {' '.join(hints['datasets'])}")
        if hints["backbones"]:
            parts.append(f"模型结构 {' '.join(hints['backbones'])}")
        if hints["methods"]:
            parts.append(f"方法名 {' '.join(hints['methods'])}")
        if hints["columns"]:
            parts.append(f"列名别名 {' '.join(hints['columns'])}")
        if hints["comparison"]:
            parts.append(f"比较关系 {' '.join(hints['comparison'])}")
        return " ".join(parts).strip()

    def build_numeric_table_cost_hint_text(self, query: str) -> str:
        hints = self.extract_numeric_table_hints(query)
        parts = [
            (
                "推理开销 额外FLOPs 推理时间 训练时间 成本 限制 讨论 附录 "
                "inference overhead extra flops inference time training time cost "
                "limitation discussion appendix no extra overhead"
            )
        ]
        if hints["methods"]:
            parts.append(f"方法名 {' '.join(hints['methods'])}")
        if hints["datasets"]:
            parts.append(f"数据集 {' '.join(hints['datasets'])}")
        return " ".join(parts).strip()

    def extract_numeric_table_hints(self, query: str) -> dict[str, list[str]]:
        if not query:
            return {
                "table_labels": [],
                "datasets": [],
                "backbones": [],
                "methods": [],
                "columns": [],
                "comparison": [],
            }

        table_labels = self._dedupe_preserve_order(self._extract_table_labels(query))
        columns = self._extract_column_aliases(query)
        datasets, backbones, methods = self._extract_named_anchors(query, column_aliases=columns)
        comparison = self._extract_comparison_aliases(query)
        return {
            "table_labels": table_labels,
            "datasets": datasets,
            "backbones": backbones,
            "methods": methods,
            "columns": columns,
            "comparison": comparison,
        }

    def _extract_table_labels(self, query: str) -> list[str]:
        labels: list[str] = []
        for match in re.finditer(r'(?:table|表)\s*\.?\s*(\d+)', query, re.IGNORECASE):
            number = match.group(1)
            labels.extend([f"Table {number}", f"表 {number}"])
        return labels

    def _extract_column_aliases(self, query: str) -> list[str]:
        aliases: list[str] = []
        sample = query or ""
        for match in re.finditer(r'[`"“”\'‘’]([^`"“”\'‘’]{1,48})[`"“”\'‘’]', sample):
            token = re.sub(r"\s+", " ", match.group(1)).strip(" ,.;:[]{}")
            if token and not self._is_numeric_column_token(token):
                aliases.append(token)

        for match in re.finditer(r'([A-Za-z][A-Za-z0-9_.+\-/]{0,48})\s*(?:指标|列|列名|metric|column)', sample, re.IGNORECASE):
            aliases.extend(self._split_column_phrase(match.group(1)))
        for match in re.finditer(r'(?:对应的?|对应|分别为|分别是)\s*([A-Za-zΔ∆△\u4e00-\u9fff][A-Za-z0-9_.+\-/|∥\s\u4e00-\u9fff]{0,80}?)(?:分别|是多少|为多少|？|\?|,|，|。|$)', sample, re.IGNORECASE):
            aliases.extend(self._split_column_phrase(match.group(1)))
        for match in re.finditer(r'的\s*([A-Za-z][A-Za-z0-9_.+\-/、，,\s]{1,80}?)\s*分别', sample, re.IGNORECASE):
            aliases.extend(self._split_column_phrase(match.group(1)))
        for match in re.finditer(r'在\s*([A-Za-z][A-Za-z0-9_.+\-/\s]{1,80}?)\s*(?:子集|groups?|subsets?)', sample, re.IGNORECASE):
            aliases.extend(self._split_column_phrase(match.group(1)))

        formula_metric_pattern = re.compile(
            r'(?:[Δ∆△A-Za-z][A-Za-z0-9]*)(?:\s*/\s*(?:\|\||∥)?\s*[A-Za-z][A-Za-z0-9_ ]*(?:\|\||∥)?)+',
            re.IGNORECASE,
        )
        for match in formula_metric_pattern.finditer(sample):
            token = re.sub(r"\s+", " ", match.group(0)).strip(" ,.;:[]{}")
            if token and not self._is_numeric_column_token(token):
                aliases.append(token)

        compact_metric_pattern = re.compile(
            r'\b(?:[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*|[A-Za-z]+[-_][A-Za-z0-9]+)(?:[-_/][A-Za-z0-9]+)*\b'
        )
        for match in compact_metric_pattern.finditer(sample):
            token = match.group(0).strip(" ,.;:[]{}")
            lowered = token.lower()
            if lowered.startswith(("table", "figure", "fig")):
                continue
            if self._looks_like_dataset_anchor(token) or self._looks_like_model_family_anchor(token):
                continue
            if re.fullmatch(r"[A-Za-z]*[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*", token):
                continue
            if "/" in token and not re.search(r"(?:\|\||∥)", token):
                aliases.extend(self._split_column_phrase(token))
            elif not self._is_numeric_column_token(token):
                aliases.append(token)

        for pattern, label in self.GENERIC_METRIC_PATTERNS:
            if pattern.search(sample):
                aliases.append(label)
        return self._remove_composite_column_parts(self._dedupe_preserve_order(aliases))

    def _extract_comparison_aliases(self, query: str) -> list[str]:
        aliases: list[str] = []
        for pattern, values in self.NUMERIC_COMPARISON_PATTERNS:
            if pattern.search(query):
                aliases.extend(values)
        return self._dedupe_preserve_order(aliases)

    def _is_numeric_table_cost_query(self, query: str) -> bool:
        normalized = (query or "").lower()
        if not normalized:
            return False
        return any(
            token in normalized
            for token in (
                'flops',
                '推理时间',
                '开销',
                'latency',
                'runtime',
                'overhead',
                'cost',
                'inference time',
                'inference overhead',
                'training time',
                '训练时间',
                '耗时',
                'extra flops',
            )
        )

    def _is_numeric_column_token(self, token: str) -> bool:
        normalized = (token or "").lower().strip()
        if not normalized:
            return True

        compact = normalized.replace(" ", "")
        if compact in self._GENERIC_NUMERIC_STOPWORDS:
            return True

        if re.fullmatch(r"(?:table|fig(?:ure)?|section|appendix)\s*\d+[a-z]?", normalized):
            return True
        return False

    def _split_column_phrase(self, phrase: str) -> list[str]:
        text = re.sub(r"\s+", " ", str(phrase or "")).strip(" ,.;:[]{}，。；：、")
        if not text:
            return []
        expanded: list[str] = []
        expanded.extend(self._extract_generic_column_aliases(text))
        pieces = [
            part.strip(" ,.;:[]{}，。；：、")
            for part in re.split(r"[、，,]|(?:\s+and\s+)|和", text, flags=re.IGNORECASE)
        ]
        for piece in pieces:
            piece = self._normalize_column_alias(piece)
            if not piece:
                continue
            if not self._looks_like_column_alias(piece):
                continue
            if "/" in piece and not re.search(r"(?:\|\||∥)", piece):
                expanded.extend(
                    normalized
                    for part in piece.split("/")
                    if (normalized := self._normalize_column_alias(part))
                )
            else:
                if not expanded or piece not in expanded:
                    expanded.append(piece)
        column_items = [item for item in expanded if item and not self._is_numeric_column_token(item)]
        return self._remove_composite_column_parts(column_items)

    def _looks_like_column_alias(self, value: str) -> bool:
        sample = re.sub(r"\s+", " ", str(value or "")).strip(" ,.;:[]{}，。；：、")
        if not sample:
            return False
        if sample in {"Acc", "FID", "All", "Many", "Med.", "Few", "||D_gen||", "ΔAcc/||D_gen||"}:
            return True
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.+\-/]*", sample):
            return True
        if re.fullmatch(r"[Δ∆△][A-Za-z0-9_.+\-/]*", sample):
            return True
        if re.fullmatch(r"[A-Za-z0-9_]+(?:[-_][A-Za-z0-9_]+)+", sample):
            return True
        if re.fullmatch(r"[\u4e00-\u9fff]{2,8}", sample):
            return True
        return False

    def _extract_generic_column_aliases(self, text: str) -> list[str]:
        aliases: list[str] = []
        patterns = (
            r"\b(?:all|overall|total)\b|总体|整体",
            r"\b(?:many)\b",
            r"\b(?:medium|med\.?)\b|中等|中尾",
            r"\b(?:few(?:-shot)?|tail)\b|少样本|尾类",
            r"\b(?:accuracy|acc(?:\.?)?)\b|分类准确率|准确率",
            r"\bfid\b",
            r"(?:[Δ∆△]\s*acc(?:/\s*(?:\|\||∥)\s*d\s*[_ ]?gen\s*(?:\|\||∥))?)",
            r"(?:\|\||∥)\s*d\s*[_ ]?gen\s*(?:\|\||∥)",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                alias = self._normalize_column_name(match.group(0))
                if alias and not self._is_numeric_column_token(alias):
                    aliases.append(alias)
        return self._dedupe_preserve_order(aliases)

    def _normalize_column_name(self, value: str) -> str:
        sample = re.sub(r"\s+", "", str(value or "").strip())
        if not sample:
            return ""
        return self._normalize_column_alias(sample)

    def _normalize_column_alias(self, value: str) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip(" ,.;:[]{}，。；：、")
        text = re.sub(
            r"(?:\s*(?:分别)?(?:是|为)?多少(?:个百分点|分|点)?|\s*(?:子集|groups?|subsets?).*|\s*(?:提升|增加|高).*)$",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip(" ,.;:[]{}，。；：、")
        text = re.sub(r"(?:分别)?(?:是|为)?多少$", "", text, flags=re.IGNORECASE).strip(" ,.;:[]{}，。；：、")
        text = re.sub(r"(?:是多少|为多少|是什么|分别)$", "", text, flags=re.IGNORECASE).strip(" ,.;:[]{}，。；：、")
        if not text:
            return ""
        normalized = re.sub(r"\s+", "", text).lower()
        if normalized in {"准确率", "分类准确率", "accuracy", "acc", "acc."}:
            return "Acc"
        return text

    def _remove_composite_column_parts(self, column_items: list[str]) -> list[str]:
        expanded: list[str] = []
        for item in column_items:
            if "/" in item and not re.search(r"(?:\|\||∥)", item):
                expanded.extend(
                    normalized
                    for part in re.split(r"[/|,&+]+", item)
                    if (normalized := self._normalize_column_alias(part))
                )
                continue
            expanded.append(item)
        return self._dedupe_preserve_order(expanded)

    def _extract_named_anchors(
        self,
        query: str,
        column_aliases: Optional[list[str]] = None,
    ) -> tuple[list[str], list[str], list[str]]:
        datasets: list[str] = []
        backbones: list[str] = []
        methods: list[str] = []
        column_keys = {
            re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").lower())
            for value in (column_aliases or [])
            if str(value or "").strip()
        }
        for value in column_aliases or []:
            for part in re.split(r"[/|,&+]+", str(value or "")):
                part_key = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", part.lower())
                if part_key:
                    column_keys.add(part_key)

        token_pattern = re.compile(
            r'\b(?:[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+|[A-Za-z]*[A-Z][A-Za-z0-9.+/_-]*)(?:\s*\(\d+\s*experts?\))?'
        )
        for match in token_pattern.finditer(query):
            token = re.sub(r"\s+", " ", match.group(0)).strip(" ,.;:[]{}")
            if not token:
                continue
            lowered = token.lower()
            compact = lowered.replace(" ", "")
            if "/" in token:
                continue
            if compact in self._GENERIC_NUMERIC_STOPWORDS:
                continue
            token_key = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", lowered)
            if token_key and token_key in column_keys:
                continue
            if self._is_numeric_column_token(token):
                continue
            if lowered.startswith("table"):
                continue
            if self._looks_like_dataset_anchor(token):
                datasets.append(token)
            elif self._looks_like_model_family_anchor(token):
                backbones.append(token)
            elif self._looks_like_method_anchor(token):
                methods.append(token)

        return (
            self._dedupe_preserve_order(datasets),
            self._dedupe_preserve_order(backbones),
            self._dedupe_preserve_order(methods),
        )

    def _looks_like_dataset_anchor(self, token: str) -> bool:
        sample = (token or "").strip()
        if not sample:
            return False
        return bool(
            re.search(r"(?:^|[-_])(?:LT|Dataset|Data)$", sample)
            or re.search(r"(?:19|20)\d{2}$", sample)
            or re.search(r"(?:^|[-_])(?:INat|Nat|Bench|Corpus|Set)(?:[-_]|$)", sample, re.IGNORECASE)
        )

    def _looks_like_method_anchor(self, token: str) -> bool:
        sample = (token or "").strip()
        if not sample:
            return False
        if self._looks_like_dataset_anchor(sample):
            return False
        if self._looks_like_model_family_anchor(sample):
            return False
        return bool(
            re.search(r"[A-Z]", sample)
            or re.search(r"[A-Za-z]+(?:[-_+/][A-Za-z0-9]+)+", sample)
            or re.search(r"\(\s*\d+\s*[^)]*\)", sample)
        )

    def _looks_like_model_family_anchor(self, token: str) -> bool:
        sample = (token or "").strip()
        if not sample:
            return False
        return bool(
            re.fullmatch(r"[A-Za-z][A-Za-z0-9]*[-_ ]?\d+[A-Za-z0-9]*", sample)
            or re.fullmatch(r"[A-Za-z]+(?:Net|Former|Encoder|Decoder|Backbone)[-_ ]?\d+[A-Za-z0-9]*", sample)
        )

    def _dedupe_preserve_order(self, items: list[str]) -> list[str]:
        seen = set()
        ordered: list[str] = []
        for item in items:
            normalized = item.strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(normalized)
        return ordered

    async def rewrite_with_llm(
        self,
        query: str,
        chat_history: list[dict] = None,
        selected_text: Optional[str] = None,
        api_key: str = "",
        model: str = "",
        provider: str = "",
        endpoint: str = "",
        evidence_need: Optional[list[str]] = None,
    ) -> str:
        """用 LLM 改写查询，支持多轮对话指代消解

        先执行 regex 规则改写（零成本），再用 LLM 改写。
        适用于多轮对话中"那它呢？""具体怎么做？"等需要上下文的查询。

        Args:
            query: 原始查询
            chat_history: 对话历史 [{"role": "user"|"assistant", "content": "..."}]
            selected_text: 用户选中的文本
            api_key: LLM API 密钥
            model: LLM 模型名称
            provider: LLM 提供商
            endpoint: LLM API 端点

        Returns:
            改写后的查询，失败时返回 regex 改写结果
        """
        # 第一步：先跑 regex 改写
        rewritten = self.rewrite(
            query,
            selected_text=selected_text,
            evidence_need=evidence_need,
        )

        if not api_key:
            return rewritten

        # 第二步：构建 LLM 改写 prompt
        history_text = ""
        if chat_history:
            recent = chat_history[-6:]  # 最近 3 轮对话
            history_lines = []
            for msg in recent:
                role = "用户" if msg.get("role") == "user" else "助手"
                content = msg.get("content", "")[:200]
                history_lines.append(f"{role}: {content}")
            history_text = "\n".join(history_lines)

        selected_hint = ""
        if selected_text and selected_text.strip():
            selected_hint = f"\n用户选中的文本：{selected_text[:200]}"

        prompt = f"""请将以下用户查询改写为一个独立的、适合检索文档的查询语句。
要求：
- 消解代词和指代（如"它"、"这个方法"等），使查询独立可理解
- 保持原始意图不变
- 直接输出改写后的查询，不要加任何前缀或解释
- 如果查询已经足够清晰，直接返回原文

{f'对话历史：{chr(10)}{history_text}' if history_text else ''}
{selected_hint}

当前查询：{rewritten}

改写后的查询："""

        try:
            from services.chat_service import call_ai_api
            from models.provider_registry import PROVIDER_CONFIG

            if not model:
                model = "gpt-4o-mini"
            if not provider:
                provider = "openai"
            if not endpoint:
                endpoint = PROVIDER_CONFIG.get(provider, {}).get("endpoint", "")

            response = await call_ai_api(
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
                model=model,
                provider=provider,
                endpoint=endpoint,
                max_tokens=100,
                temperature=0.3,
            )

            if isinstance(response, dict):
                if response.get("error"):
                    logger.warning(f"[LLM QueryRewrite] 调用失败: {response['error']}")
                    return rewritten
                content = response.get("content", "")
                if not content and "choices" in response:
                    choices = response["choices"]
                    if choices and isinstance(choices, list):
                        content = choices[0].get("message", {}).get("content", "")
            else:
                content = str(response) if response else ""

            content = content.strip()
            if content and len(content) > 3 and content != query:
                logger.info(f"[LLM QueryRewrite] '{query}' → '{content}'")
                return content

            return rewritten

        except Exception as e:
            logger.warning(f"[LLM QueryRewrite] 失败，降级为 regex 改写: {e}")
            return rewritten

    def _extract_key_content(self, text: str, max_chars: int = 50) -> str:
        """从选中文本中提取关键内容片段

        提取策略：
        1. 去除首尾空白
        2. 取第一个有意义的句子或短语
        3. 如果超过 max_chars 则截断并添加省略号

        Args:
            text: 选中的文本
            max_chars: 最大字符数，默认 50

        Returns:
            提取的关键内容片段
        """
        if not text or not text.strip():
            return ''

        # 去除首尾空白和多余换行
        cleaned = text.strip()

        # 按句子分隔符切分，取第一个有意义的句子
        # 支持中英文标点
        sentence_delimiters = re.compile(r'[。！？\n；;!?]')
        sentences = sentence_delimiters.split(cleaned)

        # 找到第一个非空句子
        first_sentence = ''
        for s in sentences:
            s = s.strip()
            if s:
                first_sentence = s
                break

        if not first_sentence:
            first_sentence = cleaned

        # 截断到 max_chars
        if len(first_sentence) > max_chars:
            return first_sentence[:max_chars]

        return first_sentence
