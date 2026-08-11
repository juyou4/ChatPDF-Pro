"""Provider-neutral reasoning effort resolution.

The UI exposes a small, stable vocabulary while providers expose different
controls.  This module is the backend source of truth for translating the
stable vocabulary into provider request fields and for downgrading unsupported
values deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from models.dynamic_store import load_dynamic_models, load_dynamic_providers
from models.provider_registry import PROVIDER_CONFIG


REASONING_EFFORTS = ("off", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")
_ACTIVE_EFFORTS = REASONING_EFFORTS[1:]
_EFFORT_RANK = {value: index for index, value in enumerate(REASONING_EFFORTS)}
REASONING_EFFORT_LABELS = {
    # 档位展示与 Provider 协议名保持一致，避免菜单文案和实际请求脱节。
    "off": "off",
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
    "ultra": "ultra",
}
_ALIASES = {
    "": "off",
    "disabled": "off",
    "none": "off",
    "on": "medium",
    "enabled": "medium",
    "extra_high": "xhigh",
    "extra-high": "xhigh",
    "extra high": "xhigh",
    "very_high": "xhigh",
    "very-high": "xhigh",
    "maximum": "max",
}

# Shared budget ladder for APIs that expose thinking_budget/budget_tokens.
# Values are intentionally conservative so a normal answer still has room.
THINKING_BUDGET_TOKENS = {
    "minimal": 1_024,
    "low": 2_048,
    "medium": 4_096,
    "high": 8_192,
    "xhigh": 16_384,
    "max": 32_768,
    "ultra": 65_536,
}


@dataclass(frozen=True)
class ReasoningProfile:
    """Capabilities of one provider/model pair."""

    mode: str
    options: tuple[str, ...]
    default: str = "medium"
    always_enabled: bool = False
    note: str = ""
    source: str = "protocol_default"
    cost_warning_from: str | None = None
    # ``off`` cannot be represented by simply omitting parameters for many
    # reasoning-first models. This names the provider-native disable knob.
    off_control: str | None = None
    # Toggle-style APIs do not share one universal enable field.  In
    # particular, MiniMax uses ``reasoning_split`` for the active path while
    # MiMo/GLM expose ``thinking.type``. Keep that distinction in the profile
    # instead of guessing from the generic ``thinking_toggle`` mode.
    on_control: str | None = None
    # Canonical UI values are intentionally independent from provider enums.
    native_levels: tuple[tuple[str, str], ...] = ()
    # Some APIs separate reasoning output with an independent response-format
    # switch. It must never be confused with the switch that enables thinking.
    split_reasoning_output: bool = False


@dataclass(frozen=True)
class ReasoningResolution:
    """The effective choice used for one request."""

    requested: str
    effective: str | None
    enabled: bool
    mode: str
    native_effort: str | None
    budget_tokens: int | None
    profile: ReasoningProfile
    native_control: str | None = None

    def public(self) -> dict[str, Any]:
        fallback = self.requested != (self.effective or "off")
        fallback_reason = None
        if fallback:
            if self.profile.mode == "unsupported":
                fallback_reason = "当前模型不支持可调思考档位"
            elif self.profile.always_enabled and self.requested == "off":
                fallback_reason = "当前模型始终启用思考"
            else:
                fallback_reason = "当前模型没有请求的档位"
        return {
            "requested": self.requested,
            "effective": self.effective,
            "enabled": self.enabled,
            "mode": self.mode,
            "native_effort": self.native_effort,
            "native_control": self.native_control,
            "budget_tokens": self.budget_tokens,
            "available": list(self.profile.options),
            "fallback": fallback,
            "fallback_reason": fallback_reason,
            "note": self.profile.note,
            "source": self.profile.source,
            "cost_warning_from": self.profile.cost_warning_from,
            "split_reasoning_output": self.profile.split_reasoning_output,
            "requested_label": REASONING_EFFORT_LABELS.get(self.requested, self.requested),
            "effective_label": REASONING_EFFORT_LABELS.get(self.effective or "off", self.effective or "off"),
        }


def ensure_reasoning_output_budget(
    max_tokens: int | None,
    resolution: ReasoningResolution,
    *,
    margin: int = 1_024,
) -> int | None:
    """确保 provider 的输出上限不会小于思考预算。

    Anthropic 的 ``budget_tokens`` 明确要求 ``max_tokens`` 大于思考预算；
    Gemini 的预算模式也容易在过小的输出上限下只返回思考内容。只在需要
    时提升上限，不覆盖用户已经设置的更大值。
    """
    if not resolution.enabled:
        return max_tokens
    budget = int(resolution.budget_tokens or 0)
    if budget <= 0:
        return max_tokens
    try:
        configured = int(max_tokens or 0)
    except (TypeError, ValueError):
        configured = 0
    return max(configured, budget + max(0, int(margin)))


def normalize_reasoning_effort(value: Any, *, default: str = "off") -> str:
    """Normalize legacy aliases without accepting arbitrary provider values."""

    raw = str(value or "").strip().lower()
    normalized = _ALIASES.get(raw, raw)
    if normalized not in REASONING_EFFORTS:
        return default if default in REASONING_EFFORTS else "off"
    return normalized


def is_valid_reasoning_effort(value: Any) -> bool:
    """Return whether a client value is a known canonical value or alias."""

    raw = str(value or "").strip().lower()
    return raw in REASONING_EFFORTS or raw in _ALIASES


def requires_preserved_reasoning_history(provider: str, model: str) -> bool:
    """判断当前模型是否要求在后续请求中回传 reasoning_content。"""

    del provider  # 兼容代理网关时以模型身份为准，不能只依赖内置 Provider ID。
    value = str(model or "").strip().lower()
    return bool(
        re.search(
            r"deepseek-v4|glm-5\.2|kimi-k3|qwen3\.8-max|mimo-v2\.5",
            value,
        )
    )


def prepare_reasoning_history_messages(
    messages: list[dict] | None,
    provider: str,
    model: str,
) -> list[dict]:
    """只向明确要求保留思考的模型发送历史 reasoning_content。

    普通 OpenAI 兼容接口可能拒绝 assistant 消息中的未知字段，因此不能把
    前端保存的思考内容无条件透传。工具调用相关字段保持原样。
    """

    preserve = requires_preserved_reasoning_history(provider, model)
    prepared: list[dict] = []
    for raw in messages or []:
        if not isinstance(raw, Mapping):
            continue
        message = dict(raw)
        if str(message.get("role") or "").strip().lower() != "assistant" or not preserve:
            message.pop("reasoning_content", None)
            message.pop("reasoning_details", None)
            message.pop("thinking", None)
        elif not isinstance(message.get("reasoning_content"), str) or not message.get("reasoning_content", "").strip():
            message.pop("reasoning_content", None)
        prepared.append(message)
    return prepared


def _dynamic_provider_config(provider: str) -> dict[str, Any]:
    try:
        value = load_dynamic_providers().get(str(provider or "").strip())
    except Exception:
        value = None
    return value if isinstance(value, dict) else {}


def _dynamic_model_config(provider: str, model: str) -> dict[str, Any]:
    """读取用户显式保存的模型能力声明。

    动态模型记录目前以 model id 为 key。旧版本记录可能没有 provider_id，
    这种记录只有在 provider 能明确匹配时才作为声明使用，避免把同名模型
    的能力错误套到另一个网关。
    """
    try:
        models = load_dynamic_models()
    except Exception:
        return {}
    exact = models.get(str(model or "").strip())
    if not isinstance(exact, dict):
        wanted = str(model or "").strip().lower()
        exact = next(
            (
                value
                for key, value in models.items()
                if str(key).strip().lower() == wanted and isinstance(value, dict)
            ),
            None,
        )
    if not isinstance(exact, dict):
        return {}
    bound_provider = str(exact.get("provider_id") or exact.get("providerId") or "").strip()
    if bound_provider and bound_provider.lower() != str(provider or "").strip().lower():
        return {}
    return exact


def _normalize_declared_options(value: Any) -> tuple[str, ...]:
    """Normalize a provider/model supplied option list into canonical values."""
    if isinstance(value, str):
        raw_values = re.split(r"[,|/\s]+", value)
    elif isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        return ()
    normalized = []
    for item in raw_values:
        raw = str(item or "").strip().lower()
        if not raw:
            continue
        value = normalize_reasoning_effort(raw)
        if value in REASONING_EFFORTS and value not in normalized:
            normalized.append(value)
    return tuple(sorted(normalized, key=lambda item: _EFFORT_RANK[item]))


def _declared_value(config: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in config:
            return config.get(key)
    return None


def _provider_protocol(provider: str, provider_type: str | None = None) -> str:
    pid = str(provider or "").strip().lower()
    dynamic = _dynamic_provider_config(pid)
    configured = str(dynamic.get("type") or PROVIDER_CONFIG.get(pid, {}).get("type") or "").strip().lower()
    return str(provider_type or configured or pid).strip().lower()


def _looks_reasoning_model(model: str) -> bool:
    value = str(model or "").strip().lower()
    return bool(
        re.search(
            r"(?:reason|reasoner|thinking|think|deepseek-(?:r|v4)|\bqwen3(?:[.\-:]|$)|\bqwq\b|\bglm-[45]|\bo[134](?:[-.]|$)|gpt-5(?:[.\-]|$)|grok-[34]|seed|(?:^|/)minimax-m[23](?:[.\-:]|$)|(?:^|/)mimo-v2\.5(?:-pro)?(?:[.\-:]|$))",
            value,
        )
    )


def _openai_options(model: str) -> tuple[str, ...]:
    value = str(model or "").strip().lower()
    if re.search(r"gpt-5\.6(?:[-.]|$)", value):
        # Ultra 不是 OpenAI-compatible 协议的通用能力；只有模型元数据
        # 明确声明后才允许展示，避免把 Cursor 的会员档位误发给上游。
        return ("off", "low", "medium", "high", "xhigh", "max")
    if re.search(r"gpt-5\.5(?:[-.]|$)", value):
        return ("off", "minimal", "low", "medium", "high", "xhigh", "max")
    if re.search(r"gpt-5\.4(?:[-.]|$)", value):
        return ("off", "minimal", "low", "medium", "high", "xhigh")
    if re.search(r"gpt-5\.[1-3](?:[-.]|$)", value):
        return ("off", "minimal", "low", "medium", "high")
    if re.search(r"\bo[134](?:[-.]|$)", value):
        return ("low", "medium", "high")
    return ("off", "low", "medium", "high")


def _openai_supports_native_off(model: str) -> bool:
    """Match the OpenAI families that accept ``reasoning_effort=none``."""

    value = str(model or "").strip().lower()
    if value.startswith("gpt-6") or "gpt-5.1" in value:
        return True
    match = re.match(r"gpt-5\.(\d+)", value)
    return bool(match and int(match.group(1)) >= 1)


def _anthropic_uses_adaptive_thinking(model: str) -> bool:
    """Return whether the Claude family uses adaptive effort, not token budgets."""

    value = str(model or "").strip().lower()
    family = r"(?:opus|sonnet|haiku|fable|mythos)"
    return bool(
        re.search(rf"claude-(?:{family}-)?5(?:[.\-]|$)", value)
        or re.search(rf"claude-(?:{family}-)?4[-.]?[6-9](?:[.\-]|$)", value)
    )


def _anthropic_adaptive_options(model: str) -> tuple[str, ...]:
    """Return the documented effort set for one adaptive Claude family."""

    value = str(model or "").strip().lower()
    supports_xhigh = bool(
        re.search(r"claude-(?:(?:fable|mythos|opus|sonnet)-)?5(?:[.\-]|$)", value)
        or re.search(r"claude-opus-?4[.\-]?[78](?:[.\-]|$)", value)
    )
    if supports_xhigh:
        return ("off", "low", "medium", "high", "xhigh", "max")
    return ("off", "low", "medium", "high", "max")


def _gemini_level_profile(model: str) -> ReasoningProfile | None:
    """Resolve Gemini 3 thinking levels at model granularity."""

    value = str(model or "").strip().lower()
    if "gemini-3.6-flash" in value:
        options, default = ("minimal", "low", "medium", "high"), "medium"
    elif "gemini-3.5-flash-lite" in value:
        options, default = ("minimal", "low", "medium", "high"), "minimal"
    elif "gemini-3.5-flash" in value:
        options, default = ("minimal", "low", "medium", "high"), "medium"
    elif "gemini-3.1-pro-preview" in value:
        options, default = ("low", "medium", "high"), "high"
    elif "gemini-3.1-flash-lite-image" in value:
        options, default = ("minimal", "high"), "minimal"
    elif "gemini-3-flash-preview" in value:
        options, default = ("minimal", "low", "medium", "high"), "high"
    elif "gemini-3-pro-preview" in value:
        options, default = ("low", "high"), "high"
    elif "gemini-3" in value:
        options, default = ("low", "medium", "high"), "high"
    else:
        return None
    return ReasoningProfile(
        "gemini_level",
        options,
        default=default,
        always_enabled=True,
        note="Gemini thinkingLevel（该型号不能完全关闭）",
        native_levels=tuple((item, item.upper()) for item in options),
    )


_REASONING_MODES = {
    "openai_effort", "anthropic_adaptive", "anthropic_budget",
    "gemini_level", "gemini_budget", "qwen_budget", "thinking_toggle",
    "ollama_think", "fixed", "unsupported",
}
_OFF_CONTROLS = {
    "reasoning_effort_none", "thinking_disabled", "enable_thinking_false",
    "gemini_budget_zero", "ollama_think_false",
}
_ON_CONTROLS = {
    "thinking_enabled", "thinking_adaptive", "enable_thinking_true",
    "reasoning_split_true", "provider_default",
}


def merge_request_body(base: dict[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    """Recursively merge request objects without replacing sibling settings.

    Provider custom parameters commonly override one nested value such as
    ``generationConfig.thinkingConfig``. A top-level ``dict.update`` would also
    discard maxOutputTokens, temperature and topP from that object.
    """

    if not isinstance(overrides, Mapping):
        return base
    for key, value in overrides.items():
        current = base.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merge_request_body(current, value)
        elif isinstance(value, Mapping):
            nested: dict[str, Any] = {}
            merge_request_body(nested, value)
            base[key] = nested
        else:
            base[key] = value
    return base


def get_reasoning_profile(
    provider: str,
    model: str,
    *,
    provider_type: str | None = None,
    supports_reasoning: bool | None = None,
    declared_options: Any = None,
    declared_mode: str | None = None,
    declared_always_enabled: bool | None = None,
    declared_default: str | None = None,
    declared_off_control: str | None = None,
    declared_on_control: str | None = None,
) -> ReasoningProfile:
    """Resolve capabilities from provider protocol and model naming.

    能力判断遵循“模型显式声明 → Provider 协议默认 → 最后的模型名兜底”。
    这样动态网关可以精确收紧能力，且不会因为一个模型名恰好包含 ``think``
    就把不支持的参数发给上游。
    """

    pid = str(provider or "").strip().lower()
    mid = str(model or "").strip().lower()
    dynamic = _dynamic_provider_config(pid)
    model_config = _dynamic_model_config(pid, model)
    protocol = _provider_protocol(pid, provider_type)
    explicit_support = supports_reasoning
    if explicit_support is None and "supports_reasoning" in model_config:
        explicit_support = bool(model_config.get("supports_reasoning"))
    if explicit_support is None and "supports_reasoning" in dynamic:
        explicit_support = bool(dynamic.get("supports_reasoning"))

    explicit_options = _normalize_declared_options(declared_options)
    options_source = "explicit_request" if explicit_options else ""
    if not explicit_options:
        explicit_options = _normalize_declared_options(
            _declared_value(
                model_config,
                "reasoning_options",
                "reasoning_efforts",
                "reasoning_levels",
                "thinking_levels",
            )
        )
        if explicit_options:
            options_source = "explicit_model"
    if not explicit_options:
        explicit_options = _normalize_declared_options(
            _declared_value(
                model_config.get("metadata") if isinstance(model_config.get("metadata"), dict) else {},
                "reasoning_options",
                "reasoning_efforts",
                "reasoning_levels",
                "thinking_levels",
            )
        )
        if explicit_options:
            options_source = "explicit_model"
    if not explicit_options:
        explicit_options = _normalize_declared_options(
            _declared_value(
                dynamic,
                "reasoning_options",
                "reasoning_efforts",
                "reasoning_levels",
                "thinking_levels",
            )
        )
        if explicit_options:
            options_source = "explicit_provider"
    metadata_config = model_config.get("metadata") if isinstance(model_config.get("metadata"), dict) else {}
    explicit_mode = str(
        declared_mode
        or _declared_value(model_config, "reasoning_mode", "thinking_mode")
        or _declared_value(metadata_config, "reasoning_mode", "thinking_mode")
        or _declared_value(dynamic, "reasoning_mode", "thinking_mode")
        or ""
    ).strip().lower()
    if explicit_mode not in _REASONING_MODES:
        explicit_mode = ""

    explicit_default = normalize_reasoning_effort(
        declared_default
        or _declared_value(model_config, "reasoning_default", "thinking_default")
        or _declared_value(metadata_config, "reasoning_default", "thinking_default")
        or _declared_value(dynamic, "reasoning_default", "thinking_default"),
        default="off",
    )
    explicit_off_control = str(
        declared_off_control
        or _declared_value(model_config, "reasoning_off_control", "thinking_off_control")
        or _declared_value(metadata_config, "reasoning_off_control", "thinking_off_control")
        or _declared_value(dynamic, "reasoning_off_control", "thinking_off_control")
        or ""
    ).strip().lower()
    if explicit_off_control not in _OFF_CONTROLS:
        explicit_off_control = ""

    explicit_on_control = str(
        declared_on_control
        or _declared_value(model_config, "reasoning_on_control", "thinking_on_control")
        or _declared_value(metadata_config, "reasoning_on_control", "thinking_on_control")
        or _declared_value(dynamic, "reasoning_on_control", "thinking_on_control")
        or ""
    ).strip().lower()
    if explicit_on_control not in _ON_CONTROLS:
        explicit_on_control = ""

    explicit_always_enabled = declared_always_enabled
    if explicit_always_enabled is None and "always_enabled" in model_config:
        explicit_always_enabled = bool(model_config.get("always_enabled"))
    if explicit_always_enabled is None and "reasoning_always_enabled" in model_config:
        explicit_always_enabled = bool(model_config.get("reasoning_always_enabled"))
    if explicit_always_enabled is None and "reasoning_always_enabled" in metadata_config:
        explicit_always_enabled = bool(metadata_config.get("reasoning_always_enabled"))
    if explicit_always_enabled is None and "reasoning_always_enabled" in dynamic:
        explicit_always_enabled = bool(dynamic.get("reasoning_always_enabled"))

    explicit_model_declaration = bool(explicit_options) or any(
        key in model_config
        for key in (
            "reasoning_options", "reasoning_efforts", "reasoning_levels", "thinking_levels",
            "always_enabled", "reasoning_always_enabled", "reasoning_mode",
            "reasoning_default", "reasoning_off_control",
            "reasoning_on_control",
        )
    ) or any(
        key in metadata_config
        for key in (
            "reasoning_options", "reasoning_efforts", "reasoning_levels", "thinking_levels",
            "reasoning_always_enabled", "reasoning_mode", "reasoning_default",
            "reasoning_off_control",
            "reasoning_on_control",
        )
    )
    explicit_provider_declaration = supports_reasoning is not None or any(
        key in dynamic
        for key in (
            "supports_reasoning", "reasoning_options", "reasoning_mode",
            "reasoning_default", "reasoning_always_enabled", "reasoning_off_control",
            "reasoning_on_control",
        )
    )

    def _mode_for_declaration() -> str:
        if explicit_mode:
            return explicit_mode
        if protocol == "anthropic":
            return "anthropic_adaptive" if _anthropic_uses_adaptive_thinking(mid) else "anthropic_budget"
        if protocol == "gemini":
            return "gemini_level" if "gemini-3" in mid else "gemini_budget"
        if pid in {"aliyun", "qwen"} or "qwen" in mid:
            return "qwen_budget"
        if pid in {"deepseek", "zhipu", "xiaomi", "minimax"}:
            return "thinking_toggle"
        if protocol in {"ollama", "local"}:
            return "ollama_think"
        return "openai_effort"

    def _finish(profile: ReasoningProfile) -> ReasoningProfile:
        options = profile.options
        mode = profile.mode
        default = profile.default
        always_enabled = profile.always_enabled
        note = profile.note
        source = profile.source
        off_control = profile.off_control
        on_control = profile.on_control
        native_levels = profile.native_levels
        split_reasoning_output = profile.split_reasoning_output

        if explicit_options:
            options = explicit_options
            if mode == "unsupported":
                mode = _mode_for_declaration()
                note = "使用模型显式声明的思考档位"
            source = options_source or "explicit_model"
            if explicit_always_enabled is None:
                always_enabled = "off" not in options
        elif explicit_model_declaration:
            source = "explicit_model"
        elif explicit_provider_declaration:
            source = "explicit_provider"

        if explicit_always_enabled is not None:
            always_enabled = bool(explicit_always_enabled)
        if explicit_mode:
            mode = explicit_mode
        if explicit_default != "off":
            default = explicit_default
        if explicit_off_control:
            off_control = explicit_off_control
        if explicit_on_control:
            on_control = explicit_on_control

        if "off" in options and not off_control:
            off_control = {
                "anthropic_adaptive": "thinking_disabled",
                "anthropic_budget": "thinking_disabled",
                "gemini_budget": "gemini_budget_zero",
                "qwen_budget": "enable_thinking_false",
                "thinking_toggle": "thinking_disabled",
                "ollama_think": "ollama_think_false",
            }.get(mode)

        if mode == "thinking_toggle" and not on_control:
            on_control = "thinking_enabled"

        if mode == "unsupported" or not any(item != "off" for item in options):
            return ReasoningProfile(
                "unsupported",
                ("off",),
                default="off",
                always_enabled=False,
                note=note or "当前模型未声明思考能力",
                source=source,
            )

        active = [item for item in options if item != "off"]
        if default not in active:
            default = active[min(len(active) - 1, 1)]
        cost_warning_from = "xhigh" if any(_EFFORT_RANK[item] >= _EFFORT_RANK["xhigh"] for item in active) else None
        return ReasoningProfile(
            mode,
            tuple(options),
            default=default,
            always_enabled=always_enabled,
            note=note,
            source=source,
            cost_warning_from=cost_warning_from,
            off_control=off_control,
            on_control=on_control,
            native_levels=native_levels,
            split_reasoning_output=split_reasoning_output,
        )

    # An explicit false declaration is authoritative for custom gateways. It
    # prevents a model-name regex from silently enabling unsupported fields.
    if dynamic and explicit_support is False and not explicit_options:
        return _finish(ReasoningProfile(
            "unsupported",
            ("off",),
            default="off",
            note="当前自定义模型未声明思考能力",
            source="explicit_provider_disabled",
        ))

    if pid in {"local", "ollama"} or protocol in {"ollama", "local"}:
        if _looks_reasoning_model(mid) or explicit_support:
            return _finish(ReasoningProfile(
                "ollama_think", ("off", "medium"),
                note="Ollama think 开关", off_control="ollama_think_false",
            ))
        return _finish(ReasoningProfile("unsupported", ("off",), note="当前本地模型未声明思考能力"))

    if pid == "moonshot":
        if "kimi-k3" in mid:
            return _finish(ReasoningProfile(
                "openai_effort", ("low", "high", "max"), default="max",
                always_enabled=True,
                note="Kimi K3 始终思考；多轮与工具调用必须完整回传 reasoning_content",
                on_control="provider_default",
            ))
        if re.search(r"kimi-k2[.\-]?(?:5|6)", mid):
            return _finish(ReasoningProfile(
                "thinking_toggle", ("off", "medium"), default="medium",
                note="Kimi K2 思考开关", off_control="thinking_disabled",
                on_control="thinking_enabled",
            ))
        if "kimi-k2.7-code" in mid or "kimi-k2-7-code" in mid:
            return _finish(ReasoningProfile(
                "fixed", ("medium",), default="medium", always_enabled=True,
                note="Kimi Code 始终思考并保留历史 reasoning_content",
            ))
        return _finish(ReasoningProfile("unsupported", ("off",), note="当前模型未声明思考能力"))

    if pid == "doubao":
        if _looks_reasoning_model(mid) or explicit_support:
            return _finish(ReasoningProfile(
                "fixed", ("medium",), default="medium", always_enabled=True,
                note="当前豆包模型由上游自动决定思考深度",
            ))
        return _finish(ReasoningProfile("unsupported", ("off",), note="当前模型未声明思考能力"))

    if pid == "anthropic" or protocol == "anthropic":
        # Newer Claude models expose adaptive effort. Older gateways still
        # accept the same semantic ladder through an enabled budget.
        if _anthropic_uses_adaptive_thinking(mid):
            return _finish(ReasoningProfile(
                "anthropic_adaptive", _anthropic_adaptive_options(mid), default="high",
                note="Claude effort 与 adaptive thinking 独立控制",
                off_control="thinking_disabled",
            ))
        return _finish(ReasoningProfile(
            "anthropic_budget", ("off", "low", "medium", "high", "max"),
            note="Anthropic extended thinking budget", off_control="thinking_disabled",
        ))

    if pid == "gemini" or protocol == "gemini":
        gemini_level = _gemini_level_profile(mid)
        if gemini_level is not None:
            return _finish(gemini_level)
        if "gemini-2.5-pro" in mid:
            return _finish(ReasoningProfile("gemini_budget", ("low", "medium", "high", "max"), default="medium", always_enabled=True, note="Gemini Pro 保留最小思考预算"))
        if "gemini-2.5" in mid or explicit_support:
            return _finish(ReasoningProfile(
                "gemini_budget", ("off", "low", "medium", "high", "max"),
                note="Gemini thinking budget", off_control="gemini_budget_zero",
            ))
        return _finish(ReasoningProfile("unsupported", ("off",), note="当前 Gemini 模型未声明思考能力"))

    if pid == "grok":
        if "grok-4.5" in mid:
            return _finish(ReasoningProfile(
                "openai_effort", ("low", "medium", "high"), default="high",
                always_enabled=True,
                note="Grok 4.5 始终思考，支持 low / medium / high",
            ))
        if "grok-3-mini" in mid:
            return _finish(ReasoningProfile(
                "openai_effort", ("low", "medium", "high"), default="medium",
                always_enabled=True, note="Grok 3 Mini reasoning effort",
            ))
        if _looks_reasoning_model(mid) or "grok-build" in mid or explicit_support:
            return _finish(ReasoningProfile(
                "fixed", ("medium",), default="medium", always_enabled=True,
                note="该 xAI 模型使用固定推理模式，不接受 reasoning_effort",
            ))
        return _finish(ReasoningProfile("unsupported", ("off",), note="当前 Grok 模型未声明思考能力"))

    if pid in {"aliyun", "qwen"} or "qwen3" in mid or "qwq" in mid:
        if "qwen3.8-max" in mid:
            return _finish(ReasoningProfile(
                "openai_effort", ("off", "low", "medium", "xhigh"),
                default="xhigh",
                note="Qwen3.8-Max 使用 reasoning_effort；不可与 thinking_budget 同时发送",
                off_control="enable_thinking_false",
                on_control="enable_thinking_true",
            ))
        if _looks_reasoning_model(mid) or explicit_support:
            return _finish(ReasoningProfile(
                "qwen_budget", ("off", "low", "medium", "high", "max"),
                note="Qwen enable_thinking + thinking_budget", off_control="enable_thinking_false",
            ))
        return _finish(ReasoningProfile("unsupported", ("off",), note="当前 Qwen 模型未声明思考能力"))

    if pid == "silicon" and "qwen" in mid:
        return _finish(ReasoningProfile(
            "qwen_budget", ("off", "low", "medium", "high", "max"),
            note="SiliconFlow Qwen thinking budget", off_control="enable_thinking_false",
        ))

    if pid == "deepseek" and "deepseek-v4" in mid:
        return _finish(ReasoningProfile(
            "openai_effort", ("off", "low", "high", "max"), default="high",
            note="DeepSeek V4 原生 reasoning_effort：low / high / max",
            off_control="thinking_disabled",
            on_control="thinking_enabled",
        ))

    if pid == "zhipu" and "glm-5.2" in mid:
        return _finish(ReasoningProfile(
            "openai_effort",
            ("off", "minimal", "low", "medium", "high", "xhigh", "max"),
            default="max",
            note="GLM-5.2：none/minimal 关闭，low/medium 映射 high，xhigh 映射 max",
            off_control="thinking_disabled",
            on_control="thinking_enabled",
        ))

    if pid in {"deepseek", "zhipu", "xiaomi"} or (pid == "silicon" and _looks_reasoning_model(mid)):
        if _looks_reasoning_model(mid) or explicit_support:
            return _finish(ReasoningProfile(
                "thinking_toggle", ("off", "medium"), note="该接口只提供思考开关",
                off_control="thinking_disabled",
                on_control="thinking_enabled",
            ))
        return _finish(ReasoningProfile("unsupported", ("off",), note="当前模型未声明思考能力"))

    if pid == "minimax":
        if _looks_reasoning_model(mid) or explicit_support:
            return _finish(ReasoningProfile(
                "thinking_toggle", ("off", "medium"), default="medium",
                note="MiniMax M3 使用 adaptive/disabled；reasoning_split 仅分离输出",
                off_control="thinking_disabled",
                on_control="thinking_adaptive",
                split_reasoning_output=True,
            ))
        return _finish(ReasoningProfile("unsupported", ("off",), note="当前 MiniMax 模型未声明思考能力"))

    if protocol in {"openai", "custom"} and (bool(explicit_support) or _looks_reasoning_model(mid)):
        options = _openai_options(mid)
        always_enabled = "off" not in options
        return _finish(ReasoningProfile(
            "openai_effort", options, default="medium", always_enabled=always_enabled,
            note="OpenAI-compatible reasoning_effort",
            off_control="reasoning_effort_none" if _openai_supports_native_off(mid) else None,
        ))

    return _finish(ReasoningProfile("unsupported", ("off",), note="当前模型未声明思考能力"))


def _nearest_active(requested: str, options: tuple[str, ...], default: str) -> str:
    active = [item for item in options if item != "off"]
    if not active:
        return default
    if requested in active:
        return requested
    requested_rank = _EFFORT_RANK.get(requested, _EFFORT_RANK.get(default, 2))
    # Prefer the closest available level, then the lower cost level on ties.
    return min(active, key=lambda item: (abs(_EFFORT_RANK[item] - requested_rank), _EFFORT_RANK[item]))


def resolve_reasoning_request(
    provider: str,
    model: str,
    *,
    enable_thinking: bool = False,
    requested_effort: Any = "off",
    provider_type: str | None = None,
    supports_reasoning: bool | None = None,
    declared_options: Any = None,
    declared_mode: str | None = None,
    declared_always_enabled: bool | None = None,
    declared_default: str | None = None,
    declared_off_control: str | None = None,
    declared_on_control: str | None = None,
) -> ReasoningResolution:
    profile = get_reasoning_profile(
        provider,
        model,
        provider_type=provider_type,
        supports_reasoning=supports_reasoning,
        declared_options=declared_options,
        declared_mode=declared_mode,
        declared_always_enabled=declared_always_enabled,
        declared_default=declared_default,
        declared_off_control=declared_off_control,
        declared_on_control=declared_on_control,
    )
    requested = normalize_reasoning_effort(requested_effort)
    if profile.mode == "unsupported":
        return ReasoningResolution(requested, None, False, profile.mode, None, None, profile)

    if profile.always_enabled:
        # 厂商强制思考时，关闭只是“未指定档位”的旧客户端语义；
        # 使用模型默认档位，用户显式选择的可用档位仍应保留。
        effective = (
            profile.default
            if requested == "off"
            else _nearest_active(requested, profile.options, profile.default)
        )
        enabled = True
    elif not enable_thinking or requested == "off":
        return ReasoningResolution(
            requested, "off", False, profile.mode, None, None, profile,
            native_control=profile.off_control,
        )
    else:
        effective = _nearest_active(requested, profile.options, profile.default)
        enabled = True

    native_level_map = dict(profile.native_levels)
    native_effort = (
        native_level_map.get(
            effective,
            effective.upper() if profile.mode == "gemini_level" else effective,
        )
        if profile.mode in {"openai_effort", "anthropic_adaptive", "gemini_level"}
        else None
    )
    budget = THINKING_BUDGET_TOKENS.get(effective) if profile.mode in {"anthropic_budget", "gemini_budget", "qwen_budget"} else None
    return ReasoningResolution(
        requested, effective, enabled, profile.mode, native_effort, budget, profile,
        native_control=profile.on_control or "enable",
    )


_REASONING_BODY_KEYS = {
    "reasoning_effort",
    "thinking",
    "enable_thinking",
    "thinking_budget",
    "thinkingBudget",
    "reasoning_split",
    "output_config",
    "reasoning",
    "think",
}


def _clear_reasoning_controls(body: dict[str, Any]) -> None:
    """Remove stale top-level and nested reasoning controls before applying one profile."""

    for key in _REASONING_BODY_KEYS:
        body.pop(key, None)
    generation = body.get("generationConfig")
    if isinstance(generation, dict):
        generation.pop("thinkingConfig", None)


def _apply_native_disable(body: dict[str, Any], resolution: ReasoningResolution) -> None:
    control = resolution.native_control
    if control == "reasoning_effort_none":
        body["reasoning_effort"] = "none"
    elif control == "thinking_disabled":
        body["thinking"] = {"type": "disabled"}
    elif control == "enable_thinking_false":
        body["enable_thinking"] = False
    elif control == "gemini_budget_zero":
        generation = body.setdefault("generationConfig", {})
        if isinstance(generation, dict):
            generation["thinkingConfig"] = {"thinkingBudget": 0}
    elif control == "ollama_think_false":
        body["think"] = False


def _apply_native_enable(body: dict[str, Any], resolution: ReasoningResolution) -> None:
    """Apply an enable switch that is independent from the effort value."""

    control = resolution.native_control
    if control == "thinking_enabled":
        body["thinking"] = {"type": "enabled"}
    elif control == "thinking_adaptive":
        body["thinking"] = {"type": "adaptive"}
    elif control == "enable_thinking_true":
        body["enable_thinking"] = True
    elif control == "reasoning_split_true":
        # Backward compatibility for custom providers that declared this old
        # output-only switch. It deliberately does not claim to enable thought.
        body["reasoning_split"] = True


def apply_reasoning_to_payload(
    body: dict[str, Any],
    provider: str,
    model: str,
    *,
    enable_thinking: bool = False,
    requested_effort: Any = "off",
    provider_type: str | None = None,
    supports_reasoning: bool | None = None,
    declared_options: Any = None,
    declared_mode: str | None = None,
    declared_always_enabled: bool | None = None,
    declared_default: str | None = None,
    declared_off_control: str | None = None,
    declared_on_control: str | None = None,
) -> ReasoningResolution:
    """Mutate a provider request body using the resolved native controls."""

    resolution = resolve_reasoning_request(
        provider,
        model,
        enable_thinking=enable_thinking,
        requested_effort=requested_effort,
        provider_type=provider_type,
        supports_reasoning=supports_reasoning,
        declared_options=declared_options,
        declared_mode=declared_mode,
        declared_always_enabled=declared_always_enabled,
        declared_default=declared_default,
        declared_off_control=declared_off_control,
        declared_on_control=declared_on_control,
    )
    _clear_reasoning_controls(body)

    if not resolution.enabled or resolution.effective is None:
        _apply_native_disable(body, resolution)
        return resolution

    mode = resolution.mode
    if mode == "openai_effort":
        body["reasoning_effort"] = resolution.native_effort
        _apply_native_enable(body, resolution)
    elif mode == "anthropic_adaptive":
        body["thinking"] = {"type": "adaptive", "display": "summarized"}
        body["output_config"] = {"effort": resolution.native_effort}
    elif mode == "anthropic_budget":
        body["thinking"] = {"type": "enabled", "budget_tokens": resolution.budget_tokens}
    elif mode == "gemini_level":
        generation = body.get("generationConfig")
        if not isinstance(generation, dict):
            generation = {}
            body["generationConfig"] = generation
        generation["thinkingConfig"] = {"thinkingLevel": resolution.native_effort}
    elif mode == "gemini_budget":
        generation = body.get("generationConfig")
        if not isinstance(generation, dict):
            generation = {}
            body["generationConfig"] = generation
        generation["thinkingConfig"] = {"thinkingBudget": resolution.budget_tokens}
    elif mode == "qwen_budget":
        body["enable_thinking"] = True
        body["thinking_budget"] = resolution.budget_tokens
    elif mode == "thinking_toggle":
        _apply_native_enable(body, resolution)
    elif mode == "ollama_think":
        body["think"] = True
    if resolution.profile.split_reasoning_output:
        body["reasoning_split"] = True
    return resolution


def reasoning_options_for_frontend(provider: str, model: str, **kwargs: Any) -> dict[str, Any]:
    """Small JSON-safe capability view for diagnostics/API consumers."""

    profile = get_reasoning_profile(provider, model, **kwargs)
    return {
        "mode": profile.mode,
        "options": list(profile.options),
        "default": profile.default,
        "always_enabled": profile.always_enabled,
        "note": profile.note,
        "source": profile.source,
        "cost_warning_from": profile.cost_warning_from,
        "off_control": profile.off_control,
        "on_control": profile.on_control,
        "split_reasoning_output": profile.split_reasoning_output,
        "can_disable": "off" in profile.options and bool(profile.off_control),
        # Kept for older frontends; unlike the old value this now means that a
        # visible Off option has an actual provider-native implementation.
        "off_is_guaranteed": "off" in profile.options and bool(profile.off_control),
        "labels": {
            option: REASONING_EFFORT_LABELS.get(option, option)
            for option in profile.options
        },
    }
