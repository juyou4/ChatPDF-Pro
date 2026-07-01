import re
import unicodedata


_GREEK_SYMBOL_TO_NAME = {
    "α": "alpha",
    "ᾱ": "alpha_bar",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ϵ": "epsilon",
    "θ": "theta",
    "λ": "lambda",
    "μ": "mu",
    "σ": "sigma",
    "ϕ": "phi",
    "φ": "phi",
    "ψ": "psi",
    "ω": "omega",
}
_GREEK_NAME_TO_SYMBOL = {value: key for key, value in _GREEK_SYMBOL_TO_NAME.items()}
_MATH_SYMBOL_REPLACEMENTS = {
    "√": " sqrt ",
    "∥": " || ",
    "‖": " || ",
    "−": "-",
    "×": " x ",
    "·": " ",
    "∗": "*",
    "≤": " <= ",
    "≥": " >= ",
    "≈": " approx ",
    "≃": " approx ",
    "∼": " ~ ",
    "∑": " sum ",
    "∏": " prod ",
}
_FORMULA_HINT_RE = re.compile(
    r"[=_^{}\\]|sqrt|frac|sum|prod|alpha|beta|theta|epsilon|lambda|gamma|delta|"
    r"[αβγδεϵθλμσϕφψω√∥‖]",
    re.IGNORECASE,
)


def looks_formula_like(text: str) -> bool:
    """Return True when text contains generic math/formula markers."""
    sample = str(text or "")
    if not sample:
        return False
    if _FORMULA_HINT_RE.search(sample):
        return True
    return bool(re.search(r"\b[a-zA-Z]\s*[_^]\s*\w+\b|\|\|[^|]{1,80}\|\|", sample))


def normalize_formula_text(text: str) -> str:
    """Normalize common OCR/LaTeX math variants without adding document-specific terms."""
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    for symbol, name in _GREEK_SYMBOL_TO_NAME.items():
        normalized = normalized.replace(symbol, f" {name} ")
    normalized = re.sub(r"\\bar\s*\{\s*([a-zA-Z]+)\s*\}", r"\1_bar", normalized)
    normalized = re.sub(r"\\overline\s*\{\s*([a-zA-Z]+)\s*\}", r"\1_bar", normalized)
    normalized = re.sub(r"([a-zA-Z]+)\s*[\u0304¯]\s*", r"\1_bar", normalized)
    normalized = normalized.replace("\\sqrt", " sqrt ")
    normalized = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r" \1 / \2 ", normalized)
    for symbol, replacement in _MATH_SYMBOL_REPLACEMENTS.items():
        normalized = normalized.replace(symbol, replacement)
    normalized = normalized.replace("{", " ").replace("}", " ")
    normalized = re.sub(r"([a-zA-Z]+)\s*_\s*\{\s*([^{}]+)\s*\}", r"\1_\2", normalized)
    normalized = re.sub(r"([a-zA-Z]+(?:_bar)?)\s*_\s*([a-z0-9]+)\b", r"\1_\2", normalized)
    normalized = re.sub(r"([a-zA-Z]+)\s*\^\s*\{\s*([^{}]+)\s*\}", r"\1^\2", normalized)
    normalized = re.sub(r"([a-zA-Z]+)\s+bar\s+([a-z0-9]+)\b", r"\1_bar_\2", normalized)
    normalized = re.sub(r"\b(alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|phi|psi|omega)\s+bar\b", r"\1_bar", normalized)
    normalized = re.sub(r"\b(alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|phi|psi|omega)_bar([a-z0-9])\b", r"\1_bar_\2", normalized)
    normalized = re.sub(r"\b([a-zA-Z])\s+([0-9t])\b", r"\1_\2", normalized)
    normalized = re.sub(r"\b([a-zA-Z]+)\s+([0-9t])\b", r"\1_\2", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def compact_formula_text(text: str) -> str:
    """Build a compact key that ignores spacing, punctuation and subscript separators."""
    normalized = normalize_formula_text(text)
    return re.sub(r"[^a-z0-9]+", "", normalized)


def _token_aliases(token: str) -> set[str]:
    token = normalize_formula_text(token)
    aliases = {token}
    compact = compact_formula_text(token)
    if compact:
        aliases.add(compact)
    for name, symbol in _GREEK_NAME_TO_SYMBOL.items():
        if name in token:
            aliases.add(token.replace(name, symbol))
    if "_bar_" in token:
        aliases.add(token.replace("_bar_", " bar "))
        aliases.add(token.replace("_bar_", "bar"))
    if "_bar" in token:
        aliases.add(token.replace("_bar", " bar"))
        aliases.add(token.replace("_bar", "bar"))
    if "_" in token:
        aliases.add(token.replace("_", " "))
        aliases.add(token.replace("_", ""))
    if token.startswith("sqrt"):
        aliases.add(token.replace("sqrt", "√", 1))
    return {alias.strip() for alias in aliases if alias and alias.strip()}


def build_formula_alias_text(text: str) -> str:
    """Append normalized aliases for formula-like text so retrieval/matching survives OCR variants."""
    source = str(text or "")
    if not source or not looks_formula_like(source):
        return source
    normalized = normalize_formula_text(source)
    aliases: set[str] = {normalized}
    for token in re.findall(r"[a-z]+(?:_bar)?(?:_[a-z0-9]+)?|[a-z]_[a-z0-9]+|sqrt|[a-z]\^[0-9]+", normalized):
        for alias in _token_aliases(token):
            alias_norm = normalize_formula_text(alias)
            if (
                alias_norm == "sqrt"
                or "_" in alias_norm
                or "bar" in alias_norm
                or len(alias_norm) >= 3
            ):
                aliases.add(alias_norm)
    aliases.discard("")
    alias_text = " ".join(sorted(aliases, key=lambda value: (len(value), value)))
    return f"{source}\n\n[formula_aliases] {alias_text}" if alias_text else source


def formula_term_matches(term: str, text: str) -> bool:
    """Generic math-aware containment check for eval and retrieval diagnostics."""
    if not term or not text:
        return False
    normalized_term = normalize_formula_text(term)
    normalized_text = normalize_formula_text(text)
    if normalized_term and normalized_term in normalized_text:
        return True
    compact_term = compact_formula_text(term)
    compact_text = compact_formula_text(text)
    if compact_term and compact_term in compact_text:
        return True
    for alias in _token_aliases(term):
        normalized_alias = normalize_formula_text(alias)
        if normalized_alias and normalized_alias in normalized_text:
            return True
        compact_alias = compact_formula_text(alias)
        if compact_alias and compact_alias in compact_text:
            return True
    return False


_TECHNICAL_BOUNDARY_CHARS = r"A-Za-z0-9_\-"


def _literal_anchor_matches(term: str, text: str) -> bool:
    clean = str(term or "").strip()
    if not clean or not text:
        return False
    if re.fullmatch(r"[\u4e00-\u9fff]{2,}", clean):
        return clean in str(text or "")
    pieces = [re.escape(piece) for piece in re.split(r"\s+", clean) if piece]
    if not pieces:
        return False
    body = r"[\s_\-]+".join(pieces)
    pattern = rf"(?<![{_TECHNICAL_BOUNDARY_CHARS}]){body}(?![{_TECHNICAL_BOUNDARY_CHARS}])"
    return re.search(pattern, str(text or ""), re.IGNORECASE) is not None


def technical_anchor_matches(term: str, text: str) -> bool:
    """Match generic question anchors without substring hits inside larger tokens.

    Academic-paper anchors often include short numbers, acronyms, dataset names,
    and formula tokens. Plain containment can match short anchors inside larger
    unrelated tokens. This helper keeps Chinese phrase containment, applies token
    boundaries for technical Latin/numeric anchors, and falls back to formula
    normalization for OCR/LaTeX variants such as ``alpha_bar_t``.
    """
    clean = str(term or "").strip()
    sample = str(text or "")
    if not clean or not sample:
        return False
    if _literal_anchor_matches(clean, sample):
        return True
    normalized = normalize_formula_text(clean)
    formula_specific = (
        looks_formula_like(clean)
        and (
            "_" in normalized
            or "^" in normalized
            or "bar" in normalized
            or normalized == "sqrt"
            or normalized in _GREEK_NAME_TO_SYMBOL
        )
    )
    return bool(formula_specific and formula_term_matches(clean, sample))
