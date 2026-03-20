"""
utils/helpers.py
Miscellaneous helper utilities.
"""
import re


def strip_rewriter_prefix(text: str) -> str:
    """Remove common LLM prefixes from rewritten queries."""
    return re.sub(
        r"(?i)(improved question:|standalone question:|rewritten[:\s]+|answer:|question:)",
        "",
        text.strip(),
    ).strip()


def truncate(text: str, max_chars: int = 80) -> str:
    """Truncate text with an ellipsis if it exceeds max_chars."""
    return text[:max_chars] + "…" if len(text) > max_chars else text


def source_label_from_docs(docs: list, used_web: bool = False) -> str:
    """
    Derive a source badge label from the retrieved document list.
    Returns: "pdf" | "web" | "hybrid" | "none"
    """
    has_pdf = any(d.metadata.get("type", "pdf") == "pdf" for d in docs)
    has_web = used_web or any(d.metadata.get("type") == "web" for d in docs)
    if has_pdf and has_web: return "hybrid"
    if has_web:             return "web"
    if has_pdf:             return "pdf"
    return "none"
