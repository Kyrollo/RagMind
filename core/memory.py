"""
core/memory.py
Simple in-memory multi-turn conversation history (Layer 7).
"""
from config import MAX_HISTORY_TURNS

_history: list[tuple[str, str]] = []


def format_history() -> str:
    """Return the last MAX_HISTORY_TURNS turns formatted as a prompt string."""
    if not _history:
        return "No previous conversation."
    return "\n".join(
        f"User: {u}\nAssistant: {a}"
        for u, a in _history[-MAX_HISTORY_TURNS:]
    )


def add_to_history(user_msg: str, assistant_msg: str) -> None:
    """Append a Q&A turn; evict oldest if over the limit."""
    _history.append((user_msg, assistant_msg))
    if len(_history) > MAX_HISTORY_TURNS:
        _history.pop(0)


def clear_history() -> None:
    """Reset the conversation history."""
    _history.clear()


def get_history() -> list[tuple[str, str]]:
    return list(_history)
