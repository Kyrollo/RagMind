"""
main.py
Public pipeline API — import this module from app.py or use standalone.

Usage:
    from main import load_document, run_pipeline, clear_history

    load_document("my_report.pdf")
    result = run_pipeline("What are the key findings?")
    print(result["answer"])
    print(result["source_label"])   # pdf | web | hybrid | none
    print(result["steps"])          # full 7-layer trace
"""
import re
from core.vectorstore import build_vectorstore_from_file
from core.memory import format_history, add_to_history, clear_history
from core.chains import question_rewriter
from core.state import GraphState
from agents.graph import app
from config import MAX_RETRIES, WEB_SEARCH_ENABLED
from core.web_search import web_search_fallback


# ── Public API ────────────────────────────────────────────────────────────────

def load_document(file_path: str) -> tuple[int, int]:
    """
    Load a PDF, DOCX, or TXT file into the vector store.
    Clears conversation history.
    Returns (num_pages, num_chunks).
    """
    clear_history()
    return build_vectorstore_from_file(file_path)


def run_pipeline(question: str, force_web: bool = False) -> dict:
    """
    Run the full 7-layer Self-RAG + CRAG pipeline.

    Args:
        question:  The user's question.
        force_web: If True, skip PDF retrieval and go straight to Tavily.

    Returns a dict with keys:
        answer       (str)       — final answer
        steps        (list[str]) — full pipeline trace
        retries      (int)       — number of query rewrites
        source_label (str)       — "pdf" | "web" | "hybrid" | "none" | "error"
    """
    history_str = format_history()

    # Layer 1 — condense follow-up with history
    try:
        condensed_q = question_rewriter.invoke(
            {"question": question, "history": history_str}
        ).strip()
        condensed_q = re.sub(
            r"(?i)(improved question:|standalone question:|rewritten[:\s]+)", "",
            condensed_q,
        ).strip() or question
    except Exception:
        condensed_q = question

    initial_state: GraphState = {
        "question":          condensed_q,
        "original_question": question,
        "generation":        "",
        "documents":         [],
        "retry_count":       0,
        "used_web":          False,
        "steps":             [
            f"🧠 **Layer 1 — Query Understanding:** `{condensed_q[:80]}{'…' if len(condensed_q)>80 else ''}`"
        ],
        "history": history_str,
    }

    # Optional force-web shortcut
    if force_web and WEB_SEARCH_ENABLED:
        web_docs = web_search_fallback(condensed_q)
        if web_docs:
            initial_state["documents"] = web_docs
            initial_state["used_web"]  = True
            initial_state["steps"].append(
                f"🌐 **Force Web Search:** Retrieved {len(web_docs)} results"
            )

    # Run graph
    final_state = None
    try:
        for output in app.stream(initial_state):
            for _, state in output.items():
                final_state = state
    except Exception as e:
        return {
            "answer":       f"Pipeline error: {e}",
            "steps":        initial_state["steps"],
            "retries":      0,
            "source_label": "error",
        }

    if final_state is None:
        return {"answer": "No output from pipeline.", "steps": [], "retries": 0, "source_label": "none"}

    answer       = final_state.get("generation", "No answer generated.")
    steps        = final_state.get("steps", [])
    retries      = final_state.get("retry_count", 0)
    used_web     = final_state.get("used_web", False)
    docs         = final_state.get("documents", [])

    has_pdf = any(d.metadata.get("type", "pdf") == "pdf" for d in docs)
    has_web = used_web or any(d.metadata.get("type") == "web" for d in docs)

    if has_pdf and has_web:   source_label = "hybrid"
    elif has_web:             source_label = "web"
    elif has_pdf:             source_label = "pdf"
    else:                     source_label = "none"

    steps.append(f"🏷️ **Layer 7 — Source Attribution:** {source_label.upper()}")

    add_to_history(question, answer)
    return {
        "answer":       answer,
        "steps":        steps,
        "retries":      retries,
        "source_label": source_label,
    }
