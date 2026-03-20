"""
agents/nodes.py
LangGraph node functions — one function per pipeline layer.
"""
import re
from langchain_core.documents import Document
from core.chains import (
    retrieval_grader, rag_chain, hallucination_grader,
    answer_grader, question_rewriter, format_docs, parse_binary_score
)
from core.vectorstore import get_retriever
from core.web_search import web_search_fallback
from core.state import GraphState
from config import MAX_RETRIES, MIN_RELEVANT, WEB_SEARCH_ENABLED


# ── Layer 2: Retrieve ─────────────────────────────────────────────────────────
def retrieve(state: GraphState) -> GraphState:
    question = state["question"]
    steps    = state.get("steps", [])
    retriever = get_retriever()

    if retriever is None:
        steps.append("❌ No document loaded — upload a file first")
        return {**state, "documents": [], "steps": steps}

    documents = retriever.invoke(question)
    steps.append(f"🔍 **Layer 2 — Retrieve:** Fetched {len(documents)} candidate chunks")
    return {**state, "documents": documents, "steps": steps}


# ── Layer 3: Grade Documents ──────────────────────────────────────────────────
def grade_documents(state: GraphState) -> GraphState:
    question  = state["question"]
    documents = state["documents"]
    steps     = state.get("steps", [])

    filtered, kept = [], 0
    for doc in documents:
        raw   = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        grade = parse_binary_score(raw)
        if grade == "yes":
            kept += 1
            filtered.append(doc)

    steps.append(f"📊 **Layer 3 — Grade Docs:** {kept}/{len(documents)} chunks relevant")
    return {**state, "documents": filtered, "steps": steps}


# ── Layer 4b: Web Search Fallback ─────────────────────────────────────────────
def web_search_node(state: GraphState) -> GraphState:
    question = state["question"]
    steps    = state.get("steps", [])

    web_docs = web_search_fallback(question)
    if web_docs:
        steps.append(f"🌐 **Layer 4b — Web Fallback:** Retrieved {len(web_docs)} results via Tavily")
    else:
        steps.append("🌐 **Layer 4b — Web Fallback:** Web search returned no results")

    return {**state, "documents": web_docs, "used_web": True, "steps": steps}


# ── Layer 4: Generate ─────────────────────────────────────────────────────────
def generate(state: GraphState) -> GraphState:
    question  = state["question"]
    documents = state["documents"]
    history   = state.get("history", "No previous conversation.")
    steps     = state.get("steps", [])

    context    = format_docs(documents) if documents else "No context available."
    generation = rag_chain.invoke({"context": context, "question": question, "history": history})

    steps.append(f"✨ **Layer 4 — Generate:** Answer produced ({len(generation.split())} words)")
    return {**state, "generation": generation, "steps": steps}


# ── Layer 1 (rewrite): Transform Query ───────────────────────────────────────
def transform_query(state: GraphState) -> GraphState:
    question    = state["question"]
    history     = state.get("history", "")
    steps       = state.get("steps", [])
    retry_count = state.get("retry_count", 0) + 1

    raw_rewrite = question_rewriter.invoke({"question": question, "history": history})
    better_q    = re.sub(
        r"(?i)(improved question:|standalone question:|rewritten[:\s]+)", "",
        raw_rewrite.strip()
    ).strip()

    label = better_q[:75] + "…" if len(better_q) > 75 else better_q
    steps.append(f"🔄 **Layer 1 — Rewrite (attempt {retry_count}/{MAX_RETRIES}):** `{label}`")
    return {**state, "question": better_q, "retry_count": retry_count, "steps": steps}


# ── Edge: Decide after grading ────────────────────────────────────────────────
def decide_to_generate(state: GraphState) -> str:
    docs        = state.get("documents", [])
    retry_count = state.get("retry_count", 0)
    used_web    = state.get("used_web", False)

    if len(docs) >= MIN_RELEVANT:
        return "generate"
    if not used_web and WEB_SEARCH_ENABLED:
        return "web_search"
    if retry_count < MAX_RETRIES:
        return "transform_query"
    return "generate"


# ── Edge: Grade generation quality ────────────────────────────────────────────
def grade_generation_quality(state: GraphState) -> str:
    question   = state["question"]
    documents  = state["documents"]
    generation = state["generation"]
    steps      = state.get("steps", [])
    retry      = state.get("retry_count", 0)

    # Layer 5 — Hallucination check
    raw_h    = hallucination_grader.invoke({
        "documents":  format_docs(documents) if documents else "No context.",
        "generation": generation,
    })
    grounded = parse_binary_score(raw_h) == "yes"

    if not grounded:
        if retry >= MAX_RETRIES:
            steps.append("⚠️ **Layer 5 — Hallucination:** Max retries reached → returning best answer")
            state["steps"] = steps
            return "useful"
        steps.append("❌ **Layer 5 — Hallucination:** Answer not grounded → regenerating…")
        state["steps"] = steps
        return "not_supported"

    steps.append("✅ **Layer 5 — Hallucination:** Answer fully grounded ✓")

    # Layer 6 — Answer quality check
    raw_a  = answer_grader.invoke({"question": question, "generation": generation})
    useful = parse_binary_score(raw_a) == "yes"

    if useful:
        steps.append("✅ **Layer 6 — Quality:** Answer resolves the question ✓")
        state["steps"] = steps
        return "useful"
    if retry >= MAX_RETRIES:
        steps.append("⚠️ **Layer 6 — Quality:** Max retries reached → returning best answer")
        state["steps"] = steps
        return "useful"

    steps.append("❌ **Layer 6 — Quality:** Answer incomplete → rewriting query…")
    state["steps"] = steps
    return "not_useful"
