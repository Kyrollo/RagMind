"""
core/chains.py
All LLM chains and the JSON parser used across the pipeline.
"""
import re, json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.llms import llm_grader, llm_generator


# ── Helpers ───────────────────────────────────────────────────────────────────
def format_docs(docs: list) -> str:
    """Render a list of Documents into a prompt-ready string with source labels."""
    parts = []
    for i, d in enumerate(docs, 1):
        src  = d.metadata.get("source", "?")
        page = d.metadata.get("page", "")
        kind = d.metadata.get("type", "pdf")
        label = f"[🌐 Web | {src}]" if kind == "web" else f"[📄 Doc p.{page} | {src}]"
        parts.append(f"{label}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)

def parse_binary_score(text: str) -> str:
    """
    Robust 4-strategy parser for binary_score JSON responses.
    Strategies (in order):
      1. json.loads()
      2. Regex on binary_score key
      3. Keyword scan (yes/no)
      4. Default "yes" (safe fallback — avoids silently dropping valid docs)
    """
    text = text.strip()
    try:
        val = str(json.loads(text).get("binary_score", "")).strip().lower()
        if val in ("yes", "no"):
            return val
    except Exception:
        pass
    m = re.search(r'["\'`]?binary_score["\'`]?\s*:\s*["\'`]?(yes|no)["\'`]?', text.lower())
    if m:
        return m.group(1)
    lower = text.lower()
    if "yes" in lower and "no"  not in lower: return "yes"
    if "no"  in lower and "yes" not in lower: return "no"
    return "yes"

# ── Layer 1: Query Rewriter ────────────────────────────────────────────────────
question_rewriter = (
    ChatPromptTemplate.from_messages([
        ("system",
         "You are a query optimization expert for vector-store retrieval. "
         "Given conversation history and a follow-up question, produce ONE improved standalone query. "
         "Resolve pronouns, expand abbreviations, add semantically related retrieval terms. "
         "Return ONLY the improved question. No explanation, no prefix."),
        ("human",
         "Conversation history:\n{history}\n\n"
         "Follow-up question: {question}\n\n"
         "Standalone improved question:"),
    ])
    | llm_generator
    | StrOutputParser()
)

# ── Layer 2: Retrieval Grader ─────────────────────────────────────────────────
retrieval_grader = (
    ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict document relevance grader. "
         "Given a question and a document chunk, decide if the chunk contains "
         "information relevant to answering the question. "
         "Consider keyword matches AND semantic meaning. "
         "Respond ONLY with valid JSON: {{\"binary_score\": \"yes\"}} or {{\"binary_score\": \"no\"}}. "
         "No other text."),
        ("human",
         "Question: {question}\n\nDocument chunk:\n{document}\n\nIs this chunk relevant?"),
    ])
    | llm_grader
    | StrOutputParser()
)

# ── Layer 3: RAG Generation Chain ─────────────────────────────────────────────
rag_chain = (
    ChatPromptTemplate.from_messages([
        ("system",
         "You are a precise, helpful assistant answering questions based ONLY on provided context.\n\n"
         "Rules:\n"
         "- Use ONLY the provided context. Never use prior knowledge.\n"
         "- Cite page numbers for document sources (According to page X...).\n"
         "- Cite titles for web sources (According to [title]...).\n"
         "- If context is a mix, label each fact clearly.\n"
         "- If context is insufficient, say: 'The document does not contain enough information.'\n"
         "- Be concise but complete (3–6 sentences)."),
        ("human",
         "Conversation history:\n{history}\n\n"
         "Context:\n{context}\n\n"
         "Question: {question}\n\nAnswer:"),
    ])
    | llm_generator
    | StrOutputParser()
)

# ── Layer 4: Hallucination Grader ─────────────────────────────────────────────
hallucination_grader = (
    ChatPromptTemplate.from_messages([
        ("system",
         "You are a hallucination detector. Check if every claim in the generated answer "
         "is supported by the source documents. If ANY claim is unsupported, it is a hallucination. "
         "Respond ONLY with valid JSON: {{\"binary_score\": \"yes\"}} (grounded) "
         "or {{\"binary_score\": \"no\"}} (hallucinated)."),
        ("human",
         "Source documents:\n{documents}\n\n"
         "Generated answer:\n{generation}\n\n"
         "Is the answer fully grounded?"),
    ])
    | llm_grader
    | StrOutputParser()
)

# ── Layer 5: Answer Quality Grader ────────────────────────────────────────────
answer_grader = (
    ChatPromptTemplate.from_messages([
        ("system",
         "You are an answer quality assessor. Decide if the answer resolves the user's question. "
         "Partial but relevant answers count as yes. "
         "Respond ONLY with valid JSON: {{\"binary_score\": \"yes\"}} or {{\"binary_score\": \"no\"}}. "
         "No other text."),
        ("human",
         "Question: {question}\n\nAnswer: {generation}\n\nDoes this answer resolve the question?"),
    ])
    | llm_grader
    | StrOutputParser()
)