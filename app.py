"""
app.py — Streamlit UI for RagMind (Self-RAG + CRAG 7-Layer Pipeline)
Run: streamlit run app.py
"""
import os, time, tempfile, pathlib
import streamlit as st

st.set_page_config(
    page_title="RagMind — Self-RAG + CRAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
:root {
    --bg:#0a0e1a; --acc:#7c3aed; --acc2:#a78bfa; --cyan:#06b6d4;
    --green:#10b981; --rose:#f43f5e; --gold:#f59e0b;
    --t1:#eef2ff; --t2:#a5b4fc; --t3:#6b7280;
}
body,[data-testid="stAppViewContainer"]{
    background:var(--bg)!important;
    font-family:'Inter',sans-serif!important;
    color:var(--t1)!important;
}
[data-testid="stChatMessage"]{
    background:rgba(30,37,70,.7)!important;
    border-radius:12px!important;
    margin-bottom:.5rem!important;
}
.badge{
    display:inline-block;padding:3px 10px;border-radius:20px;
    font-size:.72rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
    margin-bottom:.6rem;
}
.bp{background:rgba(124,58,237,.2);color:#a78bfa;border:1px solid #7c3aed;}
.bw{background:rgba(6,182,212,.2); color:#67e8f9;border:1px solid #06b6d4;}
.bh{background:rgba(16,185,129,.2);color:#6ee7b7;border:1px solid #10b981;}
.bn{background:rgba(107,114,128,.2);color:#9ca3af;border:1px solid #374151;}
.be{background:rgba(239,68,68,.2); color:#fca5a5;border:1px solid #ef4444;}
.trace{
    background:rgba(30,37,70,.8);border-left:3px solid var(--acc2);
    border-radius:8px;padding:.5rem .8rem;margin-bottom:.4rem;
    font-size:.8rem;color:var(--t2);
}
</style>
""", unsafe_allow_html=True)

# ── Pipeline import ───────────────────────────────────────────────────────────
try:
    from main import load_document, run_pipeline, clear_history
    from config import (
        GRADER_MODEL, GENERATOR_MODEL, EMBED_MODEL,
        WEB_SEARCH_ENABLED,
    )
    READY = True
except Exception as e:
    st.error(f"❌ Pipeline not loaded: {e}")
    st.info("Make sure **Ollama is running** (`ollama serve`) and all dependencies are installed.")
    st.stop()

# ── Source badge map ──────────────────────────────────────────────────────────
BADGE = {
    "pdf":    ("📄 PDF Document", "bp"),
    "web":    ("🌐 Web Search",   "bw"),
    "hybrid": ("🔀 PDF + Web",    "bh"),
    "none":   ("⬜ No Context",   "bn"),
    "error":  ("❌ Error",        "be"),
}

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in dict(messages=[], doc_loaded=False, doc_name="",
                  doc_pages=0, doc_chunks=0, queries=0, force_web=False).items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Header ────────────────────────────────────────────────────────────────────
c1, c2 = st.columns([4, 1])
with c1:
    st.markdown("## 🧠 RagMind — Self-RAG + CRAG")
    st.caption("7-Layer Adaptive Retrieval · Hallucination Guard · Web Fallback · Chat Memory")
with c2:
    st.markdown(f"""
    <div style='font-size:.72rem;color:#6b7280;text-align:right;padding-top:8px'>
    ⚙️ <b style='color:#a5b4fc'>{GRADER_MODEL}</b> grader<br>
    ⚙️ <b style='color:#a5b4fc'>{GENERATOR_MODEL}</b> generator<br>
    {'🌐 Web enabled' if WEB_SEARCH_ENABLED else '📵 Web disabled (no Tavily key)'}
    </div>""", unsafe_allow_html=True)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Upload Document")
    uploaded = st.file_uploader("PDF / DOCX / TXT", type=["pdf","docx","doc","txt"])

    if uploaded and (not st.session_state.doc_loaded or uploaded.name != st.session_state.doc_name):
        with st.spinner(f"Processing {uploaded.name}…"):
            suffix = pathlib.Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                fpath = tmp.name
            try:
                pages, chunks = load_document(fpath)
                st.session_state.update(
                    doc_loaded=True, doc_name=uploaded.name,
                    doc_pages=pages, doc_chunks=chunks, messages=[]
                )
                st.success(f"✅ {uploaded.name} loaded")
            except Exception as e:
                st.error(f"❌ {e}")
            finally:
                os.unlink(fpath)

    if st.session_state.doc_loaded:
        st.markdown(f"""
        **📄 {st.session_state.doc_name}**  
        Pages: `{st.session_state.doc_pages}` · Chunks: `{st.session_state.doc_chunks}`
        """)

    st.divider()
    st.markdown("### ⚙️ Options")
    force_web = st.toggle(
        "🌐 Force Web Search",
        value=st.session_state.force_web,
        disabled=not WEB_SEARCH_ENABLED,
        help="Bypass PDF and query Tavily directly",
    )
    st.session_state.force_web = force_web
    if not WEB_SEARCH_ENABLED:
        st.caption("Set `TAVILY_API_KEY` in `.env` to enable.")

    if st.button("🗑️ Clear Chat"):
        clear_history()
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown(f"""
    <div style='font-size:.72rem;color:#6b7280'>
    <b>Queries:</b> {st.session_state.queries}<br>
    <b>Pipeline:</b> {"✅ Ready" if READY else "❌ Error"}
    </div>""", unsafe_allow_html=True)

# ── Chat area ─────────────────────────────────────────────────────────────────
if not st.session_state.doc_loaded:
    st.info("👈 Upload a PDF, DOCX, or TXT file from the sidebar to start chatting.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "source_label" in msg:
            label, cls = BADGE.get(msg["source_label"], ("", "bn"))
            st.markdown(f'<span class="badge {cls}">{label}</span>', unsafe_allow_html=True)
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("steps"):
            with st.expander("🔍 Pipeline trace", expanded=False):
                for step in msg["steps"]:
                    st.markdown(f'<div class="trace">{step}</div>', unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
if question := st.chat_input(
    "Ask anything about your document…",
    disabled=not st.session_state.doc_loaded,
):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Running 7-layer pipeline…"):
            t0     = time.time()
            result = run_pipeline(question, force_web=st.session_state.force_web)
            elapsed = time.time() - t0

        answer       = result["answer"]
        steps        = result["steps"]
        retries      = result["retries"]
        source_label = result["source_label"]
        label, cls   = BADGE.get(source_label, ("", "bn"))

        st.markdown(f'<span class="badge {cls}">{label}</span>', unsafe_allow_html=True)
        st.markdown(answer)
        st.caption(f"⏱ {elapsed:.1f}s · {retries} rewrite(s)")

        with st.expander("🔍 Pipeline trace", expanded=False):
            for step in steps:
                st.markdown(f'<div class="trace">{step}</div>', unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant", "content": answer,
        "source_label": source_label, "steps": steps,
    })
    st.session_state.queries += 1
