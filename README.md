<div align="center">

<img src="self_rag_pipeline.jpg" alt="Self-RAG + CRAG 7-Layer Pipeline" width="800"/>

# 🧠 RagMind — Self-RAG + CRAG PDF Assistant

**A fully local, 7-layer Adaptive RAG pipeline with automatic web fallback,
hallucination detection, and multi-turn memory — powered by Ollama + LangGraph.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-7c3aed)](https://github.com/langchain-ai/langgraph)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black)](https://ollama.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[**Demo Video**](#-demo) · [**Architecture**](#-architecture) · [**Quick Start**](#-quick-start) · [**Project Structure**](#-project-structure) · [**Configuration**](#-configuration)

</div>

---

## 🎬 Demo

> 📺 **Watch the full pipeline in action:**

https://github.com/user-attachments/assets/ccbe437e-dfcb-44e1-ae9b-8be57c174486

*Upload any PDF / DOCX / TXT → ask questions in natural language → watch all 7 layers process your query in real time, with source badges and a full pipeline trace.*

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔁 **Adaptive Retrieval** | MMR-based top-K diverse chunk retrieval from Chroma |
| 🎯 **Document Grading** | Per-chunk binary relevance filter (relevant / irrelevant) |
| 🌐 **Web Fallback** | Auto Tavily search when the document has no relevant chunks |
| 🛡️ **Hallucination Guard** | LLM-graded grounding check with automatic regeneration |
| ✅ **Quality Gate** | Answer quality assessment + automatic query rewrite loop |
| 💬 **Chat Memory** | Multi-turn conversation history with context condensation |
| 🏷️ **Source Attribution** | PDF-only / Web-only / Hybrid badges on every answer |
| ♾️ **Loop Protection** | `MAX_RETRIES` cap prevents infinite regeneration |
| 📂 **Multi-Format** | PDF, DOCX, DOC, TXT documents supported |
| 🔧 **Fully Local** | No OpenAI or cloud LLM required — runs entirely on Ollama |

---

## 🏗️ Architecture

The pipeline implements **Self-RAG** (self-reflective retrieval) combined with **CRAG** (corrective RAG) in a 7-layer LangGraph state machine:

```
START
  │
  ▼
[Layer 1] Query Understanding + Rewriter
          Condenses follow-ups with chat history → standalone query
  │
  ▼
[Layer 2] Document Retrieval  (MMR · Chroma)
          Fetches top-K diverse chunks via intfloat/e5-base-v2 embeddings
  │
  ▼
[Layer 3] Document Grading  (Relevance Filter)
          Per-chunk binary score: relevant / irrelevant
  │
  ├── no relevant chunks ──► [Layer 4b] Web Fallback  (Tavily API)
  │                                      │
  ▼                                      ▼
[Layer 4] RAG Generation ◄───────────────┘
          Answer synthesised from filtered chunks + conversation history
  │
  ▼
[Layer 5] Hallucination Grader
          Is every claim grounded in context?  No → regenerate (up to MAX_RETRIES)
  │ grounded ✓
  ▼
[Layer 6] Answer Quality Grader
          Does the answer resolve the question?  No → rewrite query
  │ useful ✓
  ▼
[Layer 7] Source Attribution + Chat Memory
          Badge: PDF-only / Web-only / Hybrid  — save turn to history
  │
  ▼
 END
```
<img src="self_rag_pipeline.jpg" alt="Self-RAG + CRAG 7-Layer Pipeline" width="800"/>

**Models:**

| Role | Default | Why |
|---|---|---|
| Grader | `qwen2.5:7b-instruct` | Fast, reliable JSON output |
| Generator | `qwen2.5:14b-instruct` | Stronger reasoning, 8 K context |
| Embeddings | `intfloat/e5-base-v2` | State-of-the-art MTEB retrieval |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com/download)** installed and running locally
- *(Optional)* A free [Tavily API key](https://tavily.com) for web-search fallback

### 1 · Clone

```bash
git clone https://github.com/Kyrollo/RagMind.git
cd ragmind
```

### 2 · Create virtual environment

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3 · Install dependencies

```bash
pip install -r requirements.txt
```

### 4 · Pull Ollama models

```bash
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5:14b-instruct
```

> 💡 **Low VRAM / CPU only?** Set `GENERATOR_MODEL=qwen2.5:7b-instruct` in `.env` to use one model for everything.

### 5 · Configure environment

```bash
cp .env.example .env
# Open .env and add your TAVILY_API_KEY (optional but recommended)
```

### 6 · Run

```bash
streamlit run app.py
```

Open **http://localhost:8501**, upload a document, and start chatting.

---

## 📁 Project Structure

```
ragmind/
│
├── app.py                      # Streamlit UI — chat interface, sidebar, badges
├── main.py                     # Public API — load_document(), run_pipeline(), clear_history()
├── config.py                   # All settings loaded from .env
│
├── agents/
│   ├── __init__.py
│   ├── graph.py                # Builds & compiles the LangGraph state machine
│   └── nodes.py                # One function per pipeline layer + edge decisions
│
├── core/
│   ├── __init__.py
│   ├── chains.py               # LLM chains: retrieval_grader, rag_chain, hallucination_grader,
│   │                           #             answer_grader, question_rewriter + format_docs / parse_binary_score
│   ├── llms.py                 # Initialises Ollama LLMs and HuggingFace embeddings
│   ├── memory.py               # Multi-turn conversation history (Layer 7)
│   ├── state.py                # LangGraph TypedDict GraphState
│   ├── vectorstore.py          # Chroma vector store + MMR retriever (PDF/DOCX/TXT loader)
│   └── web_search.py           # Tavily web-search fallback (Layer 4b)
│
├── utils/
│   ├── __init__.py
│   ├── helpers.py              # strip_rewriter_prefix, truncate, source_label_from_docs
│   └── logger.py               # Centralised logging configuration
│
├── self-rag.ipynb              # Original Kaggle notebook (reference / exploration)
├── self_rag_pipeline.jpg       # Architecture diagram (used in README)
│
├── requirements.txt
├── .env.example                # Environment variable template
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Configuration

All settings live in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `TAVILY_API_KEY` | *(empty)* | Tavily key for web fallback — [get one free](https://tavily.com) |
| `GRADER_MODEL` | `qwen2.5:7b-instruct` | Ollama model for all graders |
| `GENERATOR_MODEL` | `qwen2.5:14b-instruct` | Ollama model for answer generation |
| `EMBED_MODEL` | `intfloat/e5-base-v2` | HuggingFace embedding model |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server address |
| `CHUNK_SIZE` | `400` | Tokens per document chunk |
| `CHUNK_OVERLAP` | `50` | Token overlap between adjacent chunks |
| `TOP_K_DOCS` | `6` | Chunks retrieved per query (MMR) |
| `MIN_RELEVANT` | `1` | Min relevant chunks before triggering web fallback |
| `MAX_RETRIES` | `3` | Max query-rewrite attempts before giving up |
| `MAX_HISTORY_TURNS` | `10` | Conversation turns kept in memory |
| `LANGCHAIN_API_KEY` | *(empty)* | LangSmith API key for optional tracing |

---

## 🧪 Library Usage

Use `main.py` as a standalone Python API without the Streamlit UI:

```python
from main import load_document, run_pipeline, clear_history

# 1. Load a document
load_document("research_paper.pdf")

# 2. Ask a question
result = run_pipeline("What is the main contribution of this paper?")
print(result["answer"])
print(result["source_label"])  # "pdf" | "web" | "hybrid" | "none"
print(result["steps"])         # full 7-layer trace as a list of strings

# 3. Multi-turn — history is preserved automatically
result2 = run_pipeline("Can you expand on that?")

# 4. Force web search (ignore the PDF)
result3 = run_pipeline("Latest news about LangGraph", force_web=True)

# 5. Reset conversation
clear_history()
```

---

## 🛠️ Troubleshooting

| Symptom | Fix |
|---|---|
| `Connection refused` | Run `ollama serve` in a separate terminal |
| Slow first response | Models loading into RAM/VRAM — subsequent queries are faster |
| `CUDA out of memory` | Set both models to `qwen2.5:7b-instruct` in `.env` |
| Web search disabled | Add `TAVILY_API_KEY` to `.env` |
| Empty / unhelpful answers | Enable web fallback or verify the document is on-topic |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside your venv |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: your feature description"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

MIT © 2026 [Kerollos Mansour](https://github.com/Kyrollo)

---

<div align="center">
Built with ❤️ and Coffee using
<a href="https://github.com/langchain-ai/langgraph">LangGraph</a> ·
<a href="https://ollama.com">Ollama</a> ·
<a href="https://streamlit.io">Streamlit</a> ·
<a href="https://tavily.com">Tavily</a>
</div>
