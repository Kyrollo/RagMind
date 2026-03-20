"""
config.py
Central configuration — all values are overridable via .env.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Models ────────────────────────────────────────────────────────────────
GRADER_MODEL    = os.getenv("GRADER_MODEL",    "qwen2.5:7b-instruct")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "qwen2.5:14b-instruct")
EMBED_MODEL     = os.getenv("EMBED_MODEL",     "intfloat/e5-base-v2")

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# ── API Keys ──────────────────────────────────────────────────────────────────
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY",    "")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")

WEB_SEARCH_ENABLED = bool(TAVILY_API_KEY)

# ── LangSmith (optional tracing) ──────────────────────────────────────────────
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"]   = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"]    = LANGCHAIN_API_KEY

# ── Retrieval ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "400"))   # tokens per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))    # overlap between chunks
TOP_K_DOCS    = int(os.getenv("TOP_K_DOCS",    "6"))     # chunks retrieved per query (MMR)
MIN_RELEVANT  = int(os.getenv("MIN_RELEVANT",  "1"))     # min relevant chunks before web fallback

# ── Pipeline ──────────────────────────────────────────────────────────────────
MAX_RETRIES       = int(os.getenv("MAX_RETRIES",       "3"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))
