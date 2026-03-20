"""
core/llms.py
Initialise Ollama LLMs and HuggingFace embeddings from config.
"""
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from config import (
    GRADER_MODEL, GENERATOR_MODEL, EMBED_MODEL,
    OLLAMA_BASE_URL,
)

print(f"Initialising LLMs…  grader={GRADER_MODEL}  generator={GENERATOR_MODEL}")

llm_grader = OllamaLLM(
    model=GRADER_MODEL,
    temperature=0,
    base_url=OLLAMA_BASE_URL,
    num_ctx=4096,
    format="json",          # enforces valid JSON output for all graders
)

llm_generator = OllamaLLM(
    model=GENERATOR_MODEL,
    temperature=0.1,
    base_url=OLLAMA_BASE_URL,
    num_ctx=8192,           # large context for PDF + multi-turn history
)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True},   # cosine similarity
)

print(f"LLMs ready  |  embeddings={EMBED_MODEL}")
