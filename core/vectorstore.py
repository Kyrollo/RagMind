"""
core/vectorstore.py
Build and manage a Chroma vector store with MMR retrieval.
Supports PDF, DOCX, and TXT documents.
"""
import pathlib
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.llms import embeddings
from config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_DOCS

_vectorstore = None
_retriever   = None
pdf_status   = {"loaded": False, "name": "", "pages": 0, "chunks": 0}


def _load_documents(file_path: str):
    ext = pathlib.Path(file_path).suffix.lower()
    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(file_path).load()
    elif ext in (".docx", ".doc"):
        from langchain_community.document_loaders import Docx2txtLoader
        return Docx2txtLoader(file_path).load()
    elif ext == ".txt":
        from langchain_community.document_loaders import TextLoader
        return TextLoader(file_path, encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported file type: {ext}  (supported: PDF, DOCX, TXT)")


def build_vectorstore_from_file(file_path: str) -> tuple[int, int]:
    """
    Load a document, chunk it, embed it, and store in Chroma with MMR retrieval.
    Returns (num_pages, num_chunks).
    """
    global _vectorstore, _retriever, pdf_status

    pages = _load_documents(file_path)
    print(f"📄 {len(pages)} page(s)/section(s) loaded from {pathlib.Path(file_path).name}")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)
    print(f"✂️  {len(chunks)} chunks created")

    # Drop old collection if exists
    if _vectorstore is not None:
        try:
            _vectorstore.delete_collection()
        except Exception:
            pass

    _vectorstore = Chroma.from_documents(
        documents=chunks,
        collection_name="ragmind-store",
        embedding=embeddings,
    )
    _retriever = _vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K_DOCS, "fetch_k": TOP_K_DOCS * 3},
    )

    pdf_status = {
        "loaded": True,
        "name":   pathlib.Path(file_path).name,
        "pages":  len(pages),
        "chunks": len(chunks),
    }
    print(f"Vector store ready — MMR top-{TOP_K_DOCS}")
    return len(pages), len(chunks)


def get_retriever():
    """Return the active MMR retriever, or None if no document is loaded."""
    return _retriever


def is_loaded() -> bool:
    return pdf_status["loaded"]
