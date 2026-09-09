"""
octen_rag.py
─────────────────────────────────────────────────────────────────
Real embedding-based RAG using the Octen embedding model
(HuggingFace transformers, via sentence-transformers) and LangChain's
FAISS vector store. Replaces the stdlib TF-IDF retriever previously
used in mcp_server.py's rag_search tool.

Verified against installed packages:
  langchain==1.3.11
  langchain-huggingface (HuggingFaceEmbeddings wraps SentenceTransformer
    directly: self._client = SentenceTransformer(model_name, **model_kwargs))
  langchain_community.vectorstores.FAISS (still functional despite the
    deprecation notice pushing toward standalone packages — there is no
    trustworthy standalone FAISS integration package as of this writing;
    a "langchain-faiss" package exists on PyPI but ships an EMPTY module
    with no public API. Don't use it.)

Octen models (confirmed real, HuggingFace Hub):
  Octen/Octen-Embedding-0.6B  — 1024-dim, fine-tuned from Qwen3-Embedding-0.6B
  Octen/Octen-Embedding-4B    — 2560-dim, fine-tuned from Qwen3-Embedding-4B
  Octen/Octen-Embedding-8B    — 4096-dim, fine-tuned from Qwen3-Embedding-8B
  All use last-token pooling + L2 normalization. #1 on RTEB leaderboard
  for open embedding models as of the model card's last update.

This module replaces SKILL-FILE-DRIVEN retrieval (the Pydantic AI agent
calling rag_search as an MCP tool based on instructions in
skills/rag_retrieval.md) with a CODE-DRIVEN retrieval pipeline: loader →
splitter → embed → FAISS index → similarity search. The keyword-trigger
mechanism (check_keyword_triggers) and the A2A delegation to the RAG
Agent are UNCHANGED — only what happens inside rag_search() changes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CORPUS_PATH = Path(__file__).parent / "skills" / "rag_corpus.json"
INDEX_DIR   = Path(__file__).parent / ".octen_faiss_index"

# Octen model -> embedding dimension (informational; FAISS infers this
# automatically from the first embed_query() call, but useful for sizing
# memory / sanity-checking config).
OCTEN_DIMENSIONS = {
    "Octen/Octen-Embedding-0.6B": 1024,
    "Octen/Octen-Embedding-4B":   2560,
    "Octen/Octen-Embedding-8B":   4096,
}

OCTEN_MODEL_ID = os.getenv("OCTEN_MODEL_ID", "Octen/Octen-Embedding-0.6B")


# ════════════════════════════════════════════════════════════════
#  Embeddings — Octen via langchain_huggingface
# ════════════════════════════════════════════════════════════════

def build_embeddings(
    model_id: str = OCTEN_MODEL_ID,
    device: str = "cpu",
):
    """
    Build a LangChain Embeddings object backed by an Octen model.

    HuggingFaceEmbeddings wraps sentence_transformers.SentenceTransformer
    directly:  self._client = SentenceTransformer(model_id, **model_kwargs)
    Octen models support that constructor pattern natively per their
    HF model cards, so no custom pooling code is needed here — Octen
    ships a configured SentenceTransformer wrapper that already does
    last-token pooling + normalization internally.

    Args:
        model_id: e.g. "Octen/Octen-Embedding-0.6B" (1024-dim, fastest)
                       "Octen/Octen-Embedding-4B"   (2560-dim)
                       "Octen/Octen-Embedding-8B"   (4096-dim, best quality)
        device:   "cpu", "cuda", "cuda:0", etc.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={
            "device": device,
            "trust_remote_code": True,   # Octen ships custom modeling code
        },
        encode_kwargs={
            "normalize_embeddings": True,   # match the model card's usage
        },
    )


# ════════════════════════════════════════════════════════════════
#  Corpus loading + chunking
# ════════════════════════════════════════════════════════════════

def load_corpus_documents(
    corpus_path: Path = CORPUS_PATH,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[Document]:
    """
    Load skills/rag_corpus.json ({text, source} objects) and split into
    LangChain Document chunks. Same corpus file format as the previous
    TF-IDF implementation — no migration needed.
    """
    if not corpus_path.exists():
        return []

    raw = json.loads(corpus_path.read_text(encoding="utf-8"))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    docs: list[Document] = []
    for entry in raw:
        text   = entry.get("text", "")
        source = entry.get("source", "unknown")
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata={"source": source}))
    return docs


# ════════════════════════════════════════════════════════════════
#  Vector store — built once, cached on disk, reused across calls
# ════════════════════════════════════════════════════════════════

_vectorstore = None  # module-level cache; reset by rebuild_index()


def get_vectorstore(
    embeddings=None,
    force_rebuild: bool = False,
):
    """
    Return the FAISS vector store, building it from rag_corpus.json on
    first call (or loading a cached index from disk if present).

    Note: FAISS.save_local / load_local pickle metadata — only load
    indexes you built yourself, never an index from an untrusted source.
    """
    global _vectorstore
    from langchain_community.vectorstores import FAISS

    if _vectorstore is not None and not force_rebuild:
        return _vectorstore

    embeddings = embeddings or build_embeddings()

    if INDEX_DIR.exists() and not force_rebuild:
        _vectorstore = FAISS.load_local(
            str(INDEX_DIR), embeddings,
            allow_dangerous_deserialization=True,   # safe: we wrote this index ourselves
        )
        return _vectorstore

    docs = load_corpus_documents()
    if not docs:
        # No corpus yet — return an empty-but-valid store so callers
        # get {results: [], corpus_size: 0} instead of crashing.
        docs = [Document(page_content="", metadata={"source": "empty"})]

    _vectorstore = FAISS.from_documents(docs, embeddings)
    INDEX_DIR.mkdir(exist_ok=True)
    _vectorstore.save_local(str(INDEX_DIR))
    return _vectorstore


def rebuild_index(embeddings=None) -> None:
    """Force a full re-embed + re-index of rag_corpus.json. Call this
    after editing the corpus file."""
    import shutil
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    get_vectorstore(embeddings=embeddings, force_rebuild=True)


# ════════════════════════════════════════════════════════════════
#  Query — drop-in replacement for the old TF-IDF rag_search()
# ════════════════════════════════════════════════════════════════

def query(text: str, top_k: int = 5) -> dict:
    """
    Embedding-based similarity search over the Octen-indexed corpus.

    Returns the SAME shape as the old TF-IDF rag_search tool, so
    mcp_server.py's rag_search just delegates to this function:
        {"results": [{"text", "source", "score"}, ...], "corpus_size": N}

    FAISS similarity_search_with_score returns L2 distance by default
    (lower = more similar); this converts to a 0-1 "higher = better"
    score via 1 / (1 + distance) so the API contract matches the old
    cosine-similarity tool exactly.
    """
    store = get_vectorstore()
    raw = store.similarity_search_with_score(text, k=top_k)

    results = []
    for doc, distance in raw:
        if not doc.page_content:
            continue
        results.append({
            "text":   doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score":  round(1.0 / (1.0 + float(distance)), 4),
        })

    corpus_size = store.index.ntotal if hasattr(store, "index") else len(results)
    return {"results": results, "corpus_size": corpus_size}


if __name__ == "__main__":
    print(f"Building index from {CORPUS_PATH} using {OCTEN_MODEL_ID} ...")
    rebuild_index()
    print("Done. Try: python -c \"from octen_rag import query; print(query('test'))\"")
