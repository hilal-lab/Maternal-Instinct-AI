"""
RAG Service — Manages the RAG pipeline lifecycle within the backend.

Handles:
  - Initialization (load/create FAISS index, ingest knowledge base)
  - Document upload ingestion
  - Query retrieval for the chat pipeline
"""
import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("backend.services.rag_service")

# Module-level singleton instances
_embedding_model = None
_vector_store = None
_ingester = None
_retriever = None
_initialized = False

RAG_DIR = Path(__file__).parent.parent.parent / "rag"
KNOWLEDGE_BASE_DIR = RAG_DIR / "knowledge_base"
INDEX_DIR = Path(__file__).parent.parent / "data" / "faiss_index"


def _ensure_init():
    """Lazy-initialize RAG components."""
    global _embedding_model, _vector_store, _ingester, _retriever, _initialized

    if _initialized:
        return

    try:
        from rag.embeddings import EmbeddingModel
        from rag.vector_store import VectorStore
        from rag.ingestion import DocumentIngester
        from rag.retriever import RAGRetriever

        _embedding_model = EmbeddingModel()
        _vector_store = VectorStore(dimension=1024, index_path=str(INDEX_DIR))
        _ingester = DocumentIngester(_embedding_model, _vector_store)
        _retriever = RAGRetriever(_embedding_model, _vector_store, top_k=5)

        # Auto-ingest knowledge base on first run if index is empty
        if _vector_store.count == 0 and KNOWLEDGE_BASE_DIR.is_dir():
            logger.info("Empty index — ingesting knowledge base...")
            results = _ingester.ingest_directory(KNOWLEDGE_BASE_DIR)
            total = sum(results.values())
            logger.info(f"Ingested {total} chunks from {len(results)} files.")
            _vector_store.save()

        _initialized = True
        logger.info(f"RAG service initialized. Index has {_vector_store.count} vectors.")

    except Exception as e:
        logger.error(f"RAG initialization failed: {e}. RAG features will be disabled.")
        _initialized = True  # Don't retry on every request


def retrieve_context(query: str, top_k: int = 5) -> str:
    """
    Retrieve relevant context for a query.

    Returns:
        Formatted context string for LLM injection, or empty string if unavailable.
    """
    _ensure_init()
    if _retriever is None:
        return ""

    try:
        return _retriever.retrieve_and_format(query, top_k=top_k)
    except Exception as e:
        logger.error(f"RAG retrieval error: {e}")
        return ""


def retrieve_detailed(query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve with full metadata (for the UI).

    Returns:
        List of dicts with text, source, relevance_score.
    """
    _ensure_init()
    if _retriever is None:
        return []

    try:
        contexts = _retriever.retrieve(query, top_k=top_k)
        return [
            {
                "text": ctx.text,
                "source": ctx.source,
                "relevance_score": ctx.relevance_score,
            }
            for ctx in contexts
        ]
    except Exception as e:
        logger.error(f"RAG retrieval error: {e}")
        return []


async def ingest_uploaded_file(file_path: str, doc_id: str) -> int:
    """
    Ingest an uploaded document into the RAG index.

    Returns:
        Number of chunks created.
    """
    _ensure_init()
    if _ingester is None or _vector_store is None:
        return 0

    try:
        count = _ingester.ingest_file(file_path, doc_id=doc_id)
        _vector_store.save()
        return count
    except Exception as e:
        logger.error(f"Ingestion error for {doc_id}: {e}")
        return 0


async def remove_document(doc_id: str):
    """Remove a document's chunks from the RAG index."""
    _ensure_init()
    if _vector_store is None:
        return

    try:
        _vector_store.remove_by_doc_id(doc_id)
        _vector_store.save()
    except Exception as e:
        logger.error(f"Removal error for {doc_id}: {e}")


def get_index_stats() -> dict:
    """Get vector store statistics."""
    _ensure_init()
    if _vector_store is None:
        return {"total_vectors": 0, "status": "unavailable"}

    return {
        "total_vectors": _vector_store.count,
        "status": "active" if _vector_store.count > 0 else "empty",
    }


def list_materials() -> list[dict]:
    """List all unique documents/materials in the RAG index."""
    _ensure_init()
    if _vector_store is None:
        return []

    unique_docs = {}
    for chunk in _vector_store.chunks:
        doc_id = chunk.doc_id
        if doc_id and doc_id not in unique_docs:
            unique_docs[doc_id] = {
                "doc_id": doc_id,
                "topic": chunk.metadata.get("topic", chunk.metadata.get("filename", doc_id)),
                "filename": chunk.metadata.get("filename", doc_id),
                "chunk_count": 0,
            }
        if doc_id in unique_docs:
            unique_docs[doc_id]["chunk_count"] += 1

    return list(unique_docs.values())
