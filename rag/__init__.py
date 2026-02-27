"""
RAG Pipeline — Retrieval-Augmented Generation module.

Provides:
  - EmbeddingModel: Gemini embedding API wrapper
  - VectorStore: FAISS index management
  - DocumentIngester: File chunking & indexing
  - RAGRetriever: Query → embed → search → context
"""
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from rag.ingestion import DocumentIngester
from rag.retriever import RAGRetriever

__all__ = ["EmbeddingModel", "VectorStore", "DocumentIngester", "RAGRetriever"]
