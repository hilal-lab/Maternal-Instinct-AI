"""
Vector Store — FAISS index management with metadata storage.

Handles:
  - Adding documents (embedding + metadata)
  - Similarity search (top-k nearest neighbors)
  - Disk persistence (save/load FAISS index + metadata)
"""
import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("rag.vector_store")

# Lazy-load faiss
_faiss = None


def _get_faiss():
    global _faiss
    if _faiss is None:
        try:
            import faiss
            _faiss = faiss
        except ImportError:
            raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")
    return _faiss


@dataclass
class DocumentChunk:
    """A single chunk of a document stored in the vector store."""
    text: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""
    chunk_index: int = 0


@dataclass
class SearchResult:
    """Result from a similarity search."""
    text: str
    metadata: dict
    score: float  # L2 distance (lower = more similar)
    doc_id: str = ""
    chunk_index: int = 0


class VectorStore:
    """
    FAISS-based vector store with metadata sidecar.

    Uses IndexFlatL2 (exact search) by default.
    Metadata is stored in a JSON sidecar file alongside the FAISS index.
    """

    def __init__(self, dimension: int = 768, index_path: Optional[str] = None):
        """
        Args:
            dimension: Embedding vector dimension (768 for Gemini).
            index_path: Directory to persist the index. If None, in-memory only.
        """
        faiss = _get_faiss()
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks: list[DocumentChunk] = []
        self.index_path = Path(index_path) if index_path else None

        # Try to load existing index
        if self.index_path and self._index_exists():
            self.load()

    @property
    def count(self) -> int:
        """Number of vectors in the index."""
        return self.index.ntotal

    def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadata_list: Optional[list[dict]] = None,
        doc_id: str = "",
    ):
        """
        Add documents to the vector store.

        Args:
            texts: Document chunk texts.
            embeddings: Corresponding embedding vectors.
            metadata_list: Optional metadata for each chunk.
            doc_id: Document identifier for grouping.
        """
        if len(texts) != len(embeddings):
            raise ValueError(f"texts ({len(texts)}) and embeddings ({len(embeddings)}) must have same length")

        if metadata_list is None:
            metadata_list = [{}] * len(texts)

        # Convert to numpy array for FAISS
        vectors = np.array(embeddings, dtype=np.float32)

        # Add to FAISS index
        self.index.add(vectors)

        # Store metadata
        base_idx = len(self.chunks)
        for i, (text, meta) in enumerate(zip(texts, metadata_list)):
            self.chunks.append(DocumentChunk(
                text=text,
                metadata=meta,
                doc_id=doc_id,
                chunk_index=base_idx + i,
            ))

        logger.info(f"Added {len(texts)} chunks (doc_id={doc_id}). Total: {self.count}")

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        """
        Search for the most similar documents.

        Args:
            query_embedding: Query vector (768-dim).
            top_k: Number of results to return.

        Returns:
            List of SearchResult sorted by similarity (best first).
        """
        if self.count == 0:
            return []

        query_vec = np.array([query_embedding], dtype=np.float32)
        k = min(top_k, self.count)

        distances, indices = self.index.search(query_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            results.append(SearchResult(
                text=chunk.text,
                metadata=chunk.metadata,
                score=float(dist),
                doc_id=chunk.doc_id,
                chunk_index=chunk.chunk_index,
            ))

        return results

    def remove_by_doc_id(self, doc_id: str):
        """
        Remove all chunks associated with a document.
        NOTE: FAISS IndexFlatL2 doesn't support removal, so we rebuild the index.
        """
        remaining = [(i, c) for i, c in enumerate(self.chunks) if c.doc_id != doc_id]

        if len(remaining) == len(self.chunks):
            return  # Nothing to remove

        faiss = _get_faiss()

        if not remaining:
            # All removed — reset
            self.index = faiss.IndexFlatL2(self.dimension)
            self.chunks = []
            logger.info(f"Removed all chunks for doc_id={doc_id}. Index empty.")
            return

        # Rebuild index without the removed chunks
        old_index = self.index
        new_index = faiss.IndexFlatL2(self.dimension)

        indices = [i for i, _ in remaining]
        vectors = np.array([
            old_index.reconstruct(int(i)) for i in indices
        ], dtype=np.float32)

        new_index.add(vectors)
        self.index = new_index
        self.chunks = [c for _, c in remaining]

        # Re-index chunk_index
        for i, chunk in enumerate(self.chunks):
            chunk.chunk_index = i

        logger.info(f"Removed doc_id={doc_id}. Remaining: {self.count} chunks.")

    def save(self):
        """Persist FAISS index and metadata to disk."""
        if not self.index_path:
            logger.warning("No index_path set — cannot save.")
            return

        self.index_path.mkdir(parents=True, exist_ok=True)
        faiss = _get_faiss()

        # Save FAISS index
        index_file = self.index_path / "index.faiss"
        faiss.write_index(self.index, str(index_file))

        # Save metadata sidecar
        meta_file = self.index_path / "metadata.json"
        meta_data = [
            {
                "text": c.text,
                "metadata": c.metadata,
                "doc_id": c.doc_id,
                "chunk_index": c.chunk_index,
            }
            for c in self.chunks
        ]
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved index ({self.count} vectors) to {self.index_path}")

    def load(self):
        """Load FAISS index and metadata from disk."""
        if not self.index_path:
            return

        faiss = _get_faiss()
        index_file = self.index_path / "index.faiss"
        meta_file = self.index_path / "metadata.json"

        if not index_file.exists():
            logger.info("No existing index found — starting empty.")
            return

        self.index = faiss.read_index(str(index_file))

        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
            self.chunks = [
                DocumentChunk(
                    text=m["text"],
                    metadata=m.get("metadata", {}),
                    doc_id=m.get("doc_id", ""),
                    chunk_index=m.get("chunk_index", i),
                )
                for i, m in enumerate(meta_data)
            ]

        logger.info(f"Loaded index ({self.count} vectors) from {self.index_path}")

    def _index_exists(self) -> bool:
        """Check if a persisted index exists."""
        if not self.index_path:
            return False
        return (self.index_path / "index.faiss").exists()
