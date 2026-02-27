"""
RAG Retriever — The main interface for retrieval-augmented generation.

Flow: User query → Embed → FAISS search → Format context → Return
"""
import logging
from dataclasses import dataclass
from typing import Optional

from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore, SearchResult

logger = logging.getLogger("rag.retriever")


@dataclass
class RetrievedContext:
    """A piece of retrieved context with relevance info."""
    text: str
    source: str
    relevance_score: float  # 0-1 normalized (higher = more relevant)
    metadata: dict


class RAGRetriever:
    """
    Retrieval-Augmented Generation retriever.

    Embeds queries, searches the FAISS index, and formats
    retrieved documents into LLM-ready context prompts.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        top_k: int = 5,
        relevance_threshold: float = 0.0,
    ):
        """
        Args:
            embedding_model: For embedding queries.
            vector_store: FAISS index to search.
            top_k: Default number of results.
            relevance_threshold: Min relevance score (0-1) to include.
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[RetrievedContext]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The user's query text.
            top_k: Override default number of results.

        Returns:
            List of RetrievedContext sorted by relevance (best first).
        """
        if self.vector_store.count == 0:
            logger.info("Vector store is empty — no results.")
            return []

        k = top_k or self.top_k

        # Embed the query
        query_embedding = self.embedding_model.embed_text(query)

        # Search FAISS index
        results = self.vector_store.search(query_embedding, top_k=k)

        if not results:
            return []

        # Normalize scores to 0-1 range (L2 distance → relevance)
        max_dist = max(r.score for r in results) if results else 1.0
        max_dist = max(max_dist, 0.001)  # Avoid division by zero

        contexts = []
        for result in results:
            relevance = 1.0 - (result.score / (max_dist * 1.5))  # Scale
            relevance = max(0.0, min(1.0, relevance))  # Clamp to [0, 1]

            if relevance < self.relevance_threshold:
                continue

            contexts.append(RetrievedContext(
                text=result.text,
                source=result.metadata.get("filename", result.doc_id),
                relevance_score=round(relevance, 3),
                metadata=result.metadata,
            ))

        logger.info(f"Retrieved {len(contexts)} contexts for query: '{query[:50]}...'")
        return contexts

    def build_context_prompt(
        self,
        contexts: list[RetrievedContext],
        max_chars: int = 3000,
    ) -> str:
        """
        Format retrieved contexts into a prompt section for the LLM.

        Args:
            contexts: Retrieved documents.
            max_chars: Maximum total characters for the context block.

        Returns:
            Formatted context string for LLM injection.
        """
        if not contexts:
            return ""

        lines = ["[Retrieved Knowledge Base Context]"]
        total_chars = 0

        for i, ctx in enumerate(contexts, 1):
            entry = f"\n--- Source: {ctx.source} (relevance: {ctx.relevance_score}) ---\n{ctx.text}"

            if total_chars + len(entry) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 100:
                    lines.append(entry[:remaining] + "...")
                break

            lines.append(entry)
            total_chars += len(entry)

        lines.append("\n[End of Retrieved Context]")
        return "\n".join(lines)

    def retrieve_and_format(self, query: str, top_k: Optional[int] = None, max_chars: int = 3000) -> str:
        """
        Convenience: retrieve + format in one call.

        Args:
            query: The user's query.
            top_k: Number of results.
            max_chars: Max context length.

        Returns:
            Formatted context string (empty string if no results).
        """
        contexts = self.retrieve(query, top_k)
        return self.build_context_prompt(contexts, max_chars)
