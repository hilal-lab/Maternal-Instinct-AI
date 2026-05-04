"""
Embedding Model — Local embeddings using bge-m3 via Ollama.

Produces 1024-dimensional vectors for document chunks and queries.
Fully offline, no external API dependencies.
"""
import logging
from rag.ollama_embedding_client import (
    generate_embedding,
    generate_embeddings_batch,
    get_embedding_dimension,
    check_ollama_embedding_available,
)

logger = logging.getLogger("rag.embeddings")


class EmbeddingModel:
    """
    Produces embedding vectors via local Ollama bge-m3 model.

    Supports:
      - Single text embedding
      - Batch embedding (list of texts)
      - Uses local Ollama for complete offline capability
    """

    MODEL_NAME = "bge-m3"
    DIMENSION = 1024  # bge-m3 output dimension

    def __init__(self):
        """Initialize the embedding model and check Ollama availability."""
        self.is_available = check_ollama_embedding_available()
        
        if not self.is_available:
            logger.warning(
                f"Ollama or {self.MODEL_NAME} not available. "
                "RAG features will be limited. Run: ollama pull bge-m3"
            )
        else:
            logger.info(f"EmbeddingModel initialized with {self.MODEL_NAME} ({self.DIMENSION}-dim).")

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text into a 1024-dim vector.

        Args:
            text: The text to embed.

        Returns:
            List of floats (1024-dimensional vector).
        """
        if not self.is_available:
            logger.error("Cannot generate embedding: Ollama not available")
            return [0.0] * self.DIMENSION
        
        try:
            return generate_embedding(text, model=self.MODEL_NAME)
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            return [0.0] * self.DIMENSION

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts in batch.

        Args:
            texts: List of texts to embed.

        Returns:
            List of 1024-dim vectors.
        """
        if not self.is_available:
            logger.error("Cannot generate embeddings: Ollama not available")
            return [[0.0] * self.DIMENSION for _ in texts]
        
        try:
            return generate_embeddings_batch(texts, model=self.MODEL_NAME, show_progress=True)
        except Exception as e:
            logger.error(f"Batch embedding error: {e}")
            return [[0.0] * self.DIMENSION for _ in texts]
