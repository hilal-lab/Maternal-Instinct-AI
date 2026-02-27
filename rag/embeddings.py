"""
Embedding Model — Wraps the Gemini Embedding API.

Uses `models/embedding-001` to produce 768-dimensional vectors
for document chunks and queries.
"""
import os
import logging
from typing import Optional

logger = logging.getLogger("rag.embeddings")

# Lazy-load google.genai to allow fallback mode without the package
_genai = None
_client = None


def _init_client():
    """Initialize the Gemini client lazily."""
    global _genai, _client
    if _client is not None:
        return _client

    try:
        from google import genai
        _genai = genai

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set — embedding model will use fallback mode.")
            return None

        _client = genai.Client(api_key=api_key)
        return _client
    except ImportError:
        logger.warning("google-genai not installed — embedding model will use fallback mode.")
        return None


class EmbeddingModel:
    """
    Produces embedding vectors via Gemini Embedding API.

    Supports:
      - Single text embedding
      - Batch embedding (list of texts)
      - Fallback mode (deterministic hash-based vectors) when API unavailable
    """

    MODEL_NAME = "models/embedding-001"
    DIMENSION = 768  # Gemini embedding-001 output dimension

    def __init__(self):
        self.client = _init_client()
        self.is_fallback = self.client is None
        if self.is_fallback:
            logger.info("EmbeddingModel running in FALLBACK mode (hash-based vectors).")
        else:
            logger.info(f"EmbeddingModel initialized with {self.MODEL_NAME}.")

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text into a 768-dim vector.

        Args:
            text: The text to embed.

        Returns:
            List of floats (768-dimensional vector).
        """
        if self.is_fallback:
            return self._fallback_embed(text)

        try:
            result = self.client.models.embed_content(
                model=self.MODEL_NAME,
                contents=text,
            )
            return result.embeddings[0].values
        except Exception as e:
            logger.error(f"Embedding API error: {e}. Using fallback.")
            return self._fallback_embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts in batch.

        Args:
            texts: List of texts to embed.

        Returns:
            List of 768-dim vectors.
        """
        if self.is_fallback:
            return [self._fallback_embed(t) for t in texts]

        try:
            results = []
            # Process in batches of 100 (API limit)
            for i in range(0, len(texts), 100):
                batch = texts[i:i + 100]
                result = self.client.models.embed_content(
                    model=self.MODEL_NAME,
                    contents=batch,
                )
                results.extend([e.values for e in result.embeddings])
            return results
        except Exception as e:
            logger.error(f"Batch embedding API error: {e}. Using fallback.")
            return [self._fallback_embed(t) for t in texts]

    def _fallback_embed(self, text: str) -> list[float]:
        """
        Deterministic fallback: hash-based pseudo-embedding.
        Not semantically meaningful, but consistent for testing.
        """
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        # Convert hex chars to floats in [-1, 1] range
        vector = []
        for i in range(0, min(len(h) * 12, self.DIMENSION * 2), 2):
            idx = i % len(h)
            val = (int(h[idx], 16) - 8) / 8.0  # [-1, 1]
            vector.append(val)
        # Pad to DIMENSION
        while len(vector) < self.DIMENSION:
            vector.append(0.0)
        return vector[:self.DIMENSION]
