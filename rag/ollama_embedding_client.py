"""
Ollama Embedding Client — Local embedding generation using bge-m3 via Ollama API.

Produces 1024-dimensional embeddings for semantic search and RAG.
"""
import os
import time
import logging
import requests
from typing import List, Optional

logger = logging.getLogger("rag.ollama_embedding_client")

# Model configuration
DEFAULT_MODEL = "bge-m3"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
EMBEDDING_DIMENSION = 1024
DEFAULT_TIMEOUT = 60
MAX_RETRIES = 3


def get_ollama_host() -> str:
    """Get Ollama host from environment or use default."""
    return os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)


def check_ollama_embedding_available(model: str = DEFAULT_MODEL) -> bool:
    """
    Check if Ollama is running and the embedding model is available.
    
    Returns:
        True if Ollama is accessible and model is available, False otherwise.
    """
    try:
        host = get_ollama_host()
        response = requests.get(f"{host}/api/tags", timeout=5)
        
        if response.status_code != 200:
            return False
        
        # Check if our model is in the list
        data = response.json()
        models = [m.get("name", "") for m in data.get("models", [])]
        
        # Check for exact match or with :latest tag
        model_available = any(
            model in m or f"{model}:latest" in m 
            for m in models
        )
        
        return model_available
        
    except Exception as e:
        logger.warning(f"Ollama availability check failed: {e}")
        return False


def generate_embedding(text: str, model: str = DEFAULT_MODEL) -> List[float]:
    """
    Generate a 1024-dimensional embedding for a single text.
    
    Args:
        text: The text to embed.
        model: Ollama embedding model name (default: bge-m3).
    
    Returns:
        List of 1024 floats representing the embedding vector.
    
    Raises:
        RuntimeError: If embedding generation fails after retries.
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for embedding")
        return [0.0] * EMBEDDING_DIMENSION
    
    host = get_ollama_host()
    url = f"{host}/api/embeddings"
    
    payload = {
        "model": model,
        "prompt": text.strip(),
    }
    
    # Retry logic with exponential backoff
    for attempt in range(MAX_RETRIES):
        try:
            logger.debug(f"Embedding request attempt {attempt + 1}/{MAX_RETRIES}")
            
            response = requests.post(
                url,
                json=payload,
                timeout=DEFAULT_TIMEOUT,
            )
            
            if response.status_code == 200:
                data = response.json()
                embedding = data.get("embedding", [])
                
                # Validate dimension
                if len(embedding) != EMBEDDING_DIMENSION:
                    logger.error(
                        f"Unexpected embedding dimension: {len(embedding)} "
                        f"(expected {EMBEDDING_DIMENSION})"
                    )
                    raise ValueError(f"Invalid embedding dimension: {len(embedding)}")
                
                return embedding
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            logger.warning(f"Embedding request timeout on attempt {attempt + 1}")
            
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Is it running?")
            raise RuntimeError(
                "Cannot connect to Ollama. Ensure Ollama is running at localhost:11434"
            )
            
        except ValueError:
            # Don't retry on dimension mismatch
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error during embedding request: {e}")
        
        # Exponential backoff before retry
        if attempt < MAX_RETRIES - 1:
            wait_time = (2 ** attempt)
            logger.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    # All retries failed
    raise RuntimeError(
        f"Failed to generate embedding after {MAX_RETRIES} attempts"
    )


def generate_embeddings_batch(
    texts: List[str],
    model: str = DEFAULT_MODEL,
    show_progress: bool = True,
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts.
    
    Args:
        texts: List of texts to embed.
        model: Ollama embedding model name.
        show_progress: Whether to log progress (for batch operations).
    
    Returns:
        List of embedding vectors (1024-dim each).
    """
    embeddings = []
    total = len(texts)
    
    for i, text in enumerate(texts):
        try:
            embedding = generate_embedding(text, model=model)
            embeddings.append(embedding)
            
            if show_progress and (i + 1) % 10 == 0:
                logger.info(f"Embedded {i + 1}/{total} texts")
                
        except Exception as e:
            logger.error(f"Failed to embed text at index {i}: {e}")
            # Use zero vector as fallback for failed embeddings
            embeddings.append([0.0] * EMBEDDING_DIMENSION)
    
    if show_progress:
        logger.info(f"Completed: {len(embeddings)}/{total} embeddings generated")
    
    return embeddings


def get_embedding_dimension() -> int:
    """Get the embedding dimension for bge-m3."""
    return EMBEDDING_DIMENSION


# Testing function
def test_embedding():
    """Test the Ollama embedding client."""
    print("Testing bge-m3 embedding generation...")
    
    # Check availability
    print(f"Checking if Ollama and bge-m3 are available...")
    if not check_ollama_embedding_available():
        print("❌ Ollama tidak tersedia atau model bge-m3 belum dipull")
        print("   Jalankan: ollama pull bge-m3")
        return
    
    print("✓ Ollama and bge-m3 available")
    
    # Test single embedding
    print("\nTesting single embedding...")
    text = "Machine learning adalah cabang dari artificial intelligence."
    embedding = generate_embedding(text)
    print(f"✓ Single embedding generated: {len(embedding)} dimensions")
    print(f"  Sample values: {embedding[:5]}")
    
    # Test batch embeddings
    print("\nTesting batch embeddings...")
    texts = [
        "Saya sedang belajar Python",
        "Manajemen waktu sangat penting",
        "Stres akademik bisa diatasi dengan teknik relaksasi",
    ]
    embeddings = generate_embeddings_batch(texts, show_progress=False)
    print(f"✓ Batch embeddings generated: {len(embeddings)} embeddings")
    
    # Test dimension consistency
    dimensions = [len(emb) for emb in embeddings]
    if all(d == EMBEDDING_DIMENSION for d in dimensions):
        print(f"✓ All embeddings have correct dimension: {EMBEDDING_DIMENSION}")
    else:
        print(f"❌ Dimension mismatch: {dimensions}")
        return
    
    print("\n✅ All embedding tests passed!")


if __name__ == "__main__":
    test_embedding()
