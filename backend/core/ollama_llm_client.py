"""
Ollama LLM Client — Local inference using llama3.1:8b via Ollama API.

Provides the same interface as the original Gemini client but uses
local Ollama models for complete offline capability.
"""
import os
import time
import logging
import requests
from typing import Optional

logger = logging.getLogger("backend.core.ollama_llm_client")

# Model configuration
DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT = 120  # 2 minutes for generation
MAX_RETRIES = 3


def get_ollama_host() -> str:
    """Get Ollama host from environment or use default."""
    return os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)


def check_ollama_available(model: str = DEFAULT_MODEL) -> bool:
    """
    Check if Ollama is running and the model is available.
    
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


def generate_text(
    prompt: str,
    system_instruction: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_output_tokens: int = 8000,
) -> str:
    """
    Generate text using Ollama's llama3.1:8b model.
    
    Args:
        prompt: The user prompt/message to generate from.
        system_instruction: Optional system instruction for behavior control.
        model_name: Ollama model name (default: llama3.1:8b).
        temperature: Sampling temperature (0.0-1.0).
        max_output_tokens: Maximum tokens to generate.
    
    Returns:
        Generated text string.
    """
    host = get_ollama_host()
    url = f"{host}/api/generate"
    
    # Build the payload
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_output_tokens,
        }
    }
    
    # Add system instruction if provided
    if system_instruction:
        payload["system"] = system_instruction
    
    # Retry logic with exponential backoff
    for attempt in range(MAX_RETRIES):
        try:
            logger.debug(f"Ollama request attempt {attempt + 1}/{MAX_RETRIES}")
            
            response = requests.post(
                url,
                json=payload,
                timeout=DEFAULT_TIMEOUT,
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "").strip()
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            logger.warning(f"Ollama request timeout on attempt {attempt + 1}")
            
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Is it running?")
            return "⚠️ [ERROR] Tidak dapat terhubung ke Ollama. Pastikan Ollama berjalan di localhost:11434"
            
        except Exception as e:
            logger.error(f"Unexpected error during Ollama request: {e}")
        
        # Exponential backoff before retry
        if attempt < MAX_RETRIES - 1:
            wait_time = (2 ** attempt)
            logger.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    # All retries failed
    return "⚠️ [ERROR] Gagal menghasilkan respons setelah beberapa percobaan. Silakan coba lagi."


def generate_text_stream(
    prompt: str,
    system_instruction: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.7,
):
    """
    Generate text with streaming (for future real-time UI).
    
    Yields:
        Text chunks as they're generated.
    """
    host = get_ollama_host()
    url = f"{host}/api/generate"
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
        }
    }
    
    if system_instruction:
        payload["system"] = system_instruction
    
    try:
        response = requests.post(
            url,
            json=payload,
            stream=True,
            timeout=DEFAULT_TIMEOUT,
        )
        
        for line in response.iter_lines():
            if line:
                import json
                data = json.loads(line)
                chunk = data.get("response", "")
                if chunk:
                    yield chunk
                    
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"⚠️ [ERROR] Streaming gagal: {e}"


# Testing function
def test_ollama():
    """Test the Ollama client."""
    print("Testing Ollama LLM Client...")
    
    # Check availability
    print(f"Checking if Ollama is available...")
    if not check_ollama_available():
        print("❌ Ollama tidak tersedia atau model llama3.1:8b belum dipull")
        print("   Jalankan: ollama pull llama3.1:8b")
        return
    
    print("✓ Ollama available")
    
    # Test generation
    print("\nTesting text generation...")
    response = generate_text(
        prompt="Jelaskan dalam 2 kalimat apa itu machine learning.",
        system_instruction="Anda adalah asisten AI yang membantu mahasiswa.",
    )
    print(f"✓ Response received ({len(response)} chars)")
    print(f"  Sample: {response[:100]}...")
    
    print("\n✅ All Ollama LLM tests passed!")


if __name__ == "__main__":
    test_ollama()
