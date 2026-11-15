"""
LLM Client — Local inference using llama3.2:3b via Ollama.

This module provides a unified interface for text generation across all layers.
Uses Ollama for complete offline capability.
"""
import os

from backend.core.ollama_llm_client import (
    generate_text as ollama_generate_text,
    check_ollama_available,
    DEFAULT_MODEL,
)

def generate_text(prompt, system_instruction=None, model_name=DEFAULT_MODEL):
    """
    Generate text using local Ollama llama3.2:3b model.
    
    This function maintains backward compatibility with the original Gemini interface
    while using local Ollama models for inference.
    
    Args:
        prompt: The user prompt/message.
        system_instruction: Optional system instruction for behavior control.
        model_name: Model name (default: llama3.2:3b).
    
    Returns:
        Generated text string.
    """
    return ollama_generate_text(
        prompt=prompt,
        system_instruction=system_instruction,
        model_name=model_name,
        # Default tetap 0.7 / 8000. Benchmark mengatur LACF_TEMPERATURE=0 agar reproducible.
        temperature=float(os.getenv("LACF_TEMPERATURE", "0.7")),
        max_output_tokens=int(os.getenv("LACF_MAX_TOKENS", "8000")),
    )