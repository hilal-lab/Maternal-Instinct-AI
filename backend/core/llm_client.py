"""
LLM Client — Local inference using llama3.1:8b via Ollama.

This module provides a unified interface for text generation across all layers.
Uses Ollama for complete offline capability.
"""
from backend.core.ollama_llm_client import (
    generate_text as ollama_generate_text,
    check_ollama_available,
    DEFAULT_MODEL,
)

def generate_text(prompt, system_instruction=None, model_name=DEFAULT_MODEL):
    """
    Generate text using local Ollama llama3.1:8b model.
    
    This function maintains backward compatibility with the original Gemini interface
    while using local Ollama models for inference.
    
    Args:
        prompt: The user prompt/message.
        system_instruction: Optional system instruction for behavior control.
        model_name: Model name (default: llama3.1:8b).
    
    Returns:
        Generated text string.
    """
    return ollama_generate_text(
        prompt=prompt,
        system_instruction=system_instruction,
        model_name=model_name,
        temperature=0.7,
        max_output_tokens=8000,
    )