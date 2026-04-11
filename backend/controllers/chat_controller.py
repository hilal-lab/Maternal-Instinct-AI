"""
Chat Controller — Handles chat request/response formatting.
"""
from backend.models.schemas import ChatRequest, ChatResponse
from backend.services import chat_service


async def send_message(request: ChatRequest, mode: str = "conversation") -> ChatResponse:
    """Process a chat message through the 4-layer pipeline."""
    return await chat_service.run_pipeline(request.message, mode=mode)


async def get_history(limit: int = 50, mode: str = None) -> list[dict]:
    """Get recent chat history, optionally filtered by mode."""
    return await chat_service.get_history(limit, mode=mode)


async def clear_history(mode: str = None) -> dict:
    """Clear chat history, optionally filtered by mode."""
    await chat_service.clear_history(mode=mode)
    return {"message": "Chat history cleared."}
