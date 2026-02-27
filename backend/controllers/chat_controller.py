"""
Chat Controller — Handles chat request/response formatting.
"""
from backend.models.schemas import ChatRequest, ChatResponse
from backend.services import chat_service


async def send_message(request: ChatRequest) -> ChatResponse:
    """Process a chat message through the 4-layer pipeline."""
    return await chat_service.run_pipeline(request.message)


async def get_history(limit: int = 50) -> list[dict]:
    """Get recent chat history."""
    return await chat_service.get_history(limit)


async def clear_history() -> dict:
    """Clear all chat history."""
    await chat_service.clear_history()
    return {"message": "Chat history cleared."}
