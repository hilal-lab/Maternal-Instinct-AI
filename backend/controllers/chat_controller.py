"""
Chat Controller — Handles chat request/response formatting.
"""
from backend.models.schemas import ChatRequest, ChatResponse
from backend.services import chat_service


async def send_message(request: ChatRequest, mode: str = "conversation") -> ChatResponse:
    """Process a chat message through the 4-layer pipeline."""
    return await chat_service.run_pipeline(
        request.message,
        mode=mode,
        chat_id=request.chat_id,
    )


async def get_history(limit: int = 50, mode: str = None, chat_id: str = None) -> list[dict]:
    """Get recent chat history, optionally filtered by mode and chat_id."""
    return await chat_service.get_history(limit, mode=mode, chat_id=chat_id)


async def clear_history(mode: str = None, chat_id: str = None) -> dict:
    """Clear chat history, optionally filtered by mode and chat_id."""
    await chat_service.clear_history(mode=mode, chat_id=chat_id)
    return {"message": "Chat history cleared."}
