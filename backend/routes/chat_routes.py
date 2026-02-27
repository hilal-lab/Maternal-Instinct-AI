"""
Chat Routes — API endpoint definitions.
"""
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.models.schemas import ChatRequest, ChatResponse
from backend.controllers import chat_controller
from backend.services import chat_service

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message through the 4-layer pipeline."""
    return await chat_controller.send_message(request)


@router.get("/chat/history")
async def history(limit: int = 50):
    """Get recent chat messages."""
    return await chat_controller.get_history(limit)


@router.delete("/chat/history")
async def clear():
    """Clear all chat history."""
    return await chat_controller.clear_history()


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """Real-time chat via WebSocket."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            user_message = msg.get("message", "")

            await websocket.send_json({"type": "status", "layer": 1, "message": "Analyzing intent & emotion..."})

            result = await chat_service.run_pipeline(user_message)

            await websocket.send_json({"type": "status", "layer": 2, "message": f"Agent: {result.layers.layer2.agent_used}"})
            await websocket.send_json({"type": "status", "layer": 3, "message": f"Ethics: {result.layers.layer3.status}"})
            await websocket.send_json({"type": "status", "layer": 4, "message": f"Guardrail: {'REWRITE' if result.layers.layer4.is_rewritten else 'PASS'}"})

            await websocket.send_json({"type": "response", "data": result.model_dump()})
    except WebSocketDisconnect:
        pass
