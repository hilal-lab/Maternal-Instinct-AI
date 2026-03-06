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
    """
    Real-time chat via WebSocket with per-layer progress events.

    Incoming message types
    ----------------------
    { "type": "message", "message": "..." }   — or legacy { "message": "..." }
        Start a new pipeline run.

    { "type": "tool_confirm_response", "confirm_id": "...", "approved": true/false }
        Resume a pipeline that is suspended waiting for destructive tool confirmation.

    Outgoing event types
    --------------------
    layer_start | layer_progress | layer_done  — pipeline progress
    tool_confirm                               — destructive tool needs confirmation
    response                                   — final assembled ChatResponse
    error                                      — something went wrong
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            # Determine message type — support both new typed format and
            # legacy { "message": "..." } from older frontend builds.
            msg_type = msg.get("type", "message")

            # ── Forward pipeline events live to the client ─────────────────
            async def on_event(event: dict):
                await websocket.send_json(event)

            # ── Branch on message type ─────────────────────────────────────

            if msg_type == "tool_confirm_response":
                # User approved or rejected a pending destructive tool call
                confirm_id = msg.get("confirm_id", "")
                approved = bool(msg.get("approved", False))

                result = await chat_service.resume_after_confirmation(
                    confirm_id=confirm_id,
                    approved=approved,
                    on_event=on_event,
                )

                if result is None:
                    # Unknown confirm_id — pending entry may have expired
                    await websocket.send_json({
                        "type": "error",
                        "message": "Konfirmasi tidak ditemukan atau sudah kedaluwarsa.",
                    })
                else:
                    await websocket.send_json({
                        "type": "response",
                        "data": result.model_dump(),
                    })

            else:
                # Default: new chat message (type == "message" or legacy)
                user_message = msg.get("message", "")
                if not user_message.strip():
                    continue

                result = await chat_service.run_pipeline(
                    user_message,
                    on_event=on_event,
                )

                if result is None:
                    # Pipeline suspended — tool_confirm event already sent.
                    # Don't send a final "response" here; the client will
                    # wait for the confirmation dialog and respond back.
                    pass
                else:
                    await websocket.send_json({
                        "type": "response",
                        "data": result.model_dump(),
                    })

    except WebSocketDisconnect:
        pass
