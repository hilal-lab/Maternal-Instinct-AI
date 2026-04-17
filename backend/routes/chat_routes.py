"""
Chat Routes — API endpoint definitions.
"""
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from backend.models.schemas import ChatRequest, ChatResponse, ChatMode
from backend.controllers import chat_controller
from backend.services import chat_service
from backend.services.learning_service import learning_service

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, mode: str = Query("conversation")):
    """Send a message through the 4-layer pipeline."""
    return await chat_controller.send_message(request, mode=mode)


@router.get("/chat/history")
async def history(limit: int = 50, mode: str = Query(None), chat_id: str = Query(None)):
    """Get recent chat messages, optionally filtered by mode and chat_id."""
    return await chat_controller.get_history(limit, mode=mode, chat_id=chat_id)


@router.delete("/chat/history")
async def clear(mode: str = Query(None), chat_id: str = Query(None)):
    """Clear chat history, optionally filtered by mode and chat_id."""
    return await chat_controller.clear_history(mode=mode, chat_id=chat_id)


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    Real-time chat via WebSocket with per-layer progress events.

    Supports both Conversation Mode and Learning Mode.

    Incoming message types
    ----------------------
    { "type": "message", "message": "...", "mode": "conversation|learning" }
        Start a new pipeline run.

    { "type": "tool_confirm_response", "confirm_id": "...", "approved": true/false }
        Resume a pipeline that is suspended waiting for destructive tool confirmation.

    { "type": "learning", "action": "start|continue|quiz", ... }
        Learning mode actions.

    Outgoing event types
    --------------------
    layer_start | layer_progress | layer_done  — pipeline progress
    tool_confirm                               — destructive tool needs confirmation
    response                                   — final assembled ChatResponse
    learning_start | learning_section | learning_quiz | learning_result — learning mode events
    error                                      — something went wrong
    """
    await websocket.accept()
    current_mode = "conversation"
    
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            msg_type = msg.get("type", "message")

            async def on_event(event: dict):
                await websocket.send_json(event)

            # ── Learning Mode Handler ────────────────────────────────────────
            if msg_type == "learning" or current_mode == "learning":
                learning_action = msg.get("action", "")
                
                if learning_action == "start":
                    topic = msg.get("topic", "")
                    mode = msg.get("learning_mode", "lesson")
                    level = msg.get("level", "pemula")
                    
                    result = await learning_service.start_learning(
                        topic=topic,
                        subtopic=msg.get("subtopic", ""),
                        level=level,
                        mode=mode
                    )
                    
                    await websocket.send_json({
                        "type": "learning_start",
                        "data": result
                    })
                    
                elif learning_action == "continue":
                    session_id = msg.get("session_id")
                    user_response = msg.get("response", "")
                    
                    result = await learning_service.continue_learning(
                        session_id=session_id,
                        user_response=user_response
                    )
                    
                    if result.get("done"):
                        await websocket.send_json({
                            "type": "learning_complete",
                            "data": result
                        })
                    else:
                        await websocket.send_json({
                            "type": "learning_section",
                            "data": result
                        })
                        
                elif learning_action == "submit_quiz":
                    session_id = msg.get("session_id")
                    answers = msg.get("answers", {})
                    
                    result = await learning_service.submit_quiz(
                        session_id=session_id,
                        answers=answers
                    )
                    
                    await websocket.send_json({
                        "type": "learning_result",
                        "data": result
                    })
                    
                elif learning_action == "switch_mode":
                    current_mode = msg.get("mode", "conversation")
                    await websocket.send_json({
                        "type": "mode_changed",
                        "mode": current_mode
                    })
                
                continue

            # ── Tool Confirmation Handler ───────────────────────────────────
            if msg_type == "tool_confirm_response":
                confirm_id = msg.get("confirm_id", "")
                approved = bool(msg.get("approved", False))

                result = await chat_service.resume_after_confirmation(
                    confirm_id=confirm_id,
                    approved=approved,
                    on_event=on_event,
                )

                if result is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Konfirmasi tidak ditemukan atau sudah kedaluwarsa.",
                    })
                else:
                    await websocket.send_json({
                        "type": "response",
                        "data": result.model_dump(),
                    })

            # ── Mode Switch Handler ─────────────────────────────────────────
            elif msg_type == "switch_mode":
                current_mode = msg.get("mode", "conversation")
                await websocket.send_json({
                    "type": "mode_changed",
                    "mode": current_mode
                })

            # ── Default: Chat Message ─────────────────────────────────────────
            else:
                user_message = msg.get("message", "")
                if not user_message.strip():
                    continue

                chat_id = msg.get("chat_id", "default")

                # Check if message implies mode change
                from backend.core.mode_router import detect_mode
                detected_mode, topic = detect_mode(
                    user_message, 
                    msg.get("force_mode")
                )
                current_mode = detected_mode.value
                
                # If learning mode detected, start learning session
                if current_mode == "learning" and topic:
                    result = await learning_service.start_learning(
                        topic=topic,
                        level="pemula",
                        mode="lesson"
                    )
                    await websocket.send_json({
                        "type": "learning_start",
                        "data": result,
                        "detected_topic": topic
                    })
                    continue

                result = await chat_service.run_pipeline(
                    user_message,
                    on_event=on_event,
                    chat_id=chat_id,
                )

                if result is None:
                    pass
                else:
                    await websocket.send_json({
                        "type": "response",
                        "data": result.model_dump(),
                    })

    except WebSocketDisconnect:
        pass
