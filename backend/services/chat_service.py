"""
Chat Service — Runs the 4-layer AI pipeline and manages chat history.
"""
from backend.models.database import get_db
from backend.models.schemas import (
    ChatResponse, LayersData,
    LayerOneData, LayerTwoData, LayerThreeData, LayerFourData,
)
from backend.core.layer_1_orchestrator import ExecutiveAgent
from backend.core.layer_2_specialists import SpecialistAgents
from backend.core.layer_3_ethics import PolicyAggregator
from backend.core.layer_4_guardrail import MaternalGuardrail
from backend.services import schedule_service, rag_service


async def run_pipeline(message: str) -> ChatResponse:
    """
    Full 4-layer architecture pipeline:
      Layer 1 → Layer 2 → Layer 3 → Layer 4
    """
    orchestrator = ExecutiveAgent()
    specialists = SpecialistAgents()
    ethics = PolicyAggregator()
    guardrail = MaternalGuardrail()

    # ── Layer 1: Orchestration & Emotion Detection ──
    intent, emotion, empathy_data = orchestrator.analyze_intent(message)
    intensity = empathy_data.get("intensity", 0.0)
    flags = empathy_data.get("flags", {})

    layer1 = LayerOneData(
        intent=intent, emotion=emotion, intensity=intensity, flags=flags
    )

    # --- Retrieve RAG Context ---
    # Fetch relevant knowledge base documents to augment the specialists' prompts
    rag_context = rag_service.retrieve_context(message, top_k=3)

    # ── Layer 2: Specialist Agent Execution ──
    schedule_list = await schedule_service.get_active_list()
    raw_response = ""
    agent_used = "general_chat"
    is_task_request = (intent == "TASK_OPS")

    # Always build schedule context so ALL agents can reference user's tasks
    schedule_context = specialists._format_schedule(schedule_list)

    if intent == "TASK_OPS":
        plan = specialists.run_planner(schedule_list, rag_context=rag_context)
        coach = specialists.run_coach(emotion, empathy_data, rag_context=rag_context,
                                      schedule_context=schedule_context)
        raw_response = f"{plan}\n\n**Pesan Coach:**\n{coach}"
        agent_used = "planner+coach"

    elif intent == "ACADEMIC_HELP":
        raw_response = specialists.run_tutor(message, rag_context=rag_context,
                                             schedule_context=schedule_context)
        agent_used = "tutor"

    elif intent in ("MOTIVATION_SUPPORT", "EMOTIONAL_DISTRESS"):
        coach = specialists.run_coach(emotion, empathy_data, rag_context=rag_context,
                                      schedule_context=schedule_context)
        general = specialists.run_general_chat(message, emotion, empathy_data,
                                               rag_context=rag_context,
                                               schedule_context=schedule_context)
        raw_response = f"{general}\n\n**Pesan Coach:**\n{coach}"
        agent_used = "general_chat+coach"

    else:
        raw_response = specialists.run_general_chat(message, emotion, empathy_data,
                                                     rag_context=rag_context,
                                                     schedule_context=schedule_context)

    layer2 = LayerTwoData(agent_used=agent_used, raw_response=raw_response)

    # ── Layer 3: Ethics & Deliberation ──
    reviewed_response, eth_status, eth_note = ethics.review_workload(
        schedule_list, raw_response, is_task_request
    )
    policy_recs = ethics.review_emotion_context(emotion, intensity)

    layer3 = LayerThreeData(
        status=eth_status, note=eth_note, policy_recommendations=policy_recs,
    )

    # ── Layer 4: Maternal Guardrail ──
    final_output, is_rewritten = guardrail.sanitize(
        reviewed_response, emotion, eth_status, empathy_data
    )

    layer4 = LayerFourData(
        is_rewritten=is_rewritten,
        original=raw_response if is_rewritten else "",
        final=final_output,
    )

    # ── Persist to history ──
    await _save_to_history(message, final_output, emotion, intent, intensity, eth_status, is_rewritten)

    return ChatResponse(
        response=final_output,
        layers=LayersData(layer1=layer1, layer2=layer2, layer3=layer3, layer4=layer4),
    )


async def _save_to_history(
    user_msg: str, bot_msg: str,
    emotion: str, intent: str, intensity: float,
    l3_status: str, l4_rewritten: bool,
):
    """Save user + assistant messages to chat_history."""
    db = await get_db()
    try:
        for role, content in [("user", user_msg), ("assistant", bot_msg)]:
            await db.execute(
                "INSERT INTO chat_history "
                "(role, content, emotion, intent, intensity, layer3_status, layer4_rewritten) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (role, content, emotion, intent, intensity, l3_status, int(l4_rewritten))
            )
        await db.commit()
    finally:
        await db.close()


async def get_history(limit: int = 50) -> list[dict]:
    """Get recent chat history."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, role, content, emotion, intent, intensity, "
            "layer3_status, layer4_rewritten, created_at "
            "FROM chat_history ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r[0], "role": r[1], "content": r[2], "emotion": r[3],
                "intent": r[4], "intensity": r[5], "layer3_status": r[6],
                "layer4_rewritten": bool(r[7]), "created_at": r[8],
            }
            for r in reversed(rows)
        ]
    finally:
        await db.close()


async def clear_history():
    """Clear all chat history."""
    db = await get_db()
    try:
        await db.execute("DELETE FROM chat_history")
        await db.commit()
    finally:
        await db.close()
