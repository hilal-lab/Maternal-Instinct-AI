"""
Chat Service — Full 4-Layer Pipeline Orchestrator.

Implements the exact workflow from the architecture diagram:

LAYER 1: Orchestration & Data Retrieval
  User Prompt
    → Executive Agent (intent classification)
    → Empathy Scout (emotion detection)
    → Database (via MCP Server get_schedule + get_daily_workload)
    → User Data & Context Retrieval
    → Embedding Model → FAISS Vector Database
    → Task Routing

LAYER 2: Specialist Task Agents
  Task Routing
    → MCP Server ↔ MCP Tools (schedule, notes, materials)
    → Planner Agent | Tutor Agent | Coach Agent

LAYER 3: Deliberation & Ethics
  Raw Draft
    → Policy & Ethics Aggregator
    → Workload Checking (decision: OVERLOAD / NOT_OVERLOAD)
    → Draft Logic Response

LAYER 4: Maternal Guardrail
  Draft Logic Response
    → LLM Response Wrap
    → Maternal Filter Protection Check (Good Response / Need Improve)
    → [Need Improve] → Nurturing & Empathic Rewriter
    → Final Response to User

Tool-Call Confirmation Flow
---------------------------
When Layer 2 emits a structured tool_call block and the tool is marked
"destructive", the pipeline:
  1. Stores the pending call in `pending_confirmations` (keyed by confirm_id).
  2. Emits a `tool_confirm` WebSocket event to the frontend.
  3. Returns `None` to the caller — the pipeline is suspended.

The WebSocket handler resumes the pipeline when it receives a
`tool_confirm_response` message from the frontend.
"""
import uuid
import logging
import re
from typing import Callable, Awaitable, Optional

from backend.models.database import get_db
from backend.models.schemas import (
    ChatResponse, LayersData,
    LayerOneData, LayerTwoData, LayerThreeData, LayerFourData,
)
from backend.core.layer_1_orchestrator import ExecutiveAgent
from backend.core.layer_2_specialists import SpecialistAgents
from backend.core.layer_3_ethics import PolicyAggregator
from backend.core.layer_4_guardrail import MaternalGuardrail
from backend.mcp_server.server import get_mcp_server

logger = logging.getLogger("backend.services.chat_service")

# Callback type: async fn(event: dict) -> None
ProgressCallback = Optional[Callable[[dict], Awaitable[None]]]

# ─── Pending confirmation store ───────────────────────────────────────────────
# Keyed by confirm_id (UUID string).
# Each entry holds everything needed to resume the pipeline after confirmation.
#
# Structure:
#   {
#     "confirm_id": str,
#     "tool_name": str,
#     "tool_args": dict,
#     "confirm_message": str,
#     "pipeline_context": {
#       "message": str,          # original user message
#       "emotion": str,
#       "intent": str,
#       "intensity": float,
#       "flags": dict,
#       "eth_status": str,
#       "eth_note": str,
#       "policy_recs": dict,
#       "workload_check": dict,
#       "agent_used": str,
#       "schedule_tasks": list,
#       "raw_response": str,      # text part of the agent response (no tool block)
#     }
#   }

pending_confirmations: dict[str, dict] = {}


def _extract_explicit_task_id(message: str) -> Optional[int]:
    """
    Extract exactly one explicitly labeled task ID from the current
    user message. Bare numbers, dates, and ordinal references are rejected.
    Accepted examples: 'ID 12', 'ID: 12', 'ID#12'.
    """
    if not isinstance(message, str):
        return None

    matches = re.findall(
        r"\bID\s*[:#]?\s*(\d+)\b",
        message,
        flags=re.IGNORECASE,
    )

    if len(matches) != 1:
        return None

    task_id = int(matches[0])
    return task_id if task_id > 0 else None


def _is_direct_task_delete_request(message: str) -> bool:
    """
    Detect direct requests to delete a scheduled task.

    Informational questions such as 'bagaimana cara menghapus tugas?'
    must not trigger schedule deletion handling.
    """
    if not isinstance(message, str):
        return False

    normalized = re.sub(r"\s+", " ", message.strip().lower())

    if not normalized:
        return False

    informational_prefixes = (
        "bagaimana",
        "gimana",
        "cara ",
        "apa itu",
        "kenapa",
        "mengapa",
        "apakah",
        "jelaskan",
        "tolong jelaskan",
    )

    if normalized.startswith(informational_prefixes):
        return False

    has_task_reference = bool(
        re.search(r"\b(tugas|task|jadwal)\b", normalized)
    )

    if not has_task_reference:
        return False

    direct_command = bool(
        re.search(r"\b(hapus|hapuskan|delete)\b", normalized)
    )

    stated_intention = bool(
        re.search(
            r"\b(saya|aku)\s+(ingin|mau|perlu)\s+menghapus\b",
            normalized,
        )
    )

    return direct_command or stated_intention


async def _get_task_snapshot(task_id: int) -> Optional[dict]:
    """
    Fetch a task by ID, including completed tasks.
    Returns a stable identity snapshot or None if the task does not exist.
    """
    db = await get_db()
    try:
        cursor = await db.execute(
            """
            SELECT id, task, scheduled_date, start_time, end_time, status
            FROM schedules
            WHERE id = ?
            """,
            (task_id,),
        )
        row = await cursor.fetchone()

        if row is None:
            return None

        return {
            "id": row["id"],
            "task": row["task"],
            "scheduled_date": row["scheduled_date"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "status": row["status"],
        }
    finally:
        await db.close()


def _format_delete_confirmation(snapshot: dict) -> str:
    """Build a confirmation message tied to the exact task identity."""
    date = snapshot["scheduled_date"] or "Belum ditentukan"
    start = snapshot["start_time"] or "--:--"
    end = snapshot["end_time"] or "--:--"

    return (
        "Ara akan menghapus tugas berikut secara permanen:\n"
        f"- ID: {snapshot['id']}\n"
        f"- Tugas: {snapshot['task']}\n"
        f"- Tanggal: {date}\n"
        f"- Waktu: {start}–{end}\n"
        f"- Status: {snapshot['status']}\n\n"
        "Apakah kamu yakin ingin menghapus tugas ini?"
    )


async def run_pipeline(
    message: str,
    on_event: ProgressCallback = None,
    mode: str = "conversation",
    chat_id: str = None,
) -> Optional[ChatResponse]:
    """
    Execute the full 4-layer architecture pipeline.

    Args:
        message: The user's input message.
        on_event: Optional callback for WebSocket progress events.
        mode: Chat mode — "conversation" or "learning".
        chat_id: Optional chat session ID for loading conversation history context.

    Returns:
        ChatResponse on success, or None if the pipeline was suspended
        waiting for a tool confirmation (a `tool_confirm` event will have
        been emitted to the client instead).
    """

    async def _emit(event: dict):
        if on_event:
            await on_event(event)

    # ── Instantiate layer components ──────────────────────────────────────────
    orchestrator = ExecutiveAgent()
    mcp = get_mcp_server()
    specialists = SpecialistAgents(mcp_server=mcp)
    ethics = PolicyAggregator()
    guardrail = MaternalGuardrail()

    # ── Chat ID default ──────────────────────────────────────────────────────
    if chat_id is None:
        chat_id = "default"

    # ═══════════════════════════════════════════════════════════════════════════
    # LAYER 1: Orchestration & Data Retrieval
    # ═══════════════════════════════════════════════════════════════════════════

    await _emit({"type": "layer_start", "layer": 1, "step": "intent",
                 "message": "Menganalisis intent & emosi..."})

    # Step 1a: Executive Agent + Empathy Scout (sync classification)
    intent, emotion, empathy_data, task_routing = orchestrator.analyze_intent(message)
    intensity = empathy_data.get("intensity", 0.0)
    flags = empathy_data.get("flags", {})

    await _emit({"type": "layer_progress", "layer": 1,
                 "step": "classified",
                 "intent": intent, "emotion": emotion, "intensity": intensity,
                 "message": f"Intent: {intent} | Emosi: {emotion} ({intensity:.0%})"})

    await _emit({"type": "layer_progress", "layer": 1, "step": "context",
                 "message": "Mengambil konteks jadwal & RAG..."})

    # Step 1b: Context Retrieval — DB (via MCP) + FAISS embedding search + Chat history
    context = await orchestrator.context_retriever.retrieve(message, chat_id=chat_id)

    schedule_tasks = context.get("schedule_tasks", [])
    workload = context.get("workload", {})
    rag_context = context.get("rag_context", "")
    chat_history_context = context.get("chat_history_context", "")

    await _emit({"type": "layer_done", "layer": 1,
                 "message": f"Layer 1 selesai. Routing ke: {', '.join(task_routing) or 'general_chat'}"})

    layer1 = LayerOneData(
        intent=intent,
        emotion=emotion,
        intensity=intensity,
        flags=flags,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # LAYER 2: Specialist Task Agents (via MCP Server)
    # ═══════════════════════════════════════════════════════════════════════════

    await _emit({"type": "layer_start", "layer": 2,
                 "message": "Mengirim ke agen spesialis via MCP..."})

    # MCP Server provides data context; specialists use it for generation.
    # run_with_mcp now returns a 3-tuple: (response, agent, tool_call | None)
    raw_response, agent_used, tool_call = await specialists.run_with_mcp(
        user_input=message,
        intent=intent,
        emotion=emotion,
        empathy_data=empathy_data,
        task_routing=task_routing,
        rag_context=rag_context,
        schedule_tasks=schedule_tasks,
        workload=workload,
        chat_history_context=chat_history_context,
    )

    await _emit({"type": "layer_done", "layer": 2,
                 "agent": agent_used,
                 "raw_draft": raw_response,
                 "message": f"Layer 2 selesai. Agen: {agent_used}"})

    layer2 = LayerTwoData(agent_used=agent_used, raw_response=raw_response)

    # ═══════════════════════════════════════════════════════════════════════════
    # LAYER 3: Deliberation & Ethics
    # ═══════════════════════════════════════════════════════════════════════════

    await _emit({"type": "layer_start", "layer": 3,
                 "message": "Memeriksa beban kerja & etika..."})

    is_task_request = (intent == "TASK_OPS")

    workload_check = ethics.check_workload(workload)

    await _emit({"type": "layer_progress", "layer": 3,
                 "step": "workload",
                 "decision": workload_check.get("decision", "NOT_OVERLOAD"),
                 "message": f"Workload: {workload_check.get('decision', 'NOT_OVERLOAD')}"})

    reviewed_response, eth_status, eth_note = ethics.review_workload(
        task_list=schedule_tasks,
        response_text=raw_response,
        is_plan_request=is_task_request,
        workload_check_result=workload_check,
    )

    policy_recs = ethics.review_emotion_context(emotion, intensity)

    await _emit({"type": "layer_done", "layer": 3,
                 "status": eth_status,
                 "message": f"Layer 3 selesai. Status etika: {eth_status}"})

    layer3 = LayerThreeData(
        status=eth_status,
        note=eth_note,
        policy_recommendations={
            **policy_recs,
            "workload_decision": workload_check.get("decision", "NOT_OVERLOAD"),
            "workload_message": workload_check.get("message", ""),
        },
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # TOOL CALL CHECK — runs AFTER Layer 3, BEFORE Layer 4
    # ═══════════════════════════════════════════════════════════════════════════

    policy_a_delete_request = (
        _is_direct_task_delete_request(message)
        and _extract_explicit_task_id(message) is None
    )

    if policy_a_delete_request:
        # Policy A: selalu ambil jadwal aktual, terlepas dari
        # tool_call yang dihasilkan Planner.
        tool_result = await mcp.call_tool("get_schedule")

        reviewed_response = _format_tool_result(
            "get_schedule",
            tool_result,
        )

        if tool_result.get("success") and tool_result.get("tasks"):
            reviewed_response += (
                "\n\nSilakan pilih tugas yang ingin dihapus "
                "dengan menyebutkan ID persis dari daftar di atas."
            )

    elif tool_call:
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})

        if not isinstance(tool_args, dict):
            tool_args = {}

        if mcp.is_destructive(tool_name):
            delete_snapshot = None
            validation_error = None
            handled_without_confirmation = False

            # ── Extra validation for delete_task ─────────────────────────
            if tool_name == "delete_task":
                explicit_id = _extract_explicit_task_id(message)

                if explicit_id is None:
                    # Jangan tampilkan draft Planner karena bisa mengandung
                    # daftar tugas atau ID yang tidak bersumber dari database.
                    tool_result = await mcp.call_tool("get_schedule")

                    reviewed_response = _format_tool_result(
                        "get_schedule",
                        tool_result,
                    )

                    if tool_result.get("success") and tool_result.get("tasks"):
                        reviewed_response += (
                            "\n\nSilakan pilih tugas yang ingin dihapus "
                            "dengan menyebutkan ID persis dari daftar di atas."
                        )

                    # Tandai bahwa permintaan sudah ditangani tanpa konfirmasi.
                    handled_without_confirmation = True
                else:
                    task_id = tool_args.get("task_id")

                    if type(task_id) is not int or task_id <= 0:
                        validation_error = (
                            "Ara belum bisa menghapus tugas karena ID tugas "
                            "tidak valid. Silakan sebutkan ID tugas secara eksplisit."
                        )
                    elif explicit_id != task_id:
                        validation_error = (
                            f"ID yang kamu sebutkan ({explicit_id}) tidak cocok "
                            f"dengan ID tugas yang akan diproses ({task_id}). "
                            "Silakan periksa kembali daftar jadwal dan sebutkan "
                            "ID yang tepat."
                        )
                    else:
                        delete_snapshot = await _get_task_snapshot(task_id)

                        if delete_snapshot is None:
                            validation_error = (
                                f"Tugas dengan ID {task_id} tidak ditemukan. "
                                "Silakan periksa kembali daftar jadwal."
                            )

            if handled_without_confirmation:
                pass
            elif validation_error:
                # Do not create a confirmation or execute the tool.
                reviewed_response = (
                    (reviewed_response + "\n\n" if reviewed_response else "")
                    + validation_error
                )
            else:
                # ── Destructive tool: suspend and ask for confirmation ───
                confirm_id = str(uuid.uuid4())
                confirm_message = mcp.get_confirm_message(tool_name)

                if tool_name == "delete_task":
                    confirm_message = _format_delete_confirmation(
                        delete_snapshot
                    )

                pending_confirmations[confirm_id] = {
                    "confirm_id": confirm_id,
                    "tool_name": tool_name,
                    "tool_args": dict(tool_args),
                    "confirm_message": confirm_message,
                    "delete_task_snapshot": delete_snapshot,
                    "pipeline_context": {
                        "message": message,
                        "emotion": emotion,
                        "intent": intent,
                        "intensity": intensity,
                        "flags": flags,
                        "eth_status": eth_status,
                        "eth_note": eth_note,
                        "policy_recs": policy_recs,
                        "workload_check": workload_check,
                        "agent_used": agent_used,
                        "schedule_tasks": schedule_tasks,
                        "raw_response": reviewed_response,
                    },
                }

                logger.info(
                    f"Destructive tool '{tool_name}' pending confirmation "
                    f"(confirm_id={confirm_id})"
                )

                await _emit({
                    "type": "tool_confirm",
                    "confirm_id": confirm_id,
                    "tool": tool_name,
                    "args": tool_args,
                    "description": confirm_message,
                    "message": f"Ara membutuhkan konfirmasi untuk: {tool_name}",
                })

                return None

        else:
            # ── Non-destructive tool: execute immediately ────────────────
            logger.info(
                f"Executing non-destructive tool '{tool_name}' "
                f"with args {tool_args}"
            )
            await _emit({
                "type": "layer_progress",
                "layer": 3,
                "step": "tool_call",
                "message": f"Menjalankan tool: {tool_name}...",
            })

            tool_result = await mcp.call_tool(tool_name, **tool_args)

            result_summary = _format_tool_result(tool_name, tool_result)
            reviewed_response = (
                reviewed_response + "\n\n" + result_summary
                if reviewed_response
                else result_summary
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # LAYER 4: Maternal Guardrail
    # ═══════════════════════════════════════════════════════════════════════════

    await _emit({"type": "layer_start", "layer": 4,
                 "message": "Menerapkan Maternal Guardrail..."})

    final_output, is_rewritten = guardrail.sanitize(
        raw_response=reviewed_response,
        emotion=emotion,
        layer_3_status=eth_status,
        empathy_data=empathy_data,
    )

    await _emit({"type": "layer_done", "layer": 4,
                 "is_rewritten": is_rewritten,
                 "message": f"Layer 4 selesai. {'Ditulis ulang' if is_rewritten else 'Lolos filter'}"})

    layer4 = LayerFourData(
        is_rewritten=is_rewritten,
        original=reviewed_response if is_rewritten else "",
        final=final_output,
    )

    # ── Persist to chat history ───────────────────────────────────────────────
    await _save_to_history(
        message, final_output, emotion, intent, intensity, eth_status, is_rewritten, mode, chat_id
    )

    return ChatResponse(
        response=final_output,
        layers=LayersData(layer1=layer1, layer2=layer2, layer3=layer3, layer4=layer4),
    )


async def resume_after_confirmation(
    confirm_id: str,
    approved: bool,
    on_event: ProgressCallback = None,
) -> Optional[ChatResponse]:
    """
    Resume a suspended pipeline after the user responds to a tool confirmation.

    Args:
        confirm_id: The UUID string from the `tool_confirm` event.
        approved:   True if the user approved the tool execution.
        on_event:   Same callback used in the original `run_pipeline` call.

    Returns:
        ChatResponse on success, or None if confirmation was not found.
    """

    async def _emit(event: dict):
        if on_event:
            await on_event(event)

    pending = pending_confirmations.pop(confirm_id, None)
    if not pending:
        logger.warning(f"resume_after_confirmation: unknown confirm_id={confirm_id}")
        return None

    ctx = pending["pipeline_context"]
    tool_name = pending["tool_name"]
    tool_args = pending["tool_args"]

    mcp = get_mcp_server()
    guardrail = MaternalGuardrail()
    ethics = PolicyAggregator()

    reviewed_response = ctx["raw_response"]
    emotion = ctx["emotion"]
    eth_status = ctx["eth_status"]
    eth_note = ctx["eth_note"]
    policy_recs = ctx["policy_recs"]
    workload_check = ctx["workload_check"]
    intent = ctx["intent"]
    intensity = ctx["intensity"]
    flags = ctx["flags"]
    agent_used = ctx["agent_used"]
    schedule_tasks = ctx["schedule_tasks"]
    message = ctx["message"]

    # Rebuild layer data objects
    layer1 = LayerOneData(intent=intent, emotion=emotion, intensity=intensity, flags=flags)
    layer2 = LayerTwoData(agent_used=agent_used, raw_response=reviewed_response)
    layer3 = LayerThreeData(
        status=eth_status,
        note=eth_note,
        policy_recommendations={
            **policy_recs,
            "workload_decision": workload_check.get("decision", "NOT_OVERLOAD"),
            "workload_message": workload_check.get("message", ""),
        },
    )

    if approved:
        execution_allowed = True

        if tool_name == "delete_task":
            confirmed_snapshot = pending.get("delete_task_snapshot")
            task_id = tool_args.get("task_id")
            explicit_id = _extract_explicit_task_id(message)

            if (
                type(task_id) is not int
                or task_id <= 0
                or explicit_id != task_id
                or not isinstance(confirmed_snapshot, dict)
                or confirmed_snapshot.get("id") != task_id
            ):
                execution_allowed = False
                block_reason = (
                    "Penghapusan dibatalkan karena identitas tugas "
                    "tidak cocok dengan konfirmasi yang tersimpan."
                )
            else:
                current_snapshot = await _get_task_snapshot(task_id)

                if current_snapshot is None:
                    execution_allowed = False
                    block_reason = (
                        f"Tugas dengan ID {task_id} sudah tidak ditemukan. "
                        "Tidak ada penghapusan yang dilakukan."
                    )
                elif current_snapshot != confirmed_snapshot:
                    execution_allowed = False
                    block_reason = (
                        f"Data tugas dengan ID {task_id} telah berubah "
                        "sejak konfirmasi ditampilkan. Penghapusan dibatalkan. "
                        "Silakan periksa jadwal dan ulangi permintaan."
                    )

            if not execution_allowed:
                reviewed_response = (
                    (reviewed_response + "\n\n" if reviewed_response else "")
                    + block_reason
                )

        if execution_allowed:
            logger.info(
                f"User approved tool '{tool_name}' "
                f"(confirm_id={confirm_id}). Executing."
            )
            await _emit({
                "type": "layer_progress",
                "layer": 3,
                "step": "tool_executing",
                "message": f"Menjalankan: {tool_name}...",
            })

            if tool_name == "delete_task":
                tool_result = await mcp.call_tool(
                    tool_name,
                    **tool_args,
                    expected_snapshot=confirmed_snapshot,
                )
            else:
                tool_result = await mcp.call_tool(tool_name, **tool_args)
            result_summary = _format_tool_result(tool_name, tool_result)
            reviewed_response = (
                reviewed_response + "\n\n" + result_summary
                if reviewed_response
                else result_summary
            )
    else:
        logger.info(f"User cancelled tool '{tool_name}' (confirm_id={confirm_id}).")
        reviewed_response = (
            (reviewed_response + "\n\n" if reviewed_response else "") +
            "Baik, tindakan dibatalkan. Apakah ada yang bisa Ara bantu lagi?"
        )

    # ── Layer 4 ───────────────────────────────────────────────────────────────
    await _emit({"type": "layer_start", "layer": 4,
                 "message": "Menerapkan Maternal Guardrail..."})

    final_output, is_rewritten = guardrail.sanitize(
        raw_response=reviewed_response,
        emotion=emotion,
        layer_3_status=eth_status,
        empathy_data={"intensity": intensity},
    )

    await _emit({"type": "layer_done", "layer": 4,
                 "is_rewritten": is_rewritten,
                 "message": f"Layer 4 selesai. {'Ditulis ulang' if is_rewritten else 'Lolos filter'}"})

    layer4 = LayerFourData(
        is_rewritten=is_rewritten,
        original=reviewed_response if is_rewritten else "",
        final=final_output,
    )

    await _save_to_history(
        message, final_output, emotion, intent, intensity, eth_status, is_rewritten
    )

    return ChatResponse(
        response=final_output,
        layers=LayersData(layer1=layer1, layer2=layer2, layer3=layer3, layer4=layer4),
    )


def _format_tool_result(tool_name: str, result: dict) -> str:
    """
    Format a tool execution result into a human-readable Indonesian string
    for injection into the reviewed_response before Layer 4.
    """
    if not result.get("success"):
        error = result.get("error", "Terjadi kesalahan.")
        return f"[Ara mencoba menjalankan '{tool_name}' tapi terjadi kesalahan: {error}]"

    if tool_name == "get_schedule":
        tasks = result.get("tasks", [])

        if not tasks:
            return "Tidak ada tugas aktif dalam jadwal saat ini."

        lines = [
            "Berikut daftar tugas aktual dari jadwal:",
        ]

        for task in tasks:
            task_id = task.get("id")
            title = task.get("task", "Tanpa nama")
            scheduled_date = (
                task.get("scheduled_date")
                or task.get("deadline")
                or "-"
            )
            start_time = task.get("start_time") or "-"
            end_time = task.get("end_time") or "-"
            status = task.get("status", "-")
            priority = task.get("priority", "-")

            lines.append(
                f"- ID: {task_id} | {title} | "
                f"Tanggal: {scheduled_date} | "
                f"Waktu: {start_time}-{end_time} | "
                f"Status: {status} | Prioritas: {priority}"
            )

        return "\n".join(lines)

    if tool_name == "add_task":
        task = result.get("task", {})
        return (
            f"Ara berhasil menambahkan tugas baru: **{task.get('task', '')}** "
            f"(Deadline: {task.get('deadline', '')}, Prioritas: {task.get('priority', '')})."
        )
    elif tool_name == "update_task_status":
        task = result.get("task", {})
        return (
            f"Status tugas **{task.get('task', '')}** berhasil diperbarui "
            f"menjadi **{task.get('status', '')}**."
        )
    elif tool_name == "delete_task":
        task = result.get("deleted_task", {})
        return f"Tugas **{task.get('task', '')}** berhasil dihapus dari jadwal."
    elif tool_name == "delete_document":
        return f"Dokumen (ID: {result.get('doc_id', '')}) berhasil dihapus dari basis pengetahuan."
    else:
        return f"[Tool '{tool_name}' berhasil dijalankan.]"


# ─── History helpers ──────────────────────────────────────────────────────────

async def _save_to_history(
    user_msg: str, bot_msg: str,
    emotion: str, intent: str, intensity: float,
    l3_status: str, l4_rewritten: bool,
    mode: str = "conversation",
    chat_id: str = "default",
):
    """Save user + assistant messages to chat_history."""
    db = await get_db()
    try:
        for role, content in [("user", user_msg), ("assistant", bot_msg)]:
            await db.execute(
                "INSERT INTO chat_history "
                "(chat_id, mode, role, content, emotion, intent, intensity, layer3_status, layer4_rewritten) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (chat_id, mode, role, content, emotion, intent, intensity, l3_status, int(l4_rewritten))
            )
        await db.commit()
    finally:
        await db.close()


async def get_history(limit: int = 50, mode: str = None, chat_id: str = None) -> list[dict]:
    """Get recent chat history, optionally filtered by mode and/or chat_id."""
    db = await get_db()
    try:
        if mode and chat_id:
            cursor = await db.execute(
                "SELECT id, chat_id, mode, role, content, emotion, intent, intensity, "
                "layer3_status, layer4_rewritten, created_at "
                "FROM chat_history WHERE mode = ? AND chat_id = ? ORDER BY id DESC LIMIT ?",
                (mode, chat_id, limit)
            )
        elif mode:
            cursor = await db.execute(
                "SELECT id, chat_id, mode, role, content, emotion, intent, intensity, "
                "layer3_status, layer4_rewritten, created_at "
                "FROM chat_history WHERE mode = ? ORDER BY id DESC LIMIT ?",
                (mode, limit)
            )
        elif chat_id:
            cursor = await db.execute(
                "SELECT id, chat_id, mode, role, content, emotion, intent, intensity, "
                "layer3_status, layer4_rewritten, created_at "
                "FROM chat_history WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
                (chat_id, limit)
            )
        else:
            cursor = await db.execute(
                "SELECT id, chat_id, mode, role, content, emotion, intent, intensity, "
                "layer3_status, layer4_rewritten, created_at "
                "FROM chat_history ORDER BY id DESC LIMIT ?",
                (limit,)
            )
        rows = await cursor.fetchall()
        return [
            {
                "id": r[0], "chat_id": r[1], "mode": r[2], "role": r[3], "content": r[4],
                "emotion": r[5], "intent": r[6], "intensity": r[7], "layer3_status": r[8],
                "layer4_rewritten": bool(r[9]), "created_at": r[10],
            }
            for r in reversed(rows)
        ]
    finally:
        await db.close()


async def clear_history(mode: str = None, chat_id: str = None):
    """Clear chat history, optionally filtered by mode and/or chat_id."""
    db = await get_db()
    try:
        if mode and chat_id:
            await db.execute("DELETE FROM chat_history WHERE mode = ? AND chat_id = ?", (mode, chat_id))
        elif mode:
            await db.execute("DELETE FROM chat_history WHERE mode = ?", (mode,))
        elif chat_id:
            await db.execute("DELETE FROM chat_history WHERE chat_id = ?", (chat_id,))
        else:
            await db.execute("DELETE FROM chat_history")
        await db.commit()
    finally:
        await db.close()
