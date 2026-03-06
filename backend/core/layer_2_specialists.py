"""
Layer 2: Specialist Task Agents — The Experts (Paper Section III-B).  # noqa: E501

Workflow (matches diagram):
  Task Routing (from Layer 1)
        │
        ▼
   MCP Server ◄──► MCP Tools (schedule, notes, materials)
        │
   ┌────┴──────────────────┐
   ▼           ▼           ▼
Planner     Tutor        Coach
 Agent      Agent        Agent
   └────────────────────────┘
                │
                ▼
         Raw Draft Response
         (to Layer 3)

The MCP Server is the data backbone of this layer.
Each agent receives its context (schedule, workload, materials) FROM
the MCP server rather than directly from the DB.

Tool-Call Protocol
------------------
When a user's intent requires a write operation (add task, update status,
delete task, delete document), the Planner agent embeds a structured JSON
block in its raw response using the following format:

    ```tool_call
    {"name": "add_task", "args": {"task": "...", "deadline": "...", ...}}
    ```

`extract_tool_call()` parses this block. If found, the pipeline pauses to
ask the user for confirmation before executing the tool.
"""
import json
import logging
import re
from typing import Optional
from backend.core.llm_client import generate_text

logger = logging.getLogger("backend.core.layer_2_specialists")


# ─── Tool-call extraction helper ─────────────────────────────────────────────

def extract_tool_call(text: str) -> Optional[dict]:
    """
    Parse a structured tool-call block from an LLM response.

    Looks for a fenced code block tagged ``tool_call``:

        ```tool_call
        {"name": "delete_task", "args": {"task_id": 3}}
        ```

    Also accepts plain JSON objects anywhere in the text that have a
    "tool_call" top-level key as a fallback:

        {"tool_call": {"name": "add_task", "args": {...}}}

    Args:
        text: Raw LLM output string.

    Returns:
        Dict with keys ``name`` (str) and ``args`` (dict), or None if
        no valid tool call is found.
    """
    if not text:
        return None

    # Primary: fenced ```tool_call ... ``` block
    fence_pattern = re.compile(
        r"```tool_call\s*\n(.*?)\n```",
        re.DOTALL | re.IGNORECASE,
    )
    match = fence_pattern.search(text)
    if match:
        try:
            payload = json.loads(match.group(1).strip())
            if isinstance(payload, dict) and "name" in payload:
                return {
                    "name": str(payload["name"]),
                    "args": payload.get("args", {}),
                }
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback: {"tool_call": {"name": ..., "args": ...}} anywhere in text
    fallback_pattern = re.compile(
        r'\{\s*"tool_call"\s*:\s*(\{.*?\})\s*\}',
        re.DOTALL,
    )
    match = fallback_pattern.search(text)
    if match:
        try:
            payload = json.loads(match.group(1).strip())
            if isinstance(payload, dict) and "name" in payload:
                return {
                    "name": str(payload["name"]),
                    "args": payload.get("args", {}),
                }
        except (json.JSONDecodeError, KeyError):
            pass

    return None


def strip_tool_call_block(text: str) -> str:
    """
    Remove the tool_call fenced block from an LLM response so the
    remaining text can be used as a natural-language message.
    """
    cleaned = re.sub(
        r"```tool_call\s*\n.*?\n```",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Also strip {"tool_call": ...} fallback form
    cleaned = re.sub(
        r'\{\s*"tool_call"\s*:\s*\{.*?\}\s*\}',
        "",
        cleaned,
        flags=re.DOTALL,
    )
    return cleaned.strip()




class SpecialistAgents:
    """
    Three specialist agents that work based on the task routing from Layer 1.
    All agents fetch their data context via the MCP Server.
    """

    def __init__(self, mcp_server=None):
        """
        Args:
            mcp_server: MCPServer instance. If None, a singleton will be fetched lazily.
        """
        self._mcp = mcp_server

    def _get_mcp(self):
        """Get MCPServer instance (lazy singleton)."""
        if self._mcp is None:
            from backend.mcp_server.server import get_mcp_server
            self._mcp = get_mcp_server()
        return self._mcp

    # ── Context formatting helpers ────────────────────────────────────────────

    @staticmethod
    def _format_schedule(task_list: list) -> str:
        """Format schedule tasks into a readable string for LLM context."""
        if not task_list:
            return ""
        tasks_str = "\n".join([
            f"  - {t['task']} (Deadline: {t['deadline']} | Estimasi: {t['est_hours']} jam | "
            f"Prioritas: {t['priority']} | Status: {t.get('status', 'pending')})"
            for t in task_list
        ])
        return f"\n\n📋 DATA JADWAL USER SAAT INI:\n{tasks_str}\n"

    @staticmethod
    def _format_workload(workload: dict) -> str:
        """Format workload summary into a string for LLM context."""
        if not workload or not workload.get("success"):
            return ""
        total = workload.get("total_hours", 0)
        count = workload.get("task_count", 0)
        overloaded = workload.get("overloaded", False)
        overload_by = workload.get("overload_by_hours", 0)
        if overloaded:
            return (
                f"\n⚠️ PERINGATAN BEBAN KERJA: Total {total:.1f} jam dari {count} tugas "
                f"— MELEBIHI batas aman 8 jam/hari sebesar {overload_by:.1f} jam!\n"
            )
        return f"\n📊 Beban kerja saat ini: {total:.1f} jam dari {count} tugas (dalam batas aman).\n"

    # ── Planner Agent ─────────────────────────────────────────────────────────

    def run_planner(self, task_list: list, rag_context: str = "",
                    workload: Optional[dict] = None, mcp_schedule_str: str = "",
                    user_input: str = "") -> str:
        """
        Study Planner Agent (Paper Section III-B-a).

        Capabilities: Task Chunking, Priority Structuring, Load Balancing,
        Cognitive Load Management, Adaptive Scheduling.

        The agent receives schedule data from MCP Server tools.
        When the user explicitly asks to ADD, UPDATE, or DELETE a task/document,
        the agent embeds a structured tool_call block in its response.
        """
        if not task_list and not user_input:
            return "Tidak ada data jadwal untuk diproses."

        tasks_str = mcp_schedule_str or self._format_schedule(task_list)
        workload_str = self._format_workload(workload) if workload else ""

        # List of available write tools for the agent to reference
        tool_reference = """
TOOL YANG TERSEDIA (gunakan hanya jika user SECARA EKSPLISIT meminta):
  - add_task: Tambahkan tugas baru
    args: task (str), deadline (str YYYY-MM-DD), est_hours (float), priority (str: High/Medium/Low)
  - update_task_status: Ubah status tugas
    args: task_id (int), status (str: pending/in_progress/completed)
  - delete_task: Hapus tugas secara permanen
    args: task_id (int)

Jika user meminta salah satu tindakan di atas, sertakan blok tool_call di AKHIR respons:
```tool_call
{"name": "NAMA_TOOL", "args": {"param1": "nilai1", ...}}
```
Selain itu, tetap berikan penjelasan natural language sebelum blok tersebut.
JANGAN gunakan tool_call jika user hanya bertanya atau meminta saran.
"""

        prompt = f"""
{tasks_str}
{workload_str}

Pesan user: {user_input or "Buatkan rencana pengerjaan."}

Instruksi:
Jika user meminta tindakan (tambah/hapus/ubah tugas), gunakan tool yang sesuai dengan format di atas.
Jika user meminta saran atau rencana, buatkan step-by-step menggunakan prinsip:
1. Prioritas Eisenhower (Urgent-Important dulu).
2. Task Chunking: Pecah tugas besar ke unit 25-50 menit.
3. Load Balancing: Distribusikan beban kerja merata.
4. Sisipkan waktu istirahat (minimal 10 menit per 50 menit kerja).
5. Estimasikan total jam dan pastikan tidak melebihi 8 jam/hari.
Gunakan format Markdown yang rapi.

{rag_context}

{tool_reference}
"""
        return generate_text(
            prompt,
            system_instruction=(
                "Anda adalah Study Planner Profesional yang mengutamakan produktivitas "
                "sekaligus kesejahteraan. Data jadwal diperoleh dari sistem MCP. "
                "Anda dapat menggunakan tool untuk mengelola jadwal jika user memintanya secara eksplisit."
            )
        )

    # ── Tutor Agent ───────────────────────────────────────────────────────────

    def run_tutor(self, topic: str, rag_context: str = "",
                  schedule_context: str = "") -> str:
        """
        Tutor Agent (Paper Section III-B-b).

        Capabilities: Concept Simplification, Scaffolding, Diagnostic Prompting,
        Active Recall, Difficulty Adjustment.
        """
        prompt = f"""
        Topik: {topic}

        Instruksi:
        Jelaskan topik di atas secara pedagogis:
        1. Mulai dengan analogi sederhana.
        2. Gunakan pendekatan scaffolding (step-by-step).
        3. Akhiri dengan pertanyaan active recall untuk menguji pemahaman.
        Gunakan bahasa yang mudah dipahami mahasiswa.
        {schedule_context}
        {rag_context}
        """
        return generate_text(
            prompt,
            system_instruction=(
                "Anda adalah Tutor Agent yang sabar dan pedagogis. "
                "Jelaskan dengan analogi dan pendekatan scaffolding."
            )
        )

    # ── Coach Agent ───────────────────────────────────────────────────────────

    def run_coach(self, emotion: str, empathy_data: Optional[dict] = None,
                  rag_context: str = "", schedule_context: str = "") -> str:
        """
        Coach Agent (Paper Section III-B-c).

        Capabilities: Cognitive Reframing, Micro-Goal Encouragement,
        Burnout Prevention, Habit Reinforcement, Emotional Validation.

        Always warm and supportive — like a real mother who encourages her child.
        """
        sys_inst = (
            "Anda adalah 'Ara', seorang Coach dengan naluri keibuan yang hangat dan penuh kasih sayang. "
            "Anda selalu memvalidasi perasaan user terlebih dahulu, lalu memberikan semangat "
            "dan saran praktis dengan nada lembut tapi tegas. "
            "Jika ada data jadwal, gunakan untuk memberikan saran yang spesifik dan relevan."
        )

        prompt = (
            f"Kondisi emosional user: {emotion} "
            f"(Intensitas: {empathy_data.get('intensity', 0.5) if empathy_data else 0.5})\n"
            f"Berikan motivasi dan dukungan yang tulus. "
            f"Akui perasaan user, lalu bantu mereka melihat langkah kecil yang bisa dilakukan.\n"
        )

        if schedule_context:
            prompt += f"\n{schedule_context}"
        if rag_context:
            prompt += f"\n\nReferensi Tambahan:\n{rag_context}"

        return generate_text(prompt, system_instruction=sys_inst)

    # ── General Chat ──────────────────────────────────────────────────────────

    def run_general_chat(self, user_input: str, emotion: str, empathy_data: Optional[dict] = None,
                         rag_context: str = "", schedule_context: str = "") -> str:
        """
        General chat handler — always warm, helpful, and schedule-aware.
        Acts as 'Ara', a motherly AI assistant who has full access to the
        user's schedule and can answer questions about priorities, deadlines, etc.
        """
        sys_inst = (
            "Anda adalah 'Ara', AI asisten dengan naluri keibuan yang hangat dan peduli. "
            "Anda ramah, membantu, dan selalu memvalidasi perasaan user. "
            "Jika user bertanya tentang jadwal, tugas, deadline, atau prioritas, "
            "GUNAKAN data jadwal yang tersedia untuk menjawab dengan spesifik dan akurat. "
            "Jangan pernah bilang Anda tidak bisa melihat jadwal — Anda SUDAH memiliki datanya. "
            "Jawab dalam Bahasa Indonesia yang natural dan penuh kasih sayang."
        )

        prompt = user_input
        if schedule_context:
            prompt += f"\n{schedule_context}"
        if rag_context:
            prompt += f"\n\nReferensi Tambahan:\n{rag_context}"

        return generate_text(prompt, system_instruction=sys_inst)

    # ── MCP-aware execution entry points ─────────────────────────────────────

    async def run_with_mcp(
        self,
        user_input: str,
        intent: str,
        emotion: str,
        empathy_data: dict,
        task_routing: list[str],
        rag_context: str = "",
        schedule_tasks: Optional[list] = None,
        workload: Optional[dict] = None,
    ) -> tuple[str, str, Optional[dict]]:
        """
        Execute the appropriate specialist agent(s) based on task_routing from Layer 1,
        using data already fetched via MCP Server in Layer 1 context retrieval.

        This method is the canonical entry point when running the full pipeline.

        Args:
            user_input:     Original user message.
            intent:         Classified intent from Layer 1.
            emotion:        Detected emotion from Empathy Scout.
            empathy_data:   Full empathy analysis dict.
            task_routing:   List of agents to activate (from ROUTING_MAP).
            rag_context:    Pre-fetched RAG context string.
            schedule_tasks: Pre-fetched schedule task list (from MCP get_schedule).
            workload:       Pre-fetched workload dict (from MCP get_daily_workload).

        Returns:
            (raw_response: str, agent_used: str, tool_call: dict | None)
            tool_call is {"name": str, "args": dict} if the LLM emitted one,
            or None if no tool action was requested.
        """
        schedule_tasks = schedule_tasks or []
        workload = workload or {}

        mcp = self._get_mcp()

        # Format context strings for injection into prompts
        schedule_context = mcp.format_schedule_for_prompt({"tasks": schedule_tasks})
        workload_context = mcp.format_workload_for_prompt(workload)
        combined_context = "\n".join(filter(None, [schedule_context, workload_context]))

        raw_response = ""
        agent_used = "general_chat"

        if intent == "TASK_OPS":
            plan = self.run_planner(
                schedule_tasks,
                rag_context=rag_context,
                workload=workload,
                mcp_schedule_str=schedule_context,
                user_input=user_input,
            )
            # Check for tool call in planner output BEFORE blending with coach
            tool_call = extract_tool_call(plan)
            if tool_call:
                # Strip the tool_call block from the human-readable part
                plan_text = strip_tool_call_block(plan)
                coach = self.run_coach(
                    emotion, empathy_data,
                    rag_context=rag_context,
                    schedule_context=combined_context,
                )
                raw_response = f"{plan_text}\n\n**Pesan Coach:**\n{coach}"
                agent_used = "planner+coach"
                return raw_response, agent_used, tool_call

            coach = self.run_coach(
                emotion, empathy_data,
                rag_context=rag_context,
                schedule_context=combined_context,
            )
            raw_response = f"{plan}\n\n**Pesan Coach:**\n{coach}"
            agent_used = "planner+coach"

        elif intent == "ACADEMIC_HELP":
            raw_response = self.run_tutor(
                user_input,
                rag_context=rag_context,
                schedule_context=combined_context,
            )
            agent_used = "tutor"

        elif intent in ("MOTIVATION_SUPPORT", "EMOTIONAL_DISTRESS"):
            coach = self.run_coach(
                emotion, empathy_data,
                rag_context=rag_context,
                schedule_context=combined_context,
            )
            general = self.run_general_chat(
                user_input, emotion, empathy_data,
                rag_context=rag_context,
                schedule_context=combined_context,
            )
            raw_response = f"{general}\n\n**Pesan Coach:**\n{coach}"
            agent_used = "general_chat+coach"

        else:
            raw_response = self.run_general_chat(
                user_input, emotion, empathy_data,
                rag_context=rag_context,
                schedule_context=combined_context,
            )
            agent_used = "general_chat"

        # For non-planner paths, also check if the LLM smuggled in a tool call
        tool_call = extract_tool_call(raw_response)
        if tool_call:
            raw_response = strip_tool_call_block(raw_response)

        return raw_response, agent_used, tool_call
