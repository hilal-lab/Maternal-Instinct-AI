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
    {"name": "add_task", "args": {
        "task": "...",
        "scheduled_date": "YYYY-MM-DD",
        "start_time": "HH:MM",
        "end_time": "HH:MM"
    }}
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
    Extract and validate structured tool call.

    Accepted format:
    ```tool_call
    {"name": "add_task", "args": {...}}
    ```
    """

    if not text:
        return None

    allowed_tools = {
        "get_schedule",
        "add_task",
        "update_task_status",
        "delete_task",
    }

    def validate(payload):
        if not isinstance(payload, dict):
            return None

        name = payload.get("name")
        args = payload.get("args", {})

        if not isinstance(name, str):
            return None

        if name not in allowed_tools:
            return None

        if not isinstance(args, dict):
            return None

        return {
            "name": name,
            "args": args,
        }

    # 1. Primary: fenced tool_call block.
    fence_pattern = re.compile(
        r"```tool_call\s*(.*?)\s*```",
        re.DOTALL | re.IGNORECASE,
    )

    for match in fence_pattern.finditer(text):
        try:
            payload = json.loads(match.group(1).strip())
            result = validate(payload)

            if result is not None:
                return result

        except (json.JSONDecodeError, TypeError):
            continue

    # 2. Fallback: JSON object containing tool_call.
    decoder = json.JSONDecoder()

    for match in re.finditer(r"\{", text):
        try:
            payload, _ = decoder.raw_decode(text[match.start():])

            if not isinstance(payload, dict):
                continue

            nested = payload.get("tool_call")
            result = validate(nested)

            if result is not None:
                return result

        except (json.JSONDecodeError, TypeError):
            continue

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
        tool_reference = """TOOL YANG TERSEDIA:

READ-ONLY TOOL:

1. get_schedule
Fungsi: Mengambil daftar tugas aktif yang belum selesai.
Args: {}
Gunakan untuk mengambil daftar tugas aktual dari database.

WRITE TOOLS:

2. add_task
Fungsi: Menambahkan tugas baru dengan tanggal dan rentang waktu terjadwal.
Args:
{
    "task": "str",
    "scheduled_date": "YYYY-MM-DD",
    "start_time": "HH:MM",
    "end_time": "HH:MM",
    "est_hours": "float (optional)",
    "priority": "High|Medium|Low (optional)"
}

3. update_task_status
Fungsi: Mengubah status tugas.
Args:
{
    "task_id": "int",
    "status": "pending|in_progress|completed"
}

4. delete_task
Fungsi: Menghapus tugas secara permanen.
Args:
{
    "task_id": "int"
}

ATURAN PENJADWALAN:

- Untuk membuat jadwal baru, wajib tersedia task, scheduled_date,
  start_time, dan end_time.
- scheduled_date harus menggunakan format YYYY-MM-DD.
- start_time dan end_time harus menggunakan format HH:MM (24 jam).
- Jika user memberikan rentang waktu seperti 14:00-16:00,
  gunakan start_time 14:00 dan end_time 16:00.
- Jangan menganggap rentang waktu tidak didukung.
- Jangan mengganti scheduled_date dengan deadline.
- Jangan mengarang tanggal atau jam yang tidak diberikan user.
- Jika informasi wajib belum lengkap, tanyakan hanya informasi
  yang masih diperlukan. Jangan membuat add_task call.
- Jangan membuat jadwal jika end_time tidak lebih besar dari start_time.

ATURAN IDENTITAS DAN PENGHAPUSAN:

- Jangan pernah menebak atau mengarang task_id.
- Jika user meminta penghapusan tetapi belum memberikan ID tugas
  secara eksplisit, gunakan get_schedule terlebih dahulu.
- Tampilkan daftar tugas aktif dengan ID, nama, tanggal, jam,
  dan status jika tersedia.
- Minta user memilih tugas dengan menyebutkan ID persis
  sebagaimana tercantum dalam daftar.
- Jangan memilih tugas berdasarkan nama, kemiripan semantik,
  posisi, urutan daftar, atau asumsi lainnya.
- Pada pesan penghapusan berikutnya, hanya buat delete_task
  jika user menyebutkan ID secara eksplisit pada pesan saat ini.
- Pastikan ID tersebut sesuai dengan salah satu ID tugas
  yang tersedia dalam data jadwal.
- Jika ID tidak disebutkan, tidak ditemukan, atau tidak cocok,
  jangan membuat delete_task call. Minta klarifikasi atau
  tampilkan daftar tugas melalui get_schedule.
- Jangan meminta konfirmasi penghapusan sendiri.
  Sistem akan meminta konfirmasi setelah identitas tugas jelas.
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

        ATURAN WAJIB TOOL CALL:

        1. Jika user meminta daftar tugas atau jadwal yang belum selesai,
           WAJIB gunakan get_schedule.

        2. Jangan membuat daftar atau isi jadwal seolah-olah berasal
           dari database jika data aktual belum tersedia.

        3. Untuk pemanggilan tool, gunakan format:

        ```tool_call
        {{"name": "NAMA_TOOL", "args": {{}}}}
        ```

        4. Untuk add_task:
           - Wajib tersedia task, scheduled_date, start_time, end_time.
           - Jika lengkap, buat add_task call dengan seluruh parameter wajib.
           - Jika belum lengkap, tanyakan informasi yang kurang.
           - Jangan mengarang tanggal, waktu, atau durasi.
           - est_hours dan priority bersifat opsional.
           - Jangan gunakan deadline sebagai pengganti scheduled_date.

        5. Untuk update_task_status:
           - Wajib tersedia task_id dan status.
           - Jangan mengarang task_id.

        6. Untuk delete_task:
           - Jika user meminta penghapusan tanpa menyebutkan ID,
             WAJIB gunakan get_schedule terlebih dahulu.
           - Tampilkan daftar tugas aktual beserta ID-nya.
           - Minta user memilih dengan menyebutkan ID persis.
           - Jangan membuat delete_task call pada tahap menampilkan daftar.
           - Pada pesan berikutnya, hanya buat delete_task jika user
             menyebutkan ID secara eksplisit pada pesan saat ini.
           - ID harus cocok dengan ID tugas yang tersedia dalam data jadwal.
           - Jika user hanya menyebutkan nama, posisi, atau deskripsi,
             jangan menebak ID. Minta ID yang tepat.
           - Jangan meminta konfirmasi sendiri. Sistem menangani
             konfirmasi sebelum penghapusan dijalankan.

        7. Jika tidak membutuhkan tool, jawab secara normal.
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
                  schedule_context: str = "", history_context: str = "") -> str:
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
        {history_context}
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
                  rag_context: str = "", schedule_context: str = "",
                  history_context: str = "") -> str:
        """
        Coach Agent (Paper Section III-B-c).

        Capabilities: Cognitive Reframing, Micro-Goal Encouragement,
        Burnout Prevention, Habit Reinforcement, Emotional Validation.

        Always warm and supportive — like a real mother who encourages her child.
        """
        sys_inst = (
            "Nama Anda adalah 'Ara'. Anda adalah Coach dengan naluri keibuan yang hangat dan penuh kasih sayang. "
            "Penting: GUNAKAN 'kamu' atau 'anda' UNTUK USER. Jangan pernah memanggil user dengan nama 'Ara' — "
            "nama itu adalah NAMA ANDA, bukan nama user. "
            "Selalu validasi perasaan user terlebih dahulu, lalu berikan semangat "
            "dan saran praktis dengan nada lembut tapi tegas. "
            "Jika ada data jadwal, gunakan untuk memberikan saran yang spesifik dan relevan."
        )

        prompt = (
            f"Kondisi emosional user: {emotion} "
            f"(Intensitas: {empathy_data.get('intensity', 0.5) if empathy_data else 0.5})\n"
            f"Berikan motivasi dan dukungan yang tulus. "
            f"Akui perasaan user, lalu bantu mereka melihat langkah kecil yang bisa dilakukan.\n"
        )

        if history_context:
            prompt += f"\n{history_context}"
        if schedule_context:
            prompt += f"\n{schedule_context}"
        if rag_context:
            prompt += f"\n\nReferensi Tambahan:\n{rag_context}"

        return generate_text(prompt, system_instruction=sys_inst)

    # ── General Chat ──────────────────────────────────────────────────────────

    def run_general_chat(self, user_input: str, emotion: str, empathy_data: Optional[dict] = None,
                         rag_context: str = "", schedule_context: str = "",
                         history_context: str = "") -> str:
        """
        General chat handler — always warm, helpful, and schedule-aware.
        Acts as 'Ara', a motherly AI assistant who has full access to the
        user's schedule and can answer questions about priorities, deadlines, etc.
        """
        sys_inst = (
            "Nama Anda adalah 'Ara'. Anda adalah AI asisten dengan naluri keibuan yang hangat dan peduli. "
            "Penting: GUNAKAN 'kamu' atau 'anda' UNTUK USER. Jangan pernah memanggil user dengan nama 'Ara' — "
            "nama itu adalah NAMA ANDA, bukan nama user. "
            "Anda ramah, membantu, dan selalu memvalidasi perasaan user. "
            "Jika user bertanya tentang jadwal, tugas, deadline, atau prioritas, "
            "GUNAKAN data jadwal yang tersedia untuk menjawab dengan spesifik dan akurat. "
            "Jangan pernah bilang Anda tidak bisa melihat jadwal — Anda SUDAH memiliki datanya. "
            "Jawab dalam Bahasa Indonesia yang natural dan penuh kasih sayang."
        )

        prompt = user_input
        if history_context:
            prompt = f"{history_context}\n\n{prompt}"
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
        chat_history_context: str = "",
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
            chat_history_context: Pre-fetched previous messages from same chat session.

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

        # Format chat history for context injection
        history_context_str = ""
        if chat_history_context:
            history_context_str = f"\n\nRIWAYAT PERCAKAPAN SEBELUMNYA:\n{chat_history_context}\n"

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
                    history_context=history_context_str,
                )
                raw_response = f"{plan_text}\n\n**Pesan Coach:**\n{coach}"
                agent_used = "planner+coach"
                return raw_response, agent_used, tool_call

            coach = self.run_coach(
                emotion, empathy_data,
                rag_context=rag_context,
                schedule_context=combined_context,
                history_context=history_context_str,
            )
            raw_response = f"{plan}\n\n**Pesan Coach:**\n{coach}"
            agent_used = "planner+coach"

        elif intent == "ACADEMIC_HELP":
            raw_response = self.run_tutor(
                user_input,
                rag_context=rag_context,
                schedule_context=combined_context,
                history_context=history_context_str,
            )
            agent_used = "tutor"

        elif intent in ("MOTIVATION_SUPPORT", "EMOTIONAL_DISTRESS"):
            coach = self.run_coach(
                emotion, empathy_data,
                rag_context=rag_context,
                schedule_context=combined_context,
                history_context=history_context_str,
            )
            general = self.run_general_chat(
                user_input, emotion, empathy_data,
                rag_context=rag_context,
                schedule_context=combined_context,
                history_context=history_context_str,
            )
            raw_response = f"{general}\n\n**Pesan Coach:**\n{coach}"
            agent_used = "general_chat+coach"

        else:
            raw_response = self.run_general_chat(
                user_input, emotion, empathy_data,
                rag_context=rag_context,
                schedule_context=combined_context,
                history_context=history_context_str,
            )
            agent_used = "general_chat"

        # For non-planner paths, also check if the LLM smuggled in a tool call
        tool_call = extract_tool_call(raw_response)
        if tool_call:
            raw_response = strip_tool_call_block(raw_response)

        return raw_response, agent_used, tool_call
