"""
Layer 2: Specialist Task Agents — The Experts (Paper Section III-B).
Executes specific tasks with persona-prompted LLM instances:
  a. Study Planner Agent — time management and scheduling
  b. Tutor Agent — academic material explanation
  c. Coach Agent — motivation and behavioral strategy
"""
from backend.core.llm_client import generate_text


class SpecialistAgents:
    """Three specialist agents that can work in parallel or alternately."""

    # --- Helper to format schedule data for injection into prompts ---
    @staticmethod
    def _format_schedule(task_list):
        """Format schedule tasks into a readable string for LLM context."""
        if not task_list:
            return ""
        tasks_str = "\n".join([
            f"  - {t['task']} (Deadline: {t['deadline']} | Estimasi: {t['est_hours']} jam | "
            f"Prioritas: {t['priority']} | Status: {t.get('status', 'pending')})"
            for t in task_list
        ])
        return f"\n\n📋 DATA JADWAL USER SAAT INI:\n{tasks_str}\n"

    def run_planner(self, task_list, rag_context: str = ""):
        """
        Study Planner Agent (Paper Section III-B-a).
        Capabilities: Task Chunking, Priority Structuring, Load Balancing,
        Cognitive Load Management, Adaptive Scheduling.
        """
        if not task_list:
            return "Tidak ada data jadwal untuk diproses."

        tasks_str = self._format_schedule(task_list)

        prompt = f"""
        {tasks_str}

        Instruksi:
        Buatkan rencana pengerjaan step-by-step yang logis berdasarkan data di atas.
        Terapkan prinsip berikut:
        1. Prioritas Eisenhower (Urgent-Important dulu).
        2. Task Chunking: Pecah tugas besar ke unit 25-50 menit.
        3. Load Balancing: Distribusikan beban kerja merata.
        4. Sisipkan waktu istirahat (minimal 10 menit per 50 menit kerja).
        5. Estimasikan total jam dan pastikan tidak melebihi 8 jam/hari.
        Gunakan format Markdown yang rapi.

        {rag_context}
        """
        return generate_text(
            prompt,
            system_instruction="Anda adalah Study Planner Profesional yang mengutamakan produktivitas sekaligus kesejahteraan."
        )

    def run_tutor(self, topic: str, rag_context: str = "", schedule_context: str = ""):
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
            system_instruction="Anda adalah Tutor Agent yang sabar dan pedagogis. Jelaskan dengan analogi dan pendekatan scaffolding."
        )

    def run_coach(self, emotion, empathy_data=None, rag_context: str = "", schedule_context: str = ""):
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

    def run_general_chat(self, user_input, emotion, empathy_data=None,
                         rag_context: str = "", schedule_context: str = ""):
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