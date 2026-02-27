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

    def run_planner(self, task_list, rag_context: str = ""):
        """
        Study Planner Agent (Paper Section III-B-a).
        Capabilities: Task Chunking, Priority Structuring, Load Balancing,
        Cognitive Load Management, Adaptive Scheduling.
        """
        if not task_list:
            return "Tidak ada data jadwal untuk diproses."

        tasks_str = "\n".join([
            f"- {t['task']} (DL: {t['deadline']} | {t['est_hours']}h | Pri: {t['priority']})"
            for t in task_list
        ])

        prompt = f"""
        Data Tugas User:
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

    def run_tutor(self, topic: str, rag_context: str = ""):
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

        {rag_context}
        """
        return generate_text(
            prompt,
            system_instruction="Anda adalah Tutor Agent yang sabar dan pedagogis. Jelaskan dengan analogi dan pendekatan scaffolding."
        )

    def run_coach(self, emotion, empathy_data=None, rag_context: str = ""):
        """
        Coach Agent (Paper Section III-B-c).
        Capabilities: Cognitive Reframing, Micro-Goal Encouragement,
        Burnout Prevention, Habit Reinforcement, Emotional Validation.

        Strategy: When user is stressed, Coach is intentionally 'harsh' so that
        Layer 4 (Maternal Guardrail) has material to rewrite — demonstrating
        the architecture's value.
        """
        # The Coach's persona changes based on emotion
        # In stressed scenarios, it's deliberately rigid to showcase Layer 4
        distressed_emotions = ["STRESS", "OVERWHELMED", "PANIC", "SELF-CRITICAL", "FATIGUE"]

        if emotion in distressed_emotions:
            sys_inst = (
                "Anda adalah Productivity Coach yang Keras, Tegas, dan disiplin militer. "
                "Jangan lembek. Fokus pada efisiensi dan deadline."
            )
            prompt = (
                "User sedang mengeluh dan tertekan. "
                "Berikan motivasi keras 'Tough Love'. "
                "Suruh dia berhenti mengeluh dan mulai kerja keras. "
                "Tekankan pentingnya deadline dan konsekuensi jika gagal."
            )
        else:
            sys_inst = "Anda adalah Coach yang suportif, hangat, dan memotivasi."
            prompt = "Berikan motivasi semangat yang positif untuk memulai hari."

        if rag_context:
            prompt += f"\n\nReferensi Tambahan:\n{rag_context}"

        return generate_text(prompt, system_instruction=sys_inst)

    def run_general_chat(self, user_input, emotion, empathy_data=None, rag_context: str = ""):
        """
        General chat handler for non-task requests.
        Strategy: When stressed, deliberately cold/factual to showcase
        the Layer 4 rewrite contrast.
        """
        distressed = ["STRESS", "OVERWHELMED", "PANIC", "SELF-CRITICAL", "FATIGUE"]

        if emotion in distressed:
            sys_inst = (
                "Anda adalah AI yang sangat logis, dingin, dan berbasis data. "
                "Jawab tanpa empati, hanya berdasarkan fakta."
            )
        else:
            sys_inst = "Anda adalah asisten yang ramah dan membantu."

        prompt = user_input
        if rag_context:
            prompt += f"\n\nReferensi Tambahan:\n{rag_context}"

        return generate_text(prompt, system_instruction=sys_inst)