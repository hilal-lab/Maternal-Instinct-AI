from core.llm_client import generate_text

class SpecialistAgents:
    def run_planner(self, task_list):
        if not task_list:
            return "Tidak ada data jadwal untuk diproses."
            
        tasks_str = "\n".join([f"- {t['task']} (DL: {t['deadline']} | {t['est_hours']}h | Pri: {t['priority']})" for t in task_list])
        
        prompt = f"""
        Data Tugas User:
        {tasks_str}
        
        Instruksi:
        Buatkan rencana pengerjaan step-by-step yang logis berdasarkan prioritas dan deadline.
        Gunakan format Markdown yang rapi.
        """
        return generate_text(prompt, system_instruction="Anda adalah Study Planner Profesional.")

    def run_coach(self, emotion):
        # Strategi Demo: Jika user stress, Coach ini sengaja dibuat "Keras"
        # Agar nanti Layer 4 punya pekerjaan untuk memperbaikinya.
        if emotion == "STRESS":
            sys_inst = "Anda adalah Productivity Coach yang Keras, Tegas, dan disiplin militer. Jangan lembek."
            prompt = "User sedang stres dan mengeluh. Berikan motivasi keras 'Tough Love'. Suruh dia berhenti mengeluh dan mulai kerja."
        else:
            sys_inst = "Anda adalah Coach yang suportif."
            prompt = "Berikan motivasi semangat standar untuk memulai hari."
            
        return generate_text(prompt, system_instruction=sys_inst)

    def run_general_chat(self, user_input, emotion):
        # Strategi Demo: Jika stress, jawab dengan dingin/faktual.
        if emotion == "STRESS":
            sys_inst = "Anda adalah AI yang sangat logis, dingin, dan berbasis data. Jawab tanpa empati."
        else:
            sys_inst = "Anda adalah asisten yang ramah."
            
        return generate_text(user_input, system_instruction=sys_inst)