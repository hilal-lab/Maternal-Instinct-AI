import json
from core.llm_client import generate_text

class ExecutiveAgent:
    def analyze_intent(self, user_input):
        """
        Menganalisis Niat (Intent) dan Emosi menggunakan LLM (Gemini).
        Output berupa Tuple: (intent, emotion)
        """
        
        # 1. Konstruksi Prompt Klasifikasi
        prompt = f"""
        Tugas: Analisis teks input user berikut dan tentukan INTENT dan EMOTION.
        
        INPUT USER: "{user_input}"
        
        ATURAN KLASIFIKASI:
        1. INTENT:
           - "TASK_OPS": Jika user meminta pembuatan jadwal, rencana, to-do list, atau strategi pengerjaan tugas.
           - "GENERAL_CHAT": Jika user hanya curhat, menyapa, bertanya umum, atau diskusi santai.
           
        2. EMOTION:
           - "STRESS": Jika terdeteksi nada cemas, panik, lelah, putus asa, marah, atau tertekan.
           - "NEUTRAL": Jika nada bicara santai, datar, atau positif.
           
        FORMAT OUTPUT (Wajib JSON):
        {{
            "intent": "...",
            "emotion": "..."
        }}
        """
        
        # 2. Panggil LLM
        # Minta Gemini bertindak sebagai Psikolog Analis Data
        llm_response = generate_text(
            prompt, 
            system_instruction="Anda adalah AI Orchestrator yang ahli dalam analisis sentimen dan klasifikasi niat. Keluarkan hanya JSON valid."
        )
        
        # 3. Parsing JSON (Cleaning & Error Handling)
        try:
            # Membersihkan format markdown jika Gemini memberikan ```json ... ```
            cleaned_response = llm_response.replace("```json", "").replace("```", "").strip()
            
            # Konversi string ke Dictionary Python
            data = json.loads(cleaned_response)
            
            intent = data.get("intent", "GENERAL_CHAT")
            emotion = data.get("emotion", "NEUTRAL")
            
            return intent, emotion

        except (json.JSONDecodeError, AttributeError):
            # 4. Fallback Mechanism (Jaring Pengaman)
            # Jika LLM gagal generate JSON atau API error, kembali ke Keyword Matching agar program TIDAK CRASH saat demo.
            return self._fallback_logic(user_input)

    def _fallback_logic(self, user_input):
        """Logika cadangan (Keyword Matching) jika API bermasalah."""
        task_keywords = ["jadwal", "tugas", "rencana", "plan", "atur", "deadline", "schedule", "buatkan"]
        neg_keywords = ["stres", "capek", "pusing", "takut", "panik", "nyerah", "mati", "lelah", "hancur", "berat"]
        
        intent = "TASK_OPS" if any(x in user_input.lower() for x in task_keywords) else "GENERAL_CHAT"
        emotion = "STRESS" if any(x in user_input.lower() for x in neg_keywords) else "NEUTRAL"
        
        return intent, emotion