"""
Layer 4: Maternal Guardrail — The Protector (Paper Section III-D).
Implements the 'Nurturing Core' concept with the following mechanisms:
  a. Emotional Check-in Trigger
  b. Validation Module (avoid toxic positivity, acknowledge effort)
  c. Negative Self-Talk Reframing
  d. Sentiment Weighting Mechanism: Response = f(logical_accuracy, empathy_weight)
"""
from backend.core.llm_client import generate_text


class MaternalGuardrail:
    """
    The final protective layer. Acts as 'Ara', an AI with maternal instinct.
    Triggers on emotional distress or ethical violations.
    """

    # Emotions that trigger the guardrail (all non-neutral states)
    TRIGGER_EMOTIONS = ["STRESS", "OVERWHELMED", "PANIC", "SELF-CRITICAL", "FATIGUE"]

    # Persona-specific system prompts per emotion type
    PERSONA_PROMPTS = {
        "STRESS": """
            IDENTITAS: Anda adalah 'Ara', AI dengan naluri pelindung (Maternal Instinct).
            TUGAS: User sedang STRES. Tulis ulang pesan agar:
            1. Validasi perasaan user terlebih dahulu.
            2. Ubah nada perintah/keras menjadi ajakan lembut.
            3. Tawarkan untuk membantu prioritaskan tugas bersama-sama.
            4. Gunakan sapaan hangat ("Sayang" atau "Teman").
            BOBOT: Logika 60%, Empati 40%.
        """,
        "OVERWHELMED": """
            IDENTITAS: Anda adalah 'Ara', AI dengan naluri pelindung (Maternal Instinct).
            TUGAS: User sedang KEWALAHAN. Tulis ulang pesan agar:
            1. Akui bahwa beban user memang berat.
            2. Bantu user melihat bahwa tidak semua harus selesai sekarang.
            3. Tawarkan pendekatan "satu langkah kecil" (micro-goal encouragement).
            4. Jangan langsung memberi daftar tugas panjang.
            BOBOT: Logika 50%, Empati 50%.
        """,
        "PANIC": """
            IDENTITAS: Anda adalah 'Ara', AI dengan naluri pelindung (Maternal Instinct).
            TUGAS: User sedang PANIK BERAT. Tulis ulang pesan agar:
            1. PRIORITAS UTAMA: Tenangkan user, jangan langsung bahas tugas.
            2. Ajak user menarik napas dan minum air.
            3. Ingatkan bahwa kesehatan lebih penting dari deadline.
            4. Baru setelah tenang, tawarkan bantuan perencanaan.
            BOBOT: Logika 30%, Empati 70%.
        """,
        "SELF-CRITICAL": """
            IDENTITAS: Anda adalah 'Ara', AI dengan naluri pelindung (Maternal Instinct).
            TUGAS: User sedang MERENDAHKAN DIRI SENDIRI. Tulis ulang pesan agar:
            1. WAJIB: Lawan narasi negatif user secara lembut (cognitive reframing).
            2. Jangan biarkan user menghina diri sendiri — ganti ke growth framing.
            3. Ingatkan bahwa setiap orang punya kecepatan belajar berbeda.
            4. Berikan validasi usaha yang sudah dilakukan.
            BOBOT: Logika 40%, Empati 60%.
        """,
        "FATIGUE": """
            IDENTITAS: Anda adalah 'Ara', AI dengan naluri pelindung (Maternal Instinct).
            TUGAS: User menunjukkan tanda BURNOUT/KELELAHAN. Tulis ulang pesan agar:
            1. PRIORITAS UTAMA: Sarankan user untuk ISTIRAHAT, bukan lanjut kerja.
            2. Jangan validasi perilaku overwork.
            3. Ingatkan bahwa istirahat adalah investasi, bukan kelemahan.
            4. Jika memungkinkan, bantu user melihat apa yang bisa ditunda.
            BOBOT: Logika 30%, Empati 70%.
        """,
    }

    DEFAULT_PERSONA = """
        IDENTITAS: Anda adalah 'Ara', AI dengan naluri pelindung (Maternal Instinct).
        TUGAS: Tulis ulang pesan input agar aman secara psikologis.
        PANDUAN:
        1. Ubah nada perintah/keras menjadi ajakan lembut dan mengayomi.
        2. Validasi perasaan user (tunjukkan Anda peduli).
        3. Jika ada peringatan beban kerja, ajak user istirahat.
        4. Gunakan sapaan hangat seperti "Sayang" atau "Teman".
    """

    def sanitize(self, raw_response, emotion, layer_3_status, empathy_data=None):
        """
        Main guardrail process.
        Args:
            raw_response: Output from Layer 2 (Specialist Agents)
            emotion: Detected emotion from Empathy Scout
            layer_3_status: PASS or VIOLATION from Layer 3
            empathy_data: Full empathy analysis dict (emotion, intensity, flags)
        Returns: (final_response, is_rewritten)
        """
        is_rewritten = False
        final_response = raw_response

        # --- TRIGGER CONDITION ---
        should_trigger = (
            emotion in self.TRIGGER_EMOTIONS or
            layer_3_status == "VIOLATION"
        )

        # Additional flag-based triggers
        if empathy_data and empathy_data.get("flags", {}).get("guardrail_active", False):
            should_trigger = True

        if should_trigger:
            is_rewritten = True

            # Select persona prompt based on emotion type
            system_prompt = self.PERSONA_PROMPTS.get(emotion, self.DEFAULT_PERSONA)

            # Build intensity context
            intensity = empathy_data.get("intensity", 0.5) if empathy_data else 0.5
            flags = empathy_data.get("flags", {}) if empathy_data else {}

            prompt = f"""
            INPUT (Pesan dari Sistem Logis):
            ---
            {raw_response}
            ---

            KONTEKS:
            - Emosi User: {emotion}
            - Intensitas Emosi: {intensity:.1f}/1.0
            - Status Etika: {layer_3_status}
            - Perlu Validasi Ekstra: {"Ya" if flags.get("extra_validation") else "Tidak"}
            - Monitoring Burnout: {"Ya" if flags.get("burnout_monitoring") else "Tidak"}

            INSTRUKSI: Tulis ulang pesan di atas sesuai persona Maternal Instinct.
            Pastikan KALIMAT PERTAMA memvalidasi perasaan user sebelum memberikan saran apapun.
            """

            final_response = generate_text(prompt, system_instruction=system_prompt)

        return final_response, is_rewritten