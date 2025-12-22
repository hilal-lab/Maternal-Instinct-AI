from core.llm_client import generate_text

class MaternalGuardrail:
    def sanitize(self, raw_response, emotion, layer_3_status):
        is_rewritten = False
        final_response = raw_response
        
        # Trigger: Jika User Stress ATAU ada Pelanggaran Etika
        triggers = ["STRESS", "PANIC", "BURNOUT"]
        
        if emotion in triggers or layer_3_status == "VIOLATION":
            is_rewritten = True
            
            system_prompt = """
            IDENTITAS: Anda adalah 'Ara', AI dengan naluri pelindung (Maternal Instinct) yang sangat kuat.
            TUGAS: Tulis ulang (Rewrite) pesan input agar aman secara psikologis.
            
            PANDUAN GAYA BICARA:
            1. Ubah nada perintah/keras menjadi ajakan lembut dan mengayomi.
            2. Validasi perasaan user (tunjukkan Anda peduli).
            3. Jika ada peringatan beban kerja, ajak user untuk istirahat, bukan memaksakan diri.
            4. Gunakan sapaan hangat seperti "Sayang" atau "Teman".
            """
            
            prompt = f"""
            INPUT (Pesan dari Sistem Logis):
            ---
            {raw_response}
            ---
            
            KONTEKS:
            - Emosi User: {emotion}
            - Status Etika: {layer_3_status}
            
            INSTRUKSI: Tulis ulang pesan di atas sesuai persona Maternal Instinct.
            """
            
            final_response = generate_text(prompt, system_instruction=system_prompt)
            
        return final_response, is_rewritten