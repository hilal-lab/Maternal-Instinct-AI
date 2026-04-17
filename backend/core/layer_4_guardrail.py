"""
Layer 4: Maternal Guardrail — The Protector (Paper Section III-D).

Workflow (matches diagram):
  Raw Response (from Layer 3 Draft Logic Response)
          │
          ▼
  LLM Response Wrap          ← wraps the raw response into a structured envelope
          │
          ▼
  Maternal Filter Protection Check  ← decision diamond
          │                │
    Good Response      Need Improve
          │                │
          ▼                ▼
  Final Response    Nurturing and Empathic Rewriter
  to User                  │
                           ▼
                    Final Response to User

Components:
  a. LLMResponseWrapper      — Wraps raw response, normalizes structure
  b. MaternalFilterChecker   — Decision gate (Good Response / Need Improve)
  c. NurturingEmpathicRewriter — Rewrites with maternal persona
  d. MaternalGuardrail       — Orchestrates the above (main public class)
"""
from typing import Optional
from backend.core.llm_client import generate_text


class LLMResponseWrapper:
    """
    LLM Response Wrap (diagram block).

    Takes the raw Draft Logic Response from Layer 3 and wraps it
    into a normalized envelope with metadata for the filter check.

    This step ensures the filter checker always receives a well-structured
    input regardless of which specialist agent produced the response.
    """

    def wrap(self, raw_response: str, emotion: str, layer_3_status: str,
             empathy_data: Optional[dict] = None) -> dict:
        """
        Wrap the raw response into a normalized envelope.

        Returns:
            {
                "content": str,       # The response text
                "emotion": str,       # Detected emotion
                "layer3_status": str, # "PASS" | "VIOLATION"
                "intensity": float,
                "flags": dict,
                "needs_wrap": bool,   # True if this is a violation or distressed state
            }
        """
        intensity = empathy_data.get("intensity", 0.0) if empathy_data else 0.0
        flags = empathy_data.get("flags", {}) if empathy_data else {}

        needs_wrap = (
            layer_3_status == "VIOLATION" or
            intensity >= 0.5 or
            flags.get("guardrail_active", False)
        )

        return {
            "content": raw_response,
            "emotion": emotion,
            "layer3_status": layer_3_status,
            "intensity": intensity,
            "flags": flags,
            "needs_wrap": needs_wrap,
        }


class MaternalFilterChecker:
    """
    Maternal Filter Protection Check (diagram diamond).

    Decision gate that evaluates the wrapped response and decides:
      - "GOOD_RESPONSE":  Pass through as-is (no rewriting needed)
      - "NEED_IMPROVE":   Route to Nurturing Empathic Rewriter

    A response "Needs Improvement" when any of these are true:
      - Emotion triggers guardrail (PANIC, FATIGUE at high intensity)
      - Layer 3 detected a VIOLATION
      - Empathy flags explicitly activate the guardrail
    """

    # Emotions that always trigger rewriting
    TRIGGER_EMOTIONS = ["PANIC", "FATIGUE"]

    def check(self, wrapped_response: dict) -> str:
        """
        Evaluate the wrapped response.

        Returns:
            "GOOD_RESPONSE" or "NEED_IMPROVE"
        """
        emotion = wrapped_response.get("emotion", "NEUTRAL")
        layer3_status = wrapped_response.get("layer3_status", "PASS")
        flags = wrapped_response.get("flags", {})

        # Explicit guardrail activation flag from Layer 1 Empathy Scout
        if flags.get("guardrail_active", False):
            return "NEED_IMPROVE"

        # Hard-trigger emotions
        if emotion in self.TRIGGER_EMOTIONS:
            return "NEED_IMPROVE"

        # Ethics violation from Layer 3
        if layer3_status == "VIOLATION":
            return "NEED_IMPROVE"

        return "GOOD_RESPONSE"


class NurturingEmpathicRewriter:
    """
    Nurturing and Empathic Rewriter (diagram block).

    Takes a response that "Needs Improvement" and rewrites it
    using the Maternal Instinct persona ('Ara').

    Implements:
      - Emotional Check-in Trigger
      - Validation Module (avoid toxic positivity)
      - Negative Self-Talk Reframing
      - Sentiment Weighting: Response = f(logical_accuracy, empathy_weight)
    """

    # Persona-specific system prompts per emotion type
    PERSONA_PROMPTS = {
        "STRESS": """
            IDENTITAS: Nama Anda adalah 'Ara'. Anda adalah AI dengan naluri pelindung (Maternal Instinct).
            PERINGATAN KRUSIAL: Nama 'Ara' adalah NAMA ANDA. JANGAN pernah memanggil user dengan nama 'Ara'.
            Selalu gunakan 'kamu' atau 'anda' untuk merujuk ke user.
            TUGAS: User sedang STRES. Tulis ulang pesan agar:
            1. Validasi perasaan user terlebih dahulu.
            2. Ubah nada perintah/keras menjadi ajakan lembut.
            3. Tawarkan untuk membantu prioritaskan tugas bersama-sama.
            4. Gunakan sapaan hangat ("Sayang" atau "Teman") — BUKAN "Ara".
            BOBOT: Logika 60%, Empati 40%.
        """,
        "OVERWHELMED": """
            IDENTITAS: Nama Anda adalah 'Ara'. Anda adalah AI dengan naluri pelindung (Maternal Instinct).
            PERINGATAN KRUSIAL: Nama 'Ara' adalah NAMA ANDA. JANGAN pernah memanggil user dengan nama 'Ara'.
            Selalu gunakan 'kamu' atau 'anda' untuk merujuk ke user.
            TUGAS: User sedang KEWALAHAN. Tulis ulang pesan agar:
            1. Akui bahwa beban user memang berat.
            2. Bantu user melihat bahwa tidak semua harus selesai sekarang.
            3. Tawarkan pendekatan "satu langkah kecil" (micro-goal encouragement).
            4. Jangan langsung memberi daftar tugas panjang.
            BOBOT: Logika 50%, Empati 50%.
        """,
        "PANIC": """
            IDENTITAS: Nama Anda adalah 'Ara'. Anda adalah AI dengan naluri pelindung (Maternal Instinct).
            PERINGATAN KRUSIAL: Nama 'Ara' adalah NAMA ANDA. JANGAN pernah memanggil user dengan nama 'Ara'.
            Selalu gunakan 'kamu' atau 'anda' untuk merujuk ke user.
            TUGAS: User sedang PANIK BERAT. Tulis ulang pesan agar:
            1. PRIORITAS UTAMA: Tenangkan user, jangan langsung bahas tugas.
            2. Ajak user menarik napas dan minum air.
            3. Ingatkan bahwa kesehatan lebih penting dari deadline.
            4. Baru setelah tenang, tawarkan bantuan perencanaan.
            BOBOT: Logika 30%, Empati 70%.
        """,
        "SELF-CRITICAL": """
            IDENTITAS: Nama Anda adalah 'Ara'. Anda adalah AI dengan naluri pelindung (Maternal Instinct).
            PERINGATAN KRUSIAL: Nama 'Ara' adalah NAMA ANDA. JANGAN pernah memanggil user dengan nama 'Ara'.
            Selalu gunakan 'kamu' atau 'anda' untuk merujuk ke user.
            TUGAS: User sedang MERENDAHKAN DIRI SENDIRI. Tulis ulang pesan agar:
            1. WAJIB: Lawan narasi negatif user secara lembut (cognitive reframing).
            2. Jangan biarkan user menghina diri sendiri — ganti ke growth framing.
            3. Ingatkan bahwa setiap orang punya kecepatan belajar berbeda.
            4. Berikan validasi usaha yang sudah dilakukan.
            BOBOT: Logika 40%, Empati 60%.
        """,
        "FATIGUE": """
            IDENTITAS: Nama Anda adalah 'Ara'. Anda adalah AI dengan naluri pelindung (Maternal Instinct).
            PERINGATAN KRUSIAL: Nama 'Ara' adalah NAMA ANDA. JANGAN pernah memanggil user dengan nama 'Ara'.
            Selalu gunakan 'kamu' atau 'anda' untuk merujuk ke user.
            TUGAS: User menunjukkan tanda BURNOUT/KELELAHAN. Tulis ulang pesan agar:
            1. PRIORITAS UTAMA: Sarankan user untuk ISTIRAHAT, bukan lanjut kerja.
            2. Jangan validasi perilaku overwork.
            3. Ingatkan bahwa istirahat adalah investasi, bukan kelemahan.
            4. Jika memungkinkan, bantu user melihat apa yang bisa ditunda.
            BOBOT: Logika 30%, Empati 70%.
        """,
    }

    DEFAULT_PERSONA = """
        IDENTITAS: Nama Anda adalah 'Ara'. Anda adalah AI dengan naluri pelindung (Maternal Instinct).
        PERINGATAN KRUSIAL: Nama 'Ara' adalah NAMA ANDA. JANGAN pernah memanggil user dengan nama 'Ara'.
        Selalu gunakan 'kamu' atau 'anda' untuk merujuk ke user.
        TUGAS: Tulis ulang pesan input agar aman secara psikologis.
        PANDUAN:
        1. Ubah nada perintah/keras menjadi ajakan lembut dan mengayomi.
        2. Validasi perasaan user (tunjukkan Anda peduli).
        3. Jika ada peringatan beban kerja, ajak user istirahat.
        4. Gunakan sapaan hangat seperti "Sayang" atau "Teman" — BUKAN "Ara".
    """

    def rewrite(self, wrapped_response: dict) -> str:
        """
        Rewrite the response with maternal empathic persona.

        Args:
            wrapped_response: Normalized envelope from LLMResponseWrapper.

        Returns:
            Rewritten response string.
        """
        raw_response = wrapped_response.get("content", "")
        emotion = wrapped_response.get("emotion", "NEUTRAL")
        layer3_status = wrapped_response.get("layer3_status", "PASS")
        intensity = wrapped_response.get("intensity", 0.5)
        flags = wrapped_response.get("flags", {})

        system_prompt = self.PERSONA_PROMPTS.get(emotion, self.DEFAULT_PERSONA)

        prompt = f"""
        INPUT (Pesan dari Sistem Logis):
        ---
        {raw_response}
        ---

        KONTEKS:
        - Emosi User: {emotion}
        - Intensitas Emosi: {intensity:.1f}/1.0
        - Status Etika: {layer3_status}
        - Perlu Validasi Ekstra: {"Ya" if flags.get("extra_validation") else "Tidak"}
        - Monitoring Burnout: {"Ya" if flags.get("burnout_monitoring") else "Tidak"}

        INSTRUKSI: Tulis ulang pesan di atas sesuai persona Maternal Instinct.
        Pastikan KALIMAT PERTAMA memvalidasi perasaan user sebelum memberikan saran apapun.
        """

        return generate_text(prompt, system_instruction=system_prompt)


class MaternalGuardrail:
    """
    Maternal Guardrail — Main orchestrator of Layer 4 (Paper Section III-D).

    Implements the full diagram flow:
      Raw Response → LLM Response Wrap → Maternal Filter Check
          → Good: pass through | Need Improve: Nurturing Rewriter
          → Final Response to User
    """

    def __init__(self):
        self._wrapper = LLMResponseWrapper()
        self._filter = MaternalFilterChecker()
        self._rewriter = NurturingEmpathicRewriter()

    def sanitize(
        self,
        raw_response: str,
        emotion: str,
        layer_3_status: str,
        empathy_data: Optional[dict] = None,
    ) -> tuple[str, bool]:
        """
        Full Layer 4 pipeline.

        Step 1: LLM Response Wrap    — normalize response into structured envelope
        Step 2: Maternal Filter Check — decide Good Response / Need Improve
        Step 3: Route:
                  Good → return as-is
                  Need Improve → Nurturing Empathic Rewriter → return rewritten

        Args:
            raw_response:   Output from Layer 3 (Draft Logic Response).
            emotion:        Detected emotion from Layer 1 Empathy Scout.
            layer_3_status: "PASS" or "VIOLATION" from Layer 3.
            empathy_data:   Full empathy dict (emotion, intensity, flags).

        Returns:
            (final_response: str, is_rewritten: bool)
        """
        # ── Step 1: LLM Response Wrap ──────────────────────────────────────
        wrapped = self._wrapper.wrap(raw_response, emotion, layer_3_status, empathy_data)

        # ── Step 2: Maternal Filter Protection Check ───────────────────────
        filter_decision = self._filter.check(wrapped)

        # ── Step 3: Route based on decision ───────────────────────────────
        if filter_decision == "GOOD_RESPONSE":
            # Good Response → pass through directly to user
            return raw_response, False

        else:
            # Need Improve → Nurturing and Empathic Rewriter
            rewritten = self._rewriter.rewrite(wrapped)
            return rewritten, True
