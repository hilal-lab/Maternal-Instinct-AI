import json
from backend.core.llm_client import generate_text
from backend.core.data_loader import SyntheticDataset

class EmpathyScout:
    """
    Empathy Scout — Lightweight affective signal detector (Paper Section III-A-b).
    Detects user emotional state and produces:
      1. Sentiment Classification (NEUTRAL, STRESS, OVERWHELMED, PANIC, SELF-CRITICAL, FATIGUE)
      2. Emotion Intensity Score (0.0 – 1.0)
      3. Intervention Flags (soft_tone, slower_pacing, extra_validation, burnout_monitoring)
    """

    EMOTION_LABELS = ["NEUTRAL", "STRESS", "OVERWHELMED", "PANIC", "SELF-CRITICAL", "FATIGUE"]

    INTENSITY_MAP = {
        "NEUTRAL": 0.0,
        "STRESS": 0.4,
        "OVERWHELMED": 0.6,
        "PANIC": 0.9,
        "SELF-CRITICAL": 0.5,
        "FATIGUE": 0.7,
    }

    DISTRESS_THRESHOLD = 0.5

    def analyze(self, user_input: str, few_shot_context: str = "") -> dict:
        """
        Analyze user input for affective signals.
        Returns: { emotion, intensity, flags }
        """
        prompt = f"""
        Tugas: Analisis kondisi emosional user dari teks berikut.

        INPUT USER: "{user_input}"

        {f"CONTOH REFERENSI:{chr(10)}{few_shot_context}" if few_shot_context else ""}

        KLASIFIKASI EMOSI (pilih SATU):
        - NEUTRAL: Nada bicara santai, datar, atau positif.
        - STRESS: Terdeteksi nada cemas, tertekan, atau khawatir.
        - OVERWHELMED: Kewalahan, merasa terlalu banyak beban.
        - PANIC: Panik tinggi, putus asa, ingin menyerah.
        - SELF-CRITICAL: Merendahkan diri sendiri, merasa bodoh/tidak berguna.
        - FATIGUE: Kelelahan fisik/mental berkepanjangan, burnout.

        SKALA INTENSITAS: 0.0 (tenang) hingga 1.0 (krisis).

        FORMAT OUTPUT (Wajib JSON):
        {{
            "emotion": "...",
            "intensity": 0.0
        }}
        """

        response = generate_text(
            prompt,
            system_instruction="Anda adalah Empathy Scout AI, ahli deteksi sinyal afektif. Keluarkan hanya JSON valid."
        )

        try:
            cleaned = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned)
            emotion = data.get("emotion", "NEUTRAL")
            intensity = float(data.get("intensity", 0.0))

            # Validate emotion label
            if emotion not in self.EMOTION_LABELS:
                emotion = "NEUTRAL"

            # Clamp intensity
            intensity = max(0.0, min(1.0, intensity))

        except (json.JSONDecodeError, AttributeError, ValueError):
            emotion, intensity = self._fallback(user_input)

        # Generate intervention flags
        flags = self._generate_flags(emotion, intensity)

        return {
            "emotion": emotion,
            "intensity": intensity,
            "flags": flags,
        }

    def _fallback(self, user_input: str) -> tuple:
        """Keyword-based fallback when LLM fails."""
        text = user_input.lower()

        panic_kw = ["panik", "nyerah", "mau mati", "panic", "give up", "mau pingsan", "tidak kuat"]
        fatigue_kw = ["burnout", "begadang", "capek", "lelah", "exhausted", "energi habis", "tidak tidur"]
        self_crit_kw = ["bodoh", "gagal", "tidak berguna", "stupid", "failure", "useless", "tidak kompeten"]
        overwhelm_kw = ["overwhelm", "kewalahan", "terlalu banyak", "tidak bisa", "drown"]
        stress_kw = ["stres", "stress", "cemas", "takut", "pusing", "berat", "tertekan", "anxious"]

        if any(k in text for k in panic_kw):
            return "PANIC", 0.9
        elif any(k in text for k in fatigue_kw):
            return "FATIGUE", 0.7
        elif any(k in text for k in self_crit_kw):
            return "SELF-CRITICAL", 0.5
        elif any(k in text for k in overwhelm_kw):
            return "OVERWHELMED", 0.6
        elif any(k in text for k in stress_kw):
            return "STRESS", 0.4
        else:
            return "NEUTRAL", 0.0

    def _generate_flags(self, emotion: str, intensity: float) -> dict:
        """Generate intervention flags based on distress level (Paper Section III-A-b-iii)."""
        is_distressed = intensity >= self.DISTRESS_THRESHOLD

        return {
            "soft_tone": is_distressed,
            "slower_pacing": intensity >= 0.7,
            "extra_validation": emotion in ("PANIC", "SELF-CRITICAL"),
            "burnout_monitoring": emotion == "FATIGUE",
            "guardrail_active": is_distressed or emotion != "NEUTRAL",
        }


class ExecutiveAgent:
    """
    Executive Agent / Orchestrator (Paper Section III-A-a).
    Acts as a meta-controller:
      1. Intent Classification
      2. Task Decomposition
      3. Agent Routing Strategy
    Now includes the Empathy Scout as a modular sub-component.
    """

    def __init__(self):
        self.empathy_scout = EmpathyScout()
        self._dataset = None

    def _get_dataset(self):
        if self._dataset is None:
            try:
                self._dataset = SyntheticDataset()
            except Exception:
                self._dataset = None
        return self._dataset

    def analyze_intent(self, user_input: str):
        """
        Full analysis pipeline:
        1. Empathy Scout detects emotion
        2. Executive Agent classifies intent
        Returns: (intent, emotion, empathy_data)
        """
        # --- EMPATHY SCOUT ---
        dataset = self._get_dataset()
        few_shot = ""
        if dataset:
            # Provide representative examples as few-shot context
            few_shot = (
                dataset.get_few_shot_examples("STRESS", 2) + "\n" +
                dataset.get_few_shot_examples("BURNOUT", 2) + "\n" +
                dataset.get_few_shot_examples("SELF_DEPRECATION", 2) + "\n" +
                dataset.get_few_shot_examples("NEUTRAL", 2)
            )

        empathy_data = self.empathy_scout.analyze(user_input, few_shot)

        # --- INTENT CLASSIFICATION ---
        intent = self._classify_intent(user_input)

        return intent, empathy_data["emotion"], empathy_data

    def _classify_intent(self, user_input: str) -> str:
        """Classify user intent via LLM with keyword fallback."""
        prompt = f"""
        Tugas: Klasifikasikan INTENT dari teks input user berikut.

        INPUT USER: "{user_input}"

        PILIHAN INTENT:
        - "TASK_OPS": User meminta pembuatan jadwal, rencana, to-do list, strategi pengerjaan tugas, cek deadline.
        - "ACADEMIC_HELP": User bertanya tentang materi akademik, konsep, atau butuh penjelasan.
        - "MOTIVATION_SUPPORT": User butuh motivasi, semangat, atau dukungan emosional.
        - "EMOTIONAL_DISTRESS": User sedang dalam kondisi emosional berat (curhat, mengeluh, putus asa).
        - "GENERAL_CHAT": Sapaan, diskusi santai, pertanyaan umum.

        FORMAT OUTPUT (Wajib JSON):
        {{
            "intent": "..."
        }}
        """

        response = generate_text(
            prompt,
            system_instruction="Anda adalah AI Orchestrator ahli klasifikasi niat. Keluarkan hanya JSON valid."
        )

        try:
            cleaned = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned)
            intent = data.get("intent", "GENERAL_CHAT")

            valid_intents = {"TASK_OPS", "ACADEMIC_HELP", "MOTIVATION_SUPPORT", "EMOTIONAL_DISTRESS", "GENERAL_CHAT"}
            if intent not in valid_intents:
                intent = "GENERAL_CHAT"
            return intent

        except (json.JSONDecodeError, AttributeError):
            return self._fallback_intent(user_input)

    def _fallback_intent(self, user_input: str) -> str:
        """Keyword-based intent fallback."""
        text = user_input.lower()

        task_kw = ["jadwal", "tugas", "rencana", "plan", "atur", "deadline", "schedule", "buatkan", "to-do", "cek"]
        academic_kw = ["jelaskan", "apa itu", "bagaimana cara", "materi", "rumus", "teori"]
        emotion_kw = ["stres", "capek", "nyerah", "panik", "takut", "lelah", "hancur", "berat",
                       "bodoh", "gagal", "tidak berguna", "burnout", "overwhelm"]

        if any(k in text for k in task_kw):
            return "TASK_OPS"
        elif any(k in text for k in emotion_kw):
            return "EMOTIONAL_DISTRESS"
        elif any(k in text for k in academic_kw):
            return "ACADEMIC_HELP"
        else:
            return "GENERAL_CHAT"