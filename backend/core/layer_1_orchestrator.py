"""
Layer 1: Orchestration & Data Retrieval (Paper Section III-A).

Workflow (matches diagram):
  User Prompt
      │
      ▼
  Executive Agent ──► Empathy Scout
      │                    │
      │          (emotion + intensity + flags)
      │                    │
      ▼                    ▼
  Database ──► User Data & Context Retrieval
                    │
                    ▼
              Embedding Model ──► FAISS Vector Database
                                        │
                                        ▼
                                   Task Routing
                                        │
                              (to Layer 2 specialists)

Components:
  a. ExecutiveAgent  — Intent classification + task decomposition + routing
  b. EmpathyScout    — Affective signal detector (emotion, intensity, flags)
  c. ContextRetriever — Pulls DB context (schedule/notes) + FAISS semantic search
"""
import json
import logging
from backend.core.llm_client import generate_text
from backend.core.data_loader import SyntheticDataset

logger = logging.getLogger("backend.core.layer_1_orchestrator")


# ─── Intent → Agent routing map ──────────────────────────────────────────────

ROUTING_MAP = {
    "TASK_OPS":           ["planner", "coach"],
    "ACADEMIC_HELP":      ["tutor"],
    "MOTIVATION_SUPPORT": ["coach", "general_chat"],
    "EMOTIONAL_DISTRESS": ["coach", "general_chat"],
    "GENERAL_CHAT":       ["general_chat"],
}


# ─── Empathy Scout ────────────────────────────────────────────────────────────

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
        "SELF-CRITICAL": 0.3,
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

            if emotion not in self.EMOTION_LABELS:
                emotion = "NEUTRAL"

            intensity = max(0.0, min(1.0, intensity))

        except (json.JSONDecodeError, AttributeError, ValueError):
            emotion, intensity = self._fallback(user_input)

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
            return "SELF-CRITICAL", 0.3
        elif any(k in text for k in overwhelm_kw):
            return "OVERWHELMED", 0.6
        elif any(k in text for k in stress_kw):
            return "STRESS", 0.4
        else:
            return "NEUTRAL", 0.0

    def _generate_flags(self, emotion: str, intensity: float) -> dict:
        """
        Generate intervention flags based on distress level (Paper Section III-A-b-iii).
        """
        is_distressed = intensity >= self.DISTRESS_THRESHOLD

        return {
            "soft_tone": is_distressed,
            "slower_pacing": intensity >= 0.7,
            "extra_validation": emotion == "PANIC" and intensity >= 0.8,
            "burnout_monitoring": emotion == "FATIGUE",
            "guardrail_active": emotion in ("PANIC", "FATIGUE") and intensity >= 0.7,
        }


# ─── Context Retriever ────────────────────────────────────────────────────────

class ContextRetriever:
    """
    User Data & Context Retrieval (Diagram — Layer 1 centre block).

    Pulls two types of context:
      1. DB context  — active schedule tasks and notes via MCP tools
      2. FAISS context — semantic search over knowledge base via RAG service

    The combined context is injected into Layer 2 agent prompts.
    """

    async def retrieve(self, user_query: str, top_k: int = 3) -> dict:
        """
        Retrieve all context needed for Layer 2.

        Returns:
            {
                "schedule_tasks": list[dict],   # Active tasks from DB
                "workload": dict,               # Workload stats
                "rag_context": str,             # Formatted RAG chunks for LLM injection
                "rag_results": list[dict],      # Raw RAG results (for UI)
            }
        """
        schedule_tasks = []
        workload = {}
        rag_context = ""
        rag_results = []

        # ── DB context via MCP tools ──
        try:
            from backend.mcp_server.server import get_mcp_server
            mcp = get_mcp_server()

            schedule_result = await mcp.call_tool("get_schedule")
            workload_result = await mcp.call_tool("get_daily_workload")

            schedule_tasks = schedule_result.get("tasks", [])
            workload = workload_result

        except Exception as e:
            logger.warning(f"Context retrieval (DB) failed: {e}")

        # ── FAISS semantic context ──
        try:
            from backend.services import rag_service
            rag_context = rag_service.retrieve_context(user_query, top_k=top_k)
            rag_results = rag_service.retrieve_detailed(user_query, top_k=top_k)
        except Exception as e:
            logger.warning(f"Context retrieval (FAISS) failed: {e}")

        return {
            "schedule_tasks": schedule_tasks,
            "workload": workload,
            "rag_context": rag_context,
            "rag_results": rag_results,
        }


# ─── Executive Agent ──────────────────────────────────────────────────────────

class ExecutiveAgent:
    """
    Executive Agent / Orchestrator (Paper Section III-A-a).

    Workflow:
      1. Run Empathy Scout → emotion analysis
      2. Classify intent
      3. Determine task routing (which Layer 2 agents to activate)

    Returns full orchestration result including task_routing for Layer 2.
    """

    def __init__(self):
        self.empathy_scout = EmpathyScout()
        self.context_retriever = ContextRetriever()
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
        Full analysis pipeline (synchronous — emotions + intent):
          1. Empathy Scout detects emotion
          2. Executive Agent classifies intent
          3. Determines task routing

        Returns: (intent, emotion, empathy_data, task_routing)
        """
        # ── Empathy Scout ──
        dataset = self._get_dataset()
        few_shot = ""
        if dataset:
            few_shot = (
                dataset.get_few_shot_examples("STRESS", 2) + "\n" +
                dataset.get_few_shot_examples("BURNOUT", 2) + "\n" +
                dataset.get_few_shot_examples("SELF_DEPRECATION", 2) + "\n" +
                dataset.get_few_shot_examples("NEUTRAL", 2)
            )

        empathy_data = self.empathy_scout.analyze(user_input, few_shot)

        # ── Intent Classification ──
        intent = self._classify_intent(user_input)

        # ── Task Routing ──
        task_routing = ROUTING_MAP.get(intent, ["general_chat"])

        return intent, empathy_data["emotion"], empathy_data, task_routing

    async def analyze_with_context(self, user_input: str) -> dict:
        """
        Full async pipeline including context retrieval (for use in chat_service).

        Returns:
            {
                "intent": str,
                "emotion": str,
                "empathy_data": dict,
                "task_routing": list[str],
                "context": {
                    "schedule_tasks": list,
                    "workload": dict,
                    "rag_context": str,
                    "rag_results": list,
                }
            }
        """
        intent, emotion, empathy_data, task_routing = self.analyze_intent(user_input)
        context = await self.context_retriever.retrieve(user_input)

        return {
            "intent": intent,
            "emotion": emotion,
            "empathy_data": empathy_data,
            "task_routing": task_routing,
            "context": context,
        }

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
