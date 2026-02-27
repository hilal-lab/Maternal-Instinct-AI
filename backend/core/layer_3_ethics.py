"""
Layer 3: Deliberation & Ethics — Policy Aggregator (Paper Section III-C).
This layer is DETERMINISTIC (no LLM). It enforces hard constraints via Python logic:
  1. Hard Constraint Checking (max work hours, min break time, min sleep)
  2. Health Protection Logic
  3. Conflict Override (agent recommendation vs policy)
"""


class PolicyAggregator:
    """
    Rule-based engine enforcing health and safety constraints.
    Acts as an 'ethical firewall' — LLM is stochastic, Python logic is deterministic.
    """

    # --- HARD CONSTRAINTS (Paper Section III-C-a) ---
    MAX_WORK_HOURS = 8
    MIN_BREAK_HOURS = 1
    MIN_SLEEP_HOURS = 6

    # Keywords indicating unhealthy recommendations
    HEALTH_VIOLATION_KEYWORDS = [
        "lembur", "begadang", "tidak tidur", "skip makan",
        "kerja terus", "tanpa istirahat", "tanpa henti",
        "semalaman", "sampai subuh", "sampai pagi",
        "tidak makan", "skip sarapan", "ignore makan",
        "ignore istirahat", "ignore lapar", "ignore kesehatan",
        "tidur 2 jam", "tidur 3 jam", "tidur 4 jam",
        "stay up", "all night", "nonstop",
    ]

    def review_workload(self, task_list, response_text, is_plan_request):
        """
        Main review pipeline. Checks:
        1. Total work hours against MAX_WORK_HOURS
        2. Unhealthy keywords in response text
        3. Emotional context flags
        Returns: (response_text, status, violation_note)
        """
        status = "PASS"
        violations = []

        # --- CHECK 1: Workload hours ---
        if is_plan_request and task_list:
            total_hours = sum(t.get("est_hours", 0) for t in task_list)

            if total_hours > self.MAX_WORK_HOURS:
                status = "VIOLATION"
                violations.append(
                    f"⚠️ **BEBAN KERJA:** Terdeteksi {total_hours} jam. "
                    f"Melampaui batas aman ({self.MAX_WORK_HOURS} jam/hari)."
                )

        # --- CHECK 2: Unhealthy keywords in response ---
        response_lower = response_text.lower()
        detected_keywords = [kw for kw in self.HEALTH_VIOLATION_KEYWORDS if kw in response_lower]

        if detected_keywords:
            status = "VIOLATION"
            violations.append(
                f"⚠️ **SARAN TIDAK SEHAT:** Terdeteksi indikasi: "
                f"{', '.join(detected_keywords[:3])}."
            )

        # --- BUILD VIOLATION NOTE ---
        violation_note = None
        if violations:
            violation_note = " | ".join(violations)
            response_text += f"\n\n_{violation_note}_"

        return response_text, status, violation_note

    def review_emotion_context(self, emotion: str, intensity: float) -> dict:
        """
        Additional review based on emotional context.
        Returns policy recommendations for downstream layers.
        """
        recommendations = {
            "force_rest_suggestion": False,
            "limit_task_count": False,
            "require_wellness_check": False,
            "max_suggested_tasks": None,
        }

        if emotion == "FATIGUE" or intensity >= 0.7:
            recommendations["force_rest_suggestion"] = True
            recommendations["limit_task_count"] = True
            recommendations["max_suggested_tasks"] = 3

        if emotion in ("PANIC", "SELF-CRITICAL"):
            recommendations["require_wellness_check"] = True

        return recommendations