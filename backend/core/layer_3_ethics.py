"""
Layer 3: Deliberation & Ethics — Policy Aggregator (Paper Section III-C).

Workflow (matches diagram):
  Raw Draft Response (from Layer 2 specialists)
          │
          ▼
  Policy & Ethics Aggregator  ◄──── (workload from MCP Server)
          │
          ▼
  ┌─── Workload Checking ───┐
  │                          │
  │  Overload?               │  Not Overload?
  ▼                          ▼
(signals Layer 2        Draft Logic Response
 MCP Server to               │
 redistribute)               ▼
                       (to Layer 4 as "Raw Response")

This layer is DETERMINISTIC (no LLM). It enforces hard constraints via Python:
  1. Hard Constraint Checking (max work hours, min break time, min sleep)
  2. Workload Checking — the explicit decision gate from the diagram
  3. Health Protection Logic (keyword scanning)
  4. Draft Logic Response production
"""
from typing import Optional


class PolicyAggregator:
    """
    Rule-based engine enforcing health and safety constraints.
    Acts as an 'ethical firewall' — LLM is stochastic, Python logic is deterministic.

    Diagram nodes implemented:
      - "Policy and Ethic Aggregator" — the entry block
      - "Workload Checking"           — the decision diamond
      - "Draft Logic Response"        — the output block (annotated on response)
    """

    # ── Hard Constraints (Paper Section III-C-a) ──────────────────────────────
    MAX_WORK_HOURS = 8
    MIN_BREAK_HOURS = 1
    MIN_SLEEP_HOURS = 6

    HEALTH_VIOLATION_KEYWORDS = [
        "skip sarapan",
        "ignore makan",
        "ignore istirahat",
        "ignore lapar",
        "stay up all night",
        "begadang",
        "lembur terus",
    ]

    # ── Workload Checking (diagram diamond) ───────────────────────────────────

    def check_workload(self, workload: dict) -> dict:
        """
        Workload Checking decision gate (diagram diamond node).

        Evaluates MCP workload data and decides:
          - OVERLOAD: Signal back to MCP layer to redistribute tasks
          - NOT_OVERLOAD: Proceed to Draft Logic Response

        Args:
            workload: Result from MCP get_daily_workload() tool.

        Returns:
            {
                "decision": "OVERLOAD" | "NOT_OVERLOAD",
                "total_hours": float,
                "overload_by_hours": float,
                "message": str,          # Human-readable policy note
                "should_redistribute": bool,  # Signal for MCP Server
            }
        """
        if not workload or not workload.get("success"):
            return {
                "decision": "NOT_OVERLOAD",
                "total_hours": 0,
                "overload_by_hours": 0,
                "message": "Tidak ada data beban kerja.",
                "should_redistribute": False,
            }

        total = float(workload.get("total_hours", 0))
        overloaded = workload.get("overloaded", False)
        overload_by = float(workload.get("overload_by_hours", 0))
        task_count = int(workload.get("task_count", 0))

        if overloaded:
            message = (
                f"⚠️ Beban kerja ({total:.1f} jam, {task_count} tugas) "
                f"melebihi batas aman {self.MAX_WORK_HOURS} jam/hari "
                f"sebesar {overload_by:.1f} jam. Redistribusi tugas diperlukan."
            )
            return {
                "decision": "OVERLOAD",
                "total_hours": total,
                "overload_by_hours": overload_by,
                "message": message,
                "should_redistribute": True,
            }

        return {
            "decision": "NOT_OVERLOAD",
            "total_hours": total,
            "overload_by_hours": 0,
            "message": f"Beban kerja {total:.1f} jam — dalam batas aman.",
            "should_redistribute": False,
        }

    # ── Main Review Pipeline ──────────────────────────────────────────────────

    def review_workload(
        self,
        task_list: list,
        response_text: str,
        is_plan_request: bool,
        workload_check_result: Optional[dict] = None,
    ) -> tuple[str, str, Optional[str]]:
        """
        Main review pipeline. Checks:
          1. Workload hours against MAX_WORK_HOURS (via MCP workload check)
          2. Unhealthy keywords in Layer 2 response text
          3. Produces Draft Logic Response (annotated response for Layer 4)

        Args:
            task_list:             Active task list (for fallback hour calculation).
            response_text:         Raw output from Layer 2 agents.
            is_plan_request:       True if this is a TASK_OPS request.
            workload_check_result: Pre-fetched workload check dict (from check_workload()).

        Returns:
            (annotated_response, status, violation_note)
            status: "PASS" | "VIOLATION"
        """
        status = "PASS"
        violations = []

        # ── CHECK 1: Workload Overload (from Workload Checking gate) ──
        if is_plan_request:
            if workload_check_result and workload_check_result.get("decision") == "OVERLOAD":
                status = "VIOLATION"
                violations.append(workload_check_result["message"])
            elif task_list:
                # Fallback: calculate from raw task list if workload_check not provided
                total_hours = sum(t.get("est_hours", 0) for t in task_list)
                if total_hours > self.MAX_WORK_HOURS:
                    status = "VIOLATION"
                    violations.append(
                        f"⚠️ **BEBAN KERJA:** Terdeteksi {total_hours} jam. "
                        f"Melampaui batas aman ({self.MAX_WORK_HOURS} jam/hari)."
                    )

        # ── CHECK 2: Unhealthy keywords in response (warning only, not VIOLATION) ──
        response_lower = response_text.lower()
        detected_keywords = [kw for kw in self.HEALTH_VIOLATION_KEYWORDS if kw in response_lower]

        if detected_keywords:
            violations.append(
                f"⚠️ **PERHATIAN:** Terdeteksi indikasi: "
                f"{', '.join(detected_keywords[:3])}."
            )

        # ── BUILD DRAFT LOGIC RESPONSE ──
        # This is the "Draft Logic Response" node in the diagram —
        # the policy-checked, annotated response that flows to Layer 4.
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

        if emotion == "PANIC":
            recommendations["require_wellness_check"] = True

        return recommendations
