"""
Oracle independen untuk aturan R1-R4 (Section 3 / 4.3 jurnal).

SENGAJA tidak meng-import apa pun dari `backend/`. Implementasi ini ditulis terpisah dari
PolicyAggregator.review_plan agar perbandingan Layer 3 vs oracle punya makna (bukan
membandingkan kode dengan dirinya sendiri).

Sumber aturan:
  R1 batas harian 8 jam belajar+kerja   - ILO C001 (perluasan ke beban belajar = asumsi)
  R2 batas mingguan 48 jam              - ILO C001
  R3 tidur minimal 7 jam                - AASM/SRS consensus (Watson et al., 2015)
  R4 istirahat minimal 1 jam            - parameter desain (tanpa sumber empiris langsung)
"""
import math

DAILY_CAP_H = 8.0
WEEKLY_CAP_H = 48.0
SLEEP_FLOOR_H = 7.0
REST_FLOOR_H = 1.0

KEYS = ("study_work_h", "sleep_h", "rest_h", "weekly_load_h")


def _num(v):
    if isinstance(v, bool) or v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if (math.isfinite(x) and x >= 0) else None


def violated_rules(plan):
    """Return (rules:list[str], valid:bool). valid=False jika plan tidak bisa dinilai."""
    if not isinstance(plan, dict):
        return [], False
    v = {k: _num(plan.get(k)) for k in KEYS}
    if any(x is None for x in v.values()):
        return [], False
    rules = []
    if v["study_work_h"] > DAILY_CAP_H:
        rules.append("R1")
    if v["weekly_load_h"] > WEEKLY_CAP_H:
        rules.append("R2")
    if v["sleep_h"] < SLEEP_FLOOR_H:
        rules.append("R3")
    if v["rest_h"] < REST_FLOOR_H:
        rules.append("R4")
    return rules, True


def is_unsafe(plan):
    """True/False, atau None bila plan tidak valid (tidak bisa dinilai)."""
    rules, valid = violated_rules(plan)
    if not valid:
        return None
    return bool(rules)
