"""
Adapter dataset + validasi.

Masalah yang diperbaiki (review #2):
  * nama file tidak cocok (huruf besar/kecil, nama berbeda)   -> pencarian case-insensitive + alias
  * nama field berbeda (text / user_prompt / poisoned_context)   -> adapter eksplisit per dataset
  * D1 numerik murni (tidak punya `prompt`)                      -> dikembalikan sebagai `plan`
  * prefiks "[Variation N]" mencemari prompt & menggandakan item -> dibersihkan, `cluster` dicatat
  * label D5 di luar ruang label classifier                      -> alias terdokumentasi + ceiling
  * XSTest tidak pernah termuat (URL fetch salah / 404)          -> dibaca dari data/datasets/xstest_prompts.csv

Semua loader MENOLAK dataset yang menghasilkan prompt kosong (fail fast), bukan mengirim string kosong ke LLM.

CLI:  python -m benchmarks.datasets --check
"""
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path(os.environ.get("LACF_DATA_DIR", "data/datasets"))
_FALLBACK_DIRS = [Path("data/benchmarks")]  # lokasi lama dataset (dipakai fetch_official_datasets sebelum diperbaiki)

_VARIATION_RE = re.compile(r"^\s*\[(?:Variation|Variasi)\s*\d+\]\s*", re.I)

# Alias label D5 -> label yang benar-benar bisa dikeluarkan classifier (ASUMSI, silakan dikonfirmasi)
INTENT_ALIASES = {
    "MOTIVATIONAL_SUPPORT": "MOTIVATION_SUPPORT",
}
INTENT_LABELS = {"TASK_OPS", "ACADEMIC_HELP", "MOTIVATION_SUPPORT", "EMOTIONAL_DISTRESS", "GENERAL_CHAT"}
# Gold tool (nama service di dataset) -> nama tool MCP yang diharapkan keluar dari planner.
# get_tasks -> None: di sistem ini jadwal sudah disuntikkan ke prompt, jadi TIDAK ada tool_call tulis.
TOOL_ALIASES = {
    "schedule_service.create_task": "add_task",
    "schedule_service.delete_task": "delete_task",
    "schedule_service.get_tasks": None,
}
RULE_EXPRESSIBLE_GOALS = {"G1_INFLATE_WORKLOAD", "G2_REMOVE_SLEEP_REST", "G3_OVERRIDE_HEALTH_POLICY"}


class DatasetError(RuntimeError):
    pass


def clean_prompt(text):
    return _VARIATION_RE.sub("", str(text or "")).strip()


def _find(*candidates):
    """Cari file di DATA_DIR (lalu folder cadangan) secara case-insensitive; coba beberapa nama."""
    dirs = [d for d in [DATA_DIR] + _FALLBACK_DIRS if d.is_dir()]
    if not dirs:
        raise DatasetError(f"Folder dataset tidak ada: {DATA_DIR.resolve()} (jalankan dari root repo, atau set LACF_DATA_DIR)")
    seen = []
    for d in dirs:
        existing = {p.name.lower(): p for p in d.iterdir() if p.is_file()}
        seen += sorted(existing)
        for c in candidates:
            if c.lower() in existing:
                return existing[c.lower()]
    raise DatasetError(f"Tidak menemukan {candidates} di {[str(d.resolve()) for d in dirs]}. Ada: {seen}")


def _load_json(*names):
    with open(_find(*names), "r", encoding="utf-8") as f:
        return json.load(f)


def _require_prompts(name, items, key="prompt"):
    empty = [i["id"] for i in items if not str(i.get(key, "")).strip()]
    if empty:
        raise DatasetError(f"{name}: {len(empty)} item dengan `{key}` kosong (contoh: {empty[:3]}). Periksa adapter/skema.")
    return items


# ───────────────────────────── adapters ─────────────────────────────

def load_d1():
    raw = _load_json("D1_plan_scenarios.json", "d1_plan_scenarios.json")
    items = []
    for x in raw:
        plan = {k: x.get(k) for k in ("study_work_h", "sleep_h", "rest_h", "weekly_load_h")}
        if any(v is None for v in plan.values()):
            raise DatasetError(f"D1 {x.get('id')}: field numerik hilang: {plan}")
        items.append({"id": x["id"], "category": x.get("category", ""), "plan": plan,
                      "is_harmful": bool(x["is_harmful"])})
    return items


def load_d2():
    raw = _load_json("d2_affective.json", "d2_user_affect_messages.json")
    items = []
    for x in raw:
        p = clean_prompt(x["text"])
        items.append({"id": x["id"], "prompt": p, "cluster": p.lower(), "gold_affect": x["expected_affect"],
                      "is_affective_need": bool(x["is_affective_need"]), "language": x.get("language", "")})
    return _require_prompts("D2", items)


def load_d3(include_xstest=True):
    raw = _load_json("d3_safety.json", "d3_harmfulness_prompts.json")
    items = [{"id": x["id"], "prompt": clean_prompt(x["prompt"]), "is_harmful": x.get("type", "harmful") == "harmful",
              "source": "HarmBench"} for x in raw]
    if include_xstest:
        try:
            path = _find("xstest_prompts.csv")
        except DatasetError:
            path = None
        if path:
            with open(path, "r", encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    items.append({"id": f"D3-XSTEST-{int(r['id']):04d}", "prompt": clean_prompt(r["prompt"]),
                                  "is_harmful": r["label"].strip().lower() == "unsafe", "source": "XSTest",
                                  "xs_type": r.get("type", "")})
    for it in items:
        it["cluster"] = it["prompt"].lower()
    return _require_prompts("D3", items)


_EVASION_SUFFIX_RE = re.compile(r"\s*\(Evasion Variant \d+\)\s*$", re.I)


def load_d4():
    """
    CATATAN: dataset D4 yang ada hanya berisi 14 serangan unik (G1:3 G2:3 G3:2 G4:2 G5:2 G6:2), masing-masing
    diulang ~15x dengan sufiks teks "(Evasion Variant N)" yang BUKAN teknik evasion sungguhan. Sufiks dibuang
    dan item dianggap duplikat (cluster yang sama).
    """
    raw = _load_json("d4_injection.json", "d4_prompt_injections.json", "d4_injections.json")
    items = []
    for x in raw:
        up = clean_prompt(x["user_prompt"])
        ctx = _EVASION_SUFFIX_RE.sub("", clean_prompt(x["poisoned_context"])).strip()
        items.append({"id": x["id"], "goal": x["goal"], "prompt": up, "context": ctx,
                      "rule_expressible": x["goal"] in RULE_EXPRESSIBLE_GOALS, "cluster": (up + "||" + ctx).lower()})
    _require_prompts("D4", items)
    _require_prompts("D4(context)", items, key="context")
    return items


def load_d5():
    raw = _load_json("d5_performance.json", "d5_task_benchmarks.json")
    items = []
    for x in raw:
        cat, gold = x["category"], x["expected_target"]
        if cat == "intent_classification":
            gold = INTENT_ALIASES.get(gold, gold)
            if gold not in INTENT_LABELS:
                raise DatasetError(f"D5 {x['id']}: label '{gold}' tidak ada di ruang label classifier")
        elif cat == "tool_calling":
            if gold not in TOOL_ALIASES:
                raise DatasetError(f"D5 {x['id']}: tool '{gold}' belum punya alias di TOOL_ALIASES")
            gold = TOOL_ALIASES[gold]
        items.append({"id": x["id"], "category": cat, "prompt": clean_prompt(x["prompt"]), "gold": gold,
                      "cluster": clean_prompt(x["prompt"]).lower(),
                      "scorable": cat in ("intent_classification", "tool_calling")})
    return _require_prompts("D5", items)


LOADERS = {"d1": load_d1, "d2": load_d2, "d3": load_d3, "d4": load_d4, "d5": load_d5}


def dedupe(items):
    """Pertahankan kemunculan pertama tiap `cluster` (decoding temp 0 => duplikat tidak menambah informasi)."""
    seen, out = set(), []
    for it in items:
        c = it.get("cluster")
        if c is None or c not in seen:
            out.append(it)
            seen.add(c)
    return out


# ───────────────────────────── CLI check ─────────────────────────────

def check(verbose=True):
    ok = True
    for name, fn in LOADERS.items():
        try:
            items = fn()
        except DatasetError as e:
            print(f"❌ {name.upper()}: {e}")
            ok = False
            continue
        uniq = len(dedupe(items)) if items and "cluster" in items[0] else len(items)
        extra = ""
        if name == "d3":
            n_x = sum(1 for i in items if i["source"] == "XSTest")
            extra = f" | XSTest: {n_x} (aman {sum(1 for i in items if i['source']=='XSTest' and not i['is_harmful'])})"
            if n_x == 0:
                extra += "  ⚠️ XSTest TIDAK ada -> FRR tidak bisa dihitung"
        if name == "d1":
            extra = f" | harmful {sum(i['is_harmful'] for i in items)} benign {sum(not i['is_harmful'] for i in items)}"
        print(f"✅ {name.upper()}: {len(items):4d} item, {uniq:4d} unik{extra}")
        if verbose:
            s = items[0]
            show = s.get("prompt") or json.dumps(s.get("plan"))
            print(f"     contoh: {s['id']} -> {str(show)[:90]!r}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if check() else 1)
