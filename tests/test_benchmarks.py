"""Tes cepat (tanpa Ollama) untuk instrumen benchmark. Jalankan dari root repo:  pytest -q tests/test_benchmarks.py"""
import itertools
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pytest  # noqa: E402

from backend.core.layer_3_ethics import PolicyAggregator  # noqa: E402
from benchmarks import datasets, metrics, oracle  # noqa: E402
from benchmarks.run_suite import REFUSAL_RE, main as run_main  # noqa: E402
from benchmarks.store import Store  # noqa: E402

L3 = PolicyAggregator()


def test_layer3_matches_independent_oracle_on_grid():
    def vals(a, b, s):
        return [round(a + i * s, 2) for i in range(int((b - a) / s) + 1)]
    for sw, sl, rs, wk in itertools.product(vals(0, 14, 1), vals(3, 10, 1), vals(0, 3, 0.5), vals(10, 60, 10)):
        plan = dict(study_work_h=sw, sleep_h=sl, rest_h=rs, weekly_load_h=wk)
        assert oracle.is_unsafe(plan) == (L3.review_plan(plan)["status"] == "VIOLATION"), plan


@pytest.mark.parametrize("bad", [None, {}, {"study_work_h": "5 jam", "sleep_h": 8, "rest_h": 1, "weekly_load_h": 9},
                                 {"study_work_h": True, "sleep_h": 8, "rest_h": 1, "weekly_load_h": 9},
                                 {"study_work_h": float("nan"), "sleep_h": 8, "rest_h": 1, "weekly_load_h": 9}])
def test_invalid_plan_is_never_pass_or_violation(bad):
    assert L3.review_plan(bad)["status"] == "INVALID"
    assert oracle.is_unsafe(bad) is None


def test_boundaries_are_inclusive_safe():
    assert L3.review_plan({"study_work_h": 8, "sleep_h": 7, "rest_h": 1, "weekly_load_h": 48})["status"] == "PASS"


def test_all_datasets_load_with_nonempty_prompts():
    for name, fn in datasets.LOADERS.items():
        items = fn()
        assert items, name
        if name != "d1":
            assert all(i["prompt"].strip() for i in items), name


def test_d1_labels_agree_with_oracle():
    for it in datasets.load_d1():
        assert oracle.is_unsafe(it["plan"]) == it["is_harmful"], it["id"]


def test_d3_has_benign_items_so_frr_is_measurable():
    items = datasets.load_d3()
    assert sum(1 for i in items if i["source"] == "XSTest" and not i["is_harmful"]) == 250
    assert sum(1 for i in items if i["source"] == "XSTest" and i["is_harmful"]) == 200


def test_d4_effective_sample_is_14_unique_attacks():
    items = datasets.load_d4()
    assert len(datasets.dedupe(items)) == 14
    assert all("Evasion Variant" not in i["context"] for i in items)


def test_d5_gold_labels_are_reachable():
    for it in datasets.load_d5():
        if it["category"] == "intent_classification":
            assert it["gold"] in datasets.INTENT_LABELS


def test_wilson_known_values():
    lo, hi = metrics.wilson(0, 5)
    assert lo == 0 and abs(hi - 0.434) < 0.005
    assert metrics.wilson(0, 0) is None


def test_store_drops_error_rows_so_they_are_retried(tmp_path):
    p = str(tmp_path / "r.csv")
    s = Store(p)
    s.add(config="c", item_id="a", status="ERROR")
    s.add(config="c", item_id="b", status="OK")
    s2 = Store(p)
    assert not s2.has("c", "a") and s2.has("c", "b")


def test_refusal_regex_covers_english_and_indonesian():
    assert REFUSAL_RE.search("I'm sorry, but I can't help with that.")
    assert REFUSAL_RE.search("Maaf, saya tidak bisa membantu.")
    assert not REFUSAL_RE.search("Sure, here is how to kill a Python process.")


def test_d1_end_to_end_without_llm(tmp_path):
    assert run_main(["--datasets", "d1", "--out", str(tmp_path), "--run-id", "t"]) == 0
    s = metrics.build(tmp_path / "t")["D1"]
    assert (s["TP"], s["FN"], s["FP"], s["TN"]) == (13, 0, 0, 5)


def test_guard_model_resolves_installed_tag(monkeypatch):
    from benchmarks import llm
    monkeypatch.setattr(llm, "_resolved_guard", None)
    monkeypatch.setattr(llm, "model_available", lambda name: (True, ["llama3.2:3b", "llama-guard3:1b"]))
    assert llm.resolve_guard_model() == "llama-guard3:1b"   # bukan 'llama-guard3' (=:latest, tidak terpasang)


def test_guard_missing_returns_none(monkeypatch):
    from benchmarks import llm
    monkeypatch.setattr(llm, "_resolved_guard", None)
    monkeypatch.setattr(llm, "model_available", lambda name: (False, ["llama3.2:3b"]))
    assert llm.resolve_guard_model() is None
    assert llm.guard_check("hi")[0] is None


def test_generate_falls_back_to_json_mode_when_schema_rejected(monkeypatch):
    from benchmarks import llm
    sent = []

    class R:
        def __init__(self, code, body): self.status_code, self._b, self.text = code, body, str(body)
        def json(self): return self._b

    def fake_post(url, json=None, timeout=0):
        sent.append(json.get("format"))
        return R(400, {"error": "invalid format"}) if isinstance(json.get("format"), dict) else R(200, {"response": "{}", "done_reason": "stop"})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    monkeypatch.setattr(llm, "_schema_supported", True)
    res = llm.generate("x", schema=llm.PLAN_SCHEMA, retries=1)
    assert res.ok and sent[0] == llm.PLAN_SCHEMA and sent[1] == "json"


def test_clean_prompt_removes_test_iteration_suffix():
    p1 = "Jadwalkan sesi belajar jam 14:00 - 16:00"
    p2 = p1 + " [Test Iteration 2]"
    p3 = p1 + " [Test Iteration 15]"

    assert datasets.clean_prompt(p1) == p1
    assert datasets.clean_prompt(p2) == p1
    assert datasets.clean_prompt(p3) == p1


def test_d5_repetitions_share_same_cluster():
    items = datasets.load_d5()

    intent_items = [
        x for x in items
        if x["category"] == "intent_classification"
    ]

    by_prompt = {}
    for item in intent_items:
        by_prompt.setdefault(item["prompt"], set()).add(item["cluster"])

    assert by_prompt
    assert all(len(clusters) == 1 for clusters in by_prompt.values())


def test_parse_tool_prediction_supports_json_and_legacy():
    from benchmarks.metrics import parse_tool_prediction

    new_value = (
        '{"name": "add_task", '
        '"args": {"start_time": "14:00"}}'
    )
    parsed = parse_tool_prediction(new_value)

    assert parsed["name"] == "add_task"
    assert parsed["args"]["start_time"] == "14:00"

    old_value = parse_tool_prediction("delete_task")
    assert old_value["name"] == "delete_task"
    assert old_value["args"] == {}


def test_d5_get_tasks_maps_to_registered_get_schedule():
    items = datasets.load_d5()

    get_tasks = [
        x for x in items
        if x["category"] == "tool_calling"
        and x["prompt"].lower().startswith("tampilkan daftar")
    ]

    assert get_tasks
    assert all(x["gold"] == "get_schedule" for x in get_tasks)

def test_extract_tool_call_accepts_get_schedule():
    from backend.core.layer_2_specialists import extract_tool_call

    raw = (
        '```tool_call\n'
        '{"name": "get_schedule", "args": {}}\n'
        '```'
    )

    result = extract_tool_call(raw)

    assert result == {
        "name": "get_schedule",
        "args": {},
    }