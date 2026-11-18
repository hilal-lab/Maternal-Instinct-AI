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
from benchmarks.run_suite import (
    REFUSAL_RE,
    main as run_main,
    _prepare_dataset_items,
    _d5_validate_tool_args,
    _d5_detect_clarification,
    _d5_execute_get_schedule_in_memory,
)
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


def test_d5_tool_calling_preserves_evaluation_metadata():
    items = datasets.load_d5()
    by_id = {it["id"]: it for it in items}

    # Kasus klarifikasi: penjadwalan belum lengkap.
    create_incomplete = by_id["D5-PERF-0006"]["evaluation"]
    assert create_incomplete["target_tool"] == "add_task"
    assert create_incomplete["expected_behavior"] == "clarification_required"
    assert create_incomplete["execution_eligible"] is False
    assert create_incomplete["missing_args"] == ["scheduled_date"]

    # Kasus tool call: membaca jadwal.
    get_schedule = by_id["D5-PERF-0007"]["evaluation"]
    assert get_schedule["target_tool"] == "get_schedule"
    assert get_schedule["expected_behavior"] == "tool_call_required"
    assert get_schedule["execution_eligible"] is True

    # Penghapusan tanpa ID: ambil jadwal dahulu, baru minta pengguna memilih ID.
    delete_incomplete = by_id["D5-PERF-0008"]["evaluation"]
    assert delete_incomplete["target_tool"] == "delete_task"
    assert delete_incomplete["expected_first_tool"] == "get_schedule"
    assert delete_incomplete["expected_behavior"] == "tool_call_then_clarify"
    assert delete_incomplete["execution_eligible"] is False
    assert delete_incomplete["requires_confirmation"] is True
    assert delete_incomplete["expected_tool_sequence"] == ["get_schedule"]

    # Kasus penjadwalan lengkap: tool boleh dipanggil, lalu perlu konfirmasi.
    create_complete = by_id["D5-PERF-0166"]["evaluation"]
    assert create_complete["target_tool"] == "add_task"
    assert create_complete["expected_behavior"] == "tool_call_then_confirm"
    assert create_complete["execution_eligible"] is True
    assert create_complete["expected_tool_sequence"] == ["add_task"]

    # Kategori non-tool tidak diwajibkan memiliki metadata evaluation.
    non_tool = next(it for it in items if it["category"] == "intent_classification")
    assert "evaluation" not in non_tool


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
    assert llm.resolve_guard_model() == "llama-guard3:1b"


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
    intent_items = [x for x in items if x["category"] == "intent_classification"]
    by_prompt = {}
    for item in intent_items:
        by_prompt.setdefault(item["prompt"], set()).add(item["cluster"])

    assert by_prompt
    assert all(len(clusters) == 1 for clusters in by_prompt.values())


def test_parse_tool_prediction_supports_json_and_legacy():
    from benchmarks.metrics import parse_tool_prediction

    new_value = '{"name": "add_task", "args": {"start_time": "14:00"}}'
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
    raw = '```tool_call\n{"name": "get_schedule", "args": {}}\n```'
    result = extract_tool_call(raw)
    assert result == {"name": "get_schedule", "args": {}}


def test_metrics_fmt_uses_correct_denominator():
    assert metrics.fmt(metrics.rate(7, 10)) == "70.0% (7/10)"
    assert metrics.fmt(metrics.rate(0, 0)) == "-"


def test_d3_summary_separates_c5_diagnostics():
    summary = {
        "meta": {},
        "D1": None,
        "D2": None,
        "D3": {
            "_notes": {},
            "C5_guard_diagnostics": {
                "n_attempts": 10,
                "n_successful": 10,
                "input_block_harmful": metrics.rate(6, 7),
                "input_false_block_safe": metrics.rate(1, 3),
                "output_block_harmful": metrics.rate(1, 1),
                "output_false_block_safe": metrics.rate(2, 2),
                "overall_false_block_safe": metrics.rate(3, 3),
                "guard_error_rate": metrics.rate(0, 10),
                "guard_error_count": 0,
            },
        },
        "D4": None,
        "D5": None,
    }
    md = metrics.to_md(summary)
    assert "C5 — Llama Guard input/output diagnostics" in md
    assert "input_false_block_safe" not in md
    assert "| C5_guard_diagnostics |" not in md
    assert "100.0% (3/3)" in md


def test_extra_bool_handles_json_boolean_values():
    from benchmarks.metrics import _extra_bool
    row_true = {"extra": '{"flag": true}'}
    row_false = {"extra": '{"flag": false}'}
    row_string_false = {"extra": '{"flag": "false"}'}
    row_string_true = {"extra": '{"flag": "true"}'}

    assert _extra_bool(row_true, "flag") is True
    assert _extra_bool(row_false, "flag") is False
    assert _extra_bool(row_string_false, "flag") is False
    assert _extra_bool(row_string_true, "flag") is True


def test_d5_add_task_argument_validation():
    valid = {
        "task": "Belajar matematika",
        "scheduled_date": "2030-01-10",
        "start_time": "08:00",
        "end_time": "09:00",
    }
    assert _d5_validate_tool_args("add_task", valid)["valid"] is True

    missing = dict(valid)
    missing.pop("scheduled_date")
    assert _d5_validate_tool_args("add_task", missing)["valid"] is False

    invalid_date = dict(valid, scheduled_date="2030-02-30")
    assert _d5_validate_tool_args("add_task", invalid_date)["valid"] is False

    invalid_time = dict(valid, end_time="07:00")
    assert _d5_validate_tool_args("add_task", invalid_time)["valid"] is False


def test_d5_get_schedule_argument_validation():
    assert _d5_validate_tool_args("get_schedule", {})["valid"] is True
    assert _d5_validate_tool_args("get_schedule", {"status_filter": "pending"})["valid"] is True
    assert _d5_validate_tool_args("get_schedule", {"status_filter": 123})["valid"] is False


def test_d5_clarification_detection():
    assert _d5_detect_clarification("Bisa berikan tanggal penjadwalannya terlebih dahulu?", ["scheduled_date"])
    assert _d5_detect_clarification("Mohon berikan ID tugas yang ingin dihapus?", ["task_identity"])
    assert not _d5_detect_clarification("Baik, tugas sudah diproses.", ["scheduled_date"])


def test_d5_get_schedule_executes_in_memory_and_restores_db_helper():
    from backend.mcp_server import schedule_tools
    original_get_db = schedule_tools._get_db
    result = _d5_execute_get_schedule_in_memory({})
    assert result["success"] is True
    assert result["count"] == 0
    assert result["tasks"] == []
    assert schedule_tools._get_db is original_get_db


def test_d5_metrics_use_runner_metadata_and_execution_denominator():
    def make_row(item_id, behavior, expected_tool, predicted_tool,
                 args_validation, behavior_match, execution_status,
                 execution_success=None, clarification=False,
                 confirmation_required=False, confirmation_tested=False,
                 expected_first_tool=None):
        import json
        extra = {
            "expected_behavior": behavior,
            "expected_tool": expected_tool,
            "expected_first_tool": expected_first_tool or expected_tool,
            "tool_call_found": predicted_tool != "NONE",
            "predicted_tool": predicted_tool,
            "tool_name_match": predicted_tool == (expected_first_tool or expected_tool),
            "args_validation": args_validation,
            "behavior_match": behavior_match,
            "clarification_detected": clarification,
            "execution_status": execution_status,
            "execution_success": execution_success,
            "confirmation_required": confirmation_required,
            "confirmation_flow_tested": confirmation_tested,
        }
        return {
            "config": "L2_planner_tools",
            "status": "OK",
            "item_id": item_id,
            "cluster": item_id,
            "gold": expected_tool,
            "pred": json.dumps({"name": predicted_tool, "args": {}}),
            "source": "llm",
            "latency_ms": "10",
            "extra": json.dumps(extra),
        }

    rows = [
        make_row("clarify-1", "clarification_required", "add_task", "NONE", "NOT_APPLICABLE", True, "NOT_REQUIRED_CLARIFICATION", clarification=True),
        make_row("schedule-1", "tool_call_required", "get_schedule", "get_schedule", "VALID", True, "SUCCESS", execution_success=True),
        make_row("create-1", "tool_call_then_confirm", "add_task", "add_task", "VALID", True, "NOT_EXECUTED_CONFIRMATION_REQUIRED", confirmation_required=True),
        make_row(
            "delete-1",
            "tool_call_then_clarify",
            "delete_task",
            "get_schedule",
            "VALID",
            True,
            "SUCCESS",
            execution_success=True,
            expected_first_tool="get_schedule",
        ),
    ]

    def rate_for_test(success, total):
        return metrics.rate(success, total)

    result = metrics.s_d5(rows)["tool_calling"]

    assert result["n_trials"] == 4
    assert result["name_match_trials"] == rate_for_test(3, 3)
    assert result["behavior_match_rate"] == rate_for_test(4, 4)
    assert result["clarification_trials"] == 1
    assert result["clarification_detection_rate"] == rate_for_test(1, 1)
    assert result["args_validation_trials"] == 3
    assert result["args_valid_count"] == 3
    assert result["execution_attempts"] == 2
    assert result["execution_success_count"] == 2
    assert result["confirmation_required_trials"] == 1
    assert result["confirmation_flow_status"] == "NOT_TESTED"
    assert result["tool_call_then_clarify_trials"] == 1
    assert result["tool_call_then_clarify_first_step_matches"] == 1
    assert result["exact_argument_accuracy"].startswith("NOT_MEASURED")


def test_d5_default_dedupe_preserves_existing_behavior():
    items = datasets.load_d5()
    selected, meta = _prepare_dataset_items(
        "d5", items, 0, False, False, []
    )
    assert meta["total_items"] == 180
    assert meta["excluded_items"] == 0
    assert meta["post_exclusion_items"] == 180
    assert meta["post_dedupe_items"] == 26
    assert meta["dedupe_applied"] is True
    assert meta["selected_items"] == 26
    assert len(selected) == 26


def test_d5_exclusion_and_no_dedupe_preserve_165_items():
    items = datasets.load_d5()
    excluded_ids = [
        "D5-PERF-0008",
        "D5-PERF-0019",
        "D5-PERF-0030",
        "D5-PERF-0041",
        "D5-PERF-0052",
        "D5-PERF-0063",
        "D5-PERF-0074",
        "D5-PERF-0085",
        "D5-PERF-0096",
        "D5-PERF-0107",
        "D5-PERF-0118",
        "D5-PERF-0129",
        "D5-PERF-0140",
        "D5-PERF-0151",
        "D5-PERF-0162",
    ]
    selected, meta = _prepare_dataset_items(
        "d5", items, 0, False, True, excluded_ids
    )
    assert len(selected) == 165
    assert meta["total_items"] == 180
    assert meta["excluded_items"] == 15
    assert meta["post_exclusion_items"] == 165
    assert meta["post_dedupe_items"] == 165
    assert meta["selected_items"] == 165
    assert meta["dedupe_applied"] is False
    assert set(meta["excluded_ids"]) == set(excluded_ids)


def test_d5_exclusion_rejects_unknown_ids():
    items = datasets.load_d5()
    with pytest.raises(ValueError, match="tidak ditemukan"):
        _prepare_dataset_items(
            "d5", items, 0, False, False, ["D5-NONEXISTENT-ID"]
        )


def test_no_dedupe_d5_does_not_affect_other_datasets():
    items = datasets.load_d4()
    selected_normal, meta_normal = _prepare_dataset_items(
        "d4", items, 0, False, False, []
    )
    selected_d5_flag, meta_d5_flag = _prepare_dataset_items(
        "d4", items, 0, False, True, []
    )
    assert meta_d5_flag["dedupe_applied"] is True
    assert len(selected_d5_flag) == len(selected_normal)
    assert meta_d5_flag["selected_items"] == meta_normal["selected_items"]
