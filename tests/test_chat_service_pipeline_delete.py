import asyncio
from types import SimpleNamespace

import pytest

from backend.services import chat_service as cs


TASK_SNAPSHOT = {
    "id": 12,
    "task": "Presentasi Project Agentic AI",
    "scheduled_date": "2030-01-10",
    "start_time": "14:00",
    "end_time": "16:00",
    "status": "pending",
}


@pytest.fixture
def pipeline_harness(monkeypatch):
    state = {
        "tool_calls": [],
        "snapshot_calls": [],
        "events": [],
        "snapshot": dict(TASK_SNAPSHOT),
        "tool_call": {
            "name": "delete_task",
            "args": {"task_id": 12},
        },
    }

    class FakeOrchestrator:
        def __init__(self):
            self.context_retriever = SimpleNamespace(
                retrieve=self.retrieve
            )

        async def retrieve(self, message, chat_id=None):
            return {
                "schedule_tasks": [],
                "workload": {},
                "rag_context": "",
                "chat_history_context": "",
            }

        def analyze_intent(self, message):
            return (
                "TASK_OPS",
                "neutral",
                {"intensity": 0.0, "flags": {}},
                ["planner"],
            )

    class FakeMCP:
        def is_destructive(self, tool_name):
            return tool_name == "delete_task"

        def get_confirm_message(self, tool_name):
            return "Konfirmasi penghapusan tugas."

        async def call_tool(self, tool_name, **kwargs):
            state["tool_calls"].append((tool_name, kwargs))
            if tool_name == "get_schedule":
                tasks = state.get(
                    "schedule_result_tasks",
                    [dict(TASK_SNAPSHOT)],
                )
                return {
                    "success": True,
                    "count": len(tasks),
                    "tasks": tasks,
                }
            return {"success": True}

    class FakeSpecialists:
        def __init__(self, mcp_server):
            pass

        async def run_with_mcp(self, **kwargs):
            return (
                "Draft respons Planner.",
                "planner",
                state["tool_call"],
            )

    class FakeEthics:
        def check_workload(self, workload):
            return {
                "decision": "NOT_OVERLOAD",
                "message": "",
            }

        def review_workload(self, **kwargs):
            return (
                kwargs["response_text"],
                "SAFE",
                "",
            )

        def review_emotion_context(self, emotion, intensity):
            return {}

    class FakeGuardrail:
        def sanitize(self, **kwargs):
            return kwargs["raw_response"], False

    async def fake_snapshot(task_id):
        state["snapshot_calls"].append(task_id)

        snapshot = state["snapshot"]
        if snapshot is None or snapshot["id"] != task_id:
            return None

        return dict(snapshot)

    async def fake_save(*args, **kwargs):
        pass

    async def on_event(event):
        state["events"].append(event)

    monkeypatch.setattr(cs, "ExecutiveAgent", FakeOrchestrator)
    monkeypatch.setattr(cs, "SpecialistAgents", FakeSpecialists)
    monkeypatch.setattr(cs, "get_mcp_server", lambda: FakeMCP())
    monkeypatch.setattr(cs, "PolicyAggregator", FakeEthics)
    monkeypatch.setattr(cs, "MaternalGuardrail", FakeGuardrail)
    monkeypatch.setattr(cs, "_get_task_snapshot", fake_snapshot)
    monkeypatch.setattr(cs, "_save_to_history", fake_save)

    monkeypatch.setattr(cs, "LayerOneData", lambda **kwargs: kwargs)
    monkeypatch.setattr(cs, "LayerTwoData", lambda **kwargs: kwargs)
    monkeypatch.setattr(cs, "LayerThreeData", lambda **kwargs: kwargs)
    monkeypatch.setattr(cs, "LayerFourData", lambda **kwargs: kwargs)
    monkeypatch.setattr(cs, "LayersData", lambda **kwargs: kwargs)
    monkeypatch.setattr(cs, "ChatResponse", lambda **kwargs: kwargs)

    cs.pending_confirmations.clear()

    state["on_event"] = on_event

    yield state

    cs.pending_confirmations.clear()


def run_pipeline(state, message):
    return asyncio.run(
        cs.run_pipeline(
            message=message,
            on_event=state["on_event"],
        )
    )


def test_delete_without_explicit_id_fetches_schedule_and_asks_for_id(
    pipeline_harness,
):
    state = pipeline_harness

    result = run_pipeline(state, "Hapus tugas yang ketiga")

    assert result is not None
    assert state["tool_calls"] == [("get_schedule", {})]
    assert not any(call[0] == "delete_task" for call in state["tool_calls"])
    assert state["snapshot_calls"] == []
    assert not any(
        event["type"] == "tool_confirm"
        for event in state["events"]
    )
    assert cs.pending_confirmations == {}

    response = result["response"]
    assert "Berikut daftar tugas aktual dari jadwal:" in response
    assert TASK_SNAPSHOT["task"] in response
    assert "Silakan pilih tugas yang ingin dihapus" in response
    assert "Draft respons Planner." not in response


def test_delete_without_id_handles_empty_schedule(pipeline_harness):
    state = pipeline_harness
    state["schedule_result_tasks"] = []

    result = run_pipeline(state, "Hapus tugas")

    assert result is not None
    assert state["tool_calls"] == [("get_schedule", {})]
    assert "Tidak ada tugas aktif" in result["response"]
    assert "Draft respons Planner." not in result["response"]
    assert state["snapshot_calls"] == []
    assert cs.pending_confirmations == {}


def test_delete_without_id_works_when_planner_returns_no_tool_call(
    pipeline_harness,
):
    state = pipeline_harness
    state["tool_call"] = None

    result = run_pipeline(state, "Hapus tugas")

    assert result is not None
    assert state["tool_calls"] == [("get_schedule", {})]
    assert TASK_SNAPSHOT["task"] in result["response"]
    assert "Silakan pilih tugas yang ingin dihapus" in result["response"]
    assert "Draft respons Planner." not in result["response"]
    assert state["snapshot_calls"] == []
    assert cs.pending_confirmations == {}


def test_delete_without_id_works_when_planner_returns_get_schedule(
    pipeline_harness,
):
    state = pipeline_harness
    state["tool_call"] = {
        "name": "get_schedule",
        "args": {},
    }

    result = run_pipeline(state, "Hapus tugas")

    assert result is not None
    assert state["tool_calls"] == [("get_schedule", {})]
    assert TASK_SNAPSHOT["task"] in result["response"]
    assert "Silakan pilih tugas yang ingin dihapus" in result["response"]
    assert "Draft respons Planner." not in result["response"]
    assert state["snapshot_calls"] == []
    assert cs.pending_confirmations == {}


def test_delete_with_mismatched_id_is_rejected(pipeline_harness):
    state = pipeline_harness

    result = run_pipeline(state, "Hapus tugas dengan ID 13")

    assert result is not None
    assert "tidak cocok" in result["response"]
    assert not any(
        event["type"] == "tool_confirm"
        for event in state["events"]
    )
    assert state["snapshot_calls"] == []
    assert state["tool_calls"] == []
    assert cs.pending_confirmations == {}


def test_delete_with_nonexistent_id_is_rejected(pipeline_harness):
    state = pipeline_harness
    state["snapshot"] = None

    result = run_pipeline(state, "Hapus tugas dengan ID 12")

    assert result is not None
    assert "tidak ditemukan" in result["response"]
    assert state["snapshot_calls"] == [12]
    assert not any(
        event["type"] == "tool_confirm"
        for event in state["events"]
    )
    assert state["tool_calls"] == []
    assert cs.pending_confirmations == {}


def test_valid_delete_creates_identity_specific_confirmation(pipeline_harness):
    state = pipeline_harness

    result = run_pipeline(state, "Hapus tugas dengan ID 12")

    assert result is None
    assert state["snapshot_calls"] == [12]
    assert state["tool_calls"] == []

    confirmation_events = [
        event for event in state["events"]
        if event["type"] == "tool_confirm"
    ]

    assert len(confirmation_events) == 1

    event = confirmation_events[0]
    assert event["tool"] == "delete_task"
    assert event["args"] == {"task_id": 12}
    assert "ID: 12" in event["description"]
    assert TASK_SNAPSHOT["task"] in event["description"]

    confirm_id = event["confirm_id"]
    pending = cs.pending_confirmations[confirm_id]

    assert pending["tool_args"] == {"task_id": 12}
    assert pending["delete_task_snapshot"] == TASK_SNAPSHOT
    assert pending["pipeline_context"]["message"] == (
        "Hapus tugas dengan ID 12"
    )


@pytest.mark.parametrize(
    "message",
    [
        "Hapus tugas",
        "Tolong hapus tugas yang ketiga",
        "Delete task",
        "Saya ingin menghapus tugas kuliah",
    ],
)
def test_direct_task_delete_request_is_detected(message):
    assert cs._is_direct_task_delete_request(message)


@pytest.mark.parametrize(
    "message",
    [
        "Bagaimana cara menghapus tugas?",
        "Jelaskan cara menghapus tugas",
        "Apa itu penghapusan tugas?",
        "Apakah tugas ini perlu dihapus?",
        "Tampilkan jadwal tugas",
    ],
)
def test_non_delete_request_is_not_detected(message):
    assert not cs._is_direct_task_delete_request(message)
