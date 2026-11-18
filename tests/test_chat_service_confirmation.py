import asyncio

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
def resume_harness(monkeypatch):
    state = {
        "current_snapshot": dict(TASK_SNAPSHOT),
        "snapshot_calls": [],
        "tool_calls": [],
        "events": [],
    }

    class FakeMCP:
        async def call_tool(self, tool_name, **kwargs):
            state["tool_calls"].append((tool_name, kwargs))
            return {
                "success": True,
                "deleted_task": {"task": TASK_SNAPSHOT["task"]},
            }

    class FakeGuardrail:
        def sanitize(self, **kwargs):
            return kwargs["raw_response"], False

    async def fake_get_task_snapshot(task_id):
        state["snapshot_calls"].append(task_id)
        snapshot = state["current_snapshot"]
        if snapshot is None or snapshot["id"] != task_id:
            return None
        return dict(snapshot)

    async def fake_save_to_history(*args, **kwargs):
        pass

    async def on_event(event):
        state["events"].append(event)

    monkeypatch.setattr(cs, "get_mcp_server", lambda: FakeMCP())
    monkeypatch.setattr(cs, "MaternalGuardrail", FakeGuardrail)
    monkeypatch.setattr(cs, "PolicyAggregator", lambda: object())
    monkeypatch.setattr(
        cs, "_get_task_snapshot", fake_get_task_snapshot
    )
    monkeypatch.setattr(cs, "_save_to_history", fake_save_to_history)

    # Hindari ketergantungan pada validasi schema dalam tes alur.
    monkeypatch.setattr(cs, "LayerOneData", lambda **kwargs: kwargs)
    monkeypatch.setattr(cs, "LayerTwoData", lambda **kwargs: kwargs)
    monkeypatch.setattr(cs, "LayerThreeData", lambda **kwargs: kwargs)
    monkeypatch.setattr(cs, "LayerFourData", lambda **kwargs: kwargs)
    monkeypatch.setattr(cs, "LayersData", lambda **kwargs: kwargs)
    monkeypatch.setattr(cs, "ChatResponse", lambda **kwargs: kwargs)

    cs.pending_confirmations.clear()

    confirm_id = "test-confirm-001"

    cs.pending_confirmations[confirm_id] = {
        "confirm_id": confirm_id,
        "tool_name": "delete_task",
        "tool_args": {"task_id": 12},
        "confirm_message": "Konfirmasi penghapusan",
        "delete_task_snapshot": dict(TASK_SNAPSHOT),
        "pipeline_context": {
            "message": "Hapus tugas dengan ID 12",
            "emotion": "neutral",
            "intent": "schedule_management",
            "intensity": 0.0,
            "flags": {},
            "eth_status": "SAFE",
            "eth_note": "",
            "policy_recs": {},
            "workload_check": {},
            "agent_used": "planner",
            "schedule_tasks": [],
            "raw_response": "Baik, saya akan memproses permintaanmu.",
        },
    }

    state["confirm_id"] = confirm_id
    state["on_event"] = on_event

    yield state

    cs.pending_confirmations.clear()


def test_approved_delete_executes_with_confirmed_snapshot(resume_harness):
    state = resume_harness

    result = asyncio.run(
        cs.resume_after_confirmation(
            state["confirm_id"],
            approved=True,
            on_event=state["on_event"],
        )
    )

    assert result is not None
    assert len(state["snapshot_calls"]) == 1
    assert state["snapshot_calls"] == [12]

    assert len(state["tool_calls"]) == 1

    tool_name, kwargs = state["tool_calls"][0]
    assert tool_name == "delete_task"
    assert kwargs["task_id"] == 12
    assert kwargs["expected_snapshot"] == TASK_SNAPSHOT


def test_rejected_confirmation_does_not_execute_delete(resume_harness):
    state = resume_harness

    result = asyncio.run(
        cs.resume_after_confirmation(
            state["confirm_id"],
            approved=False,
            on_event=state["on_event"],
        )
    )

    assert result is not None
    assert "dibatalkan" in result["response"].lower()
    assert state["tool_calls"] == []
    assert state["snapshot_calls"] == []


def test_changed_task_snapshot_blocks_delete(resume_harness):
    state = resume_harness
    state["current_snapshot"]["status"] = "completed"

    result = asyncio.run(
        cs.resume_after_confirmation(
            state["confirm_id"],
            approved=True,
            on_event=state["on_event"],
        )
    )

    assert result is not None
    assert "telah berubah" in result["response"].lower()
    assert state["snapshot_calls"] == [12]
    assert state["tool_calls"] == []


def test_missing_task_blocks_delete(resume_harness):
    state = resume_harness
    state["current_snapshot"] = None

    result = asyncio.run(
        cs.resume_after_confirmation(
            state["confirm_id"],
            approved=True,
            on_event=state["on_event"],
        )
    )

    assert result is not None
    assert "tidak ditemukan" in result["response"].lower()
    assert state["tool_calls"] == []


def test_confirmation_token_can_only_be_used_once(resume_harness):
    state = resume_harness

    first_result = asyncio.run(
        cs.resume_after_confirmation(
            state["confirm_id"],
            approved=False,
            on_event=state["on_event"],
        )
    )

    second_result = asyncio.run(
        cs.resume_after_confirmation(
            state["confirm_id"],
            approved=True,
            on_event=state["on_event"],
        )
    )

    assert first_result is not None
    assert second_result is None
    assert state["tool_calls"] == []
