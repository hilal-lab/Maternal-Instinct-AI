import asyncio

import aiosqlite
import pytest

from backend.mcp_server import schedule_tools


EXPECTED_TASK = {
    "id": 1,
    "task": "Presentasi Project Agentic AI",
    "scheduled_date": "2030-01-10",
    "start_time": "14:00",
    "end_time": "16:00",
    "status": "pending",
}


async def create_test_database(db_path):
    db = await aiosqlite.connect(db_path)

    await db.execute("""
        CREATE TABLE schedules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task TEXT NOT NULL,
            deadline TEXT NOT NULL,
            scheduled_date TEXT,
            start_time TEXT,
            end_time TEXT,
            est_hours REAL NOT NULL DEFAULT 2,
            priority TEXT NOT NULL DEFAULT 'Medium',
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    await db.execute(
        """
        INSERT INTO schedules (
            id, task, deadline, scheduled_date,
            start_time, end_time, est_hours, priority, status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            EXPECTED_TASK["id"],
            EXPECTED_TASK["task"],
            EXPECTED_TASK["scheduled_date"],
            EXPECTED_TASK["scheduled_date"],
            EXPECTED_TASK["start_time"],
            EXPECTED_TASK["end_time"],
            2.0,
            "Medium",
            EXPECTED_TASK["status"],
        ),
    )

    await db.commit()
    await db.close()


async def read_task(db_path, task_id=1):
    db = await aiosqlite.connect(db_path)
    try:
        cursor = await db.execute(
            "SELECT id, task, scheduled_date, start_time, "
            "end_time, status FROM schedules WHERE id = ?",
            (task_id,),
        )
        row = await cursor.fetchone()
        return row
    finally:
        await db.close()


@pytest.fixture
def isolated_schedule_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_schedule.db"

    asyncio.run(create_test_database(db_path))

    async def test_get_db():
        return await aiosqlite.connect(db_path)

    monkeypatch.setattr(schedule_tools, "_get_db", test_get_db)

    return db_path


def test_delete_task_succeeds_when_snapshot_matches(isolated_schedule_db):
    result = asyncio.run(
        schedule_tools.delete_task(
            task_id=1,
            expected_snapshot=dict(EXPECTED_TASK),
        )
    )

    assert result["success"] is True
    assert result["deleted_task"]["id"] == 1
    assert result["deleted_task"]["task"] == EXPECTED_TASK["task"]

    assert asyncio.run(read_task(isolated_schedule_db)) is None


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("task", "Nama tugas telah berubah"),
        ("scheduled_date", "2030-01-11"),
        ("start_time", "15:00"),
        ("end_time", "17:00"),
        ("status", "in_progress"),
    ],
)
def test_delete_task_rejects_changed_snapshot(
    isolated_schedule_db,
    field,
    changed_value,
):
    snapshot = dict(EXPECTED_TASK)
    snapshot[field] = changed_value

    result = asyncio.run(
        schedule_tools.delete_task(
            task_id=1,
            expected_snapshot=snapshot,
        )
    )

    assert result["success"] is False
    assert "berubah" in result["error"].lower()

    row = asyncio.run(read_task(isolated_schedule_db))
    assert row is not None
    assert row[0] == 1


def test_delete_task_rejects_missing_snapshot(isolated_schedule_db):
    result = asyncio.run(
        schedule_tools.delete_task(task_id=1)
    )

    assert result["success"] is False
    assert "snapshot" in result["error"].lower()

    assert asyncio.run(read_task(isolated_schedule_db)) is not None


def test_delete_task_rejects_snapshot_with_wrong_id(isolated_schedule_db):
    snapshot = dict(EXPECTED_TASK)
    snapshot["id"] = 2

    result = asyncio.run(
        schedule_tools.delete_task(
            task_id=1,
            expected_snapshot=snapshot,
        )
    )

    assert result["success"] is False
    assert asyncio.run(read_task(isolated_schedule_db)) is not None


def test_delete_task_rejects_nonexistent_task(isolated_schedule_db):
    result = asyncio.run(
        schedule_tools.delete_task(
            task_id=999,
            expected_snapshot={
                **EXPECTED_TASK,
                "id": 999,
            },
        )
    )

    assert result["success"] is False
    assert "tidak ditemukan" in result["error"].lower()

    assert asyncio.run(read_task(isolated_schedule_db)) is not None
