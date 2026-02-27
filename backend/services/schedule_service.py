"""
Schedule Service — Business logic for task/schedule management.
"""
from backend.models.database import get_db
from backend.models.schemas import ScheduleCreate, ScheduleUpdate, ScheduleResponse


async def list_all() -> list[ScheduleResponse]:
    """List all tasks ordered by deadline."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, task, deadline, est_hours, priority, status, created_at "
            "FROM schedules ORDER BY deadline"
        )
        rows = await cursor.fetchall()
        return [
            ScheduleResponse(
                id=r[0], task=r[1], deadline=r[2], est_hours=r[3],
                priority=r[4], status=r[5], created_at=str(r[6])
            )
            for r in rows
        ]
    finally:
        await db.close()


async def get_active_list() -> list[dict]:
    """Get active (non-completed) tasks as plain dicts for agent use."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT task, deadline, est_hours, priority FROM schedules "
            "WHERE status != 'completed' ORDER BY deadline"
        )
        rows = await cursor.fetchall()
        return [
            {"task": r[0], "deadline": r[1], "est_hours": r[2], "priority": r[3]}
            for r in rows
        ]
    finally:
        await db.close()


async def create(item: ScheduleCreate) -> ScheduleResponse:
    """Create a new task."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO schedules (task, deadline, est_hours, priority) "
            "VALUES (?, ?, ?, ?) RETURNING *",
            (item.task, item.deadline, item.est_hours, item.priority.value)
        )
        row = await cursor.fetchone()
        await db.commit()
        return ScheduleResponse(
            id=row[0], task=row[1], deadline=row[2], est_hours=row[3],
            priority=row[4], status=row[5], created_at=str(row[6])
        )
    finally:
        await db.close()


async def update(schedule_id: int, item: ScheduleUpdate) -> ScheduleResponse | None:
    """Update a task. Returns None if not found."""
    db = await get_db()
    try:
        updates, values = [], []
        for field, col in [("task","task"), ("deadline","deadline"), ("est_hours","est_hours")]:
            val = getattr(item, field)
            if val is not None:
                updates.append(f"{col} = ?")
                values.append(val)
        if item.priority is not None:
            updates.append("priority = ?")
            values.append(item.priority.value)
        if item.status is not None:
            updates.append("status = ?")
            values.append(item.status.value)

        if not updates:
            return None

        values.append(schedule_id)
        await db.execute(f"UPDATE schedules SET {', '.join(updates)} WHERE id = ?", values)
        await db.commit()

        cursor = await db.execute(
            "SELECT id, task, deadline, est_hours, priority, status, created_at "
            "FROM schedules WHERE id = ?", (schedule_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return ScheduleResponse(
            id=row[0], task=row[1], deadline=row[2], est_hours=row[3],
            priority=row[4], status=row[5], created_at=str(row[6])
        )
    finally:
        await db.close()


async def delete(schedule_id: int) -> bool:
    """Delete a task. Returns True if deleted."""
    db = await get_db()
    try:
        cursor = await db.execute("DELETE FROM schedules WHERE id = ?", (schedule_id,))
        await db.commit()
        return cursor.rowcount > 0
    finally:
        await db.close()


async def get_workload() -> dict:
    """Calculate workload for pending/in-progress tasks."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT SUM(est_hours), COUNT(*) FROM schedules WHERE status != 'completed'"
        )
        row = await cursor.fetchone()
        total = row[0] or 0
        count = row[1] or 0
        return {
            "total_hours": total,
            "task_count": count,
            "max_daily_hours": 8,
            "overloaded": total > 8,
        }
    finally:
        await db.close()
