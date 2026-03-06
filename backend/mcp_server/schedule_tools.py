"""
MCP Schedule Tools — Structured tool interface for task/schedule operations.

Wraps the schedule_service layer to expose callable tools to Layer 2 agents.
All tools return plain dicts for JSON serialization.

Tools exposed:
  - get_schedule            : List all active tasks
  - get_upcoming_deadlines  : Tasks due within N days
  - add_task                : Create a new task            [DESTRUCTIVE]
  - update_task_status      : Mark a task as pending/in_progress/completed  [DESTRUCTIVE]
  - delete_task             : Permanently delete a task    [DESTRUCTIVE]
  - get_daily_workload      : Total hours + overload flag
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger("backend.mcp_server.schedule_tools")


# ─── Internal async helpers ──────────────────────────────────────────────────

async def _get_db():
    from backend.models.database import get_db
    return await get_db()


# ─── Tool Implementations ─────────────────────────────────────────────────────

async def get_schedule(status_filter: Optional[str] = None) -> dict:
    """
    Retrieve the user's task list from the database.

    Args:
        status_filter: Optional — one of 'pending', 'in_progress', 'completed'.
                       If None, returns all non-completed tasks.

    Returns:
        {
            "success": bool,
            "tasks": [ { id, task, deadline, est_hours, priority, status, created_at } ],
            "count": int
        }
    """
    db = await _get_db()
    try:
        if status_filter:
            cursor = await db.execute(
                "SELECT id, task, deadline, est_hours, priority, status, created_at "
                "FROM schedules WHERE status = ? ORDER BY deadline",
                (status_filter,)
            )
        else:
            cursor = await db.execute(
                "SELECT id, task, deadline, est_hours, priority, status, created_at "
                "FROM schedules WHERE status != 'completed' ORDER BY deadline"
            )
        rows = await cursor.fetchall()
        tasks = [
            {
                "id": r[0], "task": r[1], "deadline": r[2],
                "est_hours": r[3], "priority": r[4],
                "status": r[5], "created_at": str(r[6]),
            }
            for r in rows
        ]
        return {"success": True, "tasks": tasks, "count": len(tasks)}
    except Exception as e:
        logger.error(f"get_schedule error: {e}")
        return {"success": False, "tasks": [], "count": 0, "error": str(e)}
    finally:
        await db.close()


async def get_upcoming_deadlines(days: int = 3) -> dict:
    """
    Retrieve tasks whose deadline falls within the next `days` days.

    Args:
        days: Number of days ahead to look (default: 3).

    Returns:
        {
            "success": bool,
            "tasks": [...],
            "count": int,
            "window_days": int
        }
    """
    db = await _get_db()
    try:
        today = datetime.now().date()
        cutoff = today + timedelta(days=days)

        cursor = await db.execute(
            "SELECT id, task, deadline, est_hours, priority, status, created_at "
            "FROM schedules WHERE status != 'completed' ORDER BY deadline"
        )
        rows = await cursor.fetchall()

        upcoming = []
        for r in rows:
            try:
                # Support multiple deadline formats
                deadline_str = str(r[2]).strip()
                for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
                    try:
                        deadline_date = datetime.strptime(deadline_str, fmt).date()
                        break
                    except ValueError:
                        continue
                else:
                    # Could not parse — include as upcoming by default
                    deadline_date = today

                if today <= deadline_date <= cutoff:
                    upcoming.append({
                        "id": r[0], "task": r[1], "deadline": r[2],
                        "est_hours": r[3], "priority": r[4],
                        "status": r[5], "created_at": str(r[6]),
                        "days_until_deadline": (deadline_date - today).days,
                    })
            except Exception:
                continue

        return {
            "success": True,
            "tasks": upcoming,
            "count": len(upcoming),
            "window_days": days,
        }
    except Exception as e:
        logger.error(f"get_upcoming_deadlines error: {e}")
        return {"success": False, "tasks": [], "count": 0, "error": str(e)}
    finally:
        await db.close()


async def add_task(
    task: str,
    deadline: str,
    est_hours: float = 2.0,
    priority: str = "Medium",
) -> dict:
    """
    Add a new task to the schedule.

    Args:
        task:      Task description.
        deadline:  Deadline string (e.g. '2026-03-10').
        est_hours: Estimated hours to complete (default: 2.0).
        priority:  'High', 'Medium', or 'Low' (default: 'Medium').

    Returns:
        { "success": bool, "task": { ... } | None, "error": str | None }
    """
    # Normalize priority
    priority = priority.capitalize()
    if priority not in ("High", "Medium", "Low"):
        priority = "Medium"

    # Clamp hours
    est_hours = max(0.5, min(24.0, float(est_hours)))

    db = await _get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO schedules (task, deadline, est_hours, priority) "
            "VALUES (?, ?, ?, ?) RETURNING id, task, deadline, est_hours, priority, status, created_at",
            (task, deadline, est_hours, priority)
        )
        row = await cursor.fetchone()
        await db.commit()
        if row:
            return {
                "success": True,
                "task": {
                    "id": row[0], "task": row[1], "deadline": row[2],
                    "est_hours": row[3], "priority": row[4],
                    "status": row[5], "created_at": str(row[6]),
                },
            }
        return {"success": False, "task": None, "error": "Insert returned no row"}
    except Exception as e:
        logger.error(f"add_task error: {e}")
        return {"success": False, "task": None, "error": str(e)}
    finally:
        await db.close()


async def update_task_status(task_id: int, status: str) -> dict:
    """
    Update the status of an existing task.

    Args:
        task_id: The task's integer ID.
        status:  One of 'pending', 'in_progress', 'completed'.

    Returns:
        { "success": bool, "task": { ... } | None, "error": str | None }
    """
    valid_statuses = ("pending", "in_progress", "completed")
    if status not in valid_statuses:
        return {
            "success": False,
            "task": None,
            "error": f"Invalid status '{status}'. Must be one of: {valid_statuses}",
        }

    db = await _get_db()
    try:
        await db.execute(
            "UPDATE schedules SET status = ? WHERE id = ?",
            (status, task_id)
        )
        await db.commit()

        cursor = await db.execute(
            "SELECT id, task, deadline, est_hours, priority, status, created_at "
            "FROM schedules WHERE id = ?",
            (task_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return {"success": False, "task": None, "error": f"Task {task_id} not found"}

        return {
            "success": True,
            "task": {
                "id": row[0], "task": row[1], "deadline": row[2],
                "est_hours": row[3], "priority": row[4],
                "status": row[5], "created_at": str(row[6]),
            },
        }
    except Exception as e:
        logger.error(f"update_task_status error: {e}")
        return {"success": False, "task": None, "error": str(e)}
    finally:
        await db.close()


async def delete_task(task_id: int) -> dict:
    """
    Permanently delete a task from the schedule.

    Args:
        task_id: The integer ID of the task to delete.

    Returns:
        { "success": bool, "deleted_task": { ... } | None, "error": str | None }
    """
    db = await _get_db()
    try:
        # Fetch the task first so we can return it in the result
        cursor = await db.execute(
            "SELECT id, task, deadline, est_hours, priority, status, created_at "
            "FROM schedules WHERE id = ?",
            (task_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return {
                "success": False,
                "deleted_task": None,
                "error": f"Task dengan ID {task_id} tidak ditemukan.",
            }

        deleted_task = {
            "id": row[0], "task": row[1], "deadline": row[2],
            "est_hours": row[3], "priority": row[4],
            "status": row[5], "created_at": str(row[6]),
        }

        await db.execute("DELETE FROM schedules WHERE id = ?", (task_id,))
        await db.commit()

        logger.info(f"Deleted task id={task_id}: '{deleted_task['task']}'")
        return {"success": True, "deleted_task": deleted_task, "error": None}

    except Exception as e:
        logger.error(f"delete_task error: {e}")
        return {"success": False, "deleted_task": None, "error": str(e)}
    finally:
        await db.close()


async def get_daily_workload() -> dict:
    """
    Calculate the current daily workload from active tasks.

    Returns:
        {
            "success": bool,
            "total_hours": float,
            "task_count": int,
            "max_daily_hours": int,        # 8h hard limit (Layer 3 policy)
            "overloaded": bool,
            "overload_by_hours": float,    # How many hours over the limit
            "high_priority_count": int,
            "pending_count": int,
            "in_progress_count": int,
        }
    """
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT SUM(est_hours), COUNT(*), "
            "SUM(CASE WHEN priority='High' THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN status='in_progress' THEN 1 ELSE 0 END) "
            "FROM schedules WHERE status != 'completed'"
        )
        row = await cursor.fetchone()
        total_hours = float(row[0] or 0)
        task_count = int(row[1] or 0)
        high_priority = int(row[2] or 0)
        pending = int(row[3] or 0)
        in_progress = int(row[4] or 0)

        max_hours = 8
        overloaded = total_hours > max_hours

        return {
            "success": True,
            "total_hours": round(total_hours, 2),
            "task_count": task_count,
            "max_daily_hours": max_hours,
            "overloaded": overloaded,
            "overload_by_hours": round(max(0.0, total_hours - max_hours), 2),
            "high_priority_count": high_priority,
            "pending_count": pending,
            "in_progress_count": in_progress,
        }
    except Exception as e:
        logger.error(f"get_daily_workload error: {e}")
        return {
            "success": False,
            "total_hours": 0,
            "task_count": 0,
            "max_daily_hours": 8,
            "overloaded": False,
            "overload_by_hours": 0,
            "error": str(e),
        }
    finally:
        await db.close()


# ─── Tool Registry ────────────────────────────────────────────────────────────

SCHEDULE_TOOLS = {
    "get_schedule": {
        "fn": get_schedule,
        "description": "Get user's active task list from database",
        "destructive": False,
        "parameters": {
            "status_filter": {
                "type": "string",
                "description": "Filter by status: 'pending', 'in_progress', 'completed'. Omit for all active.",
                "required": False,
            }
        },
    },
    "get_upcoming_deadlines": {
        "fn": get_upcoming_deadlines,
        "description": "Get tasks with deadlines in the next N days",
        "destructive": False,
        "parameters": {
            "days": {
                "type": "integer",
                "description": "Number of days to look ahead (default: 3)",
                "required": False,
            }
        },
    },
    "add_task": {
        "fn": add_task,
        "description": "Tambahkan tugas baru ke jadwal user",
        "destructive": True,
        "confirm_message": "Ara akan menambahkan tugas baru ke jadwal kamu.",
        "parameters": {
            "task": {"type": "string", "description": "Deskripsi tugas", "required": True},
            "deadline": {"type": "string", "description": "Deadline (YYYY-MM-DD)", "required": True},
            "est_hours": {"type": "number", "description": "Estimasi jam (default: 2.0)", "required": False},
            "priority": {"type": "string", "description": "'High', 'Medium', atau 'Low'", "required": False},
        },
    },
    "update_task_status": {
        "fn": update_task_status,
        "description": "Perbarui status tugas (pending/in_progress/completed)",
        "destructive": True,
        "confirm_message": "Ara akan mengubah status tugas ini.",
        "parameters": {
            "task_id": {"type": "integer", "description": "ID tugas", "required": True},
            "status": {
                "type": "string",
                "description": "'pending', 'in_progress', atau 'completed'",
                "required": True,
            },
        },
    },
    "delete_task": {
        "fn": delete_task,
        "description": "Hapus tugas secara permanen dari jadwal",
        "destructive": True,
        "confirm_message": "Ara akan menghapus tugas ini secara permanen. Tindakan ini tidak dapat dibatalkan.",
        "parameters": {
            "task_id": {"type": "integer", "description": "ID tugas yang akan dihapus", "required": True},
        },
    },
    "get_daily_workload": {
        "fn": get_daily_workload,
        "description": "Get total workload hours and overload status for today",
        "destructive": False,
        "parameters": {},
    },
}
