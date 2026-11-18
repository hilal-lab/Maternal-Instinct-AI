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
    """
    db = await _get_db()
    try:
        columns = (
            "id, task, deadline, scheduled_date, start_time, end_time, "
            "est_hours, priority, status, created_at"
        )
        if status_filter:
            cursor = await db.execute(
                f"SELECT {columns} FROM schedules "
                "WHERE status = ? ORDER BY deadline, start_time",
                (status_filter,)
            )
        else:
            cursor = await db.execute(
                f"SELECT {columns} FROM schedules "
                "WHERE status != 'completed' "
                "ORDER BY deadline, start_time"
            )
        rows = await cursor.fetchall()
        tasks = [
            {
                "id": r[0],
                "task": r[1],
                "deadline": r[2],
                "scheduled_date": r[3],
                "start_time": r[4],
                "end_time": r[5],
                "est_hours": r[6],
                "priority": r[7],
                "status": r[8],
                "created_at": str(r[9]),
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
    scheduled_date: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    est_hours: float = 2.0,
    priority: str = "Medium",
    deadline: Optional[str] = None,
) -> dict:
    """
    Add a new task to the schedule.

    Args:
        task:           Task description.
        scheduled_date: Scheduled date (YYYY-MM-DD).
        start_time:     Start time (HH:MM).
        end_time:       End time (HH:MM).
        est_hours:      Estimated hours to complete (default: 2.0).
        priority:       'High', 'Medium', or 'Low' (default: 'Medium').
        deadline:       Legacy deadline parameter for backward compatibility.

    Returns:
        { "success": bool, "task": { ... } | None, "error": str | None }
    """
    legacy_call = scheduled_date is None and deadline is not None

    if legacy_call:
        scheduled_date = deadline

    if not scheduled_date:
        return {
            "success": False,
            "task": None,
            "error": "scheduled_date is required.",
        }

    if not legacy_call and (start_time is None or end_time is None):
        return {
            "success": False,
            "task": None,
            "error": (
                "scheduled_date, start_time, and end_time are required "
                "for new scheduled tasks."
            ),
        }

    if (start_time is None) != (end_time is None):
        return {
            "success": False,
            "task": None,
            "error": "start_time and end_time must be provided together.",
        }

    # Validate task description
    task = str(task).strip() if task is not None else ""
    if not task:
        return {
            "success": False,
            "task": None,
            "error": "Task description is required.",
        }

    # Validate date format
    try:
        parsed_date = datetime.strptime(scheduled_date, "%Y-%m-%d").date()
        if parsed_date.isoformat() != scheduled_date:
            raise ValueError
    except (TypeError, ValueError):
        return {
            "success": False,
            "task": None,
            "error": "scheduled_date must use YYYY-MM-DD format.",
        }

    # Validate time range when provided
    if start_time is not None:
        try:
            parsed_start = datetime.strptime(start_time, "%H:%M").time()
            parsed_end = datetime.strptime(end_time, "%H:%M").time()

            if parsed_start.strftime("%H:%M") != start_time:
                raise ValueError
            if parsed_end.strftime("%H:%M") != end_time:
                raise ValueError
            if parsed_end <= parsed_start:
                raise ValueError

        except (TypeError, ValueError):
            return {
                "success": False,
                "task": None,
                "error": (
                    "Invalid time range. Use HH:MM format and ensure "
                    "end_time is later than start_time."
                ),
            }

    # Normalize priority
    priority = (priority or "Medium").capitalize()
    if priority not in ("High", "Medium", "Low"):
        priority = "Medium"

    # Validate and clamp estimated hours
    try:
        est_hours = float(est_hours)
        if not 0.5 <= est_hours <= 24.0:
            return {
                "success": False,
                "task": None,
                "error": "est_hours must be between 0.5 and 24.0.",
            }
    except (TypeError, ValueError):
        return {
            "success": False,
            "task": None,
            "error": "est_hours must be a valid number.",
        }

    db = await _get_db()
    try:
        cursor = await db.execute(
            """
            INSERT INTO schedules (
                task, deadline, scheduled_date,
                start_time, end_time, est_hours,
                priority
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            RETURNING id, task, deadline, scheduled_date,
                start_time, end_time, est_hours,
                priority, status, created_at
            """,
            (
                task,
                scheduled_date,
                scheduled_date,
                start_time,
                end_time,
                est_hours,
                priority,
            ),
        )
        row = await cursor.fetchone()
        await db.commit()
        if row:
            return {
                "success": True,
                "task": {
                    "id": row[0],
                    "task": row[1],
                    "deadline": row[2],
                    "scheduled_date": row[3],
                    "start_time": row[4],
                    "end_time": row[5],
                    "est_hours": row[6],
                    "priority": row[7],
                    "status": row[8],
                    "created_at": str(row[9]),
                },
            }
        return {
            "success": False,
            "task": None,
            "error": "Insert returned no row",
        }
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


async def delete_task(
    task_id: int,
    expected_snapshot: Optional[dict] = None,
) -> dict:
    """
    Delete a task only if its current identity still matches the
    snapshot that was shown to and approved by the user.
    """
    if type(task_id) is not int or task_id <= 0:
        return {
            "success": False,
            "deleted_task": None,
            "error": "ID tugas tidak valid.",
        }

    required_fields = {
        "id",
        "task",
        "scheduled_date",
        "start_time",
        "end_time",
        "status",
    }

    if (
        not isinstance(expected_snapshot, dict)
        or not required_fields.issubset(expected_snapshot)
        or expected_snapshot.get("id") != task_id
    ):
        return {
            "success": False,
            "deleted_task": None,
            "error": (
                "Snapshot konfirmasi tidak valid. "
                "Penghapusan hanya dapat dilakukan melalui "
                "konfirmasi yang telah diverifikasi."
            ),
        }

    db = await _get_db()

    try:
        # Acquire the SQLite write reservation before reading the task.
        # This prevents another writer from changing it between SELECT and DELETE.
        await db.execute("BEGIN IMMEDIATE")

        cursor = await db.execute(
            """
            SELECT
                id, task, deadline, scheduled_date,
                start_time, end_time, est_hours,
                priority, status, created_at
            FROM schedules
            WHERE id = ?
            """,
            (task_id,),
        )
        row = await cursor.fetchone()

        if row is None:
            await db.rollback()
            return {
                "success": False,
                "deleted_task": None,
                "error": f"Tugas dengan ID {task_id} tidak ditemukan.",
            }

        # Delete only if the identity still matches the approved snapshot.
        cursor = await db.execute(
            """
            DELETE FROM schedules
            WHERE id = ?
              AND task = ?
              AND scheduled_date IS ?
              AND start_time IS ?
              AND end_time IS ?
              AND status IS ?
            """,
            (
                task_id,
                expected_snapshot["task"],
                expected_snapshot["scheduled_date"],
                expected_snapshot["start_time"],
                expected_snapshot["end_time"],
                expected_snapshot["status"],
            ),
        )

        if cursor.rowcount != 1:
            await db.rollback()
            return {
                "success": False,
                "deleted_task": None,
                "error": (
                    f"Data tugas dengan ID {task_id} berubah "
                    "sejak konfirmasi. Penghapusan dibatalkan."
                ),
            }

        await db.commit()

        deleted_task = {
            "id": row[0],
            "task": row[1],
            "deadline": row[2],
            "scheduled_date": row[3],
            "start_time": row[4],
            "end_time": row[5],
            "est_hours": row[6],
            "priority": row[7],
            "status": row[8],
            "created_at": str(row[9]),
        }

        logger.info(
            f"Deleted task id={task_id}: '{deleted_task['task']}'"
        )

        return {
            "success": True,
            "deleted_task": deleted_task,
            "error": None,
        }

    except Exception as e:
        await db.rollback()
        logger.error(f"delete_task error: {e}")
        return {
            "success": False,
            "deleted_task": None,
            "error": str(e),
        }

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
        "description": (
            "Tambahkan tugas baru dengan tanggal, jam mulai, dan jam selesai."
        ),
        "destructive": True,
        "confirm_message": "Ara akan menambahkan tugas baru ke jadwal kamu.",
        "parameters": {
            "task": {
                "type": "string",
                "description": "Deskripsi tugas",
                "required": True,
            },
            "scheduled_date": {
                "type": "string",
                "description": "Tanggal jadwal dalam format YYYY-MM-DD",
                "required": True,
            },
            "start_time": {
                "type": "string",
                "description": "Jam mulai dalam format HH:MM",
                "required": True,
            },
            "end_time": {
                "type": "string",
                "description": "Jam selesai dalam format HH:MM",
                "required": True,
            },
            "est_hours": {
                "type": "number",
                "description": "Estimasi durasi pengerjaan",
                "required": False,
            },
            "priority": {
                "type": "string",
                "description": "'High', 'Medium', atau 'Low'",
                "required": False,
            },
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
