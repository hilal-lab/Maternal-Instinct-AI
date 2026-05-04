"""
MCP Notes Tools — Structured tool interface for notes/knowledge management.

Wraps the notes table to expose callable tools to Layer 2 agents.

Tools exposed:
  - get_notes     : List all notes (optionally filtered by subject)
  - search_notes  : Full-text search across note titles and content
  - add_note      : Create a new note
  - get_note      : Retrieve a single note by ID
"""
import logging
from typing import Optional

logger = logging.getLogger("backend.mcp_server.notes_tools")


# ─── Internal DB helper ───────────────────────────────────────────────────────

async def _get_db():
    from backend.models.database import get_db
    return await get_db()


# ─── Tool Implementations ─────────────────────────────────────────────────────

async def get_notes(subject: Optional[str] = None, limit: int = 20) -> dict:
    """
    Retrieve notes from the database.

    Args:
        subject: Optional subject filter (partial match, case-insensitive).
        limit:   Max number of notes to return (default: 20).

    Returns:
        {
            "success": bool,
            "notes": [ { id, title, content, subject, created_at, updated_at } ],
            "count": int
        }
    """
    db = await _get_db()
    try:
        if subject:
            cursor = await db.execute(
                "SELECT id, title, content, subject, created_at, updated_at "
                "FROM notes WHERE LOWER(subject) LIKE ? ORDER BY updated_at DESC LIMIT ?",
                (f"%{subject.lower()}%", limit)
            )
        else:
            cursor = await db.execute(
                "SELECT id, title, content, subject, created_at, updated_at "
                "FROM notes ORDER BY updated_at DESC LIMIT ?",
                (limit,)
            )
        rows = await cursor.fetchall()
        notes = [
            {
                "id": r[0], "title": r[1], "content": r[2],
                "subject": r[3], "created_at": str(r[4]), "updated_at": str(r[5]),
            }
            for r in rows
        ]
        return {"success": True, "notes": notes, "count": len(notes)}
    except Exception as e:
        logger.error(f"get_notes error: {e}")
        return {"success": False, "notes": [], "count": 0, "error": str(e)}
    finally:
        await db.close()


async def search_notes(query: str, limit: int = 10) -> dict:
    """
    Full-text search across note titles and content.

    Args:
        query: Search string (partial match in title or content).
        limit: Max results to return (default: 10).

    Returns:
        {
            "success": bool,
            "results": [ { id, title, snippet, subject, relevance_hint } ],
            "count": int,
            "query": str
        }
    """
    if not query or not query.strip():
        return {"success": False, "results": [], "count": 0, "error": "Empty query"}

    db = await _get_db()
    try:
        pattern = f"%{query.lower()}%"
        cursor = await db.execute(
            "SELECT id, title, content, subject, updated_at "
            "FROM notes WHERE LOWER(title) LIKE ? OR LOWER(content) LIKE ? "
            "ORDER BY updated_at DESC LIMIT ?",
            (pattern, pattern, limit)
        )
        rows = await cursor.fetchall()

        results = []
        for r in rows:
            content = r[2] or ""
            # Build a short snippet around the first hit
            lower_content = content.lower()
            hit_idx = lower_content.find(query.lower())
            if hit_idx >= 0:
                start = max(0, hit_idx - 60)
                end = min(len(content), hit_idx + 120)
                snippet = ("..." if start > 0 else "") + content[start:end] + ("..." if end < len(content) else "")
            else:
                snippet = content[:150] + ("..." if len(content) > 150 else "")

            # Simple relevance hint: title match > content match
            title_match = query.lower() in (r[1] or "").lower()
            results.append({
                "id": r[0],
                "title": r[1],
                "snippet": snippet,
                "subject": r[3],
                "updated_at": str(r[4]),
                "relevance_hint": "title_match" if title_match else "content_match",
            })

        return {"success": True, "results": results, "count": len(results), "query": query}
    except Exception as e:
        logger.error(f"search_notes error: {e}")
        return {"success": False, "results": [], "count": 0, "error": str(e)}
    finally:
        await db.close()


async def add_note(title: str, content: str, subject: str = "") -> dict:
    """
    Create a new note.

    Args:
        title:   Note title (required).
        content: Note body text (required).
        subject: Optional subject/tag (e.g. 'Matematika', 'Fisika').

    Returns:
        { "success": bool, "note": { ... } | None, "error": str | None }
    """
    if not title or not title.strip():
        return {"success": False, "note": None, "error": "Title cannot be empty"}
    if not content or not content.strip():
        return {"success": False, "note": None, "error": "Content cannot be empty"}

    db = await _get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO notes (title, content, subject) VALUES (?, ?, ?) "
            "RETURNING id, title, content, subject, created_at, updated_at",
            (title.strip(), content.strip(), subject.strip())
        )
        row = await cursor.fetchone()
        await db.commit()
        if row:
            return {
                "success": True,
                "note": {
                    "id": row[0], "title": row[1], "content": row[2],
                    "subject": row[3], "created_at": str(row[4]), "updated_at": str(row[5]),
                },
            }
        return {"success": False, "note": None, "error": "Insert returned no row"}
    except Exception as e:
        logger.error(f"add_note error: {e}")
        return {"success": False, "note": None, "error": str(e)}
    finally:
        await db.close()


async def get_note(note_id: int) -> dict:
    """
    Retrieve a single note by its ID.

    Args:
        note_id: The note's integer ID.

    Returns:
        { "success": bool, "note": { ... } | None, "error": str | None }
    """
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT id, title, content, subject, created_at, updated_at "
            "FROM notes WHERE id = ?",
            (note_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return {"success": False, "note": None, "error": f"Note {note_id} not found"}
        return {
            "success": True,
            "note": {
                "id": row[0], "title": row[1], "content": row[2],
                "subject": row[3], "created_at": str(row[4]), "updated_at": str(row[5]),
            },
        }
    except Exception as e:
        logger.error(f"get_note error: {e}")
        return {"success": False, "note": None, "error": str(e)}
    finally:
        await db.close()


# ─── Tool Registry ────────────────────────────────────────────────────────────

NOTES_TOOLS = {
    "get_notes": {
        "fn": get_notes,
        "description": "List user notes, optionally filtered by subject",
        "parameters": {
            "subject": {
                "type": "string",
                "description": "Optional subject filter (partial match)",
                "required": False,
            },
            "limit": {
                "type": "integer",
                "description": "Max notes to return (default: 20)",
                "required": False,
            },
        },
    },
    "search_notes": {
        "fn": search_notes,
        "description": "Full-text search across note titles and content",
        "parameters": {
            "query": {"type": "string", "description": "Search string", "required": True},
            "limit": {"type": "integer", "description": "Max results (default: 10)", "required": False},
        },
    },
    "add_note": {
        "fn": add_note,
        "description": "Create a new note in the user's knowledge base",
        "parameters": {
            "title": {"type": "string", "description": "Note title", "required": True},
            "content": {"type": "string", "description": "Note content/body", "required": True},
            "subject": {"type": "string", "description": "Optional subject tag", "required": False},
        },
    },
    "get_note": {
        "fn": get_note,
        "description": "Retrieve a specific note by its ID",
        "parameters": {
            "note_id": {"type": "integer", "description": "Note ID", "required": True},
        },
    },
}
