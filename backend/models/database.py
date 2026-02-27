"""
Database connection and table setup.
"""
import aiosqlite
from pathlib import Path

DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "student.db"


async def get_db() -> aiosqlite.Connection:
    """Get an async database connection."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(DB_PATH))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_db():
    """Create all tables if they don't exist."""
    db = await get_db()
    try:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS schedules (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                task        TEXT NOT NULL,
                deadline    TEXT NOT NULL,
                est_hours   REAL NOT NULL DEFAULT 2,
                priority    TEXT NOT NULL DEFAULT 'Medium'
                    CHECK(priority IN ('High','Medium','Low')),
                status      TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending','in_progress','completed')),
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS notes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT NOT NULL,
                content     TEXT NOT NULL,
                subject     TEXT DEFAULT '',
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS documents (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                filename        TEXT NOT NULL,
                content_type    TEXT DEFAULT 'text/plain',
                chunk_count     INTEGER DEFAULT 0,
                file_size       INTEGER DEFAULT 0,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS chat_history (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                role            TEXT NOT NULL CHECK(role IN ('user','assistant')),
                content         TEXT NOT NULL,
                emotion         TEXT DEFAULT 'NEUTRAL',
                intent          TEXT DEFAULT 'GENERAL_CHAT',
                intensity       REAL DEFAULT 0.0,
                layer3_status   TEXT DEFAULT 'PASS',
                layer4_rewritten INTEGER DEFAULT 0,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        await db.commit()
    finally:
        await db.close()
