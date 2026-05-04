"""
Database connection and table setup.
"""
import logging
import aiosqlite
from pathlib import Path

logger = logging.getLogger("backend.models.database")

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
                chat_id         TEXT NOT NULL DEFAULT 'default',
                mode            TEXT NOT NULL DEFAULT 'conversation'
                    CHECK(mode IN ('conversation', 'learning')),
                role            TEXT NOT NULL CHECK(role IN ('user','assistant')),
                content         TEXT NOT NULL,
                emotion         TEXT DEFAULT 'NEUTRAL',
                intent          TEXT DEFAULT 'GENERAL_CHAT',
                intensity       REAL DEFAULT 0.0,
                layer3_status   TEXT DEFAULT 'PASS',
                layer4_rewritten INTEGER DEFAULT 0,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS learning_sessions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                mode            TEXT NOT NULL DEFAULT 'learning'
                    CHECK(mode IN ('lesson', 'quiz', 'review')),
                topic           TEXT NOT NULL,
                subtopic        TEXT DEFAULT '',
                user_level      TEXT NOT NULL DEFAULT 'pemula'
                    CHECK(user_level IN ('pemula', 'menengah', 'mahir')),
                status          TEXT NOT NULL DEFAULT 'active'
                    CHECK(status IN ('active', 'paused', 'completed', 'abandoned')),
                current_section INTEGER DEFAULT 0,
                total_sections  INTEGER DEFAULT 0,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at    TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS lesson_progress (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      INTEGER NOT NULL,
                material_id     INTEGER,
                section_type    TEXT NOT NULL,
                content         TEXT NOT NULL,
                completed       INTEGER DEFAULT 0,
                time_spent_secs INTEGER DEFAULT 0,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES learning_sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS quiz_results (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      INTEGER NOT NULL,
                question        TEXT NOT NULL,
                question_type   TEXT NOT NULL,
                options         TEXT DEFAULT '[]',
                correct_answer  TEXT,
                user_answer     TEXT,
                is_correct      INTEGER,
                explanation     TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES learning_sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS user_topic_mastery (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                topic           TEXT NOT NULL UNIQUE,
                mastery_level   REAL NOT NULL DEFAULT 0.0,
                total_attempts  INTEGER DEFAULT 0,
                correct_attempts INTEGER DEFAULT 0,
                last_reviewed   TIMESTAMP,
                ease_factor     REAL DEFAULT 2.5,
                interval_days   INTEGER DEFAULT 1,
                next_review     TIMESTAMP,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS user_profiles (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                current_level   TEXT NOT NULL DEFAULT 'pemula'
                    CHECK(current_level IN ('pemula', 'menengah', 'mahir')),
                learning_goals  TEXT DEFAULT '[]',
                preferred_topics TEXT DEFAULT '[]',
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        await db.commit()

        await migrate_db(db)
    finally:
        await db.close()


async def migrate_db(db: aiosqlite.Connection):
    """Run database migrations for schema updates."""
    migrations = [
        ("chat_history", "mode", "TEXT NOT NULL DEFAULT 'conversation'"),
        ("chat_history", "layer3_status", "TEXT DEFAULT 'PASS'"),
        ("chat_history", "layer4_rewritten", "INTEGER DEFAULT 0"),
        ("chat_history", "chat_id", "TEXT NOT NULL DEFAULT 'default'"),
    ]

    for table, column, column_def in migrations:
        try:
            await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_def}")
            await db.commit()
            logger.info(f"Migration added column {column} to {table}")
        except Exception as e:
            if "duplicate column" in str(e).lower():
                pass
            else:
                logger.warning(f"Migration check for {table}.{column}: {e}")
