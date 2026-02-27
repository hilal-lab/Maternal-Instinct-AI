"""
Seed data for development/demo.
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend.models.database import init_db, get_db


DEMO_SCHEDULES = [
    ("Finalisasi Jurnal AI Safety", "2026-03-03 17:00", 5, "High"),
    ("Presentasi Project Agentic AI", "2026-03-04 10:00", 3, "High"),
    ("Tugas Struktur Data - Linked List", "2026-03-05 23:59", 2, "Medium"),
    ("Review Paper RAG Security", "2026-03-06 12:00", 2, "Medium"),
    ("Latihan Soal Kalkulus Integral", "2026-03-07 08:00", 1.5, "Low"),
    ("Lab Report - Jaringan Komputer", "2026-03-07 23:59", 3, "Medium"),
]

DEMO_NOTES = [
    (
        "Catatan Kuliah: Agentic AI",
        "## Agentic AI\n\n### Definisi\nAgentic AI = sistem AI otonom yang bisa observe, understand, act.\n\n### Design Patterns\n1. **ReAct** - Reasoning + Acting\n2. **Multi-Agent Collaboration**\n3. **Tool Use** - Agent memanggil API/tools\n\n### Key Paper\n- Viradia et al. (2025) - Agentic AI for Medicare",
        "AI & Machine Learning"
    ),
    (
        "Catatan: RAG System Security",
        "## RAG Security\n\n### Contextual Prompt Injection\n- Malicious instructions in retrieved docs\n\n### Mitigation\n1. Input sanitization\n2. Output validation (guardrails)\n3. Layered defense architecture",
        "Cybersecurity"
    ),
    (
        "Rumus Kalkulus - Integral",
        "## Integral\n\n### Integral Tak Tentu\n- int x^n dx = x^(n+1)/(n+1) + C\n\n### Teknik Integrasi\n1. Substitusi\n2. Parsial\n3. Trigonometri",
        "Matematika"
    ),
    (
        "Tips Manajemen Waktu",
        "## Time Management\n\n### Pomodoro: 25 min focus + 5 min break\n### Eisenhower Matrix: Urgent-Important\n### Eat the frog: hardest task first",
        "Produktivitas"
    ),
]


async def seed():
    """Seed database with demo data."""
    await init_db()
    db = await get_db()
    try:
        cursor = await db.execute("SELECT COUNT(*) FROM schedules")
        count = (await cursor.fetchone())[0]
        if count > 0:
            print(f"  DB already seeded ({count} schedules). Skipping.")
            return

        await db.executemany(
            "INSERT INTO schedules (task, deadline, est_hours, priority) VALUES (?, ?, ?, ?)",
            DEMO_SCHEDULES
        )
        await db.executemany(
            "INSERT INTO notes (title, content, subject) VALUES (?, ?, ?)",
            DEMO_NOTES
        )
        await db.commit()
        print(f"  Seeded {len(DEMO_SCHEDULES)} schedules + {len(DEMO_NOTES)} notes.")
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(seed())
