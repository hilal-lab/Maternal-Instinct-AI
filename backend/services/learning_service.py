"""
Learning Service - Manages learning sessions and integrates with chat.

Provides high-level API for learning mode operations.
"""
import json
import logging
from typing import Optional
from datetime import datetime

from backend.models.database import get_db
from backend.models.schemas import (
    UserLevel, ChatMode, LearningSessionResponse,
    LessonSection, QuizQuestion, QuizResult
)
from backend.core.learning_orchestrator import learning_orchestrator


logger = logging.getLogger("backend.services.learning_service")


class LearningService:
    def __init__(self):
        self.orchestrator = learning_orchestrator

    async def start_learning(
        self,
        topic: str,
        subtopic: str = "",
        level: UserLevel = UserLevel.PEMULA,
        mode: str = "lesson"
    ) -> dict:
        """Start a new learning session."""
        session = await self.orchestrator.start_lesson(topic, subtopic, level)
        
        if mode == "lesson":
            sections = await self.orchestrator.generate_lesson_content(
                session.id, topic, level
            )
            return {
                "session": session,
                "type": "lesson",
                "sections": [s.model_dump() for s in sections],
                "message": self._format_lesson_intro(topic, level)
            }
        
        elif mode == "quiz":
            questions = await self.orchestrator.generate_quiz(topic, level)
            return {
                "session": session,
                "type": "quiz",
                "questions": [q.model_dump() for q in questions],
                "message": f"📝 Kuis tentang **{topic}** (Level: {level.value})\n\n"
                           f"Pertanyaan: {len(questions)}\n\n"
                           f"Ketik jawaban untuk setiap pertanyaan!"
            }
        
        return {"session": session, "type": mode}

    async def continue_learning(
        self,
        session_id: int,
        user_response: str = ""
    ) -> dict:
        """Continue a learning session."""
        db = await get_db()
        try:
            cursor = await db.execute(
                """SELECT mode, topic, subtopic, user_level, status, current_section, total_sections
                   FROM learning_sessions WHERE id = ?""",
                (session_id,)
            )
            row = await cursor.fetchone()
            
            if not row:
                return {"error": "Session not found"}
            
            mode, topic, subtopic, level, status, current_section, total_sections = row
            
            if status == "completed":
                return {
                    "done": True,
                    "message": "✅ Sesi ini sudah selesai!",
                    "session_id": session_id
                }
            
            if mode == "lesson":
                return await self._continue_lesson(
                    session_id, topic, level, current_section, total_sections
                )
            
            return {"error": f"Unknown mode: {mode}"}
        finally:
            await db.close()

    async def _continue_lesson(
        self,
        session_id: int,
        topic: str,
        level: str,
        current_section: int,
        total_sections: int
    ) -> dict:
        """Continue lesson by providing next section."""
        next_section = current_section + 1
        
        if next_section >= total_sections:
            await self._complete_session(session_id)
            return {
                "done": True,
                "message": self._format_lesson_summary(topic),
                "session_id": session_id
            }
        
        db = await get_db()
        try:
            cursor = await db.execute(
                """SELECT section_type, content FROM lesson_progress 
                   WHERE session_id = ? AND completed = 0 ORDER BY id LIMIT 1""",
                (session_id,)
            )
            row = await cursor.fetchone()
            
            if row:
                await db.execute(
                    "UPDATE learning_sessions SET current_section = ? WHERE id = ?",
                    (next_section, session_id)
                )
                await db.execute(
                    "UPDATE lesson_progress SET completed = 1 WHERE session_id = ? AND section_type = ?",
                    (session_id, row[0])
                )
                await db.commit()
                
                return {
                    "done": False,
                    "section": next_section + 1,
                    "total": total_sections,
                    "type": row[0],
                    "content": row[1],
                    "message": f"Lanjut ke bagian {next_section + 1}..."
                }
            
            return {"done": True, "message": "✅ Semua bagian sudah selesai!"}
        finally:
            await db.close()

    async def submit_quiz(
        self,
        session_id: int,
        answers: dict[int, str]
    ) -> dict:
        """Submit quiz answers and get results."""
        db = await get_db()
        try:
            cursor = await db.execute(
                """SELECT qr.question, qr.question_type, qr.options, qr.correct_answer, qr.explanation
                   FROM quiz_results qr WHERE qr.session_id = ?""",
                (session_id,)
            )
            rows = await cursor.fetchall()
            
            questions = [
                QuizQuestion(
                    id=i+1,
                    type=row[1],
                    question=row[0],
                    options=json.loads(row[2]) if row[2] else [],
                    correct_answer=row[3],
                    explanation=row[4]
                ) for i, row in enumerate(rows)
            ]
            
            result = await self.orchestrator.evaluate_quiz(
                session_id, answers, questions
            )
            
            return {
                "session_id": session_id,
                "type": "quiz_result",
                **result
            }
        finally:
            await db.close()

    async def get_user_topics(self) -> list[str]:
        """Get list of topics user has learned about."""
        db = await get_db()
        try:
            cursor = await db.execute(
                """SELECT DISTINCT topic FROM learning_sessions 
                   ORDER BY created_at DESC LIMIT 20"""
            )
            rows = await cursor.fetchall()
            return [row[0] for row in rows]
        finally:
            await db.close()

    async def get_topic_progress(self, topic: str) -> dict:
        """Get detailed progress for a topic."""
        mastery = await self.orchestrator.get_topic_mastery(topic)
        sessions = await self.orchestrator.get_session_history(limit=10)
        
        topic_sessions = [s for s in sessions if s.topic.lower() == topic.lower()]
        
        return {
            "topic": topic,
            "mastery": mastery.model_dump(),
            "total_sessions": len(topic_sessions),
            "completed_sessions": len([s for s in topic_sessions if s.status == "completed"]),
            "recent_sessions": [s.model_dump() for s in topic_sessions[:5]]
        }

    async def set_user_level(self, level: UserLevel) -> bool:
        """Set user's learning level."""
        return await self.orchestrator.set_user_level(level)

    async def get_user_level(self) -> UserLevel:
        """Get user's current learning level."""
        return await self.orchestrator.get_user_level()

    def _format_lesson_intro(self, topic: str, level: UserLevel) -> str:
        """Format lesson introduction message."""
        level_text = {
            UserLevel.PEMULA: "pemula",
            UserLevel.MENENGAH: "menengah",
            UserLevel.MAHIR: "mahir"
        }
        
        return (
            f"📚 **Belajar: {topic}**\n\n"
            f"Level: {level_text.get(level, 'pemula')}\n\n"
            f"Halo! Saya Ara, guru kamu hari ini. "
            f"Mari kita pelajari **{topic}** bersama-sama!\n\n"
            f"📖 Materi sudah disiapkan. Klik 'Lanjut' untuk mulai!"
        )

    def _format_lesson_summary(self, topic: str) -> str:
        """Format lesson completion message."""
        return (
            f"✅ **Belajar {topic} Selesai!**\n\n"
            f"Bagus sekali! Kamu telah menyelesaikan pembelajaran tentang **{topic}**.\n\n"
            f"📋 Ringkasan:\n"
            f"- Materi sudah dipelajari\n"
            f"- Nanti akan ada kuis untuk menguji pemahamanmu\n"
            f"- Topik akan masuk jadwal review (spaced repetition)\n\n"
            f"Ada topik lain yang ingin kamu pelajari?"
        )

    async def _complete_session(self, session_id: int):
        """Mark session as completed."""
        db = await get_db()
        try:
            await db.execute(
                "UPDATE learning_sessions SET status = 'completed', completed_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,)
            )
            await db.commit()
        finally:
            await db.close()


learning_service = LearningService()