"""
Learning Orchestrator - Main engine for Learning Mode.

Handles:
1. Material selection from RAG
2. Lesson generation
3. Quiz generation
4. Progress tracking
5. Spaced repetition scheduling
"""
import json
import time
import logging
from typing import Optional
from datetime import datetime, timedelta

from backend.models.database import get_db
from backend.models.schemas import (
    UserLevel, LessonSection, QuizQuestion,
    LearningSessionResponse, TopicMastery, ReviewItem
)
from backend.core.llm_client import generate_text
from backend.services import rag_service


logger = logging.getLogger("backend.core.learning_orchestrator")

SYSTEM_PROMPT_LESSON = """Anda adalah Guru Pembelajaran named "Ara" yang especializadas dalam mengajarkan materi kepada mahasiswa.

Karakteristik Anda:
- Penyampaian materi jelas dan terstruktur
- Menggunakan bahasa Indonesia yang mudah dipahami
- Memberikan contoh kontekstual
- Mengatur pacing sesuai level pengguna
- Ramah dan suportif

Ketika mengajarkan, Anda akan:
1. Memulai dengan konteks/pengantar singkat
2. Menjelaskan konsep inti
3. Memberikan contoh nyata
4. Menutup dengan ringkasan

Format respons dengan heading markdown (# ## ###)."""

SYSTEM_PROMPT_QUIZ = """Anda adalah Guru Pembuatan Kuis yang membuat pertanyaan untuk menguji pemahaman siswa.

Karakteristik Anda:
- Membuat pertanyaan yang relevan dengan materi
- Tingkat kesulitan sesuai level (pemula/menengah/mahir)
- Opsi jawaban yang masuk akal (untuk multiple choice)
- Penjelasan yang membantu pembelajaran

Format output必须是 JSON dengan struktur:
{
    "questions": [
        {
            "type": "multiple_choice|true_false|short_answer",
            "question": "pertanyaan di sini",
            "options": ["A", "B", "C", "D"] (untuk multiple choice),
            "correct_answer": "jawaban benar",
            "explanation": "penjelasan jawaban"
        }
    ]
}

Buat 3-5 pertanyaan untuk setiap quiz."""


class LearningOrchestrator:
    async def get_available_topics(self) -> list[str]:
        """Get list of topics from uploaded documents."""
        try:
            materials = rag_service.list_materials()
            topics = list(set([m.get("topic", m.get("filename", "Unknown")) for m in materials]))
            return topics if topics else []
        except Exception as e:
            logger.error(f"Error getting topics: {e}")
            return []

    async def get_user_level(self, user_id: int = 1) -> UserLevel:
        """Get user's current learning level."""
        db = await get_db()
        try:
            cursor = await db.execute(
                "SELECT current_level FROM user_profiles WHERE id = ?",
                (user_id,)
            )
            row = await cursor.fetchone()
            if row:
                return UserLevel(row[0])
            await db.execute(
                "INSERT INTO user_profiles (id) VALUES (?)",
                (user_id,)
            )
            await db.commit()
            return UserLevel.PEMULA
        finally:
            await db.close()

    async def set_user_level(self, level: UserLevel, user_id: int = 1) -> bool:
        """Set user's learning level."""
        db = await get_db()
        try:
            await db.execute(
                "UPDATE user_profiles SET current_level = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (level.value, user_id)
            )
            await db.commit()
            return True
        finally:
            await db.close()

    async def get_topic_mastery(self, topic: str) -> TopicMastery:
        """Get user's mastery level for a topic."""
        db = await get_db()
        try:
            cursor = await db.execute(
                """SELECT topic, mastery_level, total_attempts, correct_attempts, 
                   last_reviewed, ease_factor, interval_days, next_review
                   FROM user_topic_mastery WHERE topic = ?""",
                (topic,)
            )
            row = await cursor.fetchone()
            if row:
                return TopicMastery(
                    topic=row[0],
                    mastery_level=row[1],
                    total_attempts=row[2],
                    last_reviewed=row[4],
                    next_review=row[7]
                )
            return TopicMastery(topic=topic, mastery_level=0.0, total_attempts=0, last_reviewed=None)
        finally:
            await db.close()

    async def start_lesson(
        self,
        topic: str,
        subtopic: str = "",
        level: UserLevel = UserLevel.PEMULA
    ) -> LearningSessionResponse:
        """Start a new learning session."""
        db = await get_db()
        try:
            cursor = await db.execute(
                """INSERT INTO learning_sessions 
                   (mode, topic, subtopic, user_level, status) 
                   VALUES (?, ?, ?, ?, ?)""",
                ("lesson", topic, subtopic, level.value, "active")
            )
            await db.commit()
            session_id = cursor.lastrowid

            cursor = await db.execute(
                "SELECT id, mode, topic, subtopic, user_level, status, current_section, "
                "total_sections, created_at, completed_at FROM learning_sessions WHERE id = ?",
                (session_id,)
            )
            row = await cursor.fetchone()
            return LearningSessionResponse(
                id=row[0], mode=row[1], topic=row[2], subtopic=row[3],
                level=row[4], status=row[5], current_section=row[6],
                total_sections=row[7], created_at=row[8], completed_at=row[9]
            )
        finally:
            await db.close()

    async def generate_lesson_content(
        self,
        session_id: int,
        topic: str,
        level: UserLevel = UserLevel.PEMULA
    ) -> list[LessonSection]:
        """Generate lesson sections from materials."""
        context = await self._get_material_context(topic, limit=5)
        
        if not context:
            return [LessonSection(
                type="introduction",
                content=f"Maaf, saya tidak menemukan materi tentang '{topic}' di dokumen yang tersedia.\n\n"
                        f"Silakan upload dokumen yang berisi materi '{topic}' terlebih dahulu.",
                duration_mins=1
            )]

        prompt = self._build_lesson_prompt(topic, level.value, context)
        response = generate_text(prompt, system_instruction=SYSTEM_PROMPT_LESSON)

        sections = self._parse_lesson_response(response)
        if not sections:
            sections = self._fallback_lesson(topic, context)

        await self._save_lesson_progress(session_id, sections)
        return sections

    async def _get_material_context(self, topic: str, limit: int = 5) -> str:
        """Get relevant material context from RAG."""
        try:
            results = rag_service.retrieve_context(topic, top_k=limit)
            return results
        except Exception as e:
            logger.error(f"Error getting material context: {e}")
            return ""

    def _build_lesson_prompt(self, topic: str, level: str, context: str) -> str:
        level_instruction = {
            "pemula": "Gunakan bahasa sederhana dan berikan penjelasan dasar yang komprehensif.",
            "menengah": "Asumsikan pengguna sudah punya pemahaman dasar. Fokus pada detail dan aplikasi.",
            "mahir": "Gunakan terminologi yang lebih advanced dan fokus pada nuance dan edge cases."
        }
        
        return f"""Buatkan pembelajaran tentang topik: {topic}

Level pengguna: {level}
{level_instruction.get(level, level_instruction['pemula'])}

Materi referensi:
{context}

Buatkan pembelajaran dengan struktur:
1. Pengenalan singkat (1-2 kalimat apa yang akan dipelajari)
2. Penjelasan konsep utama (gunakan heading ## untuk sub-bagian)
3. Contoh nyata/penerapan
4. Ringkasan singkat

Panjang pembelajaran: {10 if level == 'pemula' else 7 if level == 'menengah' else 5} menit bacaan."""

    def _parse_lesson_response(self, response: str) -> list[LessonSection]:
        """Parse LLM response into lesson sections."""
        sections = []
        
        parts = response.split("\n## ")
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            
            if i == 0:
                part = part.lstrip("# ").strip()
                section_type = "introduction"
            elif "contoh" in part.lower()[:50]:
                section_type = "example"
            elif "ringkasan" in part.lower()[:50] or "kesimpulan" in part.lower()[:50]:
                section_type = "summary"
            else:
                section_type = "explanation"
            
            lines = part.split("\n")
            heading = lines[0] if lines else ""
            content = "\n".join(lines[1:]) if len(lines) > 1 else ""
            
            sections.append(LessonSection(
                type=section_type,
                content=f"{heading}\n{content}".strip(),
                duration_mins=3
            ))
        
        return sections

    def _fallback_lesson(self, topic: str, context: str) -> list[LessonSection]:
        """Fallback lesson generation without LLM."""
        return [
            LessonSection(
                type="introduction",
                content=f"# Belajar: {topic}\n\n"
                        f"Selamat datang! Hari ini kita akan mempelajari tentang **{topic}**.\n\n"
                        "Di akhir pembelajaran, kamu akan memahami konsep dasar dan bisa menerapkannya.",
                duration_mins=2
            ),
            LessonSection(
                type="explanation",
                content=f"## Materi\n\n{context[:1000]}...",
                duration_mins=5
            ),
            LessonSection(
                type="summary",
                content=f"## Ringkasan\n\n"
                        f"Baik, itu dia pengenalan tentang **{topic}**. "
                        f"Apakah ada bagian yang ingin kamu tanyakan lebih lanjut?",
                duration_mins=1
            )
        ]

    async def _save_lesson_progress(
        self,
        session_id: int,
        sections: list[LessonSection]
    ):
        """Save lesson progress to database."""
        db = await get_db()
        try:
            total = len(sections)
            for i, section in enumerate(sections):
                await db.execute(
                    """INSERT INTO lesson_progress 
                       (session_id, section_type, content, completed) 
                       VALUES (?, ?, ?, ?)""",
                    (session_id, section.type, section.content, 1 if i == 0 else 0)
                )
            await db.execute(
                "UPDATE learning_sessions SET total_sections = ? WHERE id = ?",
                (total, session_id)
            )
            await db.commit()
        finally:
            await db.close()

    async def generate_quiz(
        self,
        topic: str,
        level: UserLevel = UserLevel.PEMULA,
        num_questions: int = 5
    ) -> list[QuizQuestion]:
        """Generate quiz questions from materials."""
        context = await self._get_material_context(topic, limit=5)
        
        if not context:
            return [QuizQuestion(
                id=1,
                type="short_answer",
                question=f"Apa yang kamu ketahui tentang '{topic}'?",
                options=[],
                correct_answer="Open-ended question",
                explanation="Tidak ada materi spesifik. Ini pertanyaan terbuka untuk eksplorasi."
            )]

        prompt = self._build_quiz_prompt(topic, level.value, context, num_questions)
        
        try:
            response = generate_text(prompt, system_instruction=SYSTEM_PROMPT_QUIZ)
            questions = self._parse_quiz_response(response)
            if questions:
                return questions
        except Exception as e:
            logger.error(f"Error generating quiz: {e}")
        
        return self._fallback_quiz(topic)

    def _build_quiz_prompt(
        self,
        topic: str,
        level: str,
        context: str,
        num_questions: int
    ) -> str:
        level_instruction = {
            "pemula": "Buatkan pertanyaan dasar yang menguji pemahaman konsep.",
            "menengah": "Pertanyaan yang memerlukan pemahaman lebih mendalam.",
            "mahir": "Pertanyaan yang menguji analisis dan aplikasi."
        }
        
        return f"""Buatkan {num_questions} pertanyaan kuis tentang: {topic}

Level: {level}
{level_instruction.get(level, level_instruction['pemula'])}

Materi:
{context}

{num_questions} pertanyaan dalam format JSON dengan struktur:
{{
    "questions": [
        {{
            "type": "multiple_choice|true_false|short_answer",
            "question": "pertanyaan",
            "options": ["A", "B", "C", "D"] (untuk multiple choice),
            "correct_answer": "jawaban benar",
            "explanation": "penjelasan"
        }}
    ]
}}"""

    def _parse_quiz_response(self, response: str) -> list[QuizQuestion]:
        """Parse quiz questions from LLM response."""
        try:
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            
            data = json.loads(json_str.strip())
            questions = []
            
            for i, q in enumerate(data.get("questions", []), 1):
                questions.append(QuizQuestion(
                    id=i,
                    type=q.get("type", "multiple_choice"),
                    question=q.get("question", ""),
                    options=q.get("options", []),
                    correct_answer=q.get("correct_answer"),
                    explanation=q.get("explanation")
                ))
            
            return questions
        except Exception as e:
            logger.error(f"Error parsing quiz response: {e}")
            return []

    def _fallback_quiz(self, topic: str) -> list[QuizQuestion]:
        """Fallback quiz when LLM fails."""
        return [
            QuizQuestion(
                id=1,
                type="short_answer",
                question=f"Jelaskan apa yang kamu ketahui tentang {topic}!",
                options=[],
                correct_answer="Open-ended",
                explanation="Pertanyaan terbuka untuk mengeksplorasi pemahaman."
            )
        ]

    async def evaluate_quiz(
        self,
        session_id: int,
        answers: dict[int, str],
        questions: list[QuizQuestion]
    ) -> dict:
        """Evaluate quiz answers and update mastery."""
        results = []
        correct_count = 0
        
        for q in questions:
            user_answer = answers.get(q.id, "")
            is_correct = user_answer.strip().lower() == q.correct_answer.strip().lower()
            if is_correct:
                correct_count += 1
            
            results.append({
                "question_id": q.id,
                "question": q.question,
                "user_answer": user_answer,
                "correct_answer": q.correct_answer,
                "is_correct": is_correct,
                "explanation": q.explanation
            })
        
        score = (correct_count / len(questions) * 100) if questions else 0
        
        await self._save_quiz_results(session_id, results, score)
        await self._update_topic_mastery(session_id, score)
        
        return {
            "session_id": session_id,
            "total_questions": len(questions),
            "correct_count": correct_count,
            "score_percentage": score,
            "results": results,
            "recommendations": self._get_recommendations(score)
        }

    async def _save_quiz_results(
        self,
        session_id: int,
        results: list[dict],
        score: float
    ):
        """Save quiz results to database."""
        db = await get_db()
        try:
            for r in results:
                await db.execute(
                    """INSERT INTO quiz_results 
                       (session_id, question, question_type, options, correct_answer, user_answer, is_correct, explanation)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session_id, r["question"], "quiz",
                        json.dumps(r.get("options", [])),
                        r["correct_answer"], r["user_answer"],
                        1 if r["is_correct"] else 0, r.get("explanation", "")
                    )
                )
            
            await db.execute(
                "UPDATE learning_sessions SET status = 'completed', completed_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,)
            )
            await db.commit()
        finally:
            await db.close()

    async def _update_topic_mastery(self, session_id: int, score: float):
        """Update topic mastery after quiz."""
        db = await get_db()
        try:
            cursor = await db.execute(
                "SELECT topic FROM learning_sessions WHERE id = ?",
                (session_id,)
            )
            row = await cursor.fetchone()
            if not row:
                return
            
            topic = row[0]
            
            cursor = await db.execute(
                "SELECT mastery_level, ease_factor, interval_days FROM user_topic_mastery WHERE topic = ?",
                (topic,)
            )
            existing = await cursor.fetchone()
            
            if existing:
                mastery, ease_factor, interval = existing
                if score >= 80:
                    new_interval = int(interval * ease_factor)
                    new_ease = min(ease_factor * 1.1, 3.0)
                elif score >= 50:
                    new_interval = interval
                    new_ease = ease_factor
                else:
                    new_interval = 1
                    new_ease = max(ease_factor - 0.1, 1.3)
                
                new_mastery = min(mastery + (score * 0.1), 100.0)
                
                await db.execute(
                    """UPDATE user_topic_mastery 
                       SET mastery_level = ?, total_attempts = total_attempts + 1,
                           correct_attempts = correct_attempts + ?, last_reviewed = CURRENT_TIMESTAMP,
                           ease_factor = ?, interval_days = ?, next_review = datetime('now', ? || ' days')
                       WHERE topic = ?""",
                    (new_mastery, 1 if score >= 50 else 0, new_ease, new_interval, new_interval, topic)
                )
            else:
                initial_mastery = score * 0.5
                await db.execute(
                    """INSERT INTO user_topic_mastery 
                       (topic, mastery_level, total_attempts, correct_attempts, last_reviewed, 
                        ease_factor, interval_days, next_review)
                       VALUES (?, ?, 1, ?, CURRENT_TIMESTAMP, 2.5, 1, datetime('now', '1 day'))""",
                    (topic, initial_mastery, 1 if score >= 50 else 0)
                )
            
            await db.commit()
        finally:
            await db.close()

    def _get_recommendations(self, score: float) -> list[str]:
        """Get learning recommendations based on score."""
        if score >= 90:
            return [
                "🎉 Luar biasa! Kamu sangat paham materi ini!",
                "💡 Mau coba topik yang lebih challenging?",
                "📚 Reposisi untuk review mingguan saja."
            ]
        elif score >= 70:
            return [
                "👍 Bagus! Pemahamanmu sudah baik.",
                "🔄 Review singkat materi untuk memperkuat.",
                "📝 Coba latihan soal tambahan."
            ]
        elif score >= 50:
            return [
                "📖 Materi perlu di-review lagi.",
                "💪 Konsisten belajar, pasti bisa!",
                "⏰ Sesi review akan dijadwalkan otomatis."
            ]
        else:
            return [
                "🤗 Tidak apa-apa! Ini bagian dari proses belajar.",
                "📚 Saya akan menjadwalkan review rutin.",
                "💡 Mari kita pelajari lagi dari awal."
            ]

    async def get_review_queue(self, limit: int = 5) -> list[ReviewItem]:
        """Get topics due for review (spaced repetition)."""
        db = await get_db()
        try:
            cursor = await db.execute(
                """SELECT topic, mastery_level, interval_days, next_review
                   FROM user_topic_mastery
                   WHERE next_review IS NOT NULL 
                   AND datetime(next_review) <= datetime('now')
                   ORDER BY mastery_level ASC
                   LIMIT ?""",
                (limit,)
            )
            rows = await cursor.fetchall()
            return [
                ReviewItem(
                    topic=row[0],
                    mastery_level=row[1],
                    due_review=row[3],
                    interval_days=row[2]
                ) for row in rows
            ]
        finally:
            await db.close()

    async def get_session_history(self, limit: int = 10) -> list[LearningSessionResponse]:
        """Get recent learning sessions."""
        db = await get_db()
        try:
            cursor = await db.execute(
                """SELECT id, mode, topic, subtopic, user_level, status, 
                   current_section, total_sections, created_at, completed_at
                   FROM learning_sessions 
                   ORDER BY created_at DESC LIMIT ?""",
                (limit,)
            )
            rows = await cursor.fetchall()
            return [
                LearningSessionResponse(
                    id=row[0], mode=row[1], topic=row[2], subtopic=row[3],
                    level=row[4], status=row[5], current_section=row[6],
                    total_sections=row[7], created_at=row[8], completed_at=row[9]
                ) for row in rows
            ]
        finally:
            await db.close()


learning_orchestrator = LearningOrchestrator()