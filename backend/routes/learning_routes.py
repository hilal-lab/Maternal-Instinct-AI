"""
Learning Routes - API endpoints for Learning Mode.

Provides REST endpoints for:
- Starting learning sessions
- Getting lesson content
- Submitting quiz answers
- Tracking progress
- Managing spaced repetition
"""
from fastapi import APIRouter, HTTPException
from typing import Optional

from backend.models.schemas import (
    UserLevel, LearningSessionResponse,
    LessonSection, QuizQuestion,
    QuizSubmission, QuizResult, TopicMastery,
    UserProfile, ReviewItem, LearningMode
)
from backend.services.learning_service import learning_service


router = APIRouter(prefix="/learning", tags=["Learning"])


@router.get("/topics")
async def get_available_topics():
    """Get list of available topics from uploaded materials."""
    topics = await learning_service.orchestrator.get_available_topics()
    user_topics = await learning_service.get_user_topics()
    
    return {
        "available_topics": topics,
        "user_topics": user_topics
    }


@router.get("/profile")
async def get_user_profile():
    """Get user's learning profile."""
    level = await learning_service.get_user_level()
    return UserProfile(
        current_level=level.value,
        learning_goals=[],
        preferred_topics=[]
    )


@router.put("/profile/level")
async def set_user_level(level: UserLevel):
    """Set user's learning level."""
    await learning_service.set_user_level(level)
    return {"message": f"Level updated to {level.value}", "level": level.value}


@router.post("/start")
async def start_learning(
    topic: str,
    mode: LearningMode = LearningMode.LESSON,
    level: UserLevel = UserLevel.PEMULA,
    subtopic: str = ""
):
    """Start a new learning session."""
    if not topic.strip():
        raise HTTPException(status_code=400, detail="Topic is required")
    
    result = await learning_service.start_learning(
        topic=topic.strip(),
        subtopic=subtopic.strip(),
        level=level,
        mode=mode.value
    )
    
    return result


@router.get("/session/{session_id}")
async def get_session(session_id: int):
    """Get learning session details."""
    sessions = await learning_service.orchestrator.get_session_history(limit=100)
    session = next((s for s in sessions if s.id == session_id), None)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session


@router.get("/session/{session_id}/continue")
async def continue_learning(session_id: int, user_response: str = ""):
    """Continue a learning session."""
    result = await learning_service.continue_learning(session_id, user_response)
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result


@router.post("/quiz/submit")
async def submit_quiz(submission: QuizSubmission):
    """Submit quiz answers and get results."""
    result = await learning_service.submit_quiz(
        session_id=submission.session_id,
        answers=submission.answers
    )
    
    return QuizResult(**result)


@router.get("/progress/{topic}")
async def get_topic_progress(topic: str):
    """Get detailed progress for a topic."""
    progress = await learning_service.get_topic_progress(topic)
    return progress


@router.get("/mastery/{topic}")
async def get_topic_mastery(topic: str):
    """Get user's mastery level for a topic."""
    mastery = await learning_service.orchestrator.get_topic_mastery(topic)
    return mastery


@router.get("/review-queue")
async def get_review_queue(limit: int = 5):
    """Get topics due for review (spaced repetition)."""
    reviews = await learning_service.orchestrator.get_review_queue(limit=limit)
    return {"reviews": [r.model_dump() for r in reviews]}


@router.get("/history")
async def get_learning_history(limit: int = 20):
    """Get learning session history."""
    sessions = await learning_service.orchestrator.get_session_history(limit=limit)
    return {"sessions": [s.model_dump() for s in sessions]}


@router.get("/quiz/generate")
async def generate_quiz(
    topic: str,
    level: UserLevel = UserLevel.PEMULA,
    num_questions: int = 5
):
    """Generate a quiz for a topic (standalone, not tied to session)."""
    questions = await learning_service.orchestrator.generate_quiz(
        topic=topic,
        level=level,
        num_questions=num_questions
    )
    
    return {
        "topic": topic,
        "level": level.value,
        "questions": [q.model_dump() for q in questions]
    }