"""
Pydantic schemas (request/response models) for all API entities.
Equivalent to DTOs in traditional backend architectures.
"""
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ─── Enums ───────────────────────────────────────────────

class Priority(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"

class EmotionType(str, Enum):
    NEUTRAL = "NEUTRAL"
    STRESS = "STRESS"
    OVERWHELMED = "OVERWHELMED"
    PANIC = "PANIC"
    SELF_CRITICAL = "SELF-CRITICAL"
    FATIGUE = "FATIGUE"

class IntentType(str, Enum):
    TASK_OPS = "TASK_OPS"
    ACADEMIC_HELP = "ACADEMIC_HELP"
    MOTIVATION_SUPPORT = "MOTIVATION_SUPPORT"
    EMOTIONAL_DISTRESS = "EMOTIONAL_DISTRESS"
    GENERAL_CHAT = "GENERAL_CHAT"


class ChatMode(str, Enum):
    CONVERSATION = "conversation"
    LEARNING = "learning"


class LearningMode(str, Enum):
    LESSON = "lesson"
    QUIZ = "quiz"
    REVIEW = "review"


class UserLevel(str, Enum):
    PEMULA = "pemula"
    MENENGAH = "menengah"
    MAHIR = "mahir"


# ─── Schedule Schemas ────────────────────────────────────

class ScheduleCreate(BaseModel):
    task: str
    deadline: str
    est_hours: float = Field(default=2, ge=0.5, le=24)
    priority: Priority = Priority.MEDIUM

class ScheduleUpdate(BaseModel):
    task: Optional[str] = None
    deadline: Optional[str] = None
    est_hours: Optional[float] = None
    priority: Optional[Priority] = None
    status: Optional[TaskStatus] = None

class ScheduleResponse(BaseModel):
    id: int
    task: str
    deadline: str
    est_hours: float
    priority: str
    status: str
    created_at: str


# ─── Note Schemas ────────────────────────────────────────

class NoteCreate(BaseModel):
    title: str
    content: str
    subject: str = ""

class NoteUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    subject: Optional[str] = None

class NoteResponse(BaseModel):
    id: int
    title: str
    content: str
    subject: str
    created_at: str
    updated_at: str


# ─── Document Schemas ────────────────────────────────────

class DocumentResponse(BaseModel):
    id: int
    filename: str
    content_type: str
    chunk_count: int
    file_size: int
    created_at: str


# ─── Chat Schemas ────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None

class LayerOneData(BaseModel):
    intent: str = "GENERAL_CHAT"
    emotion: str = "NEUTRAL"
    intensity: float = 0.0
    flags: dict = {}

class LayerTwoData(BaseModel):
    agent_used: str = "general_chat"
    raw_response: str = ""

class LayerThreeData(BaseModel):
    status: str = "PASS"
    note: Optional[str] = None
    policy_recommendations: dict = {}

class LayerFourData(BaseModel):
    is_rewritten: bool = False
    original: str = ""
    final: str = ""

class LayersData(BaseModel):
    layer1: LayerOneData = LayerOneData()
    layer2: LayerTwoData = LayerTwoData()
    layer3: LayerThreeData = LayerThreeData()
    layer4: LayerFourData = LayerFourData()

class ChatResponse(BaseModel):
    response: str
    layers: LayersData = LayersData()


# ─── Analytics Schemas ───────────────────────────────────

class AnalyticsResponse(BaseModel):
    total_chats: int = 0
    avg_intensity: float = 0.0
    emotion_distribution: dict = {}
    intent_distribution: dict = {}
    layer3_violations: int = 0
    layer4_rewrites: int = 0
    dataset_stats: dict = {}


# ─── Learning Schemas ────────────────────────────────────

class LessonSection(BaseModel):
    type: str
    content: str
    duration_mins: int = 5
    quiz_questions: list = []


class LessonResponse(BaseModel):
    session_id: int
    topic: str
    subtopic: str
    level: str
    sections: list[LessonSection]
    total_duration_mins: int


class QuizQuestion(BaseModel):
    id: int
    type: str
    question: str
    options: list[str] = []
    correct_answer: Optional[str] = None
    explanation: Optional[str] = None


class QuizSubmission(BaseModel):
    session_id: int
    answers: dict[int, str]


class QuizResult(BaseModel):
    session_id: int
    total_questions: int
    correct_count: int
    score_percentage: float
    results: list[dict]
    recommendations: list[str]


class TopicMastery(BaseModel):
    topic: str
    mastery_level: float
    total_attempts: int
    last_reviewed: Optional[str]
    next_review: Optional[str]


class UserProfile(BaseModel):
    current_level: str = "pemula"
    learning_goals: list[str] = []
    preferred_topics: list[str] = []


class LearningSessionResponse(BaseModel):
    id: int
    mode: str
    topic: str
    subtopic: str
    level: str
    status: str
    current_section: int
    total_sections: int
    created_at: str
    completed_at: Optional[str]


class ReviewItem(BaseModel):
    topic: str
    mastery_level: float
    due_review: str
    interval_days: int
