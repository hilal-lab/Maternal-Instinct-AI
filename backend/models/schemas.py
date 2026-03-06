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
