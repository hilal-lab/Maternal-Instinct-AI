"""
Mode Router - Routes user input to either Conversation Mode or Learning Mode.

Detects whether user wants to learn (study materials) or have a conversation.
"""
import re
from typing import Literal
from backend.models.schemas import ChatMode


LEARNING_KEYWORDS = [
    "belajar", "ajaran", "ajari", "kuis", "quiz", "ujian",
    "materi", "pelajaran", "latihan", "study", "learn",
    "ajari aku", "belajarin", "ngajarin", "jelaskan",
    "apa itu", "bagaimana", "mengapa", "kenapa", "apa bedanya",
    "contoh", "tolong jelaskan", "bisa jelaskan",
    "mau tahu", "ingin tahu", "butuh belajar",
    "persiapan", "siap ujian", "siap teste",
]

CONVERSATION_KEYWORDS = [
    "chat", "ngobrol", "basa-basi", "tanya dong",
    "curhat", "壟", "vy", "vy",  # emoji variations
    "lagi apa", "apasih", "gimana sih",
]

EXPLICIT_LEARNING_PATTERNS = [
    r"belajar\s+(.*)",
    r"ajari\s+(aku\s+)?(.*)",
    r"kuis\s+(tentang\s+)?(.*)",
    r"quiz\s+(about\s+)?(.*)",
    r"mau\s+belajar\s+(.*)",
    r"bisa\s+jelaskan\s+(.*)",
    r"bikinin\s+kuis\s+(.*)",
    r"generate\s+quiz",
]

EXPLICIT_MODE_CHANGE = [
    r"mode\s+(conversation|belajar|learning|chat)",
    r"ganti\s+mode",
    r"switch\s+to\s+(conversation|learning)",
]


def detect_mode(message: str, force_mode: str | None = None) -> tuple[ChatMode, str]:
    """
    Detect the appropriate mode from user message.
    
    Args:
        message: User's message
        force_mode: Explicit mode override (from UI switch)
    
    Returns:
        Tuple of (mode, extracted_topic)
    """
    msg_lower = message.lower().strip()
    
    if force_mode == "learning" or force_mode == "belajar":
        topic = _extract_learning_topic(message)
        return ChatMode.LEARNING, topic
    
    if force_mode == "conversation" or force_mode == "chat":
        return ChatMode.CONVERSATION, ""
    
    if _is_explicit_mode_change(msg_lower):
        return ChatMode.CONVERSATION, ""
    
    if _is_learning_intent(msg_lower):
        topic = _extract_learning_topic(message)
        return ChatMode.LEARNING, topic
    
    return ChatMode.CONVERSATION, ""


def _is_explicit_mode_change(msg: str) -> bool:
    for pattern in EXPLICIT_MODE_CHANGE:
        if re.search(pattern, msg, re.IGNORECASE):
            return True
    return False


def _is_learning_intent(msg: str) -> bool:
    for keyword in LEARNING_KEYWORDS:
        if keyword in msg:
            return True
    
    for pattern in EXPLICIT_LEARNING_PATTERNS:
        if re.search(pattern, msg, re.IGNORECASE):
            return True
    
    return False


def _extract_learning_topic(message: str) -> str:
    msg = message.strip()
    
    patterns = [
        (r"belajar\s+(.*)", 1),
        (r"ajari\s+(?:aku\s+)?(.*)", 1),
        (r"kuis\s+(?:tentang\s+)?(.*)", 1),
        (r"quiz\s+(?:about\s+)?(.*)", 1),
        (r"mau\s+belajar\s+(.*)", 1),
        (r"bisa\s+jelaskan\s+(.*)", 1),
        (r"bikinin\s+kuis\s+(.*)", 1),
        (r"mau\s+(?:tahu|belajar)\s+(?:tentang\s+)?(.*)", 1),
    ]
    
    for pattern, group_idx in patterns:
        match = re.search(pattern, msg, re.IGNORECASE)
        if match and match.group(group_idx).strip():
            return match.group(group_idx).strip()
    
    for keyword in LEARNING_KEYWORDS:
        if keyword in msg.lower():
            remaining = msg.lower().replace(keyword, "").strip()
            if remaining:
                return remaining
    
    return msg


def get_mode_display(mode: ChatMode) -> str:
    """Get display name for mode."""
    if mode == ChatMode.LEARNING:
        return "📚 Learning Mode"
    return "💬 Conversation Mode"


def get_mode_indicator(mode: ChatMode) -> str:
    """Get emoji indicator for mode."""
    if mode == ChatMode.LEARNING:
        return "📚"
    return "💬"
