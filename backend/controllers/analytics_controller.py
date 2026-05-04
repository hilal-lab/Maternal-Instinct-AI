"""
Analytics Controller — Handles analytics retrieval.
"""
from backend.models.schemas import AnalyticsResponse
from backend.services import analytics_service


async def get_analytics() -> AnalyticsResponse:
    return await analytics_service.get_analytics()
