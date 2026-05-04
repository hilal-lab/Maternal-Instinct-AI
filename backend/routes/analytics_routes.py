"""
Analytics Routes — API endpoint definitions.
"""
from fastapi import APIRouter

from backend.models.schemas import AnalyticsResponse
from backend.controllers import analytics_controller

router = APIRouter()


@router.get("/analytics", response_model=AnalyticsResponse)
async def analytics():
    """Get system analytics and dataset statistics."""
    return await analytics_controller.get_analytics()
