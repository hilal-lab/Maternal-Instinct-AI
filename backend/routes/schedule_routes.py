"""
Schedule Routes — API endpoint definitions.
"""
from fastapi import APIRouter

from backend.models.schemas import ScheduleCreate, ScheduleUpdate, ScheduleResponse
from backend.controllers import schedule_controller

router = APIRouter()


@router.get("/schedule", response_model=list[ScheduleResponse])
async def list_schedules():
    """List all tasks."""
    return await schedule_controller.list_all()


@router.post("/schedule", response_model=ScheduleResponse)
async def create_schedule(item: ScheduleCreate):
    """Create a new task."""
    return await schedule_controller.create(item)


@router.put("/schedule/{schedule_id}", response_model=ScheduleResponse)
async def update_schedule(schedule_id: int, item: ScheduleUpdate):
    """Update a task."""
    return await schedule_controller.update(schedule_id, item)


@router.delete("/schedule/{schedule_id}")
async def delete_schedule(schedule_id: int):
    """Delete a task."""
    return await schedule_controller.delete(schedule_id)


@router.get("/schedule/workload")
async def get_workload():
    """Get total workload hours."""
    return await schedule_controller.get_workload()
