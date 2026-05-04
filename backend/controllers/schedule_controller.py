"""
Schedule Controller — Handles schedule request/response formatting.
"""
from fastapi import HTTPException

from backend.models.schemas import ScheduleCreate, ScheduleUpdate, ScheduleResponse
from backend.services import schedule_service


async def list_all() -> list[ScheduleResponse]:
    return await schedule_service.list_all()


async def create(item: ScheduleCreate) -> ScheduleResponse:
    return await schedule_service.create(item)


async def update(schedule_id: int, item: ScheduleUpdate) -> ScheduleResponse:
    result = await schedule_service.update(schedule_id, item)
    if result is None:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return result


async def delete(schedule_id: int) -> dict:
    deleted = await schedule_service.delete(schedule_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return {"message": f"Schedule {schedule_id} deleted."}


async def get_workload() -> dict:
    return await schedule_service.get_workload()
