"""
Analytics Service — Aggregates metrics from chat history and dataset.
"""
from backend.models.database import get_db
from backend.models.schemas import AnalyticsResponse
from backend.core.data_loader import SyntheticDataset


async def get_analytics() -> AnalyticsResponse:
    """Aggregate system analytics."""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT COUNT(*) FROM chat_history")
        total_chats = (await cursor.fetchone())[0]

        cursor = await db.execute(
            "SELECT emotion, COUNT(*) FROM chat_history WHERE role='user' GROUP BY emotion"
        )
        emotion_dist = {row[0]: row[1] for row in await cursor.fetchall()}

        cursor = await db.execute(
            "SELECT intent, COUNT(*) FROM chat_history WHERE role='user' GROUP BY intent"
        )
        intent_dist = {row[0]: row[1] for row in await cursor.fetchall()}

        cursor = await db.execute(
            "SELECT COUNT(*) FROM chat_history WHERE layer3_status='VIOLATION' AND role='user'"
        )
        l3_violations = (await cursor.fetchone())[0]

        cursor = await db.execute(
            "SELECT COUNT(*) FROM chat_history WHERE layer4_rewritten=1 AND role='user'"
        )
        l4_rewrites = (await cursor.fetchone())[0]
    finally:
        await db.close()

    try:
        ds = SyntheticDataset()
        dataset_stats = ds.get_statistics()
    except Exception:
        dataset_stats = {}

    return AnalyticsResponse(
        total_chats=total_chats,
        emotion_distribution=emotion_dist,
        intent_distribution=intent_dist,
        layer3_violations=l3_violations,
        layer4_rewrites=l4_rewrites,
        dataset_stats=dataset_stats,
    )
