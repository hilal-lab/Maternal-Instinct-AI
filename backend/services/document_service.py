"""
Document Service — File upload management and (future) RAG ingestion.
"""
import os
from pathlib import Path

from backend.models.database import get_db
from backend.models.schemas import DocumentResponse
from backend.services import rag_service

UPLOAD_DIR = Path(__file__).parent.parent / "data" / "uploads"
ALLOWED_EXTENSIONS = {".md", ".txt", ".pdf", ".csv"}


async def upload(filename: str, content: bytes, content_type: str) -> DocumentResponse:
    """Save file to disk and register in DB."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"File type {ext} not supported. Use: {ALLOWED_EXTENSIONS}")

    file_path = UPLOAD_DIR / filename
    with open(file_path, "wb") as f:
        f.write(content)

    # Phase 2: Run through RAG ingestion pipeline
    chunk_count = await rag_service.ingest_uploaded_file(str(file_path), doc_id=filename)

    db = await get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO documents (filename, content_type, chunk_count, file_size) "
            "VALUES (?, ?, ?, ?) RETURNING *",
            (filename, content_type or "text/plain", chunk_count, len(content))
        )
        row = await cursor.fetchone()
        await db.commit()
        return DocumentResponse(
            id=row[0], filename=row[1], content_type=row[2],
            chunk_count=row[3], file_size=row[4], created_at=str(row[5])
        )
    finally:
        await db.close()


async def list_all() -> list[DocumentResponse]:
    """List all uploaded documents."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, filename, content_type, chunk_count, file_size, created_at "
            "FROM documents ORDER BY created_at DESC"
        )
        rows = await cursor.fetchall()
        return [
            DocumentResponse(
                id=r[0], filename=r[1], content_type=r[2],
                chunk_count=r[3], file_size=r[4], created_at=str(r[5])
            )
            for r in rows
        ]
    finally:
        await db.close()


async def delete(doc_id: int) -> bool:
    """Delete document from DB and disk."""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT filename FROM documents WHERE id = ?", (doc_id,))
        row = await cursor.fetchone()
        if not row:
            return False

        file_path = UPLOAD_DIR / row[0]
        if file_path.exists():
            os.remove(file_path)

        # Phase 2: Remove from FAISS index -> doc_id used in ingestion is the filename
        await rag_service.remove_document(doc_id=row[0])

        await db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        await db.commit()
        return True
    finally:
        await db.close()
