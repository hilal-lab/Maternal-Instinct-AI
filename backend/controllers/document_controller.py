"""
Document Controller — Handles document upload/management.
"""
from fastapi import HTTPException, UploadFile

from backend.models.schemas import DocumentResponse
from backend.services import document_service


async def upload(file: UploadFile) -> DocumentResponse:
    content = await file.read()
    try:
        return await document_service.upload(file.filename, content, file.content_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


async def list_all() -> list[DocumentResponse]:
    return await document_service.list_all()


async def delete(doc_id: int) -> dict:
    deleted = await document_service.delete(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": f"Document {doc_id} deleted."}
