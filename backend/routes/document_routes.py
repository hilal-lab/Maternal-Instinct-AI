"""
Document Routes — API endpoint definitions.
"""
from fastapi import APIRouter, UploadFile, File

from backend.models.schemas import DocumentResponse
from backend.controllers import document_controller

router = APIRouter()


@router.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a learning material."""
    return await document_controller.upload(file)


@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents():
    """List all uploaded documents."""
    return await document_controller.list_all()


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: int):
    """Delete a document."""
    return await document_controller.delete(doc_id)
