"""Upload, list, and delete documents."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import Response

from app.errors import AppError
from app.schemas import DocumentOut

router = APIRouter(tags=["documents"])


@router.post("/documents", status_code=201, response_model=DocumentOut, summary="Upload a document")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
) -> DocumentOut:
    settings = request.app.state.settings
    try:
        data = await file.read(settings.max_upload_bytes + 1)
    finally:
        await file.close()
    if len(data) > settings.max_upload_bytes:
        raise AppError(
            f"That file is larger than {settings.max_upload_bytes} bytes.",
            413,
            "file_too_large",
        )
    record = await asyncio.to_thread(
        request.app.state.rag.ingest,
        file.filename or "",
        data,
    )
    return DocumentOut.from_record(record)


@router.get("/documents", response_model=list[DocumentOut], summary="List documents")
def list_documents(request: Request) -> list[DocumentOut]:
    return [DocumentOut.from_record(record) for record in request.app.state.rag.list_documents()]


@router.delete("/documents/{document_id}", status_code=204, summary="Delete a document")
def delete_document(document_id: str, request: Request) -> Response:
    request.app.state.rag.delete(document_id)
    return Response(status_code=204)
