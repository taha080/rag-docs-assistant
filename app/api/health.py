"""Liveness and configuration summary. Never returns a secret."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app import __version__
from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    return HealthResponse(
        status="ok",
        version=__version__,
        embedding_model=settings.embedding_model,
        llm_model=settings.groq_model,
        groq_configured=settings.groq_configured,
        document_count=request.app.state.catalog.count(),
    )
