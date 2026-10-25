"""FastAPI application: JSON API plus the browser UI."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import documents, health, query
from app.config import Settings
from app.errors import AppError
from app.redact import install_redaction, redact
from app.services.catalog import Catalog
from app.services.embeddings import Embedder, SentenceTransformerEmbedder
from app.services.llm import GroqLLM, LLMClient
from app.services.rag import RagService
from app.services.vectorstore import VectorStore

logger = logging.getLogger("rag_docs")
STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
) -> FastAPI:
    """Build the app. Tests inject a fake embedder and a fake Groq client."""
    settings = settings or Settings()
    install_redaction()
    logger.setLevel(settings.log_level.upper())

    resolved_embedder = embedder or SentenceTransformerEmbedder(settings.embedding_model)
    resolved_llm = llm or GroqLLM(
        model=settings.groq_model,
        base_url=settings.groq_base_url,
        timeout=settings.groq_timeout_seconds,
        temperature=settings.groq_temperature,
        max_completion_tokens=settings.groq_max_completion_tokens,
        reasoning_effort=settings.groq_reasoning_effort,
        include_reasoning=settings.groq_include_reasoning,
    )
    catalog = Catalog(settings.sqlite_path)
    store = VectorStore(settings.chroma_persist_dir, resolved_embedder, settings.embedding_model)
    rag = RagService(settings, catalog, store, resolved_llm)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.catalog.initialize()
        Path(app.state.settings.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(app.state.settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        yield
        app.state.store.close()

    app = FastAPI(
        title="rag-docs-assistant",
        version=__version__,
        summary="Ask questions about your documents and get cited answers.",
        description=(
            "Local retrieval-augmented generation. Embeddings stay on this machine. "
            "Generation uses the Groq free API. API keys sent by the browser are not stored."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.catalog = catalog
    app.state.store = store
    app.state.rag = rag
    app.state.llm = resolved_llm

    app.include_router(health.router, prefix="/api")
    app.include_router(documents.router, prefix="/api")
    app.include_router(query.router, prefix="/api")

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        logger.info("%s %s -> %s", request.method, request.url.path, response.status_code)
        return response

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.info("Request failed code=%s status=%s", exc.code, exc.status_code)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.code, "message": redact(exc.message)},
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Drop the submitted body. A failed question must not echo an API key.
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "message": "Request validation failed.",
                "details": _sanitize(exc.errors()),
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": "Something went wrong while handling that request.",
            },
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items() if key != "input"}
    return value


app = create_app()
