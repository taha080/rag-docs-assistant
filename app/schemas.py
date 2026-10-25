"""HTTP request and response models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.domain import Answer, DocumentRecord, RetrievedChunk


class DocumentOut(BaseModel):
    id: str
    filename: str
    content_type: str
    size_bytes: int
    chunk_count: int
    created_at: str

    @classmethod
    def from_record(cls, record: DocumentRecord) -> DocumentOut:
        return cls(
            id=record.id,
            filename=record.filename,
            content_type=record.content_type,
            size_bytes=record.size_bytes,
            chunk_count=record.chunk_count,
            created_at=record.created_at,
        )


class SourceOut(BaseModel):
    document_id: str
    filename: str
    page: int | None
    chunk_index: int
    text: str
    score: float
    label: str

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk) -> SourceOut:
        return cls(
            document_id=chunk.document_id,
            filename=chunk.filename,
            page=chunk.page,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            score=chunk.score,
            label=chunk.label,
        )


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str = Field(min_length=1, max_length=4000)
    api_key: str | None = Field(
        default=None,
        max_length=300,
        description="Optional Groq API key. Prefer the X-Groq-Api-Key header. Not stored.",
    )
    top_k: int | None = Field(default=None, ge=1, le=10)


class QueryResponse(BaseModel):
    answer: str
    model: str
    sources: list[SourceOut]

    @classmethod
    def from_answer(cls, answer: Answer) -> QueryResponse:
        return cls(
            answer=answer.text,
            model=answer.model,
            sources=[SourceOut.from_chunk(chunk) for chunk in answer.sources],
        )


class HealthResponse(BaseModel):
    status: str
    version: str
    embedding_model: str
    llm_model: str
    groq_configured: bool
    document_count: int
