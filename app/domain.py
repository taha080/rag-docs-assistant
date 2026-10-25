"""Records passed between the catalog, the vector store, and the API."""

from __future__ import annotations

from dataclasses import dataclass

from app.citations import citation_label


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    filename: str
    content_type: str
    size_bytes: int
    chunk_count: int
    created_at: str


@dataclass(frozen=True)
class RetrievedChunk:
    document_id: str
    filename: str
    page: int | None
    chunk_index: int
    text: str
    score: float

    @property
    def label(self) -> str:
        return citation_label(self.filename, self.page, self.chunk_index)


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[RetrievedChunk]
    model: str
