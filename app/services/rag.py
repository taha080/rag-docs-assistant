"""Upload, delete, and answer questions from the local index."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from app.config import Settings
from app.domain import Answer, DocumentRecord, RetrievedChunk
from app.errors import AppError, EmptyDocument
from app.ingest.chunking import chunk_pages
from app.ingest.files import SUPPORTED_EXTENSIONS, sanitize_filename
from app.ingest.parsers import parse_document
from app.services.catalog import Catalog
from app.services.llm import LLMClient, resolve_api_key
from app.services.vectorstore import VectorStore

logger = logging.getLogger("rag_docs")

MAX_CHUNKS = 5000

SYSTEM_PROMPT = (
    "You answer questions about documents the user uploaded. "
    "Use only the excerpts in the user message. "
    "If the excerpts do not contain the answer, say you cannot find it in the uploaded documents. "
    "Do not invent facts, citations, or page numbers. "
    "When you use an excerpt, cite its label in square brackets, for example [notes.pdf p.2 chunk 0]. "
    "Keep the answer concise."
)


class RagService:
    def __init__(self, settings: Settings, catalog: Catalog, store: VectorStore, llm: LLMClient) -> None:
        self.settings = settings
        self.catalog = catalog
        self.store = store
        self.llm = llm

    def ingest(self, filename: str, data: bytes) -> DocumentRecord:
        if not filename or not str(filename).strip():
            raise AppError("Choose a file to upload.", 400, "missing_filename")
        if not data:
            raise EmptyDocument("The file is empty.")
        if len(data) > self.settings.max_upload_bytes:
            raise AppError(
                f"That file is larger than {self.settings.max_upload_bytes} bytes.",
                413,
                "file_too_large",
            )

        safe_name = sanitize_filename(filename)
        suffix = Path(safe_name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise AppError(
                f"Unsupported file type for '{safe_name}'. Upload a PDF, TXT, or Markdown file.",
                415,
                "unsupported_file_type",
            )

        pages = parse_document(safe_name, data)
        chunks = chunk_pages(pages, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            raise EmptyDocument("No extractable text was found in that file.")
        if len(chunks) > MAX_CHUNKS:
            raise AppError(
                f"That file produced {len(chunks)} passages, above the limit of {MAX_CHUNKS}. "
                "Split it into smaller files.",
                400,
                "too_many_chunks",
            )

        document_id = str(uuid4())
        path = self._upload_path(document_id, safe_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        record = DocumentRecord(
            id=document_id,
            filename=safe_name,
            content_type=SUPPORTED_EXTENSIONS[suffix],
            size_bytes=len(data),
            chunk_count=len(chunks),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        try:
            self.store.add(document_id, safe_name, chunks)
            self.catalog.insert(record)
        except Exception:
            self._rollback(document_id, path)
            raise

        logger.info(
            "Ingested document %s filename=%s chunks=%s bytes=%s",
            document_id,
            safe_name,
            len(chunks),
            len(data),
        )
        return record

    def list_documents(self) -> list[DocumentRecord]:
        return self.catalog.list_documents()

    def delete(self, document_id: str) -> None:
        parsed = _parse_id(document_id)
        record = self.catalog.get(parsed)
        if record is None:
            raise AppError("Document not found.", 404, "not_found")
        self.store.delete_document(parsed)
        self.catalog.delete(parsed)
        path = self._upload_path(parsed, record.filename)
        path.unlink(missing_ok=True)
        logger.info("Deleted document %s", parsed)

    def ask(
        self,
        question: str,
        header_key: str | None,
        body_key: str | None,
        top_k: int | None,
    ) -> Answer:
        cleaned = question.strip()
        if not cleaned:
            raise AppError("Enter a question.", 400, "empty_question")
        api_key = resolve_api_key(header_key, body_key, self.settings.groq_api_key)
        if self.catalog.count() == 0:
            raise AppError("Upload a document before asking a question.", 400, "no_documents")

        limit = top_k if top_k is not None else self.settings.top_k
        sources = self.store.query(cleaned, limit)
        if not sources:
            raise AppError(
                "No matching passages were found. Try a different question.",
                400,
                "no_matches",
            )

        system, user = build_prompt(cleaned, sources)
        answer = self.llm.complete(api_key=api_key, system=system, user=user)
        logger.info("Answered question chars=%s sources=%s", len(cleaned), len(sources))
        return Answer(text=answer, sources=sources, model=self.settings.groq_model)

    def _upload_path(self, document_id: str, filename: str) -> Path:
        return self.settings.upload_dir / f"{document_id}_{filename}"

    def _rollback(self, document_id: str, path: Path) -> None:
        try:
            self.store.delete_document(document_id)
        except Exception:
            logger.warning("Could not roll back vectors for document %s", document_id)
        path.unlink(missing_ok=True)


def build_prompt(question: str, sources: list[RetrievedChunk]) -> tuple[str, str]:
    blocks = [f"[{source.label}]\n{source.text}" for source in sources]
    user = "Excerpts:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
    return SYSTEM_PROMPT, user


def _parse_id(value: str) -> str:
    try:
        return str(UUID(value))
    except ValueError as exc:
        raise AppError("Document not found.", 404, "not_found") from exc
