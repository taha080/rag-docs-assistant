"""SQLite catalog of uploaded files. Vectors live in Chroma, not here."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.domain import DocumentRecord


class Catalog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._session() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def insert(self, record: DocumentRecord) -> DocumentRecord:
        with self._session() as connection:
            connection.execute(
                """
                INSERT INTO documents (id, filename, content_type, size_bytes, chunk_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.filename,
                    record.content_type,
                    record.size_bytes,
                    record.chunk_count,
                    record.created_at,
                ),
            )
        return record

    def list_documents(self) -> list[DocumentRecord]:
        with self._session() as connection:
            rows = connection.execute(
                """
                SELECT id, filename, content_type, size_bytes, chunk_count, created_at
                FROM documents
                ORDER BY created_at DESC, filename ASC
                """
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def get(self, document_id: str) -> DocumentRecord | None:
        with self._session() as connection:
            row = connection.execute(
                """
                SELECT id, filename, content_type, size_bytes, chunk_count, created_at
                FROM documents
                WHERE id = ?
                """,
                (document_id,),
            ).fetchone()
        if row is None:
            return None
        return _row_to_record(row)

    def delete(self, document_id: str) -> None:
        with self._session() as connection:
            connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    def count(self) -> int:
        with self._session() as connection:
            row = connection.execute("SELECT COUNT(*) AS total FROM documents").fetchone()
        return int(row["total"]) if row is not None else 0

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def _row_to_record(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        id=str(row["id"]),
        filename=str(row["filename"]),
        content_type=str(row["content_type"]),
        size_bytes=int(row["size_bytes"]),
        chunk_count=int(row["chunk_count"]),
        created_at=str(row["created_at"]),
    )
