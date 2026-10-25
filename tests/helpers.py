"""App factories for the API tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from tests.fakes import FakeLLM, LexicalEmbedder


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "groq_api_key": "",
        "chroma_persist_dir": tmp_path / "chroma",
        "upload_dir": tmp_path / "uploads",
        "sqlite_path": tmp_path / "catalog.db",
        "chunk_size": 180,
        "chunk_overlap": 30,
        "top_k": 3,
        "max_upload_bytes": 1_000_000,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@contextmanager
def make_client(tmp_path: Path, **overrides: object) -> Iterator[TestClient]:
    from chromadb.api.shared_system_client import SharedSystemClient

    from app.main import create_app

    llm = FakeLLM()
    application = create_app(
        make_settings(tmp_path, **overrides),
        embedder=LexicalEmbedder(),
        llm=llm,
    )
    try:
        with TestClient(application) as client:
            yield client
    finally:
        application.state.store.close()
        SharedSystemClient.clear_system_cache()


def upload(client: TestClient, name: str, content: bytes | str, content_type: str = "text/plain"):
    payload = content.encode("utf-8") if isinstance(content, str) else content
    return client.post(
        "/api/documents",
        files={"file": (name, payload, content_type)},
    )
