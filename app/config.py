"""Runtime configuration loaded from the environment and an optional .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Self


class Settings(BaseSettings):
    """Settings for the API, the local index, and the Groq client.

    Constructor arguments win over environment variables, which win over `.env`.
    Tests pass a temporary directory and `_env_file=None` so a developer `.env`
    cannot leak into the suite.
    """

    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_timeout_seconds: float = Field(default=60, gt=0, le=180)
    groq_max_completion_tokens: int = Field(default=1024, ge=64, le=8192)
    groq_temperature: float = Field(default=0.5, ge=0, le=2)
    groq_reasoning_effort: str = "low"
    groq_include_reasoning: bool = False

    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_persist_dir: Path = Path("./data/chroma")
    upload_dir: Path = Path("./data/uploads")
    sqlite_path: Path = Path("./data/catalog.db")

    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)
    top_k: int = Field(default=4, ge=1, le=10)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1, le=50 * 1024 * 1024)
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    @model_validator(mode="after")
    def overlap_fits_in_chunk(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @property
    def groq_configured(self) -> bool:
        return bool(self.groq_api_key.strip())
