"""Settings validation and the secret-file guardrails."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_size=200, chunk_overlap=200)


def test_constructor_wins_over_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fromenv")
    settings = Settings(_env_file=None, groq_api_key="")
    assert settings.groq_configured is False


def test_env_example_and_gitignore() -> None:
    root = Path(__file__).resolve().parents[1]
    example = (root / ".env.example").read_text(encoding="utf-8")
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    assert "GROQ_API_KEY=" in example
    assert "gsk_" not in example
    assert ".env" in gitignore.splitlines()
    assert "!.env.example" in gitignore.splitlines()
