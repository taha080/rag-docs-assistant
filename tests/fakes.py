"""Deterministic stand-ins for Groq and sentence-transformers."""

from __future__ import annotations

import hashlib
import math
import re


class LexicalEmbedder:
    """Bag-of-words hashing embedder.

    Shared tokens land in the same dimensions, so a question about
    photosynthesis ranks a photosynthesis passage above an unrelated one.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        values = [0.0] * self.dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            values[int.from_bytes(digest[:4], "little") % self.dim] += 1.0
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]


class FakeLLM:
    """Records prompts and returns a fixed answer. It never opens a socket."""

    def __init__(self, answer: str = "The cited passage answers the question.") -> None:
        self.answer = answer
        self.calls: list[dict[str, str]] = []

    def complete(self, *, api_key: str, system: str, user: str) -> str:
        if not api_key.strip():
            raise AssertionError("complete() was called without an API key")
        self.calls.append({"api_key": api_key, "system": system, "user": user})
        return self.answer
