"""Local sentence-transformers embeddings, plus the Chroma adapter."""

from __future__ import annotations

import logging
import threading
from typing import Protocol

import numpy as np
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function

from app.errors import EmbeddingError

logger = logging.getLogger("rag_docs")


class Embedder(Protocol):
    """Turns text into vectors. Tests supply a fake; production uses MiniLM."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages that will be stored."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a question. MiniLM uses the same space as documents."""


class SentenceTransformerEmbedder:
    """Lazy loader for a local SentenceTransformer model.

    The model is downloaded from Hugging Face on first use. Nothing is sent to
    a hosted embedding API.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: object | None = None
        self._lock = threading.RLock()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts, query=False)

    def embed_query(self, text: str) -> list[float]:
        vectors = self._encode([text], query=True)
        return vectors[0]

    def _encode(self, texts: list[str], *, query: bool) -> list[list[float]]:
        model = self._load()
        encode = model.encode_query if query else model.encode_document
        try:
            with self._lock:
                vectors = encode(
                    texts,
                    batch_size=32,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
        except Exception as exc:
            raise EmbeddingError(
                f"Could not embed text with {self.model_name}. {exc}"
            ) from exc
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[0] != len(texts):
            raise EmbeddingError("The embedding model returned an unexpected number of vectors.")
        return [row.tolist() for row in array]

    def _load(self) -> object:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is not installed. Install the project requirements and try again."
                ) from exc
            try:
                logger.info("Loading embedding model %s", self.model_name)
                self._model = SentenceTransformer(
                    self.model_name,
                    device="cpu",
                    trust_remote_code=False,
                )
            except Exception as exc:
                raise EmbeddingError(
                    f"Could not load the embedding model {self.model_name}. "
                    "The first run downloads it from Hugging Face. Check the network and the model name. "
                    f"{exc}"
                ) from exc
        return self._model


def _as_chroma(vectors: list[list[float]]) -> Embeddings:
    if not vectors:
        raise EmbeddingError("The embedding model returned no vectors.")
    return [np.asarray(vector, dtype=np.float32) for vector in vectors]


@register_embedding_function
class AppEmbeddingFunction(EmbeddingFunction[Documents]):
    """Chroma adapter. Callers normally pass vectors in themselves.

    Registering our own function keeps Chroma from downloading its separate
    ONNX copy of MiniLM. The sentence-transformers model remains the one that
    actually embeds text, via the injected embedder.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedder: Embedder | None = None) -> None:
        self.model_name = model_name
        self._embedder = embedder

    def __call__(self, input: Documents) -> Embeddings:
        texts = [str(item) for item in input]
        return _as_chroma(self._resolve().embed_documents(texts))

    def embed_query(self, input: Documents) -> Embeddings:
        texts = [str(item) for item in input]
        if len(texts) == 1:
            vectors = [self._resolve().embed_query(texts[0])]
        else:
            vectors = self._resolve().embed_documents(texts)
        return _as_chroma(vectors)

    def _resolve(self) -> Embedder:
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder(self.model_name)
        return self._embedder

    @staticmethod
    def name() -> str:
        return "rag_docs_sentence_transformer"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]

    def get_config(self) -> dict[str, str]:
        return {"model_name": self.model_name}

    @staticmethod
    def build_from_config(config: dict[str, str]) -> AppEmbeddingFunction:
        return AppEmbeddingFunction(model_name=str(config.get("model_name") or "all-MiniLM-L6-v2"))

    @staticmethod
    def validate_config(config: dict[str, str]) -> None:
        return None
