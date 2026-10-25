"""Persistent Chroma index. Embeddings are computed before they are stored."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.domain import RetrievedChunk
from app.errors import AppError, EmbeddingError
from app.ingest.chunking import Chunk
from app.services.embeddings import AppEmbeddingFunction, Embedder

logger = logging.getLogger("rag_docs")

_BATCH = 128


def relevance(distance: float) -> float:
    """Convert Chroma cosine distance into a 0-1 score where higher is closer."""
    score = 1.0 - float(distance)
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0
    return round(score, 4)


class VectorStore:
    """A persistent Chroma collection of document chunks."""

    def __init__(self, path: Path, embedder: Embedder, model_name: str) -> None:
        self.path = Path(path)
        self.embedder = embedder
        self.model_name = model_name
        self._client: object | None = None
        self._collection: object | None = None

    def add(self, document_id: str, filename: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        texts = [chunk.text for chunk in chunks]
        try:
            vectors = self.embedder.embed_documents(texts)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Could not embed the document. {exc}") from exc
        if len(vectors) != len(chunks):
            raise EmbeddingError("The embedding model returned an unexpected number of vectors.")

        ids = [f"{document_id}:{chunk.chunk_index}" for chunk in chunks]
        metadatas: list[dict[str, str | int]] = []
        for chunk in chunks:
            metadata: dict[str, str | int] = {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": chunk.chunk_index,
            }
            if chunk.page is not None:
                metadata["page"] = chunk.page
            metadatas.append(metadata)
        embeddings = [np.asarray(vector, dtype=np.float32) for vector in vectors]

        collection = self._collection_or_create()
        try:
            for start in range(0, len(ids), _BATCH):
                end = start + _BATCH
                collection.add(
                    ids=ids[start:end],
                    documents=texts[start:end],
                    metadatas=metadatas[start:end],
                    embeddings=embeddings[start:end],
                )
        except EmbeddingError:
            self._discard(document_id)
            raise
        except Exception as exc:
            self._discard(document_id)
            message = str(exc).lower()
            if "dimension" in message:
                raise AppError(
                    "The local vector store was created with a different embedding model. "
                    "Delete the data directory and upload the documents again.",
                    500,
                    "vector_store",
                ) from exc
            raise AppError("Could not store the document in the local index.", 500, "vector_store") from exc

    def delete_document(self, document_id: str) -> None:
        collection = self._collection_or_create()
        found = collection.get(where={"document_id": document_id})
        ids = found.get("ids") or []
        if ids:
            collection.delete(ids=ids)

    def query(self, text: str, top_k: int) -> list[RetrievedChunk]:
        collection = self._collection_or_create()
        count = int(collection.count())
        if count == 0:
            return []
        try:
            vector = np.asarray(self.embedder.embed_query(text), dtype=np.float32)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Could not embed the question. {exc}") from exc

        result = collection.query(query_embeddings=[vector], n_results=min(top_k, count))
        documents = (result.get("documents") or [[]])[0] or []
        metadatas = (result.get("metadatas") or [[]])[0] or []
        distances = (result.get("distances") or [[]])[0] or []

        hits: list[RetrievedChunk] = []
        for document, metadata, distance in zip(documents, metadatas, distances, strict=False):
            if not document or not metadata:
                continue
            page = metadata.get("page")
            hits.append(
                RetrievedChunk(
                    document_id=str(metadata.get("document_id", "")),
                    filename=str(metadata.get("filename", "document")),
                    page=int(page) if page is not None else None,
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    text=str(document),
                    score=relevance(float(distance)),
                )
            )
        return hits

    def close(self) -> None:
        client = self._client
        self._client = None
        self._collection = None
        if client is not None:
            client.close()

    def _discard(self, document_id: str) -> None:
        try:
            self.delete_document(document_id)
        except Exception:
            logger.warning("Could not roll back vectors for document %s", document_id)

    def _collection_or_create(self) -> object:
        if self._collection is not None:
            return self._collection
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name="chunks",
            metadata={"hnsw:space": "cosine"},
            embedding_function=AppEmbeddingFunction(
                model_name=self.model_name,
                embedder=self.embedder,
            ),
        )
        return self._collection
