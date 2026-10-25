"""Split extracted text into overlapping passages that still fit a citation."""

from __future__ import annotations

import re
from dataclasses import dataclass

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class PageText:
    """One page of a PDF, or the whole file when the format has no pages."""

    number: int | None
    text: str


@dataclass(frozen=True)
class Chunk:
    text: str
    chunk_index: int
    page: int | None


def chunk_pages(pages: list[PageText], chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk each page on its own so a citation never spans two pages."""
    chunks: list[Chunk] = []
    for page in pages:
        for text in split_text(page.text, chunk_size, overlap):
            chunks.append(Chunk(text=text, chunk_index=len(chunks), page=page.number))
    return chunks


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into pieces of at most `chunk_size` characters.

    Paragraphs are packed until another one would overflow. A paragraph that
    is still too long is split on sentences, then on a character window.
    `overlap` characters from the end of a chunk are repeated at the start of
    the next chunk when they fit, so a sentence that crosses the boundary
    remains searchable.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    normalized = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    buffer = ""
    for unit in _units(normalized, chunk_size, overlap):
        if not buffer:
            buffer = unit
            continue
        combined = f"{buffer}\n\n{unit}"
        if len(combined) <= chunk_size:
            buffer = combined
            continue
        chunks.append(buffer)
        buffer = _with_overlap(buffer, unit, overlap, chunk_size)
    if buffer:
        chunks.append(buffer)
    return [chunk for chunk in chunks if chunk.strip()]


def _units(text: str, chunk_size: int, overlap: int) -> list[str]:
    units: list[str] = []
    paragraphs = [part.strip() for part in _PARAGRAPH.split(text) if part.strip()]
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
            continue
        packed = _pack(_split_sentences(paragraph), chunk_size, " ")
        for piece in packed:
            if len(piece) <= chunk_size:
                units.append(piece)
            else:
                units.extend(_windows(piece, chunk_size, overlap))
    return units


def _split_sentences(paragraph: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE.split(paragraph) if part.strip()]
    return parts or [paragraph.strip()]


def _pack(parts: list[str], chunk_size: int, separator: str) -> list[str]:
    packed: list[str] = []
    current = ""
    for part in parts:
        if not current:
            current = part
            continue
        combined = f"{current}{separator}{part}"
        if len(combined) <= chunk_size:
            current = combined
        else:
            packed.append(current)
            current = part
    if current:
        packed.append(current)
    return packed


def _with_overlap(previous: str, unit: str, overlap: int, chunk_size: int) -> str:
    tail = _tail(previous, overlap)
    if not tail:
        return unit
    combined = f"{tail}\n\n{unit}"
    if len(combined) <= chunk_size:
        return combined
    return unit


def _tail(text: str, overlap: int) -> str:
    if overlap <= 0:
        return ""
    tail = text[-overlap:].strip()
    space = tail.find(" ")
    if 0 < space <= max(8, overlap // 5):
        tail = tail[space + 1 :].strip()
    return tail


def _windows(text: str, size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    guard = 0
    while start < len(text):
        guard += 1
        if guard > len(text) + 1:
            raise RuntimeError("chunker failed to advance")
        end = min(len(text), start + size)
        if end < len(text):
            snap = text.rfind(" ", start + (size // 2), end)
            if snap != -1:
                end = snap
        if end <= start:
            end = min(len(text), start + size)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return chunks
