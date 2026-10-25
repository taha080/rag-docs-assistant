"""Chunk boundaries, overlap, and citation labels."""

from app.citations import citation_label
from app.ingest.chunking import PageText, chunk_pages, split_text


def test_short_text_is_one_chunk() -> None:
    assert split_text("  Photosynthesis happens in leaves.  ", 200, 20) == [
        "Photosynthesis happens in leaves."
    ]


def test_blank_text_yields_nothing() -> None:
    assert split_text(" \n\t\r\n ", 200, 20) == []


def test_newlines_are_normalized_before_the_size_check() -> None:
    assert split_text("hello\r\n\r\nworld", 800, 0) == ["hello\n\nworld"]


def test_windows_overlap_on_a_long_unbroken_string() -> None:
    text = ("abcdefghijklmnopqrstuvwxyz" * 20)[:250]
    chunks = split_text(text, 100, 20)
    assert chunks
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert chunks[0] == text[:100]
    assert chunks[1][:20] == chunks[0][-20:]
    assert chunks[1][:20] == text[80:100]


def test_long_prose_terminates_and_keeps_words() -> None:
    chunks = split_text("word " * 2000, 300, 40)
    assert len(chunks) > 5
    assert all(len(chunk) <= 300 for chunk in chunks)
    assert "word" in " ".join(chunks)


def test_pages_are_not_merged() -> None:
    chunks = chunk_pages(
        [PageText(1, "Page one text."), PageText(2, "Page two text."), PageText(3, "   ")],
        chunk_size=800,
        overlap=40,
    )
    assert [(chunk.page, chunk.chunk_index, chunk.text) for chunk in chunks] == [
        (1, 0, "Page one text."),
        (2, 1, "Page two text."),
    ]


def test_overlap_must_be_smaller_than_the_chunk() -> None:
    try:
        split_text("hello", 10, 10)
    except ValueError as exc:
        assert "chunk_overlap" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_citation_labels() -> None:
    assert citation_label("notes.pdf", 2, 0) == "notes.pdf p.2 chunk 0"
    assert citation_label("notes.md", None, 3) == "notes.md chunk 3"
