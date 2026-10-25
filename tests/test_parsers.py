"""Parsers for PDF, text, and Markdown."""

from io import BytesIO

from pypdf import PdfWriter

from app.errors import EmptyDocument, UnreadableFile, UnsupportedFileType
from app.ingest.chunking import chunk_pages
from app.ingest.parsers import parse_document
from tests.pdfutil import build_pdf


def test_text_and_markdown_have_no_page_number() -> None:
    pages = parse_document("notes.md", b"# Title\n\nPhotosynthesis happens in the chloroplast.")
    assert len(pages) == 1
    assert pages[0].number is None
    assert "chloroplast" in pages[0].text


def test_utf8_bom_is_stripped() -> None:
    pages = parse_document("notes.txt", "cider".encode("utf-8-sig"))
    assert pages[0].text == "cider"


def test_utf16_is_rejected() -> None:
    # UTF-16 text contains NUL bytes, so it is treated as binary rather than text.
    try:
        parse_document("notes.txt", "cider".encode("utf-16"))
    except UnreadableFile as exc:
        assert "plain text" in exc.message
    else:
        raise AssertionError("expected UnreadableFile")


def test_invalid_utf8_is_rejected() -> None:
    try:
        parse_document("notes.txt", b"\xff\xfe cider")
    except UnreadableFile as exc:
        assert "UTF-8" in exc.message
    else:
        raise AssertionError("expected UnreadableFile")


def test_nul_bytes_are_rejected() -> None:
    try:
        parse_document("notes.txt", b"hello\x00world")
    except UnreadableFile:
        return
    raise AssertionError("expected UnreadableFile")


def test_unsupported_extension() -> None:
    try:
        parse_document("picture.png", b"not an image")
    except UnsupportedFileType as exc:
        assert exc.status_code == 415
    else:
        raise AssertionError("expected UnsupportedFileType")


def test_pdf_pages_keep_their_numbers() -> None:
    data = build_pdf(
        [
            "Photosynthesis happens inside the chloroplast.",
            "The French Revolution began in 1789.",
        ]
    )
    pages = parse_document("history.pdf", data)
    assert [page.number for page in pages] == [1, 2]
    assert "chloroplast" in pages[0].text
    assert "1789" in pages[1].text
    chunks = chunk_pages(pages, chunk_size=400, overlap=20)
    assert [chunk.page for chunk in chunks] == [1, 2]


def test_blank_pdf_has_no_passages() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buffer = BytesIO()
    writer.write(buffer)
    pages = parse_document("blank.pdf", buffer.getvalue())
    assert chunk_pages(pages, chunk_size=200, overlap=20) == []


def test_encrypted_pdf_is_rejected() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.encrypt("secret")
    buffer = BytesIO()
    writer.write(buffer)
    try:
        parse_document("secret.pdf", buffer.getvalue())
    except UnreadableFile as exc:
        assert "Encrypted" in exc.message
    else:
        raise AssertionError("expected UnreadableFile")


def test_damaged_pdf_is_rejected() -> None:
    try:
        parse_document("broken.pdf", b"%PDF-1.4 this is not a pdf")
    except UnreadableFile:
        return
    raise AssertionError("expected UnreadableFile")


def test_empty_file() -> None:
    try:
        parse_document("notes.txt", b"")
    except EmptyDocument:
        return
    raise AssertionError("expected EmptyDocument")
