"""Read PDF, plain text, and Markdown into pages of text."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.errors import EmptyDocument, UnreadableFile, UnsupportedFileType
from app.ingest.chunking import PageText
from app.ingest.files import SUPPORTED_EXTENSIONS

MAX_PDF_PAGES = 400


def parse_document(filename: str, data: bytes) -> list[PageText]:
    """Parse an uploaded file. The suffix selects the parser, not the MIME type."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileType(filename)
    if not data:
        raise EmptyDocument("The file is empty.")
    if suffix == ".pdf":
        return _parse_pdf(data)
    return _parse_text(data)


def _parse_text(data: bytes) -> list[PageText]:
    if b"\x00" in data:
        raise UnreadableFile("The file does not look like plain text.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UnreadableFile("Text and Markdown files must be UTF-8.") from exc
    return [PageText(number=None, text=text)]


def _parse_pdf(data: bytes) -> list[PageText]:
    try:
        reader = PdfReader(BytesIO(data))
    except PdfReadError as exc:
        raise UnreadableFile("Could not read that PDF. The file may be damaged.") from exc
    except Exception as exc:
        raise UnreadableFile("Could not read that PDF.") from exc

    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception as exc:
            raise UnreadableFile("Encrypted PDFs are not supported.") from exc
        if not unlocked:
            raise UnreadableFile("Encrypted PDFs are not supported.")

    if len(reader.pages) > MAX_PDF_PAGES:
        raise UnreadableFile(
            f"PDFs with more than {MAX_PDF_PAGES} pages are not supported. Split the file and upload the parts."
        )

    pages: list[PageText] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            raise UnreadableFile(f"Could not read page {index} of that PDF.") from exc
        pages.append(PageText(number=index, text=text))
    return pages
