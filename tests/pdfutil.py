"""Build a tiny text PDF that pypdf can read. Tests only."""

from __future__ import annotations


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(pages: list[str]) -> bytes:
    """Return a multi-page PDF whose visible text is exactly `pages`."""
    if not pages:
        raise ValueError("pages must not be empty")

    font_id = 3 + len(pages) * 2
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        font_id: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    kids: list[int] = []
    for index, text in enumerate(pages):
        page_id = 3 + index * 2
        content_id = page_id + 1
        kids.append(page_id)
        stream = f"BT /F1 12 Tf 72 720 Td ({_escape(text)}) Tj ET\n".encode("latin-1")
        objects[content_id] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"endstream"
        )
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        ).encode("ascii")

    kid_refs = " ".join(f"{kid} 0 R" for kid in kids)
    objects[2] = f"<< /Type /Pages /Count {len(pages)} /Kids [{kid_refs}] >>".encode("ascii")

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0] * (font_id + 1)
    for number in range(1, font_id + 1):
        offsets[number] = len(out)
        out.extend(f"{number} 0 obj\n".encode("ascii"))
        out.extend(objects[number])
        out.extend(b"\nendobj\n")

    xref = len(out)
    out.extend(f"xref\n0 {font_id + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for number in range(1, font_id + 1):
        out.extend(f"{offsets[number]:010d} 00000 n \n".encode("ascii"))
    out.extend(
        f"trailer\n<< /Size {font_id + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    return bytes(out)
