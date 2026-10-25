"""Filename checks shared by upload handling and the parsers."""

from __future__ import annotations

import re
from pathlib import Path

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}


def sanitize_filename(name: str) -> str:
    """Return a single safe file name with a lowercased extension.

    Directory pieces are dropped so a client cannot write outside the upload
    folder. Characters other than letters, numbers, spaces, dots, underscores,
    and hyphens are replaced.
    """
    base = (name or "").replace("\\", "/").split("/")[-1].replace("\x00", "")
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .")
    path = Path(base)
    suffix = path.suffix.lower()
    stem = path.stem.strip(" .") or "document"
    if suffix:
        cleaned = f"{stem}{suffix}"
    else:
        cleaned = stem
    return cleaned[:180] or "document"
