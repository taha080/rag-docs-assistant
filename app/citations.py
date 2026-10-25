"""Stable labels for a retrieved passage."""


def citation_label(filename: str, page: int | None, chunk_index: int) -> str:
    """Return the citation label shown to the user and sent to the model.

    PDF passages include a 1-based page number. Plain text has no pages, so the
    label is the file name and the chunk index only.
    """
    if page is None:
        return f"{filename} chunk {chunk_index}"
    return f"{filename} p.{page} chunk {chunk_index}"
