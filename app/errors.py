"""Application errors that map to HTTP responses."""


class AppError(Exception):
    """A failure we know how to explain to the caller."""

    def __init__(self, message: str, status_code: int = 400, code: str = "error") -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


class UnsupportedFileType(AppError):
    def __init__(self, filename: str) -> None:
        super().__init__(
            f"Unsupported file type for '{filename}'. Upload a PDF, TXT, or Markdown file.",
            415,
            "unsupported_file_type",
        )


class UnreadableFile(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, 400, "unreadable_file")


class EmptyDocument(AppError):
    def __init__(self, message: str = "No extractable text was found in that file.") -> None:
        super().__init__(message, 400, "empty_document")


class LLMError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, 502, "llm_error")


class EmbeddingError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, 500, "embedding_error")
