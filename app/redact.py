"""Redact Groq API keys before they can reach a log line or an error body."""

from __future__ import annotations

import logging
import re

_GROQ_KEY = re.compile(r"(?:Bearer\s+)?gsk_[A-Za-z0-9_\-]+", re.IGNORECASE)


def redact(value: str) -> str:
    """Replace Groq-style keys with a placeholder."""
    return _GROQ_KEY.sub("gsk_[redacted]", value)


class RedactSecretsFilter(logging.Filter):
    """Logging filter that scrubs secrets from the message and its arguments."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if isinstance(record.args, dict):
            record.args = {
                key: redact(item) if isinstance(item, str) else item
                for key, item in record.args.items()
            }
        elif isinstance(record.args, tuple):
            record.args = tuple(
                redact(item) if isinstance(item, str) else item for item in record.args
            )
        return True


def install_redaction() -> None:
    """Attach the redaction filter to the application logger once."""
    logger = logging.getLogger("rag_docs")
    if not any(isinstance(item, RedactSecretsFilter) for item in logger.filters):
        logger.addFilter(RedactSecretsFilter())
