"""Groq keys are scrubbed from log records and error text."""

import logging

from app.redact import RedactSecretsFilter, redact


def test_redact_replaces_groq_keys() -> None:
    text = "rejected bearer gsk_supersecretvalue now"
    assert "gsk_supersecretvalue" not in redact(text)
    assert "gsk_[redacted]" in redact(text)


def test_logging_filter_redacts_arguments() -> None:
    logger = logging.getLogger("rag_docs_redact_test")
    logger.setLevel(logging.INFO)
    logger.addFilter(RedactSecretsFilter())
    logger.propagate = False
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = Capture()
    logger.addHandler(handler)
    try:
        logger.info("header %s", "Bearer gsk_fromheader99")
    finally:
        logger.removeHandler(handler)
    assert records
    assert "gsk_fromheader99" not in records[0].getMessage()
    assert "gsk_[redacted]" in records[0].getMessage()
