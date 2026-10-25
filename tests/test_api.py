"""HTTP API. Embeddings are lexical and Groq is a fake."""

from pathlib import Path

from fastapi.testclient import TestClient

from tests.fakes import FakeLLM
from tests.helpers import make_client, upload
from tests.pdfutil import build_pdf

PLANTS = "Photosynthesis happens in the chloroplast of a plant."
HISTORY = "The French Revolution began in Paris in 1789."
KEY = "gsk_testkeyvalue"


def test_health_lists_models_and_hides_any_server_key(tmp_path: Path) -> None:
    with make_client(tmp_path, groq_api_key="gsk_serversecret") as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["groq_configured"] is True
    assert body["embedding_model"] == "all-MiniLM-L6-v2"
    assert body["llm_model"] == "openai/gpt-oss-20b"
    assert body["document_count"] == 0
    assert "gsk_serversecret" not in response.text


def test_index_and_static_assets(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert "rag-docs-assistant" in page.text
    assert "Groq API key" in page.text
    assert "citation" in page.text.lower() or "passage" in page.text.lower()
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    script = client.get("/static/app.js").text
    assert "X-Groq-Api-Key" in script
    assert "sessionStorage" in script
    assert "gsk_" not in script


def test_upload_list_and_delete_text(client: TestClient) -> None:
    created = upload(client, "plants.md", PLANTS, "text/markdown")
    assert created.status_code == 201
    body = created.json()
    assert body["filename"] == "plants.md"
    assert body["chunk_count"] >= 1
    assert body["content_type"] == "text/markdown"

    listed = client.get("/api/documents")
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()] == [body["id"]]

    deleted = client.delete(f"/api/documents/{body['id']}")
    assert deleted.status_code == 204
    assert client.get("/api/documents").json() == []
    upload_dir = Path(client.app.state.settings.upload_dir)
    assert list(upload_dir.iterdir()) == []


def test_filename_cannot_escape_the_upload_directory(client: TestClient) -> None:
    response = upload(client, "../../etc/passwd.txt", "Photosynthesis in leaves.")
    assert response.status_code == 201
    assert "/" not in response.json()["filename"]
    assert "\\" not in response.json()["filename"]
    assert response.json()["filename"].endswith(".txt")


def test_rejects_unsupported_empty_and_large_files(tmp_path: Path) -> None:
    with make_client(tmp_path, max_upload_bytes=32) as client:
        rejected = upload(client, "image.png", b"not-an-image")
        assert rejected.status_code == 415
        assert rejected.json()["error"] == "unsupported_file_type"

        empty = upload(client, "notes.txt", b"")
        assert empty.status_code == 400
        assert empty.json()["error"] == "empty_document"

        blank = upload(client, "notes.txt", "  \n")
        assert blank.status_code == 400
        assert blank.json()["error"] == "empty_document"

        too_big = upload(client, "notes.txt", "a" * 33)
        assert too_big.status_code == 413
        assert too_big.json()["error"] == "file_too_large"

        accepted = upload(client, "notes.txt", "a" * 32)
        assert accepted.status_code == 201


def test_query_returns_the_matching_source_and_not_the_key(client: TestClient) -> None:
    upload(client, "plants.md", PLANTS, "text/markdown")
    upload(client, "history.md", HISTORY, "text/markdown")
    response = client.post(
        "/api/query",
        headers={"X-Groq-Api-Key": KEY},
        json={"question": "Where does photosynthesis happen?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["model"] == "openai/gpt-oss-20b"
    assert body["sources"][0]["filename"] == "plants.md"
    assert body["sources"][0]["page"] is None
    assert "chloroplast" in body["sources"][0]["text"].lower()
    assert body["sources"][0]["label"] == "plants.md chunk 0"
    assert KEY not in response.text

    llm: FakeLLM = client.app.state.llm
    call = llm.calls[-1]
    assert call["api_key"] == KEY
    assert "plants.md chunk 0" in call["user"]
    assert "chloroplast" in call["user"].lower()
    assert "Do not invent" in call["system"]
    assert KEY not in call["user"]
    assert KEY not in call["system"]


def test_pdf_citation_includes_the_page(client: TestClient) -> None:
    data = build_pdf([PLANTS, HISTORY])
    created = upload(client, "notes.pdf", data, "application/pdf")
    assert created.status_code == 201
    assert created.json()["chunk_count"] == 2

    response = client.post(
        "/api/query",
        headers={"X-Groq-Api-Key": KEY},
        json={"question": "When did the French Revolution begin?"},
    )
    assert response.status_code == 200
    source = response.json()["sources"][0]
    assert source["filename"] == "notes.pdf"
    assert source["page"] == 2
    assert source["label"] == "notes.pdf p.2 chunk 1"
    assert "1789" in source["text"]


def test_header_key_beats_body_and_server_key(tmp_path: Path) -> None:
    with make_client(tmp_path, groq_api_key="gsk_envkey") as client:
        upload(client, "plants.md", PLANTS, "text/markdown")
        response = client.post(
            "/api/query",
            headers={"X-Groq-Api-Key": "gsk_headerkey"},
            json={"question": "Where does photosynthesis happen?", "api_key": "gsk_bodykey"},
        )
        assert response.status_code == 200
        assert "gsk_headerkey" not in response.text
        assert "gsk_bodykey" not in response.text
        assert "gsk_envkey" not in response.text
        assert client.app.state.llm.calls[-1]["api_key"] == "gsk_headerkey"


def test_body_key_is_used_when_the_header_is_absent(client: TestClient) -> None:
    upload(client, "plants.md", PLANTS, "text/markdown")
    response = client.post(
        "/api/query",
        json={"question": "Where does photosynthesis happen?", "api_key": "gsk_bodykey"},
    )
    assert response.status_code == 200
    assert "gsk_bodykey" not in response.text
    assert client.app.state.llm.calls[-1]["api_key"] == "gsk_bodykey"


def test_missing_key_does_not_call_the_model(client: TestClient) -> None:
    upload(client, "plants.md", PLANTS, "text/markdown")
    response = client.post("/api/query", json={"question": "Where does photosynthesis happen?"})
    assert response.status_code == 400
    assert response.json()["error"] == "missing_api_key"
    assert client.app.state.llm.calls == []


def test_validation_error_does_not_echo_the_key(client: TestClient) -> None:
    response = client.post("/api/query", json={"question": "", "api_key": "gsk_supersecret"})
    assert response.status_code == 422
    assert "gsk_supersecret" not in response.text
    assert response.json()["error"] == "validation_error"


def test_query_without_documents(client: TestClient) -> None:
    response = client.post(
        "/api/query",
        headers={"X-Groq-Api-Key": KEY},
        json={"question": "Anything?"},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "no_documents"
    assert client.app.state.llm.calls == []


def test_delete_removes_passages_from_later_answers(client: TestClient) -> None:
    plants = upload(client, "plants.md", PLANTS, "text/markdown").json()
    upload(client, "history.md", HISTORY, "text/markdown")
    assert client.delete(f"/api/documents/{plants['id']}").status_code == 204
    response = client.post(
        "/api/query",
        headers={"X-Groq-Api-Key": KEY},
        json={"question": "Where does photosynthesis happen?"},
    )
    assert response.status_code == 200
    filenames = {source["filename"] for source in response.json()["sources"]}
    assert "plants.md" not in filenames


def test_unknown_document_is_404(client: TestClient) -> None:
    response = client.delete("/api/documents/not-a-uuid")
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_index_survives_a_restart(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        created = upload(client, "plants.md", PLANTS, "text/markdown")
        assert created.status_code == 201
    with make_client(tmp_path) as client:
        listed = client.get("/api/documents").json()
        assert len(listed) == 1
        response = client.post(
            "/api/query",
            headers={"X-Groq-Api-Key": KEY},
            json={"question": "Where does photosynthesis happen?"},
        )
        assert response.status_code == 200
        assert response.json()["sources"][0]["filename"] == "plants.md"


def test_logs_do_not_contain_the_request_key(client: TestClient, caplog) -> None:
    import logging

    upload(client, "plants.md", PLANTS, "text/markdown")
    with caplog.at_level(logging.INFO, logger="rag_docs"):
        response = client.post(
            "/api/query",
            headers={"X-Groq-Api-Key": "gsk_logsecret"},
            json={"question": "Where does photosynthesis happen?"},
        )
    assert response.status_code == 200
    assert "gsk_logsecret" not in caplog.text
