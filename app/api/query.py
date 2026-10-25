"""Question answering over the local index."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Header, Request

from app.schemas import QueryRequest, QueryResponse

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse, summary="Ask a question")
def query_documents(
    payload: QueryRequest,
    request: Request,
    x_groq_api_key: Annotated[str | None, Header(alias="X-Groq-Api-Key")] = None,
) -> QueryResponse:
    answer = request.app.state.rag.ask(
        payload.question,
        x_groq_api_key,
        payload.api_key,
        payload.top_k,
    )
    return QueryResponse.from_answer(answer)
