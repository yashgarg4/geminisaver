"""OpenAI-compatible FastAPI proxy.

Phase 1: transparent passthrough to Gemini. A client only has to change its
``base_url`` to ``http://localhost:8000/v1`` — the request and response shapes
match the OpenAI Chat Completions API, so existing OpenAI/Gemini clients work
unchanged.

Later phases slot exact + semantic caching and cost-aware routing in front of
the Gemini call without changing this contract.
"""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel

from .config import Config
from .gemini import GeminiClient, classify_error, retry_delay_seconds
from .pipeline import Pipeline, PipelineResult
from .store import Store

app = FastAPI(title="GeminiSaver", version="0.1.0")


# --- OpenAI-compatible request/response schemas ---


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict] = ""


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    # Accept and ignore any other OpenAI fields (stream, top_p, ...).
    model_config = {"extra": "allow"}


class ResponseMessage(BaseModel):
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


# --- App wiring (lazy, test-overridable) ---


def get_config(request: Request) -> Config:
    cfg = getattr(request.app.state, "config", None)
    if cfg is None:
        cfg = Config.from_env()
        request.app.state.config = cfg
    return cfg


def get_gemini_client(request: Request) -> GeminiClient:
    client = getattr(request.app.state, "gemini_client", None)
    if client is None:
        cfg = get_config(request)
        client = GeminiClient(cfg.google_api_key)
        request.app.state.gemini_client = client
    return client


def get_pipeline(request: Request) -> Pipeline:
    """Lazily build the savings pipeline (exact + semantic caches + Gemini).

    Cached on ``app.state`` so caches persist across requests. Tests can set
    ``app.state.pipeline`` directly to inject fakes.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        cfg = get_config(request)
        pipeline = Pipeline(cfg, get_gemini_client(request), store=Store(cfg.db_path))
        request.app.state.pipeline = pipeline
    return pipeline


# --- Endpoints ---


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "geminisaver", "version": "0.1.0"}


@app.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest, request: Request, response: Response
) -> ChatCompletionResponse:
    try:
        pipeline = get_pipeline(request)
    except ValueError as exc:
        # e.g. missing GOOGLE_API_KEY — give the caller an actionable message.
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    if not body.messages:
        raise HTTPException(status_code=400, detail="'messages' must not be empty.")

    messages = [m.model_dump() for m in body.messages]

    try:
        result: PipelineResult = await pipeline.handle(messages)
    except Exception as exc:
        # Map upstream failures to sensible HTTP statuses for the caller.
        kind = classify_error(exc)
        if kind == "rate_limit":
            status, retry_after = 429, retry_delay_seconds(exc)
            headers = {"Retry-After": str(int(retry_after))}
        elif kind in ("server", "timeout"):
            status, headers = 503, None
        else:
            status, headers = 502, None
        raise HTTPException(
            status_code=status, detail=f"Gemini call failed ({kind}): {exc}", headers=headers
        ) from exc

    # GeminiSaver observability headers.
    response.headers["x-geminisaver-cache"] = result.cache_status
    response.headers["x-geminisaver-model"] = result.model
    response.headers["x-geminisaver-cost-usd"] = f"{result.cost:.6f}"
    response.headers["x-geminisaver-saved-usd"] = f"{result.saved:.6f}"
    if result.tier:
        response.headers["x-geminisaver-tier"] = result.tier
    if result.similarity is not None:
        response.headers["x-geminisaver-similarity"] = f"{result.similarity:.4f}"
    if result.route_reason is not None:
        response.headers["x-geminisaver-route-reason"] = result.route_reason

    return ChatCompletionResponse(
        id=result.request_id,
        created=int(time.time()),
        model=result.model,
        choices=[
            Choice(message=ResponseMessage(content=result.text)),
        ],
        usage=Usage(
            prompt_tokens=result.in_tokens,
            completion_tokens=result.out_tokens,
            total_tokens=result.in_tokens + result.out_tokens,
        ),
    )
