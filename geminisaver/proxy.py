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
import uuid

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel

from .config import Config
from .gemini import GeminiClient, GeminiResult

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


def _resolve_model(cfg: Config, requested: str) -> tuple[str, "object"]:
    """Map a requested model to (model_id, pricing).

    If the client asked for a known Gemini model we honor it; otherwise we
    fall back to the default tier. Phase 3 replaces this with real routing.
    """
    for tier in cfg.tiers.values():
        if tier.model == requested:
            return tier.model, tier.pricing
    default = cfg.tier(cfg.default_tier)
    return default.model, default.pricing


# --- Endpoints ---


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "geminisaver", "version": "0.1.0"}


@app.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest, request: Request, response: Response
) -> ChatCompletionResponse:
    cfg = get_config(request)
    try:
        client = get_gemini_client(request)
    except ValueError as exc:
        # e.g. missing GOOGLE_API_KEY — give the caller an actionable message.
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    model, pricing = _resolve_model(cfg, body.model)
    messages = [m.model_dump() for m in body.messages]

    try:
        result: GeminiResult = await client.complete(
            messages,
            model,
            temperature=body.temperature,
            max_output_tokens=body.max_tokens,
        )
    except Exception as exc:  # Phase 5 hardens this into typed error handling.
        raise HTTPException(status_code=502, detail=f"Gemini call failed: {exc}") from exc

    cost = pricing.cost(result.in_tokens, result.out_tokens)

    # GeminiSaver observability headers.
    response.headers["x-geminisaver-cache"] = "miss"
    response.headers["x-geminisaver-model"] = result.model
    response.headers["x-geminisaver-cost-usd"] = f"{cost:.6f}"

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
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
