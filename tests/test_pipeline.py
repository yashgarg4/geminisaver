"""Proxy + pipeline contract tests.

These run offline with a fake Gemini client and the semantic layer disabled
(no model download). They assert the OpenAI contract, the GeminiSaver headers,
and that the exact cache short-circuits the Gemini call.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from geminisaver.config import Config
from geminisaver.gemini import GeminiResult
from geminisaver.pipeline import Pipeline
from geminisaver.proxy import app


class FakeGemini:
    """Stand-in for GeminiClient that counts calls and returns fixed usage."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def complete(self, messages, model, *, temperature=None, max_output_tokens=None):
        self.calls.append({"messages": messages, "model": model})
        return GeminiResult(
            text="Paris is the capital of France.",
            in_tokens=12,
            out_tokens=8,
            model=model,
        )


@pytest.fixture()
def fake() -> FakeGemini:
    return FakeGemini()


@pytest.fixture()
def client(fake: FakeGemini) -> TestClient:
    cfg = Config(google_api_key="test-key")
    # semantic=None -> exact-only, no embedding model loaded in this suite.
    app.state.config = cfg
    app.state.gemini_client = fake
    app.state.pipeline = Pipeline(cfg, fake, semantic=None)
    return TestClient(app)


def _post(client: TestClient, content: str, model: str = "gemini-2.5-flash"):
    return client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": content}]},
    )


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_chat_completion_openai_shape(client: TestClient) -> None:
    resp = _post(client, "What is the capital of France?")
    assert resp.status_code == 200
    data = resp.json()

    assert data["object"] == "chat.completion"
    assert data["id"].startswith("chatcmpl-")
    assert data["model"] == "gemini-2.5-flash"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "Paris" in data["choices"][0]["message"]["content"]
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "total_tokens": 20,
    }


def test_headers_on_miss(client: TestClient) -> None:
    resp = _post(client, "hello there")
    assert resp.headers["x-geminisaver-cache"] == "miss"
    assert resp.headers["x-geminisaver-model"] == "gemini-2.5-flash"
    assert resp.headers["x-geminisaver-tier"] == "medium"
    expected = (12 * 0.30 + 8 * 2.50) / 1_000_000
    assert resp.headers["x-geminisaver-cost-usd"] == f"{expected:.6f}"


def test_exact_cache_skips_gemini(client: TestClient, fake: FakeGemini) -> None:
    first = _post(client, "How do I reset my password?")
    assert first.headers["x-geminisaver-cache"] == "miss"
    assert len(fake.calls) == 1

    second = _post(client, "How do I reset my password?")
    assert second.headers["x-geminisaver-cache"] == "hit-exact"
    assert second.headers["x-geminisaver-cost-usd"] == "0.000000"
    # No second Gemini call.
    assert len(fake.calls) == 1
    # Same answer replayed.
    assert second.json()["choices"][0]["message"]["content"] == first.json()[
        "choices"
    ][0]["message"]["content"]


def test_unknown_model_falls_back_to_default_tier(client: TestClient) -> None:
    resp = _post(client, "hi", model="gpt-4o")
    assert resp.status_code == 200
    assert resp.headers["x-geminisaver-model"] == "gemini-2.5-flash"
    assert resp.headers["x-geminisaver-tier"] == "medium"
