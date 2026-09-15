"""Phase 1: the proxy returns a valid OpenAI-shaped response.

These tests use a fake Gemini client so they run offline (no API key, no
network). They assert the OpenAI contract and the GeminiSaver headers.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from geminisaver.config import Config
from geminisaver.gemini import GeminiResult
from geminisaver.proxy import app


class FakeGemini:
    """Stand-in for GeminiClient that records calls and returns fixed usage."""

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
def client() -> TestClient:
    # Inject config + fake Gemini so nothing hits the network.
    app.state.config = Config(google_api_key="test-key")
    app.state.gemini_client = FakeGemini()
    return TestClient(app)


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_chat_completion_openai_shape(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    # OpenAI Chat Completions contract.
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


def test_geminisaver_headers(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert resp.headers["x-geminisaver-cache"] == "miss"
    assert resp.headers["x-geminisaver-model"] == "gemini-2.5-flash"
    # flash: 12 in * $0.30/1M + 8 out * $2.50/1M = 3.6e-6 + 2.0e-5 = 2.36e-5
    assert resp.headers["x-geminisaver-cost-usd"] == f"{(12 * 0.30 + 8 * 2.50) / 1_000_000:.6f}"


def test_unknown_model_falls_back_to_default_tier(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",  # not a Gemini model
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200
    # default_tier is "medium" -> gemini-2.5-flash
    assert resp.headers["x-geminisaver-model"] == "gemini-2.5-flash"
