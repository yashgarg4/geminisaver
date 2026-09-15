"""Phase 5 robustness: 429 backoff, 5xx escalation, config validation, error mapping."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from geminisaver.config import Config
from geminisaver.gemini import GeminiResult, classify_error, retry_delay_seconds
from geminisaver.pipeline import Pipeline
from geminisaver.proxy import app


class FakeErr(Exception):
    """Mimics google-genai APIError with a status ``code``."""

    def __init__(self, code: int, msg: str = "") -> None:
        self.code = code
        super().__init__(msg)


def _ok(model: str) -> GeminiResult:
    return GeminiResult(text="ok", in_tokens=10, out_tokens=5, model=model)


# --- error classification / delay parsing ---


def test_classify_error_kinds() -> None:
    assert classify_error(FakeErr(429, "quota")) == "rate_limit"
    assert classify_error(FakeErr(503, "down")) == "server"
    assert classify_error(FakeErr(500, "boom")) == "server"
    assert classify_error(FakeErr(400, "bad")) == "other"


def test_retry_delay_parsing() -> None:
    assert retry_delay_seconds(FakeErr(429, "Please retry in 15.3s")) == pytest.approx(15.3)
    assert retry_delay_seconds(FakeErr(429, "retryDelay: '7s'")) == pytest.approx(7.0)
    assert retry_delay_seconds(FakeErr(429, "no hint"), default=1.0) == 1.0
    assert retry_delay_seconds(FakeErr(429, "retry in 999s"), cap=20) == 20.0


# --- pipeline retry / escalation ---


class Flaky:
    def __init__(self, fail_n: int, err: Exception) -> None:
        self.fail_n, self.err, self.calls = fail_n, err, 0

    async def complete(self, messages, model, *, temperature=None, max_output_tokens=None):
        self.calls += 1
        if self.calls <= self.fail_n:
            raise self.err
        return _ok(model)


async def test_429_backs_off_and_retries_once() -> None:
    cfg = Config(google_api_key="x")
    gem = Flaky(fail_n=1, err=FakeErr(429, "retry in 0s"))  # 0s -> no real sleep
    pipe = Pipeline(cfg, gem, semantic=None)
    res = await pipe.handle([{"role": "user", "content": "hello"}])
    assert res.cache_status == "miss"
    assert gem.calls == 2  # one failure + one successful retry


async def test_429_gives_up_after_one_backoff() -> None:
    cfg = Config(google_api_key="x")
    gem = Flaky(fail_n=2, err=FakeErr(429, "retry in 0s"))
    pipe = Pipeline(cfg, gem, semantic=None)
    with pytest.raises(FakeErr):
        await pipe.handle([{"role": "user", "content": "hello"}])
    assert gem.calls == 2  # backs off once, then re-raises (no infinite loop)


class OnlyFrontier:
    def __init__(self, frontier_model: str) -> None:
        self.frontier_model, self.models = frontier_model, []

    async def complete(self, messages, model, *, temperature=None, max_output_tokens=None):
        self.models.append(model)
        if model != self.frontier_model:
            raise FakeErr(503, "unavailable")
        return _ok(model)


async def test_5xx_escalates_tiers() -> None:
    cfg = Config(google_api_key="x")
    frontier = cfg.tier("frontier").model
    gem = OnlyFrontier(frontier)
    pipe = Pipeline(cfg, gem, semantic=None)
    # "classify ..." routes to cheap -> 503 -> medium -> 503 -> frontier -> ok.
    res = await pipe.handle([{"role": "user", "content": "classify this text please"}])
    assert res.tier == "frontier"
    assert res.model == frontier
    assert len(gem.models) == 3  # cheap, medium, frontier


# --- config validation ---


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cache_threshold": 1.5},
        {"cache_threshold": -0.1},
        {"default_tier": "nonexistent"},
        {"port": 0},
        {"semantic_cache_max_entries": 0},
    ],
)
def test_config_rejects_bad_values(kwargs) -> None:
    with pytest.raises(Exception):
        Config(google_api_key="x", **kwargs)


def test_config_accepts_valid_defaults() -> None:
    cfg = Config(google_api_key="x")
    assert 0 <= cfg.cache_threshold <= 1
    assert cfg.default_tier in cfg.tiers


# --- proxy error mapping ---


def test_proxy_maps_429_with_retry_after() -> None:
    cfg = Config(google_api_key="x")
    app.state.config = cfg
    app.state.pipeline = Pipeline(cfg, Flaky(99, FakeErr(429, "retry in 3s")), semantic=None)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 429
    assert "retry-after" in {k.lower() for k in resp.headers}


def test_proxy_rejects_empty_messages() -> None:
    cfg = Config(google_api_key="x")
    app.state.config = cfg
    app.state.pipeline = Pipeline(cfg, Flaky(0, FakeErr(500)), semantic=None)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert resp.status_code == 400
