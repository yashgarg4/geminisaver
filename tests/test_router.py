"""Router tests.

Fast tests cover the keyword rules, length bump, escalation chain, and the
no-embedder default. One slow test exercises the real embedding fallback.
"""

from __future__ import annotations

import pytest

from geminisaver.config import Config
from geminisaver.router import Router


@pytest.fixture()
def cfg() -> Config:
    return Config(google_api_key="x")


def test_frontier_keyword(cfg: Config) -> None:
    d = Router(cfg).classify("Debug why this recursive function returns None")
    assert d.tier == "frontier"
    assert "keyword" in d.reason


def test_cheap_keyword(cfg: Config) -> None:
    d = Router(cfg).classify("Classify this customer review as positive or negative")
    assert d.tier == "cheap"
    assert "keyword" in d.reason


def test_no_embedder_uses_default(cfg: Config) -> None:
    d = Router(cfg).classify("Tell me something interesting")  # no keyword, no embedder
    assert d.tier == cfg.default_tier  # "medium"
    assert "default" in d.reason


def test_long_input_bumps_up_a_tier(cfg: Config) -> None:
    # 500 neutral words, no keyword, no embedder -> default medium, bumped to frontier.
    long_prompt = "lorem ipsum dolor sit amet " * 100  # 500 words
    d = Router(cfg).classify(long_prompt)
    assert d.tier == "frontier"
    assert "long input" in d.reason


def test_next_tier_chain(cfg: Config) -> None:
    r = Router(cfg)
    assert r.next_tier("cheap").name == "medium"
    assert r.next_tier("medium").name == "frontier"
    assert r.next_tier("frontier") is None


@pytest.mark.slow
def test_real_embedding_fallback(cfg: Config) -> None:
    pytest.importorskip("sentence_transformers")
    from geminisaver.cache.semantic import SemanticCache

    router = Router(cfg, embed_fn=SemanticCache().embed)

    # A reasoning task worded to avoid the literal keywords -> frontier.
    d = router.classify("Carefully work through this puzzle and justify each conclusion.")
    assert d.tier == "frontier"
    assert "exemplars" in d.reason

    # A mechanical task worded to avoid literal keywords -> cheap or medium.
    d2 = router.classify("Is this sentence upbeat or gloomy?")
    assert d2.tier in {"cheap", "medium"}
