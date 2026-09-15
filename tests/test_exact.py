"""Exact-match cache tests."""

from __future__ import annotations

from geminisaver.cache import CachedResponse
from geminisaver.cache.exact import ExactCache


def _resp(text: str = "cached answer") -> CachedResponse:
    return CachedResponse(
        text=text, model="gemini-2.5-flash", in_tokens=10, out_tokens=5, cost=0.001
    )


def test_hit_and_miss() -> None:
    cache = ExactCache()
    assert cache.get("hello") is None  # miss on empty

    cache.set("hello", _resp("hi!"))
    hit = cache.get("hello")
    assert hit is not None
    assert hit.text == "hi!"


def test_whitespace_normalized_but_not_case() -> None:
    cache = ExactCache()
    cache.set("  How   do I\treset my password? ", _resp())

    # Whitespace differences still hit.
    assert cache.get("How do I reset my password?") is not None

    # Case differences do NOT hit (preserves zero false positives).
    assert cache.get("how do i reset my password?") is None


def test_overwrite() -> None:
    cache = ExactCache()
    cache.set("q", _resp("first"))
    cache.set("q", _resp("second"))
    assert cache.get("q").text == "second"
    assert len(cache) == 1
