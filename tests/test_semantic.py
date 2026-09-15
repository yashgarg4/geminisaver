"""Semantic cache tests.

Most tests inject a deterministic fake embedder so they're fast and don't need
a model download. One test at the bottom uses the *real* bge-small model to
prove the actual wedge (paraphrases hit, unrelated prompts miss); it's skipped
if sentence-transformers isn't installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from geminisaver.cache import CachedResponse
from geminisaver.cache.semantic import SemanticCache


def _resp(tag: str) -> CachedResponse:
    return CachedResponse(
        text=f"answer-{tag}", model="gemini-2.5-flash", in_tokens=10, out_tokens=5, cost=0.001
    )


# A tiny deterministic embedding space. Vectors are normalized inside embed().
VECTORS = {
    "reset password": np.array([1.0, 0.0, 0.0]),
    "reset password paraphrase": np.array([0.98, 0.20, 0.0]),  # cosine ~0.98
    "weather tokyo": np.array([0.0, 1.0, 0.0]),  # cosine ~0 vs password
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


def _fake_embed(text: str) -> np.ndarray:
    return VECTORS[text]


def _cache(max_entries: int = 100) -> SemanticCache:
    return SemanticCache(max_entries=max_entries, embed_fn=_fake_embed)


def test_paraphrase_hits_above_threshold() -> None:
    cache = _cache()
    cache.add("reset password", _resp("pw"))

    hit = cache.get("reset password paraphrase", threshold=0.9)
    assert hit is not None
    entry, score = hit
    assert entry.text == "answer-pw"
    assert score > 0.9


def test_dissimilar_misses() -> None:
    cache = _cache()
    cache.add("reset password", _resp("pw"))
    assert cache.get("weather tokyo", threshold=0.9) is None


def test_empty_cache_returns_none() -> None:
    assert _cache().get("reset password", threshold=0.5) is None


def test_embeddings_are_unit_norm() -> None:
    cache = _cache()
    vec = cache.embed("reset password paraphrase")
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-3)


def test_lru_eviction() -> None:
    cache = _cache(max_entries=2)
    cache.add("x", _resp("x"))
    cache.add("y", _resp("y"))
    cache.add("z", _resp("z"))  # evicts "x" (LRU)

    assert len(cache) == 2
    assert cache.get("x", threshold=0.99) is None  # evicted
    assert cache.get("y", threshold=0.99) is not None
    assert cache.get("z", threshold=0.99) is not None


def test_lru_recency_refresh_on_hit() -> None:
    cache = _cache(max_entries=2)
    cache.add("x", _resp("x"))
    cache.add("y", _resp("y"))

    # Touch "x" so it becomes most-recently-used; "y" is now LRU.
    assert cache.get("x", threshold=0.99) is not None

    cache.add("z", _resp("z"))  # should evict "y", not "x"
    assert cache.get("y", threshold=0.99) is None
    assert cache.get("x", threshold=0.99) is not None
    assert cache.get("z", threshold=0.99) is not None


@pytest.mark.slow
def test_real_model_paraphrase_wedge() -> None:
    """The actual differentiator, with the real embedding model."""
    pytest.importorskip("sentence_transformers")
    cache = SemanticCache()  # real bge-small
    cache.add("How do I reset my password?", _resp("pw"))

    # Paraphrase should hit at a sane threshold.
    hit = cache.get("What's the way to reset my password?", threshold=0.85)
    assert hit is not None, "paraphrase should be a semantic hit"

    # Unrelated prompt should miss.
    assert cache.get("What's the weather in Tokyo?", threshold=0.85) is None
