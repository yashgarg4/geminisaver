"""Caching layers for GeminiSaver.

Two layers, checked in order by the pipeline:

- :class:`~geminisaver.cache.exact.ExactCache` — normalize + hash, instant,
  zero false positives.
- :class:`~geminisaver.cache.semantic.SemanticCache` — local embeddings +
  cosine similarity, catches paraphrases above a threshold.

Both store the same :class:`CachedResponse` payload so a hit can be replayed
as a full OpenAI response with the original token accounting.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CachedResponse:
    """What we remember about a completed Gemini call, keyed by prompt.

    ``in_tokens`` / ``out_tokens`` / ``cost`` are the *original* call's real
    figures. On a cache hit the actual cost is 0, but we keep the originals so
    the replayed response still reports meaningful usage and so the savings
    meter (Phase 3) knows exactly what was avoided.
    """

    text: str
    model: str
    in_tokens: int
    out_tokens: int
    cost: float
    tier: str | None = None


__all__ = ["CachedResponse"]
