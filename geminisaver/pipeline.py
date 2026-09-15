"""The savings pipeline: exact -> semantic -> (route) -> call -> store.

This is where a request is turned into either a cache hit (free) or a Gemini
call whose result is memoized into both caches. Phase 3 inserts routing +
savings metering at the marked point; the public contract here does not change.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cache import CachedResponse
from .cache.exact import ExactCache
from .cache.semantic import SemanticCache
from .config import Config
from .gemini import GeminiResult


@dataclass
class PipelineResult:
    """Everything the proxy needs to build a response + headers."""

    text: str
    model: str
    in_tokens: int
    out_tokens: int
    cost: float  # actual USD cost of THIS request (0 on a cache hit)
    cache_status: str  # "hit-exact" | "hit-semantic" | "miss"
    tier: str | None = None
    similarity: float | None = None  # set on a semantic hit


def _content_text(content) -> str:
    """Flatten OpenAI message content (str or list-of-parts) to plain text."""
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return content or ""


def render_messages(messages: list[dict]) -> str:
    """Canonical text for a request, used as the cache key/embedding input.

    Includes every role so different conversation histories key differently.
    """
    return "\n".join(f"{m.get('role', 'user')}: {_content_text(m.get('content'))}" for m in messages)


# Sentinel so callers can explicitly disable the semantic layer (tests) while
# still letting production lazily build a real one.
_AUTO = object()


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        gemini_client,
        exact: ExactCache | None = None,
        semantic=_AUTO,
    ) -> None:
        self.cfg = cfg
        self.gemini = gemini_client
        self.exact = exact if exact is not None else ExactCache()
        if semantic is _AUTO:
            semantic = SemanticCache(
                model_name=cfg.embedding_model,
                max_entries=cfg.semantic_cache_max_entries,
            )
        self.semantic: SemanticCache | None = semantic

    async def handle(self, messages: list[dict], requested_model: str) -> PipelineResult:
        key_text = render_messages(messages)
        tier = self.cfg.resolve_tier(requested_model)

        # 1) Exact cache — instant, free.
        exact_hit = self.exact.get(key_text)
        if exact_hit is not None:
            return PipelineResult(
                text=exact_hit.text,
                model=exact_hit.model,
                in_tokens=exact_hit.in_tokens,
                out_tokens=exact_hit.out_tokens,
                cost=0.0,
                cache_status="hit-exact",
                tier=exact_hit.tier,
            )

        # 2) Semantic cache — embed once, reuse the vector for a miss-store.
        query_embedding = None
        if self.semantic is not None:
            query_embedding = self.semantic.embed(key_text)
            sem = self.semantic.get(
                key_text, self.cfg.cache_threshold, embedding=query_embedding
            )
            if sem is not None:
                entry, score = sem
                return PipelineResult(
                    text=entry.text,
                    model=entry.model,
                    in_tokens=entry.in_tokens,
                    out_tokens=entry.out_tokens,
                    cost=0.0,
                    cache_status="hit-semantic",
                    tier=entry.tier,
                    similarity=score,
                )

        # 3) Miss -> (Phase 3 will route here) -> call Gemini.
        result: GeminiResult = await self.gemini.complete(messages, tier.model)
        cost = tier.pricing.cost(result.in_tokens, result.out_tokens)

        entry = CachedResponse(
            text=result.text,
            model=result.model,
            in_tokens=result.in_tokens,
            out_tokens=result.out_tokens,
            cost=cost,
            tier=tier.name,
        )
        # 4) Populate both caches so the next identical/similar prompt is free.
        self.exact.set(key_text, entry)
        if self.semantic is not None:
            self.semantic.add(key_text, entry, embedding=query_embedding)

        return PipelineResult(
            text=result.text,
            model=result.model,
            in_tokens=result.in_tokens,
            out_tokens=result.out_tokens,
            cost=cost,
            cache_status="miss",
            tier=tier.name,
        )
