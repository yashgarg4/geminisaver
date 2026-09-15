"""The savings pipeline: exact -> semantic -> route -> call -> store + meter.

A request becomes either a cache hit (free) or a Gemini call routed to the
cheapest sufficient tier, with a capped escalation on transient errors. Every
request is metered (baseline vs actual) so we can report honest savings.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from .cache import CachedResponse
from .cache.exact import ExactCache
from .cache.semantic import SemanticCache
from .config import Config
from .gemini import GeminiResult, classify_error
from .router import MAX_FALLBACK_DEPTH, Router
from .savings import SavingsMeter

logger = logging.getLogger("geminisaver.pipeline")


@dataclass
class PipelineResult:
    """Everything the proxy needs to build a response + headers."""

    request_id: str
    text: str
    model: str
    in_tokens: int
    out_tokens: int
    cost: float  # actual USD cost of THIS request (0 on a cache hit)
    baseline_cost: float  # frontier + no-cache cost of the same tokens
    saved: float
    cache_status: str  # "hit-exact" | "hit-semantic" | "miss"
    tier: str | None = None
    similarity: float | None = None  # set on a semantic hit
    route_reason: str | None = None  # set on a miss (why this tier)


def _content_text(content) -> str:
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return content or ""


def render_messages(messages: list[dict]) -> str:
    """Canonical text for a request; the cache key and embedding input."""
    return "\n".join(
        f"{m.get('role', 'user')}: {_content_text(m.get('content'))}" for m in messages
    )


def last_user_text(messages: list[dict]) -> str:
    """The most recent user message — what the router classifies."""
    for m in reversed(messages):
        if m.get("role", "user") == "user":
            return _content_text(m.get("content"))
    return render_messages(messages)


_AUTO = object()


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        gemini_client,
        exact: ExactCache | None = None,
        semantic=_AUTO,
        router: Router | None = None,
        meter: SavingsMeter | None = None,
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
        embed_fn = self.semantic.embed if self.semantic is not None else None
        self.router = router if router is not None else Router(cfg, embed_fn=embed_fn)
        self.meter = meter if meter is not None else SavingsMeter(cfg)

    async def handle(self, messages: list[dict]) -> PipelineResult:
        # Note: the client's ``model`` field is advisory — GeminiSaver routes to
        # the cheapest sufficient tier itself. That's the product's whole point.
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        key_text = render_messages(messages)

        # 1) Exact cache — instant, free.
        exact_hit = self.exact.get(key_text)
        if exact_hit is not None:
            return self._finish_hit(request_id, "hit-exact", exact_hit)

        # 2) Semantic cache — embed once, reuse the vector for a miss-store.
        query_embedding = None
        if self.semantic is not None:
            query_embedding = self.semantic.embed(key_text)
            sem = self.semantic.get(
                key_text, self.cfg.cache_threshold, embedding=query_embedding
            )
            if sem is not None:
                entry, score = sem
                return self._finish_hit(request_id, "hit-semantic", entry, similarity=score)

        # 3) Miss -> route to the cheapest sufficient tier, then call Gemini
        #    with a capped escalation on transient errors.
        decision = self.router.classify(last_user_text(messages))
        tier = self.cfg.tier(decision.tier)
        result, tier = await self._call_with_fallback(messages, tier)
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

        rec = self.meter.record(
            request_id=request_id,
            cache_status="miss",
            tier=tier.name,
            model=result.model,
            in_tokens=result.in_tokens,
            out_tokens=result.out_tokens,
            actual_cost=cost,
        )
        return PipelineResult(
            request_id=request_id,
            text=result.text,
            model=result.model,
            in_tokens=result.in_tokens,
            out_tokens=result.out_tokens,
            cost=cost,
            baseline_cost=rec.baseline_cost,
            saved=rec.saved,
            cache_status="miss",
            tier=tier.name,
            route_reason=decision.reason,
        )

    def _finish_hit(
        self,
        request_id: str,
        status: str,
        entry: CachedResponse,
        similarity: float | None = None,
    ) -> PipelineResult:
        # Cache hit: actual cost 0; full baseline (frontier, no cache) is saved.
        rec = self.meter.record(
            request_id=request_id,
            cache_status=status,
            tier=entry.tier,
            model=entry.model,
            in_tokens=entry.in_tokens,
            out_tokens=entry.out_tokens,
            actual_cost=0.0,
        )
        return PipelineResult(
            request_id=request_id,
            text=entry.text,
            model=entry.model,
            in_tokens=entry.in_tokens,
            out_tokens=entry.out_tokens,
            cost=0.0,
            baseline_cost=rec.baseline_cost,
            saved=rec.saved,
            cache_status=status,
            tier=entry.tier,
            similarity=similarity,
        )

    async def _call_with_fallback(self, messages, tier):
        """Call Gemini; on a transient 5xx/timeout escalate a tier (capped).

        429 is a rate limit — never escalate on it (Phase 5 adds backoff);
        re-raise so the proxy surfaces it.
        """
        depth = 0
        while True:
            try:
                result: GeminiResult = await self.gemini.complete(messages, tier.model)
                return result, tier
            except Exception as exc:
                kind = classify_error(exc)
                if kind in ("server", "timeout") and depth < MAX_FALLBACK_DEPTH:
                    nxt = self.router.next_tier(tier.name)
                    if nxt is None:
                        raise
                    logger.warning(
                        "Gemini %s on %s; escalating %s -> %s (fallback %d/%d)",
                        kind, tier.model, tier.name, nxt.name, depth + 1, MAX_FALLBACK_DEPTH,
                    )
                    tier = nxt
                    depth += 1
                    continue
                raise
