"""Cost-aware router: pick the cheapest Gemini tier that can handle a prompt.

The classifier makes **no LLM call** — that's the whole point. A classifier that
costs an API call to save an API call is pointless. It uses, in order:

1. Keyword rules  — fast, explainable ("code"/"debug" -> frontier;
   "classify"/"extract" -> cheap).
2. Length signal  — very long inputs are bumped up a tier; trivially short ones
   lean cheap.
3. Embedding similarity to per-tier *exemplars*, reusing the semantic cache's
   local embedder (free, already loaded).

It also owns the **capped fallback policy**: on a transient Gemini 5xx/timeout
the pipeline may escalate one tier (max depth 2). On a 429 it must NOT escalate
(that's a rate limit — back off instead).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import TIER_ORDER, Config, Tier

# Strong signals that a task needs real reasoning / long-form generation.
FRONTIER_KEYWORDS = (
    "debug", "refactor", "algorithm", "prove", "derive", "step by step",
    "step-by-step", "reason through", "analyze", "architecture", "design a",
    "optimize", "explain why", "trade-off", "tradeoff", "multi-step",
    "write code", "implement", "fix this", "stack trace", "complexity",
)

# Strong signals that a task is mechanical / short.
CHEAP_KEYWORDS = (
    "classify", "categorize", "label", "extract", "format", "reformat",
    "sentiment", "tag ", "yes or no", "true or false", "translate the word",
    "spelling", "capitalize", "to json", "to csv", "one word",
)

# Exemplar prompts per tier — the embedding fallback compares against these.
TIER_EXEMPLARS: dict[str, tuple[str, ...]] = {
    "cheap": (
        "Classify this review as positive or negative.",
        "Extract all email addresses from the text.",
        "Format this list as JSON.",
        "What is the sentiment of this sentence?",
    ),
    "medium": (
        "Summarize this article in three sentences.",
        "Draft a short polite reply to this email.",
        "What are the main differences between TCP and UDP?",
        "Rewrite this paragraph to be clearer.",
    ),
    "frontier": (
        "Debug why this recursive function returns the wrong value and fix it.",
        "Design a scalable rate limiter and explain the trade-offs.",
        "Prove that this algorithm runs in O(n log n).",
        "Walk through this multi-step math problem step by step.",
    ),
}

LONG_PROMPT_WORDS = 400   # inputs longer than this get bumped up a tier
SHORT_PROMPT_WORDS = 6    # trivially short inputs lean cheap
MAX_FALLBACK_DEPTH = 2    # capped escalation on transient errors


@dataclass
class Decision:
    tier: str
    model: str
    reason: str


class Router:
    def __init__(
        self,
        cfg: Config,
        embed_fn: Callable[[str], np.ndarray] | None = None,
    ) -> None:
        self.cfg = cfg
        self._embed_fn = embed_fn
        self._exemplar_vecs: dict[str, np.ndarray] | None = None

    # --- classification ---

    def classify(self, prompt: str) -> Decision:
        text = prompt.strip()
        lower = text.lower()
        words = len(text.split())

        # 1) Keyword rules (most explainable, checked first).
        for kw in FRONTIER_KEYWORDS:
            if kw in lower:
                return self._decide("frontier", f"matched complex keyword '{kw.strip()}'")
        for kw in CHEAP_KEYWORDS:
            if kw in lower:
                return self._decide("cheap", f"matched simple keyword '{kw.strip()}'")

        # 2) Embedding similarity to tier exemplars (free, local embedder).
        tier = None
        reason = ""
        if self._embed_fn is not None:
            tier, score = self._nearest_tier(text)
            reason = f"closest to {tier} exemplars (cosine {score:.2f})"
        else:
            tier = self.cfg.default_tier
            reason = "no embedder; using default tier"

        # 3) Length adjustments.
        if words >= LONG_PROMPT_WORDS and tier != "frontier":
            bumped = self._next_tier_name(tier)
            if bumped:
                return self._decide(
                    bumped, f"{reason}; bumped up for long input ({words} words)"
                )
        if words <= SHORT_PROMPT_WORDS and tier == "frontier" and self._embed_fn is None:
            # Only override the *default*-driven choice, never a real similarity one.
            return self._decide("cheap", f"very short input ({words} words)")

        return self._decide(tier, reason)

    def _nearest_tier(self, text: str) -> tuple[str, float]:
        vecs = self._ensure_exemplars()
        query = self._embed_fn(text)  # type: ignore[misc]
        query = np.asarray(query, dtype=np.float32)
        best_tier, best_score = self.cfg.default_tier, -1.0
        for tier_name, mat in vecs.items():
            score = float(np.max(mat @ query))  # max over that tier's exemplars
            if score > best_score:
                best_tier, best_score = tier_name, score
        return best_tier, best_score

    def _ensure_exemplars(self) -> dict[str, np.ndarray]:
        if self._exemplar_vecs is None:
            self._exemplar_vecs = {
                tier: np.vstack([self._embed_fn(ex) for ex in exemplars])  # type: ignore[misc]
                for tier, exemplars in TIER_EXEMPLARS.items()
                if tier in self.cfg.tiers
            }
        return self._exemplar_vecs

    def _decide(self, tier_name: str, reason: str) -> Decision:
        tier = self.cfg.tier(tier_name)
        return Decision(tier=tier.name, model=tier.model, reason=reason)

    # --- fallback policy ---

    def _next_tier_name(self, tier_name: str) -> str | None:
        idx = TIER_ORDER.index(tier_name)
        return TIER_ORDER[idx + 1] if idx + 1 < len(TIER_ORDER) else None

    def next_tier(self, tier_name: str) -> Tier | None:
        """The next tier up for escalation, or None if already at the top."""
        nxt = self._next_tier_name(tier_name)
        return self.cfg.tier(nxt) if nxt else None
