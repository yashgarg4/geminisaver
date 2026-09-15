"""Configuration: Gemini model tiers, real pricing, and cache/runtime settings.

Pricing was verified against https://ai.google.dev/gemini-api/docs/pricing
at build time. Rates are USD per 1,000,000 tokens. Model IDs change over
time, so treat these as the last-verified snapshot, not gospel.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelPricing(BaseModel):
    """Per-1M-token pricing for a single Gemini model.

    Gemini 2.5 Pro uses tiered pricing based on prompt size; the ``*_high``
    fields kick in above ``tier_threshold_tokens`` input tokens. Flat-priced
    models simply leave the high rates equal to the base rates.
    """

    input_per_1m: float
    output_per_1m: float
    input_per_1m_high: float | None = None
    output_per_1m_high: float | None = None
    tier_threshold_tokens: int = 200_000

    def cost(self, in_tokens: int, out_tokens: int) -> float:
        """USD cost for a call with the given token counts."""
        if self.input_per_1m_high is not None and in_tokens > self.tier_threshold_tokens:
            in_rate = self.input_per_1m_high
            out_rate = self.output_per_1m_high or self.output_per_1m
        else:
            in_rate = self.input_per_1m
            out_rate = self.output_per_1m
        return (in_tokens * in_rate + out_tokens * out_rate) / 1_000_000


class Tier(BaseModel):
    """A routing tier: a name, the Gemini model it maps to, and its pricing."""

    name: str
    model: str
    pricing: ModelPricing


# --- Gemini tiers (IDs + pricing verified against the LIVE API at build time) ---
# cheap:    classify / extract / format   (cheapest)
# medium:   summarize / draft / simple QA (mid)
# frontier: reasoning / code / multi-step (most capable)
#
# IMPORTANT: model IDs churn. `gemini-2.5-pro` was retired for new keys (404),
# and the true "pro" models require paid quota (429 on free keys). These
# defaults are the current, *callable* ladder verified via models.generate_content.
# To verify/refresh, run: python -m geminisaver.check_models  (see README).
#
# Paid-key upgrade: swap the frontier tier for a real Pro model, e.g.
#   model="gemini-3.1-pro-preview",
#   pricing=ModelPricing(input_per_1m=2.00, output_per_1m=12.00,
#                        input_per_1m_high=4.00, output_per_1m_high=18.00),
DEFAULT_TIERS: dict[str, Tier] = {
    "cheap": Tier(
        name="cheap",
        model="gemini-2.5-flash-lite",
        pricing=ModelPricing(input_per_1m=0.10, output_per_1m=0.40),
    ),
    "medium": Tier(
        name="medium",
        model="gemini-2.5-flash",
        pricing=ModelPricing(input_per_1m=0.30, output_per_1m=2.50),
    ),
    "frontier": Tier(
        name="frontier",
        model="gemini-3.8-flash",
        pricing=ModelPricing(input_per_1m=0.75, output_per_1m=3.75),
    ),
}

# Order from cheapest to most capable — used by the router for escalation.
TIER_ORDER: tuple[str, ...] = ("cheap", "medium", "frontier")


class Config(BaseSettings):
    """Runtime configuration, overridable via environment variables.

    Env vars use the ``GEMINISAVER_`` prefix (e.g. ``GEMINISAVER_CACHE_THRESHOLD``),
    except ``GOOGLE_API_KEY`` which keeps its conventional name.
    """

    model_config = SettingsConfigDict(
        env_prefix="GEMINISAVER_",
        env_file=".env",
        extra="ignore",
    )

    # Gemini auth (conventional name, no prefix).
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")

    # Model tiers + pricing.
    tiers: dict[str, Tier] = Field(default_factory=lambda: dict(DEFAULT_TIERS))

    # Semantic cache: cosine similarity above this counts as a hit.
    cache_threshold: float = 0.92
    semantic_cache_max_entries: int = 10_000
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Persistence.
    db_path: Path = Path("geminisaver.db")

    # Default tier when nothing is specified / routing is off.
    default_tier: str = "medium"

    # Server.
    host: str = "127.0.0.1"
    port: int = 8000

    @model_validator(mode="after")
    def _validate(self) -> "Config":
        if not self.tiers:
            raise ValueError("At least one tier must be configured.")
        if not (0.0 <= self.cache_threshold <= 1.0):
            raise ValueError("cache_threshold must be between 0 and 1 (cosine similarity).")
        if self.semantic_cache_max_entries < 1:
            raise ValueError("semantic_cache_max_entries must be >= 1.")
        if not (0 < self.port < 65536):
            raise ValueError("port must be in 1..65535.")
        if self.default_tier not in self.tiers:
            raise ValueError(
                f"default_tier '{self.default_tier}' is not one of {list(self.tiers)}."
            )
        if "frontier" not in self.tiers:
            raise ValueError("a 'frontier' tier is required for the savings baseline.")
        return self

    def tier(self, name: str) -> Tier:
        return self.tiers[name]

    def resolve_tier(self, requested_model: str) -> Tier:
        """Map a client-requested model id to a tier.

        Honors a known Gemini model id; otherwise falls back to the default
        tier (keeps the proxy drop-in for OpenAI clients that send e.g.
        ``gpt-4o``). Phase 3's router replaces this on cache misses.
        """
        for tier in self.tiers.values():
            if tier.model == requested_model:
                return tier
        return self.tier(self.default_tier)

    def frontier_pricing(self) -> ModelPricing:
        """Pricing used for the honest savings baseline (always-frontier)."""
        return self.tiers["frontier"].pricing

    @classmethod
    def from_env(cls) -> "Config":
        """Load config from environment / .env file."""
        return cls()
