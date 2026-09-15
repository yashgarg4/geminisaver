"""Honest savings metering.

The baseline is deliberately conservative and stated openly: what every request
*would* have cost with **no cache** and the **frontier model** (gemini-2.5-pro).
That's the "naive but working" setup a developer might otherwise ship. Savings
are baseline − actual, where actual is the real cost (0 on a cache hit, or the
routed tier's price on a miss). We never inflate the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config


@dataclass
class Record:
    request_id: str
    cache_status: str  # miss | hit-exact | hit-semantic
    tier: str | None
    model: str
    in_tokens: int
    out_tokens: int
    actual_cost: float
    baseline_cost: float

    @property
    def saved(self) -> float:
        return self.baseline_cost - self.actual_cost


@dataclass
class Totals:
    requests: int
    total_actual: float
    total_baseline: float
    total_saved: float
    hit_rate: float
    calls_avoided: int
    pct_reduction: float
    tier_counts: dict[str, int]
    cache_counts: dict[str, int]


class SavingsMeter:
    """In-memory accumulator of per-request savings (Phase 4 persists these)."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.records: list[Record] = []

    def baseline_cost(self, in_tokens: int, out_tokens: int) -> float:
        """Cost if this request had used the frontier model with no cache."""
        return self.cfg.frontier_pricing().cost(in_tokens, out_tokens)

    def record(
        self,
        request_id: str,
        cache_status: str,
        tier: str | None,
        model: str,
        in_tokens: int,
        out_tokens: int,
        actual_cost: float,
    ) -> Record:
        rec = Record(
            request_id=request_id,
            cache_status=cache_status,
            tier=tier,
            model=model,
            in_tokens=in_tokens,
            out_tokens=out_tokens,
            actual_cost=actual_cost,
            baseline_cost=self.baseline_cost(in_tokens, out_tokens),
        )
        self.records.append(rec)
        return rec

    def totals(self) -> Totals:
        n = len(self.records)
        total_actual = sum(r.actual_cost for r in self.records)
        total_baseline = sum(r.baseline_cost for r in self.records)
        total_saved = total_baseline - total_actual
        hits = sum(1 for r in self.records if r.cache_status != "miss")

        tier_counts: dict[str, int] = {}
        cache_counts: dict[str, int] = {}
        for r in self.records:
            if r.cache_status == "miss" and r.tier:
                tier_counts[r.tier] = tier_counts.get(r.tier, 0) + 1
            cache_counts[r.cache_status] = cache_counts.get(r.cache_status, 0) + 1

        return Totals(
            requests=n,
            total_actual=total_actual,
            total_baseline=total_baseline,
            total_saved=total_saved,
            hit_rate=(hits / n) if n else 0.0,
            calls_avoided=hits,
            pct_reduction=(total_saved / total_baseline) if total_baseline else 0.0,
            tier_counts=tier_counts,
            cache_counts=cache_counts,
        )
