"""Savings meter tests — the baseline must be honest and the math exact."""

from __future__ import annotations

import pytest

from geminisaver.config import Config
from geminisaver.savings import SavingsMeter


@pytest.fixture()
def meter() -> SavingsMeter:
    return SavingsMeter(Config(google_api_key="x"))


def test_baseline_is_always_frontier(meter: SavingsMeter) -> None:
    # baseline uses the frontier tier's pricing, whatever it's configured to.
    fp = meter.cfg.frontier_pricing()
    expected = fp.cost(1000, 500)
    assert meter.baseline_cost(1000, 500) == pytest.approx(expected)
    # sanity: baseline must be at least as expensive as the cheapest tier.
    assert expected >= meter.cfg.tier("cheap").pricing.cost(1000, 500)


def test_miss_saved_is_baseline_minus_actual(meter: SavingsMeter) -> None:
    # Routed to cheap (flash-lite): 1000*0.10/1M + 500*0.40/1M
    actual = (1000 * 0.10 + 500 * 0.40) / 1_000_000
    rec = meter.record(
        "r1", "miss", "cheap", "gemini-2.5-flash-lite", 1000, 500, actual_cost=actual
    )
    baseline = meter.cfg.frontier_pricing().cost(1000, 500)
    assert rec.baseline_cost == pytest.approx(baseline)
    assert rec.saved == pytest.approx(baseline - actual)


def test_cache_hit_saves_full_baseline(meter: SavingsMeter) -> None:
    rec = meter.record(
        "r2", "hit-semantic", "medium", "gemini-2.5-flash", 800, 400, actual_cost=0.0
    )
    assert rec.actual_cost == 0.0
    assert rec.saved == pytest.approx(rec.baseline_cost)


def test_totals_aggregate(meter: SavingsMeter) -> None:
    meter.record("a", "miss", "cheap", "gemini-2.5-flash-lite", 1000, 500, 0.0003)
    meter.record("b", "miss", "frontier", "gemini-2.5-pro", 1000, 500, 0.00625)
    meter.record("c", "hit-exact", "cheap", "gemini-2.5-flash-lite", 1000, 500, 0.0)
    meter.record("d", "hit-semantic", "cheap", "gemini-2.5-flash-lite", 1000, 500, 0.0)

    t = meter.totals()
    assert t.requests == 4
    assert t.calls_avoided == 2
    assert t.hit_rate == pytest.approx(0.5)
    assert t.cache_counts == {"miss": 2, "hit-exact": 1, "hit-semantic": 1}
    assert t.tier_counts == {"cheap": 1, "frontier": 1}  # only misses counted
    assert t.total_saved == pytest.approx(t.total_baseline - t.total_actual)
    assert 0.0 < t.pct_reduction < 1.0


def test_empty_totals(meter: SavingsMeter) -> None:
    t = meter.totals()
    assert t.requests == 0
    assert t.hit_rate == 0.0
    assert t.pct_reduction == 0.0
