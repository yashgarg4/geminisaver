"""Persistence tests: rows survive a 'restart', reads aggregate correctly."""

from __future__ import annotations

import pytest

from geminisaver.config import Config
from geminisaver.gemini import GeminiResult
from geminisaver.pipeline import Pipeline
from geminisaver.savings import Record
from geminisaver.store import Store, read_rows, read_totals


def _rec(rid: str, status: str, saved: float, actual: float, baseline: float) -> Record:
    return Record(
        request_id=rid,
        cache_status=status,
        tier="cheap",
        model="gemini-2.5-flash-lite",
        in_tokens=100,
        out_tokens=50,
        actual_cost=actual,
        baseline_cost=baseline,
    )


def test_empty_db_reads_are_safe(tmp_path) -> None:
    db = tmp_path / "none.db"
    assert read_totals(db)["requests"] == 0
    assert read_rows(db) == []


async def test_insert_and_read(tmp_path) -> None:
    db = tmp_path / "g.db"
    store = Store(db)
    await store.insert_request(_rec("r1", "miss", 0.005, 0.001, 0.006))
    await store.insert_request(_rec("r2", "hit-exact", 0.006, 0.0, 0.006))
    await store.close()

    totals = read_totals(db)
    assert totals["requests"] == 2
    assert totals["calls_avoided"] == 1
    assert totals["hit_rate"] == pytest.approx(0.5)
    assert totals["total_saved"] == pytest.approx(0.011)
    assert totals["total_baseline"] == pytest.approx(0.012)


async def test_data_survives_restart(tmp_path) -> None:
    db = tmp_path / "persist.db"
    store = Store(db)
    await store.insert_request(_rec("r1", "miss", 0.005, 0.001, 0.006))
    await store.close()

    # New Store instance == a fresh process reopening the same file.
    store2 = Store(db)
    await store2.connect()
    await store2.close()
    assert read_totals(db)["requests"] == 1


async def test_pipeline_persists_each_request(tmp_path) -> None:
    class FakeGemini:
        async def complete(self, messages, model, *, temperature=None, max_output_tokens=None):
            return GeminiResult(text="hi", in_tokens=10, out_tokens=5, model=model)

    cfg = Config(google_api_key="x")
    db = tmp_path / "pipe.db"
    store = Store(db)
    pipe = Pipeline(cfg, FakeGemini(), semantic=None, store=store)

    await pipe.handle([{"role": "user", "content": "hello"}])
    await pipe.handle([{"role": "user", "content": "hello"}])  # exact hit
    await store.close()

    rows = read_rows(db)
    assert len(rows) == 2
    assert {r["cache_status"] for r in rows} == {"miss", "hit-exact"}
