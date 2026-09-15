"""SQLite persistence so savings survive a restart.

Async writes (aiosqlite) on the request hot path; synchronous reads (stdlib
sqlite3) for the Streamlit dashboard. WAL mode lets the dashboard read while
the proxy writes. One table: one row per request.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import aiosqlite

from .savings import Record

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id            TEXT PRIMARY KEY,
    ts            REAL NOT NULL,
    cache_status  TEXT NOT NULL,
    tier          TEXT,
    model         TEXT,
    in_tokens     INTEGER NOT NULL,
    out_tokens    INTEGER NOT NULL,
    actual_cost   REAL NOT NULL,
    baseline_cost REAL NOT NULL,
    saved         REAL NOT NULL
);
"""


class Store:
    """Async writer with a lazily-opened, long-lived connection."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> aiosqlite.Connection:
        if self._db is None:
            self._db = await aiosqlite.connect(self.path)
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute(_SCHEMA)
            await self._db.commit()
        return self._db

    async def insert_request(self, rec: Record, ts: float | None = None) -> None:
        db = await self.connect()
        await db.execute(
            "INSERT OR REPLACE INTO requests "
            "(id, ts, cache_status, tier, model, in_tokens, out_tokens, "
            " actual_cost, baseline_cost, saved) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rec.request_id,
                ts if ts is not None else time.time(),
                rec.cache_status,
                rec.tier,
                rec.model,
                rec.in_tokens,
                rec.out_tokens,
                rec.actual_cost,
                rec.baseline_cost,
                rec.saved,
            ),
        )
        await db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None


# --- Synchronous reads for the dashboard (no event loop needed) ---


def _connect_ro(path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def read_totals(path: Path | str) -> dict:
    """Aggregate KPIs. Returns zeros if the db/table is empty or missing."""
    empty = {
        "requests": 0,
        "total_actual": 0.0,
        "total_baseline": 0.0,
        "total_saved": 0.0,
        "hit_rate": 0.0,
        "calls_avoided": 0,
        "pct_reduction": 0.0,
    }
    if not Path(path).exists():
        return empty
    conn = _connect_ro(path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) n, "
            "COALESCE(SUM(actual_cost),0) a, "
            "COALESCE(SUM(baseline_cost),0) b, "
            "COALESCE(SUM(saved),0) s, "
            "COALESCE(SUM(CASE WHEN cache_status != 'miss' THEN 1 ELSE 0 END),0) hits "
            "FROM requests"
        ).fetchone()
    except sqlite3.OperationalError:
        return empty
    finally:
        conn.close()

    n = row["n"] or 0
    if n == 0:
        return empty
    return {
        "requests": n,
        "total_actual": row["a"],
        "total_baseline": row["b"],
        "total_saved": row["s"],
        "hit_rate": row["hits"] / n,
        "calls_avoided": row["hits"],
        "pct_reduction": (row["s"] / row["b"]) if row["b"] else 0.0,
    }


def read_rows(path: Path | str) -> list[dict]:
    """All request rows (oldest first). Empty list if none."""
    if not Path(path).exists():
        return []
    conn = _connect_ro(path)
    try:
        rows = conn.execute("SELECT * FROM requests ORDER BY ts ASC").fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    return [dict(r) for r in rows]
