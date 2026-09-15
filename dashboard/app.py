"""GeminiSaver savings dashboard (Streamlit).

Reads the SQLite DB the proxy/demo write to and shows where the money went:
a hero "total saved" number, KPIs, cache + tier breakdowns, cumulative
baseline-vs-actual over time (the gap is the savings), and a recent-requests
table.

    make dashboard        # streamlit run dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from geminisaver.config import Config
from geminisaver.store import read_rows, read_totals

# --- Palette (validated categorical/ordinal steps from the dataviz skill) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#8a8a86"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
# Ordinal blue ramp for the cost tiers (light->dark == cheap->frontier).
TIER_COLORS = {"cheap": "#86b6ef", "medium": "#3987e5", "frontier": "#1c5cab"}
# Cache status: hits are colored; a miss is the neutral "paid full" bucket.
CACHE_COLORS = {"hit-exact": BLUE, "hit-semantic": AQUA, "miss": INK_MUTED}

st.set_page_config(page_title="GeminiSaver — Savings", page_icon="💸", layout="wide")


def _usd(x: float) -> str:
    return f"${x:,.4f}"


@st.cache_data(ttl=5)
def _load(db_path: str):
    return read_totals(db_path), read_rows(db_path)


def main() -> None:
    cfg = Config.from_env()
    db_path = str(cfg.db_path)

    st.title("💸 GeminiSaver — Savings Dashboard")
    st.caption(
        f"Reading `{db_path}`. Baseline = every request as "
        f"**{cfg.tier('frontier').model}** with no cache (honest, conservative)."
    )

    if not Path(db_path).exists():
        st.warning(
            f"No database at `{db_path}` yet. Run `make demo` (or send traffic "
            "through the proxy) to generate savings data, then refresh."
        )
        return

    totals, rows = _load(db_path)
    if totals["requests"] == 0:
        st.info("Database is empty. Run `make demo` to populate it, then refresh.")
        return

    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["ts"], unit="s")

    # --- Hero: total saved ---
    st.markdown("### ")
    hero_l, hero_r = st.columns([2, 3])
    with hero_l:
        st.metric(
            "TOTAL SAVED",
            _usd(totals["total_saved"]),
            f"{totals['pct_reduction']*100:.0f}% below baseline",
        )
    with hero_r:
        st.metric("Baseline (always-frontier, no cache)", _usd(totals["total_baseline"]))
        st.metric("Actual (with GeminiSaver)", _usd(totals["total_actual"]))

    # --- KPI row ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Requests", f"{totals['requests']:,}")
    k2.metric("Cache hit rate", f"{totals['hit_rate']*100:.0f}%")
    k3.metric("Calls avoided", f"{totals['calls_avoided']:,}")
    k4.metric("Avg saved / request", _usd(totals["total_saved"] / totals["requests"]))

    st.divider()

    # --- Cache breakdown + tier distribution ---
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Cache breakdown")
        cache_df = (
            df.groupby("cache_status").size().reset_index(name="count")
        )
        present = [s for s in CACHE_COLORS if s in cache_df["cache_status"].values]
        donut = (
            alt.Chart(cache_df)
            .mark_arc(innerRadius=60, stroke=SURFACE, strokeWidth=2)
            .encode(
                theta=alt.Theta("count:Q", stack=True),
                color=alt.Color(
                    "cache_status:N",
                    scale=alt.Scale(domain=present, range=[CACHE_COLORS[s] for s in present]),
                    legend=alt.Legend(title="status"),
                ),
                tooltip=["cache_status:N", "count:Q"],
            )
            .properties(height=280)
        )
        st.altair_chart(donut, use_container_width=True)

    with c2:
        st.subheader("Tier distribution (routed calls)")
        tier_df = df[df["cache_status"] == "miss"].copy()
        if tier_df.empty:
            st.caption("No routed calls yet (all served from cache).")
        else:
            order = [t for t in TIER_COLORS if t in tier_df["tier"].values]
            tier_counts = tier_df.groupby("tier").size().reset_index(name="count")
            bars = (
                alt.Chart(tier_counts)
                .mark_bar(cornerRadiusEnd=4, size=40)
                .encode(
                    x=alt.X("tier:N", sort=order, title=None),
                    y=alt.Y("count:Q", title="calls"),
                    color=alt.Color(
                        "tier:N",
                        scale=alt.Scale(domain=order, range=[TIER_COLORS[t] for t in order]),
                        legend=None,
                    ),
                    tooltip=["tier:N", "count:Q"],
                )
                .properties(height=280)
            )
            st.altair_chart(bars, use_container_width=True)

    st.divider()

    # --- Cumulative cost over time: baseline vs actual (gap = savings) ---
    st.subheader("Cumulative cost over time — the gap is your savings")
    ts_df = df.sort_values("ts").copy()
    ts_df["Baseline"] = ts_df["baseline_cost"].cumsum()
    ts_df["Actual"] = ts_df["actual_cost"].cumsum()
    long = ts_df.melt(
        id_vars=["dt"], value_vars=["Baseline", "Actual"],
        var_name="series", value_name="usd",
    )
    line = (
        alt.Chart(long)
        .mark_line(strokeWidth=2, interpolate="monotone")
        .encode(
            x=alt.X("dt:T", title=None),
            y=alt.Y("usd:Q", title="cumulative USD"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=["Baseline", "Actual"], range=[ORANGE, BLUE]),
                legend=alt.Legend(title=None, orient="top-left"),
            ),
            tooltip=["dt:T", "series:N", alt.Tooltip("usd:Q", format="$.4f")],
        )
        .properties(height=320)
    )
    st.altair_chart(line, use_container_width=True)

    st.divider()

    # --- Recent requests ---
    st.subheader("Recent requests")
    recent = df.sort_values("ts", ascending=False).head(25).copy()
    recent["saved"] = recent["saved"].map(_usd)
    recent["actual_cost"] = recent["actual_cost"].map(_usd)
    recent["baseline_cost"] = recent["baseline_cost"].map(_usd)
    recent["request"] = recent["id"].str.slice(0, 20)
    st.dataframe(
        recent[
            ["dt", "request", "cache_status", "tier", "model",
             "in_tokens", "out_tokens", "actual_cost", "baseline_cost", "saved"]
        ],
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
