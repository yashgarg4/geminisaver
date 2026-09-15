"""Savings demo — run a realistic batch through the pipeline and report savings.

Runs a mix of unique, repeated, and paraphrased prompts across task types, then
prints an honest savings summary (baseline = always-frontier + no-cache).

    python examples/demo.py

Requires GOOGLE_API_KEY in .env. The output is designed to be copy-pasted into
the README as the "measured results".
"""

from __future__ import annotations

import asyncio
import sys
import time

from geminisaver.config import Config
from geminisaver.gemini import GeminiClient, GeminiResult, classify_error
from geminisaver.pipeline import Pipeline
from geminisaver.store import Store

# A realistic workload: cheap (classify/extract/format), medium (summarize/
# draft/QA), frontier (reason/debug) — plus exact repeats and paraphrases that
# a real app would issue, which is where the caching pays off.
PROMPTS: list[str] = [
    # --- cheap / mechanical ---
    "Classify this review as positive or negative: 'Fast shipping, great quality!'",
    "Extract all email addresses from: contact bob@acme.io or sales@acme.io",
    "Format this as JSON: name Alice, age 30, city Paris",
    "What is the sentiment of: 'I am thrilled with the results'?",
    # --- medium / everyday ---
    "Summarize in one sentence: The mitochondria is the powerhouse of the cell.",
    "Draft a short polite reply declining a meeting invite.",
    "What are the main differences between HTTP and HTTPS?",
    "Rewrite this to be more concise: 'Due to the fact that it was raining, we stayed in.'",
    # --- frontier / reasoning ---
    "Debug why this returns None: def f(x):\n  if x>0:\n    return x\n",
    "Design a simple rate limiter and explain the main trade-offs.",
    # --- exact repeats (should be hit-exact) ---
    "What are the main differences between HTTP and HTTPS?",
    "Classify this review as positive or negative: 'Fast shipping, great quality!'",
    # --- paraphrases (should be hit-semantic) ---
    "Can you summarize this in a single sentence: The mitochondria is the powerhouse of the cell.",
    "Whats the difference between HTTP and HTTPS mainly?",
    "Write a brief courteous message turning down a meeting invitation.",
    "How is HTTP different from HTTPS?",
    # --- a few more uniques to round out the batch ---
    "Translate the word 'hello' to Spanish.",
    "Give me a one-word antonym for 'expand'.",
    "What is the capital of Japan?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
]


class SimGemini:
    """Deterministic offline stand-in for GeminiClient.

    Returns realistic token counts derived from the prompt and the routed
    model, so the demo can exercise the *real* router + caches + meter without
    spending API quota. Token counts are synthetic (flagged in the output);
    the savings math and routing distribution are exactly the production paths.
    """

    # Rough output-token scale per tier — cheaper tiers do terser tasks.
    _OUT_SCALE = {
        "gemini-2.5-flash-lite": 40,
        "gemini-2.5-flash": 180,
        "gemini-3.8-flash": 320,
    }

    async def complete(self, messages, model, *, temperature=None, max_output_tokens=None):
        prompt = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        in_tokens = max(8, int(len(prompt.split()) * 1.3) + 6)
        out_tokens = self._OUT_SCALE.get(model, 150)
        return GeminiResult(
            text=f"[simulated {model} response]",
            in_tokens=in_tokens,
            out_tokens=out_tokens,
            model=model,
        )


async def _handle_with_retry(pipeline: Pipeline, prompt: str, retries: int = 3):
    """Run one prompt, retrying transient 5xx/timeout (upstream flakiness)."""
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(retries):
        try:
            return await pipeline.handle(messages)
        except Exception as exc:
            if classify_error(exc) in ("server", "timeout") and attempt < retries - 1:
                await asyncio.sleep(2 * (attempt + 1))
                continue
            raise


def _fmt(usd: float) -> str:
    return f"${usd:.6f}"


async def main() -> None:
    simulate = "--simulate" in sys.argv
    cfg = Config.from_env()

    if simulate:
        client = SimGemini()
        print("*** SIMULATE MODE: synthetic token counts, no API calls. ***")
        print("*** Proves the real router + caches + meter; not a billing measurement. ***\n")
    else:
        if not cfg.google_api_key:
            raise SystemExit("GOOGLE_API_KEY not set — copy .env.example to .env and add your key.")
        client = GeminiClient(cfg.google_api_key)

    # Persist to the same DB the dashboard reads, so `make dashboard` shows this run.
    store = Store(cfg.db_path)
    pipeline = Pipeline(cfg, client, store=store)

    print(f"Running {len(PROMPTS)} prompts through GeminiSaver...\n", flush=True)
    start = time.time()
    skipped = 0
    quota_warned = False
    for i, prompt in enumerate(PROMPTS, 1):
        try:
            result = await _handle_with_retry(pipeline, prompt)
        except Exception as exc:
            # One flaky/unavailable prompt shouldn't abort the batch — skip it.
            kind = classify_error(exc)
            skipped += 1
            print(f"{i:>2}. [   SKIPPED  ] ({kind}) {str(exc)[:60]}", flush=True)
            if kind == "rate_limit" and not quota_warned:
                quota_warned = True
                print(
                    "     Free-tier quota hit (20 requests/day/model). Remaining\n"
                    "     prompts on that model will also be skipped.\n"
                )
            continue
        tag = result.cache_status
        extra = ""
        if result.cache_status == "hit-semantic":
            extra = f" (sim {result.similarity:.3f})"
        elif result.cache_status == "miss":
            extra = f" -> {result.tier}"
        print(
            f"{i:>2}. [{tag:^12}]{extra:<14} cost {_fmt(result.cost)} "
            f"saved {_fmt(result.saved)}  | {prompt[:48]!r}",
            flush=True,
        )
    elapsed = time.time() - start

    t = pipeline.meter.totals()
    if t.requests == 0:
        print("\nNo requests completed — nothing to summarize.")
        return
    print("\n" + "=" * 64)
    title = "GeminiSaver - savings summary" + ("  (SIMULATED tokens)" if simulate else "")
    print(title)
    print("=" * 64)
    print(f"Requests           : {t.requests}  (in {elapsed:.1f}s)" + (f"  [{skipped} skipped]" if skipped else ""))
    print(f"Baseline (frontier): {_fmt(t.total_baseline)}")
    print(f"Actual (GeminiSaver): {_fmt(t.total_actual)}")
    print(f"SAVED              : {_fmt(t.total_saved)}  ({t.pct_reduction*100:.0f}% reduction)")
    print(f"Cache hit rate     : {t.hit_rate*100:.0f}%  ({t.calls_avoided} calls avoided)")
    print(f"Cache breakdown    : {t.cache_counts}")
    print(f"Tier split (misses): {t.tier_counts}")
    print("=" * 64)
    print(
        f"Baseline assumption: every request as {cfg.tier('frontier').model} "
        f"with no cache. Savings are baseline - actual; cache hits cost $0."
    )
    await store.close()
    print(f"\nPersisted {t.requests} requests to {cfg.db_path} - run `make dashboard` to view.")


if __name__ == "__main__":
    asyncio.run(main())
