# GeminiSaver

**A self-hosted, drop-in proxy that cuts your Gemini API bill. Change one line — your `base_url` — and stop overpaying.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-41%20passing-brightgreen)

GeminiSaver sits in front of the Gemini API and transparently applies **exact caching → semantic caching → cost-aware routing**, then meters exactly how much money it saved you. It ships the **self-hosted semantic cache that LiteLLM doesn't** — paraphrased prompts hit the cache instead of paying for another call — and it's **Gemini-focused on purpose**, so its pricing and routing are deep and accurate rather than shallow across 100 providers.

In a measured run on a repetitive-but-varied workload, it cut spend **68%** (see [Measured results](#measured-results)).

---

## 60-second quick start

```bash
git clone <your-fork-url> geminisaver && cd geminisaver

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .

cp .env.example .env               # then edit .env: GOOGLE_API_KEY=...
python -m geminisaver.check_models # verify the configured models are callable

make proxy                         # -> http://127.0.0.1:8000
```

## The one-line integration

GeminiSaver speaks the OpenAI Chat Completions API, so any OpenAI client works by changing a single line:

```diff
  from openai import OpenAI

  client = OpenAI(
      api_key="your-gemini-key",
-     base_url="https://generativelanguage.googleapis.com/v1beta/openai",
+     base_url="http://localhost:8000/v1",   # <-- the only change
  )

  resp = client.chat.completions.create(
      model="gemini-2.5-flash",
      messages=[{"role": "user", "content": "What is a semantic cache?"}],
  )
```

Every request now flows through caching + cost-aware routing. Each response carries observability headers:

```
x-geminisaver-cache: hit-semantic     # miss | hit-exact | hit-semantic
x-geminisaver-model: gemini-2.5-flash
x-geminisaver-tier: medium
x-geminisaver-cost-usd: 0.000000      # actual cost of THIS request (0 on a hit)
x-geminisaver-saved-usd: 0.008378     # vs the always-frontier baseline
x-geminisaver-similarity: 0.9853      # on a semantic hit
```

## Features

| Feature | Status |
|---|---|
| OpenAI-compatible `/v1/chat/completions` | ✅ |
| Exact-match caching (hash, zero false positives) | ✅ |
| **Self-hosted semantic caching (local bge-small embeddings)** | ✅ |
| Cost-aware routing to the cheapest sufficient Gemini tier | ✅ |
| Honest savings metering (baseline vs actual) | ✅ |
| Capped fallback on 5xx/timeout · 429 backoff | ✅ |
| SQLite persistence (survives restart, WAL) | ✅ |
| Streamlit savings dashboard | ✅ |
| Live model-ID verifier (`check_models`) | ✅ |

## Architecture

```
   client (OpenAI SDK)                 base_url = http://localhost:8000/v1
        │
        ▼
  ┌───────────────┐
  │   proxy.py    │  OpenAI-compatible endpoint
  └──────┬────────┘
         ▼
  ┌───────────────┐   hit   ┌──────────────┐
  │ exact cache   ├────────▶│  return free │  (hash lookup)
  └──────┬────────┘         └──────────────┘
         ▼ miss
  ┌───────────────┐   hit   ┌──────────────┐
  │ semantic cache├────────▶│  return free │  (local embedding + cosine ≥ threshold)
  └──────┬────────┘         └──────────────┘
         ▼ miss
  ┌───────────────┐
  │    router     │  cheapest sufficient tier (keywords → length → embedding)
  └──────┬────────┘
         ▼
  ┌───────────────┐
  │   gemini.py   │  real API call + real token usage
  └──────┬────────┘
         ▼
   store in caches  ·  meter savings  ·  persist (SQLite)  ·  OpenAI-shaped response
```

## How it saves money

1. **Exact caching** — identical prompts return instantly, for free (whitespace-normalized hash; zero false positives).
2. **Semantic caching** — reworded prompts ("How do I reset my password?" vs "What's the way to reset my password?") map to nearby vectors and share one answer. Embeddings (bge-small) and vector search (faiss, numpy fallback) run **locally** — no data leaves your machine, no per-embedding cost.
3. **Cost-aware routing** — a fast, **LLM-free** classifier (keyword rules → length signal → embedding similarity to tier exemplars) sends each prompt to the cheapest Gemini tier that can handle it (`flash-lite` → `flash` → frontier) instead of always paying top prices.

**Savings are measured honestly.** Baseline = *no cache, always the frontier model*. Actual = *what it really cost* (0 on a cache hit). `saved = baseline − actual`. The baseline assumption is printed with every result — never inflated.

## Measured results

Real run of `python examples/demo.py` — 20 varied + repetitive prompts, real Gemini calls:

```
Requests            : 18   (2 frontier prompts skipped — upstream 503 during capture)
Baseline (frontier) : $0.051477
Actual (GeminiSaver): $0.016516
SAVED               : $0.034961   (68% reduction)
Cache hit rate      : 28%  (5 calls avoided: 2 exact, 3 semantic)
Tier split (misses) : {cheap: 4, medium: 9}
```

Notes for full honesty:
- The frontier model was intermittently returning 503s during capture, so those 2 prompts were skipped (the demo skips failures and continues rather than aborting). A healthy re-run adds frontier-tier calls.
- Savings scale with how repetitive your workload is: the cache layers only pay off when prompts recur or are reworded. On unique-only traffic, savings come purely from routing.
- Reproduce offline with `python examples/demo.py --simulate` (real router + caches + meter, synthetic token counts — no API quota used).

## Why not just use LiteLLM / Helicone / Portkey?

| | GeminiSaver | LiteLLM | Helicone | Portkey |
|---|---|---|---|---|
| Self-hosted **semantic** cache, no external service | ✅ core | ⚠️ not batteries-included | ❌ (observability-first) | ⚠️ hosted-leaning |
| Exact cache | ✅ | ✅ | ✅ | ✅ |
| Cost-aware routing, Gemini-accurate pricing | ✅ | ✅ generic | ➖ | ✅ |
| Honest, provider-accurate savings metering | ✅ | ➖ | ✅ | ✅ |
| Provider breadth | Gemini-only *by design* | ✅ 100+ | ✅ many | ✅ many |
| Maturity / ecosystem | 🌱 new | ✅ mature | ✅ mature | ✅ mature |

**Use GeminiSaver** if you're all-in on Gemini and want to cut spend on repetitive workloads without shipping prompts to a third party. **Use the others** if you need many providers or a mature hosted platform. Different jobs — this one goes deep on one provider instead of wide across many.

## Gemini tiers & pricing

Model IDs and pricing are verified against the **live API** at build time (they change — `gemini-2.5-pro` was retired for new keys mid-build). Run `python -m geminisaver.check_models` to re-verify yours. Defaults (USD per 1M tokens):

| Tier | Model | Input | Output |
|---|---|---|---|
| cheap | `gemini-2.5-flash-lite` | $0.10 | $0.40 |
| medium | `gemini-2.5-flash` | $0.30 | $2.50 |
| frontier | `gemini-3.8-flash` | $0.75 | $3.75 |

Real Pro (`gemini-3.1-pro-preview`) requires paid quota; it's documented as a one-line upgrade in [config.py](geminisaver/config.py). All tiers, pricing, and thresholds live there.

## Configuration

Environment variables (see [.env.example](.env.example)), prefix `GEMINISAVER_`:

| Var | Default | Meaning |
|---|---|---|
| `GOOGLE_API_KEY` | — | Gemini API key (required) |
| `GEMINISAVER_CACHE_THRESHOLD` | `0.92` | Cosine similarity for a semantic hit (the false-positive dial) |
| `GEMINISAVER_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model |
| `GEMINISAVER_DB_PATH` | `geminisaver.db` | SQLite path |
| `GEMINISAVER_HOST` / `_PORT` | `127.0.0.1` / `8000` | Proxy bind |

## Project structure

```
geminisaver/
  proxy.py        OpenAI-compatible FastAPI app
  pipeline.py     exact -> semantic -> route -> call -> store + meter
  cache/          exact.py (hash) · semantic.py (embeddings + cosine + LRU)
  router.py       LLM-free complexity -> cheapest tier + fallback policy
  gemini.py       async google-genai wrapper + error classification
  savings.py      baseline-vs-actual metering
  store.py        aiosqlite persistence (async write, sync read, WAL)
  config.py       tiers, verified pricing, validated settings
  check_models.py live model-ID verifier
dashboard/app.py  Streamlit savings dashboard
examples/         integrate.py (one-line swap) · demo.py (savings batch)
tests/            41 tests
```

## Commands

```bash
make proxy       # run the proxy on :8000
make demo        # run the savings batch (add --simulate for offline)
make dashboard   # open the Streamlit dashboard
make test        # run the test suite
```

## Running tests

```bash
make test                  # fast suite
python -m pytest           # includes the slow real-model tests (downloads bge-small once)
```

## Limitations & roadmap

- **Single-node** (in-memory caches + local SQLite). Multi-node would need a shared vector store + DB.
- **Gemini-only by design.** Multi-provider is explicitly out of scope (that's LiteLLM's lane).
- **Threshold needs per-workload tuning.** 0.92 is a safe default; raise it if you see false hits, lower it to save more.
- **Next:** frontier-tier results in the measured numbers, a hosted option, and per-workload threshold auto-tuning.

## License

MIT
