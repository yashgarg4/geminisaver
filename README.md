# GeminiSaver

**A self-hosted, drop-in proxy that cuts your Gemini API bill. Change one line — your `base_url` — and stop overpaying.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

GeminiSaver sits in front of the Gemini API and transparently applies **exact caching → semantic caching → cost-aware routing**, then meters exactly how much money it saved you. It ships the **self-hosted semantic cache that LiteLLM doesn't** — paraphrased prompts hit the cache instead of paying for another call — and it's **Gemini-focused on purpose**, so its pricing and routing are deep and accurate rather than shallow across 100 providers.

> ⚠️ **Status:** Phase 1 (OpenAI-compatible proxy + Gemini passthrough) is complete. Caching, routing, savings metering, and the dashboard land in Phases 2–4. Measured savings numbers will be filled in from the demo once routing/caching ship.

---

## 60-second quick start

```bash
# 1. clone, then create the venv
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install
pip install -r requirements.txt
pip install -e .

# 3. add your key
cp .env.example .env               # then edit .env: GOOGLE_API_KEY=...

# 4. run the proxy
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

That's it. Every request now flows through the savings pipeline.

## Features

| Feature | Status |
|---|---|
| OpenAI-compatible `/v1/chat/completions` | ✅ Phase 1 |
| Real Gemini token usage + per-request cost header | ✅ Phase 1 |
| Exact-match caching | 🔜 Phase 2 |
| **Self-hosted semantic caching (local embeddings)** | 🔜 Phase 2 |
| Cost-aware routing to cheapest sufficient tier | 🔜 Phase 3 |
| Honest savings metering (baseline vs actual) | 🔜 Phase 3 |
| Persistence + Streamlit savings dashboard | 🔜 Phase 4 |

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
  │ exact cache   ├────────▶│  return free │
  └──────┬────────┘         └──────────────┘
         ▼ miss
  ┌───────────────┐   hit   ┌──────────────┐
  │ semantic cache├────────▶│  return free │
  └──────┬────────┘         └──────────────┘
         ▼ miss
  ┌───────────────┐
  │    router     │  cheapest sufficient Gemini tier
  └──────┬────────┘
         ▼
  ┌───────────────┐
  │   gemini.py   │  real API call + real token usage
  └──────┬────────┘
         ▼
   store in caches  ·  meter savings  ·  OpenAI-shaped response
   headers: x-geminisaver-cache / -model / -tier / -cost-usd / -saved-usd
```

## How it saves money

1. **Exact caching** — identical prompts return instantly, for free (hash lookup, zero false positives).
2. **Semantic caching** — reworded prompts ("How do I reset my password?" vs "What's the way to reset my password?") map to nearby vectors and share one answer. Embeddings and vector search run **locally** — no data leaves your machine, no per-embedding cost.
3. **Cost-aware routing** — a fast, LLM-free classifier sends each prompt to the cheapest Gemini tier that can handle it (`flash-lite` → `flash` → `pro`) instead of always paying frontier prices.

Savings are measured **honestly**: baseline = *no cache, always frontier model*; actual = *what it really cost* (0 on a cache hit). Saved = baseline − actual.

## Why not just use LiteLLM / Helicone / Portkey?

| | GeminiSaver | LiteLLM | Helicone | Portkey |
|---|---|---|---|---|
| Self-hosted **semantic** cache, no external service | ✅ core | ⚠️ not batteries-included | ❌ | ⚠️ hosted-leaning |
| Cost-aware routing, Gemini-accurate pricing | ✅ | ✅ generic | ➖ | ✅ |
| Provider breadth | Gemini-only *by design* | ✅ 100+ | ✅ many | ✅ many |
| Maturity | 🌱 new | ✅ mature | ✅ mature | ✅ mature |

**Use GeminiSaver** if you're all-in on Gemini and want to cut spend on repetitive workloads without shipping prompts to a third party. **Use the others** if you need many providers or a mature hosted platform. Different jobs.

## Gemini tiers & pricing

Pricing (USD per 1M tokens) verified against [ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing):

| Tier | Model | Input | Output |
|---|---|---|---|
| cheap | `gemini-2.5-flash-lite` | $0.10 | $0.40 |
| medium | `gemini-2.5-flash` | $0.30 | $2.50 |
| frontier | `gemini-2.5-pro` | $1.25 / $2.50 (>200k) | $10.00 / $15.00 (>200k) |

All tiers, pricing, and thresholds live in [config.py](geminisaver/config.py).

## Project structure

```
geminisaver/      proxy, pipeline, cache/, router, gemini, savings, store, config
dashboard/        streamlit savings dashboard
examples/         integrate.py (one-line swap), demo.py (batch savings run)
tests/            pytest suite
```

## Running tests

```bash
make test          # or: python -m pytest -q
```

## License

MIT
