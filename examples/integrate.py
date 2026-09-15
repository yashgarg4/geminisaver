"""One-line integration demo.

GeminiSaver is OpenAI-compatible, so any OpenAI client works by changing a
single line: the ``base_url``. Point it at the proxy and every request now
flows through caching + cost-aware routing — with no other code changes.

Run the proxy first:

    make proxy         # or: uvicorn geminisaver.proxy:app --port 8000

Then run this file:

    python examples/integrate.py

This uses the stdlib (httpx) rather than the openai package so it runs with no
extra install, but the shape is identical to what the openai SDK sends.
"""

from __future__ import annotations

import httpx

# ---------------------------------------------------------------------------
# BEFORE — talking straight to Gemini's OpenAI-compatible endpoint:
#
#   base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
#
# AFTER — the ONE line you change to route through GeminiSaver:
BASE_URL = "http://localhost:8000/v1"
# ---------------------------------------------------------------------------


def main() -> None:
    payload = {
        "model": "gemini-2.5-flash",
        "messages": [
            {"role": "user", "content": "In one sentence, what is a semantic cache?"}
        ],
    }

    resp = httpx.post(f"{BASE_URL}/chat/completions", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    print("Answer:", data["choices"][0]["message"]["content"].strip())
    print()
    print("--- GeminiSaver headers ---")
    print("cache   :", resp.headers.get("x-geminisaver-cache"))
    print("model   :", resp.headers.get("x-geminisaver-model"))
    print("cost usd:", resp.headers.get("x-geminisaver-cost-usd"))
    print("tokens  :", data["usage"])


if __name__ == "__main__":
    main()
