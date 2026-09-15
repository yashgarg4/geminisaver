"""Verify the configured Gemini model IDs against the live API.

Model IDs and access change without notice (a model can be retired, or gated
behind paid quota). This utility pings each configured tier's model with a
tiny request and reports whether it's actually callable on your key, so you
never ship a config that 404s or 429s in production.

    python -m geminisaver.check_models
"""

from __future__ import annotations

import asyncio

from google import genai

from .config import Config
from .gemini import classify_error


async def _ping(client: genai.Client, model: str) -> tuple[str, str]:
    try:
        resp = await client.aio.models.generate_content(model=model, contents="ping")
        usage = resp.usage_metadata
        tokens = usage.total_token_count if usage else "?"
        return "OK", f"responded ({tokens} tokens)"
    except Exception as exc:  # noqa: BLE001 - we want to report any failure
        code = getattr(exc, "code", "?")
        return f"FAIL {code}", f"{classify_error(exc)}: {str(exc)[:80]}"


async def main() -> None:
    cfg = Config.from_env()
    if not cfg.google_api_key:
        raise SystemExit("GOOGLE_API_KEY not set — copy .env.example to .env and add your key.")

    client = genai.Client(api_key=cfg.google_api_key)

    print("Verifying configured tier models against the live Gemini API:\n")
    all_ok = True
    for name in ("cheap", "medium", "frontier"):
        tier = cfg.tier(name)
        status, detail = await _ping(client, tier.model)
        all_ok = all_ok and status == "OK"
        print(f"  {name:9} {tier.model:24} {status:10} {detail}")

    print()
    if all_ok:
        print("All configured models are callable on this key. [OK]")
    else:
        print(
            "Some models are not callable. Edit DEFAULT_TIERS in geminisaver/config.py.\n"
            "List models available to your key with:\n"
            "  python -c \"from google import genai; from geminisaver.config import Config; "
            "[print(m.name) for m in genai.Client(api_key=Config.from_env().google_api_key)"
            ".models.list() if 'generateContent' in (m.supported_actions or [])]\""
        )


if __name__ == "__main__":
    asyncio.run(main())
