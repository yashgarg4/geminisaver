"""Async Gemini client wrapper.

Translates OpenAI-style ``messages`` into google-genai ``contents`` +
``system_instruction``, calls the model, and returns the text alongside the
*real* token usage reported by the API (which we need for honest costing).
"""

from __future__ import annotations

from dataclasses import dataclass

from google import genai
from google.genai import errors as genai_errors
from google.genai import types


def classify_error(exc: Exception) -> str:
    """Bucket a Gemini exception for routing decisions.

    Returns one of: ``"rate_limit"`` (429 — back off, never escalate),
    ``"server"`` (5xx — transient, safe to escalate a tier), ``"timeout"``
    (also escalate), or ``"other"``.
    """
    code = getattr(exc, "code", None)
    if code == 429:
        return "rate_limit"
    if isinstance(code, int) and 500 <= code <= 599:
        return "server"
    if isinstance(exc, genai_errors.ServerError):
        return "server"
    if isinstance(exc, TimeoutError) or "timeout" in str(exc).lower():
        return "timeout"
    return "other"


@dataclass
class GeminiResult:
    """Outcome of a single Gemini completion."""

    text: str
    in_tokens: int
    out_tokens: int
    model: str


def _split_messages(
    messages: list[dict],
) -> tuple[str | None, list[types.Content]]:
    """Convert OpenAI chat messages to (system_instruction, contents).

    OpenAI roles: system / user / assistant.
    Gemini roles: user / model  (system goes into system_instruction).
    Consecutive system messages are concatenated.
    """
    system_parts: list[str] = []
    contents: list[types.Content] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        # OpenAI allows content to be a list of parts; flatten to text.
        if isinstance(content, list):
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )

        if role == "system":
            if content:
                system_parts.append(content)
        else:
            gemini_role = "model" if role == "assistant" else "user"
            contents.append(
                types.Content(role=gemini_role, parts=[types.Part(text=content)])
            )

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents


class GeminiClient:
    """Thin async wrapper over google-genai."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        self._client = genai.Client(api_key=api_key)

    async def complete(
        self,
        messages: list[dict],
        model: str,
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> GeminiResult:
        """Run a completion and return text + real token usage."""
        system_instruction, contents = _split_messages(messages)

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        response = await self._client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        text = response.text or ""
        in_tokens, out_tokens = _extract_usage(response)
        return GeminiResult(
            text=text, in_tokens=in_tokens, out_tokens=out_tokens, model=model
        )


def _extract_usage(response: types.GenerateContentResponse) -> tuple[int, int]:
    """Pull real input/output token counts from the response.

    Output tokens include *thinking* tokens (billed as output on Gemini 2.5),
    so we derive output as ``total - prompt`` when possible and fall back to
    candidates + thoughts otherwise.
    """
    usage = response.usage_metadata
    if usage is None:
        return 0, 0

    in_tokens = usage.prompt_token_count or 0
    total = usage.total_token_count or 0
    if total > in_tokens:
        out_tokens = total - in_tokens
    else:
        out_tokens = (usage.candidates_token_count or 0) + (
            usage.thoughts_token_count or 0
        )
    return in_tokens, out_tokens
