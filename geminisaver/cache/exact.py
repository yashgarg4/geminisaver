"""Exact-match cache: normalize -> hash -> dict lookup.

Instant and free, with zero false positives. Normalization only collapses
surrounding/duplicate whitespace so that prompts differing solely in
formatting still match; it deliberately does *not* lowercase or otherwise
rewrite text, because that could merge prompts with genuinely different intent
(e.g. "polish" vs "Polish"). Anything reworded is the semantic layer's job.
"""

from __future__ import annotations

import hashlib

from . import CachedResponse


class ExactCache:
    """A dict keyed by the SHA-256 of the whitespace-normalized prompt."""

    def __init__(self) -> None:
        self._store: dict[str, CachedResponse] = {}

    @staticmethod
    def normalize(text: str) -> str:
        """Collapse runs of whitespace and trim. No case/semantic changes."""
        return " ".join(text.split())

    @classmethod
    def _key(cls, text: str) -> str:
        return hashlib.sha256(cls.normalize(text).encode("utf-8")).hexdigest()

    def get(self, text: str) -> CachedResponse | None:
        return self._store.get(self._key(text))

    def set(self, text: str, response: CachedResponse) -> None:
        self._store[self._key(text)] = response

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()
