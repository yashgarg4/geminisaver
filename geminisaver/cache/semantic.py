"""Semantic cache: local embeddings + cosine similarity.

The wedge. A prompt is embedded with a small *local* model (bge-small, no API
cost, no data egress), unit-normalized so a dot product equals cosine
similarity. If the nearest previously-seen prompt scores above the threshold,
we replay its answer instead of calling Gemini.

Vector search uses faiss ``IndexFlatIP`` when available and falls back to a
numpy brute-force dot product otherwise — identical results, faiss is just
faster at scale.

LRU eviction bounds memory: the least-recently-used entry is dropped when the
cache is full. A hit refreshes recency.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import numpy as np

from . import CachedResponse

try:  # faiss is optional; numpy brute-force is a correct fallback.
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # pragma: no cover - depends on platform wheels
    faiss = None  # type: ignore
    _HAS_FAISS = False


DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class SemanticCache:
    """Embedding + cosine-similarity cache with LRU eviction.

    Parameters
    ----------
    model_name:
        sentence-transformers model id used for embeddings.
    max_entries:
        LRU capacity.
    embed_fn:
        Optional injected embedding function ``str -> np.ndarray`` (used by
        tests to avoid loading a real model). If omitted, the model is loaded
        lazily on first use.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_entries: int = 10_000,
        embed_fn: Callable[[str], np.ndarray] | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_entries = max_entries
        self._model = None
        self._embed_fn = embed_fn

        # id -> entry, in LRU order (most-recently-used last).
        self._entries: "OrderedDict[int, CachedResponse]" = OrderedDict()
        self._embeddings: dict[int, np.ndarray] = {}
        self._next_id = 0

        # Search structures, rebuilt lazily when marked dirty.
        self._dirty = True
        self._matrix: np.ndarray | None = None
        self._row_ids: list[int] = []
        self._index = None  # faiss index when available

    # --- embedding ---

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """Return a unit-normalized float32 embedding for ``text``."""
        if self._embed_fn is not None:
            vec = np.asarray(self._embed_fn(text), dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        else:
            model = self._ensure_model()
            vec = model.encode(
                text, normalize_embeddings=True, convert_to_numpy=True
            ).astype(np.float32)
        # Cosine correctness depends on unit norm.
        assert np.isclose(
            float(np.linalg.norm(vec)), 1.0, atol=1e-3
        ), "embedding must be unit-normalized for cosine similarity"
        # faiss requires contiguous float32; normalization above can upcast.
        return np.ascontiguousarray(vec, dtype=np.float32)

    # --- search index maintenance ---

    def _rebuild(self) -> None:
        self._row_ids = list(self._entries.keys())
        if self._row_ids:
            self._matrix = np.vstack([self._embeddings[i] for i in self._row_ids])
        else:
            self._matrix = None

        if _HAS_FAISS and self._matrix is not None:
            index = faiss.IndexFlatIP(self._matrix.shape[1])
            index.add(self._matrix)
            self._index = index
        else:
            self._index = None
        self._dirty = False

    def _search(self, query: np.ndarray) -> tuple[int, float] | None:
        """Return (entry_id, score) of the nearest entry, or None if empty."""
        if self._dirty:
            self._rebuild()
        if not self._row_ids:
            return None

        if self._index is not None:  # faiss path
            scores, rows = self._index.search(query[None, :], 1)
            row = int(rows[0][0])
            score = float(scores[0][0])
        else:  # numpy brute-force cosine (unit vectors -> dot == cosine)
            sims = self._matrix @ query
            row = int(np.argmax(sims))
            score = float(sims[row])
        return self._row_ids[row], score

    # --- public API ---

    def get(
        self,
        text: str,
        threshold: float,
        *,
        embedding: np.ndarray | None = None,
    ) -> tuple[CachedResponse, float] | None:
        """Return (entry, score) if the nearest prompt is above ``threshold``.

        Pass a precomputed ``embedding`` to avoid embedding the same text twice
        (the pipeline reuses it for :meth:`add` on a miss).
        """
        if not self._entries:
            return None
        query = embedding if embedding is not None else self.embed(text)
        result = self._search(query)
        if result is None:
            return None
        entry_id, score = result
        if score >= threshold:
            self._entries.move_to_end(entry_id)  # refresh LRU recency
            return self._entries[entry_id], score
        return None

    def add(
        self,
        text: str,
        response: CachedResponse,
        *,
        embedding: np.ndarray | None = None,
    ) -> None:
        """Store a prompt/response, evicting the LRU entry if at capacity."""
        vec = embedding if embedding is not None else self.embed(text)

        entry_id = self._next_id
        self._next_id += 1
        self._entries[entry_id] = response
        self._embeddings[entry_id] = vec

        while len(self._entries) > self.max_entries:
            old_id, _ = self._entries.popitem(last=False)  # LRU = oldest
            self._embeddings.pop(old_id, None)

        self._dirty = True

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self._embeddings.clear()
        self._dirty = True
