"""The whole online path in one place: question -> answer with sources.

    lib = Librarian()
    out = lib.ask("What does vm.swappiness control?")
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict
import threading

from .config import cost_usd, settings
from .generate import IDK, get_generator, parse_citations
from .retrieve import Retriever


def needs_rewrite(q: str) -> bool:
    """Cheap heuristic for ``rewrite=auto``: only spend an LLM call on
    questions that are very short or obviously multi-part."""
    words = q.split()
    return (len(words) <= 3 or q.count("?") > 1
            or bool(re.search(r"\b(vs\.?|versus|difference between|compare)\b", q, re.I))
            or (len(words) > 12 and " and " in q and q.rstrip().endswith("?")))


class Librarian:
    def __init__(self, retriever: Retriever | None = None, generator=None):
        self.retriever = retriever or Retriever()
        self.generator = generator or get_generator()
        self._cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._cache_lock = threading.RLock()
        self._inflight: dict[str, threading.Event] = {}

    def clear_cache(self):
        """Clear stored answers (active requests may subsequently fill entries)."""
        with self._cache_lock:
            self._cache.clear()

    def _cache_key(self, question, filters, mode, k):
        # Technical identifiers can be case sensitive (%p vs %P, config names).
        # Include the response policy and index generation to avoid stale answers.
        identity = {
            "question": question, "filters": filters or {}, "mode": mode,
            "k": k or settings.top_k, "generator": self.generator.model,
            "settings": asdict(settings), "index": self.retriever.index.meta,
        }
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def ask(self, question: str, filters: dict | None = None, mode: str = "hybrid_rerank",
            k: int | None = None) -> dict:
        t0 = time.perf_counter()
        question = question.strip()
        if not settings.answer_cache:
            return self._ask_uncached(question, filters, mode, k, t0)
        if settings.answer_cache_ttl <= 0 or settings.answer_cache_size <= 0:
            raise ValueError("Answer cache TTL and size must be positive")
        key = self._cache_key(question, filters, mode, k)
        waited = False
        while True:
            with self._cache_lock:
                entry = self._cache.get(key)
                now = time.monotonic()
                if entry and now - entry[0] < settings.answer_cache_ttl:
                    self._cache.move_to_end(key)
                    out = deepcopy(entry[1])
                    out.update(cached=True, cost_usd=0.0, usage=[],
                               cache_age_s=round(now - entry[0], 3),
                               cache_coalesced=waited,
                               timings_ms={"total": (time.perf_counter() - t0) * 1000})
                    return out
                if entry:
                    del self._cache[key]
                event = self._inflight.get(key)
                if event is None:
                    event = threading.Event()
                    self._inflight[key] = event
                    break
            # Coalesce duplicate concurrent misses without blocking other keys.
            event.wait()
            waited = True
        try:
            out = self._ask_uncached(question, filters, mode, k, t0)
            with self._cache_lock:
                self._cache[key] = (time.monotonic(), deepcopy(out))
                self._cache.move_to_end(key)
                while len(self._cache) > settings.answer_cache_size:
                    self._cache.popitem(last=False)
            return out
        finally:
            # Failures are never cached, and waiters are always released.
            with self._cache_lock:
                self._inflight.pop(key, None)
                event.set()

    def _ask_uncached(self, question, filters, mode, k, t0):

        usage_all: list[tuple[str, dict]] = []
        timings: dict[str, float] = {}

        # 1. optional query rewrite
        extra = []
        if settings.rewrite == "always" or (settings.rewrite == "auto" and needs_rewrite(question)):
            t = time.perf_counter()
            extra, u = self.generator.rewrite(question)
            extra = [q for q in dict.fromkeys(extra) if q.lower() != question.lower()]
            if u:
                usage_all.append((settings.rewrite_model, u))
            timings["rewrite"] = (time.perf_counter() - t) * 1000

        # 2. retrieve (+ rerank)
        res = self.retriever.search(question, k, mode, filters, extra)
        timings.update(res.timings_ms)
        chunks = [self.retriever.index.chunks[h.idx] for h in res.hits]

        # 3. "I don't know" gate: if the reranker says even the best chunk is
        #    irrelevant, do not pay for an LLM call that can only guess.
        best = res.hits[0].rerank_score if res.hits and res.reranked else None
        if not chunks or (best is not None and best < settings.idk_threshold):
            answer_text, gen_usage = f"{IDK}\nNothing in the documentation index matched this question.", {}
            gated = True
        else:
            t = time.perf_counter()
            answer_text, gen_usage = self.generator.answer(question, chunks)
            timings["generate"] = (time.perf_counter() - t) * 1000
            gated = False
        if gen_usage:
            usage_all.append((self.generator.model, gen_usage))

        parsed = parse_citations(answer_text, len(chunks))
        timings["total"] = (time.perf_counter() - t0) * 1000
        sources = [{
            "n": n, "id": c.id, "path": c.path, "section": c.section, "url": c.url,
            "text": c.text, "cited": n in parsed["cited"],
            "rerank_score": res.hits[n - 1].rerank_score,
            "bm25_rank": res.hits[n - 1].bm25_rank, "vector_rank": res.hits[n - 1].vector_rank,
        } for n, c in enumerate(chunks, 1)]
        out = {
            "question": question, **parsed, "sources": sources,
            "queries": res.queries, "reranked": res.reranked, "idk_gated": gated,
            "timings_ms": {k: round(v, 1) for k, v in timings.items()},
            "usage": [{"model": m, **u} for m, u in usage_all],
            "cost_usd": round(sum(cost_usd(m, u) for m, u in usage_all), 6),
            "generator": self.generator.model, "cached": False,
        }
        return out
