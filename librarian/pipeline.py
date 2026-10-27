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
        self._cache: OrderedDict[str, dict] = OrderedDict()

    def ask(self, question: str, filters: dict | None = None, mode: str = "hybrid_rerank",
            k: int | None = None) -> dict:
        t0 = time.perf_counter()
        question = question.strip()
        key = hashlib.sha1(json.dumps([question.lower(), filters, mode, k]).encode()).hexdigest()
        if settings.answer_cache and key in self._cache:
            out = dict(self._cache[key])
            out.update(cached=True, cost_usd=0.0,
                       timings_ms={"total": (time.perf_counter() - t0) * 1000})
            return out

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
        if settings.answer_cache:
            self._cache[key] = out
            if len(self._cache) > 1000:
                self._cache.popitem(last=False)
        return out
