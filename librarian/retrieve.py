"""Online retrieval: (rewrite) -> BM25 + vector -> RRF fusion -> rerank.

``mode`` exists so the ablation runs the *same code path* with stages
switched off: ``bm25`` | ``vector`` | ``hybrid`` | ``hybrid_rerank``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import settings
from .index import Index
from .rerank import get_reranker

MODES = ("bm25", "vector", "hybrid", "hybrid_rerank")


@dataclass
class Hit:
    idx: int
    score: float                    # final score used for ordering
    bm25_rank: int | None = None
    vector_rank: int | None = None
    rerank_score: float | None = None


@dataclass
class RetrievalResult:
    hits: list[Hit]
    queries: list[str]
    timings_ms: dict = field(default_factory=dict)
    reranked: bool = False


def rrf(rankings: list[list[int]], k: int, weights: list[float] | None = None) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of w / (k + rank).

    Uses ranks, not raw scores, so BM25 scores (0..30) and cosines (0..1)
    never have to be put on the same scale. Known weakness: a chunk ranked
    #1 by one retriever and absent from the other (typical for rare kernel
    identifiers like "nosmp") scores below a chunk both rank ~#5. Weights
    let the ablation test leaning on BM25.
    """
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = {}
    for ranking, w in zip(rankings, weights):
        for r, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + w / (k + r + 1)
    return fused


class Retriever:
    def __init__(self, index: Index | None = None, reranker_spec: str | None = None):
        self.index = index or Index()
        spec = settings.reranker if reranker_spec is None else reranker_spec
        self.reranker = get_reranker(spec)

    def search(self, query: str, k: int | None = None, mode: str = "hybrid_rerank",
               filters: dict | None = None, extra_queries: list[str] | None = None) -> RetrievalResult:
        k = k or settings.top_k
        if mode == "hybrid_rerank" and self.reranker is None:
            mode = "hybrid"
        ix = self.index
        mask = ix.mask(filters)
        queries = [query] + list(extra_queries or [])
        t = {}

        t0 = time.perf_counter()
        bm_lists, vec_lists = [], []
        bm_rank, vec_rank = {}, {}
        for q in queries:
            if mode != "vector":
                hits = ix.bm25_search(q, settings.candidates, mask)
                bm_lists.append([i for i, _ in hits])
            if mode != "bm25":
                hits = ix.vector_search(q, settings.candidates, mask)
                vec_lists.append([i for i, _ in hits])
        for lst in bm_lists:
            for r, i in enumerate(lst):
                bm_rank[i] = min(bm_rank.get(i, r + 1), r + 1)
        for lst in vec_lists:
            for r, i in enumerate(lst):
                vec_rank[i] = min(vec_rank.get(i, r + 1), r + 1)
        t["search"] = (time.perf_counter() - t0) * 1000

        weights = [settings.bm25_weight] * len(bm_lists) + [settings.vector_weight] * len(vec_lists)
        fused = rrf(bm_lists + vec_lists, settings.rrf_k, weights)
        order = sorted(fused, key=fused.get, reverse=True)
        hits = [Hit(i, fused[i], bm_rank.get(i), vec_rank.get(i)) for i in order]

        reranked = False
        if mode == "hybrid_rerank" and hits:
            confident = (settings.skip_rerank_when_confident and bm_lists and vec_lists
                         and bm_lists[0][:1] == vec_lists[0][:1])
            if not confident:
                t1 = time.perf_counter()
                pool = hits[: settings.rerank_pool]
                scores = self.reranker.score(query, [ix.chunks[h.idx].index_text for h in pool])
                for h, s in zip(pool, scores):
                    h.rerank_score = s
                    h.score = s
                hits = sorted(pool, key=lambda h: h.score, reverse=True)
                t["rerank"] = (time.perf_counter() - t1) * 1000
                reranked = True
        return RetrievalResult(hits[:k], queries, t, reranked)
