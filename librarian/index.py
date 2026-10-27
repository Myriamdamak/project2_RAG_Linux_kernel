"""Offline index build + the in-memory index the service searches.

Storage choice: flat files (JSONL + .npy + scipy .npz), loaded into RAM.
35k chunks x 384 dims x float16 = 27 MB; brute-force cosine over that is a
single matrix-vector product (~5 ms on a laptop CPU). A vector database would
add a server, a network hop and a deployment dependency to save milliseconds
we do not need. We revisit this at 100x corpus size (design.md, cost model).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .chunker import Chunk, chunk_documents
from .config import settings
from .corpus import load_corpus
from .embed import get_embedder


def build(out_dir: str | None = None, strategy: str | None = None,
          size: int | None = None, overlap: int | None = None,
          embedder: str | None = None) -> dict:
    out = Path(out_dir or settings.index_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = embedder or settings.embedder
    t0 = time.time()
    docs = load_corpus()
    chunks = chunk_documents(docs, strategy, size, overlap)
    t1 = time.time()
    print(f"[index] {len(docs)} docs -> {len(chunks)} chunks ({t1 - t0:.1f}s)")

    with open(out / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict()) + "\n")

    texts = [c.index_text for c in chunks]
    BM25().fit(texts).save(str(out / "bm25"))
    t2 = time.time()
    print(f"[index] BM25 built ({t2 - t1:.1f}s)")

    emb = get_embedder(spec, settings.embed_batch)
    emb.fit(texts)
    vecs = emb.encode_docs(texts)
    np.save(out / "vectors.npy", vecs.astype(np.float16))
    if spec == "lsa":
        emb.save(str(out / "lsa.pkl"))
    t3 = time.time()
    print(f"[index] embeddings {vecs.shape} with {spec} ({t3 - t2:.1f}s)")

    meta = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "embedder": spec, "chunker": strategy or settings.chunker,
        "chunk_tokens": size or settings.chunk_tokens,
        "chunk_overlap": settings.chunk_overlap if overlap is None else overlap,
        "n_docs": len(docs), "n_chunks": len(chunks), "dims": int(vecs.shape[1]),
        "seconds": {"chunk": t1 - t0, "bm25": t2 - t1, "embed": t3 - t2},
        "bytes": {p.name: p.stat().st_size for p in out.iterdir() if p.is_file()},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


class Index:
    def __init__(self, index_dir: str | None = None):
        d = Path(index_dir or settings.index_dir)
        if not (d / "meta.json").exists():
            raise SystemExit(f"No index in {d}. Build it first: make index")
        self.dir = d
        self.meta = json.loads((d / "meta.json").read_text())
        with open(d / "chunks.jsonl") as f:
            self.chunks = [Chunk(**json.loads(line)) for line in f]
        self.by_id = {c.id: i for i, c in enumerate(self.chunks)}
        self.bm25 = BM25.load(str(d / "bm25"))
        self.vectors = np.load(d / "vectors.npy").astype(np.float32)
        self.embedder = get_embedder(self.meta["embedder"], index_dir=str(d))
        # metadata columns as arrays so filters are vectorised masks
        self.paths = np.array([c.path for c in self.chunks])
        self.subsystems = np.array([c.subsystem for c in self.chunks])
        self.doc_types = np.array([c.doc_type for c in self.chunks])
        # "<2021-01-01" means "not changed since then": sort it before any real date
        self.dates = np.array([("0000" + c.date[1:]) if c.date.startswith("<") else (c.date or "0000")
                               for c in self.chunks])

    def mask(self, filters: dict | None) -> np.ndarray | None:
        """Metadata filter -> boolean mask (None = everything)."""
        if not filters:
            return None
        m = np.ones(len(self.chunks), dtype=bool)
        if filters.get("subsystem"):
            m &= np.isin(self.subsystems, np.atleast_1d(filters["subsystem"]))
        if filters.get("doc_type"):
            m &= np.isin(self.doc_types, np.atleast_1d(filters["doc_type"]))
        if filters.get("updated_after"):          # "YYYY" or "YYYY-MM-DD"
            m &= self.dates >= str(filters["updated_after"])
        if filters.get("path_prefix"):
            m &= np.char.startswith(self.paths, filters["path_prefix"])
        return m

    def bm25_search(self, query: str, n: int, mask=None) -> list[tuple[int, float]]:
        s = self.bm25.scores(query)
        return _top(s, n, mask, positive_only=True)

    def vector_search(self, query: str, n: int, mask=None) -> list[tuple[int, float]]:
        q = self.embedder.encode_query(query)
        s = self.vectors @ q
        return _top(s, n, mask)


def _top(scores: np.ndarray, n: int, mask, positive_only: bool = False) -> list[tuple[int, float]]:
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    n = min(n, len(scores))
    idx = np.argpartition(-scores, n - 1)[:n]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx if np.isfinite(scores[i]) and (scores[i] > 0 or not positive_only)]
