"""BM25 keyword index, written out in ~80 lines instead of imported so every
line can be explained (and because the tokenizer matters more than the
library for this corpus).

Kernel docs are full of identifiers: ``CONFIG_PREEMPT_RT``, ``kmalloc()``,
``/proc/sys/vm/dirty_ratio``, ``GFP_KERNEL``. A tokenizer that splits on
underscores loses the exact identifier; one that does not split loses the
parts ("dirty ratio"). We emit both.
"""
from __future__ import annotations

import re

import numpy as np
from scipy import sparse

_WORD = re.compile(r"[A-Za-z0-9_]+")
STOP = set("""a an and are as at be but by for from has have how i if in into is it
its of on or that the their then there these this to was what when where which
who why will with does do can you your we our should would""".split())


def _stem(t: str) -> str:
    # deliberately tiny: plurals only. Aggressive stemming merges kernel
    # identifiers that must stay distinct (e.g. "mapping" vs "map").
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def tokenize(text: str) -> list[str]:
    out = []
    for raw in _WORD.findall(text):
        t = raw.lower()
        if t in STOP or len(t) < 2:
            continue
        out.append(_stem(t))
        if "_" in t:                              # CONFIG_FOO_BAR -> also config, foo, bar
            out.extend(_stem(p) for p in t.split("_") if len(p) > 1 and p not in STOP)
    return out


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vocab: dict[str, int] = {}
        self.matrix: sparse.csc_matrix | None = None   # docs x terms, BM25 weights
        self.idf: np.ndarray | None = None

    def fit(self, texts: list[str]) -> "BM25":
        rows, cols, vals = [], [], []
        lengths = np.zeros(len(texts), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = tokenize(text)
            lengths[i] = len(toks)
            counts: dict[int, int] = {}
            for t in toks:
                j = self.vocab.setdefault(t, len(self.vocab))
                counts[j] = counts.get(j, 0) + 1
            rows.extend([i] * len(counts)); cols.extend(counts.keys()); vals.extend(counts.values())
        tf = sparse.csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)),
                               shape=(len(texts), len(self.vocab)))
        n = len(texts)
        df = np.bincount(tf.indices, minlength=len(self.vocab))
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        # Precompute the per-(doc, term) BM25 weight so a query is just a sum
        # over its terms' columns.
        avgdl = lengths.mean() or 1.0
        norm = self.k1 * (1 - self.b + self.b * lengths / avgdl)
        tf = tf.tocoo()
        w = tf.data * (self.k1 + 1) / (tf.data + norm[tf.row])
        self.matrix = sparse.csc_matrix((w * self.idf[tf.col], (tf.row, tf.col)), shape=tf.shape)
        return self

    def scores(self, query: str) -> np.ndarray:
        ids = [self.vocab[t] for t in set(tokenize(query)) if t in self.vocab]
        if not ids:
            return np.zeros(self.matrix.shape[0], dtype=np.float32)
        return np.asarray(self.matrix[:, ids].sum(axis=1)).ravel()

    def save(self, path: str):
        sparse.save_npz(path + ".npz", self.matrix)
        np.save(path + ".idf.npy", self.idf)
        import json
        with open(path + ".vocab.json", "w") as f:
            json.dump(self.vocab, f)

    @classmethod
    def load(cls, path: str) -> "BM25":
        import json
        self = cls()
        self.matrix = sparse.load_npz(path + ".npz").tocsc()
        self.idf = np.load(path + ".idf.npy")
        with open(path + ".vocab.json") as f:
            self.vocab = json.load(f)
        return self
