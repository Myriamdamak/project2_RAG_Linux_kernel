"""Dense embeddings behind one tiny interface.

    emb = get_embedder(spec)
    emb.fit(texts)             # only LSA learns from the corpus
    X = emb.encode_docs(texts) # float32, L2-normalised -> dot product = cosine
    q = emb.encode_query(text)

Backends
--------
``st:<model>``  sentence-transformers (default ``BAAI/bge-small-en-v1.5``:
                384 dims, 33M params, runs on a laptop CPU).
``lsa``         TF-IDF + truncated SVD. No download, no GPU, deterministic.
                It is the honest offline baseline, and lets the whole
                pipeline run where HuggingFace is unreachable.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .config import ROOT, settings

# Asymmetric models expect an instruction on the query side only.
QUERY_PREFIX = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "intfloat/e5-small-v2": "query: ",
}
DOC_PREFIX = {"intfloat/e5-small-v2": "passage: "}


def _normalise(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1
    return (x / n).astype(np.float32)


class STEmbedder:
    def __init__(self, model: str, batch: int = 64):
        from sentence_transformers import SentenceTransformer
        self.name = model
        source = model
        if settings.embed_model_path and model == settings.embedder.removeprefix("st:"):
            local = Path(settings.embed_model_path)
            source = str(local if local.is_absolute() else ROOT / local)
        self.model = SentenceTransformer(source, device=None)
        self.batch = batch

    def fit(self, texts):
        return self

    def encode_docs(self, texts):
        p = DOC_PREFIX.get(self.name, "")
        return self.model.encode([p + t for t in texts], batch_size=self.batch,
                                 normalize_embeddings=True, show_progress_bar=True,
                                 convert_to_numpy=True).astype(np.float32)

    def encode_query(self, text):
        p = QUERY_PREFIX.get(self.name, "")
        return self.model.encode([p + text], normalize_embeddings=True,
                                 convert_to_numpy=True).astype(np.float32)[0]

    def save(self, path):
        pass


class LSAEmbedder:
    name = "lsa"

    def __init__(self, dims: int = 256):
        self.dims = dims
        self.vec = self.svd = None

    def fit(self, texts):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from .bm25 import tokenize
        self.vec = TfidfVectorizer(analyzer=tokenize, min_df=2, max_df=0.5,
                                   sublinear_tf=True, dtype=np.float32)
        X = self.vec.fit_transform(texts)
        self.svd = TruncatedSVD(self.dims, random_state=0).fit(X)
        return self

    def encode_docs(self, texts):
        return _normalise(self.svd.transform(self.vec.transform(texts)))

    def encode_query(self, text):
        return self.encode_docs([text])[0]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump((self.vec, self.svd), f)

    @classmethod
    def load(cls, path):
        self = cls()
        with open(path, "rb") as f:
            self.vec, self.svd = pickle.load(f)
        return self


def get_embedder(spec: str, batch: int = 64, index_dir: str | None = None):
    if spec.startswith("st:"):
        return STEmbedder(spec[3:], batch)
    if spec == "lsa":
        if index_dir:
            import os
            p = os.path.join(index_dir, "lsa.pkl")
            if os.path.exists(p):
                return LSAEmbedder.load(p)
        return LSAEmbedder()
    raise ValueError(f"unknown embedder {spec!r}")
