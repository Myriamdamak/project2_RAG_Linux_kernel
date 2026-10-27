"""Second-stage reranking.

The first stage (BM25 + vectors) scores query and chunk *independently*; a
cross-encoder reads them *together*, which is far more accurate and far too
slow to run over 35k chunks. So we run it only over the fused top
``rerank_pool`` candidates.
"""
from __future__ import annotations


class CrossEncoderReranker:
    def __init__(self, model: str):
        from sentence_transformers import CrossEncoder
        self.name = model
        self.model = CrossEncoder(model, max_length=512)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self.model.predict([(query, t) for t in texts],
                                                      batch_size=32, show_progress_bar=False)]


def get_reranker(spec: str):
    if not spec or spec == "none":
        return None
    if spec.startswith("ce:"):
        return CrossEncoderReranker(spec[3:])
    raise ValueError(f"unknown reranker {spec!r}")
