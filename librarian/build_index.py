"""CLI: python -m librarian.build_index [--out DIR] [--embedder lsa] [--chunker fixed] [--size N] [--overlap N]"""
import argparse
import json

from .index import build

ap = argparse.ArgumentParser(description="Build the chunk, BM25 and vector index.")
ap.add_argument("--out")
ap.add_argument("--embedder")
ap.add_argument("--chunker", choices=["structure", "section", "fixed"])
ap.add_argument("--size", type=int)
ap.add_argument("--overlap", type=int)
ap.add_argument("--reuse-index", help="Reuse exact matching chunks from an index with the same ST model")
a = ap.parse_args()
meta = build(a.out, a.chunker, a.size, a.overlap, a.embedder, a.reuse_index)
print(json.dumps({k: meta[k] for k in ("n_docs", "n_chunks", "embedder", "chunker",
                                       "chunk_tokens", "chunk_overlap", "seconds")}, indent=2))
