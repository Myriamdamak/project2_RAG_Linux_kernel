"""Inspect/tune HNSW without re-embedding or uploading the corpus.

python -m librarian.tune_qdrant --apply --benchmark
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .config import settings
from .qdrant import QdrantStore


def status(store):
    info = store.request("GET")
    return {key: info.get(key) for key in (
        "status", "optimizer_status", "points_count", "indexed_vectors_count",
    )} | {"hnsw_config": info["config"]["hnsw_config"],
         "optimizer_config": info["config"]["optimizer_config"]}


def benchmark(store, directory, samples, k):
    """Use cached document vectors to isolate database/network search latency."""
    meta = json.loads((directory / "meta.json").read_text())
    if meta.get("vector_store") != "qdrant" or meta["qdrant_collection"] != store.collection:
        raise ValueError("Benchmark index must refer to the selected Qdrant collection")
    vectors = np.load(directory / "vectors.npy", mmap_mode="r")
    rows = np.random.default_rng(42).choice(len(vectors), min(samples, len(vectors)), replace=False)
    if not len(rows):
        raise ValueError("No benchmark vectors found")
    times = {"exact": [], "hnsw": []}
    recall = []
    # Warm both paths; alternate measured order to reduce ordering bias.
    for exact in (True, False):
        store.search(vectors[rows[0]].astype(np.float32), k, meta["qdrant_generation"], exact=exact)
    for j, row in enumerate(rows):
        results = {}
        for exact in ((True, False) if j % 2 == 0 else (False, True)):
            mode = "exact" if exact else "hnsw"
            start = time.perf_counter()
            hits = store.search(vectors[row].astype(np.float32), k, meta["qdrant_generation"], exact=exact)
            times[mode].append((time.perf_counter() - start) * 1000)
            results[mode] = {cid for cid, _ in hits}
        if not results["exact"]:
            raise ValueError("Exact search returned no hits for the active generation")
        recall.append(len(results["exact"] & results["hnsw"]) / len(results["exact"]))
    return {
        "samples": len(rows), "k": k, "hnsw_ef": max(k, settings.qdrant_hnsw_ef),
        "query_source": "cached document vectors; excludes BGE encoding and reranking",
        "latency_includes": "HTTP round trip and Qdrant search",
        "recall_vs_exact": float(np.mean(recall)),
        **{mode: {"p50_ms": round(float(np.median(values)), 2),
                  "p95_ms": round(float(np.percentile(values, 95)), 2)}
           for mode, values in times.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply configured HNSW settings in place")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--index-dir", default=settings.index_dir)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--k", type=int, default=settings.candidates)
    parser.add_argument("--ef-values", type=int, nargs="+", help="Compare several HNSW query search efforts")
    args = parser.parse_args()
    if args.samples <= 0 or args.k <= 0:
        parser.error("--samples and --k must be positive")
    if args.ef_values and (not args.benchmark or min(args.ef_values) <= 0):
        parser.error("--ef-values requires --benchmark and positive values")
    directory = Path(args.index_dir)
    meta = json.loads((directory / "meta.json").read_text())
    if meta.get("vector_store") != "qdrant":
        parser.error("The selected index does not use Qdrant")
    store = QdrantStore(meta["qdrant_collection"])
    print(json.dumps({"before": status(store)}, indent=2), flush=True)
    if args.apply:
        store.configure_hnsw()
        print(json.dumps({"after": status(store)}, indent=2), flush=True)
    if args.benchmark:
        original_ef = settings.qdrant_hnsw_ef
        try:
            for ef in args.ef_values or [original_ef]:
                settings.qdrant_hnsw_ef = ef
                print(json.dumps({"benchmark": benchmark(store, directory, args.samples, args.k)}, indent=2), flush=True)
        finally:
            settings.qdrant_hnsw_ef = original_ef


if __name__ == "__main__":
    main()
