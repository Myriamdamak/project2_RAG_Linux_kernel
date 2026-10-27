"""Chunk size / overlap / strategy are decisions to test, not defaults.

Builds one index per configuration under index_sweep/<name>/ and runs the
retrieval eval on each, then prints one table. Gold labels are (path,
evidence), so they stay valid across configurations.

    python eval/chunk_sweep.py                  # with the configured embedder (slow: re-embeds each config)
    python eval/chunk_sweep.py --embedder lsa   # fast, offline approximation
"""
import argparse
from pathlib import Path

from common import ROOT, load_qa, save
from retrieval_eval import evaluate

from librarian.chunker import approx_tokens
from librarian.index import Index, build
from librarian.retrieve import Retriever

CONFIGS = [  # (strategy, size, overlap)
    ("fixed", 350, 50),
    ("section", 200, 0),
    ("section", 200, 50),
    ("section", 350, 0),
    ("section", 350, 50),
    ("section", 350, 100),
    ("section", 500, 50),
    ("section", 800, 100),
]

ap = argparse.ArgumentParser()
ap.add_argument("--embedder", default=None)
ap.add_argument("--modes", nargs="+", default=["bm25", "vector", "hybrid"])
ap.add_argument("--rebuild", action="store_true")
args = ap.parse_args()

qa = load_qa()
rows = []
for strategy, size, overlap in CONFIGS:
    name = f"{strategy}-{size}-{overlap}" + (f"-{args.embedder}" if args.embedder else "")
    d = ROOT / "index_sweep" / name
    if args.rebuild or not (d / "meta.json").exists():
        build(str(d), strategy, size, overlap, args.embedder)
    ix = Index(str(d))
    mean_tok = sum(approx_tokens(c.text) for c in ix.chunks) / len(ix.chunks)
    r = Retriever(ix, reranker_spec="none")
    for mode in args.modes:
        res = evaluate(r, qa, mode)
        # bigger chunks trivially contain the evidence more often, so report
        # what top-5 costs in prompt tokens next to the recall it buys
        rows.append({"config": name, "chunks": len(ix.chunks), "mean_tokens": round(mean_tok),
                     "tokens@5": round(5 * mean_tok), "mode": mode,
                     **{k: res[k] for k in ("recall@1", "recall@5", "recall@10", "mrr", "p95_ms")}})
        print(f"{name:28s} {mode:7s} chunks={len(ix.chunks):6d} R@5={res['recall@5']:.3f} MRR={res['mrr']:.3f}")

save("chunk_sweep.json", rows)
md = ["| config | chunks | mean tok | ~tokens in top-5 | mode | R@1 | R@5 | R@10 | MRR |",
      "|---|---|---|---|---|---|---|---|---|"]
md += [f"| {r['config']} | {r['chunks']} | {r['mean_tokens']} | {r['tokens@5']} | {r['mode']} | "
       f"{r['recall@1']:.3f} | {r['recall@5']:.3f} | "
       f"{r['recall@10']:.3f} | {r['mrr']:.3f} |" for r in rows]
(ROOT / "results" / "chunk_sweep.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
