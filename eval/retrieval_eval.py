"""Retrieval eval + ablation, measured separately from generation.

For every answerable question and every retrieval mode:
  recall@k   share of questions whose gold passage(s) are all in the top k
  MRR        mean of 1/rank of the first relevant chunk (0 if none in top 20)
  doc@5      looser: right *file* in the top 5, even if the wrong section
  latency    p50 / p95 of the search step (+ rerank for hybrid_rerank)

Modes = the ablation: bm25 | vector | hybrid | hybrid_rerank.
For hybrid_rerank it also records the top rerank score of answerable vs
unanswerable questions and suggests an "I don't know" threshold.

    python eval/retrieval_eval.py                      # all modes, default index
    python eval/retrieval_eval.py --modes bm25 hybrid --index index_lsa --tag lsa
"""
import argparse
import time
from collections import defaultdict

import numpy as np
from common import chunk_matches, load_qa, percentile, save

from librarian.config import settings
from librarian.index import Index
from librarian.retrieve import MODES, Retriever

KS = (1, 3, 5, 10, 20)


def rank_of(hits, chunks, source):
    for r, h in enumerate(hits, 1):
        if chunk_matches(chunks[h.idx], source):
            return r
    return None


def evaluate(retriever: Retriever, qa: list[dict], mode: str, use_rewrite: bool = False) -> dict:
    chunks = retriever.index.chunks
    recall = {k: [] for k in KS}
    rr, doc5, lat = [], [], []
    per_cat = defaultdict(list)
    top_scores = {"answerable": [], "unanswerable": []}
    misses = []
    for item in qa:
        t = time.perf_counter()
        res = retriever.search(item["question"], k=max(KS), mode=mode)
        lat.append((time.perf_counter() - t) * 1000)
        if res.reranked and res.hits:
            top_scores["answerable" if item["answerable"] else "unanswerable"].append(res.hits[0].rerank_score)
        if not item["answerable"]:
            continue
        ranks = [rank_of(res.hits, chunks, s) for s in item["sources"]]
        # all gold sources must be found (multi-hop needs every piece)
        worst = None if any(r is None for r in ranks) else max(ranks)
        first = min((r for r in ranks if r), default=None)
        for k in KS:
            recall[k].append(1.0 if worst is not None and worst <= k else 0.0)
        rr.append(1.0 / first if first else 0.0)
        gold_paths = {s["path"] for s in item["sources"]}
        doc5.append(1.0 if gold_paths & {chunks[h.idx].path for h in res.hits[:5]} else 0.0)
        per_cat[item["category"]].append(1.0 if worst is not None and worst <= 5 else 0.0)
        if worst is None or worst > 5:
            misses.append({"id": item["id"], "question": item["question"], "gold": item["sources"][0]["path"],
                           "rank": worst, "got": [chunks[h.idx].path + " :: " + chunks[h.idx].section[-60:]
                                                  for h in res.hits[:3]]})
    out = {
        "mode": mode, "n": len(rr),
        **{f"recall@{k}": round(float(np.mean(v)), 4) for k, v in recall.items()},
        "mrr": round(float(np.mean(rr)), 4), "doc@5": round(float(np.mean(doc5)), 4),
        "p50_ms": round(percentile(lat, 50), 1), "p95_ms": round(percentile(lat, 95), 1),
        "recall@5_by_category": {c: round(float(np.mean(v)), 3) for c, v in sorted(per_cat.items())},
        "misses@5": misses,
    }
    if top_scores["answerable"]:
        out["idk_threshold"] = suggest_threshold(top_scores)
    return out


def suggest_threshold(s: dict) -> dict:
    """Pick the rerank-score cut that best separates answerable from
    unanswerable questions, but never refuses more than 5% of answerable
    ones: wrongly saying "I don't know" to a real question is the worse error."""
    a, u = np.array(s["answerable"]), np.array(s["unanswerable"])
    best = None
    for t in np.unique(np.concatenate([a, u])):
        false_idk = float(np.mean(a < t))
        if false_idk > 0.05:
            continue
        caught = float(np.mean(u < t)) if len(u) else 0.0
        if best is None or caught > best["unanswerable_caught"]:
            best = {"threshold": round(float(t), 3), "unanswerable_caught": round(caught, 3),
                    "answerable_refused": round(false_idk, 3)}
    return {"suggested": best,
            "answerable_top_score_p10": round(float(np.percentile(a, 10)), 3),
            "unanswerable_top_score_p90": round(float(np.percentile(u, 90)), 3) if len(u) else None}


def markdown(rows: list[dict], meta: dict) -> str:
    head = "| mode | " + " | ".join(f"R@{k}" for k in KS) + " | MRR | doc@5 | p50 ms | p95 ms |"
    lines = [f"Index: {meta['n_chunks']} chunks, embedder `{meta['embedder']}`, chunker "
             f"`{meta['chunker']}` {meta['chunk_tokens']}/{meta['chunk_overlap']} tokens. "
             f"{rows[0]['n']} answerable questions.\n", head,
             "|" + "---|" * (len(KS) + 5)]
    for r in rows:
        lines.append(f"| {r['mode']} | " + " | ".join(f"{r[f'recall@{k}']:.3f}" for k in KS)
                     + f" | {r['mrr']:.3f} | {r['doc@5']:.3f} | {r['p50_ms']} | {r['p95_ms']} |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=settings.index_dir)
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    qa = load_qa()
    retriever = Retriever(Index(args.index))
    rows = []
    for mode in args.modes:
        if mode == "hybrid_rerank" and retriever.reranker is None:
            print("[skip] hybrid_rerank: reranker disabled (LIB_RERANKER=none)")
            continue
        r = evaluate(retriever, qa, mode)
        rows.append(r)
        print(f"{mode:14s} R@1 {r['recall@1']:.3f}  R@5 {r['recall@5']:.3f}  R@10 {r['recall@10']:.3f}"
              f"  MRR {r['mrr']:.3f}  p95 {r['p95_ms']} ms")
    suffix = f"_{args.tag}" if args.tag else ""
    p = save(f"retrieval{suffix}.json", {"index": retriever.index.meta, "rows": rows})
    md = markdown(rows, retriever.index.meta)
    (p.parent / f"retrieval{suffix}.md").write_text(md + "\n")
    print("\n" + md)
