"""Generation eval: run the full pipeline on every question and score the
answers. Retrieval is scored separately (retrieval_eval.py); here we ask
whether the *answer* is right and honest about its sources.

Computed without any judge (cheap, deterministic):
  citation_coverage     share of factual sentences that carry a citation
  invalid_citation_rate share of answers that cited a non-existent source
  cites_gold            answer cites at least one chunk containing the gold evidence
  idk                   precision/recall of "I don't know" (unanswerable = positive)
  latency, tokens, cost

With an LLM judge (--judge, needs OPENAI_API_KEY):
  correctness           correct / partial / wrong against the reference answer
  faithfulness          share of claims supported by the chunks the answer cites
  citation_accuracy     share of citations whose chunk really says the claim

The judge sees the cited chunk text, so "faithful" means "supported by what
was cited", not "true according to the judge's own knowledge".

    python eval/generation_eval.py --judge --limit 40 --tag baseline
"""
import argparse
import hashlib
import os
from dataclasses import asdict
from pathlib import Path
import json
import time

import numpy as np
from common import RESULTS, chunk_matches, load_qa, percentile, save

from librarian.config import cost_usd, settings
from librarian.generate import OpenAIGenerator, get_generator
from librarian.pipeline import Librarian
from librarian.retrieve import Retriever

JUDGE_PROMPT = """You grade answers produced by a documentation assistant. You receive:
- QUESTION, REFERENCE answer written by a human, and the system's ANSWER;
- SOURCES: the numbered excerpts the system was given.

Return JSON:
{"correctness": "correct" | "partial" | "wrong",
 "claims": [{"claim": "<one factual claim from ANSWER>", "cited": [<numbers cited for it>],
             "supported_by_cited": true|false, "supported_by_any_source": true|false}]}

Rules:
- correctness compares ANSWER with REFERENCE: "correct" if it contains the reference's key facts and nothing contradicting them, "partial" if some key facts are missing, "wrong" if it contradicts the reference or answers a different question. An "I don't know" is "wrong" when the reference has an answer.
- List every factual claim in ANSWER (skip filler). supported_by_cited: the excerpts it cites state the claim. supported_by_any_source: some excerpt states it. Judge only from SOURCES, never from your own knowledge.
- If ANSWER is "I don't know", return an empty claims list."""


def judge(judge_gen: OpenAIGenerator, item: dict, out: dict) -> dict:
    sources = "\n\n".join(f"[{s['n']}] {s['path']}\n{s['text']}" for s in out["sources"])
    user = (f"QUESTION: {item['question']}\nREFERENCE: {item['answer']}\n"
            f"ANSWER: {out['answer']}\n\nSOURCES:\n{sources}")
    text, usage = judge_gen._call(judge_gen.model, JUDGE_PROMPT, user, 1500, json_mode=True)
    try:
        verdict = json.loads(text)
    except json.JSONDecodeError:
        verdict = {"correctness": "unparsed", "claims": [], "raw": text[:500]}
    verdict["cost_usd"] = cost_usd(judge_gen.model, usage)
    return verdict


def write_checkpoint(path, metadata, rows, outputs):
    """Atomic replacement keeps the previous checkpoint intact if interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "rows": rows, "outputs": outputs}, f)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def read_checkpoint(path, metadata):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["metadata"] != metadata:
        raise ValueError("Checkpoint configuration/data changed; use a new --tag")
    return data["rows"], data["outputs"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", help="reuse saved answers and completed judgments")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--judge-model", default=settings.rewrite_model)
    ap.add_argument("--limit", type=int, default=0, help="first N questions only")
    ap.add_argument("--mode", default="hybrid_rerank")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    qa = load_qa()
    if args.limit:
        qa = qa[: args.limit] + [q for q in qa if not q["answerable"]][: max(3, args.limit // 10)]
    lib = Librarian(Retriever(), get_generator())
    judge_gen = OpenAIGenerator(args.judge_model) if args.judge else None

    suffix = f"_{args.tag}" if args.tag else ""
    checkpoint = RESULTS / f"generation{suffix}.checkpoint.json"
    identity = {"settings": asdict(settings), "index": lib.retriever.index.meta,
                "qa": qa, "mode": args.mode, "judge": args.judge,
                "judge_model": args.judge_model, "judge_prompt": JUDGE_PROMPT}
    from librarian.generate import SYSTEM_PROMPT, REWRITE_PROMPT
    identity.update(system_prompt=SYSTEM_PROMPT, rewrite_prompt=REWRITE_PROMPT)
    metadata = {"fingerprint": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()}
    saved_rows, outputs = {}, {}
    if checkpoint.exists():
        if not args.resume:
            raise SystemExit(f"Checkpoint exists: {checkpoint}. Use --resume or a new --tag.")
        saved_rows, outputs = read_checkpoint(checkpoint, metadata)
    rows = []
    for i, item in enumerate(qa, 1):
        previous = saved_rows.get(item["id"], {})
        if previous and "error" not in previous and (not args.judge or
                previous.get("judge", {}).get("correctness") in {"correct", "partial", "wrong"}):
            rows.append(previous)
            print(f"[{i}/{len(qa)}] {item['id']} resumed")
            continue
        t = time.perf_counter()
        try:
            out = outputs.get(item["id"])
            if out is None:
                out = lib.ask(item["question"], mode=args.mode)
                outputs[item["id"]] = out
                write_checkpoint(checkpoint, metadata, saved_rows, outputs)
        except Exception as e:
            row = {"id": item["id"], "error": str(e)}
            rows.append(row)
            saved_rows[item["id"]] = row
            write_checkpoint(checkpoint, metadata, saved_rows, outputs)
            print(f"[{i}] error {e}")
            continue
        wall = out["timings_ms"]["total"]  # retain original latency for resumed answers
        idx_by_n = {s["n"]: s for s in out["sources"]}
        cited_chunks = [idx_by_n[n] for n in out["cited"] if n in idx_by_n]
        golds = item["sources"]

        class _C:  # adapt dict -> attribute access for chunk_matches
            def __init__(self, d): self.path, self.text = d["path"], d["text"]
        row = {
            "id": item["id"], "category": item["category"], "answerable": item["answerable"],
            "question": item["question"], "answer": out["answer"], "idk": out["idk"],
            "idk_gated": out["idk_gated"], "citation_coverage": out["citation_coverage"],
            "invalid_citations": out["invalid_citations"],
            "cites_gold": any(chunk_matches(_C(c), g) for c in cited_chunks for g in golds) if golds else None,
            "retrieved_gold": any(chunk_matches(_C(c), g) for c in out["sources"] for g in golds) if golds else None,
            "ms": wall, "cost_usd": out["cost_usd"], "usage": out["usage"],
        }
        if judge_gen:
            try:
                row["judge"] = judge(judge_gen, item, out)
            except Exception as e:
                row["judge_error"] = str(e)
                print(f"[{i}] judge failed; answer saved: {e}")
        rows.append(row)
        saved_rows[item["id"]] = row
        write_checkpoint(checkpoint, metadata, saved_rows, outputs)
        print(f"[{i}/{len(qa)}] {item['id']} idk={out['idk']} cov={out['citation_coverage']:.2f} "
              f"gold={row['cites_gold']} {wall:.0f}ms" + (f" judge={row.get('judge', {}).get('correctness', 'failed')}" if judge_gen else ""))

    ok = [r for r in rows if "error" not in r]
    ans = [r for r in ok if r["answerable"]]
    una = [r for r in ok if not r["answerable"]]
    idk_tp = sum(r["idk"] for r in una)
    idk_fp = sum(r["idk"] for r in ans)
    summary = {
        "generator": settings.generator, "reasoning_effort": settings.reasoning_effort,
        "mode": args.mode, "reranker": settings.reranker, "embedder": lib.retriever.index.meta["embedder"],
        "n": len(ok), "errors": len(rows) - len(ok),
        "citation_coverage": round(float(np.mean([r["citation_coverage"] for r in ans if not r["idk"]] or [0])), 3),
        "invalid_citation_rate": round(float(np.mean([bool(r["invalid_citations"]) for r in ok] or [0])), 3),
        "cites_gold": round(float(np.mean([r["cites_gold"] for r in ans if not r["idk"]] or [0])), 3),
        "retrieved_gold": round(float(np.mean([r["retrieved_gold"] for r in ans] or [0])), 3),
        "idk_recall": round(idk_tp / len(una), 3) if una else None,     # unanswerable correctly refused
        "idk_false_rate": round(idk_fp / len(ans), 3) if ans else None,  # answerable wrongly refused
        "p50_ms": round(percentile([r["ms"] for r in ok], 50), 1),
        "p95_ms": round(percentile([r["ms"] for r in ok], 95), 1),
        "mean_cost_usd": round(float(np.mean([r["cost_usd"] for r in ok] or [0])), 6),
    }
    if judge_gen:
        judged = [r for r in ans if r.get("judge", {}).get("correctness") in {"correct", "partial", "wrong"}]
        verdicts = [r["judge"]["correctness"] for r in judged]
        claims = [c for r in judged for c in r["judge"].get("claims", [])]
        cited_claims = [c for c in claims if c.get("cited")]
        summary.update({
            "correct": round(verdicts.count("correct") / len(judged), 3) if judged else None,
            "partial": round(verdicts.count("partial") / len(judged), 3) if judged else None,
            "wrong": round(verdicts.count("wrong") / len(judged), 3) if judged else None,
            # "accuracy" used for the Week 8 budget: correct + half credit for partial
            "accuracy": round((verdicts.count("correct") + 0.5 * verdicts.count("partial")) / len(judged), 3) if judged else None,
            "faithfulness": round(float(np.mean([bool(c.get("supported_by_any_source")) for c in claims] or [0])), 3),
            "citation_accuracy": round(float(np.mean([bool(c.get("supported_by_cited")) for c in cited_claims] or [0])), 3),
            "judge_cost_usd": round(sum(r.get("judge", {}).get("cost_usd", 0) for r in ok), 4),
        })
    summary["judge_errors"] = sum("judge_error" in r for r in rows)
    summary["judged_answerable"] = len(judged) if judge_gen else 0
    summary["complete"] = not summary["errors"] and (not judge_gen or all(
        r.get("judge", {}).get("correctness") in {"correct", "partial", "wrong"} for r in ok))
    save(f"generation{suffix}.json", {"summary": summary, "rows": rows})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
