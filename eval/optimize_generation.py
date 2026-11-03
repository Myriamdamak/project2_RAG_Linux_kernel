"""Paired live evaluation of a generation-only optimization.

Both models receive exactly the same fresh retrieval/rewrite results. Cost
includes each model's measured generation usage plus the shared rewrite usage.
Only the baseline is timed end to end; candidate timing is generation-only.
Acceptance requires >=50% lower cost and <=2 percentage points accuracy loss.
No answer cache or gold-label-based routing is used.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
from common import RESULTS, load_qa, save


def summarize(rows, variant, expected):
    outputs = [r[variant] for r in rows if variant in r]
    judged = [o for o in outputs if o.get("judge", {}).get("correctness") in {"correct", "partial", "wrong"}]
    answerable = [o for o in judged if o["answerable"]]
    scores = {"correct": 1., "partial": .5, "wrong": 0.}
    accuracy = float(np.mean([scores[o["judge"]["correctness"]] for o in answerable])) if answerable else None
    overall = [scores[o["judge"]["correctness"]] if o["answerable"] else float(o["idk"]) for o in judged]
    gen_ms = [o["timings_ms"].get("generate", 0.) for o in outputs]
    return {
        "n": len(outputs), "judged": len(judged), "judged_answerable": len(answerable),
        "complete": len(judged) == expected,
        "accuracy": accuracy,
        "overall_accuracy_with_idk": float(np.mean(overall)) if overall else None,
        "idk_recall": float(np.mean([o["idk"] for o in judged if not o["answerable"]])) if any(not o["answerable"] for o in judged) else None,
        "mean_cost_usd": float(np.mean([o["cost_usd"] for o in outputs])) if outputs else None,
        "mean_generation_cost_usd": float(np.mean([o["generation_cost_usd"] for o in outputs])) if outputs else None,
        "generation_p50_ms": float(np.percentile(gen_ms, 50)) if gen_ms else None,
        "generation_p95_ms": float(np.percentile(gen_ms, 95)) if gen_ms else None,
        "judge_cost_usd": sum(o.get("judge", {}).get("cost_usd", 0.) for o in outputs),
    }


def compare(before, after):
    complete = before["complete"] and after["complete"]
    cost_reduction = (1 - after["mean_cost_usd"] / before["mean_cost_usd"]) if complete and before["mean_cost_usd"] else None
    accuracy_change = 100 * (after["accuracy"] - before["accuracy"]) if complete and before["accuracy"] is not None and after["accuracy"] is not None else None
    overall_change = 100 * (after["overall_accuracy_with_idk"] - before["overall_accuracy_with_idk"]) if complete else None
    return {
        "complete": complete, "cost_reduction_fraction": cost_reduction,
        "accuracy_change_percentage_points": accuracy_change,
        "overall_accuracy_change_percentage_points": overall_change,
        "target_met": bool(complete and cost_reduction is not None and cost_reduction >= .5
                           and accuracy_change is not None and accuracy_change >= -2. - 1e-9
                           and overall_change >= -2. - 1e-9),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-model", default="gpt-5.4-mini")
    parser.add_argument("--candidate-model", default="gpt-5.4-nano")
    parser.add_argument("--judge-model", default="gpt-5.4-mini")
    parser.add_argument("--tag", default="optimization_nano")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="Pilot only; resume without this flag to finish the same manifest")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-dotenv-key", action="store_true", help="Use the local .env API key instead of an inherited shell key")
    args = parser.parse_args()
    if args.workers <= 0 or args.limit < 0:
        parser.error("workers must be positive and limit nonnegative")
    if args.use_dotenv_key:
        os.environ.pop("OPENAI_API_KEY", None)

    # Import only after choosing the credential source. Never persist credentials.
    from librarian.config import PRICES, cost_usd, settings
    from librarian.generate import OpenAIGenerator, REWRITE_PROMPT, SYSTEM_PROMPT, parse_citations
    from librarian.index import Index
    from librarian.pipeline import Librarian
    from librarian.retrieve import Retriever
    from generation_eval import JUDGE_PROMPT, judge, write_checkpoint

    for model in (args.baseline_model, args.candidate_model, args.judge_model):
        if model not in PRICES:
            parser.error(f"No verified pricing for {model}; refusing to report zero cost")
    settings.answer_cache = False
    qa = load_qa()
    index = Index()
    retriever = Retriever(index)
    configuration = asdict(settings)
    configuration.pop("generator")  # baseline/candidate are explicit above
    # Cache policy has no effect in this explicitly uncached comparison.
    configuration.pop("answer_cache_ttl", None)
    configuration.pop("answer_cache_size", None)
    identity = {
        "qa": qa, "index": index.meta, "settings": configuration,
        "baseline_model": args.baseline_model, "candidate_model": args.candidate_model,
        "judge_model": args.judge_model, "system_prompt": SYSTEM_PROMPT,
        "rewrite_prompt": REWRITE_PROMPT, "judge_prompt": JUDGE_PROMPT,
    }
    metadata = {"fingerprint": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()}
    path = RESULTS / f"{args.tag}.checkpoint.json"
    saved = {}
    if path.exists():
        if not args.resume:
            parser.error(f"Checkpoint exists: {path}; use --resume or a new tag")
        checkpoint = json.loads(path.read_text())
        if checkpoint["metadata"] != metadata:
            parser.error("The questions/index/configuration changed; use a new tag")
        saved = checkpoint["rows"]
    lock = threading.RLock()
    local = threading.local()
    chunks = {c.id: c for c in index.chunks}

    def persist(qid, variant, output):
        with lock:
            saved.setdefault(qid, {})[variant] = output
            write_checkpoint(path, metadata, saved, {})

    def run(item):
        qid = item["id"]
        with lock:
            previous = deepcopy(saved.get(qid, {}))
        if all(previous.get(v, {}).get("judge", {}).get("correctness") in {"correct", "partial", "wrong"}
               for v in ("before", "after")):
            return
        if not hasattr(local, "base"):
            local.base = Librarian(retriever, OpenAIGenerator(args.baseline_model))
            local.candidate = OpenAIGenerator(args.candidate_model)
            local.judge = OpenAIGenerator(args.judge_model)
        before = previous.get("before")
        if before is None:
            before = local.base.ask(item["question"])
            before["answerable"] = item["answerable"]
            before["generation_cost_usd"] = cost_usd(args.baseline_model, before["usage"][-1]) if not before["idk_gated"] else 0.
            persist(qid, "before", before)
        after = previous.get("after")
        if after is None:
            after = deepcopy(before)
            after.pop("judge", None)
            after.pop("judge_error", None)
            after["generator"] = args.candidate_model
            if not before["idk_gated"]:
                start = time.perf_counter()
                text, usage = local.candidate.answer(item["question"], [chunks[s["id"]] for s in before["sources"]])
                after.update(parse_citations(text, len(before["sources"])))
                after["timings_ms"] = {"generate": (time.perf_counter() - start) * 1000}
                after["usage"] = before["usage"][:-1] + [{"model": args.candidate_model, **usage}]
                after["generation_cost_usd"] = cost_usd(args.candidate_model, usage)
                after["cost_usd"] = sum(cost_usd(u["model"], u) for u in after["usage"])
                for source in after["sources"]:
                    source["cited"] = source["n"] in after["cited"]
            else:
                after["timings_ms"] = {"generate": 0.}
            persist(qid, "after", after)
        for variant, output in (("before", before), ("after", after)):
            if output.get("judge", {}).get("correctness") not in {"correct", "partial", "wrong"}:
                try:
                    output["judge"] = judge(local.judge, item, output)
                    output.pop("judge_error", None)
                except Exception as exc:
                    output["judge_error"] = str(exc)
                    persist(qid, variant, output)
                    raise
                persist(qid, variant, output)
        print(f"[{qid}] before={before['judge']['correctness']} after={after['judge']['correctness']} "
              f"cost=${before['cost_usd']:.6f}->${after['cost_usd']:.6f}", flush=True)

    # Exclude model load/cold-start from generation measurements.
    retriever.search(qa[0]["question"])
    selected = qa[:args.limit] if args.limit else qa
    error = None
    with ThreadPoolExecutor(args.workers) as executor:
        futures = [executor.submit(run, item) for item in selected]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                error = str(exc)
                print(f"[stopped] {error}; checkpoint saved, resume after resolving the error", flush=True)
                for pending in futures:
                    pending.cancel()
                break
    before = summarize(list(saved.values()), "before", len(qa))
    after = summarize(list(saved.values()), "after", len(qa))
    report = {
        "method": "paired generation change with identical live retrieval, rewrite results, and judge; no answer cache",
        "cost_scope": "measured generation tokens plus the same measured rewrite tokens; excludes judge, hosting, and local compute",
        "latency_scope": "generation only; candidate end-to-end p95 is not measured by this experiment",
        "manifest": identity, "before": before, "after": after,
        "comparison": compare(before, after), "error": error,
    }
    output = save(f"{args.tag}.json", report)
    print(json.dumps({"before": before, "after": after, "comparison": report["comparison"], "report": str(output)}, indent=2), flush=True)
    return 1 if error else (0 if args.limit or report["comparison"]["target_met"] else 2)


if __name__ == "__main__":
    raise SystemExit(main())
