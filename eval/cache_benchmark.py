"""Compare nano HTTP latency with answer caching off/on on the same repeated workload.

Both arms start with an empty answer cache. Cold fills remain in mixed results.
The benchmark starts its own local server; an existing port-8000 app is untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
import time

import numpy as np
import requests
from common import ROOT, load_qa, save


def content_signature(output):
    keys = ("answer", "sources", "cited", "invalid_citations", "idk", "idk_gated",
            "queries", "reranked", "generator")
    return hashlib.sha256(json.dumps({key: output[key] for key in keys}, sort_keys=True).encode()).hexdigest()


def summarize(rows):
    def latency(selected):
        values = [r["http_ms"] for r in selected]
        return {"n": len(values), "p50_ms": float(np.percentile(values, 50)) if values else None,
                "p95_ms": float(np.percentile(values, 95)) if values else None}

    return {
        **latency(rows), "mean_cost_usd": float(np.mean([r["cost_usd"] for r in rows])) if rows else None,
        "cache_hits": sum(r["cached"] for r in rows),
        "cache_hit_rate": sum(r["cached"] for r in rows) / len(rows) if rows else None,
        "hits": latency([r for r in rows if r["cached"]]),
        "misses": latency([r for r in rows if not r["cached"]]),
        "repeated_requests": latency([r for r in rows if r["pass"] > 0]),
    }


def select_questions(count):
    path = ROOT / "logs" / "requests.jsonl"
    frequencies = Counter()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("question", "").strip():
                frequencies[row["question"].strip()] += 1
    popular = sorted(frequencies, key=lambda q: (-frequencies[q], q))
    questions = list(dict.fromkeys(popular + [q["question"] for q in load_qa()]))[:count]
    return questions, {q: frequencies[q] for q in questions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, default=10)
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--tag", default="cache_nano")
    parser.add_argument("--use-dotenv-key", action="store_true")
    args = parser.parse_args()
    if args.questions <= 0 or args.passes < 2:
        parser.error("questions must be positive and passes >= 2")
    if args.use_dotenv_key:
        os.environ.pop("OPENAI_API_KEY", None)

    from librarian.config import settings
    from librarian.pipeline import Librarian
    from librarian import server as service
    from werkzeug.serving import make_server

    questions, frequencies = select_questions(args.questions)
    if not questions:
        parser.error("No benchmark questions available")
    library = Librarian()
    if library.generator.model != "gpt-5.4-nano":
        parser.error("This benchmark expects the validated gpt-5.4-nano generation profile")
    # Model loading, embedding initialization, and a reranker warmup are excluded.
    library.retriever.search(questions[0])
    service._lib = library
    service.LOG = ROOT / "logs" / "cache_benchmark"
    service.LOG.mkdir(exist_ok=True)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    server = make_server("127.0.0.1", 0, service.app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    session = requests.Session()
    original_policy = settings.answer_cache
    recorded = {}
    first_answers = {}
    error = None
    try:
        for label, enabled in (("before", False), ("after", True)):
            settings.answer_cache = enabled
            library.clear_cache()
            recorded[label] = []
            for repeat in range(args.passes):
                for i, question in enumerate(questions):
                    start = time.perf_counter()
                    response = session.post(f"http://127.0.0.1:{server.server_port}/api/ask",
                                            json={"question": question}, timeout=180)
                    elapsed = (time.perf_counter() - start) * 1000
                    if not response.ok:
                        data = response.json()
                        raise RuntimeError(f"HTTP {response.status_code}: {data.get('detail') or data.get('error')}")
                    output = response.json()
                    signature = content_signature(output)
                    cached = output["cached"]
                    if label == "before" and cached:
                        raise RuntimeError("Uncached baseline unexpectedly returned a cache hit")
                    if label == "after":
                        if repeat == 0:
                            if cached:
                                raise RuntimeError("Cache was not empty at the start of the candidate arm")
                            first_answers[question] = signature
                        elif not cached or signature != first_answers[question]:
                            raise RuntimeError("Repeated answer/citations changed or cache unexpectedly missed")
                        if cached and (output["usage"] or output["cost_usd"]):
                            raise RuntimeError("Cache hit incorrectly reports new token usage or cost")
                    recorded[label].append({
                        "question_index": i, "pass": repeat, "http_ms": elapsed,
                        "pipeline_ms": output["timings_ms"]["total"],
                        "cost_usd": output["cost_usd"], "cached": cached,
                        "content_signature": signature,
                    })
                print(f"[{label}] pass {repeat + 1}/{args.passes}; "
                      f"{len(recorded[label])} requests complete", flush=True)
    except Exception as exc:
        error = str(exc)
        print(f"[failed] {error}", flush=True)
    finally:
        settings.answer_cache = original_policy
        session.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    expected = len(questions) * args.passes
    complete = not error and all(len(recorded.get(label, [])) == expected for label in ("before", "after"))
    before, after = (summarize(recorded.get(label, [])) for label in ("before", "after"))
    report = {
        "method": "same sequential HTTP workload, cache off then on, both initially empty",
        "manifest": {"questions": questions, "prior_logged_counts": frequencies,
                     "passes": args.passes, "concurrency": 1, "settings": asdict(settings),
                     "index": library.retriever.index.meta},
        "selection": "most frequent logged questions; QA questions fill missing slots",
        "scope": "loopback HTTP + complete nano pipeline; excludes model startup; includes candidate cold fills",
        "cache_policy": {"before_enabled": False, "after_enabled": True,
                         "ttl_seconds": settings.answer_cache_ttl,
                         "max_entries": settings.answer_cache_size},
        "complete": complete, "error": error, "before": before, "after": after,
        "cached_content_unchanged": complete,
        "comparison": {
            "mixed_p50_reduction_fraction": 1 - after["p50_ms"] / before["p50_ms"] if complete else None,
            "mixed_p95_reduction_fraction": 1 - after["p95_ms"] / before["p95_ms"] if complete else None,
            "repeat_p95_reduction_fraction": 1 - after["repeated_requests"]["p95_ms"] / before["repeated_requests"]["p95_ms"] if complete else None,
            "cost_reduction_fraction": 1 - after["mean_cost_usd"] / before["mean_cost_usd"] if complete and before["mean_cost_usd"] else None,
        },
        "rows": recorded,
        "limitations": "Small repeated-question workload, not a traffic forecast. Exact questions only. "
                       "Cold misses still invoke retrieval/reranking/generation. Provider prompt caching and "
                       "network variation can affect miss latency. Cache is per process and is lost on restart.",
    }
    path = save(f"{args.tag}.json", report)
    print(json.dumps({"complete": complete, "before": before, "after": after,
                      "comparison": report["comparison"], "report": str(path)}, indent=2), flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
