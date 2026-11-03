"""End-to-end latency and cost against the *running service*, plus the cost model.

Sends the eval questions to POST /api/ask with a few concurrent clients and
reports p50/p95 wall-clock latency as a user sees it (network + retrieval +
rerank + LLM), cost per request from the token usage the service reports,
and what that means at today's traffic and at 100x.

    python eval/load_test.py --url http://localhost:8000 --n 60 --concurrency 4 --tag baseline
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from common import load_qa, save

def one(q, url, timeout=120):
    t = time.perf_counter()
    try:
        r = requests.post(f"{url.rstrip('/')}/api/ask", json={"question": q}, timeout=timeout)
    except requests.RequestException as exc:
        return {"ok": False, "question": q, "status": None,
                "ms": (time.perf_counter() - t) * 1000, "error": str(exc)[:800]}
    ms = (time.perf_counter() - t) * 1000
    try:
        d = r.json()
    except ValueError:
        d = None
    if not r.ok or not isinstance(d, dict) or "error" in d:
        detail = (d.get("detail") or d.get("error") or str(d)) if isinstance(d, dict) else r.text
        return {"ok": False, "question": q, "status": r.status_code, "ms": ms,
                "error": str(detail or "Empty or invalid JSON response")[:800]}
    return {"ok": True, "status": r.status_code, "ms": ms, "cost": d.get("cost_usd", 0.0), "server": d.get("timings_ms", {}),
            "cached": bool(d.get("cached", False)),
            "tokens": sum(u.get("input_tokens", 0) + u.get("cached_input_tokens", 0) for u in d.get("usage", [])),
            "out_tokens": sum(u.get("output_tokens", 0) + u.get("thinking_tokens", 0) for u in d.get("usage", []))}


def summarize(res, elapsed, concurrency, per_day):
    ok = [r for r in res if r["ok"]]
    lat = [r["ms"] for r in ok]
    stage = {}
    for k in ("rewrite", "search", "rerank", "generate"):
        v = [r["server"][k] for r in ok if k in r["server"]]
        if v:
            stage[k] = {"p50": round(float(np.percentile(v, 50)), 1), "p95": round(float(np.percentile(v, 95)), 1)}
    cost = float(np.mean([r["cost"] for r in ok])) if ok else None
    return {
        "n": len(res), "successful_requests": len(ok), "errors": len(res) - len(ok), "concurrency": concurrency,
        "complete": len(ok) == len(res) and bool(res),
        "metrics_scope": "successful requests only; failed requests may still incur upstream charges",
        "error_details": [r for r in res if not r["ok"]],
        "throughput_rps": round(len(ok) / elapsed, 2) if elapsed > 0 else 0.0,
        "p50_ms": round(float(np.percentile(lat, 50)), 1) if lat else None,
        "p95_ms": round(float(np.percentile(lat, 95)), 1) if lat else None,
        "stages_ms": stage,
        "cache_hits": sum(r.get("cached", False) for r in ok),
        "cache_hit_rate": sum(r.get("cached", False) for r in ok) / len(ok) if ok else None,
        "cache_latency_ms": {
            label: {"n": len(values), "p50": round(float(np.percentile(values, 50)), 2),
                    "p95": round(float(np.percentile(values, 95)), 2)}
            for label in ("hit", "miss")
            if (values := [r["ms"] for r in ok if bool(r.get("cached", False)) == (label == "hit")])
        },
        "mean_input_tokens": round(float(np.mean([r["tokens"] for r in ok])), 1) if ok else None,
        "mean_output_tokens": round(float(np.mean([r["out_tokens"] for r in ok])), 1) if ok else None,
        "cost_per_request_usd": round(cost, 6) if cost is not None else None,
        "cost_model": {
            "today": {"requests_per_day": per_day, "usd_per_month": round(cost * per_day * 30, 2) if cost is not None else None},
            "100x": {"requests_per_day": per_day * 100, "usd_per_month": round(cost * per_day * 100 * 30, 2) if cost is not None else None},
            "note": "LLM tokens only. Add hosting (one CPU VM holds the index in ~1 GB RAM) and, at 100x, "
                    "enough replicas for the rerank+LLM latency at peak.",
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--requests-per-day", type=int, default=200)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.n <= 0 or args.concurrency <= 0 or args.timeout <= 0 or args.requests_per_day < 0:
        ap.error("n, concurrency, and timeout must be positive; requests-per-day must be nonnegative")
    qs = [q["question"] for q in load_qa()]
    if not qs:
        ap.error("No evaluation questions found")
    qs = (qs * (args.n // len(qs) + 1))[:args.n]

    def run(q):
        result = one(q, args.url, args.timeout)
        if not result["ok"]:
            print(f"[failed] HTTP {result['status'] or 'connection error'}: {result['error']}", flush=True)
        return result

    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.concurrency) as ex:
        res = list(ex.map(run, qs))
    summary = summarize(res, time.perf_counter() - t0, args.concurrency, args.requests_per_day)
    path = save(f"load{'_' + args.tag if args.tag else ''}.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Report saved: {path}")
    if not summary["successful_requests"]:
        print("No requests succeeded. Check error_details and logs/errors.jsonl before benchmarking again.")
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
