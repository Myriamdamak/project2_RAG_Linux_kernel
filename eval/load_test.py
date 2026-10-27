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

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--concurrency", type=int, default=4)
ap.add_argument("--requests-per-day", type=int, default=200,
                help="today's expected traffic (classmates + a few kernel users)")
ap.add_argument("--tag", default="")
args = ap.parse_args()

qs = [q["question"] for q in load_qa()]
qs = (qs * (args.n // len(qs) + 1))[: args.n]


def one(q):
    t = time.perf_counter()
    r = requests.post(f"{args.url}/api/ask", json={"question": q}, timeout=120)
    ms = (time.perf_counter() - t) * 1000
    d = r.json() if r.ok else {}
    return {"ok": r.ok, "ms": ms, "cost": d.get("cost_usd", 0.0), "server": d.get("timings_ms", {}),
            "tokens": sum(u.get("input_tokens", 0) + u.get("cached_input_tokens", 0) for u in d.get("usage", [])),
            "out_tokens": sum(u.get("output_tokens", 0) + u.get("thinking_tokens", 0) for u in d.get("usage", []))}


t0 = time.perf_counter()
with ThreadPoolExecutor(args.concurrency) as ex:
    res = list(ex.map(one, qs))
elapsed = time.perf_counter() - t0
ok = [r for r in res if r["ok"]]
lat = [r["ms"] for r in ok]
stage = {}
for k in ("rewrite", "search", "rerank", "generate"):
    v = [r["server"][k] for r in ok if k in r["server"]]
    if v:
        stage[k] = {"p50": round(float(np.percentile(v, 50)), 1), "p95": round(float(np.percentile(v, 95)), 1)}
cost = float(np.mean([r["cost"] for r in ok])) if ok else 0.0
per_day = args.requests_per_day
summary = {
    "n": len(res), "errors": len(res) - len(ok), "concurrency": args.concurrency,
    "throughput_rps": round(len(ok) / elapsed, 2),
    "p50_ms": round(float(np.percentile(lat, 50)), 1), "p95_ms": round(float(np.percentile(lat, 95)), 1),
    "stages_ms": stage,
    "mean_input_tokens": round(float(np.mean([r["tokens"] for r in ok])), 1),
    "mean_output_tokens": round(float(np.mean([r["out_tokens"] for r in ok])), 1),
    "cost_per_request_usd": round(cost, 6),
    "cost_model": {
        "today": {"requests_per_day": per_day, "usd_per_month": round(cost * per_day * 30, 2)},
        "100x": {"requests_per_day": per_day * 100, "usd_per_month": round(cost * per_day * 100 * 30, 2)},
        "note": "LLM tokens only. Add hosting (one CPU VM holds the index in ~1 GB RAM) and, at 100x, "
                "enough replicas for the rerank+LLM latency at peak.",
    },
}
save(f"load{'_' + args.tag if args.tag else ''}.json", summary)
print(json.dumps(summary, indent=2))
