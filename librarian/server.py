"""HTTP service: a small web UI plus a JSON API.

    POST /api/ask        {"question": "...", "filters": {"subsystem": "admin-guide"}}
    POST /api/feedback   {"id": "<request id>", "verdict": "up|down|wrong_source", "comment": "..."}
    GET  /api/stats      live p50/p95 latency, cost, IDK rate, feedback counts
    GET  /api/health

Every request and every piece of feedback is appended to logs/*.jsonl. Those
logs are the source for the "real users" section of the report.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from .config import settings
from .pipeline import Librarian

app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"))
LOG = Path(settings.log_dir)
LOG.mkdir(parents=True, exist_ok=True)
_lock = threading.Lock()
_lib: Librarian | None = None


def lib() -> Librarian:
    global _lib
    if _lib is None:
        with _lock:
            if _lib is None:
                _lib = Librarian()
    return _lib


def _append(name: str, record: dict):
    with _lock, open(LOG / name, "a") as f:
        f.write(json.dumps(record) + "\n")


def _read(name: str) -> list[dict]:
    p = LOG / name
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


@app.get("/")
def home():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    ix = lib().retriever.index
    return jsonify(ok=True, chunks=len(ix.chunks), embedder=ix.meta["embedder"],
                   reranker=settings.reranker, generator=lib().generator.model,
                   kernel=ix.chunks[0].version if ix.chunks else None)


@app.get("/api/subsystems")
def subsystems():
    ix = lib().retriever.index
    names, counts = np.unique(ix.subsystems, return_counts=True)
    return jsonify(sorted(({"name": n, "chunks": int(c)} for n, c in zip(names, counts)),
                          key=lambda x: -x["chunks"]))


@app.post("/api/ask")
def ask():
    body = request.get_json(force=True, silent=True) or {}
    q = (body.get("question") or "").strip()
    if not q:
        return jsonify(error="question is required"), 400
    if len(q) > 1000:
        return jsonify(error="question too long (max 1000 chars)"), 400
    filters = {k: v for k, v in (body.get("filters") or {}).items() if v}
    rid = uuid.uuid4().hex[:12]
    try:
        out = lib().ask(q, filters or None)
    except Exception as e:                     # surface, log, don't crash the server
        _append("errors.jsonl", {"id": rid, "ts": time.time(), "question": q, "error": str(e)})
        return jsonify(error="The librarian failed to answer.", detail=str(e)[:300]), 502
    out["id"] = rid
    _append("requests.jsonl", {
        "id": rid, "ts": time.time(), "question": q, "filters": filters,
        "idk": out["idk"], "idk_gated": out["idk_gated"], "cached": out["cached"],
        "timings_ms": out["timings_ms"], "cost_usd": out["cost_usd"], "usage": out["usage"],
        "cited_paths": [s["path"] for s in out["sources"] if s["cited"]],
        "generator": out["generator"], "citation_coverage": out["citation_coverage"],
    })
    return jsonify(out)


@app.post("/api/feedback")
def feedback():
    body = request.get_json(force=True, silent=True) or {}
    if body.get("verdict") not in {"up", "down", "wrong_source"} or not body.get("id"):
        return jsonify(error="need id and verdict in up|down|wrong_source"), 400
    _append("feedback.jsonl", {"id": body["id"], "ts": time.time(), "verdict": body["verdict"],
                               "comment": (body.get("comment") or "")[:2000]})
    return jsonify(ok=True)


@app.get("/api/stats")
def stats():
    reqs = _read("requests.jsonl")
    fb = _read("feedback.jsonl")
    lat = np.array([r["timings_ms"]["total"] for r in reqs]) if reqs else np.array([0.0])
    cost = [r["cost_usd"] for r in reqs]
    verdicts: dict[str, int] = {}
    for f in fb:
        verdicts[f["verdict"]] = verdicts.get(f["verdict"], 0) + 1
    return jsonify(
        requests=len(reqs),
        p50_ms=round(float(np.percentile(lat, 50)), 1),
        p95_ms=round(float(np.percentile(lat, 95)), 1),
        mean_cost_usd=round(float(np.mean(cost)), 6) if cost else 0.0,
        idk_rate=round(sum(r["idk"] for r in reqs) / len(reqs), 3) if reqs else 0.0,
        feedback=verdicts,
    )


def main():
    lib()   # load index and models before accepting traffic
    app.run(host=settings.host, port=settings.port, threaded=True)


if __name__ == "__main__":
    main()
