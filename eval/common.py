"""Shared helpers for the evals.

Relevance is defined by (path, evidence), not by chunk id. A retrieved chunk
is relevant to a gold source when it comes from the same file AND contains
the evidence span. This keeps the gold labels valid when chunk size,
overlap or strategy change, which is exactly what the chunking sweep varies.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QA_PATH = ROOT / "data" / "qa.jsonl"
RESULTS = ROOT / "results"


def norm(s: str) -> str:
    """Lower-case, drop RST inline markup (`` ** |) and collapse whitespace."""
    s = s.lower().replace("`", "").replace("*", "").replace("|", " ")
    return re.sub(r"\s+", " ", s).strip()


def load_qa(path: Path = QA_PATH) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_matches(chunk, source: dict) -> bool:
    return chunk.path == source["path"] and norm(source["evidence"]) in norm(chunk.text)


def save(name: str, obj) -> Path:
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=2))
    return p


def percentile(xs, q):
    import numpy as np
    return float(np.percentile(xs, q)) if len(xs) else 0.0
