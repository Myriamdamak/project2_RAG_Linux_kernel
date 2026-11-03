"""Central configuration.

Every knob that the ablation or the Week 8 optimisation touches lives here, so
experiments are "change one setting, re-run the eval" rather than code edits.
Values can be overridden with environment variables of the same name prefixed
with ``LIB_`` (e.g. ``LIB_CHUNK_TOKENS=300``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def _load_dotenv(path: Path):
    """Minimal .env support (KEY=VALUE lines) so the API key never lives in code."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

# Pinned corpus version: answers are only "correct" relative to one release.
KERNEL_TAG = "v7.3"


def _env(name: str, default):
    raw = os.environ.get(f"LIB_{name.upper()}")
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


@dataclass
class Settings:
    # --- paths -------------------------------------------------------------
    corpus_dir: str = str(ROOT / "corpus" / "linux" / "Documentation")
    index_dir: str = str(ROOT / "index")
    log_dir: str = str(ROOT / "logs")

    # --- corpus selection --------------------------------------------------
    # Translations duplicate English content in other languages; we index
    # English only (see design.md).
    exclude_dirs: str = "translations,sphinx,sphinx-static,sphinx-includes,images"

    # --- chunking (both are decisions to test, see eval/chunk_sweep.py) ----
    chunker: str = "structure"        # "structure" | "section" (alias) | "fixed"
    chunk_tokens: int = 350           # target size, approx tokens
    chunk_overlap: int = 50           # approx tokens carried into next chunk
    min_chunk_words: int = 5          # drop toctree-only / empty sections

    qdrant_url: str = os.environ.get("QDRANT_URL", "")
    qdrant_collection: str = "kernel_documentation"
    qdrant_batch: int = 256
    qdrant_hnsw_m: int = 16
    qdrant_hnsw_ef_construct: int = 100
    qdrant_hnsw_ef: int = 256      # higher values favor recall over latency
    qdrant_full_scan_threshold: int = 1000  # KB; scan small filtered sets
    qdrant_indexing_threshold: int = 2000   # KB; index smaller segments too

    # --- embeddings --------------------------------------------------------
    # "st:<hf model>"  sentence-transformers (default, needs HuggingFace)
    # "lsa"            TF-IDF + SVD, fully offline (fallback / baseline)
    embedder: str = "st:BAAI/bge-small-en-v1.5"
    embed_batch: int = 64
    embed_model_path: str = ""        # optional local copy of the configured ST model

    # --- retrieval ---------------------------------------------------------
    candidates: int = 50              # per retriever, before fusion
    rrf_k: int = 60                   # Reciprocal Rank Fusion constant
    bm25_weight: float = 1.0          # RRF weight of the keyword list
    vector_weight: float = 1.0        # RRF weight of the vector list
    top_k: int = 6                    # chunks passed to the generator

    # --- reranking ---------------------------------------------------------
    # "ce:<hf model>"  cross-encoder | "none"
    reranker: str = "ce:cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_pool: int = 30             # how many fused candidates to rerank
    # Below this cross-encoder logit for the best chunk we answer
    # "I don't know" without calling the LLM. Tuned on data/qa.jsonl.
    idk_threshold: float = -6.0
    # Week 8 knob: skip the reranker when BM25 and vector agree on the top hit.
    skip_rerank_when_confident: bool = False

    # --- query rewriting ---------------------------------------------------
    rewrite: str = "auto"             # "auto" | "always" | "never"

    # --- generation --------------------------------------------------------
    # "openai:<model id>" | "extractive" (offline, no API key needed)
    generator: str = "openai:gpt-5.4-nano"  # validated cost optimization; see optimisation.md
    rewrite_model: str = "gpt-5.4-mini"
    reasoning_effort: str = "none"    # model-specific; empty omits the parameter
    max_answer_tokens: int = 1600     # includes reasoning tokens when enabled
    max_rewrite_tokens: int = 800
    openai_base_url: str = "https://api.openai.com/v1"
    answer_cache: bool = False        # Week 8: cache repeated questions
    answer_cache_ttl: float = 3600.0  # seconds; per-process exact-answer cache
    answer_cache_size: int = 1000     # least-recently-used entries

    # --- service -----------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000

    excluded: list[str] = field(default_factory=list, init=False)

    def __post_init__(self):
        for f in fields(self):
            if f.init:
                setattr(self, f.name, _env(f.name, getattr(self, f.name)))
        self.excluded = [d for d in self.exclude_dirs.split(",") if d]


# Standard USD / million tokens. Verified 2026-10-27:
# https://developers.openai.com/api/docs/models/gpt-5.4-mini
# Other models need a verified pricing entry before cost reporting is accurate.
PRICES = {
    "gpt-5.4-mini": {"in": 0.75, "out": 4.50, "cached_in": 0.075},
    "gpt-5.4-mini-2026-03-17": {"in": 0.75, "out": 4.50, "cached_in": 0.075},
    # https://developers.openai.com/api/docs/models/gpt-5.4-nano (2026-11-03)
    "gpt-5.4-nano": {"in": 0.20, "out": 1.25, "cached_in": 0.02},
    "gpt-5.4-nano-2026-03-17": {"in": 0.20, "out": 1.25, "cached_in": 0.02},
}


def cost_usd(model: str, usage: dict) -> float:
    p = PRICES.get(model)
    if not p or not usage:
        return 0.0
    return (
        usage.get("input_tokens", 0) * p["in"]
        + usage.get("cached_input_tokens", 0) * p["cached_in"]
        + (usage.get("output_tokens", 0) + usage.get("thinking_tokens", 0)) * p["out"]
    ) / 1e6


settings = Settings()
