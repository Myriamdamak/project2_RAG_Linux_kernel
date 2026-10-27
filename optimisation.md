# Week 8 — Optimisation report

**Target:** cut cost per request **or** p95 latency by **50%**, losing at most
**2 points** of accuracy on our own eval. Every row below is measured before
and after on the same 125 questions (`data/qa.jsonl`) and the same load test.

> Fill the tables with your runs. The commands are exact; the "idea" column
> explains why each change should help so you can predict before you measure.

## Baseline (end of Week 7)

```bash
make index
python eval/retrieval_eval.py --tag baseline
python eval/generation_eval.py --judge --tag baseline
make run &   python eval/load_test.py --n 100 --concurrency 4 --tag baseline
```

| metric | baseline |
|---|---|
| accuracy (judge: correct + ½ partial) | |
| faithfulness | |
| citation accuracy | |
| recall@5 (hybrid + rerank) | |
| p50 / p95 latency, end to end (ms) | |
| — of which rerank / generate p95 (ms) | |
| input / output (+thinking) tokens per request | |
| cost per request (USD) | |
| cost per month today / at 100× | |

## Changes (one at a time, then combined)

Each knob is a setting in `librarian/config.py`, set with an env var, so a
change is a re-run, not a code edit.

| # | change | how | idea | Δ cost | Δ p95 | Δ accuracy | keep? |
|---|---|---|---|---|---|---|---|
| 1 | Fewer chunks to the model | `LIB_TOP_K=4` (from 6) | Input tokens dominate cost; the reranker puts the gold chunk in the top 3 most of the time — check `recall@3` vs `recall@5` first. | | | | |
| 2 | Smaller chunks | rebuild with `--size 200` | Fewer tokens per chunk; check the chunk sweep's recall-per-token column. | | | | |
| 3 | Skip the reranker when retrievers agree | `LIB_SKIP_RERANK_WHEN_CONFIDENT=1` | The cross-encoder is the largest CPU cost in the search step. When BM25 and vector agree on #1, reranking rarely changes the answer. | | | | |
| 4 | Smaller reranker pool | `LIB_RERANK_POOL=15` | Cross-encoder time is linear in pool size. | | | | |
| 5 | Never rewrite | `LIB_REWRITE=never` | Removes a whole LLM round-trip on short questions. Watch recall on 1–3 word questions. | | | | |
| 6 | Cache repeated questions | `LIB_ANSWER_CACHE=1` | Classmates ask the same questions; a hit costs $0 and ~1 ms. Measure the hit rate from `logs/requests.jsonl`. | | | | |
| 7 | IDK gate | tune `LIB_IDK_THRESHOLD` from `results/retrieval_*.json` | Unanswerable questions stop before the LLM call. | | | | |
| 8 | Thinking level | `LIB_THINKING_LEVEL=minimal` vs `low` | Thinking tokens are billed as output (8× input price). | | | | |
| 9 | Context caching | Gemini implicit caching of the fixed system prompt | Only helps once the fixed prefix passes the model's minimum cacheable size; our system prompt is short, so measure `cached_input_tokens` before claiming anything. | | | | |
| 10 | Smaller embedding model / index | e.g. 256-d or int8 vectors | Index RAM and search time; search is already ~5 ms, so expect little. | | | | |

## After

| metric | baseline | optimised | change |
|---|---|---|---|
| accuracy | | | (must be ≥ −2 pts) |
| cost per request | | | (target −50%) |
| p95 latency | | | (target −50%) |

## What did not work

List every change that failed to pay off, with its numbers. Negative results
are results.
